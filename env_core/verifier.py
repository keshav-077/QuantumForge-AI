"""Zero-trust server-side episode verifier and batch scorer for CryoForge RL.

Recomputes every mechanical, thermal, compatibility, crane, bay-overlap, route-clearance,
permit-quota, SLA, budget, and batch-completion invariant directly from raw `OrderBooking`
specifications against the scenario's active post-shock state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from env_core.contracts import (
    ASSEMBLY_COMMISSION_WEEKS,
    BUDGET_BONUS_HEADROOM_FRAC,
    BayBlackoutWindow,
    BayId,
    CustomerOrder,
    FittingsGrade,
    OrderBooking,
    OrderVerificationResult,
    PhysicsEvaluation,
    SIM_BUDGET_DEFAULT,
    TransportRoute,
    VerificationReport,
)
from env_core.physics import evaluate_design_physics, validate_catalogue_bounds
from env_core.scenarios import ScenarioSpec, get_scenario


def _build_post_shock_world(
    scenario: ScenarioSpec,
) -> tuple[
    dict[str, CustomerOrder],
    dict[str, TransportRoute],
    float,
    int,
    float,
    dict[FittingsGrade, int],
    list[BayBlackoutWindow],
    dict[BayId, float],
]:
    dis = scenario.disruption
    orders: dict[str, CustomerOrder] = {}
    for o in scenario.orders:
        updates: dict[str, object] = {}
        if o.order_id in dis.order_min_pressure_overrides:
            updates["min_pressure_mpa"] = dis.order_min_pressure_overrides[o.order_id]
        if o.order_id in dis.order_sla_overrides:
            updates["sla_week"] = dis.order_sla_overrides[o.order_id]
        orders[o.order_id] = o.model_copy(update=updates)

    routes: dict[str, TransportRoute] = {}
    for r in scenario.routes:
        r_updates: dict[str, object] = {}
        if r.route_id in dis.route_mass_cap_overrides:
            r_updates["max_unit_transport_mass_kg"] = dis.route_mass_cap_overrides[r.route_id]
        if r.route_id in dis.route_std_height_overrides:
            r_updates["std_max_height_mm"] = dis.route_std_height_overrides[r.route_id]
        if r.route_id in dis.route_permit_allowed_overrides:
            r_updates["escort_permit_allowed"] = dis.route_permit_allowed_overrides[r.route_id]
        routes[r.route_id] = r.model_copy(update=r_updates)

    effective_budget_eur = round(scenario.total_budget_eur + dis.budget_delta_eur, 2)
    effective_permits = max(0, scenario.escort_permits_total + dis.escort_permits_delta)
    return (
        orders,
        routes,
        effective_budget_eur,
        effective_permits,
        dis.pvi_k_multiplier,
        dict(dis.fittings_lead_deltas),
        list(dis.bay_blackouts),
        dict(dis.bay_max_unit_tare_kg),
    )


def _would_pass_pre_shock(
    scenario: ScenarioSpec,
    booking: OrderBooking,
    all_bookings: dict[str, OrderBooking],
) -> bool:
    """Check if `booking` would have passed all order-level checks prior to DisruptionEvent."""
    pre_orders = {o.order_id: o for o in scenario.orders}
    pre_routes = {r.route_id: r for r in scenario.routes}
    order = pre_orders.get(booking.order_id)
    if order is None or booking.assigned_route_id != order.route_id:
        return False
    route = pre_routes.get(order.route_id)
    if route is None or booking.bay_start_week < 1:
        return False

    ev = evaluate_design_physics(order, booking.spec, pvi_k_multiplier=1.0, fittings_lead_delta=0)
    if not ev.all_engineering_pass:
        return False

    fab_end = booking.bay_start_week + ev.fab_weeks
    for other_id, other_b in all_bookings.items():
        if other_id == booking.order_id or other_b.bay_id != booking.bay_id:
            continue
        other_order = pre_orders.get(other_id)
        if other_order is None:
            continue
        other_ev = evaluate_design_physics(other_order, other_b.spec)
        other_end = other_b.bay_start_week + other_ev.fab_weeks
        if not (fab_end <= other_b.bay_start_week or other_end <= booking.bay_start_week):
            return False

    if ev.unit_transport_mass_kg > route.max_unit_transport_mass_kg:
        return False

    dims = ev.dimensions
    permits_used = sum(1 for b in all_bookings.values() if b.use_escort_permit)
    if booking.use_escort_permit:
        if (
            not route.escort_permit_allowed
            or dims.shipping_width_mm > route.permit_max_width_mm
            or dims.shipping_height_mm > route.permit_max_height_mm
            or permits_used > scenario.escort_permits_total
        ):
            return False
    else:
        if (
            dims.shipping_width_mm > route.std_max_width_mm
            or dims.shipping_height_mm > route.std_max_height_mm
        ):
            return False

    release_wk = max(1, booking.order_release_week)
    delivery_wk = (
        max(fab_end, release_wk + ev.fittings_lead_weeks)
        + ASSEMBLY_COMMISSION_WEEKS
        + route.transit_weeks
    )
    return delivery_wk <= order.sla_week and delivery_wk <= scenario.horizon_weeks


def _add_cat(cats: list[str], category: str) -> None:
    if category not in cats:
        cats.append(category)


def evaluate_episode(
    scenario: ScenarioSpec | str,
    bookings: Mapping[str, OrderBooking] | Sequence[OrderBooking],
    *,
    finalized: bool = False,
    retooling_fees_eur: float = 0.0,
    sim_calls_used: int = 0,
    sim_budget_total: int = SIM_BUDGET_DEFAULT,
    step_count: int = 0,
    disruption_triggered: bool = True,
) -> VerificationReport:
    """Zero-trust episode verification and batch scoring per ENGINE_API_SPEC Section 2.7."""
    sc = get_scenario(scenario) if isinstance(scenario, str) else scenario

    if isinstance(bookings, Mapping):
        booking_map: dict[str, OrderBooking] = dict(bookings)
    else:
        booking_map = {b.order_id: b for b in bookings}

    (
        post_orders,
        post_routes,
        effective_budget_eur,
        effective_permits,
        pvi_k_mult,
        lead_deltas,
        bay_blackouts,
        crane_caps,
    ) = _build_post_shock_world(sc)

    failure_modes: list[str] = []
    failure_categories: list[str] = []

    if not finalized:
        _add_cat(failure_categories, "UNFINALIZED")
        failure_modes.append(
            "UNFINALIZED_EPISODE: finalize_commissioning() was never called; final_score forced to 0.0."
        )

    # Precompute zero-trust physics & fab intervals for every booked known order
    recomputed_physics: dict[str, PhysicsEvaluation] = {}
    recomputed_intervals: dict[str, tuple[BayId, int, int]] = {}
    for oid, order in post_orders.items():
        b = booking_map.get(oid)
        if b is None:
            continue
        ev = evaluate_design_physics(
            order,
            b.spec,
            pvi_k_multiplier=pvi_k_mult,
            fittings_lead_delta=lead_deltas.get(b.spec.fittings, 0),
        )
        recomputed_physics[oid] = ev
        recomputed_intervals[oid] = (b.bay_id, b.bay_start_week, b.bay_start_week + ev.fab_weeks)

    permits_used = sum(
        1 for oid in post_orders if oid in booking_map and booking_map[oid].use_escort_permit
    )
    permit_quota_exceeded = permits_used > effective_permits
    if permit_quota_exceeded:
        _add_cat(failure_categories, "PERMIT_QUOTA")
        failure_modes.append(
            f"PERMIT_QUOTA_EXCEEDED: batch used {permits_used} escort permits > post-shock quota {effective_permits}."
        )

    order_results: list[OrderVerificationResult] = []
    for oid, order in post_orders.items():
        b = booking_map.get(oid)
        if b is None:
            _add_cat(failure_categories, "INCOMPLETE_BATCH")
            msg = f"{oid}: ORDER_UNBOOKED — order was never reserved on a fabrication bay."
            failure_modes.append(msg)
            order_results.append(
                OrderVerificationResult(
                    order_id=oid,
                    booked=False,
                    route_assigned=False,
                    passed_all_hard_checks=False,
                    order_raw_score=0.0,
                    weighted_contribution=0.0,
                    delivery_week=None,
                    sla_week=order.sla_week,
                    total_order_cost_eur=0.0,
                    physics=None,
                    violations=[msg],
                )
            )
            continue

        ev = recomputed_physics[oid]
        order_violations: list[str] = list(ev.violations)

        # Categorize physics violations
        if validate_catalogue_bounds(b.spec):
            _add_cat(failure_categories, "CATALOGUE_BOUNDS")
        if not ev.mass_pass:
            _add_cat(failure_categories, "CAPACITY_SHORTFALL")
        if not ev.pressure_pass or not ev.stress_pass:
            _add_cat(failure_categories, "HOOP_STRESS")
        if not ev.material_compat_pass:
            _add_cat(failure_categories, "HAZARD_COMPATIBILITY")
        if not ev.boil_off_pass:
            _add_cat(failure_categories, "BOIL_OFF")
        if not ev.site_clearance_pass:
            _add_cat(failure_categories, "SITE_CLEARANCE")

        # Bay & crane checks
        bay_id, start_wk, end_wk = recomputed_intervals[oid]
        if start_wk < 1:
            order_violations.append(f"INVALID_START_WEEK: bay_start_week={start_wk} < 1")
            _add_cat(failure_categories, "BAY_CONFLICT")

        if bay_id in crane_caps and ev.unit_tare_mass_kg > crane_caps[bay_id]:
            order_violations.append(
                f"CRANE_TARE_EXCEEDED: {bay_id.value} unit_tare_mass_kg={ev.unit_tare_mass_kg} kg "
                f"> crane limit {crane_caps[bay_id]} kg"
            )
            _add_cat(failure_categories, "BAY_CONFLICT")

        for other_oid, (other_bay, other_start, other_end) in recomputed_intervals.items():
            if other_oid == oid or other_bay != bay_id:
                continue
            if not (end_wk <= other_start or other_end <= start_wk):
                order_violations.append(
                    f"BAY_OVERLAP: {bay_id.value} [{start_wk}, {end_wk}) overlaps {other_oid} [{other_start}, {other_end})"
                )
                _add_cat(failure_categories, "BAY_CONFLICT")

        for bw in bay_blackouts:
            if bw.bay_id == bay_id and not (end_wk <= bw.start_week or bw.end_week <= start_wk):
                order_violations.append(
                    f"BAY_BLACKOUT_CONFLICT: {bay_id.value} [{start_wk}, {end_wk}) overlaps blackout "
                    f"[{bw.start_week}, {bw.end_week}) ({bw.reason})"
                )
                _add_cat(failure_categories, "BAY_CONFLICT")

        # Route & transport checks
        route_assigned = b.assigned_route_id is not None
        transport_cost_eur = 0.0
        delivery_week: int | None = None

        if not route_assigned:
            order_violations.append("ROUTE_UNASSIGNED: assign_transport was never called for this order")
            _add_cat(failure_categories, "INCOMPLETE_BATCH")
        elif b.assigned_route_id not in post_routes:
            order_violations.append(f"UNKNOWN_ROUTE: assigned_route_id='{b.assigned_route_id}' does not exist")
            _add_cat(failure_categories, "TRANSPORT_CLEARANCE")
        else:
            assigned_route = post_routes[b.assigned_route_id]
            transport_cost_eur = b.spec.unit_count * assigned_route.base_freight_eur_per_unit + (
                assigned_route.permit_fee_eur if b.use_escort_permit else 0.0
            )
            release_wk = max(1, b.order_release_week)
            delivery_week = (
                max(end_wk, release_wk + ev.fittings_lead_weeks)
                + ASSEMBLY_COMMISSION_WEEKS
                + assigned_route.transit_weeks
            )

            if b.assigned_route_id != order.route_id:
                order_violations.append(
                    f"WRONG_ROUTE: assigned_route_id='{b.assigned_route_id}' != customer route '{order.route_id}'"
                )
                _add_cat(failure_categories, "TRANSPORT_CLEARANCE")

            # Evaluate gauge & bridge limits against customer's designated route if wrong, or assigned route
            check_route = post_routes[order.route_id]
            if ev.unit_transport_mass_kg > check_route.max_unit_transport_mass_kg:
                order_violations.append(
                    f"BRIDGE_MASS_CAP_EXCEEDED: unit_transport_mass_kg={ev.unit_transport_mass_kg} kg "
                    f"> {check_route.route_id} limit {check_route.max_unit_transport_mass_kg} kg"
                )
                _add_cat(failure_categories, "BRIDGE_MASS_CAP")

            dims = ev.dimensions
            if b.use_escort_permit:
                if not check_route.escort_permit_allowed:
                    order_violations.append(
                        f"PERMIT_FORBIDDEN_ON_ROUTE: {check_route.route_id} forbids escort permits"
                    )
                    _add_cat(failure_categories, "PERMIT_QUOTA")
                if (
                    dims.shipping_width_mm > check_route.permit_max_width_mm
                    or dims.shipping_height_mm > check_route.permit_max_height_mm
                ):
                    order_violations.append(
                        f"PERMIT_GAUGE_EXCEEDED: shipping ({dims.shipping_width_mm}x{dims.shipping_height_mm} mm) "
                        f"exceeds {check_route.route_id} permit envelope "
                        f"({check_route.permit_max_width_mm}x{check_route.permit_max_height_mm} mm)"
                    )
                    _add_cat(failure_categories, "TRANSPORT_CLEARANCE")
                if permit_quota_exceeded:
                    order_violations.append(
                        f"PERMIT_QUOTA_EXCEEDED: batch used {permits_used} permits > {effective_permits} available"
                    )
                    _add_cat(failure_categories, "PERMIT_QUOTA")
            else:
                if (
                    dims.shipping_width_mm > check_route.std_max_width_mm
                    or dims.shipping_height_mm > check_route.std_max_height_mm
                ):
                    order_violations.append(
                        f"STD_GAUGE_EXCEEDED: shipping ({dims.shipping_width_mm}x{dims.shipping_height_mm} mm) "
                        f"exceeds {check_route.route_id} standard envelope "
                        f"({check_route.std_max_width_mm}x{check_route.std_max_height_mm} mm) without escort permit"
                    )
                    _add_cat(failure_categories, "TRANSPORT_CLEARANCE")

            # Schedule SLA & Horizon check
            if delivery_week > order.sla_week:
                order_violations.append(
                    f"SLA_MISSED: delivery_week={delivery_week} > customer sla_week={order.sla_week}"
                )
                _add_cat(failure_categories, "SLA_MISSED")
            if delivery_week > sc.horizon_weeks:
                order_violations.append(
                    f"HORIZON_EXCEEDED: delivery_week={delivery_week} > episode horizon_weeks={sc.horizon_weeks}"
                )
                _add_cat(failure_categories, "SLA_MISSED")

        total_order_cost = round(ev.manufacturing_cost_eur + transport_cost_eur, 2)
        passed = (
            ev.all_engineering_pass
            and route_assigned
            and b.assigned_route_id == order.route_id
            and len(order_violations) == 0
        )

        if not passed:
            if _would_pass_pre_shock(sc, b, booking_map):
                _add_cat(failure_categories, "SHOCK_UNADAPTED")
                shock_msg = (
                    f"{oid}: SHOCK_UNADAPTED — pre-shock valid design failed after {sc.disruption.disruption_id} "
                    f"({sc.disruption.title})"
                )
                failure_modes.append(shock_msg)
            for v in order_violations:
                failure_modes.append(f"{oid}: {v}")

        order_raw = 1.0 if passed else 0.0
        weighted_contrib = round(0.85 * order.priority_weight, 6) if passed else 0.0

        order_results.append(
            OrderVerificationResult(
                order_id=oid,
                booked=True,
                route_assigned=route_assigned,
                passed_all_hard_checks=passed,
                order_raw_score=order_raw,
                weighted_contribution=weighted_contrib,
                delivery_week=delivery_week,
                sla_week=order.sla_week,
                total_order_cost_eur=total_order_cost,
                physics=ev,
                violations=order_violations,
            )
        )

    total_order_count = len(sc.orders)
    valid_order_count = sum(1 for r in order_results if r.passed_all_hard_checks)
    if valid_order_count < total_order_count:
        _add_cat(failure_categories, "INCOMPLETE_BATCH")

    total_spent_eur = round(
        sum(r.total_order_cost_eur for r in order_results) + max(0.0, float(retooling_fees_eur)),
        2,
    )
    budget_exceeded = total_spent_eur > effective_budget_eur + 1e-6
    if budget_exceeded:
        _add_cat(failure_categories, "BUDGET_OVERRUN")
        failure_modes.append(
            f"BUDGET_EXCEEDED: total_spent_eur={total_spent_eur:.2f} EUR > effective_budget_eur={effective_budget_eur:.2f} EUR"
        )

    raw_score = sum(r.weighted_contribution for r in order_results)

    budget_efficiency_bonus = 0.0
    schedule_margin_bonus = 0.0
    sim_efficiency_bonus = 0.0

    if valid_order_count == total_order_count and not budget_exceeded and effective_budget_eur > 0.0:
        budget_efficiency_bonus = round(
            0.07
            * min(
                1.0,
                max(
                    0.0,
                    (effective_budget_eur - total_spent_eur)
                    / (BUDGET_BONUS_HEADROOM_FRAC * effective_budget_eur),
                ),
            ),
            4,
        )
        avg_slack_weeks = (
            sum(
                min(post_orders[r.order_id].sla_week, sc.horizon_weeks) - (r.delivery_week or 0)
                for r in order_results
            )
            / float(total_order_count)
        )
        schedule_margin_bonus = round(
            0.04 * min(1.0, max(0.0, avg_slack_weeks / 2.0)),
            4,
        )
        if sim_budget_total > 0:
            sim_efficiency_bonus = round(
                0.04 * max(0.0, (sim_budget_total - sim_calls_used) / float(sim_budget_total)),
                4,
            )

    # Batch Completion Gate
    batch_gate_cap_applied = False
    if valid_order_count == 0:
        score = 0.0
    elif valid_order_count == 1:
        score = min(raw_score, 0.08)
        batch_gate_cap_applied = True
    elif valid_order_count == 2:
        score = raw_score * 0.80
        batch_gate_cap_applied = True
    elif valid_order_count == 3:
        score = raw_score * 0.92
        batch_gate_cap_applied = True
    else:
        score = min(
            1.0,
            max(
                0.0,
                raw_score + budget_efficiency_bonus + schedule_margin_bonus + sim_efficiency_bonus,
            ),
        )

    # Budget Breach Penalty
    if budget_exceeded:
        score = min(score * 0.20, 0.15)

    # Mandatory Finalization Gate
    if not finalized:
        score = 0.0

    final_score = round(score, 4)

    return VerificationReport(
        seed=sc.seed,
        finalized=finalized,
        final_score=final_score,
        valid_order_count=valid_order_count,
        total_order_count=total_order_count,
        batch_gate_cap_applied=batch_gate_cap_applied,
        budget_exceeded=budget_exceeded,
        permit_quota_exceeded=permit_quota_exceeded,
        total_budget_eur=effective_budget_eur,
        total_spent_eur=total_spent_eur,
        budget_efficiency_bonus=budget_efficiency_bonus,
        schedule_margin_bonus=schedule_margin_bonus,
        sim_efficiency_bonus=sim_efficiency_bonus,
        sim_calls_used=sim_calls_used,
        sim_budget_total=sim_budget_total,
        step_count=step_count,
        disruption_triggered=disruption_triggered,
        order_results=order_results,
        failure_modes=failure_modes,
        primary_failure_categories=failure_categories,
    )
