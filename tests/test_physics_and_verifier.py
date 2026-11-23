"""Comprehensive pytest suite for CryoForge RL physics, handbook, scenarios, engine & verifier."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from env_core import (
    CryoForgeEngine,
    ScenarioSpec,
    evaluate_episode,
    get_handbook_section,
    get_scenario,
    list_scenarios,
)
from env_core.contracts import (
    ALL_SEEDS,
    AMBIENT_TEMP_K,
    ASSEMBLY_COMMISSION_WEEKS,
    BASE_ALLOWABLE_STRESS_MPA,
    BayId,
    EVAL_SEEDS,
     ExecutionType,
    FILL_FRACTION,
    FITTINGS_PROPS,
    FittingsGrade,
    HANDBOOK_SECTIONS,
    INSULATION_PROPS,
    InsulationType,
    MAX_EPISODE_STEPS,
    MEDIUM_PROPS,
    Medium,
    OrderBooking,
    POST_SHOCK_FIRST_REVISION_FEE_EUR,
    SIM_BUDGET_DEFAULT,
    STANDARD_REVISION_FEE_EUR,
    STANDARD_VOLUMES_M3,
    TIEBREAKER_SEEDS,
    UNIT_COUNTS_ALLOWED,
    VesselDesignSpec,
)
from env_core.physics import (
    compute_dimensions,
    compute_fab_and_cost,
    compute_masses_and_area,
    evaluate_design_physics,
    nominal_outer_diameter_mm,
    validate_catalogue_bounds,
)

FORBIDDEN_ORACLE_KEYS = {
    "hoop_stress_mpa",
    "boil_off_pct_day",
    "stress_pass",
    "boil_off_pass",
}


def _collect_all_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found |= _collect_all_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            found |= _collect_all_keys(item)
    return found


def _find_order_candidates(
    order,
    route,
    *,
    pvi_k: float = 1.0,
    lead_deltas: dict[FittingsGrade, int] | None = None,
    release_week: int = 1,
    horizon_weeks: int = 14,
):
    lead_deltas = lead_deltas or {}
    candidates = []
    thicknesses = [round(4.0 + 0.5 * i, 1) for i in range(41)]
    design_p = round(order.min_pressure_mpa, 2)

    for uc in UNIT_COUNTS_ALLOWED:
        for vol in STANDARD_VOLUMES_M3:
            for ins in InsulationType:
                for fit in FittingsGrade:
                    for wt in thicknesses:
                        spec = VesselDesignSpec(
                            unit_count=uc,
                            unit_volume_m3=vol,
                            design_pressure_mpa=design_p,
                            wall_thickness_mm=wt,
                            insulation=ins,
                            fittings=fit,
                        )
                        ev = evaluate_design_physics(
                            order,
                            spec,
                            pvi_k_multiplier=pvi_k,
                            fittings_lead_delta=lead_deltas.get(fit, 0),
                        )
                        if not ev.all_engineering_pass:
                            continue
                        if ev.unit_transport_mass_kg > route.max_unit_transport_mass_kg:
                            break
                        dims = ev.dimensions
                        std_ok = (
                            dims.shipping_width_mm <= route.std_max_width_mm
                            and dims.shipping_height_mm <= route.std_max_height_mm
                        )
                        perm_ok = (
                            route.escort_permit_allowed
                            and dims.shipping_width_mm <= route.permit_max_width_mm
                            and dims.shipping_height_mm <= route.permit_max_height_mm
                        )
                        if not std_ok and not perm_ok:
                            break
                        need_perm = not std_ok
                        tcost = uc * route.base_freight_eur_per_unit + (
                            route.permit_fee_eur if need_perm else 0.0
                        )
                        total_cost = round(ev.manufacturing_cost_eur + tcost, 2)
                        deadline = min(order.sla_week, horizon_weeks)
                        max_fab_end = deadline - ASSEMBLY_COMMISSION_WEEKS - route.transit_weeks
                        fit_ready = release_week + ev.fittings_lead_weeks
                        if fit_ready > max_fab_end or 1 + ev.fab_weeks > max_fab_end:
                            break
                        candidates.append((total_cost, need_perm, spec, ev, max_fab_end, fit_ready))
                        break
    candidates.sort(key=lambda x: x[0])
    return candidates


def _solve_batch_schedule(
    orders,
    routes,
    horizon_weeks: int,
    eff_permits: int,
    blackouts,
    crane_caps,
    pvi_k: float,
    lead_deltas,
    release_weeks: dict[str, int],
):
    oids = list(orders.keys())
    per_order = {}
    for oid in oids:
        cands = _find_order_candidates(
            orders[oid],
            routes[orders[oid].route_id],
            pvi_k=pvi_k,
            lead_deltas=lead_deltas,
            release_week=release_weeks[oid],
            horizon_weeks=horizon_weeks,
        )
        per_order[oid] = [c for c in cands if not c[1]][:5] + [c for c in cands if c[1]][:4]

    best = None
    for combo in itertools.product(*(per_order[oid] for oid in oids)):
        if sum(1 for c in combo if c[1]) > eff_permits:
            continue
        total_cost = sum(c[0] for c in combo)
        for bays in itertools.product([BayId.BAY_A, BayId.BAY_B], repeat=4):
            if any(
                bays[i] in crane_caps and combo[i][3].unit_tare_mass_kg > crane_caps[bays[i]]
                for i in range(4)
            ):
                continue
            bay_lists = {BayId.BAY_A: [], BayId.BAY_B: []}
            for i, b in enumerate(bays):
                bay_lists[b].append(i)

            def _sched_single_bay(bay_id: BayId, idxs: list[int]):
                if not idxs:
                    return {}
                best_sched = None
                best_slack = -999
                bwins = [bw for bw in blackouts if bw.bay_id == bay_id]
                for perm in itertools.permutations(idxs):
                    cur = 1
                    sched = {}
                    possible = True
                    slack = 0
                    for idx in perm:
                        _, need_perm, spec, ev, max_fab_end, fit_ready = combo[idx]
                        st = cur
                        while True:
                            en = st + ev.fab_weeks
                            if en > max_fab_end:
                                possible = False
                                break
                            if not any(
                                not (en <= bw.start_week or st >= bw.end_week) for bw in bwins
                            ):
                                break
                            st += 1
                        if not possible:
                            break
                        en = st + ev.fab_weeks
                        o = orders[oids[idx]]
                        r = routes[o.route_id]
                        deliv = max(en, fit_ready) + ASSEMBLY_COMMISSION_WEEKS + r.transit_weeks
                        slack += min(o.sla_week, horizon_weeks) - deliv
                        sched[oids[idx]] = (bay_id, st, en, deliv, need_perm, spec, ev)
                        cur = en
                    if possible and slack > best_slack:
                        best_slack = slack
                        best_sched = sched
                return best_sched

            sa = _sched_single_bay(BayId.BAY_A, bay_lists[BayId.BAY_A])
            sb = _sched_single_bay(BayId.BAY_B, bay_lists[BayId.BAY_B])
            if sa is None or sb is None:
                continue
            merged = {**sa, **sb}
            avg_slack = (
                sum(min(orders[oid].sla_week, horizon_weeks) - merged[oid][3] for oid in oids)
                / 4.0
            )
            key = (total_cost, -avg_slack)
            if best is None or key < best[0]:
                best = (key, total_cost, avg_slack, merged)
    return best


# ---------------------------------------------------------------------------
# 1. Physics & Hazard Compatibility Tests
# ---------------------------------------------------------------------------


def test_physics_geometry_and_catalogue_bounds() -> None:
    assert nominal_outer_diameter_mm(2.0) == 1400
    assert nominal_outer_diameter_mm(5.0) == 1600
    assert nominal_outer_diameter_mm(10.0) == 2000
    assert nominal_outer_diameter_mm(15.0) == 2400
    assert nominal_outer_diameter_mm(25.0) == 2600
    assert nominal_outer_diameter_mm(30.0) == 3000

    d_stat = compute_dimensions(15.0, ExecutionType.STATIONARY, InsulationType.EVI)
    assert d_stat.outer_diameter_mm == 2400
    assert d_stat.inner_diameter_mm == 2160
    assert d_stat.shipping_width_mm == 2400
    assert d_stat.shipping_height_mm == 2750

    d_trans = compute_dimensions(15.0, ExecutionType.TRANSPORTABLE, InsulationType.EVI)
    assert d_trans.shipping_width_mm == 2600
    assert d_trans.shipping_height_mm == 3000

    bad_vol_spec = VesselDesignSpec(
        unit_count=1,
        unit_volume_m3=12.5,
        design_pressure_mpa=1.5,
        wall_thickness_mm=8.0,
        insulation=InsulationType.EVI,
        fittings=FittingsGrade.CN,
    )
    errs = validate_catalogue_bounds(bad_vol_spec)
    assert any("not a standard shell volume" in e for e in errs)


def test_hazard_compatibility_and_pvi_multiplier() -> None:
    sc = get_scenario("S01")
    lh2_order = sc.orders[2]  # ORD-3 is LH2
    assert lh2_order.medium == Medium.LH2

    bad_lh2_spec = VesselDesignSpec(
        unit_count=1,
        unit_volume_m3=15.0,
        design_pressure_mpa=1.8,
        wall_thickness_mm=12.0,
        insulation=InsulationType.PVI,
        fittings=FittingsGrade.CN,
    )
    ev_bad = evaluate_design_physics(lh2_order, bad_lh2_spec)
    assert not ev_bad.material_compat_pass
    assert not ev_bad.all_engineering_pass
    assert any("HAZARD_INSULATION" in v for v in ev_bad.violations)
    assert any("HAZARD_FITTINGS" in v for v in ev_bad.violations)

    lox_order = sc.orders[3]  # ORD-4 is LOX with min_pressure 2.4 > 2.0
    lox_cn_spec = VesselDesignSpec(
        unit_count=1,
        unit_volume_m3=20.0,
        design_pressure_mpa=2.4,
        wall_thickness_mm=18.0,
        insulation=InsulationType.EVI,
        fittings=FittingsGrade.CN,
    )
    ev_lox = evaluate_design_physics(lox_order, lox_cn_spec)
    assert not ev_lox.material_compat_pass
    assert any("LOX_IGNITION_RISK" in v for v in ev_lox.violations)

    # PVI k_eff multiplier affects PVI boil-off and leaves EVI untouched
    lin_order = sc.orders[0]
    pvi_spec = VesselDesignSpec(
        unit_count=1,
        unit_volume_m3=10.0,
        design_pressure_mpa=1.5,
        wall_thickness_mm=8.0,
        insulation=InsulationType.PVI,
        fittings=FittingsGrade.CN,
    )
    evi_spec = pvi_spec.model_copy(update={"insulation": InsulationType.EVI})
    assert (
        evaluate_design_physics(lin_order, pvi_spec, pvi_k_multiplier=1.55).boil_off_pct_day
        > evaluate_design_physics(lin_order, pvi_spec, pvi_k_multiplier=1.0).boil_off_pct_day
    )
    assert (
        evaluate_design_physics(lin_order, evi_spec, pvi_k_multiplier=1.55).boil_off_pct_day
        == evaluate_design_physics(lin_order, evi_spec, pvi_k_multiplier=1.0).boil_off_pct_day
    )


# ---------------------------------------------------------------------------
# 2. Handbook Completeness & Zero-Drift Tests
# ---------------------------------------------------------------------------


def test_handbook_sections_and_zero_drift_against_contracts() -> None:
    for sec in HANDBOOK_SECTIONS:
        data = get_handbook_section(sec)
        assert data["section"] == sec
        # Must be 100% JSON serializable without custom encoders
        encoded = json.dumps(data)
        assert len(encoded) > 100

    with pytest.raises(ValueError, match="Unknown handbook section"):
        get_handbook_section("nonexistent_section")

    full = get_handbook_section("all")["sections"]
    assert full["overview"]["catalogue_bounds"]["standard_volumes_m3"] == list(STANDARD_VOLUMES_M3)
    assert full["overview"]["catalogue_bounds"]["unit_counts_allowed"] == list(UNIT_COUNTS_ALLOWED)
    assert full["physics"]["constants"]["fill_fraction"] == FILL_FRACTION
    assert full["physics"]["constants"]["ambient_temp_k"] == AMBIENT_TEMP_K
    assert full["physics"]["constants"]["base_allowable_stress_mpa"] == BASE_ALLOWABLE_STRESS_MPA
    assert (
        full["manufacturing_and_bays"]["constants"]["assembly_commission_weeks"]
        == ASSEMBLY_COMMISSION_WEEKS
    )
    assert (
        full["disruptions_and_revisions"]["constants"]["post_shock_first_revision_fee_eur"]
        == POST_SHOCK_FIRST_REVISION_FEE_EUR
    )
    assert (
        full["disruptions_and_revisions"]["constants"]["standard_revision_fee_eur"]
        == STANDARD_REVISION_FEE_EUR
    )

    # Zero drift on MEDIUM_PROPS, INSULATION_PROPS, FITTINGS_PROPS
    mat = full["materials_and_insulation"]
    for m, props in MEDIUM_PROPS.items():
        h_m = mat["medium_props"][m.value]
        assert h_m["density_kg_m3"] == props["density_kg_m3"]
        assert h_m["boil_temp_k"] == props["boil_temp_k"]
        assert h_m["latent_heat_kj_kg"] == props["latent_heat_kj_kg"]
        assert h_m["safety_factor"] == props["safety_factor"]
        assert h_m["requires_evi"] == props["requires_evi"]

    for ins, props in INSULATION_PROPS.items():
        h_i = mat["insulation_props"][ins.value]
        assert h_i["annulus_mm"] == props["annulus_mm"]
        assert h_i["k_eff_w_mk"] == props["k_eff_w_mk"]
        assert h_i["cost_multiplier"] == props["cost_multiplier"]

    for fit, props in FITTINGS_PROPS.items():
        h_f = mat["fittings_props"][fit.value]
        assert h_f["weld_efficiency"] == props["weld_efficiency"]
        assert h_f["valve_heat_leak_w"] == props["valve_heat_leak_w"]
        assert h_f["base_lead_weeks"] == props["base_lead_weeks"]
        assert h_f["fixed_kit_eur"] == props["fixed_kit_eur"]


# ---------------------------------------------------------------------------
# 3. All 8 Scenarios Feasibility & Post-Shock Adaptation Proof
# ---------------------------------------------------------------------------


def test_scenario_registry_structure() -> None:
    eval_list = list_scenarios(include_tiebreaker=False)
    all_list = list_scenarios(include_tiebreaker=True)
    assert [s.seed for s in eval_list] == list(EVAL_SEEDS)
    assert [s.seed for s in all_list] == list(ALL_SEEDS)
    assert len(TIEBREAKER_SEEDS) == 3

    seen_disruptions = set()
    for sc in all_list:
        assert len(sc.orders) == 4
        assert [o.order_id for o in sc.orders] == ["ORD-1", "ORD-2", "ORD-3", "ORD-4"]
        assert all(abs(o.priority_weight - 0.25) < 1e-9 for o in sc.orders)
        seen_disruptions.add(sc.disruption.disruption_id)
    assert len(seen_disruptions) == 8


@pytest.mark.parametrize("seed", ALL_SEEDS)
def test_every_scenario_online_post_shock_feasibility_and_shock_bite(
    seed: str, tmp_path: Path
) -> None:
    sc: ScenarioSpec = get_scenario(seed)
    pre_orders = {o.order_id: o for o in sc.orders}
    pre_routes = {r.route_id: r for r in sc.routes}

    pre_plan = _solve_batch_schedule(
        pre_orders,
        pre_routes,
        sc.horizon_weeks,
        sc.escort_permits_total,
        [],
        {},
        1.0,
        {},
        {"ORD-1": 1, "ORD-2": 1, "ORD-3": 1, "ORD-4": 1},
    )
    assert pre_plan is not None, f"{seed} pre-shock schedule missing"

    # Prove ignoring the mid-episode shock fails post-shock verification
    unadapted_bookings = {
        oid: OrderBooking(
            order_id=oid,
            bay_id=bay_id,
            bay_start_week=st,
            fab_weeks=ev.fab_weeks,
            fab_end_week=en,
            booked_at_step=1,
            order_release_week=(1 if oid in ("ORD-1", "ORD-2") else 2),
            spec=spec,
            manufacturing_cost_eur=ev.manufacturing_cost_eur,
            assigned_route_id=pre_orders[oid].route_id,
            use_escort_permit=need_perm,
        )
        for oid, (bay_id, st, en, _, need_perm, spec, ev) in pre_plan[3].items()
    }
    unadapted_report = evaluate_episode(sc, unadapted_bookings, finalized=True)
    assert unadapted_report.valid_order_count < 4
    assert "SHOCK_UNADAPTED" in unadapted_report.primary_failure_categories
    assert unadapted_report.final_score < 0.60

    # Prove online adapted trajectory through CryoForgeEngine scores >= 0.94 with 12-22% headroom
    eng = CryoForgeEngine(
        seed=seed,
        run_id=f"test_{seed}",
        agent_name="verified_solver",
        agent_type="reference",
        log_dir=tmp_path,
    )
    brief = eng.get_episode_brief()
    assert brief["ok"] and not brief["disruption_triggered"]

    for oid in ("ORD-1", "ORD-2"):
        bay_id, st, _, _, _, spec, _ = pre_plan[3][oid]
        res = eng.reserve_bay_and_order(
            order_id=oid,
            bay_id=bay_id.value,
            bay_start_week=st,
            unit_count=spec.unit_count,
            unit_volume_m3=spec.unit_volume_m3,
            design_pressure_mpa=spec.design_pressure_mpa,
            wall_thickness_mm=spec.wall_thickness_mm,
            insulation=spec.insulation.value,
            fittings=spec.fittings.value,
        )
        assert res["ok"], f"{seed} pre-shock reserve failed for {oid}: {res}"
        assert not (FORBIDDEN_ORACLE_KEYS & _collect_all_keys(res))

    assert eng.disruption_triggered

    post_plan = _solve_batch_schedule(
        eng.active_orders,
        eng.active_routes,
        sc.horizon_weeks,
        eng.effective_escort_permits_total,
        eng.active_bay_blackouts,
        eng.active_bay_max_unit_tare_kg,
        eng.active_pvi_k_multiplier,
        eng.active_fittings_lead_deltas,
        {"ORD-1": 1, "ORD-2": 2, "ORD-3": 2, "ORD-4": 2},
    )
    assert post_plan is not None, f"{seed} post-shock schedule missing"

    for oid in ("ORD-1", "ORD-2"):
        bay_id, st, _, _, need_perm, spec, _ = post_plan[3][oid]
        cur = eng.booked_orders[oid]
        if cur.bay_id != bay_id or cur.bay_start_week != st or cur.spec != spec:
            rev = eng.revise_order(
                order_id=oid,
                bay_id=bay_id.value,
                bay_start_week=st,
                unit_count=spec.unit_count,
                unit_volume_m3=spec.unit_volume_m3,
                design_pressure_mpa=spec.design_pressure_mpa,
                wall_thickness_mm=spec.wall_thickness_mm,
                insulation=spec.insulation.value,
                fittings=spec.fittings.value,
                route_id=eng.active_orders[oid].route_id,
                use_escort_permit=need_perm,
            )
            assert rev["ok"], f"{seed} revise failed for {oid}: {rev}"
            assert not (FORBIDDEN_ORACLE_KEYS & _collect_all_keys(rev))
        else:
            at = eng.assign_transport(oid, eng.active_orders[oid].route_id, need_perm)
            assert at["ok"]

    for oid in ("ORD-3", "ORD-4"):
        bay_id, st, _, _, need_perm, spec, _ = post_plan[3][oid]
        rb = eng.reserve_bay_and_order(
            order_id=oid,
            bay_id=bay_id.value,
            bay_start_week=st,
            unit_count=spec.unit_count,
            unit_volume_m3=spec.unit_volume_m3,
            design_pressure_mpa=spec.design_pressure_mpa,
            wall_thickness_mm=spec.wall_thickness_mm,
            insulation=spec.insulation.value,
            fittings=spec.fittings.value,
        )
        assert rb["ok"]
        at = eng.assign_transport(oid, eng.active_orders[oid].route_id, need_perm)
        assert at["ok"]

    fin = eng.finalize_commissioning()
    assert fin["ok"]
    assert fin["valid_order_count"] == 4
    assert fin["final_score"] >= 0.94

    saved_file = tmp_path / f"test_{seed}.json"
    assert saved_file.exists()
    traj = eng.export_trajectory_log()
    assert eng.verify_trajectory_hmac(traj)

    headroom_frac = (
        traj.verification.total_budget_eur - traj.verification.total_spent_eur
    ) / traj.verification.total_budget_eur
    assert 0.02 <= headroom_frac <= 0.08, f"{seed} headroom={headroom_frac:.3f} outside [0.02, 0.08]"


# ---------------------------------------------------------------------------
# 4. Baselines, Anti-Oracle, Sim/Step Budgets, Immutability & Penalties
# ---------------------------------------------------------------------------


def test_do_nothing_and_unfinalized_and_stop_early_gates() -> None:
    # 1. Unfinalized do-nothing -> 0.0
    eng_unfin = CryoForgeEngine("S01", agent_type="do_nothing")
    eng_unfin.get_episode_brief()
    log_unfin = eng_unfin.export_trajectory_log()
    assert log_unfin.verification.final_score == 0.0
    assert "UNFINALIZED" in log_unfin.verification.primary_failure_categories

    # 2. Finalized do-nothing -> 0.0
    eng_dn = CryoForgeEngine("S01", agent_type="do_nothing")
    fin_dn = eng_dn.finalize_commissioning()
    assert fin_dn["final_score"] == 0.0
    assert fin_dn["valid_order_count"] == 0

    # 3. 1-order stop_early -> capped at <= 0.08
    eng_se = CryoForgeEngine("S01", agent_type="stop_early")
    r1 = eng_se.reserve_bay_and_order(
        order_id="ORD-1",
        bay_id="BAY-A",
        bay_start_week=1,
        unit_count=1,
        unit_volume_m3=10.0,
        design_pressure_mpa=1.5,
        wall_thickness_mm=7.0,
        insulation="EVI",
        fittings="CN",
    )
    assert r1["ok"]
    a1 = eng_se.assign_transport("ORD-1", "R-CENTRAL", False)
    assert a1["ok"]
    # Before finalizing, score is 0.0 even though ORD-1 is valid
    assert eng_se.export_trajectory_log().verification.final_score == 0.0
    fin_se = eng_se.finalize_commissioning()
    assert fin_se["valid_order_count"] == 1
    assert fin_se["final_score"] == 0.08
    assert fin_se["verification"]["batch_gate_cap_applied"] is True


def test_anti_oracle_accepts_failing_stress_and_boiloff_without_leaking() -> None:
    eng = CryoForgeEngine("S01", agent_type="exploit")
    # Book a wildly under-thick PVI shell on LH2 order ORD-3: passes catalogue bounds, fails physics!
    resp = eng.reserve_bay_and_order(
        order_id="ORD-3",
        bay_id="BAY-A",
        bay_start_week=1,
        unit_count=1,
        unit_volume_m3=1.0,
        design_pressure_mpa=0.5,
        wall_thickness_mm=4.0,
        insulation="PVI",
        fittings="CN",
    )
    assert resp["ok"] is True
    assert not (FORBIDDEN_ORACLE_KEYS & _collect_all_keys(resp))
    assert not (FORBIDDEN_ORACLE_KEYS & _collect_all_keys(eng.get_factory_state()))
    assert not (FORBIDDEN_ORACLE_KEYS & _collect_all_keys(eng.inspect_order("ORD-3")))

    eng.assign_transport("ORD-3", "R-CENTRAL", False)
    fin = eng.finalize_commissioning()
    assert fin["valid_order_count"] == 0
    assert fin["final_score"] == 0.0
    cats = fin["verification"]["primary_failure_categories"]
    assert "HAZARD_COMPATIBILITY" in cats
    assert "BOIL_OFF" in cats


def test_sim_budget_exhaustion_and_post_finalization_immutability() -> None:
    eng = CryoForgeEngine("S01")
    for expected_rem in range(SIM_BUDGET_DEFAULT - 1, -1, -1):
        res = eng.run_physics_sim(
            order_id="ORD-1",
            unit_count=1,
            unit_volume_m3=10.0,
            design_pressure_mpa=1.5,
            wall_thickness_mm=7.0,
            insulation="PVI",
            fittings="CN",
        )
        assert res["ok"] is True
        assert res["sim_calls_remaining"] == expected_rem
        assert "hoop_stress_mpa" in res["physics"]

    # 9th call must fail with SIM_BUDGET_EXHAUSTED
    exhausted = eng.run_physics_sim(
        order_id="ORD-1",
        unit_count=1,
        unit_volume_m3=10.0,
        design_pressure_mpa=1.5,
        wall_thickness_mm=7.0,
        insulation="EVI",
        fittings="CN",
    )
    assert exhausted["ok"] is False
    assert exhausted["error"] == "SIM_BUDGET_EXHAUSTED"
    assert exhausted["sim_calls_remaining"] == 0

    fin = eng.finalize_commissioning()
    assert fin["ok"] is True
    log_before = eng.export_trajectory_log()

    # All 5 mutating methods must reject post-finalization without altering state or HMAC
    assert eng.run_physics_sim("ORD-1", 1, 10.0, 1.5, 7.0, "PVI", "CN")["error"] == "EPISODE_ALREADY_FINALIZED"
    assert eng.reserve_bay_and_order("ORD-1", "BAY-A", 1, 1, 10.0, 1.5, 7.0, "PVI", "CN")["error"] == "EPISODE_ALREADY_FINALIZED"
    assert eng.assign_transport("ORD-1", "R-CENTRAL", False)["error"] == "EPISODE_ALREADY_FINALIZED"
    assert eng.revise_order("ORD-1", "BAY-A", 1, 1, 10.0, 1.5, 7.0, "PVI", "CN")["error"] == "EPISODE_ALREADY_FINALIZED"
    assert eng.finalize_commissioning()["error"] == "EPISODE_ALREADY_FINALIZED"

    log_after = eng.export_trajectory_log()
    assert log_after.state_hmac == log_before.state_hmac
    assert log_after.step_count if hasattr(log_after, "step_count") else log_after.verification.step_count == log_before.verification.step_count


def test_disruption_trigger_on_fourth_mutation_and_revision_fees() -> None:
    eng = CryoForgeEngine("S03")
    # 1st mutation: reserve ORD-1
    m1 = eng.reserve_bay_and_order("ORD-1", "BAY-A", 1, 1, 10.0, 1.8, 8.0, "PVI", "CN")
    assert m1["ok"] and not eng.disruption_triggered
    # 2nd mutation: assign transport ORD-1
    m2 = eng.assign_transport("ORD-1", "R-EAST", False)
    assert m2["ok"] and not eng.disruption_triggered
    # 3rd mutation: pre-shock revise ORD-1 -> charges STANDARD_REVISION_FEE_EUR (4500)
    m3 = eng.revise_order("ORD-1", "BAY-A", 1, 1, 10.0, 1.8, 8.5, "PVI", "CN")
    assert m3["ok"] and not eng.disruption_triggered
    assert m3["revision_fee_charged_eur"] == STANDARD_REVISION_FEE_EUR
    # 4th mutation on single order triggers disruption!
    m4 = eng.assign_transport("ORD-1", "R-EAST", False)
    assert m4["ok"] and eng.disruption_triggered
    assert m4["disruption_alert"]["disruption_id"] == "DIS-S03"

    # First post-shock revision of ORD-1 charges POST_SHOCK_FIRST_REVISION_FEE_EUR (1500)
    m5 = eng.revise_order("ORD-1", "BAY-A", 1, 1, 10.0, 1.8, 8.5, "EVI", "EU_STD")
    assert m5["ok"]
    assert m5["revision_fee_charged_eur"] == POST_SHOCK_FIRST_REVISION_FEE_EUR
    assert m5["booking"]["order_release_week"] == 2  # fittings changed CN -> EU_STD post-shock!

    # Second post-shock revision of ORD-1 charges STANDARD_REVISION_FEE_EUR (4500)
    m6 = eng.revise_order("ORD-1", "BAY-A", 1, 1, 10.0, 1.8, 9.0, "EVI", "EU_STD")
    assert m6["ok"]
    assert m6["revision_fee_charged_eur"] == STANDARD_REVISION_FEE_EUR


def test_budget_overrun_penalty_and_permit_quota_and_step_limit() -> None:
    eng_solve = CryoForgeEngine("S05")
    eng_solve._apply_disruption()
    post_plan = _solve_batch_schedule(
        eng_solve.active_orders,
        eng_solve.active_routes,
        eng_solve.scenario.horizon_weeks,
        eng_solve.effective_escort_permits_total,
        eng_solve.active_bay_blackouts,
        eng_solve.active_bay_max_unit_tare_kg,
        eng_solve.active_pvi_k_multiplier,
        eng_solve.active_fittings_lead_deltas,
        {"ORD-1": 1, "ORD-2": 2, "ORD-3": 2, "ORD-4": 2},
    )
    assert post_plan is not None

    # Set fake manufacturing_cost_eur=1.0 on valid bookings to prove zero-trust recomputation,
    # plus high retooling_fees_eur to trigger budget_exceeded on a 4/4 valid batch!
    over_budget_bookings = {
        oid: OrderBooking(
            order_id=oid,
            bay_id=bay_id,
            bay_start_week=st,
            fab_weeks=1,  # deliberately tampered to prove zero-trust fab_weeks recomputation
            fab_end_week=st + 1,
            booked_at_step=1,
            order_release_week=(1 if oid == "ORD-1" else 2),
            spec=spec,
            manufacturing_cost_eur=1.0,  # deliberately tampered to prove zero-trust cost recomputation
            assigned_route_id=eng_solve.active_orders[oid].route_id,
            use_escort_permit=need_perm,
        )
        for oid, (bay_id, st, _, _, need_perm, spec, _) in post_plan[3].items()
    }
    rep = evaluate_episode(
        eng_solve.scenario,
        over_budget_bookings,
        finalized=True,
        retooling_fees_eur=80000.0,
    )
    assert rep.valid_order_count == 4
    assert rep.budget_exceeded is True
    assert rep.total_spent_eur > rep.total_budget_eur
    assert rep.final_score == 0.15  # 0.85 * 0.20 = 0.17 -> capped at 0.15!
    assert "BUDGET_OVERRUN" in rep.primary_failure_categories

    # Also test permit quota breach: claim escort permit on all 4 orders when quota is 1
    permit_breach_bookings = {
        oid: b.model_copy(update={"use_escort_permit": True})
        for oid, b in over_budget_bookings.items()
    }
    rep_perm = evaluate_episode(eng_solve.scenario, permit_breach_bookings, finalized=True)
    assert rep_perm.permit_quota_exceeded is True
    assert rep_perm.valid_order_count == 0
    assert "PERMIT_QUOTA" in rep_perm.primary_failure_categories

    # Input fuzzing (NaN / Inf / bool) & MAX_EPISODE_STEPS
    eng = CryoForgeEngine("S01", agent_type="exploit")
    nan_res = eng.reserve_bay_and_order("ORD-1", "BAY-A", 1, 1, 10.0, float("nan"), 8.0, "EVI", "CN")
    assert nan_res["ok"] is False and nan_res["error"] == "INVALID_SPEC"
    bool_res = eng.reserve_bay_and_order("ORD-1", "BAY-A", True, 1, 10.0, 1.5, 8.0, "EVI", "CN")  # type: ignore[arg-type]
    assert bool_res["ok"] is False and bool_res["error"] == "INVALID_START_WEEK"

    while eng.step_count < MAX_EPISODE_STEPS:
        assert eng.get_factory_state()["ok"] is True
    over = eng.get_factory_state()
    assert over["ok"] is False and over["error"] == "MAX_STEPS_EXCEEDED"
    # Exported log must remain strictly valid JSON even after NaN fuzzing
    raw_json = eng.export_trajectory_log().model_dump_json()
    assert json.loads(raw_json)["seed"] == "S01"
