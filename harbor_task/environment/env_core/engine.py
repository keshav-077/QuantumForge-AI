"""Stateful CryoForge RL Commissioning Engine.

Implements all 10 agent tool methods, anti-oracle guarantees, mid-episode supply/infrastructure
disruption triggers, revision retooling fee accounting, step/simulation budgets, post-finalization
immutability, and HMAC-signed trajectory export.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
from pathlib import Path
import secrets
from typing import Any
import uuid

from pydantic import ValidationError

from env_core.contracts import (
    ASSEMBLY_COMMISSION_WEEKS,
    BayBlackoutWindow,
    BayId,
    CustomerOrder,
    DISRUPTION_TRIGGER_BOOKINGS,
    DISRUPTION_TRIGGER_MUTATIONS,
    FITTINGS_PROPS,
    FittingsGrade,
    HANDBOOK_SECTIONS,
    InsulationType,
    MAX_EPISODE_STEPS,
    OrderBooking,
    POST_SHOCK_FIRST_REVISION_FEE_EUR,
    SIM_BUDGET_DEFAULT,
    STANDARD_REVISION_FEE_EUR,
    STANDARD_VOLUMES_M3,
    TrajectoryLog,
    TrajectoryStepLog,
    TransportRoute,
    UNIT_COUNTS_ALLOWED,
    VerificationReport,
    VesselDesignSpec,
)
from env_core.handbook import get_handbook_section
from env_core.physics import (
    compute_dimensions,
    compute_fab_and_cost,
    compute_masses_and_area,
    evaluate_design_physics,
    validate_catalogue_bounds,
)
from env_core.scenarios import ScenarioSpec, get_scenario
from env_core.verifier import evaluate_episode

_VALID_AGENT_TYPES = {
    "reference",
    "frontier",
    "do_nothing",
    "stop_early",
    "small_before",
    "small_after",
    "exploit",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_json_value(val: Any) -> Any:
    """Ensure logged tool arguments are 100% RFC-8259 JSON-safe even on NaN/Inf fuzzing."""
    if isinstance(val, float):
        if not math.isfinite(val):
            return str(val)
        return val
    if isinstance(val, dict):
        return {str(k): _sanitize_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_sanitize_json_value(v) for v in val]
    if isinstance(val, Path):
        return str(val)
    return val


class CryoForgeEngine:
    """Stateful single-episode environment engine for CryoForge RL."""

    def __init__(
        self,
        seed: str = "S01",
        *,
        run_id: str | None = None,
        agent_name: str = "agent",
        agent_type: str = "frontier",
        model_id: str = "unknown",
        log_dir: str | Path | None = None,
    ) -> None:
        self.scenario: ScenarioSpec = get_scenario(seed)
        self.seed: str = self.scenario.seed
        self.run_id: str = run_id or f"{self.seed}_{agent_name}_{uuid.uuid4().hex[:8]}"
        self.agent_name: str = agent_name
        self.agent_type: str = agent_type if agent_type in _VALID_AGENT_TYPES else "frontier"
        self.model_id: str = model_id
        self.log_dir: Path | None = Path(log_dir) if log_dir is not None else None

        self.started_at: str = _utc_now_iso()
        self.completed_at: str | None = None
        self._hmac_secret: bytes = secrets.token_bytes(32)

        # Episode counters & budgets
        self.step_count: int = 0
        self.mutation_count: int = 0
        self.sim_budget_total: int = SIM_BUDGET_DEFAULT
        self.sim_calls_remaining: int = SIM_BUDGET_DEFAULT
        self.sim_calls_used: int = 0
        self.finalized: bool = False
        self.disruption_triggered: bool = False
        self.retooling_fees_eur: float = 0.0

        # Active dynamic world state (mutated in-place when DisruptionEvent fires)
        self.active_orders: dict[str, CustomerOrder] = {
            o.order_id: o.model_copy(deep=True) for o in self.scenario.orders
        }
        self.active_routes: dict[str, TransportRoute] = {
            r.route_id: r.model_copy(deep=True) for r in self.scenario.routes
        }
        self.effective_budget_eur: float = float(self.scenario.total_budget_eur)
        self.effective_escort_permits_total: int = int(self.scenario.escort_permits_total)
        self.active_fittings_lead_deltas: dict[FittingsGrade, int] = {}
        self.active_pvi_k_multiplier: float = 1.0
        self.active_bay_blackouts: list[BayBlackoutWindow] = []
        self.active_bay_max_unit_tare_kg: dict[BayId, float] = {}

        # Active factory bookings & step audit log
        self.booked_orders: dict[str, OrderBooking] = {}
        self.steps: list[TrajectoryStepLog] = []

    @property
    def current_factory_week(self) -> int:
        return 2 if self.disruption_triggered else 1

    @property
    def escort_permits_used(self) -> int:
        return sum(1 for b in self.booked_orders.values() if b.use_escort_permit)

    def _order_transport_cost(self, booking: OrderBooking) -> float:
        if booking.assigned_route_id is None or booking.assigned_route_id not in self.active_routes:
            return 0.0
        route = self.active_routes[booking.assigned_route_id]
        return round(
            booking.spec.unit_count * route.base_freight_eur_per_unit
            + (route.permit_fee_eur if booking.use_escort_permit else 0.0),
            2,
        )

    @property
    def total_spent_eur(self) -> float:
        orders_cost = sum(
            b.manufacturing_cost_eur + self._order_transport_cost(b)
            for b in self.booked_orders.values()
        )
        return round(orders_cost + self.retooling_fees_eur, 2)

    @property
    def budget_remaining_eur(self) -> float:
        return round(self.effective_budget_eur - self.total_spent_eur, 2)

    def _record_step(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        observation: dict[str, Any],
        *,
        disruption_triggered_this_step: bool = False,
    ) -> dict[str, Any]:
        self.steps.append(
            TrajectoryStepLog(
                step_index=self.step_count,
                timestamp_iso=_utc_now_iso(),
                tool_name=tool_name,
                arguments=_sanitize_json_value(arguments),
                observation=observation,
                disruption_triggered_this_step=disruption_triggered_this_step,
            )
        )
        return observation

    def _check_step_budget(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Increment step_count and enforce MAX_EPISODE_STEPS (42)."""
        self.step_count += 1
        if self.step_count > MAX_EPISODE_STEPS:
            obs = {
                "ok": False,
                "error": "MAX_STEPS_EXCEEDED",
                "message": f"Episode step limit ({MAX_EPISODE_STEPS}) exceeded.",
                "step_count": self.step_count,
                "max_steps": MAX_EPISODE_STEPS,
            }
            return self._record_step(tool_name, arguments, obs)
        return None

    def _apply_disruption(self) -> dict[str, Any]:
        """Apply scenario's DisruptionEvent to active world state and return serialized alert."""
        dis = self.scenario.disruption
        self.disruption_triggered = True

        self.effective_budget_eur = round(self.scenario.total_budget_eur + dis.budget_delta_eur, 2)
        self.effective_escort_permits_total = max(
            0, self.scenario.escort_permits_total + dis.escort_permits_delta
        )
        self.active_fittings_lead_deltas = dict(dis.fittings_lead_deltas)
        self.active_pvi_k_multiplier = float(dis.pvi_k_multiplier)
        self.active_bay_blackouts = [b.model_copy(deep=True) for b in dis.bay_blackouts]
        self.active_bay_max_unit_tare_kg = dict(dis.bay_max_unit_tare_kg)

        for oid, min_p in dis.order_min_pressure_overrides.items():
            if oid in self.active_orders:
                self.active_orders[oid] = self.active_orders[oid].model_copy(
                    update={"min_pressure_mpa": min_p}
                )
        for oid, new_sla in dis.order_sla_overrides.items():
            if oid in self.active_orders:
                self.active_orders[oid] = self.active_orders[oid].model_copy(
                    update={"sla_week": new_sla}
                )

        for rid, mass_cap in dis.route_mass_cap_overrides.items():
            if rid in self.active_routes:
                self.active_routes[rid] = self.active_routes[rid].model_copy(
                    update={"max_unit_transport_mass_kg": mass_cap}
                )
        for rid, std_h in dis.route_std_height_overrides.items():
            if rid in self.active_routes:
                self.active_routes[rid] = self.active_routes[rid].model_copy(
                    update={"std_max_height_mm": std_h}
                )
        for rid, perm_ok in dis.route_permit_allowed_overrides.items():
            if rid in self.active_routes:
                self.active_routes[rid] = self.active_routes[rid].model_copy(
                    update={"escort_permit_allowed": perm_ok}
                )

        return dis.model_dump(mode="json")

    def _maybe_trigger_disruption(self) -> tuple[bool, dict[str, Any] | None]:
        if self.disruption_triggered:
            return False, None
        if (
            len(self.booked_orders) >= DISRUPTION_TRIGGER_BOOKINGS
            or self.mutation_count >= DISRUPTION_TRIGGER_MUTATIONS
        ):
            alert = self._apply_disruption()
            return True, alert
        return False, None

    def _parse_and_validate_spec(
        self,
        unit_count: Any,
        unit_volume_m3: Any,
        design_pressure_mpa: Any,
        wall_thickness_mm: Any,
        insulation: Any,
        fittings: Any,
    ) -> tuple[VesselDesignSpec | None, str | None]:
        if isinstance(unit_count, bool) or not isinstance(unit_count, int):
            return None, f"unit_count must be an integer in {list(UNIT_COUNTS_ALLOWED)}."
        for name, num in (
            ("unit_volume_m3", unit_volume_m3),
            ("design_pressure_mpa", design_pressure_mpa),
            ("wall_thickness_mm", wall_thickness_mm),
        ):
            if isinstance(num, bool) or not isinstance(num, (int, float)) or not math.isfinite(float(num)):
                return None, f"{name} must be a finite number, got {num!r}."
        if not isinstance(insulation, str) or insulation.strip().upper() not in InsulationType.__members__:
            return None, f"insulation must be one of {[e.value for e in InsulationType]}, got {insulation!r}."
        if not isinstance(fittings, str) or fittings.strip().upper() not in FittingsGrade.__members__:
            return None, f"fittings must be one of {[e.value for e in FittingsGrade]}, got {fittings!r}."

        try:
            spec = VesselDesignSpec(
                unit_count=unit_count,
                unit_volume_m3=float(unit_volume_m3),
                design_pressure_mpa=float(design_pressure_mpa),
                wall_thickness_mm=float(wall_thickness_mm),
                insulation=InsulationType(insulation.strip().upper()),
                fittings=FittingsGrade(fittings.strip().upper()),
            )
        except ValidationError as exc:
            return None, f"Invalid vessel specification: {exc.errors()[0]['msg']}"

        cat_errors = validate_catalogue_bounds(spec)
        if cat_errors:
            return None, "; ".join(cat_errors)
        return spec, None

    def _validate_bay_and_schedule(
        self,
        *,
        bay_id: Any,
        bay_start_week: Any,
        fab_weeks: int,
        unit_tare_kg: float,
        exclude_order_id: str | None = None,
    ) -> tuple[BayId | None, str | None, str | None]:
        if not isinstance(bay_id, str):
            return None, "INVALID_BAY", f"bay_id must be one of {[b.value for b in BayId]}."
        normalized_bay = bay_id.strip().upper()
        try:
            bay_enum = BayId(normalized_bay)
        except ValueError:
            return (
                None,
                "INVALID_BAY",
                f"Unknown bay_id '{bay_id}'. Valid bays: {[b.value for b in BayId]}.",
            )

        if isinstance(bay_start_week, bool) or not isinstance(bay_start_week, int) or bay_start_week < 1:
            return (
                None,
                "INVALID_START_WEEK",
                f"bay_start_week must be an integer >= 1, got {bay_start_week!r}.",
            )
        if bay_start_week > self.scenario.horizon_weeks:
            return (
                None,
                "INVALID_START_WEEK",
                f"bay_start_week={bay_start_week} exceeds scenario horizon_weeks={self.scenario.horizon_weeks}.",
            )

        if bay_enum in self.active_bay_max_unit_tare_kg:
            crane_cap = self.active_bay_max_unit_tare_kg[bay_enum]
            if unit_tare_kg > crane_cap:
                return (
                    None,
                    "CRANE_TARE_LIMIT_EXCEEDED",
                    f"{bay_enum.value} overhead crane cap is {crane_cap} kg, but unit_tare_mass_kg is {unit_tare_kg} kg.",
                )

        end_week = bay_start_week + fab_weeks
        for oid, existing in self.booked_orders.items():
            if oid == exclude_order_id or existing.bay_id != bay_enum:
                continue
            if not (end_week <= existing.bay_start_week or existing.fab_end_week <= bay_start_week):
                return (
                    None,
                    "BAY_OVERLAP",
                    f"{bay_enum.value} slot [{bay_start_week}, {end_week}) overlaps {oid} "
                    f"[{existing.bay_start_week}, {existing.fab_end_week}).",
                )

        for bw in self.active_bay_blackouts:
            if bw.bay_id == bay_enum and not (end_week <= bw.start_week or bw.end_week <= bay_start_week):
                return (
                    None,
                    "BAY_BLACKOUT_CONFLICT",
                    f"{bay_enum.value} slot [{bay_start_week}, {end_week}) overlaps maintenance blackout "
                    f"[{bw.start_week}, {bw.end_week}) ({bw.reason}).",
                )

        return bay_enum, None, None

    # ------------------------------------------------------------------
    # 10 Public Agent Tool Methods
    # ------------------------------------------------------------------

    def get_episode_brief(self) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if not self.finalized:
            err = self._check_step_budget("get_episode_brief", args)
            if err is not None:
                return err

        public_title = (
            self.scenario.title
            if self.disruption_triggered
            else self.scenario.title.split(" — ", 1)[0].strip()
        )
        public_description = (
            self.scenario.description
            if self.disruption_triggered
            else (
                "Commission all 4 client cryogenic vessel orders across the active transport "
                "corridors within plant budget, fabrication bay capacity, and delivery SLAs."
            )
        )

        obs = {
            "ok": True,
            "seed": self.scenario.seed,
            "title": public_title,
            "description": public_description,
            "horizon_weeks": self.scenario.horizon_weeks,
            "current_factory_week": self.current_factory_week,
            "effective_budget_eur": self.effective_budget_eur,
            "total_spent_eur": self.total_spent_eur,
            "budget_remaining_eur": self.budget_remaining_eur,
            "escort_permits_total": self.effective_escort_permits_total,
            "escort_permits_used": self.escort_permits_used,
            "bays": [b.value for b in BayId],
            "sim_calls_remaining": self.sim_calls_remaining,
            "sim_budget_total": self.sim_budget_total,
            "steps_used": self.step_count,
            "max_steps": MAX_EPISODE_STEPS,
            "standard_volumes_m3": list(STANDARD_VOLUMES_M3),
            "unit_counts_allowed": list(UNIT_COUNTS_ALLOWED),
            "orders": [o.model_dump(mode="json") for o in self.active_orders.values()],
            "routes": [r.model_dump(mode="json") for r in self.active_routes.values()],
            "disruption_triggered": self.disruption_triggered,
            "disruption_alert": (
                self.scenario.disruption.model_dump(mode="json") if self.disruption_triggered else None
            ),
        }
        if self.finalized:
            return obs
        return self._record_step("get_episode_brief", args, obs)

    def query_handbook(self, section: str = "all") -> dict[str, Any]:
        args = {"section": section}
        if not self.finalized:
            err = self._check_step_budget("query_handbook", args)
            if err is not None:
                return err

        try:
            data = get_handbook_section(section)
            obs: dict[str, Any] = {"ok": True, **data}
        except ValueError as exc:
            obs = {
                "ok": False,
                "error": "INVALID_HANDBOOK_SECTION",
                "message": str(exc),
                "valid_sections": list(HANDBOOK_SECTIONS),
            }

        if self.finalized:
            return obs
        return self._record_step("query_handbook", args, obs)

    def inspect_order(self, order_id: str) -> dict[str, Any]:
        args = {"order_id": order_id}
        if not self.finalized:
            err = self._check_step_budget("inspect_order", args)
            if err is not None:
                return err

        key = order_id.strip().upper() if isinstance(order_id, str) else ""
        if key not in self.active_orders:
            obs: dict[str, Any] = {
                "ok": False,
                "error": "UNKNOWN_ORDER",
                "message": f"Unknown order_id '{order_id}'. Valid order IDs: {list(self.active_orders)}.",
            }
        else:
            order = self.active_orders[key]
            obs = {
                "ok": True,
                "order": order.model_dump(mode="json"),
                "designated_route": self.active_routes[order.route_id].model_dump(mode="json"),
                "current_booking": (
                    self.booked_orders[key].model_dump(mode="json") if key in self.booked_orders else None
                ),
                "disruption_triggered": self.disruption_triggered,
            }

        if self.finalized:
            return obs
        return self._record_step("inspect_order", args, obs)

    def inspect_route(self, route_id: str) -> dict[str, Any]:
        args = {"route_id": route_id}
        if not self.finalized:
            err = self._check_step_budget("inspect_route", args)
            if err is not None:
                return err

        key = route_id.strip().upper() if isinstance(route_id, str) else ""
        if key not in self.active_routes:
            obs: dict[str, Any] = {
                "ok": False,
                "error": "UNKNOWN_ROUTE",
                "message": f"Unknown route_id '{route_id}'. Valid route IDs: {list(self.active_routes)}.",
            }
        else:
            obs = {
                "ok": True,
                "route": self.active_routes[key].model_dump(mode="json"),
                "escort_permits_total": self.effective_escort_permits_total,
                "escort_permits_used": self.escort_permits_used,
                "disruption_triggered": self.disruption_triggered,
            }

        if self.finalized:
            return obs
        return self._record_step("inspect_route", args, obs)

    def get_factory_state(self) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if not self.finalized:
            err = self._check_step_budget("get_factory_state", args)
            if err is not None:
                return err

        bay_schedules: dict[str, list[dict[str, Any]]] = {b.value: [] for b in BayId}
        for b in sorted(self.booked_orders.values(), key=lambda x: (x.bay_start_week, x.order_id)):
            bay_schedules[b.bay_id.value].append(
                {
                    "order_id": b.order_id,
                    "bay_start_week": b.bay_start_week,
                    "fab_end_week": b.fab_end_week,
                    "fab_weeks": b.fab_weeks,
                    "order_release_week": b.order_release_week,
                    "assigned_route_id": b.assigned_route_id,
                    "use_escort_permit": b.use_escort_permit,
                }
            )

        obs = {
            "ok": True,
            "seed": self.scenario.seed,
            "current_factory_week": self.current_factory_week,
            "horizon_weeks": self.scenario.horizon_weeks,
            "finalized": self.finalized,
            "step_count": self.step_count,
            "max_steps": MAX_EPISODE_STEPS,
            "mutation_count": self.mutation_count,
            "sim_calls_remaining": self.sim_calls_remaining,
            "sim_budget_total": self.sim_budget_total,
            "effective_budget_eur": self.effective_budget_eur,
            "total_spent_eur": self.total_spent_eur,
            "retooling_fees_eur": self.retooling_fees_eur,
            "budget_remaining_eur": self.budget_remaining_eur,
            "escort_permits_total": self.effective_escort_permits_total,
            "escort_permits_used": self.escort_permits_used,
            "bay_schedules": bay_schedules,
            "bay_blackouts": [bw.model_dump(mode="json") for bw in self.active_bay_blackouts],
            "bay_max_unit_tare_kg": {k.value: v for k, v in self.active_bay_max_unit_tare_kg.items()},
            "active_fittings_lead_deltas": {
                k.value: v for k, v in self.active_fittings_lead_deltas.items()
            },
            "active_pvi_k_multiplier": self.active_pvi_k_multiplier,
            "booked_orders": {oid: b.model_dump(mode="json") for oid, b in self.booked_orders.items()},
            "unbooked_orders": [oid for oid in self.active_orders if oid not in self.booked_orders],
            "disruption_triggered": self.disruption_triggered,
            "disruption_alert": (
                self.scenario.disruption.model_dump(mode="json") if self.disruption_triggered else None
            ),
        }
        if self.finalized:
            return obs
        return self._record_step("get_factory_state", args, obs)

    def run_physics_sim(
        self,
        order_id: str,
        unit_count: int,
        unit_volume_m3: float,
        design_pressure_mpa: float,
        wall_thickness_mm: float,
        insulation: str,
        fittings: str,
    ) -> dict[str, Any]:
        if self.finalized:
            return {
                "ok": False,
                "error": "EPISODE_ALREADY_FINALIZED",
                "message": "Episode is already finalized; state is immutable.",
            }

        args = {
            "order_id": order_id,
            "unit_count": unit_count,
            "unit_volume_m3": unit_volume_m3,
            "design_pressure_mpa": design_pressure_mpa,
            "wall_thickness_mm": wall_thickness_mm,
            "insulation": insulation,
            "fittings": fittings,
        }
        err = self._check_step_budget("run_physics_sim", args)
        if err is not None:
            return err

        if self.sim_calls_remaining <= 0:
            obs: dict[str, Any] = {
                "ok": False,
                "error": "SIM_BUDGET_EXHAUSTED",
                "message": "Physics simulation budget exhausted (0 remaining). Compute equations from handbook.",
                "sim_calls_remaining": 0,
            }
            return self._record_step("run_physics_sim", args, obs)

        key = order_id.strip().upper() if isinstance(order_id, str) else ""
        if key not in self.active_orders:
            obs = {
                "ok": False,
                "error": "UNKNOWN_ORDER",
                "message": f"Unknown order_id '{order_id}'. Valid order IDs: {list(self.active_orders)}.",
                "sim_calls_remaining": self.sim_calls_remaining,
            }
            return self._record_step("run_physics_sim", args, obs)

        spec, spec_err = self._parse_and_validate_spec(
            unit_count,
            unit_volume_m3,
            design_pressure_mpa,
            wall_thickness_mm,
            insulation,
            fittings,
        )
        if spec is None:
            obs = {
                "ok": False,
                "error": "INVALID_SPEC",
                "message": spec_err,
                "sim_calls_remaining": self.sim_calls_remaining,
            }
            return self._record_step("run_physics_sim", args, obs)

        order = self.active_orders[key]
        physics_eval = evaluate_design_physics(
            order,
            spec,
            pvi_k_multiplier=self.active_pvi_k_multiplier,
            fittings_lead_delta=self.active_fittings_lead_deltas.get(spec.fittings, 0),
        )
        self.sim_calls_remaining -= 1
        self.sim_calls_used += 1

        obs = {
            "ok": True,
            "order_id": key,
            "sim_calls_remaining": self.sim_calls_remaining,
            "disruption_triggered": self.disruption_triggered,
            "physics": physics_eval.model_dump(mode="json"),
        }
        return self._record_step("run_physics_sim", args, obs)

    def reserve_bay_and_order(
        self,
        order_id: str,
        bay_id: str,
        bay_start_week: int,
        unit_count: int,
        unit_volume_m3: float,
        design_pressure_mpa: float,
        wall_thickness_mm: float,
        insulation: str,
        fittings: str,
    ) -> dict[str, Any]:
        if self.finalized:
            return {
                "ok": False,
                "error": "EPISODE_ALREADY_FINALIZED",
                "message": "Episode is already finalized; state is immutable.",
            }

        args = {
            "order_id": order_id,
            "bay_id": bay_id,
            "bay_start_week": bay_start_week,
            "unit_count": unit_count,
            "unit_volume_m3": unit_volume_m3,
            "design_pressure_mpa": design_pressure_mpa,
            "wall_thickness_mm": wall_thickness_mm,
            "insulation": insulation,
            "fittings": fittings,
        }
        err = self._check_step_budget("reserve_bay_and_order", args)
        if err is not None:
            return err

        key = order_id.strip().upper() if isinstance(order_id, str) else ""
        if key not in self.active_orders:
            return self._record_step(
                "reserve_bay_and_order",
                args,
                {
                    "ok": False,
                    "error": "UNKNOWN_ORDER",
                    "message": f"Unknown order_id '{order_id}'. Valid order IDs: {list(self.active_orders)}.",
                },
            )
        if key in self.booked_orders:
            return self._record_step(
                "reserve_bay_and_order",
                args,
                {
                    "ok": False,
                    "error": "ORDER_ALREADY_BOOKED",
                    "message": f"Order {key} is already booked. Use revise_order to modify an existing booking.",
                },
            )

        spec, spec_err = self._parse_and_validate_spec(
            unit_count,
            unit_volume_m3,
            design_pressure_mpa,
            wall_thickness_mm,
            insulation,
            fittings,
        )
        if spec is None:
            return self._record_step(
                "reserve_bay_and_order",
                args,
                {"ok": False, "error": "INVALID_SPEC", "message": spec_err},
            )

        order = self.active_orders[key]
        # Anti-oracle: ONLY compute geometry, tare/transport mass, and fab/cost — NEVER hoop stress or boil-off!
        dims = compute_dimensions(spec.unit_volume_m3, order.execution, spec.insulation)
        _, unit_tare_kg, unit_transport_kg = compute_masses_and_area(spec, order.execution, dims)
        fab_weeks, fittings_lead_weeks, mfg_cost_eur = compute_fab_and_cost(
            spec,
            order.execution,
            unit_tare_kg,
            fittings_lead_delta=self.active_fittings_lead_deltas.get(spec.fittings, 0),
        )

        bay_enum, bay_err_code, bay_err_msg = self._validate_bay_and_schedule(
            bay_id=bay_id,
            bay_start_week=bay_start_week,
            fab_weeks=fab_weeks,
            unit_tare_kg=unit_tare_kg,
        )
        if bay_enum is None:
            return self._record_step(
                "reserve_bay_and_order",
                args,
                {"ok": False, "error": bay_err_code, "message": bay_err_msg},
            )

        release_week = 2 if self.disruption_triggered else 1
        booking = OrderBooking(
            order_id=key,
            bay_id=bay_enum,
            bay_start_week=bay_start_week,
            fab_weeks=fab_weeks,
            fab_end_week=bay_start_week + fab_weeks,
            booked_at_step=self.step_count,
            order_release_week=release_week,
            spec=spec,
            manufacturing_cost_eur=mfg_cost_eur,
            assigned_route_id=None,
            use_escort_permit=False,
            revision_count=0,
            post_shock_revised=False,
        )
        self.booked_orders[key] = booking
        self.mutation_count += 1

        triggered_now, alert = self._maybe_trigger_disruption()

        obs = {
            "ok": True,
            "booking": booking.model_dump(mode="json"),
            "unit_tare_mass_kg": unit_tare_kg,
            "unit_transport_mass_kg": unit_transport_kg,
            "dimensions": dims.model_dump(mode="json"),
            "fittings_lead_weeks": fittings_lead_weeks,
            "budget_remaining_eur": self.budget_remaining_eur,
            "disruption_triggered": self.disruption_triggered,
            "disruption_alert": alert,
        }
        return self._record_step(
            "reserve_bay_and_order",
            args,
            obs,
            disruption_triggered_this_step=triggered_now,
        )

    def assign_transport(
        self,
        order_id: str,
        route_id: str,
        use_escort_permit: bool = False,
    ) -> dict[str, Any]:
        if self.finalized:
            return {
                "ok": False,
                "error": "EPISODE_ALREADY_FINALIZED",
                "message": "Episode is already finalized; state is immutable.",
            }

        args = {
            "order_id": order_id,
            "route_id": route_id,
            "use_escort_permit": use_escort_permit,
        }
        err = self._check_step_budget("assign_transport", args)
        if err is not None:
            return err

        okey = order_id.strip().upper() if isinstance(order_id, str) else ""
        if okey not in self.active_orders:
            return self._record_step(
                "assign_transport",
                args,
                {
                    "ok": False,
                    "error": "UNKNOWN_ORDER",
                    "message": f"Unknown order_id '{order_id}'. Valid order IDs: {list(self.active_orders)}.",
                },
            )
        if okey not in self.booked_orders:
            return self._record_step(
                "assign_transport",
                args,
                {
                    "ok": False,
                    "error": "ORDER_NOT_BOOKED",
                    "message": f"Order {okey} is not booked yet. Call reserve_bay_and_order first.",
                },
            )

        rkey = route_id.strip().upper() if isinstance(route_id, str) else ""
        if rkey not in self.active_routes:
            return self._record_step(
                "assign_transport",
                args,
                {
                    "ok": False,
                    "error": "UNKNOWN_ROUTE",
                    "message": f"Unknown route_id '{route_id}'. Valid route IDs: {list(self.active_routes)}.",
                },
            )
        if not isinstance(use_escort_permit, bool):
            return self._record_step(
                "assign_transport",
                args,
                {
                    "ok": False,
                    "error": "INVALID_ARGUMENT",
                    "message": "use_escort_permit must be a boolean.",
                },
            )

        booking = self.booked_orders[okey].model_copy(
            update={"assigned_route_id": rkey, "use_escort_permit": use_escort_permit}
        )
        self.booked_orders[okey] = booking
        self.mutation_count += 1

        triggered_now, alert = self._maybe_trigger_disruption()

        eff_lead = max(
            1,
            int(
                FITTINGS_PROPS[booking.spec.fittings]["base_lead_weeks"]
                + self.active_fittings_lead_deltas.get(booking.spec.fittings, 0)
            ),
        )
        route = self.active_routes[rkey]
        delivery_week = (
            max(booking.fab_end_week, booking.order_release_week + eff_lead)
            + ASSEMBLY_COMMISSION_WEEKS
            + route.transit_weeks
        )

        obs = {
            "ok": True,
            "order_id": okey,
            "booking": booking.model_dump(mode="json"),
            "transport_cost_eur": self._order_transport_cost(booking),
            "estimated_delivery_week": delivery_week,
            "escort_permits_used": self.escort_permits_used,
            "escort_permits_total": self.effective_escort_permits_total,
            "budget_remaining_eur": self.budget_remaining_eur,
            "disruption_triggered": self.disruption_triggered,
            "disruption_alert": alert,
        }
        return self._record_step(
            "assign_transport",
            args,
            obs,
            disruption_triggered_this_step=triggered_now,
        )

    def revise_order(
        self,
        order_id: str,
        bay_id: str,
        bay_start_week: int,
        unit_count: int,
        unit_volume_m3: float,
        design_pressure_mpa: float,
        wall_thickness_mm: float,
        insulation: str,
        fittings: str,
        route_id: str | None = None,
        use_escort_permit: bool | None = None,
    ) -> dict[str, Any]:
        if self.finalized:
            return {
                "ok": False,
                "error": "EPISODE_ALREADY_FINALIZED",
                "message": "Episode is already finalized; state is immutable.",
            }

        args = {
            "order_id": order_id,
            "bay_id": bay_id,
            "bay_start_week": bay_start_week,
            "unit_count": unit_count,
            "unit_volume_m3": unit_volume_m3,
            "design_pressure_mpa": design_pressure_mpa,
            "wall_thickness_mm": wall_thickness_mm,
            "insulation": insulation,
            "fittings": fittings,
            "route_id": route_id,
            "use_escort_permit": use_escort_permit,
        }
        err = self._check_step_budget("revise_order", args)
        if err is not None:
            return err

        okey = order_id.strip().upper() if isinstance(order_id, str) else ""
        if okey not in self.active_orders:
            return self._record_step(
                "revise_order",
                args,
                {
                    "ok": False,
                    "error": "UNKNOWN_ORDER",
                    "message": f"Unknown order_id '{order_id}'. Valid order IDs: {list(self.active_orders)}.",
                },
            )
        if okey not in self.booked_orders:
            return self._record_step(
                "revise_order",
                args,
                {
                    "ok": False,
                    "error": "ORDER_NOT_BOOKED",
                    "message": f"Order {okey} is not booked yet. Use reserve_bay_and_order first.",
                },
            )

        existing = self.booked_orders[okey]
        new_route_id = existing.assigned_route_id
        if route_id is not None:
            rkey = route_id.strip().upper() if isinstance(route_id, str) else ""
            if rkey not in self.active_routes:
                return self._record_step(
                    "revise_order",
                    args,
                    {
                        "ok": False,
                        "error": "UNKNOWN_ROUTE",
                        "message": f"Unknown route_id '{route_id}'. Valid route IDs: {list(self.active_routes)}.",
                    },
                )
            new_route_id = rkey

        new_permit = existing.use_escort_permit
        if use_escort_permit is not None:
            if not isinstance(use_escort_permit, bool):
                return self._record_step(
                    "revise_order",
                    args,
                    {
                        "ok": False,
                        "error": "INVALID_ARGUMENT",
                        "message": "use_escort_permit must be a boolean or None.",
                    },
                )
            new_permit = use_escort_permit

        spec, spec_err = self._parse_and_validate_spec(
            unit_count,
            unit_volume_m3,
            design_pressure_mpa,
            wall_thickness_mm,
            insulation,
            fittings,
        )
        if spec is None:
            return self._record_step(
                "revise_order",
                args,
                {"ok": False, "error": "INVALID_SPEC", "message": spec_err},
            )

        order = self.active_orders[okey]
        # Anti-oracle: ONLY compute geometry, tare/transport mass, and fab/cost — NEVER hoop stress or boil-off!
        dims = compute_dimensions(spec.unit_volume_m3, order.execution, spec.insulation)
        _, unit_tare_kg, unit_transport_kg = compute_masses_and_area(spec, order.execution, dims)
        fab_weeks, fittings_lead_weeks, mfg_cost_eur = compute_fab_and_cost(
            spec,
            order.execution,
            unit_tare_kg,
            fittings_lead_delta=self.active_fittings_lead_deltas.get(spec.fittings, 0),
        )

        bay_enum, bay_err_code, bay_err_msg = self._validate_bay_and_schedule(
            bay_id=bay_id,
            bay_start_week=bay_start_week,
            fab_weeks=fab_weeks,
            unit_tare_kg=unit_tare_kg,
            exclude_order_id=okey,
        )
        if bay_enum is None:
            return self._record_step(
                "revise_order",
                args,
                {"ok": False, "error": bay_err_code, "message": bay_err_msg},
            )

        # Retooling fee rules (ENGINE_API_SPEC invariant 6)
        if self.disruption_triggered and not existing.post_shock_revised:
            fee_eur = POST_SHOCK_FIRST_REVISION_FEE_EUR
            post_shock_revised = True
        else:
            fee_eur = STANDARD_REVISION_FEE_EUR
            post_shock_revised = existing.post_shock_revised
        self.retooling_fees_eur = round(self.retooling_fees_eur + fee_eur, 2)

        # Order release week rule (ENGINE_API_SPEC invariant 5)
        if self.disruption_triggered and (
            spec.fittings != existing.spec.fittings or spec.unit_count != existing.spec.unit_count
        ):
            release_week = 2
        else:
            release_week = existing.order_release_week

        updated_booking = OrderBooking(
            order_id=okey,
            bay_id=bay_enum,
            bay_start_week=bay_start_week,
            fab_weeks=fab_weeks,
            fab_end_week=bay_start_week + fab_weeks,
            booked_at_step=existing.booked_at_step,
            order_release_week=release_week,
            spec=spec,
            manufacturing_cost_eur=mfg_cost_eur,
            assigned_route_id=new_route_id,
            use_escort_permit=new_permit,
            revision_count=existing.revision_count + 1,
            post_shock_revised=post_shock_revised,
        )
        self.booked_orders[okey] = updated_booking
        self.mutation_count += 1

        triggered_now, alert = self._maybe_trigger_disruption()

        obs = {
            "ok": True,
            "booking": updated_booking.model_dump(mode="json"),
            "revision_fee_charged_eur": fee_eur,
            "retooling_fees_total_eur": self.retooling_fees_eur,
            "unit_tare_mass_kg": unit_tare_kg,
            "unit_transport_mass_kg": unit_transport_kg,
            "dimensions": dims.model_dump(mode="json"),
            "fittings_lead_weeks": fittings_lead_weeks,
            "budget_remaining_eur": self.budget_remaining_eur,
            "disruption_triggered": self.disruption_triggered,
            "disruption_alert": alert,
        }
        return self._record_step(
            "revise_order",
            args,
            obs,
            disruption_triggered_this_step=triggered_now,
        )

    def finalize_commissioning(self) -> dict[str, Any]:
        if self.finalized:
            return {
                "ok": False,
                "error": "EPISODE_ALREADY_FINALIZED",
                "message": "Episode is already finalized; state is immutable.",
            }

        args: dict[str, Any] = {}
        err = self._check_step_budget("finalize_commissioning", args)
        if err is not None:
            return err

        self.finalized = True
        self.completed_at = _utc_now_iso()
        report = self._compute_verification_report()

        obs = {
            "ok": True,
            "finalized": True,
            "final_score": report.final_score,
            "valid_order_count": report.valid_order_count,
            "total_order_count": report.total_order_count,
            "verification": report.model_dump(mode="json"),
        }
        self._record_step("finalize_commissioning", args, obs)

        if self.log_dir is not None:
            self.export_trajectory_log()

        return obs

    # ------------------------------------------------------------------
    # Harness-only Verification & Signed Trajectory Export
    # ------------------------------------------------------------------

    def _compute_verification_report(self) -> VerificationReport:
        return evaluate_episode(
            scenario=self.scenario,
            bookings=self.booked_orders,
            finalized=self.finalized,
            retooling_fees_eur=self.retooling_fees_eur,
            sim_calls_used=self.sim_calls_used,
            sim_budget_total=self.sim_budget_total,
            step_count=self.step_count,
            disruption_triggered=self.disruption_triggered,
        )

    def _sign_payload(self, log_without_hmac: dict[str, Any]) -> str:
        payload_copy = dict(log_without_hmac)
        payload_copy["state_hmac"] = ""
        canonical = json.dumps(payload_copy, sort_keys=True, separators=(",", ":"))
        return hmac.new(self._hmac_secret, canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_trajectory_hmac(self, log: TrajectoryLog) -> bool:
        """Verify that `log.state_hmac` matches this engine instance's per-episode HMAC secret."""
        expected = self._sign_payload(log.model_dump(mode="json"))
        return hmac.compare_digest(expected, log.state_hmac)

    def export_trajectory_log(self) -> TrajectoryLog:
        """Export HMAC-signed TrajectoryLog (and persist to `log_dir/<run_id>.json` if configured)."""
        verification = self._compute_verification_report()
        completed_iso = self.completed_at or _utc_now_iso()

        unsigned_log = TrajectoryLog(
            run_id=self.run_id,
            seed=self.seed,
            agent_name=self.agent_name,
            agent_type=self.agent_type,  # type: ignore[arg-type]
            model_id=self.model_id,
            started_at=self.started_at,
            completed_at=completed_iso,
            steps=list(self.steps),
            verification=verification,
            state_hmac="",
        )
        signature = self._sign_payload(unsigned_log.model_dump(mode="json"))
        signed_log = unsigned_log.model_copy(update={"state_hmac": signature})

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.log_dir / f"{self.run_id}.json"
            out_path.write_text(signed_log.model_dump_json(indent=2), encoding="utf-8")

        return signed_log
