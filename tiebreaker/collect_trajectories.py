"""Tiebreaker Trajectory Collector, Contrastive Pair Builder & S01-S05 Experience Distiller.

Collects verified expert (`>= 0.94`) and suboptimal/failure trajectories strictly on
training seeds `S01`–`S05` (`EVAL_SEEDS`) — NEVER reading, simulating, or referencing
held-out tiebreaker seeds `S06`, `S07`, `S08` (`TIEBREAKER_SEEDS`).

Exports:
1. `runs/tiebreaker/sft_train_messages.jsonl`:
   OpenAI / HuggingFace TRL multi-turn tool-calling `messages` format with explicit
   Chain-of-Thought cryogenic engineering calculations before each tool call.
2. `runs/tiebreaker/preference_pairs.jsonl`:
   DPO / GRPO contrast pairs (`chosen` >= 0.94 vs `rejected` shock-unadapted,
   sim-exhausted, or batch-gated failure rollouts on `S01`–`S05`) labeled with
   full `VerificationReport` rewards.
3. `runs/tiebreaker/experience_playbook.json`:
   Structured empirical policy, failure-mode reward deltas, handbook formula summary,
   and compact `S01`–`S05` trajectory demonstrations distilled automatically from
   `S01`–`S05` environment rollouts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from env_core.contracts import (
    AMBIENT_TEMP_K,
    ASSEMBLY_COMMISSION_WEEKS,
    BASE_ALLOWABLE_STRESS_MPA,
    BayBlackoutWindow,
    BayId,
    CustomerOrder,
    DISRUPTION_TRIGGER_BOOKINGS,
    DISRUPTION_TRIGGER_MUTATIONS,
    DisruptionEvent,
    EVAL_SEEDS,
    ExecutionType,
    FILL_FRACTION,
    FITTINGS_PROPS,
    FittingsGrade,
    HANDBOOK_SECTIONS,
    INSULATION_PROPS,
    InsulationType,
    MAX_DESIGN_PRESSURE_MPA,
    MAX_EPISODE_STEPS,
    MAX_WALL_THICKNESS_MM,
    MEDIUM_PROPS,
    MIN_DESIGN_PRESSURE_MPA,
    MIN_WALL_THICKNESS_MM,
    OrderBooking,
    OrderVerificationResult,
    POST_SHOCK_FIRST_REVISION_FEE_EUR,
    PhysicsEvaluation,
    SIM_BUDGET_DEFAULT,
    STANDARD_REVISION_FEE_EUR,
    STANDARD_VOLUMES_M3,
    TIEBREAKER_SEEDS,
    TrajectoryLog,
    TrajectoryStepLog,
    TransportRoute,
    UNIT_COUNTS_ALLOWED,
    VerificationReport,
    VesselDesignSpec,
)
from env_core.physics import (
    compute_dimensions,
    compute_fab_and_cost,
    compute_masses_and_area,
    evaluate_design_physics,
    validate_catalogue_bounds,
)
from tiebreaker import (
    ALLOWED_TRAIN_SEEDS,
    FORBIDDEN_HOLDOUT_SEEDS,
    REPO_ROOT,
    TIEBREAKER_RUNS_DIR,
    TRAJECTORIES_DIR,
    assert_train_seed_only,
)


# ---------------------------------------------------------------------------
# Exact 10 MCP Tool Schemas (OpenAI / HuggingFace TRL Function Format)
# ---------------------------------------------------------------------------

CRYOFORGE_OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_episode_brief",
            "description": (
                "Retrieve the initial commissioning episode brief: active order IDs, "
                "route IDs, planning horizon weeks, factory budget, escort permit quota, "
                "and FEA simulation call budget."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_handbook",
            "description": (
                "Read the CryoForge engineering handbook containing exact physical constants, "
                "hoop-stress and boil-off equations, medium compatibility rules, shipping/site "
                "envelope formulas, bay scheduling rules, disruption revision fees, and scoring."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                        "enum": list(HANDBOOK_SECTIONS),
                        "description": "Handbook section name or 'all' for full specification.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_order",
            "description": (
                "Inspect a customer cryogenic vessel order's current requirements: medium, "
                "execution type, required deliverable mass, minimum design pressure, max daily "
                "boil-off, SLA delivery week, designated route, and site clearance limits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Customer order identifier."}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_route",
            "description": (
                "Inspect a transport route's current transit time, standard vs escort-permit "
                "width/height limits, unit transport mass cap, freight cost, and permit fee."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "route_id": {"type": "string", "description": "Transport route identifier."}
                },
                "required": ["route_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_factory_state",
            "description": (
                "Inspect live factory bay bookings, active bay blackout windows, crane unit "
                "tare caps, remaining budget, escort permit quota, remaining FEA simulation "
                "calls, and any active mid-episode supply/site disruption alert."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_physics_sim",
            "description": (
                "Run a deterministic FEA mechanical, thermal boil-off, hazard compatibility, "
                "and site-clearance check for a candidate vessel specification. Consumes 1 "
                "simulation call from the episode simulation budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "unit_count": {"type": "integer", "enum": list(UNIT_COUNTS_ALLOWED)},
                    "unit_volume_m3": {"type": "number", "enum": list(STANDARD_VOLUMES_M3)},
                    "design_pressure_mpa": {
                        "type": "number",
                        "minimum": MIN_DESIGN_PRESSURE_MPA,
                        "maximum": MAX_DESIGN_PRESSURE_MPA,
                    },
                    "wall_thickness_mm": {
                        "type": "number",
                        "minimum": MIN_WALL_THICKNESS_MM,
                        "maximum": MAX_WALL_THICKNESS_MM,
                    },
                    "insulation": {"type": "string", "enum": [e.value for e in InsulationType]},
                    "fittings": {"type": "string", "enum": [e.value for e in FittingsGrade]},
                },
                "required": [
                    "order_id",
                    "unit_count",
                    "unit_volume_m3",
                    "design_pressure_mpa",
                    "wall_thickness_mm",
                    "insulation",
                    "fittings",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reserve_bay_and_order",
            "description": (
                "Book fabrication bay schedule and release valve/fittings procurement for an "
                "order. Validates catalogue bounds, bay non-overlap, blackouts, and crane tare "
                "limits (does NOT act as a physics oracle for hoop stress or boil-off)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "bay_id": {"type": "string", "enum": [b.value for b in BayId]},
                    "bay_start_week": {"type": "integer", "minimum": 1},
                    "unit_count": {"type": "integer", "enum": list(UNIT_COUNTS_ALLOWED)},
                    "unit_volume_m3": {"type": "number", "enum": list(STANDARD_VOLUMES_M3)},
                    "design_pressure_mpa": {"type": "number"},
                    "wall_thickness_mm": {"type": "number"},
                    "insulation": {"type": "string", "enum": [e.value for e in InsulationType]},
                    "fittings": {"type": "string", "enum": [e.value for e in FittingsGrade]},
                },
                "required": [
                    "order_id",
                    "bay_id",
                    "bay_start_week",
                    "unit_count",
                    "unit_volume_m3",
                    "design_pressure_mpa",
                    "wall_thickness_mm",
                    "insulation",
                    "fittings",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_transport",
            "description": (
                "Assign a transport route and optional heavy-haul escort permit to an already "
                "booked customer order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "route_id": {"type": "string"},
                    "use_escort_permit": {"type": "boolean", "default": False},
                },
                "required": ["order_id", "route_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revise_order",
            "description": (
                "Revise an already booked order's bay schedule, vessel specification, and/or "
                "transport assignment (incurs 1,500 EUR first post-shock revision fee or "
                "4,500 EUR standard revision fee)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "bay_id": {"type": "string", "enum": [b.value for b in BayId]},
                    "bay_start_week": {"type": "integer", "minimum": 1},
                    "unit_count": {"type": "integer", "enum": list(UNIT_COUNTS_ALLOWED)},
                    "unit_volume_m3": {"type": "number", "enum": list(STANDARD_VOLUMES_M3)},
                    "design_pressure_mpa": {"type": "number"},
                    "wall_thickness_mm": {"type": "number"},
                    "insulation": {"type": "string", "enum": [e.value for e in InsulationType]},
                    "fittings": {"type": "string", "enum": [e.value for e in FittingsGrade]},
                    "route_id": {"type": "string"},
                    "use_escort_permit": {"type": "boolean"},
                },
                "required": [
                    "order_id",
                    "bay_id",
                    "bay_start_week",
                    "unit_count",
                    "unit_volume_m3",
                    "design_pressure_mpa",
                    "wall_thickness_mm",
                    "insulation",
                    "fittings",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_commissioning",
            "description": (
                "Lock and commission the plant batch across all customer orders. Must be "
                "called before episode end; unfinalized episodes receive a 0.0000 score."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


DEFAULT_SYSTEM_PROMPT = (
    "You are Lead Cryogenic Commissioning & Operations Engineer for CryoForge RL. "
    "Use the MCP tools to inspect the episode brief, engineering handbook, customer orders, "
    "transport routes, and factory bay state. Size ASME/GOST-compliant cryogenic vessels, "
    "conserve the finite 8-call FEA simulation budget, adapt already-booked and remaining "
    "orders when a mid-episode supply/site disruption triggers, assign compliant transport "
    "routes within permit and budget caps, and call finalize_commissioning() before step 42."
)

FALLBACK_TASK_INSTRUCTION = (
    "Commission all 4 cryogenic customer orders in this CryoForge factory batch within the "
    "planning horizon, factory EUR budget, escort permit quota, 8-call FEA simulation budget, "
    "and 42-step limit.\n\n"
    "Requirements:\n"
    "1. Inspect the episode brief, engineering handbook, all 4 customer orders, their designated "
    "transport routes, and factory bay state.\n"
    "2. Book valid vessel designs onto BAY-A and BAY-B without bay or blackout overlaps, meeting "
    "cryogenic deliverable mass, hoop stress, material/oxygen/hydrogen/helium compatibility, "
    "daily boil-off rate, installed site clearance, and transport envelope limits.\n"
    "3. Monitor for mid-episode disruption alerts; re-inspect affected constraints and revise any "
    "invalidated bookings using revise_order while keeping total spend within budget.\n"
    "4. Assign each order to its required transport route and call finalize_commissioning()."
)


def load_task_instruction() -> str:
    """Load `harbor_task/instruction.md` if available, else return standard task instruction."""
    inst_path = REPO_ROOT / "harbor_task" / "instruction.md"
    if inst_path.exists():
        text = inst_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return FALLBACK_TASK_INSTRUCTION


def _load_engine_cls() -> type[Any]:
    """Import `CryoForgeEngine` from `env_core` lazily so `py_compile` works standalone."""
    try:
        from env_core import CryoForgeEngine  # type: ignore[attr-defined]

        return CryoForgeEngine
    except ImportError:
        from env_core.engine import CryoForgeEngine  # type: ignore[no-redef]

        return CryoForgeEngine


# ---------------------------------------------------------------------------
# Flexible Observation Extractors (Compatible with any wrapper keys)
# ---------------------------------------------------------------------------


def _extract_order_ids(brief_obs: dict[str, Any], state_obs: dict[str, Any] | None = None) -> list[str]:
    found: list[str] = []
    for src in (brief_obs, state_obs or {}):
        for key in ("order_ids", "orders", "customer_orders", "order_summaries"):
            val = src.get(key)
            if isinstance(val, list):
                for item in val:
                    oid = item if isinstance(item, str) else (item.get("order_id") if isinstance(item, dict) else None)
                    if oid and oid not in found:
                        found.append(str(oid))
            elif isinstance(val, dict):
                for k, v in val.items():
                    oid = v.get("order_id", k) if isinstance(v, dict) else k
                    if oid and str(oid) not in found:
                        found.append(str(oid))
    return found


def _extract_route_ids(brief_obs: dict[str, Any], orders: dict[str, CustomerOrder]) -> list[str]:
    found: list[str] = []
    for key in ("route_ids", "routes", "transport_routes", "available_routes"):
        val = brief_obs.get(key)
        if isinstance(val, list):
            for item in val:
                rid = item if isinstance(item, str) else (item.get("route_id") if isinstance(item, dict) else None)
                if rid and rid not in found:
                    found.append(str(rid))
        elif isinstance(val, dict):
            for k, v in val.items():
                rid = v.get("route_id", k) if isinstance(v, dict) else k
                if rid and str(rid) not in found:
                    found.append(str(rid))
    for order in orders.values():
        if order.route_id not in found:
            found.append(order.route_id)
    return found


def _parse_customer_order(obs: dict[str, Any]) -> CustomerOrder:
    payload = obs.get("order") if isinstance(obs.get("order"), dict) else obs
    clean = {k: payload[k] for k in CustomerOrder.model_fields if k in payload}
    return CustomerOrder.model_validate(clean)


def _parse_transport_route(obs: dict[str, Any]) -> TransportRoute:
    payload = obs.get("route") if isinstance(obs.get("route"), dict) else obs
    clean = {k: payload[k] for k in TransportRoute.model_fields if k in payload}
    return TransportRoute.model_validate(clean)


def _parse_disruption(obs: dict[str, Any]) -> DisruptionEvent | None:
    for key in ("disruption_alert", "active_disruption", "disruption", "disruption_event"):
        raw = obs.get(key)
        if isinstance(raw, dict) and "disruption_id" in raw:
            clean = {k: raw[k] for k in DisruptionEvent.model_fields if k in raw}
            return DisruptionEvent.model_validate(clean)
    return None


@dataclass
class ObservedWorldState:
    seed: str
    horizon_weeks: int = 16
    effective_budget_eur: float = 280000.0
    retooling_fees_eur: float = 0.0
    effective_escort_permits_total: int = 2
    sim_calls_remaining: int = SIM_BUDGET_DEFAULT
    pvi_k_multiplier: float = 1.0
    fittings_lead_deltas: dict[FittingsGrade, int] = field(default_factory=dict)
    bay_blackouts: list[BayBlackoutWindow] = field(default_factory=list)
    bay_max_unit_tare_kg: dict[BayId, float] = field(default_factory=dict)
    disruption_triggered: bool = False
    disruption_event: DisruptionEvent | None = None
    orders: dict[str, CustomerOrder] = field(default_factory=dict)
    routes: dict[str, TransportRoute] = field(default_factory=dict)

    def update_from_brief_or_factory(self, obs: dict[str, Any]) -> None:
        for k in ("horizon_weeks", "episode_horizon_weeks", "planning_horizon_weeks", "max_weeks"):
            if isinstance(obs.get(k), int):
                self.horizon_weeks = int(obs[k])
                break

        for k in ("effective_budget_eur", "total_budget_eur", "factory_budget_eur", "budget_eur", "budget_total_eur"):
            if isinstance(obs.get(k), (int, float)):
                self.effective_budget_eur = float(obs[k])
                break
        else:
            if isinstance(obs.get("budget_remaining_eur"), (int, float)) and isinstance(obs.get("total_spent_eur"), (int, float)):
                self.effective_budget_eur = float(obs["budget_remaining_eur"]) + float(obs["total_spent_eur"])

        if isinstance(obs.get("retooling_fees_eur"), (int, float)):
            self.retooling_fees_eur = float(obs["retooling_fees_eur"])

        for k in (
            "effective_escort_permits_total",
            "escort_permits_total",
            "escort_permit_quota",
            "permit_quota",
            "permits_total",
            "escort_permits",
        ):
            if isinstance(obs.get(k), int):
                self.effective_escort_permits_total = int(obs[k])
                break

        if isinstance(obs.get("sim_calls_remaining"), int):
            self.sim_calls_remaining = int(obs["sim_calls_remaining"])

        for pk in ("pvi_k_multiplier", "active_pvi_k_multiplier"):
            if isinstance(obs.get(pk), (int, float)):
                self.pvi_k_multiplier = float(obs[pk])
                break

        for lk in ("fittings_lead_deltas", "active_fittings_lead_deltas"):
            raw_lead = obs.get(lk)
            if isinstance(raw_lead, dict):
                self.fittings_lead_deltas = {FittingsGrade(str(k)): int(v) for k, v in raw_lead.items()}
                break

        for k in ("bay_blackouts", "blackouts", "active_blackouts"):
            raw_bo = obs.get(k)
            if isinstance(raw_bo, list):
                parsed_bo: list[BayBlackoutWindow] = []
                for item in raw_bo:
                    if isinstance(item, dict) and {"bay_id", "start_week", "end_week"} <= set(item.keys()):
                        parsed_bo.append(
                            BayBlackoutWindow(
                                bay_id=BayId(str(item["bay_id"])),
                                start_week=int(item["start_week"]),
                                end_week=int(item["end_week"]),
                                reason=str(item.get("reason", "maintenance")),
                            )
                        )
                self.bay_blackouts = parsed_bo
                break

        for k in ("bay_max_unit_tare_kg", "crane_limits_kg", "bay_crane_caps_kg"):
            raw_crane = obs.get(k)
            if isinstance(raw_crane, dict):
                self.bay_max_unit_tare_kg = {BayId(str(bk)): float(bv) for bk, bv in raw_crane.items() if bv is not None}
                break

        if obs.get("disruption_triggered") is True:
            self.disruption_triggered = True

        alert = _parse_disruption(obs)
        if alert is not None:
            self.apply_disruption_alert(alert)

    def apply_disruption_alert(self, alert: DisruptionEvent) -> None:
        already_applied = self.disruption_event is not None and self.disruption_event.disruption_id == alert.disruption_id
        self.disruption_triggered = True
        self.disruption_event = alert

        if not already_applied:
            self.pvi_k_multiplier = float(alert.pvi_k_multiplier)
            for fg, delta in alert.fittings_lead_deltas.items():
                self.fittings_lead_deltas[FittingsGrade(fg)] = int(delta)
            for bo in alert.bay_blackouts:
                if all(
                    (existing.bay_id, existing.start_week, existing.end_week)
                    != (bo.bay_id, bo.start_week, bo.end_week)
                    for existing in self.bay_blackouts
                ):
                    self.bay_blackouts.append(bo)
            for bay_id, cap in alert.bay_max_unit_tare_kg.items():
                self.bay_max_unit_tare_kg[BayId(bay_id)] = float(cap)
            for rid, mass_cap in alert.route_mass_cap_overrides.items():
                if rid in self.routes:
                    self.routes[rid] = self.routes[rid].model_copy(update={"max_unit_transport_mass_kg": float(mass_cap)})
            for rid, std_h in alert.route_std_height_overrides.items():
                if rid in self.routes:
                    self.routes[rid] = self.routes[rid].model_copy(update={"std_max_height_mm": int(std_h)})
            for rid, perm_ok in alert.route_permit_allowed_overrides.items():
                if rid in self.routes:
                    self.routes[rid] = self.routes[rid].model_copy(update={"escort_permit_allowed": bool(perm_ok)})
            for oid, min_p in alert.order_min_pressure_overrides.items():
                if oid in self.orders:
                    self.orders[oid] = self.orders[oid].model_copy(update={"min_pressure_mpa": float(min_p)})
            for oid, sla_w in alert.order_sla_overrides.items():
                if oid in self.orders:
                    self.orders[oid] = self.orders[oid].model_copy(update={"sla_week": int(sla_w)})


# ---------------------------------------------------------------------------
# Exact Multi-Order CryoForge Batch Optimizer (for S01-S05 Rollout Generation)
# ---------------------------------------------------------------------------

CANDIDATE_WALL_MM: tuple[float, ...] = (
    4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5,
    10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 13.9,
    14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 24.0,
)


@dataclass(frozen=True)
class CandidateOrderOption:
    order_id: str
    bay_id: BayId
    spec: VesselDesignSpec
    physics: PhysicsEvaluation
    use_escort_permit: bool
    order_cost_eur: float
    revision_fee_eur: float
    order_release_week: int


@dataclass
class PlannedOrderAssignment:
    option: CandidateOrderOption
    bay_start_week: int
    delivery_week: int
    needs_revision: bool


def _overlaps_window(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _enumerate_order_options(
    order: CustomerOrder,
    route: TransportRoute,
    world: ObservedWorldState,
    *,
    existing_booking: dict[str, Any] | None = None,
    prefer_bay_b_first: bool = False,
) -> list[CandidateOrderOption]:
    """Enumerate dominance-pruned valid (bay, spec, permit) options for one order."""
    design_p = round(min(MAX_DESIGN_PRESSURE_MPA, max(MIN_DESIGN_PRESSURE_MPA, order.min_pressure_mpa)), 3)
    best_by_signature: dict[tuple[str, int, int, bool, bool], CandidateOrderOption] = {}
    bays = [BayId.BAY_B, BayId.BAY_A] if prefer_bay_b_first else [BayId.BAY_A, BayId.BAY_B]

    for unit_count in UNIT_COUNTS_ALLOWED:
        for vol in STANDARD_VOLUMES_M3:
            for ins in (InsulationType.PVI, InsulationType.EVI):
                for fit in (FittingsGrade.CN, FittingsGrade.EU_STD, FittingsGrade.DE_PRECISION):
                    lead_delta = world.fittings_lead_deltas.get(fit, 0)
                    # Pick minimum wall thickness that passes hoop stress & all engineering checks
                    chosen_phys: PhysicsEvaluation | None = None
                    chosen_spec: VesselDesignSpec | None = None
                    for wall in CANDIDATE_WALL_MM:
                        spec = VesselDesignSpec(
                            unit_count=unit_count,
                            unit_volume_m3=vol,
                            design_pressure_mpa=design_p,
                            wall_thickness_mm=wall,
                            insulation=ins,
                            fittings=fit,
                        )
                        phys = evaluate_design_physics(
                            order,
                            spec,
                            pvi_k_multiplier=world.pvi_k_multiplier,
                            fittings_lead_delta=lead_delta,
                        )
                        if phys.all_engineering_pass:
                            chosen_phys = phys
                            chosen_spec = spec
                            break
                        # If non-stress check failed (mass/boil-off/compat/site), thicker wall will not fix it
                        if not (phys.mass_pass and phys.pressure_pass and phys.material_compat_pass and phys.boil_off_pass and phys.site_clearance_pass):
                            break

                    if chosen_phys is None or chosen_spec is None:
                        continue

                    # Transport mass check
                    if chosen_phys.unit_transport_mass_kg > route.max_unit_transport_mass_kg + 1e-9:
                        continue

                    dims = chosen_phys.dimensions
                    fits_std = (
                        dims.shipping_width_mm <= route.std_max_width_mm
                        and dims.shipping_height_mm <= route.std_max_height_mm
                    )
                    fits_permit = (
                        route.escort_permit_allowed
                        and dims.shipping_width_mm <= route.permit_max_width_mm
                        and dims.shipping_height_mm <= route.permit_max_height_mm
                    )
                    if not fits_std and not fits_permit:
                        continue
                    use_permit = False if fits_std else True

                    freight_eur = chosen_spec.unit_count * route.base_freight_eur_per_unit
                    permit_eur = route.permit_fee_eur if use_permit else 0.0
                    order_cost = round(chosen_phys.manufacturing_cost_eur + freight_eur + permit_eur, 2)

                    for bay_id in bays:
                        crane_cap = world.bay_max_unit_tare_kg.get(bay_id)
                        if crane_cap is not None and chosen_phys.unit_tare_mass_kg > crane_cap + 1e-9:
                            continue

                        same_spec_as_existing = False
                        rev_fee = 0.0
                        release_wk = 2 if world.disruption_triggered else 1

                        if existing_booking is not None:
                            ex_spec = existing_booking["spec"]
                            same_fittings_and_count = (
                                ex_spec["fittings"] == chosen_spec.fittings.value
                                and int(ex_spec["unit_count"]) == chosen_spec.unit_count
                            )
                            release_wk = int(existing_booking.get("order_release_week", 1)) if same_fittings_and_count else 2
                            same_spec_as_existing = (
                                existing_booking["bay_id"] == bay_id.value
                                and same_fittings_and_count
                                and abs(float(ex_spec["unit_volume_m3"]) - chosen_spec.unit_volume_m3) < 1e-6
                                and abs(float(ex_spec["design_pressure_mpa"]) - chosen_spec.design_pressure_mpa) < 1e-6
                                and abs(float(ex_spec["wall_thickness_mm"]) - chosen_spec.wall_thickness_mm) < 1e-6
                                and ex_spec["insulation"] == chosen_spec.insulation.value
                            )
                            if not same_spec_as_existing:
                                rev_fee = (
                                    POST_SHOCK_FIRST_REVISION_FEE_EUR
                                    if (world.disruption_triggered and not existing_booking.get("post_shock_revised", False))
                                    else STANDARD_REVISION_FEE_EUR
                                )

                        sig = (
                            bay_id.value,
                            chosen_phys.fab_weeks,
                            release_wk + chosen_phys.fittings_lead_weeks,
                            use_permit,
                            same_spec_as_existing,
                        )
                        cand = CandidateOrderOption(
                            order_id=order.order_id,
                            bay_id=bay_id,
                            spec=chosen_spec,
                            physics=chosen_phys,
                            use_escort_permit=use_permit,
                            order_cost_eur=order_cost,
                            revision_fee_eur=rev_fee,
                            order_release_week=release_wk,
                        )
                        prev = best_by_signature.get(sig)
                        if prev is None or (cand.order_cost_eur + cand.revision_fee_eur) < (prev.order_cost_eur + prev.revision_fee_eur):
                            best_by_signature[sig] = cand

    options = list(best_by_signature.values())
    options.sort(
        key=lambda o: (
            0 if (existing_booking and o.revision_fee_eur == 0.0) else 1,
            o.order_cost_eur + o.revision_fee_eur,
            o.physics.fab_weeks,
            1 if o.use_escort_permit else 0,
        )
    )
    return options


def solve_batch_plan(
    world: ObservedWorldState,
    order_ids: list[str],
    *,
    existing_bookings: dict[str, dict[str, Any]] | None = None,
    min_new_bay_start_week: int = 1,
    prefer_bay_b_first: bool = False,
) -> dict[str, PlannedOrderAssignment]:
    """Find the globally highest-scoring 4-order vessel + bay + route plan under `world`."""
    existing_bookings = existing_bookings or {}
    options_per_order: dict[str, list[CandidateOrderOption]] = {}
    for oid in order_ids:
        order = world.orders[oid]
        route = world.routes[order.route_id]
        opts = _enumerate_order_options(
            order,
            route,
            world,
            existing_booking=existing_bookings.get(oid),
            prefer_bay_b_first=prefer_bay_b_first,
        )
        options_per_order[oid] = opts[:14]

    # Sort orders most-constrained-first for fast branch-and-bound
    sorted_oids = sorted(order_ids, key=lambda x: (len(options_per_order[x]), world.orders[x].sla_week))

    best_score = -1.0
    best_plan: dict[str, PlannedOrderAssignment] | None = None

    blackouts_by_bay: dict[BayId, list[tuple[int, int]]] = {BayId.BAY_A: [], BayId.BAY_B: []}
    for bo in world.bay_blackouts:
        blackouts_by_bay[bo.bay_id].append((bo.start_week, bo.end_week))

    def _valid_start_weeks(
        opt: CandidateOrderOption,
        occupied: list[tuple[int, int]],
        sla_week: int,
        transit_weeks: int,
        ex_booking: dict[str, Any] | None,
    ) -> list[int]:
        fab = opt.physics.fab_weeks
        fittings_ready = opt.order_release_week + opt.physics.fittings_lead_weeks
        max_deadline = min(sla_week, world.horizon_weeks)
        max_fab_end = max_deadline - ASSEMBLY_COMMISSION_WEEKS - transit_weeks
        max_start = max_fab_end - fab
        if max_start < 1 or fittings_ready + ASSEMBLY_COMMISSION_WEEKS + transit_weeks > max_deadline:
            return []

        candidates: list[int] = []
        # If keeping exact existing booking spec & bay, test existing bay_start_week first (zero fee!)
        if ex_booking is not None and opt.revision_fee_eur == 0.0:
            ex_start = int(ex_booking["bay_start_week"])
            if 1 <= ex_start <= max_start:
                candidates.append(ex_start)

        low_w = min_new_bay_start_week if world.disruption_triggered else 1
        for w in range(low_w, max_start + 1):
            if w not in candidates:
                candidates.append(w)
        if low_w > 1:
            for w in range(1, min(low_w, max_start + 1)):
                if w not in candidates:
                    candidates.append(w)

        valid: list[int] = []
        for w in candidates:
            end_w = w + fab
            if any(_overlaps_window(w, end_w, bs, be) for bs, be in blackouts_by_bay[opt.bay_id]):
                continue
            if any(_overlaps_window(w, end_w, os_, oe_) for os_, oe_ in occupied):
                continue
            valid.append(w)
        return valid

    def _dfs(
        idx: int,
        occ_a: list[tuple[int, int]],
        occ_b: list[tuple[int, int]],
        permits_used: int,
        spent_eur: float,
        current: dict[str, PlannedOrderAssignment],
    ) -> None:
        nonlocal best_score, best_plan
        if permits_used > world.effective_escort_permits_total:
            return
        if spent_eur > world.effective_budget_eur + 1e-6:
            return

        if idx == len(sorted_oids):
            # Ensure sequential 1-pass revisability if 2 pre-shock bookings both change windows
            rev_booked = [
                oid for oid in existing_bookings
                if oid in current and current[oid].needs_revision
            ]
            if len(rev_booked) == 2:
                o1, o2 = rev_booked[0], rev_booked[1]
                b1, b2 = existing_bookings[o1], existing_bookings[o2]
                p1, p2 = current[o1], current[o2]
                old1_end = int(b1["bay_start_week"]) + int(b1.get("fab_weeks", 6))
                old2_end = int(b2["bay_start_week"]) + int(b2.get("fab_weeks", 6))
                p1_hits_old2 = (
                    p1.option.bay_id.value == b2["bay_id"]
                    and _overlaps_window(p1.bay_start_week, p1.bay_start_week + p1.option.physics.fab_weeks, int(b2["bay_start_week"]), old2_end)
                )
                p2_hits_old1 = (
                    p2.option.bay_id.value == b1["bay_id"]
                    and _overlaps_window(p2.bay_start_week, p2.bay_start_week + p2.option.physics.fab_weeks, int(b1["bay_start_week"]), old1_end)
                )
                if p1_hits_old2 and p2_hits_old1:
                    return

            # Evaluate exact verifier objective proxy (base 0.85 + budget 0.07 + slack 0.04)
            budget_bonus = 0.07 * min(
                1.0,
                max(0.0, (world.effective_budget_eur - spent_eur) / max(1.0, 0.03 * world.effective_budget_eur)),
            )
            avg_slack = sum(
                world.orders[k].sla_week - v.delivery_week for k, v in current.items()
            ) / max(1, len(current))
            slack_bonus = 0.04 * min(1.0, max(0.0, avg_slack / 2.0))
            score = 0.85 + budget_bonus + slack_bonus
            if score > best_score:
                best_score = score
                best_plan = dict(current)
            return

        oid = sorted_oids[idx]
        order = world.orders[oid]
        route = world.routes[order.route_id]
        ex_b = existing_bookings.get(oid)

        for opt in options_per_order[oid]:
            new_permits = permits_used + (1 if opt.use_escort_permit else 0)
            if new_permits > world.effective_escort_permits_total:
                continue
            occ = occ_a if opt.bay_id == BayId.BAY_A else occ_b
            starts = _valid_start_weeks(opt, occ, order.sla_week, route.transit_weeks, ex_b)
            for start_w in starts:
                needs_rev = False
                extra_fee = opt.revision_fee_eur
                if ex_b is not None:
                    if opt.revision_fee_eur > 0.0 or int(ex_b["bay_start_week"]) != start_w:
                        needs_rev = True
                        if extra_fee == 0.0:
                            extra_fee = (
                                POST_SHOCK_FIRST_REVISION_FEE_EUR
                                if (world.disruption_triggered and not ex_b.get("post_shock_revised", False))
                                else STANDARD_REVISION_FEE_EUR
                            )
                new_spent = spent_eur + opt.order_cost_eur + extra_fee
                if new_spent > world.effective_budget_eur + 1e-6:
                    continue

                deliv_w = (
                    max(start_w + opt.physics.fab_weeks, opt.order_release_week + opt.physics.fittings_lead_weeks)
                    + ASSEMBLY_COMMISSION_WEEKS
                    + route.transit_weeks
                )
                current[oid] = PlannedOrderAssignment(
                    option=opt,
                    bay_start_week=start_w,
                    delivery_week=deliv_w,
                    needs_revision=needs_rev,
                )
                interval = (start_w, start_w + opt.physics.fab_weeks)
                if opt.bay_id == BayId.BAY_A:
                    _dfs(idx + 1, occ_a + [interval], occ_b, new_permits, new_spent, current)
                else:
                    _dfs(idx + 1, occ_a, occ_b + [interval], new_permits, new_spent, current)
                current.pop(oid, None)

    _dfs(0, [], [], 0, world.retooling_fees_eur, {})
    if best_plan is None:
        raise RuntimeError(f"No feasible 4-order batch plan found for training seed {world.seed}")
    return best_plan


# ---------------------------------------------------------------------------
# Live S01-S05 Rollout Generators (Expert + Contrastive Failures + Augmentations)
# ---------------------------------------------------------------------------


def _discover_episode(engine: Any, seed: str) -> tuple[ObservedWorldState, list[str]]:
    """Execute standard Phase-1 discovery tools on `engine` and build `ObservedWorldState`."""
    assert_train_seed_only(seed)
    world = ObservedWorldState(seed=seed)
    brief = engine.get_episode_brief()
    world.update_from_brief_or_factory(brief)
    engine.query_handbook(section="all")
    fstate = engine.get_factory_state()
    world.update_from_brief_or_factory(fstate)

    order_ids = _extract_order_ids(brief, fstate)
    for oid in order_ids:
        obs_o = engine.inspect_order(order_id=oid)
        world.orders[oid] = _parse_customer_order(obs_o)

    route_ids = _extract_route_ids(brief, world.orders)
    for rid in route_ids:
        obs_r = engine.inspect_route(route_id=rid)
        world.routes[rid] = _parse_transport_route(obs_r)

    return world, order_ids


def _extract_booked_snapshot(
    order_id: str,
    plan_item: PlannedOrderAssignment,
    resp: dict[str, Any],
    *,
    post_shock_revised: bool = False,
) -> dict[str, Any]:
    booking = resp.get("booking") if isinstance(resp.get("booking"), dict) else {}
    spec = plan_item.option.spec
    return {
        "order_id": order_id,
        "bay_id": booking.get("bay_id", plan_item.option.bay_id.value),
        "bay_start_week": int(booking.get("bay_start_week", plan_item.bay_start_week)),
        "fab_weeks": int(booking.get("fab_weeks", plan_item.option.physics.fab_weeks)),
        "order_release_week": int(booking.get("order_release_week", plan_item.option.order_release_week)),
        "post_shock_revised": bool(booking.get("post_shock_revised", post_shock_revised)),
        "spec": {
            "unit_count": spec.unit_count,
            "unit_volume_m3": spec.unit_volume_m3,
            "design_pressure_mpa": spec.design_pressure_mpa,
            "wall_thickness_mm": spec.wall_thickness_mm,
            "insulation": spec.insulation.value,
            "fittings": spec.fittings.value,
        },
    }


def rollout_expert_episode(
    seed: str,
    *,
    engine_factory: Any | None = None,
    sim_mode: str = "targeted_2sim",
    prefer_bay_b_first: bool = False,
    run_tag: str = "expert",
) -> TrajectoryLog:
    """Run an honest 2-phase (pre-shock -> shock -> post-shock adaptation) rollout on `S01`–`S05`."""
    assert_train_seed_only(seed)
    cls = engine_factory or _load_engine_cls()
    run_id = f"tb_{seed.lower()}_{run_tag}"
    engine = cls(
        seed=seed,
        run_id=run_id,
        agent_name=f"cryoforge_{run_tag}",
        agent_type="reference",
        model_id="cryoforge-s01-s05-expert",
        log_dir=None,
    )

    world, order_ids = _discover_episode(engine, seed)
    pre_plan = solve_batch_plan(world, order_ids, prefer_bay_b_first=prefer_bay_b_first)

    # Order the initial 2 pre-shock bookings by earliest bay start week
    ordered_pre = sorted(order_ids, key=lambda oid: (pre_plan[oid].bay_start_week, oid))
    booked_records: dict[str, dict[str, Any]] = {}

    # Optional sim-recovery demonstration: test a slightly thin wall first on Order 0, observe stress/boil-off, then fix
    if sim_mode == "sim_recovery":
        first_oid = ordered_pre[0]
        opt0 = pre_plan[first_oid].option
        probe_wall = max(MIN_WALL_THICKNESS_MM, round(opt0.spec.wall_thickness_mm - 2.5, 1))
        if probe_wall < opt0.spec.wall_thickness_mm:
            engine.run_physics_sim(
                order_id=first_oid,
                unit_count=opt0.spec.unit_count,
                unit_volume_m3=opt0.spec.unit_volume_m3,
                design_pressure_mpa=opt0.spec.design_pressure_mpa,
                wall_thickness_mm=probe_wall,
                insulation=opt0.spec.insulation.value,
                fittings=opt0.spec.fittings.value,
            )

    # Book first 2 orders (triggers mid-episode disruption on booking #2)
    for idx, oid in enumerate(ordered_pre[:2]):
        item = pre_plan[oid]
        spec = item.option.spec
        if (sim_mode == "targeted_2sim" and idx == 0) or sim_mode in ("sim_4orders", "sim_recovery"):
            sim_obs = engine.run_physics_sim(
                order_id=oid,
                unit_count=spec.unit_count,
                unit_volume_m3=spec.unit_volume_m3,
                design_pressure_mpa=spec.design_pressure_mpa,
                wall_thickness_mm=spec.wall_thickness_mm,
                insulation=spec.insulation.value,
                fittings=spec.fittings.value,
            )
            world.update_from_brief_or_factory(sim_obs)

        resp = engine.reserve_bay_and_order(
            order_id=oid,
            bay_id=item.option.bay_id.value,
            bay_start_week=item.bay_start_week,
            unit_count=spec.unit_count,
            unit_volume_m3=spec.unit_volume_m3,
            design_pressure_mpa=spec.design_pressure_mpa,
            wall_thickness_mm=spec.wall_thickness_mm,
            insulation=spec.insulation.value,
            fittings=spec.fittings.value,
        )
        world.update_from_brief_or_factory(resp)
        booked_records[oid] = _extract_booked_snapshot(oid, item, resp)

    # Post-shock state refresh & re-inspection of overridden orders/routes
    fstate_post = engine.get_factory_state()
    world.update_from_brief_or_factory(fstate_post)
    if world.disruption_event is not None:
        ev = world.disruption_event
        for oid in sorted(set(ev.order_min_pressure_overrides) | set(ev.order_sla_overrides)):
            if oid in world.orders:
                world.orders[oid] = _parse_customer_order(engine.inspect_order(order_id=oid))
        for rid in sorted(
            set(ev.route_mass_cap_overrides)
            | set(ev.route_std_height_overrides)
            | set(ev.route_permit_allowed_overrides)
        ):
            if rid in world.routes:
                world.routes[rid] = _parse_transport_route(engine.inspect_route(route_id=rid))

    # Re-optimize entire 4-order batch under post-shock world state
    post_plan = solve_batch_plan(
        world,
        order_ids,
        existing_bookings=booked_records,
        min_new_bay_start_week=1,
        prefer_bay_b_first=prefer_bay_b_first,
    )

    # 1) Revise any pre-shock bookings that require revision (ordered free-target-first)
    revised_oids = [oid for oid in ordered_pre[:2] if post_plan[oid].needs_revision]
    if len(revised_oids) == 2:
        o1, o2 = revised_oids[0], revised_oids[1]
        b2 = booked_records[o2]
        p1 = post_plan[o1]
        if p1.option.bay_id.value == b2["bay_id"] and _overlaps_window(
            p1.bay_start_week,
            p1.bay_start_week + p1.option.physics.fab_weeks,
            int(b2["bay_start_week"]),
            int(b2["bay_start_week"]) + int(b2.get("fab_weeks", 6)),
        ):
            revised_oids = [o2, o1]
    for r_idx, oid in enumerate(revised_oids):
        item = post_plan[oid]
        spec = item.option.spec
        if (sim_mode == "targeted_2sim" and r_idx == 0 and world.sim_calls_remaining >= 4) or (
            sim_mode == "sim_4orders" and world.sim_calls_remaining >= 3
        ):
            sim_obs = engine.run_physics_sim(
                order_id=oid,
                unit_count=spec.unit_count,
                unit_volume_m3=spec.unit_volume_m3,
                design_pressure_mpa=spec.design_pressure_mpa,
                wall_thickness_mm=spec.wall_thickness_mm,
                insulation=spec.insulation.value,
                fittings=spec.fittings.value,
            )
            world.update_from_brief_or_factory(sim_obs)

        order = world.orders[oid]
        rev_resp = engine.revise_order(
            order_id=oid,
            bay_id=item.option.bay_id.value,
            bay_start_week=item.bay_start_week,
            unit_count=spec.unit_count,
            unit_volume_m3=spec.unit_volume_m3,
            design_pressure_mpa=spec.design_pressure_mpa,
            wall_thickness_mm=spec.wall_thickness_mm,
            insulation=spec.insulation.value,
            fittings=spec.fittings.value,
            route_id=order.route_id,
            use_escort_permit=item.option.use_escort_permit,
        )
        if not rev_resp.get("ok", True):
            # Fallback if bay_start_week=1 was disallowed on revise_order post-shock
            alt_plan = solve_batch_plan(
                world,
                order_ids,
                existing_bookings=booked_records,
                min_new_bay_start_week=2,
                prefer_bay_b_first=prefer_bay_b_first,
            )
            post_plan = alt_plan
            item = post_plan[oid]
            spec = item.option.spec
            rev_resp = engine.revise_order(
                order_id=oid,
                bay_id=item.option.bay_id.value,
                bay_start_week=item.bay_start_week,
                unit_count=spec.unit_count,
                unit_volume_m3=spec.unit_volume_m3,
                design_pressure_mpa=spec.design_pressure_mpa,
                wall_thickness_mm=spec.wall_thickness_mm,
                insulation=spec.insulation.value,
                fittings=spec.fittings.value,
                route_id=order.route_id,
                use_escort_permit=item.option.use_escort_permit,
            )
        world.update_from_brief_or_factory(rev_resp)
        booked_records[oid] = _extract_booked_snapshot(oid, item, rev_resp, post_shock_revised=True)

    # 2) Book remaining 2 unbooked orders under post-shock plan
    for u_idx, oid in enumerate(ordered_pre[2:]):
        item = post_plan[oid]
        spec = item.option.spec
        if sim_mode == "sim_4orders" and world.sim_calls_remaining >= 2:
            sim_obs = engine.run_physics_sim(
                order_id=oid,
                unit_count=spec.unit_count,
                unit_volume_m3=spec.unit_volume_m3,
                design_pressure_mpa=spec.design_pressure_mpa,
                wall_thickness_mm=spec.wall_thickness_mm,
                insulation=spec.insulation.value,
                fittings=spec.fittings.value,
            )
            world.update_from_brief_or_factory(sim_obs)
        elif sim_mode == "targeted_2sim" and u_idx == 0 and len(revised_oids) == 0 and world.sim_calls_remaining >= 5:
            sim_obs = engine.run_physics_sim(
                order_id=oid,
                unit_count=spec.unit_count,
                unit_volume_m3=spec.unit_volume_m3,
                design_pressure_mpa=spec.design_pressure_mpa,
                wall_thickness_mm=spec.wall_thickness_mm,
                insulation=spec.insulation.value,
                fittings=spec.fittings.value,
            )
            world.update_from_brief_or_factory(sim_obs)

        res_resp = engine.reserve_bay_and_order(
            order_id=oid,
            bay_id=item.option.bay_id.value,
            bay_start_week=item.bay_start_week,
            unit_count=spec.unit_count,
            unit_volume_m3=spec.unit_volume_m3,
            design_pressure_mpa=spec.design_pressure_mpa,
            wall_thickness_mm=spec.wall_thickness_mm,
            insulation=spec.insulation.value,
            fittings=spec.fittings.value,
        )
        world.update_from_brief_or_factory(res_resp)
        booked_records[oid] = _extract_booked_snapshot(oid, item, res_resp)

    # 3) Explicitly assign transport for all 4 orders matching designated client routes
    for oid in order_ids:
        item = post_plan[oid]
        order = world.orders[oid]
        engine.assign_transport(
            order_id=oid,
            route_id=order.route_id,
            use_escort_permit=item.option.use_escort_permit,
        )

    # 4) Finalize commissioning
    engine.finalize_commissioning()
    return engine.export_trajectory_log()


def rollout_failure_episode(
    seed: str,
    failure_mode: str,
    *,
    engine_factory: Any | None = None,
) -> TrajectoryLog:
    """Generate realistic suboptimal/failure rollouts on `S01`–`S05` for DPO/GRPO contrast."""
    assert_train_seed_only(seed)
    cls = engine_factory or _load_engine_cls()
    agent_type = "stop_early" if failure_mode == "stop_early" else "small_before"
    engine = cls(
        seed=seed,
        run_id=f"tb_{seed.lower()}_rej_{failure_mode}",
        agent_name=f"contrast_{failure_mode}",
        agent_type=agent_type,
        model_id="cryoforge-contrastive-baseline",
        log_dir=None,
    )

    if failure_mode == "shock_unadapted":
        # Solves pre-shock state, triggers disruption after 2nd booking, but ignores disruption_alert
        world, order_ids = _discover_episode(engine, seed)
        pre_plan = solve_batch_plan(world, order_ids)
        ordered = sorted(order_ids, key=lambda oid: (pre_plan[oid].bay_start_week, oid))
        for oid in ordered:
            item = pre_plan[oid]
            spec = item.option.spec
            resp = engine.reserve_bay_and_order(
                order_id=oid,
                bay_id=item.option.bay_id.value,
                bay_start_week=item.bay_start_week,
                unit_count=spec.unit_count,
                unit_volume_m3=spec.unit_volume_m3,
                design_pressure_mpa=spec.design_pressure_mpa,
                wall_thickness_mm=max(MIN_WALL_THICKNESS_MM, round(spec.wall_thickness_mm - 1.5, 1)),
                insulation="PVI",
                fittings="CN",
            )
            # If blackout/crane rejected booking 3 or 4, shift week/bay blindly without fixing physics
            if not resp.get("ok", True):
                placed = False
                for fb_bay in (BayId.BAY_A.value, BayId.BAY_B.value):
                    if placed:
                        break
                    for fallback_w in range(2, 13):
                        r2 = engine.reserve_bay_and_order(
                            order_id=oid,
                            bay_id=fb_bay,
                            bay_start_week=fallback_w,
                            unit_count=1,
                            unit_volume_m3=min(5.0, spec.unit_volume_m3),
                            design_pressure_mpa=spec.design_pressure_mpa,
                            wall_thickness_mm=MIN_WALL_THICKNESS_MM,
                            insulation="PVI",
                            fittings="CN",
                        )
                        if r2.get("ok", True):
                            placed = True
                            break
        for oid in ordered:
            engine.assign_transport(order_id=oid, route_id=world.orders[oid].route_id, use_escort_permit=False)
        engine.finalize_commissioning()
        return engine.export_trajectory_log()

    if failure_mode == "sim_exhausted":
        # Skips handbook, burns all 8 FEA simulation calls on Order 1, then guesses undersized vessels
        brief = engine.get_episode_brief()
        fstate = engine.get_factory_state()
        order_ids = _extract_order_ids(brief, fstate)
        orders: dict[str, CustomerOrder] = {}
        for oid in order_ids:
            orders[oid] = _parse_customer_order(engine.inspect_order(order_id=oid))
        first_oid = order_ids[0]
        for v_try in (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0):
            engine.run_physics_sim(
                order_id=first_oid,
                unit_count=1,
                unit_volume_m3=v_try,
                design_pressure_mpa=MIN_DESIGN_PRESSURE_MPA,
                wall_thickness_mm=MIN_WALL_THICKNESS_MM,
                insulation="PVI",
                fittings="CN",
            )
        for idx, oid in enumerate(order_ids):
            bay = BayId.BAY_A.value if idx % 2 == 0 else BayId.BAY_B.value
            engine.reserve_bay_and_order(
                order_id=oid,
                bay_id=bay,
                bay_start_week=1 + (idx // 2) * 4,
                unit_count=1,
                unit_volume_m3=5.0,
                design_pressure_mpa=max(MIN_DESIGN_PRESSURE_MPA, orders[oid].min_pressure_mpa),
                wall_thickness_mm=MIN_WALL_THICKNESS_MM,
                insulation="PVI",
                fittings="CN",
            )
            engine.assign_transport(order_id=oid, route_id=orders[oid].route_id, use_escort_permit=True)
        engine.finalize_commissioning()
        return engine.export_trajectory_log()

    # stop_early: Cherry-picks only 1 order and finalizes immediately (capped <= 0.08 by batch gate)
    world, order_ids = _discover_episode(engine, seed)
    pre_plan = solve_batch_plan(world, order_ids)
    first_oid = order_ids[0]
    item = pre_plan[first_oid]
    spec = item.option.spec
    engine.reserve_bay_and_order(
        order_id=first_oid,
        bay_id=item.option.bay_id.value,
        bay_start_week=item.bay_start_week,
        unit_count=spec.unit_count,
        unit_volume_m3=spec.unit_volume_m3,
        design_pressure_mpa=spec.design_pressure_mpa,
        wall_thickness_mm=spec.wall_thickness_mm,
        insulation=spec.insulation.value,
        fittings=spec.fittings.value,
    )
    engine.assign_transport(
        order_id=first_oid,
        route_id=world.orders[first_oid].route_id,
        use_escort_permit=item.option.use_escort_permit,
    )
    engine.finalize_commissioning()
    return engine.export_trajectory_log()


# ---------------------------------------------------------------------------
# Spec-Exact Perturbed S01-S05 Simulator for `--augment` (ZERO S06-S08 Access)
# ---------------------------------------------------------------------------


class _PerturbedTrainScenarioEngine:
    """In-memory spec-compliant engine for perturbed `S01`–`S05` variations (`--augment`)."""

    def __init__(
        self,
        *,
        seed_label: str,
        orders: dict[str, CustomerOrder],
        routes: dict[str, TransportRoute],
        disruption: DisruptionEvent,
        horizon_weeks: int,
        initial_budget_eur: float,
        initial_permits: int,
        handbook_obs: dict[str, Any],
    ) -> None:
        assert_train_seed_only(seed_label)
        self.seed = seed_label
        self._orders = {k: v.model_copy(deep=True) for k, v in orders.items()}
        self._routes = {k: v.model_copy(deep=True) for k, v in routes.items()}
        self._disruption = disruption.model_copy(deep=True)
        self.horizon_weeks = horizon_weeks
        self.effective_budget_eur = initial_budget_eur
        self.effective_permits = initial_permits
        self._handbook_obs = handbook_obs
        self.sim_calls_remaining = SIM_BUDGET_DEFAULT
        self.step_count = 0
        self.mutation_count = 0
        self.disruption_triggered = False
        self.pvi_k_multiplier = 1.0
        self.fittings_lead_deltas: dict[FittingsGrade, int] = {}
        self.bay_blackouts: list[BayBlackoutWindow] = []
        self.bay_max_unit_tare_kg: dict[BayId, float] = {}
        self.bookings: dict[str, OrderBooking] = {}
        self.retooling_fees_eur = 0.0
        self.finalized = False
        self.steps: list[TrajectoryStepLog] = []
        self.started_at = datetime.now(timezone.utc).isoformat()

    def _log(self, tool_name: str, args: dict[str, Any], obs: dict[str, Any], shock_now: bool = False) -> dict[str, Any]:
        self.step_count += 1
        self.steps.append(
            TrajectoryStepLog(
                step_index=self.step_count,
                timestamp_iso=datetime.now(timezone.utc).isoformat(),
                tool_name=tool_name,
                arguments=args,
                observation=obs,
                disruption_triggered_this_step=shock_now,
            )
        )
        return obs

    def _total_spent(self) -> float:
        total = self.retooling_fees_eur
        for b in self.bookings.values():
            total += b.manufacturing_cost_eur
            if b.assigned_route_id and b.assigned_route_id in self._routes:
                r = self._routes[b.assigned_route_id]
                total += b.spec.unit_count * r.base_freight_eur_per_unit
                if b.use_escort_permit:
                    total += r.permit_fee_eur
        return round(total, 2)

    def _maybe_trigger_shock(self) -> bool:
        if self.disruption_triggered:
            return False
        if len(self.bookings) >= DISRUPTION_TRIGGER_BOOKINGS or self.mutation_count >= DISRUPTION_TRIGGER_MUTATIONS:
            self.disruption_triggered = True
            ev = self._disruption
            self.pvi_k_multiplier = ev.pvi_k_multiplier
            self.effective_budget_eur = round(self.effective_budget_eur + ev.budget_delta_eur, 2)
            self.effective_permits = max(0, self.effective_permits + ev.escort_permits_delta)
            for fg, d in ev.fittings_lead_deltas.items():
                self.fittings_lead_deltas[FittingsGrade(fg)] = int(d)
            self.bay_blackouts.extend(ev.bay_blackouts)
            for bid, cap in ev.bay_max_unit_tare_kg.items():
                self.bay_max_unit_tare_kg[BayId(bid)] = float(cap)
            for rid, mcap in ev.route_mass_cap_overrides.items():
                if rid in self._routes:
                    self._routes[rid] = self._routes[rid].model_copy(update={"max_unit_transport_mass_kg": float(mcap)})
            for rid, sh in ev.route_std_height_overrides.items():
                if rid in self._routes:
                    self._routes[rid] = self._routes[rid].model_copy(update={"std_max_height_mm": int(sh)})
            for rid, pok in ev.route_permit_allowed_overrides.items():
                if rid in self._routes:
                    self._routes[rid] = self._routes[rid].model_copy(update={"escort_permit_allowed": bool(pok)})
            for oid, mp in ev.order_min_pressure_overrides.items():
                if oid in self._orders:
                    self._orders[oid] = self._orders[oid].model_copy(update={"min_pressure_mpa": float(mp)})
            for oid, sw in ev.order_sla_overrides.items():
                if oid in self._orders:
                    self._orders[oid] = self._orders[oid].model_copy(update={"sla_week": int(sw)})
            return True
        return False

    def get_episode_brief(self) -> dict[str, Any]:
        obs = {
            "ok": True,
            "seed": self.seed,
            "horizon_weeks": self.horizon_weeks,
            "total_budget_eur": self.effective_budget_eur,
            "escort_permits_total": self.effective_permits,
            "sim_budget_total": SIM_BUDGET_DEFAULT,
            "order_ids": list(self._orders.keys()),
            "route_ids": list(self._routes.keys()),
            "bays": [BayId.BAY_A.value, BayId.BAY_B.value],
        }
        return self._log("get_episode_brief", {}, obs)

    def query_handbook(self, section: str = "all") -> dict[str, Any]:
        return self._log("query_handbook", {"section": section}, dict(self._handbook_obs))

    def inspect_order(self, order_id: str) -> dict[str, Any]:
        obs = {"ok": True, "order": self._orders[order_id].model_dump(mode="json")}
        return self._log("inspect_order", {"order_id": order_id}, obs)

    def inspect_route(self, route_id: str) -> dict[str, Any]:
        obs = {"ok": True, "route": self._routes[route_id].model_dump(mode="json")}
        return self._log("inspect_route", {"route_id": route_id}, obs)

    def get_factory_state(self) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "ok": True,
            "effective_budget_eur": self.effective_budget_eur,
            "total_spent_eur": self._total_spent(),
            "budget_remaining_eur": round(self.effective_budget_eur - self._total_spent(), 2),
            "retooling_fees_eur": self.retooling_fees_eur,
            "effective_escort_permits_total": self.effective_permits,
            "sim_calls_remaining": self.sim_calls_remaining,
            "pvi_k_multiplier": self.pvi_k_multiplier,
            "fittings_lead_deltas": {k.value: v for k, v in self.fittings_lead_deltas.items()},
            "bay_blackouts": [b.model_dump(mode="json") for b in self.bay_blackouts],
            "bay_max_unit_tare_kg": {k.value: v for k, v in self.bay_max_unit_tare_kg.items()},
            "bookings": {k: v.model_dump(mode="json") for k, v in self.bookings.items()},
            "disruption_triggered": self.disruption_triggered,
            "disruption_alert": self._disruption.model_dump(mode="json") if self.disruption_triggered else None,
        }
        return self._log("get_factory_state", {}, obs)

    def run_physics_sim(self, **kwargs: Any) -> dict[str, Any]:
        if self.sim_calls_remaining <= 0:
            return self._log("run_physics_sim", kwargs, {"ok": False, "error": "SIM_BUDGET_EXHAUSTED", "sim_calls_remaining": 0})
        self.sim_calls_remaining -= 1
        order = self._orders[kwargs["order_id"]]
        spec = VesselDesignSpec(
            unit_count=int(kwargs["unit_count"]),
            unit_volume_m3=float(kwargs["unit_volume_m3"]),
            design_pressure_mpa=float(kwargs["design_pressure_mpa"]),
            wall_thickness_mm=float(kwargs["wall_thickness_mm"]),
            insulation=InsulationType(kwargs["insulation"]),
            fittings=FittingsGrade(kwargs["fittings"]),
        )
        phys = evaluate_design_physics(
            order,
            spec,
            pvi_k_multiplier=self.pvi_k_multiplier,
            fittings_lead_delta=self.fittings_lead_deltas.get(spec.fittings, 0),
        )
        return self._log(
            "run_physics_sim",
            kwargs,
            {"ok": True, "sim_calls_remaining": self.sim_calls_remaining, "physics": phys.model_dump(mode="json")},
        )

    def reserve_bay_and_order(self, **kwargs: Any) -> dict[str, Any]:
        order_id = str(kwargs["order_id"])
        bay_id = BayId(kwargs["bay_id"])
        start_w = int(kwargs["bay_start_week"])
        spec = VesselDesignSpec(
            unit_count=int(kwargs["unit_count"]),
            unit_volume_m3=float(kwargs["unit_volume_m3"]),
            design_pressure_mpa=float(kwargs["design_pressure_mpa"]),
            wall_thickness_mm=float(kwargs["wall_thickness_mm"]),
            insulation=InsulationType(kwargs["insulation"]),
            fittings=FittingsGrade(kwargs["fittings"]),
        )
        errs = validate_catalogue_bounds(spec)
        if errs:
            return self._log("reserve_bay_and_order", kwargs, {"ok": False, "error": "; ".join(errs)})

        order = self._orders[order_id]
        dims = compute_dimensions(spec.unit_volume_m3, order.execution, spec.insulation)
        _, tare_kg, _ = compute_masses_and_area(spec, order.execution, dims)
        fab_w, _, mfg_eur = compute_fab_and_cost(
            spec, order.execution, tare_kg, fittings_lead_delta=self.fittings_lead_deltas.get(spec.fittings, 0)
        )
        end_w = start_w + fab_w
        for other in self.bookings.values():
            if other.bay_id == bay_id and _overlaps_window(start_w, end_w, other.bay_start_week, other.fab_end_week):
                return self._log("reserve_bay_and_order", kwargs, {"ok": False, "error": "BAY_OVERLAP"})
        for bo in self.bay_blackouts:
            if bo.bay_id == bay_id and _overlaps_window(start_w, end_w, bo.start_week, bo.end_week):
                return self._log("reserve_bay_and_order", kwargs, {"ok": False, "error": "BAY_BLACKOUT_CONFLICT"})
        if bay_id in self.bay_max_unit_tare_kg and tare_kg > self.bay_max_unit_tare_kg[bay_id]:
            return self._log("reserve_bay_and_order", kwargs, {"ok": False, "error": "CRANE_TARE_EXCEEDED"})

        release_w = 2 if self.disruption_triggered else 1
        booking = OrderBooking(
            order_id=order_id,
            bay_id=bay_id,
            bay_start_week=start_w,
            fab_weeks=fab_w,
            fab_end_week=end_w,
            booked_at_step=self.step_count + 1,
            order_release_week=release_w,
            spec=spec,
            manufacturing_cost_eur=mfg_eur,
        )
        self.bookings[order_id] = booking
        self.mutation_count += 1
        shock_now = self._maybe_trigger_shock()
        obs = {
            "ok": True,
            "booking": booking.model_dump(mode="json"),
            "budget_remaining_eur": round(self.effective_budget_eur - self._total_spent(), 2),
            "disruption_alert": self._disruption.model_dump(mode="json") if self.disruption_triggered else None,
        }
        return self._log("reserve_bay_and_order", kwargs, obs, shock_now=shock_now)

    def assign_transport(self, order_id: str, route_id: str, use_escort_permit: bool = False) -> dict[str, Any]:
        args = {"order_id": order_id, "route_id": route_id, "use_escort_permit": use_escort_permit}
        if order_id not in self.bookings:
            return self._log("assign_transport", args, {"ok": False, "error": "ORDER_NOT_BOOKED"})
        b = self.bookings[order_id]
        b.assigned_route_id = route_id
        b.use_escort_permit = bool(use_escort_permit)
        self.mutation_count += 1
        shock_now = self._maybe_trigger_shock()
        obs = {
            "ok": True,
            "booking": b.model_dump(mode="json"),
            "budget_remaining_eur": round(self.effective_budget_eur - self._total_spent(), 2),
            "disruption_alert": self._disruption.model_dump(mode="json") if self.disruption_triggered else None,
        }
        return self._log("assign_transport", args, obs, shock_now=shock_now)

    def revise_order(self, **kwargs: Any) -> dict[str, Any]:
        order_id = str(kwargs["order_id"])
        if order_id not in self.bookings:
            return self._log("revise_order", kwargs, {"ok": False, "error": "ORDER_NOT_BOOKED"})
        prev = self.bookings[order_id]
        bay_id = BayId(kwargs["bay_id"])
        start_w = int(kwargs["bay_start_week"])
        spec = VesselDesignSpec(
            unit_count=int(kwargs["unit_count"]),
            unit_volume_m3=float(kwargs["unit_volume_m3"]),
            design_pressure_mpa=float(kwargs["design_pressure_mpa"]),
            wall_thickness_mm=float(kwargs["wall_thickness_mm"]),
            insulation=InsulationType(kwargs["insulation"]),
            fittings=FittingsGrade(kwargs["fittings"]),
        )
        order = self._orders[order_id]
        dims = compute_dimensions(spec.unit_volume_m3, order.execution, spec.insulation)
        _, tare_kg, _ = compute_masses_and_area(spec, order.execution, dims)
        fab_w, _, mfg_eur = compute_fab_and_cost(
            spec, order.execution, tare_kg, fittings_lead_delta=self.fittings_lead_deltas.get(spec.fittings, 0)
        )
        end_w = start_w + fab_w
        for oid, other in self.bookings.items():
            if oid != order_id and other.bay_id == bay_id and _overlaps_window(start_w, end_w, other.bay_start_week, other.fab_end_week):
                return self._log("revise_order", kwargs, {"ok": False, "error": "BAY_OVERLAP"})
        for bo in self.bay_blackouts:
            if bo.bay_id == bay_id and _overlaps_window(start_w, end_w, bo.start_week, bo.end_week):
                return self._log("revise_order", kwargs, {"ok": False, "error": "BAY_BLACKOUT_CONFLICT"})
        if bay_id in self.bay_max_unit_tare_kg and tare_kg > self.bay_max_unit_tare_kg[bay_id]:
            return self._log("revise_order", kwargs, {"ok": False, "error": "CRANE_TARE_EXCEEDED"})

        fee = (
            POST_SHOCK_FIRST_REVISION_FEE_EUR
            if (self.disruption_triggered and not prev.post_shock_revised)
            else STANDARD_REVISION_FEE_EUR
        )
        self.retooling_fees_eur = round(self.retooling_fees_eur + fee, 2)
        changed_proc = (spec.fittings != prev.spec.fittings) or (spec.unit_count != prev.spec.unit_count)
        release_w = 2 if (self.disruption_triggered and changed_proc) else prev.order_release_week
        updated = OrderBooking(
            order_id=order_id,
            bay_id=bay_id,
            bay_start_week=start_w,
            fab_weeks=fab_w,
            fab_end_week=end_w,
            booked_at_step=self.step_count + 1,
            order_release_week=release_w,
            spec=spec,
            manufacturing_cost_eur=mfg_eur,
            assigned_route_id=kwargs.get("route_id") if kwargs.get("route_id") is not None else prev.assigned_route_id,
            use_escort_permit=bool(kwargs["use_escort_permit"]) if kwargs.get("use_escort_permit") is not None else prev.use_escort_permit,
            revision_count=prev.revision_count + 1,
            post_shock_revised=True if self.disruption_triggered else prev.post_shock_revised,
        )
        self.bookings[order_id] = updated
        self.mutation_count += 1
        shock_now = self._maybe_trigger_shock()
        obs = {
            "ok": True,
            "booking": updated.model_dump(mode="json"),
            "revision_fee_eur": fee,
            "budget_remaining_eur": round(self.effective_budget_eur - self._total_spent(), 2),
            "disruption_alert": self._disruption.model_dump(mode="json") if self.disruption_triggered else None,
        }
        return self._log("revise_order", kwargs, obs, shock_now=shock_now)

    def finalize_commissioning(self) -> dict[str, Any]:
        self.finalized = True
        report = self._verify()
        obs = {"ok": True, "finalized": True, "verification": report.model_dump(mode="json")}
        return self._log("finalize_commissioning", {}, obs)

    def _verify(self) -> VerificationReport:
        permits_used = sum(1 for b in self.bookings.values() if b.use_escort_permit)
        permit_exceeded = permits_used > self.effective_permits
        spent = self._total_spent()
        budget_exceeded = spent > self.effective_budget_eur + 1e-6

        results: list[OrderVerificationResult] = []
        valid_count = 0
        slacks: list[int] = []
        all_failures: list[str] = []

        for oid, order in self._orders.items():
            b = self.bookings.get(oid)
            if b is None:
                results.append(
                    OrderVerificationResult(
                        order_id=oid,
                        booked=False,
                        route_assigned=False,
                        passed_all_hard_checks=False,
                        order_raw_score=0.0,
                        weighted_contribution=0.0,
                        sla_week=order.sla_week,
                        violations=["ORDER_NOT_BOOKED"],
                    )
                )
                all_failures.append("ORDER_NOT_BOOKED")
                continue

            phys = evaluate_design_physics(
                order,
                b.spec,
                pvi_k_multiplier=self.pvi_k_multiplier,
                fittings_lead_delta=self.fittings_lead_deltas.get(b.spec.fittings, 0),
            )
            viols = list(phys.violations)
            route_ok = b.assigned_route_id == order.route_id
            if not route_ok:
                viols.append("ROUTE_NOT_ASSIGNED_OR_MISMATCH")

            deliv_w: int | None = None
            order_cost = b.manufacturing_cost_eur
            if route_ok and b.assigned_route_id in self._routes:
                r = self._routes[b.assigned_route_id]
                order_cost += b.spec.unit_count * r.base_freight_eur_per_unit + (r.permit_fee_eur if b.use_escort_permit else 0.0)
                deliv_w = max(b.fab_end_week, b.order_release_week + phys.fittings_lead_weeks) + ASSEMBLY_COMMISSION_WEEKS + r.transit_weeks
                if deliv_w > order.sla_week or deliv_w > self.horizon_weeks:
                    viols.append("SLA_OR_HORIZON_EXCEEDED")
                if phys.unit_transport_mass_kg > r.max_unit_transport_mass_kg + 1e-9:
                    viols.append("ROUTE_MASS_EXCEEDED")
                dims = phys.dimensions
                if b.use_escort_permit:
                    if not r.escort_permit_allowed or dims.shipping_width_mm > r.permit_max_width_mm or dims.shipping_height_mm > r.permit_max_height_mm:
                        viols.append("PERMIT_ENVELOPE_EXCEEDED")
                    if permit_exceeded:
                        viols.append("PERMIT_QUOTA_EXCEEDED")
                else:
                    if dims.shipping_width_mm > r.std_max_width_mm or dims.shipping_height_mm > r.std_max_height_mm:
                        viols.append("STD_TRANSPORT_ENVELOPE_EXCEEDED")

            passed = len(viols) == 0 and phys.all_engineering_pass and route_ok
            if passed and deliv_w is not None:
                valid_count += 1
                slacks.append(order.sla_week - deliv_w)
            all_failures.extend(viols)
            contrib = round(order.priority_weight * 0.85, 4) if passed else 0.0
            results.append(
                OrderVerificationResult(
                    order_id=oid,
                    booked=True,
                    route_assigned=route_ok,
                    passed_all_hard_checks=passed,
                    order_raw_score=1.0 if passed else 0.0,
                    weighted_contribution=contrib,
                    delivery_week=deliv_w,
                    sla_week=order.sla_week,
                    total_order_cost_eur=round(order_cost, 2),
                    physics=phys,
                    violations=viols,
                )
            )

        raw_score = sum(r.weighted_contribution for r in results)
        sim_used = SIM_BUDGET_DEFAULT - self.sim_calls_remaining
        budget_bonus = 0.0
        slack_bonus = 0.0
        sim_bonus = 0.0
        if valid_count == 4 and not budget_exceeded:
            budget_bonus = round(
                0.07 * min(1.0, max(0.0, (self.effective_budget_eur - spent) / max(1.0, 0.18 * self.effective_budget_eur))),
                4,
            )
            slack_bonus = round(0.04 * min(1.0, max(0.0, (sum(slacks) / 4.0) / 2.0)), 4)
            sim_bonus = round(0.04 * ((SIM_BUDGET_DEFAULT - sim_used) / SIM_BUDGET_DEFAULT), 4)

        total_score = raw_score + budget_bonus + slack_bonus + sim_bonus
        cap_applied = False
        if valid_count == 0:
            total_score = 0.0
        elif valid_count == 1:
            total_score = min(total_score, 0.08)
            cap_applied = True
        elif valid_count == 2:
            total_score *= 0.80
            cap_applied = True
        elif valid_count == 3:
            total_score *= 0.92
            cap_applied = True
        if budget_exceeded:
            total_score = min(total_score * 0.20, 0.15)
        if not self.finalized:
            total_score = 0.0

        return VerificationReport(
            seed=self.seed,
            finalized=self.finalized,
            final_score=round(total_score, 4),
            valid_order_count=valid_count,
            total_order_count=len(self._orders),
            batch_gate_cap_applied=cap_applied,
            budget_exceeded=budget_exceeded,
            permit_quota_exceeded=permit_exceeded,
            total_budget_eur=self.effective_budget_eur,
            total_spent_eur=spent,
            budget_efficiency_bonus=budget_bonus,
            schedule_margin_bonus=slack_bonus,
            sim_efficiency_bonus=sim_bonus,
            sim_calls_used=sim_used,
            sim_budget_total=SIM_BUDGET_DEFAULT,
            step_count=self.step_count,
            disruption_triggered=self.disruption_triggered,
            order_results=results,
            failure_modes=sorted(set(all_failures)),
            primary_failure_categories=sorted({f.split(":")[0] for f in all_failures}),
        )

    def export_trajectory_log(self) -> TrajectoryLog:
        return TrajectoryLog(
            run_id=f"tb_{self.seed.lower()}_perturbed",
            seed=self.seed,
            agent_name="cryoforge_augmented_expert",
            agent_type="reference",
            model_id="cryoforge-s01-s05-augmented",
            started_at=self.started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            steps=self.steps,
            verification=self._verify(),
            state_hmac="augmented_s01_s05_only",
        )


def _generate_perturbed_s01_s05_rollouts() -> list[TrajectoryLog]:
    """Generate parameterized perturbations strictly from `S01`–`S05` (NEVER `S06`–`S08`)."""
    cls = _load_engine_cls()
    augmented: list[TrajectoryLog] = []
    perturbations = [
        ("aug_mass_up", 1.08, 1.00, 0.0),
        ("aug_boil_tight", 0.96, 0.92, 0.15),
    ]

    for seed in EVAL_SEEDS:
        assert_train_seed_only(seed)
        probe = cls(seed=seed, agent_name="probe", agent_type="reference", model_id="probe", log_dir=None)
        brief = probe.get_episode_brief()
        handbook_obs = probe.query_handbook(section="all")
        fstate = probe.get_factory_state()
        world_init = ObservedWorldState(seed=seed)
        world_init.update_from_brief_or_factory(brief)
        world_init.update_from_brief_or_factory(fstate)
        oids = _extract_order_ids(brief, fstate)
        base_orders = {oid: _parse_customer_order(probe.inspect_order(order_id=oid)) for oid in oids}
        rids = _extract_route_ids(brief, base_orders)
        base_routes = {rid: _parse_transport_route(probe.inspect_route(route_id=rid)) for rid in rids}

        # Trigger shock on probe solely to extract S01-S05's DisruptionEvent
        pre_plan = solve_batch_plan(
            ObservedWorldState(
                seed=seed,
                horizon_weeks=world_init.horizon_weeks,
                effective_budget_eur=world_init.effective_budget_eur,
                effective_escort_permits_total=world_init.effective_escort_permits_total,
                orders={k: v.model_copy(deep=True) for k, v in base_orders.items()},
                routes={k: v.model_copy(deep=True) for k, v in base_routes.items()},
                bay_blackouts=list(world_init.bay_blackouts),
                bay_max_unit_tare_kg=dict(world_init.bay_max_unit_tare_kg),
            ),
            oids,
        )
        disruption_ev: DisruptionEvent | None = None
        for oid in oids[:2]:
            it = pre_plan[oid]
            sp = it.option.spec
            r_obs = probe.reserve_bay_and_order(
                order_id=oid,
                bay_id=it.option.bay_id.value,
                bay_start_week=it.bay_start_week,
                unit_count=sp.unit_count,
                unit_volume_m3=sp.unit_volume_m3,
                design_pressure_mpa=sp.design_pressure_mpa,
                wall_thickness_mm=sp.wall_thickness_mm,
                insulation=sp.insulation.value,
                fittings=sp.fittings.value,
            )
            disruption_ev = _parse_disruption(r_obs) or disruption_ev
        if disruption_ev is None:
            disruption_ev = _parse_disruption(probe.get_factory_state())
        if disruption_ev is None:
            continue

        for tag, mass_mult, boil_mult, press_delta in perturbations:
            pert_orders: dict[str, CustomerOrder] = {}
            for oid, o in base_orders.items():
                pert_orders[oid] = o.model_copy(
                    update={
                        "required_product_mass_kg": round(o.required_product_mass_kg * mass_mult, 1),
                        "max_boil_off_pct_day": round(max(0.08, o.max_boil_off_pct_day * boil_mult), 4),
                        "min_pressure_mpa": round(min(3.8, o.min_pressure_mpa + press_delta), 2),
                    }
                )
            seed_label = f"{seed}_{tag}"
            assert_train_seed_only(seed_label)

            def _factory(*_a: Any, _orders: dict[str, CustomerOrder] = pert_orders, _lbl: str = seed_label, **_kw: Any) -> _PerturbedTrainScenarioEngine:
                return _PerturbedTrainScenarioEngine(
                    seed_label=_lbl,
                    orders=_orders,
                    routes=base_routes,
                    disruption=disruption_ev,
                    horizon_weeks=world_init.horizon_weeks,
                    initial_budget_eur=round(world_init.effective_budget_eur * 1.06, 2),
                    initial_permits=world_init.effective_escort_permits_total,
                    handbook_obs=handbook_obs,
                )

            try:
                traj = rollout_expert_episode(
                    seed_label,
                    engine_factory=_factory,
                    sim_mode="targeted_2sim",
                    run_tag=tag,
                )
                if traj.verification.final_score >= 0.94 and traj.verification.valid_order_count == 4:
                    augmented.append(traj)
            except RuntimeError:
                # Perturbation made scenario infeasible; skip cleanly
                continue

    return augmented


# ---------------------------------------------------------------------------
# Explicit Engineering Chain-of-Thought Synthesizer for Multi-Turn SFT
# ---------------------------------------------------------------------------


def _synthesize_step_cot(
    step: TrajectoryStepLog,
    prev_step: TrajectoryStepLog | None,
    tracker: ObservedWorldState,
    booked_tracker: dict[str, dict[str, Any]],
) -> str:
    """Construct grounded, analytical Chain-of-Thought before `step.tool_name`."""
    shock_prefix = ""
    if prev_step is not None and prev_step.disruption_triggered_this_step and tracker.disruption_event is not None:
        ev = tracker.disruption_event
        impacts: list[str] = [f"Disruption '{ev.disruption_id}' ({ev.title}) just fired!"]
        if ev.pvi_k_multiplier != 1.0:
            impacts.append(f"PVI thermal conductivity multiplied by {ev.pvi_k_multiplier}x (re-check PVI boil-off).")
        if ev.fittings_lead_deltas:
            impacts.append(
                "Fittings lead deltas: "
                + ", ".join(f"{k.value}={v:+d}w" for k, v in ev.fittings_lead_deltas.items())
                + " (and post-shock procurement release jumps to week 2)."
            )
        if ev.bay_blackouts:
            impacts.append(
                "New bay blackouts: "
                + ", ".join(f"{b.bay_id.value}[w{b.start_week},w{b.end_week})" for b in ev.bay_blackouts)
                + "."
            )
        if ev.bay_max_unit_tare_kg:
            impacts.append(
                "Crane unit tare caps: "
                + ", ".join(f"{k.value}<={v:.0f}kg" for k, v in ev.bay_max_unit_tare_kg.items())
                + "."
            )
        if ev.order_min_pressure_overrides or ev.order_sla_overrides:
            impacts.append(
                f"Order overrides on {sorted(set(ev.order_min_pressure_overrides) | set(ev.order_sla_overrides))}."
            )
        if ev.route_mass_cap_overrides or ev.route_std_height_overrides or ev.route_permit_allowed_overrides:
            impacts.append(
                "Route overrides on "
                + str(
                    sorted(
                        set(ev.route_mass_cap_overrides)
                        | set(ev.route_std_height_overrides)
                        | set(ev.route_permit_allowed_overrides)
                    )
                )
                + "."
            )
        if ev.budget_delta_eur != 0.0 or ev.escort_permits_delta != 0:
            impacts.append(
                f"Budget delta={ev.budget_delta_eur:+.0f} EUR, escort permits delta={ev.escort_permits_delta:+d}."
            )
        shock_prefix = "[POST-SHOCK AUDIT] " + " ".join(impacts) + "\n"

    tname = step.tool_name
    args = step.arguments

    if tname == "get_episode_brief":
        return shock_prefix + (
            "Step 1: Query get_episode_brief() to obtain the batch order IDs, transport routes, "
            "planning horizon, total budget, escort permit quota, and 8-call FEA simulation budget."
        )
    if tname == "query_handbook":
        return shock_prefix + (
            "Read the complete CryoForge engineering handbook ('all') so we apply exact GOST/ASME "
            "hoop-stress, boil-off, medium compatibility, shipping envelope, and delivery-week "
            "equations analytically rather than wasting our 8 FEA simulation calls."
        )
    if tname == "get_factory_state":
        if tracker.disruption_triggered:
            return shock_prefix + (
                "Inspect live post-shock factory state to verify updated effective budget, permit quota, "
                "bay blackout intervals, crane tare limits, and existing bay bookings before revising."
            )
        return shock_prefix + (
            "Check initial factory bay state (active blackouts, crane tare caps, remaining budget, "
            "and permit quota) before inspecting individual orders."
        )
    if tname == "inspect_order":
        return shock_prefix + (
            f"Inspect customer order {args.get('order_id')} to record working cryogenic medium, "
            "execution type, required deliverable product mass, minimum pressure, max daily boil-off, "
            "SLA delivery week, site clearance envelope, and designated route."
        )
    if tname == "inspect_route":
        return shock_prefix + (
            f"Inspect transport route {args.get('route_id')} for transit weeks, standard vs escort-permit "
            "width/height clearances, unit transport mass cap, freight rate, and permit fee."
        )

    if tname in ("run_physics_sim", "reserve_bay_and_order", "revise_order"):
        oid = str(args.get("order_id", ""))
        order = tracker.orders.get(oid)
        if order is not None and all(
            k in args for k in ("unit_count", "unit_volume_m3", "design_pressure_mpa", "wall_thickness_mm", "insulation", "fittings")
        ):
            spec = VesselDesignSpec(
                unit_count=int(args["unit_count"]),
                unit_volume_m3=float(args["unit_volume_m3"]),
                design_pressure_mpa=float(args["design_pressure_mpa"]),
                wall_thickness_mm=float(args["wall_thickness_mm"]),
                insulation=InsulationType(str(args["insulation"])),
                fittings=FittingsGrade(str(args["fittings"])),
            )
            phys = evaluate_design_physics(
                order,
                spec,
                pvi_k_multiplier=tracker.pvi_k_multiplier,
                fittings_lead_delta=tracker.fittings_lead_deltas.get(spec.fittings, 0),
            )
            route = tracker.routes.get(order.route_id)
            route_note = ""
            if route is not None:
                dims = phys.dimensions
                std_ok = dims.shipping_width_mm <= route.std_max_width_mm and dims.shipping_height_mm <= route.std_max_height_mm
                route_note = (
                    f" Shipping WxH={dims.shipping_width_mm}x{dims.shipping_height_mm}mm "
                    f"({'fits std' if std_ok else 'requires escort permit'} on {route.route_id}), "
                    f"transport_mass={phys.unit_transport_mass_kg}kg <= {route.max_unit_transport_mass_kg}kg."
                )
            calc_line = (
                f"Analytical check for {oid} ({order.medium.value}, {order.execution.value}): "
                f"{spec.unit_count}x{spec.unit_volume_m3}m3, P={spec.design_pressure_mpa}MPa >= {order.min_pressure_mpa}MPa, "
                f"wall={spec.wall_thickness_mm}mm, {spec.insulation.value}/{spec.fittings.value} -> "
                f"usable_mass={phys.usable_total_mass_kg}kg (req {order.required_product_mass_kg}kg), "
                f"hoop_stress={phys.hoop_stress_mpa}MPa <= {phys.allowable_stress_mpa}MPa, "
                f"boil_off={phys.boil_off_pct_day}%/d <= {order.max_boil_off_pct_day}%/d, "
                f"tare={phys.unit_tare_mass_kg}kg, fab={phys.fab_weeks}w, lead={phys.fittings_lead_weeks}w, "
                f"mfg_cost={phys.manufacturing_cost_eur:.0f} EUR.{route_note}"
            )
            if tname == "run_physics_sim":
                return shock_prefix + (
                    f"{calc_line} Spend 1 FEA simulation call ({tracker.sim_calls_remaining} remaining) "
                    f"to confirm full engineering compliance on {oid} while conserving reserve sims."
                )
            bay_s = int(args.get("bay_start_week", 1))
            bay_id = str(args.get("bay_id", "BAY-A"))
            rel_w = 2 if tracker.disruption_triggered else 1
            if tname == "revise_order" and oid in booked_tracker:
                prev_b = booked_tracker[oid]
                same_proc = (
                    prev_b.get("fittings") == spec.fittings.value
                    and int(prev_b.get("unit_count", 0)) == spec.unit_count
                )
                rel_w = int(prev_b.get("order_release_week", 1)) if same_proc else 2
            transit_w = route.transit_weeks if route is not None else 0
            deliv_w = max(bay_s + phys.fab_weeks, rel_w + phys.fittings_lead_weeks) + ASSEMBLY_COMMISSION_WEEKS + transit_w
            sched_line = (
                f" Schedule on {bay_id} [w{bay_s}, w{bay_s + phys.fab_weeks}): "
                f"delivery_week = max({bay_s + phys.fab_weeks}, {rel_w + phys.fittings_lead_weeks}) + 1 + {transit_w} "
                f"= w{deliv_w} <= SLA w{order.sla_week}."
            )
            action_verb = "Revise post-shock invalidated booking" if tname == "revise_order" else "Book bay and procurement"
            return shock_prefix + f"{calc_line}{sched_line} {action_verb} for {oid}."

    if tname == "assign_transport":
        oid = str(args.get("order_id", ""))
        rid = str(args.get("route_id", ""))
        perm = bool(args.get("use_escort_permit", False))
        return shock_prefix + (
            f"Assign order {oid} to its designated client route {rid} with "
            f"use_escort_permit={perm}, respecting shipping envelope and global permit quota."
        )

    if tname == "finalize_commissioning":
        return shock_prefix + (
            "Final pre-commissioning audit complete: all 4 orders are booked on non-overlapping "
            "bay windows outside blackouts, post-shock mechanical/thermal/material checks pass, "
            "all 4 designated transport routes are assigned within permit and budget limits. "
            "Call finalize_commissioning() to lock the batch score."
        )

    return shock_prefix + f"Execute {tname} with verified parameters."


def trajectory_to_messages(
    traj: TrajectoryLog,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    task_instruction: str | None = None,
    include_cot: bool = True,
) -> list[dict[str, Any]]:
    """Convert a `TrajectoryLog` into OpenAI / HuggingFace TRL multi-turn tool-calling `messages`."""
    assert_train_seed_only(traj.seed)
    instruction = task_instruction or load_task_instruction()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    tracker = ObservedWorldState(seed=traj.seed)
    booked_tracker: dict[str, dict[str, Any]] = {}
    prev_step: TrajectoryStepLog | None = None

    for step in traj.steps:
        cot = _synthesize_step_cot(step, prev_step, tracker, booked_tracker) if include_cot else ""
        call_id = f"call_{traj.seed.lower()}_{step.step_index:02d}"
        messages.append(
            {
                "role": "assistant",
                "content": cot,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": step.tool_name,
                            "arguments": json.dumps(step.arguments, ensure_ascii=False, sort_keys=True),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": step.tool_name,
                "content": json.dumps(step.observation, ensure_ascii=False),
            }
        )

        # Advance causal state tracker AFTER observation arrives (zero lookahead!)
        obs = step.observation if isinstance(step.observation, dict) else {}
        if step.tool_name in ("get_episode_brief", "get_factory_state", "run_physics_sim", "reserve_bay_and_order", "revise_order", "assign_transport"):
            tracker.update_from_brief_or_factory(obs)
        if step.tool_name == "inspect_order" and obs.get("ok", True):
            try:
                o = _parse_customer_order(obs)
                tracker.orders[o.order_id] = o
            except Exception:
                pass
        elif step.tool_name == "inspect_route" and obs.get("ok", True):
            try:
                r = _parse_transport_route(obs)
                tracker.routes[r.route_id] = r
            except Exception:
                pass
        if step.tool_name in ("reserve_bay_and_order", "revise_order") and obs.get("ok", True):
            oid = str(step.arguments.get("order_id", ""))
            b_raw = obs.get("booking") if isinstance(obs.get("booking"), dict) else {}
            booked_tracker[oid] = {
                "fittings": str(step.arguments.get("fittings", "")),
                "unit_count": int(step.arguments.get("unit_count", 1)),
                "order_release_week": int(b_raw.get("order_release_week", 2 if tracker.disruption_triggered else 1)),
            }
        prev_step = step

    v = traj.verification
    messages.append(
        {
            "role": "assistant",
            "content": (
                f"Batch commissioning finalized on seed {traj.seed}: "
                f"{v.valid_order_count}/{v.total_order_count} valid orders, "
                f"sim_calls_used={v.sim_calls_used}/{v.sim_budget_total}, "
                f"spent={v.total_spent_eur:.2f}/{v.total_budget_eur:.2f} EUR."
            ),
        }
    )
    return messages


# ---------------------------------------------------------------------------
# Automatic Empirical Experience Distillation (`experience_playbook.json`)
# ---------------------------------------------------------------------------


def build_experience_playbook(
    chosen_logs: list[TrajectoryLog],
    rejected_logs: list[TrajectoryLog],
) -> dict[str, Any]:
    """Automatically distill `S01`–`S05` rollouts & contrast failure deltas into an empirical policy."""
    for log in chosen_logs + rejected_logs:
        assert_train_seed_only(log.seed)

    # 1. Empirical step/sim statistics across S01-S05 winners
    expert_scores = [t.verification.final_score for t in chosen_logs]
    expert_steps = [t.verification.step_count for t in chosen_logs]
    expert_sims = [t.verification.sim_calls_used for t in chosen_logs]
    rejected_scores = [t.verification.final_score for t in rejected_logs]

    # 2. Empirical tool-transition mining from S01-S05 winning rollouts
    transition_counts: Counter[str] = Counter()
    post_shock_next_tools: Counter[str] = Counter()
    for traj in chosen_logs:
        for i in range(len(traj.steps) - 1):
            cur_t = traj.steps[i].tool_name
            nxt_t = traj.steps[i + 1].tool_name
            transition_counts[f"{cur_t} -> {nxt_t}"] += 1
            if traj.steps[i].disruption_triggered_this_step:
                post_shock_next_tools[nxt_t] += 1

    # 3. Contrastive failure-mode reward-delta analysis on S01-S05
    chosen_by_base_seed: dict[str, float] = {}
    for t in chosen_logs:
        bseed = t.seed.split("_")[0]
        chosen_by_base_seed[bseed] = max(chosen_by_base_seed.get(bseed, 0.0), t.verification.final_score)

    failure_deltas: dict[str, list[float]] = defaultdict(list)
    for r in rejected_logs:
        bseed = r.seed.split("_")[0]
        ref_score = chosen_by_base_seed.get(bseed, 0.96)
        loss = round(max(0.0, ref_score - r.verification.final_score), 4)
        cats = list(r.verification.primary_failure_categories)
        if r.verification.batch_gate_cap_applied:
            cats.append("BATCH_COMPLETION_GATE_CAP")
        if r.verification.sim_calls_used >= SIM_BUDGET_DEFAULT:
            cats.append("SIM_BUDGET_EXHAUSTED")
        if r.verification.permit_quota_exceeded:
            cats.append("PERMIT_QUOTA_EXCEEDED")
        if r.verification.budget_exceeded:
            cats.append("BUDGET_EXCEEDED")
        if not r.verification.finalized:
            cats.append("UNFINALIZED_ZERO_SCORE")
        for c in set(cats):
            failure_deltas[c].append(loss)

    empirical_failure_lessons: list[dict[str, Any]] = []
    mitigation_lookup = {
        "BOIL_OFF": "For tight boil-off limits or after PVI multiplier shocks, compute Q_cond = k_eff * A_inner * (293.15 - T_boil) / (annulus_mm/1000) and Q_total = Q_cond + valve_leak_w * (1.25 if TRANSPORTABLE else 1.0); switch insulation to EVI (k=0.0022 vs PVI 0.024 W/mK) or upgrade fittings (EU_STD=11W, DE_PRECISION=5.5W vs CN=18W) so boil_off_pct_day <= max_boil_off_pct_day.",
        "BOIL_OFF_EXCEEDED": "Switch insulation to EVI or upgrade fittings to reduce conductive and valve heat leak below max_boil_off_pct_day.",
        "HOOP_STRESS": "Compute exact minimum wall t_min = P * inner_d_mm / (2 * weld_eff * sigma_allow - P) where sigma_allow = round(235.0 / safety_factor, 2) and weld_eff = {CN:0.85, EU_STD:0.90, DE_PRECISION:0.96}, then add +1.0 mm safety margin (keep < 14.0 mm when valid to avoid +1 fab_week). Re-check after order_min_pressure_overrides.",
        "HOOP_STRESS_EXCEEDED": "Increase wall_thickness_mm or weld_eff so P*(inner_d_mm + t)/(2*weld_eff*t) <= allowable_stress.",
        "CAPACITY_SHORTFALL": "Never confuse m3 with kg! Deliverable mass is unit_count * round(unit_volume_m3 * density_kg_m3 * 0.95, 2) >= required_product_mass_kg. Low-density cryogens LH2 (70.85 kg/m3) and LHe (124.9 kg/m3) require large volumes (e.g. 15-30 m3 or 2-3 units) to deliver 1,000+ kg.",
        "HAZARD_COMPATIBILITY": "LH2 and LHe strictly REQUIRE EVI insulation and FORBID CN fittings; LOX with design_pressure_mpa > 2.0 MPa strictly FORBIDS CN fittings (use EU_STD or DE_PRECISION).",
        "SLA_MISSED": "Verify delivery_week = max(bay_start_week + fab_weeks, order_release_week + effective_fittings_lead_weeks) + 1 + route.transit_weeks <= min(sla_week, horizon_weeks). Post-shock new bookings or revisions changing fittings/unit_count set order_release_week = 2!",
        "PERMIT_QUOTA": "Count total orders with use_escort_permit=True across all 4 orders and ensure total <= post-shock effective_escort_permits_total; exceeding quota fails ALL permit orders.",
        "PERMIT_QUOTA_EXCEEDED": "Count total orders with use_escort_permit=True across all 4 orders and ensure total <= post-shock effective_escort_permits_total; exceeding quota fails ALL permit orders.",
        "TRANSPORT_CLEARANCE": "Any shell with shipping_width_mm > route.std_max_width_mm (e.g. outer_d >= 2400 on TRANSPORTABLE -> 2600mm > 2500mm, or outer_d >= 2600 on STATIONARY -> 2600mm > 2500mm) MUST either set use_escort_permit=True (within permit quota) OR split into unit_count=2 or 3 smaller standard volumes (<=8 m3 TRANSPORTABLE, <=10 m3 STATIONARY) that fit standard gauge without a permit.",
        "BRIDGE_MASS_CAP": "Keep unit_transport_mass_kg <= post-shock route max_unit_transport_mass_kg; splitting into unit_count=2 or 3 smaller shells cuts per-unit tare and transport mass in half.",
        "BAY_CONFLICT": "Ensure [bay_start_week, bay_start_week + fab_weeks) on BAY-A/BAY-B never overlaps another order or post-shock bay_blackouts, and unit_tare_mass_kg <= bay_max_unit_tare_kg.",
        "SHOCK_UNADAPTED": "Immediately after disruption_alert triggers, call get_factory_state() + inspect_order/inspect_route on affected IDs and call revise_order() + assign_transport() to restore post-shock compliance.",
        "SIM_BUDGET_EXHAUSTED": f"Total run_physics_sim quota is strictly {SIM_BUDGET_DEFAULT} calls per episode! Pre-compute specs analytically from query_handbook('all'), use <= 1 sim call pre-shock on the tightest boil-off order, and save >= 2 sim calls for post-shock verification.",
        "BATCH_COMPLETION_GATE_CAP": "All 4 orders must pass hard checks to unlock full score + 0.15 efficiency bonuses (1 valid order is capped at <= 0.0800; 2 orders x0.80; 3 orders x0.92).",
        "BUDGET_OVERRUN": "Keep sum(manufacturing_cost + unit_count*base_freight + permit_fee) + retooling_fees <= effective_budget_eur (breaching budget multiplies score by 0.20 and caps at 0.15).",
        "BUDGET_EXCEEDED": "Keep sum(manufacturing_cost + unit_count*base_freight + permit_fee) + retooling_fees <= effective_budget_eur (breaching budget multiplies score by 0.20 and caps at 0.15). First post-shock revision per order costs 1,500 EUR; extra revisions cost 4,500 EUR.",
        "SITE_CLEARANCE": "Check STATIONARY installed_height = shell_axis_mm + 900 <= max_site_height_mm AND bank footprint = unit_count * installed_width_mm + (unit_count - 1) * 400 <= max_site_footprint_mm (note: 2x2000mm bank = 4400mm; if max_site_footprint_mm < 4400mm, use 1 unit or <=5m3 1600mm shells where 2x1600+400 = 3600mm).",
        "INCOMPLETE_BATCH": "Every one of ORD-1..ORD-4 must be booked on BAY-A/BAY-B AND assigned transport on its exact order.route_id before finalize_commissioning().",
        "UNFINALIZED": "Always call finalize_commissioning() before step 40; unfinalized episodes receive 0.0000.",
        "UNFINALIZED_ZERO_SCORE": "Always call finalize_commissioning() before step 40; unfinalized episodes receive 0.0000.",
    }
    for cat, losses in sorted(failure_deltas.items(), key=lambda kv: (-sum(kv[1]) / len(kv[1]), -len(kv[1]))):
        empirical_failure_lessons.append(
            {
                "failure_category": cat,
                "occurrences_in_s01_s05_contrasts": len(losses),
                "mean_reward_loss": round(sum(losses) / len(losses), 4),
                "mitigation_rule": mitigation_lookup.get(
                    cat,
                    "Re-verify post-shock order, route, bay blackout, crane tare, and physics constraints before finalize_commissioning().",
                ),
            }
        )

    # 4. Compact few-shot episode action traces distilled from S01 and S02
    demo_traces: list[dict[str, Any]] = []
    seen_demo_seeds: set[str] = set()
    for traj in chosen_logs:
        if traj.seed in ("S01", "S02") and traj.seed not in seen_demo_seeds:
            seen_demo_seeds.add(traj.seed)
            condensed_steps: list[dict[str, Any]] = []
            for st in traj.steps:
                if st.tool_name in (
                    "get_episode_brief",
                    "query_handbook",
                    "run_physics_sim",
                    "reserve_bay_and_order",
                    "get_factory_state",
                    "revise_order",
                    "assign_transport",
                    "finalize_commissioning",
                ):
                    condensed_steps.append(
                        {
                            "step": st.step_index,
                            "tool": st.tool_name,
                            "args": st.arguments,
                            "disruption_triggered": st.disruption_triggered_this_step,
                        }
                    )
            demo_traces.append(
                {
                    "train_seed": traj.seed,
                    "final_score": traj.verification.final_score,
                    "valid_orders": f"{traj.verification.valid_order_count}/{traj.verification.total_order_count}",
                    "sim_calls_used": traj.verification.sim_calls_used,
                    "step_count": traj.verification.step_count,
                    "key_tool_sequence": condensed_steps,
                }
            )

    mean_exp = round(sum(expert_scores) / max(1, len(expert_scores)), 4)
    mean_rej = round(sum(rejected_scores) / max(1, len(rejected_scores)), 4)

    # 5. Render compact, zero-leakage Experience-Conditioned Policy prompt for small models
    lessons_bullets = "\n".join(
        f"- {item['failure_category']} (empirical S01-S05 reward loss -{item['mean_reward_loss']:.3f}): {item['mitigation_rule']}"
        for item in empirical_failure_lessons[:10]
    )

    compiled_policy_prompt = f"""### EXPERIENCE-DISTILLED COMMISSIONING POLICY (LEARNED FROM S01-S05 ENVIRONMENT ROLLOUTS)
Empirical S01-S05 benchmark stats: winning trajectories achieved mean score {mean_exp:.4f} in {sum(expert_steps)/max(1,len(expert_steps)):.1f} steps using {sum(expert_sims)/max(1,len(expert_sims)):.1f}/{SIM_BUDGET_DEFAULT} FEA sims (vs {mean_rej:.4f} on unadapted/sim-exhausted rollouts).

#### 1. Mandatory 4-Phase Execution Protocol (Max 42 Steps, Max {SIM_BUDGET_DEFAULT} Sims)
- **Phase A — Compact Discovery First (Steps 1–5, 0 sims):**
  1. Call `get_episode_brief()` (which already lists all 4 orders and routes!), `query_handbook(section="all")`, and `get_factory_state()`.
- **Phase B — Closed-Form Analytical Sizing + First 2 Bookings (Conserve Sims: Use <= 1 Sim Pre-Shock!):**
  1. Solve every order in closed form BEFORE booking or calling `run_physics_sim`:
     - Standard volumes `V` in `[1, 2, 3, 5, 8, 10, 15, 20, 25, 30]` m3, `unit_count` `N` in `[1, 2, 3]`.
     - **Deliverable mass (DO NOT confuse m3 with kg!):** `usable_unit_mass_kg = round(V * density * 0.95, 2)` and `N * usable_unit_mass_kg >= required_product_mass_kg`, where `density` (`kg/m3`) = `LIN:808, LOX:1141, LAR:1395, LCO2:1030, LNG:430, LH2:70.85, LHe:124.9`. Note `LH2` & `LHe` have ultra-low density: e.g. `1x20m3 LHe` delivers `20*124.9*0.95 = 2373 kg` (whereas `1x1m3` is only `118.6 kg`!), and `1x20m3 LH2` delivers `20*70.85*0.95 = 1346 kg`!
     - **Outer & Inner Diameter:** Exact `outer_d_mm`: `1-2m3:1400, 3-5m3:1600, 8-10m3:2000, 15m3:2400, 20-25m3:2600, 30m3:3000`. Then `inner_d_mm = outer_d_mm - 2*annulus_mm` (`EVI: -240mm`, `PVI: -400mm`).
     - **Closed-Form Wall Thickness (NEVER under-size hoop stress!):**
       * Set `P = max(min_pressure_mpa, 0.4)`.
       * `sigma_allow = round(235.0 / safety_factor, 2)` (`LIN/LCO2: 235.0`, `LAR: 223.81`, `LNG: 204.35`, `LOX/LH2/LHe: 188.0` MPa).
       * `weld_eff`: `CN=0.85`, `EU_STD=0.90`, `DE_PRECISION=0.96`.
       * Exact formula: `t_min_mm = P * inner_d_mm / (2.0 * weld_eff * sigma_allow - P)`. Round UP to the next integer `mm` and add `+1.0 mm` margin so `round(P * (inner_d_mm + t) / (2 * weld_eff * t), 2) <= sigma_allow` is guaranteed! Keep `t < 14.0 mm` when valid (since `t >= 14.0 mm` adds `+1` `fab_weeks`).
     - **Material Hazard Rules:** `LH2` & `LHe` strictly REQUIRE `EVI` and FORBID `CN`. `LOX` with `design_pressure_mpa > 2.0` FORBIDS `CN`.
     - **Boil-Off & Budget Discipline (DO NOT overspend on `DE_PRECISION`!):**
       * `EU_STD` (`4w` lead, `1.22x` cost, `11W` leak) + `EVI` (`k=0.0022`) is the workhorse for cryogenic orders; use `PVI` + `CN` (`2w` lead, `1.00x` cost) only on warm/lenient orders (`LCO2` or `max_boil_off >= 0.40%/day`) when PVI boil-off & height fit.
       * Percentage boil-off `%/day` drops sharply with larger `unit_volume_m3`: if `1x10m3 EVI+EU_STD` slightly exceeds `max_boil_off_pct_day` (especially on `LHe` or `LH2`), upsize to `1x15m3` or `1x20m3` `EVI+EU_STD` FIRST! Reserve expensive/slow `DE_PRECISION` (`7w` lead, `1.68x` cost, `5.5W` leak) ONLY when `EVI+EU_STD` still misses boil-off AND `sla_week >= 10` AND total batch spend stays `<= effective_budget_eur`!
     - **Exact Catalogue Heights & Route Envelope (Avoid `SITE_HEIGHT`, `SITE_FOOTPRINT`, `STD_GAUGE`!):**
       * Exact `STATIONARY` `installed_height_mm` (`PVI` is 700-900mm taller than `EVI`!): `5m3: EVI=4350/PVI=5330`, `8m3: EVI=4190/PVI=4880`, `10m3: EVI=5020/PVI=5880`, `15m3: EVI=5000/PVI=5680`, `20m3: EVI=5480/PVI=6170`, `25m3: EVI=6620/PVI=7480`. Always check `installed_height_mm <= max_site_height_mm`!
       * Multi-unit bank width: `N * installed_width + (N - 1) * 400 <= max_site_footprint_mm` (`2x10m3 STATIONARY` = `4400mm`, so if `max_site_footprint_mm < 4400`, use `1x20m3 EVI` (`width=2600mm, height=5480mm`) instead of `2x10m3`!).
       * Shipping width: `STATIONARY` = `outer_d_mm` (`20-25m3` -> `2600mm`); `TRANSPORTABLE` = `outer_d_mm + 200` (`15m3` -> `2600mm`, `10m3` -> `2200mm`). Any order with `shipping_width > route.std_max_width_mm` (typically `2500mm`) or `shipping_height > route.std_max_height_mm` MUST either set `use_escort_permit=True` (within batch permit quota!) OR use `<=10m3` (`outer_d <= 2000`).
  2. Spend **at most 1 `run_physics_sim` call** pre-shock (on your tightest `LHe`/`LH2`/`LOX` boil-off order), keeping **2 `run_physics_sim` calls in reserve** for post-shock verification! Book the first 2 orders via `reserve_bay_and_order`.
- **Phase C — Mid-Episode Supply/Site Disruption Audit & Adaptation (CRITICAL):**
  - Booking the 2nd order (or 4th mutating step) triggers `disruption_alert`!
  - IMMEDIATELY call `get_factory_state()`, plus `inspect_order` and `inspect_route` on EVERY overridden order/route ID.
  - Audit BOTH already-booked orders AND the 2 unbooked orders against ALL post-shock changes:
    1. `pvi_k_multiplier > 1.0`: any tight `PVI` order must be revised (`revise_order`) to `EVI`!
    2. `order_min_pressure_overrides` / `order_sla_overrides`: raise `design_pressure_mpa`, recompute `t_min_mm` for hoop stress, and verify SLA!
    3. `bay_blackouts` / `bay_max_unit_tare_kg`: ensure `[bay_start_week, bay_start_week + fab_weeks)` has ZERO overlap with blackouts and `unit_tare_mass_kg <= crane_cap`.
    4. `fittings_lead_deltas`: if `EU_STD` (`4w`) or `DE_PRECISION` (`7w`) receives a `+2w` or `+3w` customs delay so `order_release_week + effective_fittings_lead_weeks + 1 + transit_weeks > sla_week`, IMMEDIATELY revise that order to faster permitted fittings (`CN` `2w` lead for `LIN/LAR/LCO2/LNG` or `LOX <= 2.0 MPa`; `EU_STD` `4w` lead if `DE_PRECISION` was delayed)! Note: post-shock new bookings or revisions changing `fittings`/`unit_count` set `order_release_week = 2`.
    5. `route_*_overrides` / `escort_permits_delta` / `budget_delta_eur`: check `unit_transport_mass_kg <= max_unit_transport_mass_kg`, shipping WxH vs std/permit limits, total permits across all 4 orders `<= effective_escort_permits_total`, and total batch spend (mfg + freight + permits + revision fees) `<= effective_budget_eur`!
  - Use your 2 reserved `run_physics_sim` calls to confirm your tightest post-shock / low-temperature (`LH2`/`LHe`/`LOX`) specs pass `all_engineering_pass == true`, call `revise_order(...)` once per invalidated pre-shock order, and book the remaining 2 orders via `reserve_bay_and_order`.
- **Phase D — Complete Transport Assignment & Mandatory Commissioning:**
  - Call `assign_transport(order_id, route_id=order.route_id, use_escort_permit=...)` on ALL 4 orders (`ORD-1`..`ORD-4`), double-checking whether each order's `shipping_width_mm` or `shipping_height_mm` requires `use_escort_permit=True` and that total permits used `<= effective_escort_permits_total`.
  - Confirm `total_spent_eur <= effective_budget_eur` in `get_factory_state()`, then ALWAYS call `finalize_commissioning()` before step 40!

#### 2. Empirical Failure-Mode Priorities from S01-S05 Contrast Rollouts
{lessons_bullets}
"""

    playbook = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_seeds_included": sorted({t.seed.split("_")[0] for t in chosen_logs}),
        "augmented_rollout_count": sum(1 for t in chosen_logs if "_" in t.seed),
        "expert_trajectory_count": len(chosen_logs),
        "contrast_rejected_count": len(rejected_logs),
        "zero_leakage_verified": True,
        "empirical_summary": {
            "expert_mean_score_s01_s05": mean_exp,
            "rejected_mean_score_s01_s05": mean_rej,
            "mean_contrast_reward_gain": round(mean_exp - mean_rej, 4),
            "expert_mean_steps": round(sum(expert_steps) / max(1, len(expert_steps)), 2),
            "expert_mean_sim_calls": round(sum(expert_sims) / max(1, len(expert_sims)), 2),
            "top_tool_transitions": transition_counts.most_common(12),
            "post_shock_immediate_actions": post_shock_next_tools.most_common(5),
        },
        "failure_mode_reward_deltas": empirical_failure_lessons,
        "few_shot_s01_s05_traces": demo_traces,
        "compiled_policy_prompt": compiled_policy_prompt,
    }
    return playbook


def verify_zero_holdout_leakage_in_text(payload_text: str, label: str) -> None:
    """Assert that zero held-out seed identifiers (`S06`, `S07`, `S08`) exist in exported artifacts."""
    for holdout_seed in sorted(FORBIDDEN_HOLDOUT_SEEDS):
        if holdout_seed in payload_text:
            raise AssertionError(
                f"CRITICAL ZERO-LEAKAGE VIOLATION: Holdout seed identifier {holdout_seed!r} "
                f"detected inside {label}!"
            )


# ---------------------------------------------------------------------------
# Main Trajectory Collection & Export Pipeline
# ---------------------------------------------------------------------------


def collect_and_export(
    *,
    trajectories_dir: Path = TRAJECTORIES_DIR,
    output_dir: Path = TIEBREAKER_RUNS_DIR,
    augment: bool = False,
) -> dict[str, Any]:
    """Collect `S01`–`S05` trajectories and export SFT JSONL, DPO JSONL, and Experience Playbook."""
    output_dir.mkdir(parents=True, exist_ok=True)
    task_instruction = load_task_instruction()

    ingested_chosen: list[TrajectoryLog] = []
    ingested_rejected: list[TrajectoryLog] = []
    quarantined_holdout_files: list[str] = []

    if trajectories_dir.exists():
        for jpath in sorted(trajectories_dir.glob("*.json")):
            try:
                raw = json.loads(jpath.read_text(encoding="utf-8"))
                tlog = TrajectoryLog.model_validate(raw)
            except Exception:
                continue
            base_seed = tlog.seed.split("_")[0]
            if base_seed in FORBIDDEN_HOLDOUT_SEEDS or base_seed not in ALLOWED_TRAIN_SEEDS:
                quarantined_holdout_files.append(jpath.name)
                continue
            assert_train_seed_only(tlog.seed)
            if tlog.verification.final_score >= 0.94 and tlog.verification.valid_order_count == 4:
                ingested_chosen.append(tlog)
            elif tlog.verification.final_score < 0.70:
                ingested_rejected.append(tlog)

    # Ensure complete, balanced coverage across all 5 training seeds S01-S05 via live CryoForgeEngine rollouts
    chosen_by_seed: dict[str, list[TrajectoryLog]] = defaultdict(list)
    for t in ingested_chosen:
        chosen_by_seed[t.seed.split("_")[0]].append(t)

    rejected_by_seed: dict[str, list[tuple[str, TrajectoryLog]]] = defaultdict(list)
    for t in ingested_rejected:
        rejected_by_seed[t.seed.split("_")[0]].append((t.agent_type, t))

    for seed in EVAL_SEEDS:
        assert_train_seed_only(seed)
        # Always generate a clean, sim-disciplined expert rollout + 4-sim small-model demonstration on S01-S05
        exp_2sim = rollout_expert_episode(seed, sim_mode="targeted_2sim", run_tag="expert_2sim")
        if exp_2sim.verification.final_score < 0.94:
            raise AssertionError(
                f"Expert rollout on training seed {seed} scored {exp_2sim.verification.final_score} < 0.94"
            )
        chosen_by_seed[seed].append(exp_2sim)

        exp_4sim = rollout_expert_episode(seed, sim_mode="sim_4orders", prefer_bay_b_first=True, run_tag="expert_4sim")
        if exp_4sim.verification.final_score >= 0.94:
            chosen_by_seed[seed].append(exp_4sim)

        # Generate contrastive failure rollouts on S01-S05 for DPO/GRPO pairs & failure distillation
        for fmode in ("shock_unadapted", "sim_exhausted", "stop_early"):
            f_traj = rollout_failure_episode(seed, fmode)
            rejected_by_seed[seed].append((fmode, f_traj))

    if augment:
        for seed in EVAL_SEEDS:
            rec_traj = rollout_expert_episode(seed, sim_mode="sim_recovery", run_tag="aug_sim_recovery")
            if rec_traj.verification.final_score >= 0.94:
                chosen_by_seed[seed].append(rec_traj)
        for pert_traj in _generate_perturbed_s01_s05_rollouts():
            chosen_by_seed[pert_traj.seed.split("_")[0]].append(pert_traj)

    all_chosen: list[TrajectoryLog] = []
    for seed in EVAL_SEEDS:
        # Deduplicate identical run_ids while preserving diversity
        seen_runs: set[str] = set()
        for t in chosen_by_seed[seed]:
            if t.run_id not in seen_runs:
                seen_runs.add(t.run_id)
                all_chosen.append(t)

    all_rejected: list[TrajectoryLog] = [t for s in EVAL_SEEDS for _, t in rejected_by_seed[s]]

    # 1. Write `sft_train_messages.jsonl`
    sft_path = output_dir / "sft_train_messages.jsonl"
    sft_lines: list[str] = []
    for idx, traj in enumerate(all_chosen, start=1):
        assert_train_seed_only(traj.seed)
        msgs = trajectory_to_messages(traj, task_instruction=task_instruction, include_cot=True)
        record = {
            "id": f"sft_{traj.seed.lower()}_{idx:03d}",
            "seed": traj.seed,
            "base_seed": traj.seed.split("_")[0],
            "source_run_id": traj.run_id,
            "final_score": traj.verification.final_score,
            "valid_order_count": traj.verification.valid_order_count,
            "sim_calls_used": traj.verification.sim_calls_used,
            "step_count": traj.verification.step_count,
            "tools": CRYOFORGE_OPENAI_TOOLS,
            "messages": msgs,
        }
        line = json.dumps(record, ensure_ascii=False)
        verify_zero_holdout_leakage_in_text(line, f"sft_train_messages.jsonl:{record['id']}")
        sft_lines.append(line)
    sft_path.write_text("\n".join(sft_lines) + "\n", encoding="utf-8")

    # 2. Write `preference_pairs.jsonl`
    pref_path = output_dir / "preference_pairs.jsonl"
    pref_lines: list[str] = []
    pair_idx = 0
    for seed in EVAL_SEEDS:
        assert_train_seed_only(seed)
        best_chosen = max(chosen_by_seed[seed], key=lambda x: x.verification.final_score)
        chosen_msgs = trajectory_to_messages(best_chosen, task_instruction=task_instruction, include_cot=True)
        for fmode, rej_traj in rejected_by_seed[seed]:
            assert_train_seed_only(rej_traj.seed)
            if best_chosen.verification.final_score < 0.94 or rej_traj.verification.final_score >= best_chosen.verification.final_score:
                continue
            pair_idx += 1
            rej_msgs = trajectory_to_messages(rej_traj, task_instruction=task_instruction, include_cot=False)
            pair_record = {
                "pair_id": f"dpo_{seed.lower()}_{pair_idx:03d}",
                "seed": seed,
                "failure_archetype": fmode,
                "chosen_reward": best_chosen.verification.final_score,
                "rejected_reward": rej_traj.verification.final_score,
                "reward_margin": round(
                    best_chosen.verification.final_score - rej_traj.verification.final_score, 4
                ),
                "prompt": chosen_msgs[:2],
                "tools": CRYOFORGE_OPENAI_TOOLS,
                "chosen": chosen_msgs[2:],
                "rejected": rej_msgs[2:],
                "chosen_verification": best_chosen.verification.model_dump(mode="json"),
                "rejected_verification": rej_traj.verification.model_dump(mode="json"),
            }
            pline = json.dumps(pair_record, ensure_ascii=False)
            verify_zero_holdout_leakage_in_text(pline, f"preference_pairs.jsonl:{pair_record['pair_id']}")
            pref_lines.append(pline)
    pref_path.write_text("\n".join(pref_lines) + "\n", encoding="utf-8")

    # 3. Write `experience_playbook.json`
    playbook = build_experience_playbook(all_chosen, all_rejected)
    playbook["quarantined_holdout_trajectory_files"] = len(quarantined_holdout_files)
    playbook["sft_dataset_sha256"] = hashlib.sha256(sft_path.read_bytes()).hexdigest()
    playbook_json = json.dumps(playbook, indent=2, ensure_ascii=False)
    verify_zero_holdout_leakage_in_text(playbook_json, "experience_playbook.json")
    playbook_path = output_dir / "experience_playbook.json"
    playbook_path.write_text(playbook_json + "\n", encoding="utf-8")

    return {
        "train_seeds_used": list(EVAL_SEEDS),
        "quarantined_holdout_files_skipped": len(quarantined_holdout_files),
        "sft_examples_count": len(sft_lines),
        "preference_pairs_count": len(pref_lines),
        "sft_train_messages_path": str(sft_path),
        "preference_pairs_path": str(pref_path),
        "experience_playbook_path": str(playbook_path),
        "expert_mean_score": playbook["empirical_summary"]["expert_mean_score_s01_s05"],
        "rejected_mean_score": playbook["empirical_summary"]["rejected_mean_score_s01_s05"],
        "zero_leakage_verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect S01-S05 expert & contrast trajectories and distill zero-leakage tiebreaker datasets."
    )
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=TRAJECTORIES_DIR,
        help="Directory containing existing TrajectoryLog JSON files (holdout seeds S06-S08 are strictly quarantined).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TIEBREAKER_RUNS_DIR,
        help="Output directory for sft_train_messages.jsonl, preference_pairs.jsonl, and experience_playbook.json.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Include sim-recovery rollouts and parameterized perturbations derived strictly from S01-S05.",
    )
    args = parser.parse_args(argv)

    summary = collect_and_export(
        trajectories_dir=args.trajectories_dir,
        output_dir=args.output_dir,
        augment=args.augment,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
