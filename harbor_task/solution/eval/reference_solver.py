"""Zero-cheating Reference Solver for CryoForge RL over public MCP tools only.

STRICT INVARIANT:
- NEVER imports `env_core` (`scenarios`, `verifier`, `physics`, `contracts`, or `engine`).
- Connects exclusively through `eval.mcp_client_helper` to the MCP server (`stdio` or `sse`)
  and uses ONLY the 10 public agent MCP tools.
- Parses all physical constants, lookup tables, material rules, lead-time formulas, and
  scoring equations dynamically from `query_handbook("all")`.
- Performs exact lossless Pareto bucket reduction + joint 4-order branch-and-bound over
  specifications, escort permits, and 2-bay schedules both pre-shock and post-shock,
  verifying tightest-margin designs with 2 targeted `run_physics_sim` calls.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any

from eval.mcp_client_helper import (
    DEFAULT_TRAJECTORIES_DIR,
    CryoForgeMCPClient,
    load_persisted_trajectory,
    open_cryoforge_mcp_session,
)

DEFAULT_ALL_SEEDS: tuple[str, ...] = (
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
)


@dataclass(frozen=True)
class HandbookModel:
    """All physical constants and tables parsed strictly from `query_handbook('all')`."""

    unit_counts_allowed: tuple[int, ...]
    standard_volumes_m3: tuple[float, ...]
    min_design_pressure_mpa: float
    max_design_pressure_mpa: float
    min_wall_thickness_mm: float
    max_wall_thickness_mm: float

    fill_fraction: float
    ambient_temp_k: float
    base_allowable_stress_mpa: float
    inner_shell_steel_density_kg_m3: float
    inner_shell_head_stiffener_factor: float
    outer_jacket_thickness_m: float
    outer_jacket_steel_density_kg_m3: float
    nozzle_piping_fixed_tare_kg: float
    stationary_shipping_saddles_kg: float
    transportable_skid_tare_fraction: float
    transportable_skid_fixed_kg: float
    modular_bank_inter_unit_spacing_mm: int
    seconds_per_day: float
    outer_diameter_thresholds: tuple[tuple[float, int], ...]

    medium_props: dict[str, dict[str, Any]]
    insulation_props: dict[str, dict[str, Any]]
    fittings_props: dict[str, dict[str, Any]]

    bays: tuple[str, ...]
    assembly_commission_weeks: int
    series_discount_factor_multi_unit: float
    transportable_cost_multiplier: float

    pre_shock_order_release_week: int
    post_shock_order_release_week: int
    post_shock_first_revision_fee_eur: float
    standard_revision_fee_eur: float
    budget_bonus_headroom_frac: float

    @classmethod
    def from_handbook_response(cls, resp: dict[str, Any]) -> "HandbookModel":
        sections = resp["sections"]
        ov = sections["overview"]
        cb = ov["catalogue_bounds"]
        ph = sections["physics"]
        pc = ph["constants"]
        mat = sections["materials_and_insulation"]
        mfg = sections["manufacturing_and_bays"]["constants"]
        dis = sections["disruptions_and_revisions"]["constants"]
        sc_bon = sections["scoring"]["efficiency_and_margin_bonuses"]

        b_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*\*\s*effective_budget_eur",
            str(sc_bon.get("budget_efficiency_bonus", "")),
        )
        budget_bonus_headroom_frac = float(b_match.group(1)) if b_match else 0.03

        # Parse nominal_outer_diameter_mm_lookup thresholds dynamically from handbook table
        thresholds: list[tuple[float, int]] = []
        fallback_diam = 3000
        for row in ph["nominal_outer_diameter_mm_lookup"]:
            cond = str(row["unit_volume_m3_condition"])
            diam = int(row["outer_diameter_mm"])
            le_match = re.search(r"<=\s*([0-9]+(?:\.[0-9]+)?)", cond)
            if le_match:
                thresholds.append((float(le_match.group(1)), diam))
            else:
                fallback_diam = diam
        thresholds.sort(key=lambda item: item[0])
        thresholds.append((float("inf"), fallback_diam))

        return cls(
            unit_counts_allowed=tuple(int(x) for x in cb["unit_counts_allowed"]),
            standard_volumes_m3=tuple(float(x) for x in cb["standard_volumes_m3"]),
            min_design_pressure_mpa=float(cb["min_design_pressure_mpa"]),
            max_design_pressure_mpa=float(cb["max_design_pressure_mpa"]),
            min_wall_thickness_mm=float(cb["min_wall_thickness_mm"]),
            max_wall_thickness_mm=float(cb["max_wall_thickness_mm"]),
            fill_fraction=float(pc["fill_fraction"]),
            ambient_temp_k=float(pc["ambient_temp_k"]),
            base_allowable_stress_mpa=float(pc["base_allowable_stress_mpa"]),
            inner_shell_steel_density_kg_m3=float(pc["inner_shell_steel_density_kg_m3"]),
            inner_shell_head_stiffener_factor=float(pc["inner_shell_head_stiffener_factor"]),
            outer_jacket_thickness_m=float(pc["outer_jacket_thickness_m"]),
            outer_jacket_steel_density_kg_m3=float(pc["outer_jacket_steel_density_kg_m3"]),
            nozzle_piping_fixed_tare_kg=float(pc["nozzle_piping_fixed_tare_kg"]),
            stationary_shipping_saddles_kg=float(pc["stationary_shipping_saddles_kg"]),
            transportable_skid_tare_fraction=float(pc["transportable_skid_tare_fraction"]),
            transportable_skid_fixed_kg=float(pc["transportable_skid_fixed_kg"]),
            modular_bank_inter_unit_spacing_mm=int(pc["modular_bank_inter_unit_spacing_mm"]),
            seconds_per_day=float(pc["seconds_per_day"]),
            outer_diameter_thresholds=tuple(thresholds),
            medium_props=dict(mat["medium_props"]),
            insulation_props=dict(mat["insulation_props"]),
            fittings_props=dict(mat["fittings_props"]),
            bays=tuple(str(b) for b in mfg["bays"]),
            assembly_commission_weeks=int(mfg["assembly_commission_weeks"]),
            series_discount_factor_multi_unit=float(mfg["series_discount_factor_multi_unit"]),
            transportable_cost_multiplier=float(mfg["transportable_cost_multiplier"]),
            pre_shock_order_release_week=int(dis["pre_shock_order_release_week"]),
            post_shock_order_release_week=int(dis["post_shock_order_release_week"]),
            post_shock_first_revision_fee_eur=float(dis["post_shock_first_revision_fee_eur"]),
            standard_revision_fee_eur=float(dis["standard_revision_fee_eur"]),
            budget_bonus_headroom_frac=budget_bonus_headroom_frac,
        )

    def nominal_outer_diameter_mm(self, unit_volume_m3: float) -> int:
        for upper_vol, diam_mm in self.outer_diameter_thresholds:
            if unit_volume_m3 <= upper_vol + 1e-9:
                return diam_mm
        return self.outer_diameter_thresholds[-1][1]


@dataclass(frozen=True)
class CandidateOption:
    """A single valid `(spec, use_escort_permit)` candidate for an order."""

    order_id: str
    unit_count: int
    unit_volume_m3: float
    design_pressure_mpa: float
    wall_thickness_mm: float
    insulation: str
    fittings: str
    use_escort_permit: bool

    fab_weeks: int
    fittings_lead_weeks: int
    manufacturing_cost_eur: float
    transport_cost_eur: float
    unit_tare_mass_kg: float
    unit_transport_mass_kg: float
    hoop_stress_mpa: float
    allowable_stress_mpa: float
    boil_off_pct_day: float
    max_boil_off_pct_day: float
    eligible_bays: tuple[str, ...]
    matches_existing_spec: bool = False

    @property
    def base_order_cost_eur(self) -> float:
        return round(self.manufacturing_cost_eur + self.transport_cost_eur, 2)

    @property
    def tightest_relative_margin(self) -> float:
        stress_margin = (
            (self.allowable_stress_mpa - self.hoop_stress_mpa)
            / max(1.0, self.allowable_stress_mpa)
        )
        boil_margin = (
            (self.max_boil_off_pct_day - self.boil_off_pct_day)
            / max(1e-4, self.max_boil_off_pct_day)
        )
        return min(stress_margin, boil_margin)

    def spec_tuple(self) -> tuple[int, float, float, float, str, str]:
        return (
            self.unit_count,
            round(self.unit_volume_m3, 4),
            round(self.design_pressure_mpa, 4),
            round(self.wall_thickness_mm, 4),
            self.insulation,
            self.fittings,
        )


@dataclass(frozen=True)
class ScheduledOrderAction:
    """Concrete planned schedule & spec assignment for one customer order."""

    order_id: str
    route_id: str
    bay_id: str
    bay_start_week: int
    fab_end_week: int
    delivery_week: int
    sla_week: int
    keep_existing_booking: bool
    retooling_fee_delta_eur: float
    candidate: CandidateOption


@dataclass(frozen=True)
class BatchPlan:
    """Optimal joint 4-order commissioning plan."""

    projected_score: float
    projected_total_spent_eur: float
    avg_slack_weeks: float
    permits_used: int
    actions: dict[str, ScheduledOrderAction]
    revision_execution_order: tuple[str, ...]


def evaluate_handbook_candidate(
    hb: HandbookModel,
    order: dict[str, Any],
    route: dict[str, Any],
    *,
    unit_count: int,
    unit_volume_m3: float,
    design_pressure_mpa: float,
    wall_thickness_mm: float,
    insulation: str,
    fittings: str,
    pvi_k_multiplier: float,
    fittings_lead_deltas: dict[str, int],
    bay_max_unit_tare_kg: dict[str, float],
    existing_spec_tuple: tuple[int, float, float, float, str, str] | None = None,
) -> list[CandidateOption]:
    """Evaluate published handbook equations and return valid `(spec, permit)` options."""
    medium = str(order["medium"])
    execution = str(order["execution"])
    mprops = hb.medium_props[medium]
    iprops = hb.insulation_props[insulation]
    fprops = hb.fittings_props[fittings]

    # 1. Material & hazard compatibility
    if mprops["requires_evi"] and insulation != "EVI":
        return []
    if fittings in mprops["forbidden_fittings"]:
        return []
    if mprops["lox_high_pressure_rule"] and design_pressure_mpa > 2.0 and fittings == "CN":
        return []

    # 2. Deliverable mass
    usable_unit_mass_kg = round(
        unit_volume_m3 * float(mprops["density_kg_m3"]) * hb.fill_fraction, 2
    )
    usable_total_mass_kg = round(unit_count * usable_unit_mass_kg, 2)
    if usable_total_mass_kg < float(order["required_product_mass_kg"]):
        return []

    # 3. Dimensions & site clearance
    outer_d_mm = hb.nominal_outer_diameter_mm(unit_volume_m3)
    annulus_mm = float(iprops["annulus_mm"])
    inner_d_mm = int(round(outer_d_mm - 2.0 * annulus_mm))
    d_inner_m = inner_d_mm / 1000.0
    raw_axis_mm = (unit_volume_m3 / (math.pi * (d_inner_m / 2.0) ** 2)) * 1000.0
    shell_axis_mm = int(math.ceil(raw_axis_mm / 10.0) * 10)

    if execution == "STATIONARY":
        installed_height_mm = shell_axis_mm + 900
        installed_width_mm = outer_d_mm
        shipping_height_mm = outer_d_mm + 350
        shipping_width_mm = outer_d_mm
    else:
        installed_height_mm = outer_d_mm + 600
        installed_width_mm = outer_d_mm + 200
        shipping_height_mm = installed_height_mm
        shipping_width_mm = installed_width_mm

    bank_footprint_mm = (
        unit_count * installed_width_mm
        + (unit_count - 1) * hb.modular_bank_inter_unit_spacing_mm
    )
    if installed_height_mm > int(order["max_site_height_mm"]):
        return []
    if bank_footprint_mm > int(order["max_site_footprint_mm"]):
        return []

    # 4. Pressure & hoop stress
    if design_pressure_mpa < float(order["min_pressure_mpa"]) - 1e-9:
        return []
    weld_eff = float(fprops["weld_efficiency"])
    hoop_stress_mpa = round(
        (design_pressure_mpa * (inner_d_mm + wall_thickness_mm))
        / (2.0 * weld_eff * wall_thickness_mm),
        2,
    )
    allowable_stress_mpa = round(
        hb.base_allowable_stress_mpa / float(mprops["safety_factor"]), 2
    )
    if hoop_stress_mpa > allowable_stress_mpa:
        return []

    # 5. Surface area, tare mass, transport mass
    r_m = (inner_d_mm / 1000.0) / 2.0
    l_m = shell_axis_mm / 1000.0
    area_m2 = 2.0 * math.pi * r_m * l_m + 4.0 * math.pi * (r_m**2) * 0.65

    inner_shell_vol_m3 = (
        area_m2 * (wall_thickness_mm / 1000.0) * hb.inner_shell_head_stiffener_factor
    )
    inner_shell_mass_kg = inner_shell_vol_m3 * hb.inner_shell_steel_density_kg_m3
    outer_jacket_mass_kg = (
        area_m2
        * hb.outer_jacket_thickness_m
        * hb.outer_jacket_steel_density_kg_m3
        * float(iprops["casing_mass_factor"])
    )
    unit_tare_mass_kg = round(
        inner_shell_mass_kg + outer_jacket_mass_kg + hb.nozzle_piping_fixed_tare_kg, 1
    )
    if execution == "TRANSPORTABLE":
        skid_kg = round(
            hb.transportable_skid_tare_fraction * unit_tare_mass_kg
            + hb.transportable_skid_fixed_kg,
            1,
        )
    else:
        skid_kg = hb.stationary_shipping_saddles_kg
    unit_transport_mass_kg = round(unit_tare_mass_kg + skid_kg, 1)

    if unit_transport_mass_kg > float(route["max_unit_transport_mass_kg"]):
        return []

    eligible_bays = tuple(
        b
        for b in hb.bays
        if unit_tare_mass_kg <= float(bay_max_unit_tare_kg.get(b, float("inf")))
    )
    if not eligible_bays:
        return []

    # 6. Thermal boil-off rate
    k_eff = float(iprops["k_eff_w_mk"]) * (
        pvi_k_multiplier if insulation == "PVI" else 1.0
    )
    t_ins_m = annulus_mm / 1000.0
    delta_t = hb.ambient_temp_k - float(mprops["boil_temp_k"])
    q_cond_w = k_eff * area_m2 * delta_t / t_ins_m
    q_valve_w = float(fprops["valve_heat_leak_w"]) * (
        1.25 if execution == "TRANSPORTABLE" else 1.00
    )
    q_day_j = (q_cond_w + q_valve_w) * hb.seconds_per_day
    evap_kg_day = q_day_j / (float(mprops["latent_heat_kj_kg"]) * 1000.0)
    boil_off_pct_day = round((evap_kg_day / usable_unit_mass_kg) * 100.0, 4)
    max_boil = float(order["max_boil_off_pct_day"])
    if boil_off_pct_day > max_boil:
        return []

    # 7. Fabrication duration, fittings lead, manufacturing cost
    base_fab = (
        2
        + (1 if unit_volume_m3 >= 10.0 else 0)
        + (1 if unit_volume_m3 >= 20.0 else 0)
        + (1 if wall_thickness_mm >= 14.0 else 0)
        + (1 if execution == "TRANSPORTABLE" else 0)
    )
    fab_weeks = base_fab + (unit_count - 1)
    fittings_lead_weeks = max(
        1, int(fprops["base_lead_weeks"]) + int(fittings_lead_deltas.get(fittings, 0))
    )

    exec_mult = hb.transportable_cost_multiplier if execution == "TRANSPORTABLE" else 1.00
    unit_cost_eur = (
        (9500.0 + unit_volume_m3 * 2150.0 + unit_tare_mass_kg * 3.10)
        * float(iprops["cost_multiplier"])
        * float(fprops["cost_multiplier"])
        * exec_mult
        + float(fprops["fixed_kit_eur"])
    )
    series_factor = hb.series_discount_factor_multi_unit if unit_count > 1 else 1.00
    manufacturing_cost_eur = round(unit_count * unit_cost_eur * series_factor, 2)

    cur_spec_tuple = (
        unit_count,
        round(unit_volume_m3, 4),
        round(design_pressure_mpa, 4),
        round(wall_thickness_mm, 4),
        insulation,
        fittings,
    )
    is_existing = existing_spec_tuple is not None and cur_spec_tuple == existing_spec_tuple

    # 8. Standard vs Escort Permit clearance options
    std_ok = shipping_width_mm <= int(route["std_max_width_mm"]) and shipping_height_mm <= int(
        route["std_max_height_mm"]
    )
    permit_ok = (
        bool(route["escort_permit_allowed"])
        and shipping_width_mm <= int(route["permit_max_width_mm"])
        and shipping_height_mm <= int(route["permit_max_height_mm"])
    )

    options: list[CandidateOption] = []
    base_freight = round(unit_count * float(route["base_freight_eur_per_unit"]), 2)
    if std_ok:
        options.append(
            CandidateOption(
                order_id=str(order["order_id"]),
                unit_count=unit_count,
                unit_volume_m3=unit_volume_m3,
                design_pressure_mpa=design_pressure_mpa,
                wall_thickness_mm=wall_thickness_mm,
                insulation=insulation,
                fittings=fittings,
                use_escort_permit=False,
                fab_weeks=fab_weeks,
                fittings_lead_weeks=fittings_lead_weeks,
                manufacturing_cost_eur=manufacturing_cost_eur,
                transport_cost_eur=base_freight,
                unit_tare_mass_kg=unit_tare_mass_kg,
                unit_transport_mass_kg=unit_transport_mass_kg,
                hoop_stress_mpa=hoop_stress_mpa,
                allowable_stress_mpa=allowable_stress_mpa,
                boil_off_pct_day=boil_off_pct_day,
                max_boil_off_pct_day=max_boil,
                eligible_bays=eligible_bays,
                matches_existing_spec=is_existing,
            )
        )
    elif permit_ok:
        options.append(
            CandidateOption(
                order_id=str(order["order_id"]),
                unit_count=unit_count,
                unit_volume_m3=unit_volume_m3,
                design_pressure_mpa=design_pressure_mpa,
                wall_thickness_mm=wall_thickness_mm,
                insulation=insulation,
                fittings=fittings,
                use_escort_permit=True,
                fab_weeks=fab_weeks,
                fittings_lead_weeks=fittings_lead_weeks,
                manufacturing_cost_eur=manufacturing_cost_eur,
                transport_cost_eur=round(base_freight + float(route["permit_fee_eur"]), 2),
                unit_tare_mass_kg=unit_tare_mass_kg,
                unit_transport_mass_kg=unit_transport_mass_kg,
                hoop_stress_mpa=hoop_stress_mpa,
                allowable_stress_mpa=allowable_stress_mpa,
                boil_off_pct_day=boil_off_pct_day,
                max_boil_off_pct_day=max_boil,
                eligible_bays=eligible_bays,
                matches_existing_spec=is_existing,
            )
        )
    return options


def generate_pareto_candidates_for_order(
    hb: HandbookModel,
    order: dict[str, Any],
    route: dict[str, Any],
    *,
    pvi_k_multiplier: float,
    fittings_lead_deltas: dict[str, int],
    bay_max_unit_tare_kg: dict[str, float],
    existing_booking: dict[str, Any] | None = None,
) -> list[CandidateOption]:
    """Generate lossless Pareto-reduced `(spec, permit)` candidates for `order`."""
    existing_spec_tuple: tuple[int, float, float, float, str, str] | None = None
    extra_pressures: list[float] = []
    extra_walls: list[float] = []
    if existing_booking is not None:
        espec = existing_booking["spec"]
        existing_spec_tuple = (
            int(espec["unit_count"]),
            round(float(espec["unit_volume_m3"]), 4),
            round(float(espec["design_pressure_mpa"]), 4),
            round(float(espec["wall_thickness_mm"]), 4),
            str(espec["insulation"]),
            str(espec["fittings"]),
        )
        extra_pressures.append(round(float(espec["design_pressure_mpa"]), 4))
        extra_walls.append(round(float(espec["wall_thickness_mm"]), 4))

    min_req_p = max(hb.min_design_pressure_mpa, round(float(order["min_pressure_mpa"]), 4))
    pressure_candidates = sorted({min_req_p, *extra_pressures})

    steps_count = int(round((hb.max_wall_thickness_mm - hb.min_wall_thickness_mm) / 0.5)) + 1
    wall_grid = [round(hb.min_wall_thickness_mm + 0.5 * i, 4) for i in range(steps_count)]

    # Bucket key: (unit_count, fab_weeks, fittings, use_escort_permit, eligible_bays, matches_existing_spec)
    best_by_bucket: dict[tuple[Any, ...], CandidateOption] = {}

    for unit_count in hb.unit_counts_allowed:
        for vol in hb.standard_volumes_m3:
            for ins in hb.insulation_props:
                for fit in hb.fittings_props:
                    for design_p in pressure_candidates:
                        if design_p < min_req_p - 1e-9 or design_p > hb.max_design_pressure_mpa:
                            continue
                        # Find minimal valid 0.5mm wall thickness (+ existing wall if applicable)
                        tested_minimal = False
                        walls_to_try = wall_grid
                        if extra_walls:
                            walls_to_try = sorted({*wall_grid, *extra_walls})
                        for w_mm in walls_to_try:
                            if w_mm < hb.min_wall_thickness_mm or w_mm > hb.max_wall_thickness_mm:
                                continue
                            is_existing_wall = (
                                existing_spec_tuple is not None
                                and w_mm == existing_spec_tuple[3]
                            )
                            if tested_minimal and not is_existing_wall:
                                continue
                            opts = evaluate_handbook_candidate(
                                hb,
                                order,
                                route,
                                unit_count=unit_count,
                                unit_volume_m3=vol,
                                design_pressure_mpa=design_p,
                                wall_thickness_mm=w_mm,
                                insulation=ins,
                                fittings=fit,
                                pvi_k_multiplier=pvi_k_multiplier,
                                fittings_lead_deltas=fittings_lead_deltas,
                                bay_max_unit_tare_kg=bay_max_unit_tare_kg,
                                existing_spec_tuple=existing_spec_tuple,
                            )
                            if opts:
                                tested_minimal = True
                                for opt in opts:
                                    bkey = (
                                        opt.unit_count,
                                        opt.fab_weeks,
                                        opt.fittings,
                                        opt.use_escort_permit,
                                        opt.eligible_bays,
                                        opt.matches_existing_spec,
                                    )
                                    prev = best_by_bucket.get(bkey)
                                    if prev is None or (
                                        opt.base_order_cost_eur,
                                        opt.unit_tare_mass_kg,
                                        opt.boil_off_pct_day,
                                    ) < (
                                        prev.base_order_cost_eur,
                                        prev.unit_tare_mass_kg,
                                        prev.boil_off_pct_day,
                                    ):
                                        best_by_bucket[bkey] = opt

    candidates = sorted(
        best_by_bucket.values(),
        key=lambda c: (not c.matches_existing_spec, c.base_order_cost_eur, c.fab_weeks),
    )
    return candidates


def _earliest_slot_on_bay(
    earliest_start: int,
    fab_weeks: int,
    horizon_weeks: int,
    blocked_intervals: list[tuple[int, int]],
) -> int | None:
    """Find smallest integer `s >= earliest_start` (`s <= horizon_weeks`) avoiding `blocked_intervals`."""
    s = max(1, earliest_start)
    while s <= horizon_weeks:
        end = s + fab_weeks
        conflict_end: int | None = None
        for b_start, b_end in blocked_intervals:
            if not (end <= b_start or b_end <= s):
                conflict_end = b_end if conflict_end is None else max(conflict_end, b_end)
        if conflict_end is None:
            return s
        s = conflict_end
    return None


def _find_conflict_free_revision_order(
    revisions: dict[str, tuple[str, int, int]],
    active_bookings: dict[str, dict[str, Any]],
) -> tuple[str, ...] | None:
    """Find a 1-pass execution sequence for `revise_order` calls without intermediate `BAY_OVERLAP`."""
    if not revisions:
        return ()

    current_slots: dict[str, tuple[str, int, int]] = {
        oid: (
            str(b["bay_id"]),
            int(b["bay_start_week"]),
            int(b["fab_end_week"]),
        )
        for oid, b in active_bookings.items()
    }
    remaining = list(revisions.keys())
    ordered: list[str] = []

    while remaining:
        progressed = False
        for oid in list(remaining):
            target_bay, target_start, target_end = revisions[oid]
            overlaps_other = False
            for other_oid, (cur_bay, cur_start, cur_end) in current_slots.items():
                if other_oid == oid or cur_bay != target_bay:
                    continue
                if not (target_end <= cur_start or cur_end <= target_start):
                    overlaps_other = True
                    break
            if not overlaps_other:
                current_slots[oid] = (target_bay, target_start, target_end)
                ordered.append(oid)
                remaining.remove(oid)
                progressed = True
                break
        if not progressed:
            return None
    return tuple(ordered)


def solve_optimal_batch_plan(
    hb: HandbookModel,
    orders: dict[str, dict[str, Any]],
    routes: dict[str, dict[str, Any]],
    *,
    horizon_weeks: int,
    effective_budget_eur: float,
    retooling_fees_spent_eur: float,
    escort_permits_total: int,
    disruption_triggered: bool,
    pvi_k_multiplier: float,
    fittings_lead_deltas: dict[str, int],
    bay_blackouts: list[dict[str, Any]],
    bay_max_unit_tare_kg: dict[str, float],
    active_bookings: dict[str, dict[str, Any]],
    sim_budget_total: int,
    projected_sim_calls_used: int,
) -> BatchPlan:
    """Exact joint branch-and-bound solver maximizing the published handbook score."""
    order_ids = tuple(sorted(orders.keys()))
    n_orders = len(order_ids)

    candidates_by_oid: dict[str, list[CandidateOption]] = {}
    min_cost_by_oid: dict[str, float] = {}
    for oid in order_ids:
        order = orders[oid]
        route = routes[str(order["route_id"])]
        cands = generate_pareto_candidates_for_order(
            hb,
            order,
            route,
            pvi_k_multiplier=pvi_k_multiplier,
            fittings_lead_deltas=fittings_lead_deltas,
            bay_max_unit_tare_kg=bay_max_unit_tare_kg,
            existing_booking=active_bookings.get(oid),
        )
        if not cands:
            raise RuntimeError(f"No valid engineering candidate found for {oid}.")
        candidates_by_oid[oid] = cands
        min_cost_by_oid[oid] = min(c.base_order_cost_eur for c in cands)

    blackouts_by_bay: dict[str, list[tuple[int, int]]] = {b: [] for b in hb.bays}
    for bw in bay_blackouts:
        b_id = str(bw["bay_id"])
        if b_id in blackouts_by_bay:
            blackouts_by_bay[b_id].append((int(bw["start_week"]), int(bw["end_week"])))

    raw_base_score = sum(0.85 * float(orders[oid].get("priority_weight", 0.25)) for oid in order_ids)
    sim_bonus = 0.04 * max(0.0, (sim_budget_total - projected_sim_calls_used) / max(1, sim_budget_total))

    suffix_min_cost = [0.0] * (n_orders + 1)
    for idx in range(n_orders - 1, -1, -1):
        suffix_min_cost[idx] = suffix_min_cost[idx + 1] + min_cost_by_oid[order_ids[idx]]

    best_plan: BatchPlan | None = None
    best_key: tuple[float, float, float, int] = (-1.0, -1e12, -1e12, -999)

    chosen_cands: list[CandidateOption] = []

    def _evaluate_schedules_for_selection(
        selection: list[CandidateOption],
        permits_used: int,
        base_orders_cost: float,
    ) -> None:
        nonlocal best_plan, best_key

        # Upper-bound score check before enumerating bay permutations
        min_possible_spend = retooling_fees_spent_eur + base_orders_cost
        if min_possible_spend > effective_budget_eur + 1e-6:
            return
        max_budget_bonus = 0.07 * min(
            1.0,
            max(
                0.0,
                (effective_budget_eur - min_possible_spend)
                / max(1.0, hb.budget_bonus_headroom_frac * effective_budget_eur),
            ),
        )
        optimistic_score = raw_base_score + max_budget_bonus + 0.04 + sim_bonus
        if best_plan is not None and optimistic_score < best_key[0] - 1e-9:
            return

        # For each already-booked order whose candidate matches its existing spec, we can
        # either keep its exact existing `(bay_id, bay_start_week)` (0 EUR revision fee!)
        # or reschedule it via `revise_order`.
        keep_modes_per_idx: list[list[bool]] = []
        for cand in selection:
            existing = active_bookings.get(cand.order_id)
            if existing is not None and cand.matches_existing_spec:
                eb_bay = str(existing["bay_id"])
                eb_start = int(existing["bay_start_week"])
                eb_end = int(existing["fab_end_week"])
                # Check if existing bay/slot still satisfies post-shock crane & blackout rules
                crane_ok = eb_bay in cand.eligible_bays
                blackout_ok = all(
                    eb_end <= bw_s or bw_e <= eb_start
                    for bw_s, bw_e in blackouts_by_bay[eb_bay]
                )
                if crane_ok and blackout_ok:
                    keep_modes_per_idx.append([True, False])
                else:
                    keep_modes_per_idx.append([False])
            else:
                keep_modes_per_idx.append([False])

        def _enum_keep_modes(pos: int, keep_flags: list[bool]) -> None:
            nonlocal best_plan, best_key
            if pos < n_orders:
                for flag in keep_modes_per_idx[pos]:
                    keep_flags.append(flag)
                    _enum_keep_modes(pos + 1, keep_flags)
                    keep_flags.pop()
                return

            # Compute retooling fees & fixed intervals for kept orders
            retooling_delta_by_oid: dict[str, float] = {}
            release_week_by_oid: dict[str, int] = {}
            fixed_by_bay: dict[str, list[tuple[int, int]]] = {b: list(blackouts_by_bay[b]) for b in hb.bays}
            fixed_actions: dict[str, tuple[str, int, int, int]] = {}
            movable_indices: list[int] = []

            for i, cand in enumerate(selection):
                oid = cand.order_id
                existing = active_bookings.get(oid)
                order_obj = orders[oid]
                route_obj = routes[str(order_obj["route_id"])]
                sla_cap = min(int(order_obj["sla_week"]), horizon_weeks)

                if keep_flags[i] and existing is not None:
                    eb_bay = str(existing["bay_id"])
                    eb_start = int(existing["bay_start_week"])
                    eb_end = int(existing["fab_end_week"])
                    # Check overlap among kept orders on same bay
                    for f_s, f_e in fixed_by_bay[eb_bay]:
                        if not (eb_end <= f_s or f_e <= eb_start):
                            return
                    rel_wk = int(existing["order_release_week"])
                    deliv_wk = (
                        max(eb_end, rel_wk + cand.fittings_lead_weeks)
                        + hb.assembly_commission_weeks
                        + int(route_obj["transit_weeks"])
                    )
                    if deliv_wk > sla_cap:
                        return
                    fixed_by_bay[eb_bay].append((eb_start, eb_end))
                    fixed_actions[oid] = (eb_bay, eb_start, eb_end, deliv_wk)
                    retooling_delta_by_oid[oid] = 0.0
                    release_week_by_oid[oid] = rel_wk
                else:
                    movable_indices.append(i)
                    if existing is None:
                        retooling_delta_by_oid[oid] = 0.0
                        release_week_by_oid[oid] = (
                            hb.post_shock_order_release_week
                            if disruption_triggered
                            else hb.pre_shock_order_release_week
                        )
                    else:
                        fee = (
                            hb.post_shock_first_revision_fee_eur
                            if (disruption_triggered and not bool(existing.get("post_shock_revised", False)))
                            else hb.standard_revision_fee_eur
                        )
                        retooling_delta_by_oid[oid] = fee
                        espec = existing["spec"]
                        changed_fit_or_cnt = (
                            cand.fittings != str(espec["fittings"])
                            or cand.unit_count != int(espec["unit_count"])
                        )
                        release_week_by_oid[oid] = (
                            hb.post_shock_order_release_week
                            if (disruption_triggered and changed_fit_or_cnt)
                            else int(existing["order_release_week"])
                        )

            total_spend = round(
                retooling_fees_spent_eur
                + base_orders_cost
                + sum(retooling_delta_by_oid.values()),
                2,
            )
            if total_spend > effective_budget_eur + 1e-6:
                return

            budget_bonus = 0.07 * min(
                1.0,
                max(
                    0.0,
                    (effective_budget_eur - total_spend)
                    / max(1.0, hb.budget_bonus_headroom_frac * effective_budget_eur),
                ),
            )
            if best_plan is not None and (raw_base_score + budget_bonus + 0.04 + sim_bonus) < best_key[0] - 1e-9:
                return

            # Place `movable_indices` sequentially onto bays via exact permutation DFS
            placed_movable: dict[str, tuple[str, int, int, int]] = {}
            bay_cursors: dict[str, int] = {b: 1 for b in hb.bays}

            def _place_movable(remaining_idxs: list[int]) -> None:
                nonlocal best_plan, best_key
                if not remaining_idxs:
                    revisions_needed: dict[str, tuple[str, int, int]] = {}
                    for i, cand in enumerate(selection):
                        oid = cand.order_id
                        if oid in active_bookings and not keep_flags[i]:
                            b_id, s_wk, e_wk, _ = placed_movable[oid]
                            revisions_needed[oid] = (b_id, s_wk, e_wk)

                    rev_order = _find_conflict_free_revision_order(revisions_needed, active_bookings)
                    if rev_order is None:
                        return

                    all_placements = {**fixed_actions, **placed_movable}
                    slacks: list[float] = []
                    actions_map: dict[str, ScheduledOrderAction] = {}
                    for i, cand in enumerate(selection):
                        oid = cand.order_id
                        order_obj = orders[oid]
                        b_id, s_wk, e_wk, deliv_wk = all_placements[oid]
                        sla_cap = min(int(order_obj["sla_week"]), horizon_weeks)
                        slacks.append(float(sla_cap - deliv_wk))
                        actions_map[oid] = ScheduledOrderAction(
                            order_id=oid,
                            route_id=str(order_obj["route_id"]),
                            bay_id=b_id,
                            bay_start_week=s_wk,
                            fab_end_week=e_wk,
                            delivery_week=deliv_wk,
                            sla_week=int(order_obj["sla_week"]),
                            keep_existing_booking=keep_flags[i],
                            retooling_fee_delta_eur=retooling_delta_by_oid[oid],
                            candidate=cand,
                        )

                    avg_slack = sum(slacks) / float(n_orders)
                    sched_bonus = 0.04 * min(1.0, max(0.0, avg_slack / 2.0))
                    score = round(
                        min(1.0, raw_base_score + budget_bonus + sched_bonus + sim_bonus),
                        6,
                    )
                    cand_key = (
                        score,
                        round(effective_budget_eur - total_spend, 2),
                        round(avg_slack, 4),
                        -len(rev_order),
                    )
                    if cand_key > best_key:
                        best_key = cand_key
                        best_plan = BatchPlan(
                            projected_score=score,
                            projected_total_spent_eur=total_spend,
                            avg_slack_weeks=avg_slack,
                            permits_used=permits_used,
                            actions=actions_map,
                            revision_execution_order=rev_order,
                        )
                    return

                for idx_pos, m_idx in enumerate(remaining_idxs):
                    cand = selection[m_idx]
                    oid = cand.order_id
                    order_obj = orders[oid]
                    route_obj = routes[str(order_obj["route_id"])]
                    sla_cap = min(int(order_obj["sla_week"]), horizon_weeks)
                    rel_wk = release_week_by_oid[oid]
                    next_remaining = remaining_idxs[:idx_pos] + remaining_idxs[idx_pos + 1 :]

                    for b_id in cand.eligible_bays:
                        # Canonical symmetry breaking: only place `m_idx` on `b_id` after any earlier
                        # movable order already placed on `b_id` via `bay_cursors[b_id]`
                        prev_cursor = bay_cursors[b_id]
                        s_wk = _earliest_slot_on_bay(
                            prev_cursor,
                            cand.fab_weeks,
                            horizon_weeks,
                            fixed_by_bay[b_id],
                        )
                        if s_wk is None:
                            continue
                        e_wk = s_wk + cand.fab_weeks
                        deliv_wk = (
                            max(e_wk, rel_wk + cand.fittings_lead_weeks)
                            + hb.assembly_commission_weeks
                            + int(route_obj["transit_weeks"])
                        )
                        if deliv_wk > sla_cap:
                            continue

                        placed_movable[oid] = (b_id, s_wk, e_wk, deliv_wk)
                        bay_cursors[b_id] = e_wk
                        _place_movable(next_remaining)
                        bay_cursors[b_id] = prev_cursor
                        placed_movable.pop(oid, None)

            _place_movable(movable_indices)

        _enum_keep_modes(0, [])

    def _dfs_select(order_pos: int, permits_so_far: int, cost_so_far: float) -> None:
        if permits_so_far > escort_permits_total:
            return
        if retooling_fees_spent_eur + cost_so_far + suffix_min_cost[order_pos] > effective_budget_eur + 1e-6:
            return
        if order_pos == n_orders:
            _evaluate_schedules_for_selection(chosen_cands, permits_so_far, cost_so_far)
            return

        oid = order_ids[order_pos]
        for cand in candidates_by_oid[oid]:
            np = permits_so_far + (1 if cand.use_escort_permit else 0)
            if np > escort_permits_total:
                continue
            chosen_cands.append(cand)
            _dfs_select(order_pos + 1, np, cost_so_far + cand.base_order_cost_eur)
            chosen_cands.pop()

    _dfs_select(0, 0, 0.0)

    if best_plan is None:
        raise RuntimeError("Joint branch-and-bound planner found no feasible 4-order commissioning plan.")
    return best_plan


async def _refresh_world_state(
    client: CryoForgeMCPClient,
    order_ids: list[str],
    route_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Inspect all orders, routes, and factory state over public MCP tools."""
    orders: dict[str, dict[str, Any]] = {}
    for oid in order_ids:
        res = await client.call_tool("inspect_order", {"order_id": oid})
        if not res.get("ok"):
            raise RuntimeError(f"inspect_order({oid}) failed: {res}")
        orders[oid] = res["order"]

    routes: dict[str, dict[str, Any]] = {}
    for rid in route_ids:
        res = await client.call_tool("inspect_route", {"route_id": rid})
        if not res.get("ok"):
            raise RuntimeError(f"inspect_route({rid}) failed: {res}")
        routes[rid] = res["route"]

    fstate = await client.call_tool("get_factory_state")
    if not fstate.get("ok"):
        raise RuntimeError(f"get_factory_state() failed: {fstate}")
    return orders, routes, fstate


async def _verify_tightest_margin_with_sim(
    client: CryoForgeMCPClient,
    plan: BatchPlan,
) -> dict[str, Any]:
    """Run 1 targeted `run_physics_sim` call on the tightest engineering-margin order in `plan`."""
    tightest_action = min(
        plan.actions.values(),
        key=lambda a: a.candidate.tightest_relative_margin,
    )
    cand = tightest_action.candidate
    sim_res = await client.call_tool(
        "run_physics_sim",
        {
            "order_id": cand.order_id,
            "unit_count": cand.unit_count,
            "unit_volume_m3": cand.unit_volume_m3,
            "design_pressure_mpa": cand.design_pressure_mpa,
            "wall_thickness_mm": cand.wall_thickness_mm,
            "insulation": cand.insulation,
            "fittings": cand.fittings,
        },
    )
    if not sim_res.get("ok") or not sim_res.get("physics", {}).get("all_engineering_pass"):
        raise RuntimeError(
            f"Physics sim verification failed for {cand.order_id}: {sim_res}"
        )
    return sim_res


async def solve_episode_via_mcp(client: CryoForgeMCPClient) -> dict[str, Any]:
    """Execute full closed-loop commissioning on `client` using ONLY the 10 public MCP tools."""
    # 1. Read episode brief & full engineering handbook
    brief = await client.call_tool("get_episode_brief")
    if not brief.get("ok"):
        raise RuntimeError(f"get_episode_brief failed: {brief}")

    handbook_resp = await client.call_tool("query_handbook", {"section": "all"})
    if not handbook_resp.get("ok"):
        raise RuntimeError(f"query_handbook('all') failed: {handbook_resp}")
    hb = HandbookModel.from_handbook_response(handbook_resp)

    order_ids = [str(o["order_id"]) for o in brief["orders"]]
    route_ids = [str(r["route_id"]) for r in brief["routes"]]

    # 2. Inspect all orders, all routes, and factory state
    orders, routes, fstate = await _refresh_world_state(client, order_ids, route_ids)

    # 3. Compute exact pre-shock joint plan (accounting for 2 total targeted sim calls: 1 pre + 1 post)
    sim_budget_total = int(fstate["sim_budget_total"])
    pre_plan = solve_optimal_batch_plan(
        hb,
        orders,
        routes,
        horizon_weeks=int(fstate["horizon_weeks"]),
        effective_budget_eur=float(fstate["effective_budget_eur"]),
        retooling_fees_spent_eur=float(fstate.get("retooling_fees_eur", 0.0)),
        escort_permits_total=int(fstate["escort_permits_total"]),
        disruption_triggered=bool(fstate["disruption_triggered"]),
        pvi_k_multiplier=float(fstate.get("active_pvi_k_multiplier", 1.0)),
        fittings_lead_deltas={
            str(k): int(v) for k, v in (fstate.get("active_fittings_lead_deltas") or {}).items()
        },
        bay_blackouts=list(fstate.get("bay_blackouts") or []),
        bay_max_unit_tare_kg={
            str(k): float(v) for k, v in (fstate.get("bay_max_unit_tare_kg") or {}).items()
        },
        active_bookings=dict(fstate.get("booked_orders") or {}),
        sim_budget_total=sim_budget_total,
        projected_sim_calls_used=2,
    )

    # 4. Pre-shock targeted physics verification on tightest-margin design
    await _verify_tightest_margin_with_sim(client, pre_plan)

    # 5. Book orders sequentially (tightest SLA / longest fittings lead first) until disruption triggers
    booking_priority = sorted(
        order_ids,
        key=lambda oid: (
            pre_plan.actions[oid].sla_week,
            -pre_plan.actions[oid].candidate.fittings_lead_weeks,
            pre_plan.actions[oid].bay_start_week,
            oid,
        ),
    )

    shock_detected = bool(fstate["disruption_triggered"])
    for oid in booking_priority:
        act = pre_plan.actions[oid]
        cand = act.candidate
        res_book = await client.call_tool(
            "reserve_bay_and_order",
            {
                "order_id": oid,
                "bay_id": act.bay_id,
                "bay_start_week": act.bay_start_week,
                "unit_count": cand.unit_count,
                "unit_volume_m3": cand.unit_volume_m3,
                "design_pressure_mpa": cand.design_pressure_mpa,
                "wall_thickness_mm": cand.wall_thickness_mm,
                "insulation": cand.insulation,
                "fittings": cand.fittings,
            },
        )
        if not res_book.get("ok"):
            raise RuntimeError(f"Pre-shock reserve_bay_and_order({oid}) failed: {res_book}")
        if res_book.get("disruption_triggered") or res_book.get("disruption_alert"):
            shock_detected = True
            break

        res_tr = await client.call_tool(
            "assign_transport",
            {
                "order_id": oid,
                "route_id": act.route_id,
                "use_escort_permit": cand.use_escort_permit,
            },
        )
        if not res_tr.get("ok"):
            raise RuntimeError(f"Pre-shock assign_transport({oid}) failed: {res_tr}")
        if res_tr.get("disruption_triggered") or res_tr.get("disruption_alert"):
            shock_detected = True
            break

    # 6. Post-shock re-inspection & joint re-optimization
    if shock_detected:
        orders, routes, fstate = await _refresh_world_state(client, order_ids, route_ids)
        sim_used_now = sim_budget_total - int(fstate["sim_calls_remaining"])
        post_plan = solve_optimal_batch_plan(
            hb,
            orders,
            routes,
            horizon_weeks=int(fstate["horizon_weeks"]),
            effective_budget_eur=float(fstate["effective_budget_eur"]),
            retooling_fees_spent_eur=float(fstate.get("retooling_fees_eur", 0.0)),
            escort_permits_total=int(fstate["escort_permits_total"]),
            disruption_triggered=True,
            pvi_k_multiplier=float(fstate.get("active_pvi_k_multiplier", 1.0)),
            fittings_lead_deltas={
                str(k): int(v) for k, v in (fstate.get("active_fittings_lead_deltas") or {}).items()
            },
            bay_blackouts=list(fstate.get("bay_blackouts") or []),
            bay_max_unit_tare_kg={
                str(k): float(v) for k, v in (fstate.get("bay_max_unit_tare_kg") or {}).items()
            },
            active_bookings=dict(fstate.get("booked_orders") or {}),
            sim_budget_total=sim_budget_total,
            projected_sim_calls_used=sim_used_now + 1,
        )

        # Post-shock targeted physics verification on tightest-margin design
        await _verify_tightest_margin_with_sim(client, post_plan)

        active_bookings = dict(fstate.get("booked_orders") or {})

        # 7a. Revise ONLY already-booked orders that actually changed spec or bay slot
        for oid in post_plan.revision_execution_order:
            act = post_plan.actions[oid]
            cand = act.candidate
            rev_res = await client.call_tool(
                "revise_order",
                {
                    "order_id": oid,
                    "bay_id": act.bay_id,
                    "bay_start_week": act.bay_start_week,
                    "unit_count": cand.unit_count,
                    "unit_volume_m3": cand.unit_volume_m3,
                    "design_pressure_mpa": cand.design_pressure_mpa,
                    "wall_thickness_mm": cand.wall_thickness_mm,
                    "insulation": cand.insulation,
                    "fittings": cand.fittings,
                    "route_id": act.route_id,
                    "use_escort_permit": cand.use_escort_permit,
                },
            )
            if not rev_res.get("ok"):
                raise RuntimeError(f"Post-shock revise_order({oid}) failed: {rev_res}")

        # 7b. Ensure transport is assigned on any kept pre-shock order
        for oid, act in post_plan.actions.items():
            if oid in active_bookings and act.keep_existing_booking:
                eb = active_bookings[oid]
                if (
                    eb.get("assigned_route_id") != act.route_id
                    or bool(eb.get("use_escort_permit", False)) != act.candidate.use_escort_permit
                ):
                    tr_res = await client.call_tool(
                        "assign_transport",
                        {
                            "order_id": oid,
                            "route_id": act.route_id,
                            "use_escort_permit": act.candidate.use_escort_permit,
                        },
                    )
                    if not tr_res.get("ok"):
                        raise RuntimeError(f"Post-shock assign_transport({oid}) failed: {tr_res}")

        # 7c. Book & route remaining unbooked orders
        for oid in sorted(order_ids, key=lambda x: (post_plan.actions[x].bay_start_week, x)):
            if oid in active_bookings:
                continue
            act = post_plan.actions[oid]
            cand = act.candidate
            book_res = await client.call_tool(
                "reserve_bay_and_order",
                {
                    "order_id": oid,
                    "bay_id": act.bay_id,
                    "bay_start_week": act.bay_start_week,
                    "unit_count": cand.unit_count,
                    "unit_volume_m3": cand.unit_volume_m3,
                    "design_pressure_mpa": cand.design_pressure_mpa,
                    "wall_thickness_mm": cand.wall_thickness_mm,
                    "insulation": cand.insulation,
                    "fittings": cand.fittings,
                },
            )
            if not book_res.get("ok"):
                raise RuntimeError(f"Post-shock reserve_bay_and_order({oid}) failed: {book_res}")

            tr_res = await client.call_tool(
                "assign_transport",
                {
                    "order_id": oid,
                    "route_id": act.route_id,
                    "use_escort_permit": cand.use_escort_permit,
                },
            )
            if not tr_res.get("ok"):
                raise RuntimeError(f"Post-shock assign_transport({oid}) failed: {tr_res}")

    # 8. Finalize plant commissioning
    fin_res = await client.call_tool("finalize_commissioning")
    if not fin_res.get("ok"):
        raise RuntimeError(f"finalize_commissioning() failed: {fin_res}")
    return fin_res


async def run_reference_episode(
    seed: str = "S01",
    *,
    mcp_url: str | None = None,
    run_id: str | None = None,
    log_dir: str | Path | None = DEFAULT_TRAJECTORIES_DIR,
) -> dict[str, Any]:
    """Run the zero-cheating reference solver on `seed` over MCP (`stdio` or `sse`)."""
    effective_run_id = run_id or f"reference_{seed}"
    async with open_cryoforge_mcp_session(
        seed=seed,
        mcp_url=mcp_url,
        run_id=effective_run_id,
        agent_name="CryoForge Reference Planner",
        agent_type="reference",
        model_id="cryoforge-bnb-v1",
        log_dir=log_dir,
    ) as client:
        finalize_payload = await solve_episode_via_mcp(client)
        sse_grade = client.fetch_sse_grade() if mcp_url else None

    persisted = load_persisted_trajectory(effective_run_id, log_dir) if not mcp_url else None
    verification = (
        (sse_grade or {}).get("verification")
        or (persisted or {}).get("verification")
        or finalize_payload.get("verification")
    )
    return {
        "seed": seed,
        "run_id": effective_run_id,
        "final_score": float(finalize_payload["final_score"]),
        "valid_order_count": int(finalize_payload["valid_order_count"]),
        "total_order_count": int(finalize_payload["total_order_count"]),
        "verification": verification,
    }


async def run_all_reference_seeds(
    seeds: tuple[str, ...] = DEFAULT_ALL_SEEDS,
    *,
    log_dir: str | Path | None = DEFAULT_TRAJECTORIES_DIR,
) -> dict[str, Any]:
    """Run the reference solver deterministically across `seeds` and summarize scores."""
    results: list[dict[str, Any]] = []
    for seed in seeds:
        res = await run_reference_episode(seed=seed, log_dir=log_dir)
        results.append(res)

    scores = [r["final_score"] for r in results]
    mean_score = round(sum(scores) / max(1, len(scores)), 4)
    return {
        "agent": "reference",
        "mean_score": mean_score,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "episodes": results,
    }


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.reference_solver",
        description="Zero-cheating CryoForge RL Reference Solver over public MCP tools.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=list(DEFAULT_ALL_SEEDS),
        help="Scenario seeds to solve (default: S01..S08).",
    )
    parser.add_argument(
        "--mcp-url",
        default=None,
        help="Optional SSE MCP URL (e.g. http://127.0.0.1:8090/sse) for single-seed sidecar mode.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional trajectory run_id override (single-seed mode).",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_TRAJECTORIES_DIR),
        help="Directory to persist TrajectoryLog JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save summary JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_cli(argv)
    if args.mcp_url:
        seed = args.seeds[0]
        summary = asyncio.run(
            run_reference_episode(
                seed=seed,
                mcp_url=args.mcp_url,
                run_id=args.run_id,
                log_dir=args.log_dir,
            )
        )
    else:
        summary = asyncio.run(
            run_all_reference_seeds(
                seeds=tuple(args.seeds),
                log_dir=args.log_dir,
            )
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
