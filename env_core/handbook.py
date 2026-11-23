"""CryoForge RL Engineering & Commissioning Handbook.

Zero-drift dynamic handbook pulling all physical constants, material tables,
catalogue limits, disruption mechanics, and scoring equations directly from
`env_core.contracts` so agent-visible documentation and server-side verification
are 100% aligned with zero hidden rules.
"""

from __future__ import annotations

import copy
from typing import Any

from env_core.contracts import (
    AMBIENT_TEMP_K,
    ASSEMBLY_COMMISSION_WEEKS,
    BASE_ALLOWABLE_STRESS_MPA,
    BUDGET_BONUS_HEADROOM_FRAC,
    BayId,
    DISRUPTION_TRIGGER_BOOKINGS,
    DISRUPTION_TRIGGER_MUTATIONS,
    FILL_FRACTION,
    FITTINGS_PROPS,
    HANDBOOK_SECTIONS,
    INSULATION_PROPS,
    MAX_DESIGN_PRESSURE_MPA,
    MAX_EPISODE_STEPS,
    MAX_WALL_THICKNESS_MM,
    MEDIUM_PROPS,
    MIN_DESIGN_PRESSURE_MPA,
    MIN_WALL_THICKNESS_MM,
    POST_SHOCK_FIRST_REVISION_FEE_EUR,
    SIM_BUDGET_DEFAULT,
    STANDARD_REVISION_FEE_EUR,
    STANDARD_VOLUMES_M3,
    UNIT_COUNTS_ALLOWED,
)


def _serialize_medium_table() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for medium, props in MEDIUM_PROPS.items():
        table[medium.value] = {
            "label": props["label"],
            "density_kg_m3": float(props["density_kg_m3"]),
            "boil_temp_k": float(props["boil_temp_k"]),
            "latent_heat_kj_kg": float(props["latent_heat_kj_kg"]),
            "safety_factor": float(props["safety_factor"]),
            "allowable_stress_mpa": round(
                BASE_ALLOWABLE_STRESS_MPA / float(props["safety_factor"]), 2
            ),
            "requires_evi": bool(props["requires_evi"]),
            "forbidden_fittings": [f.value for f in props["forbidden_fittings"]],
            "lox_high_pressure_rule": bool(props["lox_high_pressure_rule"]),
        }
    return table


def _serialize_insulation_table() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for ins, props in INSULATION_PROPS.items():
        table[ins.value] = {
            "label": props["label"],
            "annulus_mm": float(props["annulus_mm"]),
            "k_eff_w_mk": float(props["k_eff_w_mk"]),
            "casing_mass_factor": float(props["casing_mass_factor"]),
            "cost_multiplier": float(props["cost_multiplier"]),
        }
    return table


def _serialize_fittings_table() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for fit, props in FITTINGS_PROPS.items():
        table[fit.value] = {
            "label": props["label"],
            "weld_efficiency": float(props["weld_efficiency"]),
            "valve_heat_leak_w": float(props["valve_heat_leak_w"]),
            "base_lead_weeks": int(props["base_lead_weeks"]),
            "cost_multiplier": float(props["cost_multiplier"]),
            "fixed_kit_eur": float(props["fixed_kit_eur"]),
        }
    return table


def _build_sections() -> dict[str, dict[str, Any]]:
    return {
        "overview": {
            "section": "overview",
            "title": "CryoForge Commissioning Environment Overview & Catalogue Bounds",
            "mission": (
                "Configure, schedule, ship, and commission all 4 cryogenic vessel customer orders "
                "(ORD-1..ORD-4) across parallel fabrication bays (BAY-A, BAY-B) within the factory "
                "budget, escort-permit quota, route mass/gauge envelopes, and customer SLA deadlines, "
                "while adapting to a mid-episode supply/infrastructure shock."
            ),
            "episode_limits": {
                "max_episode_steps": MAX_EPISODE_STEPS,
                "sim_budget_default": SIM_BUDGET_DEFAULT,
                "bays_available": [b.value for b in BayId],
                "reset_available": False,
                "finalization_required": True,
            },
            "catalogue_bounds": {
                "unit_counts_allowed": list(UNIT_COUNTS_ALLOWED),
                "standard_volumes_m3": list(STANDARD_VOLUMES_M3),
                "min_design_pressure_mpa": MIN_DESIGN_PRESSURE_MPA,
                "max_design_pressure_mpa": MAX_DESIGN_PRESSURE_MPA,
                "min_wall_thickness_mm": MIN_WALL_THICKNESS_MM,
                "max_wall_thickness_mm": MAX_WALL_THICKNESS_MM,
            },
            "anti_oracle_invariant": (
                "reserve_bay_and_order and revise_order validate only discrete catalogue bounds, "
                "bay existence, bay schedule/blackout non-overlap, and active crane tare limits. "
                "They NEVER reveal hoop_stress_mpa, boil_off_pct_day, stress_pass, or boil_off_pass. "
                "Use run_physics_sim (limited to sim_budget_default calls) and/or compute exact "
                "equations from the physics section before finalizing."
            ),
            "workflow_checklist": [
                "1. Call get_episode_brief (and query_handbook) to inspect orders, routes, budget, permits, and equations.",
                "2. Design each order (unit_count, unit_volume_m3, design_pressure_mpa, wall_thickness_mm, insulation, fittings).",
                "3. Reserve bay slots via reserve_bay_and_order and assign delivery corridors via assign_transport.",
                "4. When the mid-episode disruption triggers, inspect disruption_alert and get_factory_state, revise impacted bookings via revise_order, and book remaining orders under post-shock constraints.",
                "5. Call finalize_commissioning before step 42; unfinalized episodes score 0.0.",
            ],
        },
        "physics": {
            "section": "physics",
            "title": "Exact Cryogenic Vessel Geometry, Mass, Hoop Stress & Boil-Off Equations",
            "constants": {
                "fill_fraction": FILL_FRACTION,
                "ambient_temp_k": AMBIENT_TEMP_K,
                "base_allowable_stress_mpa": BASE_ALLOWABLE_STRESS_MPA,
                "inner_shell_steel_density_kg_m3": 7900.0,
                "inner_shell_head_stiffener_factor": 1.12,
                "outer_jacket_thickness_m": 0.0055,
                "outer_jacket_steel_density_kg_m3": 7850.0,
                "nozzle_piping_fixed_tare_kg": 420.0,
                "stationary_shipping_saddles_kg": 480.0,
                "transportable_skid_tare_fraction": 0.18,
                "transportable_skid_fixed_kg": 650.0,
                "modular_bank_inter_unit_spacing_mm": 400,
                "seconds_per_day": 86400.0,
            },
            "nominal_outer_diameter_mm_lookup": [
                {"unit_volume_m3_condition": "unit_volume_m3 <= 2.0", "outer_diameter_mm": 1400},
                {"unit_volume_m3_condition": "2.0 < unit_volume_m3 <= 5.0", "outer_diameter_mm": 1600},
                {"unit_volume_m3_condition": "5.0 < unit_volume_m3 <= 10.0", "outer_diameter_mm": 2000},
                {"unit_volume_m3_condition": "10.0 < unit_volume_m3 <= 15.0", "outer_diameter_mm": 2400},
                {"unit_volume_m3_condition": "15.0 < unit_volume_m3 <= 25.0", "outer_diameter_mm": 2600},
                {"unit_volume_m3_condition": "unit_volume_m3 > 25.0 (30.0)", "outer_diameter_mm": 3000},
            ],
            "equations": {
                "compute_dimensions": {
                    "inner_diameter_mm": "int(round(outer_diameter_mm - 2.0 * annulus_mm))",
                    "d_inner_m": "inner_diameter_mm / 1000.0",
                    "raw_axis_mm": "(unit_volume_m3 / (pi * (d_inner_m / 2.0)**2)) * 1000.0",
                    "shell_axis_mm": "int(ceil(raw_axis_mm / 10.0) * 10)",
                    "STATIONARY": {
                        "installed_height_mm": "shell_axis_mm + 900",
                        "installed_width_mm": "outer_diameter_mm",
                        "installed_length_mm": "outer_diameter_mm",
                        "shipping_height_mm": "outer_diameter_mm + 350",
                        "shipping_width_mm": "outer_diameter_mm",
                        "shipping_length_mm": "shell_axis_mm + 900",
                    },
                    "TRANSPORTABLE": {
                        "installed_height_mm": "outer_diameter_mm + 600",
                        "installed_width_mm": "outer_diameter_mm + 200",
                        "installed_length_mm": "shell_axis_mm + 1200",
                        "shipping_height_mm": "installed_height_mm",
                        "shipping_width_mm": "installed_width_mm",
                        "shipping_length_mm": "installed_length_mm",
                    },
                },
                "site_clearance": {
                    "bank_footprint_mm": "unit_count * installed_width_mm + (unit_count - 1) * 400",
                    "pass_condition": (
                        "installed_height_mm <= order.max_site_height_mm "
                        "AND bank_footprint_mm <= order.max_site_footprint_mm"
                    ),
                },
                "usable_unit_mass_kg": {
                    "usable_unit_mass_kg": "round(unit_volume_m3 * density_kg_m3 * FILL_FRACTION, 2)",
                    "usable_total_mass_kg": "round(unit_count * usable_unit_mass_kg, 2)",
                    "pass_condition": "usable_total_mass_kg >= order.required_product_mass_kg",
                },
                "pressure_and_hoop_stress_mpa": {
                    "pressure_pass": "design_pressure_mpa >= order.min_pressure_mpa",
                    "hoop_stress_mpa": (
                        "round((design_pressure_mpa * (inner_diameter_mm + wall_thickness_mm)) "
                        "/ (2.0 * weld_efficiency * wall_thickness_mm), 2)"
                    ),
                    "allowable_stress_mpa": "round(BASE_ALLOWABLE_STRESS_MPA / safety_factor, 2)",
                    "stress_pass": "hoop_stress_mpa <= allowable_stress_mpa",
                },
                "compute_masses_and_area": {
                    "r_m": "(inner_diameter_mm / 1000.0) / 2.0",
                    "l_m": "shell_axis_mm / 1000.0",
                    "area_m2": "2.0 * pi * r_m * l_m + 4.0 * pi * (r_m**2) * 0.65",
                    "inner_shell_vol_m3": "area_m2 * (wall_thickness_mm / 1000.0) * 1.12",
                    "inner_shell_mass_kg": "inner_shell_vol_m3 * 7900.0",
                    "outer_jacket_mass_kg": "area_m2 * 0.0055 * 7850.0 * casing_mass_factor",
                    "unit_tare_mass_kg": "round(inner_shell_mass_kg + outer_jacket_mass_kg + 420.0, 1)",
                    "skid_or_cradle_kg": (
                        "round(0.18 * unit_tare_mass_kg + 650.0, 1) if execution == 'TRANSPORTABLE' else 480.0"
                    ),
                    "unit_transport_mass_kg": "round(unit_tare_mass_kg + skid_or_cradle_kg, 1)",
                },
                "boil_off_pct_day": {
                    "k_eff": "k_eff_w_mk * (pvi_k_multiplier if insulation == 'PVI' else 1.0)",
                    "t_ins_m": "annulus_mm / 1000.0",
                    "delta_t": "AMBIENT_TEMP_K - boil_temp_k",
                    "q_cond_w": "k_eff * area_m2 * delta_t / t_ins_m",
                    "q_valve_w": "valve_heat_leak_w * (1.25 if execution == 'TRANSPORTABLE' else 1.00)",
                    "q_day_j": "(q_cond_w + q_valve_w) * 86400.0",
                    "evap_kg_day": "q_day_j / (latent_heat_kj_kg * 1000.0)",
                    "boil_off_pct_day": "round((evap_kg_day / usable_unit_mass_kg) * 100.0, 4)",
                    "boil_off_pass": "boil_off_pct_day <= order.max_boil_off_pct_day",
                },
            },
        },
        "materials_and_insulation": {
            "section": "materials_and_insulation",
            "title": "Cryogenic Media, Vacuum Insulation Systems, Fittings & Hazard Rules",
            "medium_props": _serialize_medium_table(),
            "insulation_props": _serialize_insulation_table(),
            "fittings_props": _serialize_fittings_table(),
            "hazard_compatibility_rules": [
                "LH2 and LHe strictly require EVI multi-layer screen-vacuum insulation (requires_evi=True); PVI fails HAZARD_INSULATION.",
                "LH2 and LHe strictly forbid CN fittings (forbidden_fittings=['CN']); only EU_STD or DE_PRECISION are allowed.",
                "LOX with design_pressure_mpa > 2.0 MPa strictly forbids CN fittings (LOX_IGNITION_RISK); oxygen-clean EU_STD or DE_PRECISION fittings are mandatory.",
            ],
        },
        "transport_and_routes": {
            "section": "transport_and_routes",
            "title": "Heavy Cryogenic Transport Corridors, Bridge Caps & Escort Permits",
            "rules": {
                "designated_route_requirement": (
                    "Each booked order MUST be assigned via assign_transport (or revise_order) "
                    "to its customer-designated route: assigned_route_id == order.route_id."
                ),
                "bridge_mass_cap": (
                    "unit_transport_mass_kg <= route.max_unit_transport_mass_kg. "
                    "Bridge/axle mass caps apply PER UNIT and CANNOT be waived by an escort permit. "
                    "Splitting an order into unit_count = 2 or 3 smaller shells reduces per-unit transport mass."
                ),
                "standard_clearance_no_permit": (
                    "When use_escort_permit == False: "
                    "shipping_width_mm <= route.std_max_width_mm AND shipping_height_mm <= route.std_max_height_mm."
                ),
                "oversize_clearance_with_permit": (
                    "When use_escort_permit == True: "
                    "route.escort_permit_allowed == True AND "
                    "shipping_width_mm <= route.permit_max_width_mm AND "
                    "shipping_height_mm <= route.permit_max_height_mm."
                ),
                "batch_permit_quota": (
                    "Total orders claiming use_escort_permit == True across the batch must be "
                    "<= effective_escort_permits_total. If exceeded, permit_quota_exceeded becomes True "
                    "and ALL orders claiming use_escort_permit == True fail PERMIT_QUOTA_EXCEEDED."
                ),
                "transport_cost_formula_eur": (
                    "unit_count * route.base_freight_eur_per_unit "
                    "+ (route.permit_fee_eur if use_escort_permit else 0.0)"
                ),
            },
        },
        "manufacturing_and_bays": {
            "section": "manufacturing_and_bays",
            "title": "Bay Scheduling, Crane Limits, Manufacturing Cost & Delivery Timeline",
            "constants": {
                "bays": [b.value for b in BayId],
                "assembly_commission_weeks": ASSEMBLY_COMMISSION_WEEKS,
                "series_discount_factor_multi_unit": 0.94,
                "transportable_cost_multiplier": 1.18,
            },
            "compute_fab_and_cost": {
                "base_fab": (
                    "2 + (1 if unit_volume_m3 >= 10.0 else 0) "
                    "+ (1 if unit_volume_m3 >= 20.0 else 0) "
                    "+ (1 if wall_thickness_mm >= 14.0 else 0) "
                    "+ (1 if execution == 'TRANSPORTABLE' else 0)"
                ),
                "fab_weeks": "base_fab + (unit_count - 1)",
                "fab_interval": "[bay_start_week, bay_start_week + fab_weeks) with bay_start_week >= 1",
                "effective_fittings_lead_weeks": "max(1, base_lead_weeks + fittings_lead_delta)",
                "unit_cost_eur": (
                    "(9500.0 + unit_volume_m3 * 2150.0 + unit_tare_mass_kg * 3.10) "
                    "* insulation_cost_multiplier * fittings_cost_multiplier "
                    "* (1.18 if execution == 'TRANSPORTABLE' else 1.00) + fixed_kit_eur"
                ),
                "series_factor": "0.94 if unit_count > 1 else 1.00",
                "manufacturing_cost_eur": "round(unit_count * unit_cost_eur * series_factor, 2)",
            },
            "bay_constraints": {
                "non_overlap": (
                    "Two orders booked on the same bay cannot overlap half-open intervals "
                    "[start_a, end_a) and [start_b, end_b): must have end_a <= start_b or end_b <= start_a."
                ),
                "blackout_windows": (
                    "A booking [bay_start_week, fab_end_week) cannot overlap any active BayBlackoutWindow "
                    "[start_week, end_week) on the same bay."
                ),
                "crane_tare_cap": (
                    "If bay_max_unit_tare_kg is active on bay_id, unit_tare_mass_kg <= bay_max_unit_tare_kg[bay_id]."
                ),
            },
            "delivery_week_formula": {
                "fittings_ready_week": "order_release_week + effective_fittings_lead_weeks",
                "delivery_week": (
                    "max(bay_start_week + fab_weeks, order_release_week + effective_fittings_lead_weeks) "
                    "+ ASSEMBLY_COMMISSION_WEEKS + route.transit_weeks"
                ),
                "sla_pass_condition": (
                    "delivery_week <= order.sla_week AND delivery_week <= episode.horizon_weeks"
                ),
            },
        },
        "disruptions_and_revisions": {
            "section": "disruptions_and_revisions",
            "title": "Mid-Episode Supply/Infrastructure Shocks, Release Weeks & Revision Fees",
            "constants": {
                "disruption_trigger_bookings": DISRUPTION_TRIGGER_BOOKINGS,
                "disruption_trigger_mutations": DISRUPTION_TRIGGER_MUTATIONS,
                "pre_shock_order_release_week": 1,
                "post_shock_order_release_week": 2,
                "post_shock_first_revision_fee_eur": POST_SHOCK_FIRST_REVISION_FEE_EUR,
                "standard_revision_fee_eur": STANDARD_REVISION_FEE_EUR,
            },
            "rules": {
                "trigger_condition": (
                    f"Successful factory mutations (reserve_bay_and_order, assign_transport, revise_order) "
                    f"increment mutation_count. The scenario's DisruptionEvent fires automatically once "
                    f"len(booked_orders) >= {DISRUPTION_TRIGGER_BOOKINGS} OR "
                    f"mutation_count >= {DISRUPTION_TRIGGER_MUTATIONS}, whichever comes first."
                ),
                "world_state_updates": (
                    "When triggered, the DisruptionEvent immediately updates active fittings_lead_deltas, "
                    "pvi_k_multiplier, effective_budget_eur (+ budget_delta_eur), effective_escort_permits_total "
                    "(+ escort_permits_delta), bay_blackouts, bay_max_unit_tare_kg, route mass/height/permit "
                    "overrides, and customer order min_pressure_mpa / sla_week overrides."
                ),
                "order_release_week_rule": (
                    "Orders booked before the shock start with order_release_week = 1. "
                    "Once disruption_triggered is True, any NEW reserve_bay_and_order gets order_release_week = 2, "
                    "and any revise_order that changes fittings or unit_count updates order_release_week = 2 "
                    "(revisions that keep both fittings and unit_count unchanged retain their prior order_release_week)."
                ),
                "revision_retooling_fees": (
                    f"First post-shock revision of a booked order (disruption_triggered == True and "
                    f"post_shock_revised == False) costs {POST_SHOCK_FIRST_REVISION_FEE_EUR:.1f} EUR and marks "
                    f"post_shock_revised = True. Pre-shock revisions or repeat revisions of the same order cost "
                    f"{STANDARD_REVISION_FEE_EUR:.1f} EUR. All revision fees accumulate in retooling_fees_eur "
                    f"and count toward total_spent_eur."
                ),
            },
        },
        "scoring": {
            "section": "scoring",
            "title": "Zero-Trust Verification & Batch Commissioning Scoring Formula",
            "finalization_gate": (
                "If finalize_commissioning() was NOT called before the episode ends, final_score = 0.0."
            ),
            "order_hard_checks": [
                "1. Order is booked AND assigned_route_id == order.route_id.",
                "2. Post-shock evaluate_design_physics has all_engineering_pass == True (capacity, pressure, hoop stress, material hazard compatibility, boil-off, site height & bank footprint).",
                "3. Zero bay overlap with other booked orders on the same bay, zero overlap with active bay_blackouts, and unit_tare_mass_kg <= bay_max_unit_tare_kg (if active).",
                "4. Transport envelope satisfied: unit_transport_mass_kg <= route.max_unit_transport_mass_kg, shipping width/height within std or permit gauge, route.escort_permit_allowed if permit used, and total batch permits <= effective_escort_permits_total.",
                "5. Schedule satisfied: delivery_week <= order.sla_week AND delivery_week <= horizon_weeks.",
            ],
            "base_progress_credit": {
                "max_credit": 0.85,
                "formula": "raw_score = sum(0.85 * order.priority_weight for each order passing all 5 hard checks)",
                "per_order_credit_for_equal_weights": 0.2125,
            },
            "efficiency_and_margin_bonuses": {
                "eligibility": "Awarded ONLY when valid_order_count == 4 and total_spent_eur <= effective_budget_eur",
                "max_total_bonus": 0.15,
                "budget_efficiency_bonus": (
                    f"0.07 * min(1.0, max(0.0, (effective_budget_eur - total_spent_eur) / ({BUDGET_BONUS_HEADROOM_FRAC:.2f} * effective_budget_eur)))"
                ),
                "schedule_margin_bonus": (
                    "0.04 * min(1.0, max(0.0, avg_slack_weeks / 2.0)), "
                    "where avg_slack_weeks = mean(min(order.sla_week, horizon_weeks) - delivery_week)"
                ),
                "sim_efficiency_bonus": (
                    "0.04 * max(0.0, (sim_budget_total - sim_calls_used) / sim_budget_total)"
                ),
            },
            "batch_completion_gate": {
                "0_valid_orders": "0.0000",
                "1_valid_order": "min(raw_score, 0.0800)  # anti-cherry-picking cap",
                "2_valid_orders": "raw_score * 0.80       # ~0.3400",
                "3_valid_orders": "raw_score * 0.92       # ~0.5865",
                "4_valid_orders": "min(1.0, max(0.0, raw_score + bonuses))  # 0.8500 to 1.0000",
            },
            "budget_breach_penalty": {
                "total_spent_eur": (
                    "sum(manufacturing_cost_eur + unit_count * base_freight_eur_per_unit "
                    "+ (permit_fee_eur if use_escort_permit else 0.0)) + retooling_fees_eur"
                ),
                "penalty_if_exceeded": (
                    "If total_spent_eur > effective_budget_eur: bonuses = 0.0 and "
                    "score = min(score * 0.20, 0.15)."
                ),
            },
        },
    }


def get_handbook_section(section: str = "all") -> dict[str, Any]:
    """Return exact zero-drift handbook documentation for `section` or `'all'`."""
    if not isinstance(section, str):
        raise ValueError(f"Handbook section must be a string, got {type(section).__name__}")
    normalized = section.strip().lower()
    if normalized not in HANDBOOK_SECTIONS:
        raise ValueError(
            f"Unknown handbook section '{section}'. Valid sections: {list(HANDBOOK_SECTIONS)}"
        )

    all_sections = _build_sections()
    if normalized == "all":
        return copy.deepcopy(
            {
                "section": "all",
                "available_sections": [s for s in HANDBOOK_SECTIONS if s != "all"],
                "sections": all_sections,
            }
        )
    return copy.deepcopy(all_sections[normalized])
