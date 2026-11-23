"""Frozen shared domain contracts, physical constants, and trajectory schemas for CryoForge RL.

All core environment modules, MCP tool schemas, reference solvers, exploit tests,
tiebreaker scripts, and replay viewers rely on these exact definitions so handbook
documentation and server-side verification can never drift.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class Medium(StrEnum):
    LIN = "LIN"    # Liquid Nitrogen (-196 C)
    LOX = "LOX"    # Liquid Oxygen (-183 C)
    LAR = "LAR"    # Liquid Argon (-186 C)
    LCO2 = "LCO2"  # Liquid Carbon Dioxide (-30 C)
    LNG = "LNG"    # Liquefied Natural Gas (-162 C)
    LH2 = "LH2"    # Liquid Hydrogen (-253 C, high hazard)
    LHe = "LHe"    # Liquid Helium (-269 C, extreme cryo)


class ExecutionType(StrEnum):
    STATIONARY = "STATIONARY"        # Vertical GHK tank; shipped horizontal on flatbed, erected vertical on site
    TRANSPORTABLE = "TRANSPORTABLE"  # Horizontal CTK tank permanently mounted on heavy transport skid


class InsulationType(StrEnum):
    EVI = "EVI"  # Multi-layer screen-vacuum insulation (ЭВИ)
    PVI = "PVI"  # Perlite-vacuum powder insulation (ПВИ)


class FittingsGrade(StrEnum):
    CN = "CN"                        # Standard certified fittings (shortest lead, lowest weld/valve grade)
    EU_STD = "EU_STD"                # European industrial cryo valves (oxygen/cryo certified)
    DE_PRECISION = "DE_PRECISION"    # German precision cryo-valves (highest weld efficiency, low heat leak, long lead)


class BayId(StrEnum):
    BAY_A = "BAY-A"
    BAY_B = "BAY-B"


EVAL_SEEDS: tuple[str, ...] = ("S01", "S02", "S03", "S04", "S05")
TIEBREAKER_SEEDS: tuple[str, ...] = ("S06", "S07", "S08")
ALL_SEEDS: tuple[str, ...] = EVAL_SEEDS + TIEBREAKER_SEEDS

STANDARD_VOLUMES_M3: tuple[float, ...] = (
    1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0
)

UNIT_COUNTS_ALLOWED: tuple[int, ...] = (1, 2, 3)
MIN_WALL_THICKNESS_MM: float = 4.0
MAX_WALL_THICKNESS_MM: float = 24.0
MIN_DESIGN_PRESSURE_MPA: float = 0.25
MAX_DESIGN_PRESSURE_MPA: float = 4.50

FILL_FRACTION: float = 0.95
AMBIENT_TEMP_K: float = 293.15
BASE_ALLOWABLE_STRESS_MPA: float = 235.0  # Cryo-strengthened 12X18H10T / AISI 321 design stress at cryo service

SIM_BUDGET_DEFAULT: int = 3
MAX_EPISODE_STEPS: int = 42
DISRUPTION_TRIGGER_BOOKINGS: int = 2
DISRUPTION_TRIGGER_MUTATIONS: int = 4
BUDGET_BONUS_HEADROOM_FRAC: float = 0.03

POST_SHOCK_FIRST_REVISION_FEE_EUR: float = 1500.0
STANDARD_REVISION_FEE_EUR: float = 4500.0
ASSEMBLY_COMMISSION_WEEKS: int = 1


# Exact physical & hazard properties per working cryogenic medium
MEDIUM_PROPS: dict[Medium, dict[str, Any]] = {
    Medium.LIN: {
        "label": "Liquid Nitrogen (LIN)",
        "density_kg_m3": 808.0,
        "boil_temp_k": 77.36,
        "latent_heat_kj_kg": 199.0,
        "safety_factor": 1.00,
        "requires_evi": False,
        "forbidden_fittings": (),
        "lox_high_pressure_rule": False,
    },
    Medium.LOX: {
        "label": "Liquid Oxygen (LOX)",
        "density_kg_m3": 1141.0,
        "boil_temp_k": 90.19,
        "latent_heat_kj_kg": 213.0,
        "safety_factor": 1.25,
        "requires_evi": False,
        "forbidden_fittings": (),
        "lox_high_pressure_rule": True,  # If design_pressure_mpa > 2.0, CN fittings are forbidden (oxygen fire hazard)
    },
    Medium.LAR: {
        "label": "Liquid Argon (LAR)",
        "density_kg_m3": 1395.0,
        "boil_temp_k": 87.30,
        "latent_heat_kj_kg": 161.0,
        "safety_factor": 1.05,
        "requires_evi": False,
        "forbidden_fittings": (),
        "lox_high_pressure_rule": False,
    },
    Medium.LCO2: {
        "label": "Liquid Carbon Dioxide (LCO2)",
        "density_kg_m3": 1030.0,
        "boil_temp_k": 243.15,
        "latent_heat_kj_kg": 302.0,
        "safety_factor": 1.00,
        "requires_evi": False,
        "forbidden_fittings": (),
        "lox_high_pressure_rule": False,
    },
    Medium.LNG: {
        "label": "Liquefied Natural Gas (LNG)",
        "density_kg_m3": 430.0,
        "boil_temp_k": 111.65,
        "latent_heat_kj_kg": 510.0,
        "safety_factor": 1.15,
        "requires_evi": False,
        "forbidden_fittings": (),
        "lox_high_pressure_rule": False,
    },
    Medium.LH2: {
        "label": "Liquid Hydrogen (LH2)",
        "density_kg_m3": 70.85,
        "boil_temp_k": 20.28,
        "latent_heat_kj_kg": 446.0,
        "safety_factor": 1.25,
        "requires_evi": True,
        "forbidden_fittings": (FittingsGrade.CN,),
        "lox_high_pressure_rule": False,
    },
    Medium.LHe: {
        "label": "Liquid Helium (LHe)",
        "density_kg_m3": 124.9,
        "boil_temp_k": 4.22,
        "latent_heat_kj_kg": 20.7,
        "safety_factor": 1.25,
        "requires_evi": True,
        "forbidden_fittings": (FittingsGrade.CN,),
        "lox_high_pressure_rule": False,
    },
}


INSULATION_PROPS: dict[InsulationType, dict[str, Any]] = {
    InsulationType.EVI: {
        "label": "Multi-Layer Screen-Vacuum (EVI / ЭВИ)",
        "annulus_mm": 120.0,
        "k_eff_w_mk": 0.00028,
        "casing_mass_factor": 1.00,
        "cost_multiplier": 1.18,
    },
    InsulationType.PVI: {
        "label": "Perlite-Vacuum Powder (PVI / ПВИ)",
        "annulus_mm": 200.0,
        "k_eff_w_mk": 0.00185,
        "casing_mass_factor": 1.14,
        "cost_multiplier": 1.00,
    },
}


FITTINGS_PROPS: dict[FittingsGrade, dict[str, Any]] = {
    FittingsGrade.CN: {
        "label": "Certified Standard (CN)",
        "weld_efficiency": 0.85,
        "valve_heat_leak_w": 18.0,
        "base_lead_weeks": 2,
        "cost_multiplier": 1.00,
        "fixed_kit_eur": 2200.0,
    },
    FittingsGrade.EU_STD: {
        "label": "European Cryo Standard (EU_STD)",
        "weld_efficiency": 0.90,
        "valve_heat_leak_w": 11.0,
        "base_lead_weeks": 4,
        "cost_multiplier": 1.12,
        "fixed_kit_eur": 4800.0,
    },
    FittingsGrade.DE_PRECISION: {
        "label": "German Precision Cryo (DE_PRECISION)",
        "weld_efficiency": 0.96,
        "valve_heat_leak_w": 5.5,
        "base_lead_weeks": 6,
        "cost_multiplier": 1.32,
        "fixed_kit_eur": 8500.0,
    },
}


HANDBOOK_SECTIONS: tuple[str, ...] = (
    "overview",
    "physics",
    "materials_and_insulation",
    "transport_and_routes",
    "manufacturing_and_bays",
    "disruptions_and_revisions",
    "scoring",
    "all",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VesselDesignSpec(StrictModel):
    unit_count: int = Field(..., ge=1, le=3)
    unit_volume_m3: float = Field(..., gt=0.0)
    design_pressure_mpa: float = Field(..., ge=MIN_DESIGN_PRESSURE_MPA, le=MAX_DESIGN_PRESSURE_MPA)
    wall_thickness_mm: float = Field(..., ge=MIN_WALL_THICKNESS_MM, le=MAX_WALL_THICKNESS_MM)
    insulation: InsulationType
    fittings: FittingsGrade


class DimensionsMM(StrictModel):
    outer_diameter_mm: int
    inner_diameter_mm: int
    shell_axis_mm: int
    installed_height_mm: int
    installed_width_mm: int
    installed_length_mm: int
    shipping_height_mm: int
    shipping_width_mm: int
    shipping_length_mm: int


class PhysicsEvaluation(StrictModel):
    usable_unit_mass_kg: float
    usable_total_mass_kg: float
    required_product_mass_kg: float
    mass_pass: bool

    hoop_stress_mpa: float
    allowable_stress_mpa: float
    stress_pass: bool
    pressure_pass: bool

    boil_off_pct_day: float
    max_boil_off_pct_day: float
    boil_off_pass: bool

    material_compat_pass: bool
    material_notes: list[str] = Field(default_factory=list)

    unit_tare_mass_kg: float
    unit_transport_mass_kg: float
    dimensions: DimensionsMM
    site_clearance_pass: bool

    fab_weeks: int
    fittings_lead_weeks: int
    manufacturing_cost_eur: float

    all_engineering_pass: bool
    violations: list[str] = Field(default_factory=list)


class CustomerOrder(StrictModel):
    order_id: str
    client_name: str
    medium: Medium
    execution: ExecutionType
    required_product_mass_kg: float
    min_pressure_mpa: float
    max_boil_off_pct_day: float
    sla_week: int
    route_id: str
    max_site_height_mm: int
    max_site_footprint_mm: int
    priority_weight: float = 0.25
    notes: str = ""


class TransportRoute(StrictModel):
    route_id: str
    name: str
    transit_weeks: int
    std_max_width_mm: int = 2500
    std_max_height_mm: int = 3800
    permit_max_width_mm: int = 3100
    permit_max_height_mm: int = 4350
    escort_permit_allowed: bool = True
    max_unit_transport_mass_kg: float = 16500.0
    base_freight_eur_per_unit: float = 2800.0
    permit_fee_eur: float = 4200.0
    notes: str = ""


class BayBlackoutWindow(StrictModel):
    bay_id: BayId
    start_week: int
    end_week: int  # exclusive [start_week, end_week)
    reason: str


class DisruptionEvent(StrictModel):
    disruption_id: str
    title: str
    headline: str
    details: str
    fittings_lead_deltas: dict[FittingsGrade, int] = Field(default_factory=dict)
    pvi_k_multiplier: float = 1.0
    budget_delta_eur: float = 0.0
    escort_permits_delta: int = 0
    bay_blackouts: list[BayBlackoutWindow] = Field(default_factory=list)
    bay_max_unit_tare_kg: dict[BayId, float] = Field(default_factory=dict)
    route_mass_cap_overrides: dict[str, float] = Field(default_factory=dict)
    route_std_height_overrides: dict[str, int] = Field(default_factory=dict)
    route_permit_allowed_overrides: dict[str, bool] = Field(default_factory=dict)
    order_min_pressure_overrides: dict[str, float] = Field(default_factory=dict)
    order_sla_overrides: dict[str, int] = Field(default_factory=dict)


class OrderBooking(StrictModel):
    order_id: str
    bay_id: BayId
    bay_start_week: int
    fab_weeks: int
    fab_end_week: int
    booked_at_step: int
    order_release_week: int = 1
    spec: VesselDesignSpec
    manufacturing_cost_eur: float
    assigned_route_id: str | None = None
    use_escort_permit: bool = False
    revision_count: int = 0
    post_shock_revised: bool = False


class OrderVerificationResult(StrictModel):
    order_id: str
    booked: bool
    route_assigned: bool
    passed_all_hard_checks: bool
    order_raw_score: float
    weighted_contribution: float
    delivery_week: int | None = None
    sla_week: int
    total_order_cost_eur: float = 0.0
    physics: PhysicsEvaluation | None = None
    violations: list[str] = Field(default_factory=list)


class VerificationReport(StrictModel):
    seed: str
    finalized: bool
    final_score: float
    valid_order_count: int
    total_order_count: int
    batch_gate_cap_applied: bool
    budget_exceeded: bool
    permit_quota_exceeded: bool
    total_budget_eur: float
    total_spent_eur: float
    budget_efficiency_bonus: float
    schedule_margin_bonus: float
    sim_efficiency_bonus: float
    sim_calls_used: int
    sim_budget_total: int
    step_count: int
    disruption_triggered: bool
    order_results: list[OrderVerificationResult]
    failure_modes: list[str]
    primary_failure_categories: list[str]


class TrajectoryStepLog(StrictModel):
    step_index: int
    timestamp_iso: str
    tool_name: str
    arguments: dict[str, Any]
    observation: dict[str, Any]
    disruption_triggered_this_step: bool = False


class TrajectoryLog(StrictModel):
    run_id: str
    seed: str
    agent_name: str
    agent_type: Literal["reference", "frontier", "do_nothing", "stop_early", "small_before", "small_after", "exploit"]
    model_id: str
    started_at: str
    completed_at: str
    steps: list[TrajectoryStepLog]
    verification: VerificationReport
    state_hmac: str = ""
