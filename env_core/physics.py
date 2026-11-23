"""Deterministic cryogenic physics, geometry, stress, thermal boil-off, and cost model.

Adapted and extended from industrial cryogenic vessel rules (GOST 34347 / ASME VIII)
so every equation is smooth, physically grounded, and 100% documented in handbook.py.
"""

from __future__ import annotations

import math
from env_core.contracts import (
    AMBIENT_TEMP_K,
    BASE_ALLOWABLE_STRESS_MPA,
    CustomerOrder,
    DimensionsMM,
    ExecutionType,
    FILL_FRACTION,
    FITTINGS_PROPS,
    FittingsGrade,
    INSULATION_PROPS,
    InsulationType,
    MAX_DESIGN_PRESSURE_MPA,
    MAX_WALL_THICKNESS_MM,
    MEDIUM_PROPS,
    MIN_DESIGN_PRESSURE_MPA,
    MIN_WALL_THICKNESS_MM,
    PhysicsEvaluation,
    STANDARD_VOLUMES_M3,
    UNIT_COUNTS_ALLOWED,
    VesselDesignSpec,
)


def nominal_outer_diameter_mm(unit_volume_m3: float) -> int:
    """Nominal outer vacuum-jacket diameter (mm) for standard shell volumes."""
    if unit_volume_m3 <= 2.0:
        return 1400
    if unit_volume_m3 <= 5.0:
        return 1600
    if unit_volume_m3 <= 10.0:
        return 2000
    if unit_volume_m3 <= 15.0:
        return 2400
    if unit_volume_m3 <= 25.0:
        return 2600
    return 3000


def compute_dimensions(
    unit_volume_m3: float,
    execution: ExecutionType,
    insulation: InsulationType,
) -> DimensionsMM:
    """Compute exact installed and shipping envelope dimensions (mm) per unit."""
    outer_d_mm = nominal_outer_diameter_mm(unit_volume_m3)
    annulus_mm = INSULATION_PROPS[insulation]["annulus_mm"]
    inner_d_mm = int(round(outer_d_mm - 2.0 * annulus_mm))
    d_inner_m = inner_d_mm / 1000.0

    # Cylindrical equivalent axis length rounded up to nearest 10 mm
    raw_axis_mm = (unit_volume_m3 / (math.pi * (d_inner_m / 2.0) ** 2)) * 1000.0
    shell_axis_mm = int(math.ceil(raw_axis_mm / 10.0) * 10)

    if execution == ExecutionType.STATIONARY:
        installed_height_mm = shell_axis_mm + 900
        installed_width_mm = outer_d_mm
        installed_length_mm = outer_d_mm
        # Shipped horizontally on temporary transport saddles, erected vertically on site
        shipping_height_mm = outer_d_mm + 350
        shipping_width_mm = outer_d_mm
        shipping_length_mm = shell_axis_mm + 900
    else:
        # Transportable horizontal CTK vessel permanently mounted on heavy road skid
        installed_height_mm = outer_d_mm + 600
        installed_width_mm = outer_d_mm + 200
        installed_length_mm = shell_axis_mm + 1200
        shipping_height_mm = installed_height_mm
        shipping_width_mm = installed_width_mm
        shipping_length_mm = installed_length_mm

    return DimensionsMM(
        outer_diameter_mm=outer_d_mm,
        inner_diameter_mm=inner_d_mm,
        shell_axis_mm=shell_axis_mm,
        installed_height_mm=installed_height_mm,
        installed_width_mm=installed_width_mm,
        installed_length_mm=installed_length_mm,
        shipping_height_mm=shipping_height_mm,
        shipping_width_mm=shipping_width_mm,
        shipping_length_mm=shipping_length_mm,
    )


def validate_catalogue_bounds(spec: VesselDesignSpec) -> list[str]:
    """Validate that discrete shell volumes and engineering bounds match factory catalogue."""
    errors: list[str] = []
    if spec.unit_count not in UNIT_COUNTS_ALLOWED:
        errors.append(f"unit_count={spec.unit_count} not in allowed {UNIT_COUNTS_ALLOWED}")
    if all(abs(spec.unit_volume_m3 - v) > 1e-6 for v in STANDARD_VOLUMES_M3):
        errors.append(
            f"unit_volume_m3={spec.unit_volume_m3} is not a standard shell volume {list(STANDARD_VOLUMES_M3)}"
        )
    if not (math.isfinite(spec.design_pressure_mpa) and MIN_DESIGN_PRESSURE_MPA <= spec.design_pressure_mpa <= MAX_DESIGN_PRESSURE_MPA):
        errors.append(
            f"design_pressure_mpa={spec.design_pressure_mpa} outside [{MIN_DESIGN_PRESSURE_MPA}, {MAX_DESIGN_PRESSURE_MPA}]"
        )
    if not (math.isfinite(spec.wall_thickness_mm) and MIN_WALL_THICKNESS_MM <= spec.wall_thickness_mm <= MAX_WALL_THICKNESS_MM):
        errors.append(
            f"wall_thickness_mm={spec.wall_thickness_mm} outside [{MIN_WALL_THICKNESS_MM}, {MAX_WALL_THICKNESS_MM}]"
        )
    return errors


def compute_fab_and_cost(
    spec: VesselDesignSpec,
    execution: ExecutionType,
    unit_tare_mass_kg: float,
    fittings_lead_delta: int = 0,
) -> tuple[int, int, float]:
    """Return (fab_weeks, fittings_lead_weeks, manufacturing_cost_eur) without FEA checks."""
    base_fab = (
        2
        + (1 if spec.unit_volume_m3 >= 10.0 else 0)
        + (1 if spec.unit_volume_m3 >= 20.0 else 0)
        + (1 if spec.wall_thickness_mm >= 14.0 else 0)
        + (1 if execution == ExecutionType.TRANSPORTABLE else 0)
    )
    fab_weeks = base_fab + (spec.unit_count - 1)

    fittings_lead_weeks = max(
        1, int(FITTINGS_PROPS[spec.fittings]["base_lead_weeks"] + fittings_lead_delta)
    )

    ins_mult = INSULATION_PROPS[spec.insulation]["cost_multiplier"]
    fit_mult = FITTINGS_PROPS[spec.fittings]["cost_multiplier"]
    exec_mult = 1.18 if execution == ExecutionType.TRANSPORTABLE else 1.00
    fixed_kit = FITTINGS_PROPS[spec.fittings]["fixed_kit_eur"]

    unit_cost = (
        (9500.0 + spec.unit_volume_m3 * 2150.0 + unit_tare_mass_kg * 3.10)
        * ins_mult
        * fit_mult
        * exec_mult
        + fixed_kit
    )
    series_factor = 0.94 if spec.unit_count > 1 else 1.00
    manufacturing_cost_eur = round(spec.unit_count * unit_cost * series_factor, 2)
    return fab_weeks, fittings_lead_weeks, manufacturing_cost_eur


def compute_masses_and_area(
    spec: VesselDesignSpec,
    execution: ExecutionType,
    dims: DimensionsMM,
) -> tuple[float, float, float]:
    """Return (inner_surface_area_m2, unit_tare_mass_kg, unit_transport_mass_kg)."""
    r_m = (dims.inner_diameter_mm / 1000.0) / 2.0
    l_m = dims.shell_axis_mm / 1000.0
    area_m2 = 2.0 * math.pi * r_m * l_m + 4.0 * math.pi * (r_m ** 2) * 0.65

    inner_shell_vol_m3 = area_m2 * (spec.wall_thickness_mm / 1000.0) * 1.12
    inner_shell_mass_kg = inner_shell_vol_m3 * 7900.0
    outer_jacket_mass_kg = (
        area_m2 * 0.0055 * 7850.0 * INSULATION_PROPS[spec.insulation]["casing_mass_factor"]
    )
    unit_tare_mass_kg = round(inner_shell_mass_kg + outer_jacket_mass_kg + 420.0, 1)

    if execution == ExecutionType.TRANSPORTABLE:
        skid_or_cradle_kg = round(0.18 * unit_tare_mass_kg + 650.0, 1)
    else:
        skid_or_cradle_kg = 480.0

    unit_transport_mass_kg = round(unit_tare_mass_kg + skid_or_cradle_kg, 1)
    return area_m2, unit_tare_mass_kg, unit_transport_mass_kg


def evaluate_design_physics(
    order: CustomerOrder,
    spec: VesselDesignSpec,
    *,
    pvi_k_multiplier: float = 1.0,
    fittings_lead_delta: int = 0,
) -> PhysicsEvaluation:
    """Run full deterministic mechanical, thermal, compatibility, and clearance checks."""
    violations = validate_catalogue_bounds(spec)
    mprops = MEDIUM_PROPS[order.medium]
    dims = compute_dimensions(spec.unit_volume_m3, order.execution, spec.insulation)
    area_m2, unit_tare_kg, unit_transport_kg = compute_masses_and_area(
        spec, order.execution, dims
    )

    # 1. Deliverable Cryogenic Mass (95% max liquid fill)
    usable_unit_mass_kg = round(spec.unit_volume_m3 * mprops["density_kg_m3"] * FILL_FRACTION, 2)
    usable_total_mass_kg = round(spec.unit_count * usable_unit_mass_kg, 2)
    mass_pass = usable_total_mass_kg >= order.required_product_mass_kg
    if not mass_pass:
        violations.append(
            f"INSUFFICIENT_CAPACITY: delivered {usable_total_mass_kg} kg < required {order.required_product_mass_kg} kg"
        )

    # 2. Pressure & Hoop Stress
    pressure_pass = spec.design_pressure_mpa >= order.min_pressure_mpa - 1e-9
    if not pressure_pass:
        violations.append(
            f"UNDER_PRESSURE: design_pressure_mpa={spec.design_pressure_mpa} < min_pressure_mpa={order.min_pressure_mpa}"
        )

    weld_eff = FITTINGS_PROPS[spec.fittings]["weld_efficiency"]
    hoop_stress_mpa = round(
        (spec.design_pressure_mpa * (dims.inner_diameter_mm + spec.wall_thickness_mm))
        / (2.0 * weld_eff * spec.wall_thickness_mm),
        2,
    )
    allowable_stress_mpa = round(BASE_ALLOWABLE_STRESS_MPA / mprops["safety_factor"], 2)
    stress_pass = hoop_stress_mpa <= allowable_stress_mpa
    if not stress_pass:
        violations.append(
            f"HOOP_STRESS_EXCEEDED: {hoop_stress_mpa} MPa > allowable {allowable_stress_mpa} MPa"
        )

    # 3. Cryogenic Material & Hazard Compatibility
    material_compat_pass = True
    material_notes: list[str] = []
    if mprops["requires_evi"] and spec.insulation != InsulationType.EVI:
        material_compat_pass = False
        msg = f"HAZARD_INSULATION: {order.medium.value} strictly requires EVI insulation"
        material_notes.append(msg)
        violations.append(msg)
    if spec.fittings in mprops["forbidden_fittings"]:
        material_compat_pass = False
        msg = f"HAZARD_FITTINGS: {order.medium.value} forbids {spec.fittings.value} fittings"
        material_notes.append(msg)
        violations.append(msg)
    if mprops["lox_high_pressure_rule"] and spec.design_pressure_mpa > 2.0 and spec.fittings == FittingsGrade.CN:
        material_compat_pass = False
        msg = "LOX_IGNITION_RISK: LOX above 2.0 MPa requires EU_STD or DE_PRECISION oxygen-clean fittings"
        material_notes.append(msg)
        violations.append(msg)

    # 4. Daily Thermal Boil-Off Rate (%/day)
    k_base = INSULATION_PROPS[spec.insulation]["k_eff_w_mk"]
    k_eff = k_base * (pvi_k_multiplier if spec.insulation == InsulationType.PVI else 1.0)
    t_ins_m = INSULATION_PROPS[spec.insulation]["annulus_mm"] / 1000.0
    delta_t = AMBIENT_TEMP_K - mprops["boil_temp_k"]

    q_cond_w = k_eff * area_m2 * delta_t / t_ins_m
    q_valve_w = FITTINGS_PROPS[spec.fittings]["valve_heat_leak_w"] * (
        1.25 if order.execution == ExecutionType.TRANSPORTABLE else 1.00
    )
    q_day_j = (q_cond_w + q_valve_w) * 86400.0
    evap_kg_day = q_day_j / (mprops["latent_heat_kj_kg"] * 1000.0)
    boil_off_pct_day = round((evap_kg_day / usable_unit_mass_kg) * 100.0, 4)
    boil_off_pass = boil_off_pct_day <= order.max_boil_off_pct_day
    if not boil_off_pass:
        violations.append(
            f"BOIL_OFF_EXCEEDED: {boil_off_pct_day}%/day > max {order.max_boil_off_pct_day}%/day"
        )

    # 5. Installed Site Clearance (height + modular bank footprint width with 400mm inter-unit spacing)
    bank_footprint_mm = spec.unit_count * dims.installed_width_mm + (spec.unit_count - 1) * 400
    height_ok = dims.installed_height_mm <= order.max_site_height_mm
    footprint_ok = bank_footprint_mm <= order.max_site_footprint_mm
    site_clearance_pass = height_ok and footprint_ok
    if not height_ok:
        violations.append(
            f"SITE_HEIGHT_EXCEEDED: installed {dims.installed_height_mm} mm > site max {order.max_site_height_mm} mm"
        )
    if not footprint_ok:
        violations.append(
            f"SITE_FOOTPRINT_EXCEEDED: bank width {bank_footprint_mm} mm > site max {order.max_site_footprint_mm} mm"
        )

    fab_weeks, fittings_lead_weeks, manufacturing_cost_eur = compute_fab_and_cost(
        spec, order.execution, unit_tare_kg, fittings_lead_delta=fittings_lead_delta
    )

    all_pass = (
        len(violations) == 0
        and mass_pass
        and pressure_pass
        and stress_pass
        and material_compat_pass
        and boil_off_pass
        and site_clearance_pass
    )

    return PhysicsEvaluation(
        usable_unit_mass_kg=usable_unit_mass_kg,
        usable_total_mass_kg=usable_total_mass_kg,
        required_product_mass_kg=order.required_product_mass_kg,
        mass_pass=mass_pass,
        hoop_stress_mpa=hoop_stress_mpa,
        allowable_stress_mpa=allowable_stress_mpa,
        stress_pass=stress_pass,
        pressure_pass=pressure_pass,
        boil_off_pct_day=boil_off_pct_day,
        max_boil_off_pct_day=order.max_boil_off_pct_day,
        boil_off_pass=boil_off_pass,
        material_compat_pass=material_compat_pass,
        material_notes=material_notes,
        unit_tare_mass_kg=unit_tare_kg,
        unit_transport_mass_kg=unit_transport_kg,
        dimensions=dims,
        site_clearance_pass=site_clearance_pass,
        fab_weeks=fab_weeks,
        fittings_lead_weeks=fittings_lead_weeks,
        manufacturing_cost_eur=manufacturing_cost_eur,
        all_engineering_pass=all_pass,
        violations=violations,
    )
