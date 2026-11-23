import type {
  CustomerOrder,
  DimensionsMM,
  ExecutionType,
  FittingsGrade,
  InsulationType,
  Medium,
  PhysicsEvaluation,
  VesselDesignSpec,
} from "../types/cryoforge";

export const STANDARD_VOLUMES_M3 = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30] as const;
export const FILL_FRACTION = 0.95;
export const AMBIENT_TEMP_K = 293.15;
export const BASE_ALLOWABLE_STRESS_MPA = 235.0;
export const SIM_BUDGET_DEFAULT = 3;
export const MAX_EPISODE_STEPS = 42;
export const ASSEMBLY_COMMISSION_WEEKS = 1;

export const MEDIUM_PROPS: Record<
  Medium,
  {
    label: string;
    shortLabel: string;
    density_kg_m3: number;
    boil_temp_k: number;
    boil_temp_c: number;
    latent_heat_kj_kg: number;
    safety_factor: number;
    requires_evi: boolean;
    forbidden_fittings: FittingsGrade[];
    lox_high_pressure_rule: boolean;
    liquidColor: string;
    hazardClass: string;
  }
> = {
  LIN: {
    label: "Liquid Nitrogen (LIN)",
    shortLabel: "LIN · -196°C",
    density_kg_m3: 808.0,
    boil_temp_k: 77.36,
    boil_temp_c: -196,
    latent_heat_kj_kg: 199.0,
    safety_factor: 1.0,
    requires_evi: false,
    forbidden_fittings: [],
    lox_high_pressure_rule: false,
    liquidColor: "#38bdf8",
    hazardClass: "2.2 CRYO INERT",
  },
  LOX: {
    label: "Liquid Oxygen (LOX)",
    shortLabel: "LOX · -183°C",
    density_kg_m3: 1141.0,
    boil_temp_k: 90.19,
    boil_temp_c: -183,
    latent_heat_kj_kg: 213.0,
    safety_factor: 1.25,
    requires_evi: false,
    forbidden_fittings: [],
    lox_high_pressure_rule: true,
    liquidColor: "#0ea5e9",
    hazardClass: "2.2 + 5.1 OXIDIZER",
  },
  LAR: {
    label: "Liquid Argon (LAR)",
    shortLabel: "LAR · -186°C",
    density_kg_m3: 1395.0,
    boil_temp_k: 87.3,
    boil_temp_c: -186,
    latent_heat_kj_kg: 161.0,
    safety_factor: 1.05,
    requires_evi: false,
    forbidden_fittings: [],
    lox_high_pressure_rule: false,
    liquidColor: "#818cf8",
    hazardClass: "2.2 HEAVY CRYO",
  },
  LCO2: {
    label: "Liquid Carbon Dioxide (LCO2)",
    shortLabel: "LCO2 · -30°C",
    density_kg_m3: 1030.0,
    boil_temp_k: 243.15,
    boil_temp_c: -30,
    latent_heat_kj_kg: 302.0,
    safety_factor: 1.0,
    requires_evi: false,
    forbidden_fittings: [],
    lox_high_pressure_rule: false,
    liquidColor: "#64748b",
    hazardClass: "2.2 HIGH-P LIQ",
  },
  LNG: {
    label: "Liquefied Natural Gas (LNG)",
    shortLabel: "LNG · -162°C",
    density_kg_m3: 430.0,
    boil_temp_k: 111.65,
    boil_temp_c: -162,
    latent_heat_kj_kg: 510.0,
    safety_factor: 1.15,
    requires_evi: false,
    forbidden_fittings: [],
    lox_high_pressure_rule: false,
    liquidColor: "#14b8a6",
    hazardClass: "2.1 FLAMMABLE",
  },
  LH2: {
    label: "Liquid Hydrogen (LH2)",
    shortLabel: "LH2 · -253°C",
    density_kg_m3: 70.85,
    boil_temp_k: 20.28,
    boil_temp_c: -253,
    latent_heat_kj_kg: 446.0,
    safety_factor: 1.25,
    requires_evi: true,
    forbidden_fittings: ["CN"],
    lox_high_pressure_rule: false,
    liquidColor: "#f472b6",
    hazardClass: "2.1 EXTREME H2",
  },
  LHe: {
    label: "Liquid Helium (LHe)",
    shortLabel: "LHe · -269°C",
    density_kg_m3: 124.9,
    boil_temp_k: 4.22,
    boil_temp_c: -269,
    latent_heat_kj_kg: 20.7,
    safety_factor: 1.25,
    requires_evi: true,
    forbidden_fittings: ["CN"],
    lox_high_pressure_rule: false,
    liquidColor: "#c084fc",
    hazardClass: "2.2 DEEP 4.2K",
  },
};

export const INSULATION_PROPS: Record<
  InsulationType,
  {
    label: string;
    shortCode: string;
    annulus_mm: number;
    k_eff_w_mk: number;
    casing_mass_factor: number;
    cost_multiplier: number;
  }
> = {
  EVI: {
    label: "Multi-Layer Screen-Vacuum (EVI / ЭВИ)",
    shortCode: "EVI (ЭВИ 120mm)",
    annulus_mm: 120.0,
    k_eff_w_mk: 0.00028,
    casing_mass_factor: 1.0,
    cost_multiplier: 1.18,
  },
  PVI: {
    label: "Perlite-Vacuum Powder (PVI / ПВИ)",
    shortCode: "PVI (ПВИ 200mm)",
    annulus_mm: 200.0,
    k_eff_w_mk: 0.00185,
    casing_mass_factor: 1.14,
    cost_multiplier: 1.0,
  },
};

export const FITTINGS_PROPS: Record<
  FittingsGrade,
  {
    label: string;
    weld_efficiency: number;
    valve_heat_leak_w: number;
    base_lead_weeks: number;
    cost_multiplier: number;
    fixed_kit_eur: number;
  }
> = {
  CN: {
    label: "Certified Standard (CN)",
    weld_efficiency: 0.85,
    valve_heat_leak_w: 18.0,
    base_lead_weeks: 2,
    cost_multiplier: 1.0,
    fixed_kit_eur: 2200.0,
  },
  EU_STD: {
    label: "European Cryo Standard (EU_STD)",
    weld_efficiency: 0.9,
    valve_heat_leak_w: 11.0,
    base_lead_weeks: 4,
    cost_multiplier: 1.12,
    fixed_kit_eur: 4800.0,
  },
  DE_PRECISION: {
    label: "German Precision Cryo (DE_PRECISION)",
    weld_efficiency: 0.96,
    valve_heat_leak_w: 5.5,
    base_lead_weeks: 6,
    cost_multiplier: 1.32,
    fixed_kit_eur: 8500.0,
  },
};

const round1 = (n: number) => Math.round(n * 10) / 10;
const round2 = (n: number) => Math.round(n * 100) / 100;
const round4 = (n: number) => Math.round(n * 10000) / 10000;

export function nominalOuterDiameterMm(unitVolumeM3: number): number {
  if (unitVolumeM3 <= 2.0) return 1400;
  if (unitVolumeM3 <= 5.0) return 1600;
  if (unitVolumeM3 <= 10.0) return 2000;
  if (unitVolumeM3 <= 15.0) return 2400;
  if (unitVolumeM3 <= 25.0) return 2600;
  return 3000;
}

export function computeDimensions(
  unitVolumeM3: number,
  execution: ExecutionType,
  insulation: InsulationType
): DimensionsMM {
  const outer_d_mm = nominalOuterDiameterMm(unitVolumeM3);
  const annulus_mm = INSULATION_PROPS[insulation].annulus_mm;
  const inner_d_mm = Math.round(outer_d_mm - 2.0 * annulus_mm);
  const d_inner_m = inner_d_mm / 1000.0;

  const raw_axis_mm = (unitVolumeM3 / (Math.PI * (d_inner_m / 2.0) ** 2)) * 1000.0;
  const shell_axis_mm = Math.ceil(raw_axis_mm / 10.0) * 10;

  if (execution === "STATIONARY") {
    return {
      outer_diameter_mm: outer_d_mm,
      inner_diameter_mm: inner_d_mm,
      shell_axis_mm,
      installed_height_mm: shell_axis_mm + 900,
      installed_width_mm: outer_d_mm,
      installed_length_mm: outer_d_mm,
      shipping_height_mm: outer_d_mm + 350,
      shipping_width_mm: outer_d_mm,
      shipping_length_mm: shell_axis_mm + 900,
    };
  }

  const installed_height_mm = outer_d_mm + 600;
  const installed_width_mm = outer_d_mm + 200;
  const installed_length_mm = shell_axis_mm + 1200;
  return {
    outer_diameter_mm: outer_d_mm,
    inner_diameter_mm: inner_d_mm,
    shell_axis_mm,
    installed_height_mm,
    installed_width_mm,
    installed_length_mm,
    shipping_height_mm: installed_height_mm,
    shipping_width_mm: installed_width_mm,
    shipping_length_mm: installed_length_mm,
  };
}

export function evaluateDesignPhysics(
  order: CustomerOrder,
  spec: VesselDesignSpec,
  opts?: {
    pvi_k_multiplier?: number;
    fittings_lead_delta?: number;
  }
): PhysicsEvaluation {
  const pvi_k_multiplier = opts?.pvi_k_multiplier ?? 1.0;
  const fittings_lead_delta = opts?.fittings_lead_delta ?? 0;

  const violations: string[] = [];
  if (![1, 2, 3].includes(spec.unit_count)) {
    violations.push(`unit_count=${spec.unit_count} not in allowed (1, 2, 3)`);
  }
  if (STANDARD_VOLUMES_M3.every((v) => Math.abs(spec.unit_volume_m3 - v) > 1e-6)) {
    violations.push(`unit_volume_m3=${spec.unit_volume_m3} is not a standard shell volume`);
  }
  if (!(Number.isFinite(spec.design_pressure_mpa) && spec.design_pressure_mpa >= 0.25 && spec.design_pressure_mpa <= 4.5)) {
    violations.push(`design_pressure_mpa=${spec.design_pressure_mpa} outside [0.25, 4.5]`);
  }
  if (!(Number.isFinite(spec.wall_thickness_mm) && spec.wall_thickness_mm >= 4.0 && spec.wall_thickness_mm <= 24.0)) {
    violations.push(`wall_thickness_mm=${spec.wall_thickness_mm} outside [4.0, 24.0]`);
  }

  const mprops = MEDIUM_PROPS[order.medium] ?? MEDIUM_PROPS.LIN;
  const dims = computeDimensions(spec.unit_volume_m3, order.execution, spec.insulation);

  const r_m = dims.inner_diameter_mm / 1000.0 / 2.0;
  const l_m = dims.shell_axis_mm / 1000.0;
  const area_m2 = 2.0 * Math.PI * r_m * l_m + 4.0 * Math.PI * r_m ** 2 * 0.65;

  const inner_shell_vol_m3 = area_m2 * (spec.wall_thickness_mm / 1000.0) * 1.12;
  const inner_shell_mass_kg = inner_shell_vol_m3 * 7900.0;
  const outer_jacket_mass_kg =
    area_m2 * 0.0055 * 7850.0 * INSULATION_PROPS[spec.insulation].casing_mass_factor;
  const unit_tare_mass_kg = round1(inner_shell_mass_kg + outer_jacket_mass_kg + 420.0);
  const skid_or_cradle_kg =
    order.execution === "TRANSPORTABLE" ? round1(0.18 * unit_tare_mass_kg + 650.0) : 480.0;
  const unit_transport_mass_kg = round1(unit_tare_mass_kg + skid_or_cradle_kg);

  // 1. Deliverable Cryogenic Mass
  const usable_unit_mass_kg = round2(spec.unit_volume_m3 * mprops.density_kg_m3 * FILL_FRACTION);
  const usable_total_mass_kg = round2(spec.unit_count * usable_unit_mass_kg);
  const mass_pass = usable_total_mass_kg >= order.required_product_mass_kg;
  if (!mass_pass) {
    violations.push(
      `INSUFFICIENT_CAPACITY: delivered ${usable_total_mass_kg} kg < required ${order.required_product_mass_kg} kg`
    );
  }

  // 2. Pressure & Hoop Stress
  const pressure_pass = spec.design_pressure_mpa >= order.min_pressure_mpa - 1e-9;
  if (!pressure_pass) {
    violations.push(
      `UNDER_PRESSURE: design_pressure_mpa=${spec.design_pressure_mpa} < min_pressure_mpa=${order.min_pressure_mpa}`
    );
  }

  const weld_eff = FITTINGS_PROPS[spec.fittings].weld_efficiency;
  const hoop_stress_mpa = round2(
    (spec.design_pressure_mpa * (dims.inner_diameter_mm + spec.wall_thickness_mm)) /
      (2.0 * weld_eff * spec.wall_thickness_mm)
  );
  const allowable_stress_mpa = round2(BASE_ALLOWABLE_STRESS_MPA / mprops.safety_factor);
  const stress_pass = hoop_stress_mpa <= allowable_stress_mpa;
  if (!stress_pass) {
    violations.push(
      `HOOP_STRESS_EXCEEDED: ${hoop_stress_mpa} MPa > allowable ${allowable_stress_mpa} MPa`
    );
  }

  // 3. Cryogenic Material & Hazard Compatibility
  let material_compat_pass = true;
  const material_notes: string[] = [];
  if (mprops.requires_evi && spec.insulation !== "EVI") {
    material_compat_pass = false;
    const msg = `HAZARD_INSULATION: ${order.medium} strictly requires EVI insulation`;
    material_notes.push(msg);
    violations.push(msg);
  }
  if (mprops.forbidden_fittings.includes(spec.fittings)) {
    material_compat_pass = false;
    const msg = `HAZARD_FITTINGS: ${order.medium} forbids ${spec.fittings} fittings`;
    material_notes.push(msg);
    violations.push(msg);
  }
  if (mprops.lox_high_pressure_rule && spec.design_pressure_mpa > 2.0 && spec.fittings === "CN") {
    material_compat_pass = false;
    const msg = "LOX_IGNITION_RISK: LOX above 2.0 MPa requires EU_STD or DE_PRECISION oxygen-clean fittings";
    material_notes.push(msg);
    violations.push(msg);
  }

  // 4. Daily Thermal Boil-Off Rate
  const k_base = INSULATION_PROPS[spec.insulation].k_eff_w_mk;
  const k_eff = k_base * (spec.insulation === "PVI" ? pvi_k_multiplier : 1.0);
  const t_ins_m = INSULATION_PROPS[spec.insulation].annulus_mm / 1000.0;
  const delta_t = AMBIENT_TEMP_K - mprops.boil_temp_k;

  const q_cond_w = (k_eff * area_m2 * delta_t) / t_ins_m;
  const q_valve_w =
    FITTINGS_PROPS[spec.fittings].valve_heat_leak_w *
    (order.execution === "TRANSPORTABLE" ? 1.25 : 1.0);
  const q_day_j = (q_cond_w + q_valve_w) * 86400.0;
  const evap_kg_day = q_day_j / (mprops.latent_heat_kj_kg * 1000.0);
  const boil_off_pct_day = round4((evap_kg_day / usable_unit_mass_kg) * 100.0);
  const boil_off_pass = boil_off_pct_day <= order.max_boil_off_pct_day;
  if (!boil_off_pass) {
    violations.push(
      `BOIL_OFF_EXCEEDED: ${boil_off_pct_day}%/day > max ${order.max_boil_off_pct_day}%/day`
    );
  }

  // 5. Site Clearance
  const bank_footprint_mm = spec.unit_count * dims.installed_width_mm + (spec.unit_count - 1) * 400;
  const height_ok = dims.installed_height_mm <= order.max_site_height_mm;
  const footprint_ok = bank_footprint_mm <= order.max_site_footprint_mm;
  const site_clearance_pass = height_ok && footprint_ok;
  if (!height_ok) {
    violations.push(
      `SITE_HEIGHT_EXCEEDED: installed ${dims.installed_height_mm} mm > site max ${order.max_site_height_mm} mm`
    );
  }
  if (!footprint_ok) {
    violations.push(
      `SITE_FOOTPRINT_EXCEEDED: bank width ${bank_footprint_mm} mm > site max ${order.max_site_footprint_mm} mm`
    );
  }

  // Fab & Cost
  const base_fab =
    2 +
    (spec.unit_volume_m3 >= 10.0 ? 1 : 0) +
    (spec.unit_volume_m3 >= 20.0 ? 1 : 0) +
    (spec.wall_thickness_mm >= 14.0 ? 1 : 0) +
    (order.execution === "TRANSPORTABLE" ? 1 : 0);
  const fab_weeks = base_fab + (spec.unit_count - 1);
  const fittings_lead_weeks = Math.max(
    1,
    Math.trunc(FITTINGS_PROPS[spec.fittings].base_lead_weeks + fittings_lead_delta)
  );

  const ins_mult = INSULATION_PROPS[spec.insulation].cost_multiplier;
  const fit_mult = FITTINGS_PROPS[spec.fittings].cost_multiplier;
  const exec_mult = order.execution === "TRANSPORTABLE" ? 1.18 : 1.0;
  const fixed_kit = FITTINGS_PROPS[spec.fittings].fixed_kit_eur;
  const unit_cost =
    (9500.0 + spec.unit_volume_m3 * 2150.0 + unit_tare_mass_kg * 3.1) *
      ins_mult *
      fit_mult *
      exec_mult +
    fixed_kit;
  const series_factor = spec.unit_count > 1 ? 0.94 : 1.0;
  const manufacturing_cost_eur = round2(spec.unit_count * unit_cost * series_factor);

  const all_engineering_pass =
    violations.length === 0 &&
    mass_pass &&
    pressure_pass &&
    stress_pass &&
    material_compat_pass &&
    boil_off_pass &&
    site_clearance_pass;

  return {
    usable_unit_mass_kg,
    usable_total_mass_kg,
    required_product_mass_kg: order.required_product_mass_kg,
    mass_pass,
    hoop_stress_mpa,
    allowable_stress_mpa,
    stress_pass,
    pressure_pass,
    boil_off_pct_day,
    max_boil_off_pct_day: order.max_boil_off_pct_day,
    boil_off_pass,
    material_compat_pass,
    material_notes,
    unit_tare_mass_kg,
    unit_transport_mass_kg,
    dimensions: dims,
    site_clearance_pass,
    fab_weeks,
    fittings_lead_weeks,
    manufacturing_cost_eur,
    all_engineering_pass,
    violations,
  };
}

export interface FittedVessel {
  width: number;
  height: number;
  aspect: number;
  calloutLayout: "sides" | "stacked";
}

/**
 * True-proportion GOST drawing scaler adapted from TechGas VesselBlueprint.tsx.
 */
export function fitVessel(
  H: number,
  D: number,
  boxWidth = 440,
  boxHeight = 380
): FittedVessel {
  const safeH = Math.max(10, H);
  const safeD = Math.max(10, D);
  const rawRatio = safeD / safeH;

  let ratio = rawRatio;
  if (rawRatio > 8) {
    ratio = 8 + Math.log(1 + (rawRatio - 8));
  } else if (rawRatio < 1 / 8) {
    const inv = 1 / rawRatio;
    const compInv = 8 + Math.log(1 + (inv - 8));
    ratio = 1 / compInv;
  }

  let width: number;
  let height: number;
  if (ratio > boxWidth / boxHeight) {
    width = boxWidth;
    height = Math.max(28, Math.round(boxWidth / ratio));
  } else {
    height = boxHeight;
    width = Math.max(28, Math.round(boxHeight * ratio));
  }

  const aspect = width / height;
  const calloutLayout: "sides" | "stacked" = aspect > 0.78 ? "stacked" : "sides";

  return { width, height, aspect, calloutLayout };
}
