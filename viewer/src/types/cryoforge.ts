export type Medium = "LIN" | "LOX" | "LAR" | "LCO2" | "LNG" | "LH2" | "LHe";
export type ExecutionType = "STATIONARY" | "TRANSPORTABLE";
export type InsulationType = "EVI" | "PVI";
export type FittingsGrade = "CN" | "EU_STD" | "DE_PRECISION";
export type BayId = "BAY-A" | "BAY-B";

export type AgentType =
  | "reference"
  | "frontier"
  | "do_nothing"
  | "stop_early"
  | "small_before"
  | "small_after"
  | "exploit";

export const EVAL_SEEDS = ["S01", "S02", "S03", "S04", "S05"] as const;
export const TIEBREAKER_SEEDS = ["S06", "S07", "S08"] as const;
export const ALL_SEEDS = [...EVAL_SEEDS, ...TIEBREAKER_SEEDS] as const;

export interface VesselDesignSpec {
  unit_count: number;
  unit_volume_m3: number;
  design_pressure_mpa: number;
  wall_thickness_mm: number;
  insulation: InsulationType;
  fittings: FittingsGrade;
}

export interface DimensionsMM {
  outer_diameter_mm: number;
  inner_diameter_mm: number;
  shell_axis_mm: number;
  installed_height_mm: number;
  installed_width_mm: number;
  installed_length_mm: number;
  shipping_height_mm: number;
  shipping_width_mm: number;
  shipping_length_mm: number;
}

export interface PhysicsEvaluation {
  usable_unit_mass_kg: number;
  usable_total_mass_kg: number;
  required_product_mass_kg: number;
  mass_pass: boolean;

  hoop_stress_mpa: number;
  allowable_stress_mpa: number;
  stress_pass: boolean;
  pressure_pass: boolean;

  boil_off_pct_day: number;
  max_boil_off_pct_day: number;
  boil_off_pass: boolean;

  material_compat_pass: boolean;
  material_notes: string[];

  unit_tare_mass_kg: number;
  unit_transport_mass_kg: number;
  dimensions: DimensionsMM;
  site_clearance_pass: boolean;

  fab_weeks: number;
  fittings_lead_weeks: number;
  manufacturing_cost_eur: number;

  all_engineering_pass: boolean;
  violations: string[];
}

export interface CustomerOrder {
  order_id: string;
  client_name: string;
  medium: Medium;
  execution: ExecutionType;
  required_product_mass_kg: number;
  min_pressure_mpa: number;
  max_boil_off_pct_day: number;
  sla_week: number;
  route_id: string;
  max_site_height_mm: number;
  max_site_footprint_mm: number;
  priority_weight: number;
  notes?: string;
}

export interface TransportRoute {
  route_id: string;
  name: string;
  transit_weeks: number;
  std_max_width_mm: number;
  std_max_height_mm: number;
  permit_max_width_mm: number;
  permit_max_height_mm: number;
  escort_permit_allowed: boolean;
  max_unit_transport_mass_kg: number;
  base_freight_eur_per_unit: number;
  permit_fee_eur: number;
  notes?: string;
}

export interface BayBlackoutWindow {
  bay_id: BayId;
  start_week: number;
  end_week: number; // exclusive [start_week, end_week)
  reason: string;
}

export interface DisruptionEvent {
  disruption_id: string;
  title: string;
  headline: string;
  details: string;
  fittings_lead_deltas: Partial<Record<FittingsGrade, number>>;
  pvi_k_multiplier: number;
  budget_delta_eur: number;
  escort_permits_delta: number;
  bay_blackouts: BayBlackoutWindow[];
  bay_max_unit_tare_kg: Partial<Record<BayId, number>>;
  route_mass_cap_overrides: Record<string, number>;
  route_std_height_overrides: Record<string, number>;
  route_permit_allowed_overrides: Record<string, boolean>;
  order_min_pressure_overrides: Record<string, number>;
  order_sla_overrides: Record<string, number>;
}

export interface OrderBooking {
  order_id: string;
  bay_id: BayId;
  bay_start_week: number;
  fab_weeks: number;
  fab_end_week: number;
  booked_at_step: number;
  order_release_week: number;
  spec: VesselDesignSpec;
  manufacturing_cost_eur: number;
  assigned_route_id: string | null;
  use_escort_permit: boolean;
  revision_count: number;
  post_shock_revised: boolean;
}

export interface OrderVerificationResult {
  order_id: string;
  booked: boolean;
  route_assigned: boolean;
  passed_all_hard_checks: boolean;
  order_raw_score: number;
  weighted_contribution: number;
  delivery_week: number | null;
  sla_week: number;
  total_order_cost_eur: number;
  physics: PhysicsEvaluation | null;
  violations: string[];
}

export interface VerificationReport {
  seed: string;
  finalized: boolean;
  final_score: number;
  valid_order_count: number;
  total_order_count: number;
  batch_gate_cap_applied: boolean;
  budget_exceeded: boolean;
  permit_quota_exceeded: boolean;
  total_budget_eur: number;
  total_spent_eur: number;
  budget_efficiency_bonus: number;
  schedule_margin_bonus: number;
  sim_efficiency_bonus: number;
  sim_calls_used: number;
  sim_budget_total: number;
  step_count: number;
  disruption_triggered: boolean;
  order_results: OrderVerificationResult[];
  failure_modes: string[];
  primary_failure_categories: string[];
}

export interface TrajectoryStepLog {
  step_index: number;
  timestamp_iso: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  observation: Record<string, unknown>;
  disruption_triggered_this_step: boolean;
}

export interface TrajectoryLog {
  run_id: string;
  seed: string;
  agent_name: string;
  agent_type: AgentType;
  model_id: string;
  started_at: string;
  completed_at: string;
  steps: TrajectoryStepLog[];
  verification: VerificationReport;
  state_hmac?: string;
}

export type CanonicalFailureCode =
  | "HOOP_STRESS"
  | "BOIL_OFF"
  | "SHOCK_UNADAPTED"
  | "BRIDGE_MASS_CAP"
  | "SIM_EXHAUSTION"
  | "BAY_CONFLICT";

export interface FailureIncidentLink {
  seed: string;
  run_id: string;
  agent_name: string;
  order_id: string;
  step_index: number;
  tool_name: string;
  summary: string;
  violation_badge: string;
}

export interface FailureTaxonomyEntry {
  code: CanonicalFailureCode;
  title: string;
  subtitle: string;
  occurrences: number;
  affected_seeds: string[];
  root_cause: string;
  incidents: FailureIncidentLink[];
}

export interface ExploitAuditItem {
  exploit_id: string;
  name: string;
  category: string;
  attack_vector: string;
  defense_mechanism: string;
  invariant_ref: string;
  observed_score: number;
  max_allowed_score: number;
  blocked: boolean;
  sample_tool_trace: string[];
}

export interface SeedScoreSummary {
  seed: string;
  split: "eval" | "tiebreaker";
  scenario_title: string;
  shock_headline: string;
  reference_score: number;
  frontier_scores: number[];
  frontier_mean: number;
  frontier_std: number;
  gap_delta: number;
  small_before_score?: number;
  small_after_score?: number;
}

export interface TiebreakerSummary {
  model_family: string;
  adapter_spec: string;
  train_seeds: string[];
  held_out_seeds: string[];
  before_mean: number;
  after_mean: number;
  delta_improvement: number;
  relative_gain_pct: number;
  per_seed: Array<{
    seed: string;
    scenario_title: string;
    before_score: number;
    after_score: number;
    delta: number;
    before_valid_orders: number;
    after_valid_orders: number;
    key_learned_adaptation: string;
  }>;
}

export interface ScorecardData {
  generated_at: string;
  environment_version: string;
  rubric: {
    reference_mean: number;
    frontier_mean: number;
    frontier_std: number;
    gap_delta: number;
    do_nothing_score: number;
    stop_early_score: number;
    max_exploit_score: number;
    exploits_blocked: number;
    exploits_total: number;
  };
  per_seed: SeedScoreSummary[];
  failure_taxonomy: FailureTaxonomyEntry[];
  exploits: ExploitAuditItem[];
  tiebreaker: TiebreakerSummary;
  trajectories: TrajectoryLog[];
}
