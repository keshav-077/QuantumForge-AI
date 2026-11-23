import type { CanonicalFailureCode, FittingsGrade, InsulationType } from "../types/cryoforge";

export interface ScenarioPlainMeta {
  shortName: string;
  title: string;
  disruptionSummary: string;
  disruptionImpacts: string[];
}

export const SCENARIO_PLAIN_META: Record<string, ScenarioPlainMeta> = {
  S01: {
    shortName: "Scenario 1",
    title: "Customs Hold, Tier-1 Efficiency Drop & North Route Weight Cap",
    disruptionSummary:
      "At Step 4: Tier-3 Supplier is delayed by +3 weeks, Tier-1 Config loses 25% efficiency, and Route North caps single-unit weight at 5,800 kg.",
    disruptionImpacts: [
      "Tier-3 Supplier lead time increases by +3 weeks",
      "Tier-1 Config efficiency loss rises 1.25×",
      "Route North max unit weight drops to 5,800 kg",
    ],
  },
  S02: {
    shortName: "Scenario 2",
    title: "Lane B Mid-Schedule Outage & Budget Freeze",
    disruptionSummary:
      "At Step 4: Execution Lane B goes offline for maintenance during Weeks 5–7, and total project budget is cut by €15,000.",
    disruptionImpacts: [
      "Lane B unavailable during Weeks 5–7 (maintenance window)",
      "Global project budget reduced by €15,000",
    ],
  },
  S03: {
    shortName: "Scenario 3",
    title: "Tier-1 Efficiency Degradation & Oversize Permit Cut",
    disruptionSummary:
      "At Step 4: Tier-1 Config efficiency degrades by 1.55×, and available oversize transport permits are cut from 2 to 1.",
    disruptionImpacts: [
      "Tier-1 Config efficiency degradation spikes 1.55×",
      "Oversize route permit quota reduced from 2 to 1",
    ],
  },
  S04: {
    shortName: "Scenario 4",
    title: "Task 3 Load Uprate & Route Alpine Clearance Lock",
    disruptionSummary:
      "At Step 4: Task 3 minimum load rating jumps from 1.8 to 2.5 MPa, and Route Alpine caps height at 2,700 mm with oversize permits suspended.",
    disruptionImpacts: [
      "Task 3 required load rating increases to 2.5 MPa",
      "Route Alpine max clearance drops to 2,700 mm (no permits)",
    ],
  },
  S05: {
    shortName: "Scenario 5",
    title: "Tier-2 Supplier Delay & Lane A Capacity Cap",
    disruptionSummary:
      "At Step 4: Tier-2 Supplier experiences a +2 week backlog, and Lane A caps single-unit hardware weight at 5,200 kg.",
    disruptionImpacts: [
      "Tier-2 Supplier lead time increases by +2 weeks",
      "Lane A max unit weight capped at 5,200 kg",
    ],
  },
  S06: {
    shortName: "Scenario 6 (Held-Out)",
    title: "Tier-3 Supplier Delay, Route Fjord Weight Cap & Budget Cut",
    disruptionSummary:
      "At Step 4: Tier-3 Supplier adds +2 weeks delay, Route Fjord caps unit weight at 5,600 kg, and budget drops by €10,000.",
    disruptionImpacts: [
      "Tier-3 Supplier delayed by +2 weeks",
      "Route Fjord max unit weight capped at 5,600 kg",
      "Global project budget reduced by €10,000",
    ],
  },
  S07: {
    shortName: "Scenario 7 (Held-Out)",
    title: "Lane A Outage, Tier-1 Degradation & Permit Cut",
    disruptionSummary:
      "At Step 4: Lane A goes offline during Weeks 4–6, Tier-1 Config degrades 1.50×, and oversize permits are cut from 2 to 1.",
    disruptionImpacts: [
      "Lane A unavailable during Weeks 4–6",
      "Tier-1 Config efficiency degrades 1.50×",
      "Oversize route permits cut from 2 to 1",
    ],
  },
  S08: {
    shortName: "Scenario 8 (Held-Out)",
    title: "Task 2 Load Uprate, Route Gorge Clearance Lock & Lane B Weight Cap",
    disruptionSummary:
      "At Step 4: Task 2 load target jumps to 2.5 MPa, Route Gorge caps clearance at 2,700 mm, Lane B caps weight at 5,100 kg, and Tier-2 Supplier adds +2 weeks.",
    disruptionImpacts: [
      "Task 2 load target uprated to 2.5 MPa",
      "Route Gorge max height capped at 2,700 mm (no permits)",
      "Lane B max unit weight capped at 5,100 kg",
    ],
  },
};

export interface PlainFailureTaxonomyMeta {
  title: string;
  subtitle: string;
  plainExplanation: string;
}

export const FAILURE_TAXONOMY_PLAIN: Record<CanonicalFailureCode, PlainFailureTaxonomyMeta> = {
  BAY_CONFLICT: {
    title: "Schedule Lane Collision or Clearance Breach",
    subtitle: "Double-booked execution lanes, maintenance window overlap, or site clearance breach",
    plainExplanation:
      "The agent books two tasks into the same lane during overlapping weeks, schedules work during a Step-4 maintenance outage, or exceeds physical clearance limits.",
  },
  BOIL_OFF: {
    title: "Efficiency Limit Breach",
    subtitle: "Selected resource tier fails daily efficiency threshold after Step-4 degradation",
    plainExplanation:
      "The agent chooses a low-cost Tier-1 configuration or splits into too many small units without verifying that efficiency loss stays within the task's strict daily limit.",
  },
  SIM_EXHAUSTION: {
    title: "Simulation Tool Budget Exhausted (3/3 Used)",
    subtitle: "Agent burns all 3 simulation calls guessing instead of reading the rules handbook",
    plainExplanation:
      "Agents are given only 3 calls to the simulation tool (`run_physics_sim`) across 4 tasks. Frontier models waste all 3 calls guessing parameters on Task 1–2 and fly blind after Step 4.",
  },
  SHOCK_UNADAPTED: {
    title: "Ignored Step-4 Disruption (Did Not Re-Plan)",
    subtitle: "Agent commits to its initial plan and fails to revise tasks after the Step-4 disruption",
    plainExplanation:
      "At Step 4, supplier lead times, lane availability, or route limits change. The agent fails to call `revise_order` on affected tasks, causing missed SLA deadlines.",
  },
  HOOP_STRESS: {
    title: "Safety Margin Breach",
    subtitle: "Structural wall rating is too thin for the task's required pressure and safety factor",
    plainExplanation:
      "The agent under-allocates structural wall thickness or ignores the 1.25× high-load safety multiplier defined in the rules handbook.",
  },
  BRIDGE_MASS_CAP: {
    title: "Route Capacity / Oversize Limit Breach",
    subtitle: "Single-unit weight or width exceeds route limit without an oversize permit",
    plainExplanation:
      "The agent books a heavy single-unit configuration that exceeds the route's weight or width limit instead of splitting the workload into 2 modular units.",
  },
};

export function formatSeedLabel(seed: string): string {
  const num = seed.replace("S0", "");
  return `Scenario ${num}`;
}

export function formatOrderLabel(orderId: string): string {
  if (orderId === "BATCH") return "All Tasks (Batch)";
  const num = orderId.replace("ORD-", "");
  return `Task ${num}`;
}

export function formatBayLabel(bayId: string): string {
  if (bayId === "BAY-A") return "Lane A";
  if (bayId === "BAY-B") return "Lane B";
  return bayId;
}

export function formatConfigTier(insulation: InsulationType): string {
  return insulation === "EVI" ? "High-Efficiency Tier" : "Standard Tier";
}

export function formatSupplierTier(fittings: FittingsGrade): string {
  if (fittings === "DE_PRECISION") return "Tier-3 Fast/Precision";
  if (fittings === "EU_STD") return "Tier-2 Standard";
  return "Tier-1 Economy";
}

export function formatAgentDisplayName(agentName: string, agentType?: string): string {
  if (agentType === "reference" || agentName.toLowerCase().includes("reference")) {
    return "Reference Planner (Analytical)";
  }
  if (agentType === "frontier" || agentName.toLowerCase().includes("frontier")) {
    return "Frontier Agent (gemini-3.8-flash)";
  }
  if (agentType === "small_after" || agentName.toLowerCase().includes("after")) {
    return "Small Model (After Training)";
  }
  if (agentType === "small_before" || agentName.toLowerCase().includes("before")) {
    return "Small Model (Before Training)";
  }
  return agentName;
}

export function humanizeViolationString(raw: string): string {
  return raw
    .replace(/ORD-(\d+)/g, "Task $1")
    .replace(/BAY-A/g, "Lane A")
    .replace(/BAY-B/g, "Lane B")
    .replace(/HOOP_STRESS_EXCEEDED/g, "Safety Margin Breach")
    .replace(/BOIL_OFF_EXCEEDED/g, "Efficiency Limit Breach")
    .replace(/SHOCK_UNADAPTED/g, "Ignored Step-4 Disruption")
    .replace(/STD_GAUGE_EXCEEDED/g, "Route Width Limit Exceeded")
    .replace(/BRIDGE_MASS_EXCEEDED/g, "Route Weight Cap Exceeded")
    .replace(/SITE_HEIGHT_EXCEEDED/g, "Max Height Clearance Exceeded")
    .replace(/SLA_MISSED/g, "Missed SLA Deadline")
    .replace(/HORIZON_EXCEEDED/g, "Exceeded Schedule Horizon")
    .replace(/BAY_OVERLAP/g, "Lane Schedule Collision")
    .replace(/SIM_BUDGET_EXHAUSTED/g, "Simulation Budget Exhausted (3/3)")
    .replace(/DE_PRECISION/g, "Tier-3 Supplier")
    .replace(/EU_STD/g, "Tier-2 Supplier")
    .replace(/PVI/g, "Standard Tier")
    .replace(/EVI/g, "High-Efficiency Tier")
    .replace(/boil-off/gi, "efficiency loss")
    .replace(/hoop stress/gi, "safety stress");
}
