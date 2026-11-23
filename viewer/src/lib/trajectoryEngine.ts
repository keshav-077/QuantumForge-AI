import type {
  BayBlackoutWindow,
  BayId,
  CustomerOrder,
  DisruptionEvent,
  ExecutionType,
  FittingsGrade,
  InsulationType,
  Medium,
  OrderBooking,
  PhysicsEvaluation,
  TrajectoryLog,
  TransportRoute,
  VesselDesignSpec,
} from "../types/cryoforge";
import { ASSEMBLY_COMMISSION_WEEKS, SIM_BUDGET_DEFAULT, evaluateDesignPhysics } from "./cryoPhysics";

export interface SeedWorldMetadata {
  seed: string;
  horizonWeeks: number;
  initialBudgetEur: number;
  initialEscortPermits: number;
  orders: CustomerOrder[];
  routes: Record<string, TransportRoute>;
  initialBlackouts: BayBlackoutWindow[];
  disruption: DisruptionEvent | null;
}

export interface BlueprintViolationBadge {
  code: string;
  label: string;
  severity: "crimson" | "amber";
}

export interface OrderLiveSnapshot {
  order: CustomerOrder;
  route: TransportRoute;
  spec: VesselDesignSpec;
  specSource: "booked" | "simulated" | "unconfigured";
  booking: OrderBooking | null;
  physics: PhysicsEvaluation;
  fittingsReadyWeek: number | null;
  fabEndWeek: number | null;
  assemblyEndWeek: number | null;
  deliveryWeek: number | null;
  slaPass: boolean;
  bayConflictReasons: string[];
  transportViolations: string[];
  allViolations: string[];
  passedAllHardChecks: boolean;
  blueprintBadges: BlueprintViolationBadge[];
}

export interface ReplayedFrameState {
  stepArrayIndex: number;
  stepNumber: number;
  totalSteps: number;
  shockArrayIndex: number | null;
  shockStepNumber: number | null;
  disruptionActive: boolean;
  disruptionJustTriggered: boolean;
  disruption: DisruptionEvent | null;
  simCallsUsed: number;
  simBudgetTotal: number;
  effectiveBudgetEur: number;
  spentEur: number;
  retoolingFeesEur: number;
  permitsUsed: number;
  effectivePermitsTotal: number;
  activeBlackouts: BayBlackoutWindow[];
  bayMaxTareKg: Partial<Record<BayId, number>>;
  finalized: boolean;
  orders: OrderLiveSnapshot[];
}

function asObj(v: unknown): Record<string, unknown> | null {
  return v !== null && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;
}

function parseOrderCandidate(raw: unknown): CustomerOrder | null {
  const o = asObj(raw);
  if (!o || typeof o.order_id !== "string") return null;
  return {
    order_id: o.order_id,
    client_name: typeof o.client_name === "string" ? o.client_name : o.order_id,
    medium: (typeof o.medium === "string" ? o.medium : "LIN") as Medium,
    execution: (typeof o.execution === "string" ? o.execution : "STATIONARY") as ExecutionType,
    required_product_mass_kg: Number(o.required_product_mass_kg ?? 8000),
    min_pressure_mpa: Number(o.min_pressure_mpa ?? 1.6),
    max_boil_off_pct_day: Number(o.max_boil_off_pct_day ?? 0.35),
    sla_week: Number(o.sla_week ?? 12),
    route_id: typeof o.route_id === "string" ? o.route_id : "RT-NORTH",
    max_site_height_mm: Number(o.max_site_height_mm ?? 6500),
    max_site_footprint_mm: Number(o.max_site_footprint_mm ?? 6000),
    priority_weight: Number(o.priority_weight ?? 0.25),
    notes: typeof o.notes === "string" ? o.notes : "",
  };
}

function parseRouteCandidate(raw: unknown): TransportRoute | null {
  const r = asObj(raw);
  if (!r || typeof r.route_id !== "string") return null;
  return {
    route_id: r.route_id,
    name: typeof r.name === "string" ? r.name : r.route_id,
    transit_weeks: Number(r.transit_weeks ?? 2),
    std_max_width_mm: Number(r.std_max_width_mm ?? 2500),
    std_max_height_mm: Number(r.std_max_height_mm ?? 3800),
    permit_max_width_mm: Number(r.permit_max_width_mm ?? 3100),
    permit_max_height_mm: Number(r.permit_max_height_mm ?? 4350),
    escort_permit_allowed: r.escort_permit_allowed !== false,
    max_unit_transport_mass_kg: Number(r.max_unit_transport_mass_kg ?? 16500),
    base_freight_eur_per_unit: Number(r.base_freight_eur_per_unit ?? 2800),
    permit_fee_eur: Number(r.permit_fee_eur ?? 4200),
    notes: typeof r.notes === "string" ? r.notes : "",
  };
}

function parseDisruptionCandidate(raw: unknown): DisruptionEvent | null {
  const d = asObj(raw);
  if (!d || typeof d.disruption_id !== "string") return null;
  const blackoutsRaw = Array.isArray(d.bay_blackouts) ? d.bay_blackouts : [];
  const bay_blackouts: BayBlackoutWindow[] = blackoutsRaw
    .map((b) => {
      const bo = asObj(b);
      if (!bo) return null;
      return {
        bay_id: (bo.bay_id === "BAY-B" ? "BAY-B" : "BAY-A") as BayId,
        start_week: Number(bo.start_week ?? 4),
        end_week: Number(bo.end_week ?? 6),
        reason: typeof bo.reason === "string" ? bo.reason : "Unplanned Maintenance",
      };
    })
    .filter((x): x is BayBlackoutWindow => x !== null);

  return {
    disruption_id: d.disruption_id,
    title: typeof d.title === "string" ? d.title : "Supply & Corridor Disruption",
    headline: typeof d.headline === "string" ? d.headline : String(d.title ?? "Disruption Alert"),
    details: typeof d.details === "string" ? d.details : "",
    fittings_lead_deltas: (asObj(d.fittings_lead_deltas) ?? {}) as Partial<Record<FittingsGrade, number>>,
    pvi_k_multiplier: Number(d.pvi_k_multiplier ?? 1.0),
    budget_delta_eur: Number(d.budget_delta_eur ?? 0),
    escort_permits_delta: Number(d.escort_permits_delta ?? 0),
    bay_blackouts,
    bay_max_unit_tare_kg: (asObj(d.bay_max_unit_tare_kg) ?? {}) as Partial<Record<BayId, number>>,
    route_mass_cap_overrides: (asObj(d.route_mass_cap_overrides) ?? {}) as Record<string, number>,
    route_std_height_overrides: (asObj(d.route_std_height_overrides) ?? {}) as Record<string, number>,
    route_permit_allowed_overrides: (asObj(d.route_permit_allowed_overrides) ?? {}) as Record<string, boolean>,
    order_min_pressure_overrides: (asObj(d.order_min_pressure_overrides) ?? {}) as Record<string, number>,
    order_sla_overrides: (asObj(d.order_sla_overrides) ?? {}) as Record<string, number>,
  };
}

export function findShockArrayIndex(run: TrajectoryLog): number | null {
  for (let i = 0; i < run.steps.length; i++) {
    const st = run.steps[i];
    if (st.disruption_triggered_this_step || st.observation?.disruption_alert != null) {
      return i;
    }
  }
  return null;
}

export function extractSeedWorldMetadata(seed: string, runs: TrajectoryLog[]): SeedWorldMetadata {
  const orderMap = new Map<string, CustomerOrder>();
  const routeMap: Record<string, TransportRoute> = {};
  let initialBlackouts: BayBlackoutWindow[] = [];
  let disruption: DisruptionEvent | null = null;
  let horizonWeeks = 16;
  let initialBudgetEur = runs[0]?.verification?.total_budget_eur ?? 320000;
  let initialEscortPermits = 2;

  // Prefer pre-shock inspection observations first so initial order specs are clean
  for (const run of runs) {
    const shockIdx = findShockArrayIndex(run);
    for (let i = 0; i < run.steps.length; i++) {
      const st = run.steps[i];
      const obs = st.observation ?? {};
      const beforeShock = shockIdx === null || i < shockIdx;

      if (typeof obs.horizon_weeks === "number") horizonWeeks = obs.horizon_weeks;
      if (beforeShock && typeof obs.total_budget_eur === "number") {
        initialBudgetEur = obs.total_budget_eur;
      }
      if (beforeShock && typeof obs.escort_permits_total === "number") {
        initialEscortPermits = obs.escort_permits_total;
      }

      if (beforeShock && Array.isArray(obs.bay_blackouts) && initialBlackouts.length === 0) {
        initialBlackouts = obs.bay_blackouts
          .map((b) => {
            const bo = asObj(b);
            if (!bo) return null;
            return {
              bay_id: (bo.bay_id === "BAY-B" ? "BAY-B" : "BAY-A") as BayId,
              start_week: Number(bo.start_week ?? 1),
              end_week: Number(bo.end_week ?? 2),
              reason: String(bo.reason ?? "Scheduled Calibration"),
            };
          })
          .filter((x): x is BayBlackoutWindow => x !== null);
      }

      // Extract orders from episode brief or inspect_order
      if (Array.isArray(obs.orders)) {
        for (const item of obs.orders) {
          const parsed = parseOrderCandidate(item);
          if (parsed && (!orderMap.has(parsed.order_id) || beforeShock)) {
            if (!orderMap.has(parsed.order_id)) orderMap.set(parsed.order_id, parsed);
          }
        }
      }
      const singleOrder = parseOrderCandidate(obs.order ?? (st.tool_name === "inspect_order" ? obs : null));
      if (singleOrder && (!orderMap.has(singleOrder.order_id) || beforeShock)) {
        orderMap.set(singleOrder.order_id, singleOrder);
      }

      // Extract routes
      if (Array.isArray(obs.routes)) {
        for (const item of obs.routes) {
          const parsed = parseRouteCandidate(item);
          if (parsed && (!routeMap[parsed.route_id] || beforeShock)) {
            if (!routeMap[parsed.route_id]) routeMap[parsed.route_id] = parsed;
          }
        }
      }
      const singleRoute = parseRouteCandidate(obs.route ?? (st.tool_name === "inspect_route" ? obs : null));
      if (singleRoute && (!routeMap[singleRoute.route_id] || beforeShock)) {
        routeMap[singleRoute.route_id] = singleRoute;
      }

      // Extract disruption
      if (!disruption && obs.disruption_alert) {
        disruption = parseDisruptionCandidate(obs.disruption_alert);
      }
      if (!disruption && obs.active_disruption) {
        disruption = parseDisruptionCandidate(obs.active_disruption);
      }
    }
  }

  // Fallback synthesis if any order in verification.order_results was never inspected
  for (const run of runs) {
    for (const res of run.verification?.order_results ?? []) {
      if (!orderMap.has(res.order_id)) {
        orderMap.set(res.order_id, {
          order_id: res.order_id,
          client_name: res.order_id,
          medium: "LIN",
          execution: "STATIONARY",
          required_product_mass_kg: res.physics?.required_product_mass_kg ?? 8000,
          min_pressure_mpa: 1.6,
          max_boil_off_pct_day: res.physics?.max_boil_off_pct_day ?? 0.35,
          sla_week: res.sla_week,
          route_id: "RT-STD",
          max_site_height_mm: 6500,
          max_site_footprint_mm: 6000,
          priority_weight: 0.25,
        });
      }
    }
  }

  const orders = Array.from(orderMap.values());
  for (const ord of orders) {
    if (!routeMap[ord.route_id]) {
      routeMap[ord.route_id] = {
        route_id: ord.route_id,
        name: `${ord.route_id} Industrial Corridor`,
        transit_weeks: 2,
        std_max_width_mm: 2500,
        std_max_height_mm: 3800,
        permit_max_width_mm: 3100,
        permit_max_height_mm: 4350,
        escort_permit_allowed: true,
        max_unit_transport_mass_kg: 16500,
        base_freight_eur_per_unit: 2800,
        permit_fee_eur: 4200,
      };
    }
  }

  if (disruption && runs[0]?.verification?.total_budget_eur) {
    // Adjust initialBudgetEur if verification reflects post-shock budget
    if (runs[0].verification.disruption_triggered && disruption.budget_delta_eur !== 0) {
      // Keep explicit pre-shock budget if found, else derive from verification
      const hasPreShockObs = runs.some((r) =>
        r.steps.some((s) => typeof s.observation?.total_budget_eur === "number" && !s.disruption_triggered_this_step)
      );
      if (!hasPreShockObs) {
        initialBudgetEur = runs[0].verification.total_budget_eur - disruption.budget_delta_eur;
      }
    }
  }

  return {
    seed,
    horizonWeeks,
    initialBudgetEur,
    initialEscortPermits,
    orders,
    routes: routeMap,
    initialBlackouts,
    disruption,
  };
}

function parseSpecFromArgs(args: Record<string, unknown>, fallback?: VesselDesignSpec): VesselDesignSpec {
  return {
    unit_count: Number(args.unit_count ?? fallback?.unit_count ?? 1),
    unit_volume_m3: Number(args.unit_volume_m3 ?? fallback?.unit_volume_m3 ?? 10),
    design_pressure_mpa: Number(args.design_pressure_mpa ?? fallback?.design_pressure_mpa ?? 1.6),
    wall_thickness_mm: Number(args.wall_thickness_mm ?? fallback?.wall_thickness_mm ?? 8),
    insulation: (typeof args.insulation === "string"
      ? args.insulation
      : fallback?.insulation ?? "EVI") as InsulationType,
    fittings: (typeof args.fittings === "string"
      ? args.fittings
      : fallback?.fittings ?? "EU_STD") as FittingsGrade,
  };
}

function buildBlueprintBadges(
  order: CustomerOrder,
  spec: VesselDesignSpec,
  physics: PhysicsEvaluation,
  route: TransportRoute,
  booking: OrderBooking | null,
  deliveryWeek: number | null,
  horizonWeeks: number,
  bayConflictReasons: string[],
  permitQuotaExceeded: boolean
): BlueprintViolationBadge[] {
  const badges: BlueprintViolationBadge[] = [];

  if (!physics.stress_pass) {
    badges.push({
      code: "HOOP_STRESS",
      label: `HOOP STRESS ${physics.hoop_stress_mpa.toFixed(1)} > ${physics.allowable_stress_mpa.toFixed(1)} MPa`,
      severity: "crimson",
    });
  }
  if (!physics.pressure_pass) {
    badges.push({
      code: "UNDER_PRESSURE",
      label: `DESIGN ${spec.design_pressure_mpa.toFixed(2)} < REQ ${order.min_pressure_mpa.toFixed(2)} MPa`,
      severity: "crimson",
    });
  }
  if (!physics.boil_off_pass) {
    badges.push({
      code: "BOIL_OFF",
      label: `BOIL-OFF ${physics.boil_off_pct_day.toFixed(3)}% > MAX ${order.max_boil_off_pct_day.toFixed(2)}%/d`,
      severity: "amber",
    });
  }
  if (!physics.material_compat_pass) {
    const isLox = physics.material_notes.some((n) => n.includes("LOX"));
    badges.push({
      code: "HAZARD",
      label: isLox
        ? `LOX > 2.0 MPa CN IGNITION HAZARD`
        : `${order.medium}/CRYO HAZARD (${spec.insulation}/${spec.fittings})`,
      severity: "crimson",
    });
  }
  if (!physics.mass_pass) {
    badges.push({
      code: "CAPACITY",
      label: `MASS ${Math.round(physics.usable_total_mass_kg)} < ${Math.round(order.required_product_mass_kg)} kg`,
      severity: "crimson",
    });
  }
  if (!physics.site_clearance_pass) {
    const bankW = spec.unit_count * physics.dimensions.installed_width_mm + (spec.unit_count - 1) * 400;
    if (physics.dimensions.installed_height_mm > order.max_site_height_mm) {
      badges.push({
        code: "SITE_HEIGHT",
        label: `HEIGHT ${physics.dimensions.installed_height_mm}mm > ${order.max_site_height_mm}mm SITE MAX`,
        severity: "amber",
      });
    }
    if (bankW > order.max_site_footprint_mm) {
      badges.push({
        code: "SITE_FOOTPRINT",
        label: `BANK WIDTH ${bankW}mm > ${order.max_site_footprint_mm}mm SITE MAX`,
        severity: "amber",
      });
    }
  }

  if (booking) {
    const usePermit = booking.use_escort_permit;
    const widthLimit = usePermit ? route.permit_max_width_mm : route.std_max_width_mm;
    const heightLimit = usePermit ? route.permit_max_height_mm : route.std_max_height_mm;

    if (usePermit && !route.escort_permit_allowed) {
      badges.push({
        code: "PERMIT_FORBIDDEN",
        label: `ESCORT PERMIT REVOKED ON ${route.route_id}`,
        severity: "crimson",
      });
    } else if (usePermit && permitQuotaExceeded) {
      badges.push({
        code: "PERMIT_QUOTA",
        label: `PERMIT QUOTA EXCEEDED`,
        severity: "crimson",
      });
    }

    if (physics.dimensions.shipping_width_mm > widthLimit) {
      badges.push({
        code: "SHIP_WIDTH",
        label: `SHIP WIDTH ${physics.dimensions.shipping_width_mm}mm > ${widthLimit}mm ROAD LIMIT`,
        severity: "amber",
      });
    }
    if (physics.dimensions.shipping_height_mm > heightLimit) {
      badges.push({
        code: "SHIP_HEIGHT",
        label: `SHIP HEIGHT ${physics.dimensions.shipping_height_mm}mm > ${heightLimit}mm CLEARANCE`,
        severity: "amber",
      });
    }
    if (physics.unit_transport_mass_kg > route.max_unit_transport_mass_kg) {
      badges.push({
        code: "BRIDGE_MASS_CAP",
        label: `TRANSPORT ${Math.round(physics.unit_transport_mass_kg)}kg > ${Math.round(route.max_unit_transport_mass_kg)}kg BRIDGE CAP`,
        severity: "crimson",
      });
    }
    if (!booking.assigned_route_id) {
      badges.push({
        code: "NO_ROUTE",
        label: `TRANSPORT ROUTE UNASSIGNED`,
        severity: "amber",
      });
    } else if (booking.assigned_route_id !== order.route_id) {
      badges.push({
        code: "WRONG_ROUTE",
        label: `WRONG ROUTE ${booking.assigned_route_id} != ${order.route_id}`,
        severity: "crimson",
      });
    }
    if (deliveryWeek !== null && (deliveryWeek > order.sla_week || deliveryWeek > horizonWeeks)) {
      badges.push({
        code: "SLA_MISS",
        label: `DELIVERY W${deliveryWeek} > SLA W${order.sla_week}`,
        severity: "crimson",
      });
    }
    if (bayConflictReasons.length > 0) {
      badges.push({
        code: "BAY_CONFLICT",
        label: bayConflictReasons[0],
        severity: "crimson",
      });
    }
  }

  return badges;
}

export function replayTrajectoryAtStep(
  run: TrajectoryLog,
  uptoArrayIdx: number,
  world: SeedWorldMetadata
): ReplayedFrameState {
  const totalSteps = run.steps.length;
  const clampedIdx = totalSteps === 0 ? -1 : Math.max(0, Math.min(uptoArrayIdx, totalSteps - 1));
  const shockArrayIndex = findShockArrayIndex(run);

  let disruptionActive = false;
  let disruptionJustTriggered = false;
  let simCallsUsed = 0;
  const simBudgetTotal = run.verification?.sim_budget_total ?? SIM_BUDGET_DEFAULT;
  let retoolingFeesEur = 0;
  let finalized = false;

  const bookings: Record<string, OrderBooking> = {};
  const latestSimSpecs: Record<string, VesselDesignSpec> = {};

  for (let i = 0; i <= clampedIdx; i++) {
    const step = run.steps[i];
    const obs = step.observation ?? {};
    const args = step.arguments ?? {};
    const ok = obs.ok !== false;

    const triggeredHere =
      step.disruption_triggered_this_step || obs.disruption_alert != null || i === shockArrayIndex;
    if (triggeredHere) {
      disruptionActive = true;
      if (i === clampedIdx) disruptionJustTriggered = true;
    } else if (i === clampedIdx) {
      disruptionJustTriggered = false;
    }

    if (step.tool_name === "run_physics_sim") {
      if (ok) {
        simCallsUsed += 1;
        const oid = typeof args.order_id === "string" ? args.order_id : "";
        if (oid) {
          latestSimSpecs[oid] = parseSpecFromArgs(args, latestSimSpecs[oid]);
        }
      } else if (obs.error === "SIM_BUDGET_EXHAUSTED") {
        simCallsUsed = simBudgetTotal;
      }
    }

    if (step.tool_name === "reserve_bay_and_order" && ok) {
      const bookingObs = asObj(obs.booking);
      const oid = String(bookingObs?.order_id ?? args.order_id ?? "");
      if (oid) {
        const ord = world.orders.find((o) => o.order_id === oid) ?? world.orders[0];
        const spec = bookingObs?.spec
          ? parseSpecFromArgs(asObj(bookingObs.spec) ?? {}, bookings[oid]?.spec)
          : parseSpecFromArgs(args, bookings[oid]?.spec);
        const phys = evaluateDesignPhysics(ord, spec);
        const bayStart = Number(bookingObs?.bay_start_week ?? args.bay_start_week ?? 1);
        const fabWeeks = Number(bookingObs?.fab_weeks ?? phys.fab_weeks);
        const preTriggered = disruptionActive && !triggeredHere;
        bookings[oid] = {
          order_id: oid,
          bay_id: ((bookingObs?.bay_id ?? args.bay_id) === "BAY-B" ? "BAY-B" : "BAY-A") as BayId,
          bay_start_week: bayStart,
          fab_weeks: fabWeeks,
          fab_end_week: Number(bookingObs?.fab_end_week ?? bayStart + fabWeeks),
          booked_at_step: step.step_index,
          order_release_week: Number(bookingObs?.order_release_week ?? (preTriggered ? 2 : 1)),
          spec,
          manufacturing_cost_eur: Number(bookingObs?.manufacturing_cost_eur ?? phys.manufacturing_cost_eur),
          assigned_route_id:
            typeof bookingObs?.assigned_route_id === "string" ? bookingObs.assigned_route_id : null,
          use_escort_permit: Boolean(bookingObs?.use_escort_permit ?? false),
          revision_count: 0,
          post_shock_revised: false,
        };
      }
    }

    if (step.tool_name === "assign_transport" && ok) {
      const oid = String(args.order_id ?? "");
      if (oid && bookings[oid]) {
        bookings[oid] = {
          ...bookings[oid],
          assigned_route_id: typeof args.route_id === "string" ? args.route_id : bookings[oid].assigned_route_id,
          use_escort_permit: Boolean(args.use_escort_permit ?? false),
        };
      }
    }

    if (step.tool_name === "revise_order" && ok) {
      const bookingObs = asObj(obs.booking);
      const oid = String(bookingObs?.order_id ?? args.order_id ?? "");
      if (oid) {
        const prev = bookings[oid];
        const ord = world.orders.find((o) => o.order_id === oid) ?? world.orders[0];
        const spec = bookingObs?.spec
          ? parseSpecFromArgs(asObj(bookingObs.spec) ?? {}, prev?.spec)
          : parseSpecFromArgs(args, prev?.spec);
        const phys = evaluateDesignPhysics(ord, spec);
        const bayStart = Number(bookingObs?.bay_start_week ?? args.bay_start_week ?? prev?.bay_start_week ?? 1);
        const fabWeeks = Number(bookingObs?.fab_weeks ?? phys.fab_weeks);

        const isFirstPostShock = disruptionActive && prev && !prev.post_shock_revised;
        retoolingFeesEur += isFirstPostShock ? 1500 : 4500;

        const changedProcurement =
          !prev || prev.spec.fittings !== spec.fittings || prev.spec.unit_count !== spec.unit_count;
        const releaseWeek = Number(
          bookingObs?.order_release_week ??
            (disruptionActive && changedProcurement ? 2 : prev?.order_release_week ?? 1)
        );

        bookings[oid] = {
          order_id: oid,
          bay_id: ((bookingObs?.bay_id ?? args.bay_id ?? prev?.bay_id) === "BAY-B"
            ? "BAY-B"
            : "BAY-A") as BayId,
          bay_start_week: bayStart,
          fab_weeks: fabWeeks,
          fab_end_week: Number(bookingObs?.fab_end_week ?? bayStart + fabWeeks),
          booked_at_step: step.step_index,
          order_release_week: releaseWeek,
          spec,
          manufacturing_cost_eur: Number(bookingObs?.manufacturing_cost_eur ?? phys.manufacturing_cost_eur),
          assigned_route_id:
            typeof args.route_id === "string"
              ? args.route_id
              : typeof bookingObs?.assigned_route_id === "string"
                ? bookingObs.assigned_route_id
                : prev?.assigned_route_id ?? null,
          use_escort_permit:
            typeof args.use_escort_permit === "boolean"
              ? args.use_escort_permit
              : Boolean(bookingObs?.use_escort_permit ?? prev?.use_escort_permit ?? false),
          revision_count: (prev?.revision_count ?? 0) + 1,
          post_shock_revised: Boolean(prev?.post_shock_revised || disruptionActive),
        };
      }
    }

    if (step.tool_name === "finalize_commissioning" && ok) {
      finalized = true;
    }
  }

  const dis = disruptionActive ? world.disruption : null;
  const effectiveBudgetEur = world.initialBudgetEur + (dis?.budget_delta_eur ?? 0);
  const effectivePermitsTotal = Math.max(0, world.initialEscortPermits + (dis?.escort_permits_delta ?? 0));
  const activeBlackouts = [...world.initialBlackouts, ...(dis?.bay_blackouts ?? [])];
  const bayMaxTareKg = dis?.bay_max_unit_tare_kg ?? {};

  // Build effective routes & orders at current step
  const activeRoutes: Record<string, TransportRoute> = {};
  for (const [rid, baseR] of Object.entries(world.routes)) {
    activeRoutes[rid] = {
      ...baseR,
      max_unit_transport_mass_kg: dis?.route_mass_cap_overrides?.[rid] ?? baseR.max_unit_transport_mass_kg,
      std_max_height_mm: dis?.route_std_height_overrides?.[rid] ?? baseR.std_max_height_mm,
      escort_permit_allowed:
        dis?.route_permit_allowed_overrides?.[rid] !== undefined
          ? dis.route_permit_allowed_overrides[rid]
          : baseR.escort_permit_allowed,
    };
  }

  const permitsUsed = Object.values(bookings).filter((b) => b.use_escort_permit).length;
  const permitQuotaExceeded = permitsUsed > effectivePermitsTotal;

  let spentEur = retoolingFeesEur;
  for (const b of Object.values(bookings)) {
    const r = activeRoutes[b.assigned_route_id ?? ""] ?? Object.values(activeRoutes)[0];
    const freight = b.assigned_route_id && r ? b.spec.unit_count * r.base_freight_eur_per_unit : 0;
    const permitFee = b.assigned_route_id && b.use_escort_permit && r ? r.permit_fee_eur : 0;
    spentEur += b.manufacturing_cost_eur + freight + permitFee;
  }

  const isTerminalStep = clampedIdx === totalSteps - 1;
  if (isTerminalStep && run.verification) {
    spentEur = run.verification.total_spent_eur || spentEur;
  }

  const orderSnapshots: OrderLiveSnapshot[] = world.orders.map((baseOrder) => {
    const order: CustomerOrder = {
      ...baseOrder,
      min_pressure_mpa: dis?.order_min_pressure_overrides?.[baseOrder.order_id] ?? baseOrder.min_pressure_mpa,
      sla_week: dis?.order_sla_overrides?.[baseOrder.order_id] ?? baseOrder.sla_week,
    };

    const booking = bookings[order.order_id] ?? null;
    const simSpec = latestSimSpecs[order.order_id];
    const specSource: OrderLiveSnapshot["specSource"] = booking
      ? "booked"
      : simSpec
        ? "simulated"
        : "unconfigured";

    const defaultSpec: VesselDesignSpec = {
      unit_count: 1,
      unit_volume_m3: 10,
      design_pressure_mpa: order.min_pressure_mpa,
      wall_thickness_mm: 8,
      insulation: order.medium === "LH2" || order.medium === "LHe" ? "EVI" : "PVI",
      fittings: "EU_STD",
    };
    const spec = booking?.spec ?? simSpec ?? defaultSpec;

    const routeId = booking?.assigned_route_id ?? order.route_id;
    const route = activeRoutes[routeId] ?? activeRoutes[order.route_id] ?? Object.values(activeRoutes)[0];

    const leadDelta = dis?.fittings_lead_deltas?.[spec.fittings] ?? 0;
    const pviMult = dis?.pvi_k_multiplier ?? 1.0;

    const verifOrder = isTerminalStep
      ? run.verification?.order_results?.find((r) => r.order_id === order.order_id)
      : undefined;

    const physics =
      verifOrder?.physics ??
      evaluateDesignPhysics(order, spec, {
        pvi_k_multiplier: pviMult,
        fittings_lead_delta: leadDelta,
      });

    let fittingsReadyWeek: number | null = null;
    let fabEndWeek: number | null = null;
    let assemblyEndWeek: number | null = null;
    let deliveryWeek: number | null = null;
    const bayConflictReasons: string[] = [];
    const transportViolations: string[] = [];

    if (booking) {
      fittingsReadyWeek = booking.order_release_week + physics.fittings_lead_weeks;
      fabEndWeek = booking.bay_start_week + booking.fab_weeks;
      assemblyEndWeek = Math.max(fabEndWeek, fittingsReadyWeek) + ASSEMBLY_COMMISSION_WEEKS;
      deliveryWeek = verifOrder?.delivery_week ?? assemblyEndWeek + route.transit_weeks;

      // Check blackout overlaps
      for (const blk of activeBlackouts) {
        if (
          blk.bay_id === booking.bay_id &&
          booking.bay_start_week < blk.end_week &&
          fabEndWeek > blk.start_week
        ) {
          bayConflictReasons.push(
            `${booking.bay_id} BLACKOUT [W${blk.start_week}-W${blk.end_week}) CONFLICT`
          );
        }
      }
      // Check peer order bay overlaps
      for (const other of Object.values(bookings)) {
        if (
          other.order_id !== booking.order_id &&
          other.bay_id === booking.bay_id &&
          booking.bay_start_week < other.bay_start_week + other.fab_weeks &&
          fabEndWeek > other.bay_start_week
        ) {
          bayConflictReasons.push(`${booking.bay_id} OVERLAP WITH ${other.order_id}`);
        }
      }
      // Check bay crane tare cap
      const maxTare = bayMaxTareKg[booking.bay_id];
      if (maxTare !== undefined && physics.unit_tare_mass_kg > maxTare) {
        bayConflictReasons.push(
          `${booking.bay_id} CRANE CAP ${Math.round(physics.unit_tare_mass_kg)} > ${Math.round(maxTare)} kg`
        );
      }

      // Check transport envelope
      if (!booking.assigned_route_id) {
        transportViolations.push("ROUTE_UNASSIGNED");
      } else if (booking.assigned_route_id !== order.route_id) {
        transportViolations.push(`WRONG_ROUTE: ${booking.assigned_route_id} != ${order.route_id}`);
      }
      if (physics.unit_transport_mass_kg > route.max_unit_transport_mass_kg) {
        transportViolations.push(
          `BRIDGE_MASS_CAP: ${physics.unit_transport_mass_kg} kg > ${route.max_unit_transport_mass_kg} kg`
        );
      }
      const maxW = booking.use_escort_permit ? route.permit_max_width_mm : route.std_max_width_mm;
      const maxH = booking.use_escort_permit ? route.permit_max_height_mm : route.std_max_height_mm;
      if (booking.use_escort_permit && !route.escort_permit_allowed) {
        transportViolations.push(`ESCORT_PERMIT_FORBIDDEN on ${route.route_id}`);
      }
      if (booking.use_escort_permit && permitQuotaExceeded) {
        transportViolations.push("PERMIT_QUOTA_EXCEEDED");
      }
      if (physics.dimensions.shipping_width_mm > maxW) {
        transportViolations.push(
          `SHIP_WIDTH_EXCEEDED: ${physics.dimensions.shipping_width_mm} mm > ${maxW} mm`
        );
      }
      if (physics.dimensions.shipping_height_mm > maxH) {
        transportViolations.push(
          `SHIP_HEIGHT_EXCEEDED: ${physics.dimensions.shipping_height_mm} mm > ${maxH} mm`
        );
      }
    }

    const slaPass =
      deliveryWeek !== null && deliveryWeek <= order.sla_week && deliveryWeek <= world.horizonWeeks;

    const allViolations = verifOrder
      ? verifOrder.violations
      : [
          ...(!booking ? ["ORDER_NOT_BOOKED"] : []),
          ...physics.violations,
          ...bayConflictReasons,
          ...transportViolations,
          ...(booking && !slaPass && deliveryWeek !== null
            ? [`SLA_EXCEEDED: delivery W${deliveryWeek} > SLA W${order.sla_week}`]
            : []),
        ];

    const passedAllHardChecks = verifOrder
      ? verifOrder.passed_all_hard_checks
      : Boolean(
          booking &&
            booking.assigned_route_id === order.route_id &&
            physics.all_engineering_pass &&
            bayConflictReasons.length === 0 &&
            transportViolations.length === 0 &&
            slaPass
        );

    const blueprintBadges = buildBlueprintBadges(
      order,
      spec,
      physics,
      route,
      booking,
      deliveryWeek,
      world.horizonWeeks,
      bayConflictReasons,
      permitQuotaExceeded
    );

    return {
      order,
      route,
      spec,
      specSource,
      booking,
      physics,
      fittingsReadyWeek,
      fabEndWeek,
      assemblyEndWeek,
      deliveryWeek,
      slaPass,
      bayConflictReasons,
      transportViolations,
      allViolations,
      passedAllHardChecks,
      blueprintBadges,
    };
  });

  return {
    stepArrayIndex: clampedIdx,
    stepNumber: clampedIdx >= 0 ? run.steps[clampedIdx].step_index : 0,
    totalSteps,
    shockArrayIndex,
    shockStepNumber: shockArrayIndex !== null ? run.steps[shockArrayIndex].step_index : null,
    disruptionActive,
    disruptionJustTriggered,
    disruption: world.disruption,
    simCallsUsed,
    simBudgetTotal,
    effectiveBudgetEur,
    spentEur: Math.round(spentEur),
    retoolingFeesEur: Math.round(retoolingFeesEur),
    permitsUsed,
    effectivePermitsTotal,
    activeBlackouts,
    bayMaxTareKg,
    finalized,
    orders: orderSnapshots,
  };
}
