import { FALLBACK_SCORECARD } from "../data/fallbackScorecard";
import type { ExploitAuditItem, ScorecardData, SeedScoreSummary, TrajectoryLog } from "../types/cryoforge";

export type DataSourceMode = "live_api" | "static_artifact" | "bundled_fallback" | "custom_upload";

export interface LoadedStudioBundle {
  scorecard: ScorecardData;
  sourceMode: DataSourceMode;
  sourceLabel: string;
}

function asObj(v: unknown): Record<string, unknown> | null {
  return v !== null && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : null;
}

function isTrajectoryLog(raw: unknown): raw is TrajectoryLog {
  const o = asObj(raw);
  return Boolean(o && typeof o.run_id === "string" && typeof o.seed === "string" && Array.isArray(o.steps));
}

/**
 * Normalizes any `/api/scorecard` or `/scorecard.json` payload produced by `harbor_runner.py` / `api_bridge.py`
 * and backfills any missing seeds/fields from `FALLBACK_SCORECARD` so the UI never breaks.
 */
export function normalizeScorecardPayload(
  rawScorecard: unknown,
  extraRuns: TrajectoryLog[] = []
): ScorecardData {
  const base = FALLBACK_SCORECARD;
  const root = asObj(rawScorecard);
  if (!root) {
    if (extraRuns.length === 0) return base;
    return mergeTrajectoriesIntoScorecard(base, extraRuns);
  }

  // If rawScorecard itself is a single TrajectoryLog uploaded by user
  if (isTrajectoryLog(root)) {
    return mergeTrajectoriesIntoScorecard(base, [root, ...extraRuns]);
  }

  const rub = asObj(root.rubric ?? root.summary ?? root.metrics) ?? root;
  const reference_mean = Number(
    rub.reference_mean ?? rub.reference_solver_mean ?? rub.ref_mean ?? base.rubric.reference_mean
  );
  const frontier_mean = Number(
    rub.frontier_mean ?? rub.frontier_agent_mean ?? base.rubric.frontier_mean
  );
  const frontier_std = Number(
    rub.frontier_std ?? rub.frontier_agent_std ?? base.rubric.frontier_std
  );
  const gap_delta = Number(
    rub.gap_delta ?? rub.the_gap ?? Math.round((reference_mean - frontier_mean) * 1000) / 1000
  );
  const do_nothing_score = Number(
    rub.do_nothing_score ?? asObj(root.baselines)?.do_nothing ?? base.rubric.do_nothing_score
  );
  const stop_early_score = Number(
    rub.stop_early_score ?? asObj(root.baselines)?.stop_early ?? base.rubric.stop_early_score
  );

  // Merge trajectories
  const incomingTrajs: TrajectoryLog[] = [];
  const rawTrajList = Array.isArray(root.trajectories)
    ? root.trajectories
    : Array.isArray(root.runs)
      ? root.runs
      : [];
  for (const item of rawTrajList) {
    if (isTrajectoryLog(item)) incomingTrajs.push(item);
  }
  incomingTrajs.push(...extraRuns);

  // Merge exploits if provided
  let exploits: ExploitAuditItem[] = base.exploits;
  const rawExploits = root.exploits ?? root.exploit_results ?? root.exploit_suite;
  if (Array.isArray(rawExploits) && rawExploits.length > 0) {
    exploits = base.exploits.map((fbExp, idx) => {
      const match =
        rawExploits.find((e) => asObj(e)?.exploit_id === fbExp.exploit_id) ?? rawExploits[idx];
      const mo = asObj(match);
      if (!mo) return fbExp;
      const obsScore = Number(mo.observed_score ?? mo.final_score ?? mo.score ?? fbExp.observed_score);
      const maxAllowed = Number(mo.max_allowed_score ?? 0.08);
      return {
        ...fbExp,
        name: typeof mo.name === "string" ? mo.name : fbExp.name,
        attack_vector:
          typeof mo.attack_vector === "string"
            ? mo.attack_vector
            : typeof mo.description === "string"
              ? mo.description
              : fbExp.attack_vector,
        defense_mechanism:
          typeof mo.defense_mechanism === "string" ? mo.defense_mechanism : fbExp.defense_mechanism,
        observed_score: obsScore,
        max_allowed_score: maxAllowed,
        blocked: typeof mo.blocked === "boolean" ? mo.blocked : obsScore <= maxAllowed,
      };
    });
  }

  const max_exploit_score = Math.max(0, ...exploits.map((e) => e.observed_score));
  const exploits_blocked = exploits.filter((e) => e.blocked).length;

  // Merge per_seed
  let per_seed: SeedScoreSummary[] = base.per_seed;
  if (Array.isArray(root.per_seed) && root.per_seed.length > 0) {
    per_seed = base.per_seed.map((fbSeed) => {
      const found = asObj(root.per_seed && (root.per_seed as unknown[]).find((s) => asObj(s)?.seed === fbSeed.seed));
      if (!found) return fbSeed;
      const refS = Number(found.reference_score ?? fbSeed.reference_score);
      const fMean = Number(found.frontier_mean ?? fbSeed.frontier_mean);
      return {
        ...fbSeed,
        scenario_title: typeof found.scenario_title === "string" ? found.scenario_title : fbSeed.scenario_title,
        shock_headline: typeof found.shock_headline === "string" ? found.shock_headline : fbSeed.shock_headline,
        reference_score: refS,
        frontier_scores: Array.isArray(found.frontier_scores)
          ? (found.frontier_scores as unknown[]).map(Number)
          : fbSeed.frontier_scores,
        frontier_mean: fMean,
        frontier_std: Number(found.frontier_std ?? fbSeed.frontier_std),
        gap_delta: Number(found.gap_delta ?? Math.round((refS - fMean) * 1000) / 1000),
        small_before_score:
          found.small_before_score !== undefined ? Number(found.small_before_score) : fbSeed.small_before_score,
        small_after_score:
          found.small_after_score !== undefined ? Number(found.small_after_score) : fbSeed.small_after_score,
      };
    });
  }

  // Merge tiebreaker
  const rawTb = asObj(root.tiebreaker ?? root.small_model);
  const tiebreaker = rawTb
    ? {
        ...base.tiebreaker,
        model_family: typeof rawTb.model_family === "string" ? rawTb.model_family : base.tiebreaker.model_family,
        adapter_spec: typeof rawTb.adapter_spec === "string" ? rawTb.adapter_spec : base.tiebreaker.adapter_spec,
        train_seeds: Array.isArray(rawTb.train_seeds)
          ? (rawTb.train_seeds as string[])
          : base.tiebreaker.train_seeds,
        held_out_seeds: Array.isArray(rawTb.held_out_seeds)
          ? (rawTb.held_out_seeds as string[])
          : base.tiebreaker.held_out_seeds,
        before_mean: Number(rawTb.before_mean ?? base.tiebreaker.before_mean),
        after_mean: Number(rawTb.after_mean ?? base.tiebreaker.after_mean),
        delta_improvement: Number(
          rawTb.delta_improvement ??
            rawTb.delta ??
            Math.round(
              (Number(rawTb.after_mean ?? base.tiebreaker.after_mean) -
                Number(rawTb.before_mean ?? base.tiebreaker.before_mean)) *
                1000
            ) /
              1000
        ),
        relative_gain_pct:
          rawTb.relative_gain_pct !== undefined
            ? Number(rawTb.relative_gain_pct)
            : base.tiebreaker.relative_gain_pct,
        per_seed:
          Array.isArray(rawTb.per_seed) && rawTb.per_seed.length > 0
            ? (rawTb.per_seed as ScorecardData["tiebreaker"]["per_seed"])
            : base.tiebreaker.per_seed,
      }
    : base.tiebreaker;

  const mergedCard: ScorecardData = {
    generated_at: typeof root.generated_at === "string" ? root.generated_at : base.generated_at,
    environment_version:
      typeof root.environment_version === "string" ? root.environment_version : base.environment_version,
    rubric: {
      reference_mean,
      frontier_mean,
      frontier_std,
      gap_delta,
      do_nothing_score,
      stop_early_score,
      max_exploit_score,
      exploits_blocked,
      exploits_total: exploits.length,
    },
    per_seed,
    failure_taxonomy: Array.isArray(root.failure_taxonomy) && root.failure_taxonomy.length > 0
      ? (root.failure_taxonomy as ScorecardData["failure_taxonomy"])
      : base.failure_taxonomy,
    exploits,
    tiebreaker,
    trajectories: base.trajectories,
  };

  return mergeTrajectoriesIntoScorecard(mergedCard, incomingTrajs);
}

function mergeTrajectoriesIntoScorecard(
  card: ScorecardData,
  incoming: TrajectoryLog[]
): ScorecardData {
  if (incoming.length === 0) return card;
  const byId = new Map<string, TrajectoryLog>();
  const coveredPairs = new Set<string>();
  const coveredSeeds = new Set<string>();

  // Put real incoming trajectories FIRST so they are always default in replay selectors
  for (const inc of incoming) {
    byId.set(inc.run_id, inc);
    coveredPairs.add(`${inc.seed}::${inc.agent_type}`);
    coveredSeeds.add(inc.seed);
  }

  // Only backfill fallback trajectories when incoming doesn't cover all 8 seeds
  if (coveredSeeds.size < 8) {
    for (const t of card.trajectories) {
      if (!byId.has(t.run_id) && !coveredPairs.has(`${t.seed}::${t.agent_type}`)) {
        byId.set(t.run_id, t);
      }
    }
  }

  return {
    ...card,
    trajectories: Array.from(byId.values()),
  };
}

async function fetchJsonQuiet(url: string, timeoutMs = 1800): Promise<unknown | null> {
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      signal: ctrl.signal,
      headers: { Accept: "application/json" },
    });
    if (!res.ok) return null;
    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("text/html")) return null; // SPA fallback guard
    return await res.json();
  } catch {
    return null;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function loadStudioData(): Promise<LoadedStudioBundle> {
  // 1. Try live FastAPI / MCP api_bridge endpoints: /api/scorecard + /api/runs
  const [apiScorecard, apiRunsRaw] = await Promise.all([
    fetchJsonQuiet("/api/scorecard"),
    fetchJsonQuiet("/api/runs"),
  ]);

  if (apiScorecard || apiRunsRaw) {
    const extraRuns: TrajectoryLog[] = [];
    const listCandidate = Array.isArray(apiRunsRaw)
      ? apiRunsRaw
      : Array.isArray(asObj(apiRunsRaw)?.runs)
        ? (asObj(apiRunsRaw)!.runs as unknown[])
        : Array.isArray(asObj(apiRunsRaw)?.trajectories)
          ? (asObj(apiRunsRaw)!.trajectories as unknown[])
          : [];

    const summaryIdsToFetch: string[] = [];
    for (const item of listCandidate) {
      if (isTrajectoryLog(item)) {
        extraRuns.push(item);
      } else {
        const rid = asObj(item)?.run_id;
        if (typeof rid === "string") summaryIdsToFetch.push(rid);
      }
    }

    if (summaryIdsToFetch.length > 0) {
      const detailPayloads = await Promise.all(
        summaryIdsToFetch.slice(0, 24).map((rid) => fetchJsonQuiet(`/api/runs/${encodeURIComponent(rid)}`))
      );
      for (const dp of detailPayloads) {
        if (isTrajectoryLog(dp)) extraRuns.push(dp);
        else if (isTrajectoryLog(asObj(dp)?.trajectory)) {
          extraRuns.push(asObj(dp)!.trajectory as TrajectoryLog);
        }
      }
    }

    return {
      scorecard: normalizeScorecardPayload(apiScorecard, extraRuns),
      sourceMode: "live_api",
      sourceLabel: "LIVE API BRIDGE (/api/scorecard + /api/runs)",
    };
  }

  // 2. Try static public artifact ./scorecard.json
  const staticJson = await fetchJsonQuiet("./scorecard.json");
  if (staticJson) {
    return {
      scorecard: normalizeScorecardPayload(staticJson),
      sourceMode: "static_artifact",
      sourceLabel: "BUNDLED ARTIFACT (/scorecard.json)",
    };
  }

  // 3. Zero-error offline TypeScript fallback
  return {
    scorecard: FALLBACK_SCORECARD,
    sourceMode: "bundled_fallback",
    sourceLabel: "OFFLINE DETERMINISTIC BUNDLE (fallbackScorecard.ts)",
  };
}
