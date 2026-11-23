import React, { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  FastForward,
  Pause,
  Play,
  RotateCcw,
  Terminal,
  Zap,
} from "lucide-react";
import type { TrajectoryLog } from "../types/cryoforge";
import {
  extractSeedWorldMetadata,
  findShockArrayIndex,
  replayTrajectoryAtStep,
} from "../lib/trajectoryEngine";
import {
  formatAgentDisplayName,
  formatBayLabel,
  formatConfigTier,
  formatOrderLabel,
  formatSeedLabel,
  humanizeViolationString,
  SCENARIO_PLAIN_META,
} from "../lib/plainEnglishLabels";
import { BayGanttTimeline } from "./BayGanttTimeline";

export interface FocusedIncidentTarget {
  seed: string;
  run_id: string;
  order_id: string;
  step_index: number;
  nonce: number;
}

export interface TrajectoryDiffProps {
  seed: string;
  onSelectSeed: (seed: string) => void;
  seedRuns: TrajectoryLog[];
  focusedIncident?: FocusedIncidentTarget | null;
  preferTiebreakerPair?: boolean;
}

const ALL_SCENARIO_KEYS = ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"] as const;

export const TrajectoryDiff: React.FC<TrajectoryDiffProps> = ({
  seed,
  onSelectSeed,
  seedRuns,
  focusedIncident,
  preferTiebreakerPair = false,
}) => {
  const hasSmallPair = useMemo(
    () =>
      seedRuns.some((r) => r.agent_type === "small_after") &&
      seedRuns.some((r) => r.agent_type === "small_before"),
    [seedRuns]
  );

  const [leftRunId, setLeftRunId] = useState<string>("");
  const [rightRunId, setRightRunId] = useState<string>("");
  const [scrubStep, setScrubStep] = useState<number>(999);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [selectedOrderId, setSelectedOrderId] = useState<string>("");
  const [showRawLogs, setShowRawLogs] = useState<boolean>(false);

  useEffect(() => {
    setIsPlaying(false);
    const refRun = seedRuns.find((r) => r.agent_type === "reference") ?? seedRuns[0];
    const frontierCandidates = seedRuns
      .filter((r) => r.agent_type === "frontier")
      .sort((a, b) => a.verification.final_score - b.verification.final_score);
    const frontRun = frontierCandidates[0] ?? seedRuns[1] ?? seedRuns[0];
    const saRun = seedRuns.find((r) => r.agent_type === "small_after");
    const sbRun = seedRuns.find((r) => r.agent_type === "small_before");

    if (preferTiebreakerPair && saRun && sbRun) {
      setLeftRunId(saRun.run_id);
      setRightRunId(sbRun.run_id);
    } else {
      setLeftRunId(refRun?.run_id ?? "");
      setRightRunId(frontRun?.run_id ?? "");
    }
    const maxLen = Math.max(...seedRuns.map((r) => r.steps.length), 1);
    setScrubStep(maxLen - 1);
  }, [seed, seedRuns, preferTiebreakerPair]);

  const world = useMemo(() => extractSeedWorldMetadata(seed, seedRuns), [seed, seedRuns]);

  useEffect(() => {
    if (world.orders.length > 0 && !world.orders.some((o) => o.order_id === selectedOrderId)) {
      setSelectedOrderId(world.orders[0].order_id);
    }
  }, [world, selectedOrderId]);

  useEffect(() => {
    if (!focusedIncident || focusedIncident.seed !== seed) return;
    setIsPlaying(false);
    const targetRun = seedRuns.find((r) => r.run_id === focusedIncident.run_id);
    if (targetRun) {
      if (targetRun.agent_type === "small_before") {
        const sa = seedRuns.find((r) => r.agent_type === "small_after");
        if (sa) setLeftRunId(sa.run_id);
      } else {
        const ref = seedRuns.find((r) => r.agent_type === "reference");
        if (ref) setLeftRunId(ref.run_id);
      }
      setRightRunId(targetRun.run_id);
      const arrIdx = targetRun.steps.findIndex((s) => s.step_index >= focusedIncident.step_index);
      setScrubStep(arrIdx >= 0 ? Math.max(arrIdx, targetRun.steps.length - 1) : targetRun.steps.length - 1);
    }
    if (focusedIncident.order_id && focusedIncident.order_id !== "BATCH") {
      setSelectedOrderId(focusedIncident.order_id);
    }
    setShowRawLogs(true);
  }, [focusedIncident, seed, seedRuns]);

  const leftRun = seedRuns.find((r) => r.run_id === leftRunId) ?? seedRuns[0];
  const rightRun = seedRuns.find((r) => r.run_id === rightRunId) ?? seedRuns[1] ?? seedRuns[0];

  const maxSteps = Math.max(leftRun?.steps.length ?? 1, rightRun?.steps.length ?? 1);
  const clampedStep = Math.max(0, Math.min(scrubStep, maxSteps - 1));

  useEffect(() => {
    if (!isPlaying) return;
    const id = window.setInterval(() => {
      setScrubStep((prev) => {
        if (prev >= maxSteps - 1) {
          setIsPlaying(false);
          return maxSteps - 1;
        }
        return prev + 1;
      });
    }, 850);
    return () => window.clearInterval(id);
  }, [isPlaying, maxSteps]);

  const leftFrame = useMemo(
    () => replayTrajectoryAtStep(leftRun, clampedStep, world),
    [leftRun, clampedStep, world]
  );
  const rightFrame = useMemo(
    () => replayTrajectoryAtStep(rightRun, clampedStep, world),
    [rightRun, clampedStep, world]
  );

  const shockIdx = useMemo(() => {
    const rShock = rightRun ? findShockArrayIndex(rightRun) : null;
    const lShock = leftRun ? findShockArrayIndex(leftRun) : null;
    return rShock ?? lShock ?? 3;
  }, [leftRun, rightRun]);

  if (!leftRun || !rightRun) {
    return (
      <div className="p-8 text-center text-ink-muted text-sm bg-surface border border-hairline rounded-xl">
        No recorded runs available for {formatSeedLabel(seed)}.
      </div>
    );
  }

  const activeOrderId = selectedOrderId || world.orders[0]?.order_id || "";
  const leftStepObj = leftRun.steps[Math.min(clampedStep, leftRun.steps.length - 1)];
  const rightStepObj = rightRun.steps[Math.min(clampedStep, rightRun.steps.length - 1)];
  const scenarioMeta = SCENARIO_PLAIN_META[seed];

  return (
    <div className="space-y-6">
      {/* Clean Scenario Picker & Step-4 Disruption Summary */}
      <div className="bg-surface border border-hairline rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-1 flex-1 min-w-[260px]">
            <label className="block text-xs font-medium text-ink-muted">
              Select benchmark scenario (Scenarios 1–5 evaluate Frontier AI; 6–8 are held-out generalization tests)
            </label>
            <select
              value={seed}
              onChange={(e) => onSelectSeed(e.target.value)}
              className="w-full sm:w-auto min-w-[340px] bg-canvas border border-hairline rounded-lg px-3 py-2 text-sm font-medium text-ink-primary focus:outline-none focus:border-indigo-500"
            >
              {ALL_SCENARIO_KEYS.map((key) => {
                const m = SCENARIO_PLAIN_META[key];
                return (
                  <option key={key} value={key}>
                    {m.shortName}: {m.title}
                  </option>
                );
              })}
            </select>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {hasSmallPair && (
              <div className="inline-flex rounded-lg bg-canvas p-1 border border-hairline text-xs">
                <button
                  type="button"
                  onClick={() => {
                    const ref = seedRuns.find((r) => r.agent_type === "reference") ?? seedRuns[0];
                    const fr =
                      seedRuns
                        .filter((r) => r.agent_type === "frontier")
                        .sort((a, b) => a.verification.final_score - b.verification.final_score)[0] ??
                      seedRuns[1];
                    if (ref && fr) {
                      setLeftRunId(ref.run_id);
                      setRightRunId(fr.run_id);
                    }
                  }}
                  className={`px-3 py-1 rounded-md font-medium transition ${
                    leftRun.agent_type === "reference"
                      ? "bg-surface text-ink-primary shadow-sm border border-hairline"
                      : "text-ink-muted hover:text-ink-primary"
                  }`}
                >
                  Reference vs Frontier
                </button>
                <button
                  type="button"
                  onClick={() => {
                    const sa = seedRuns.find((r) => r.agent_type === "small_after");
                    const sb = seedRuns.find((r) => r.agent_type === "small_before");
                    if (sa && sb) {
                      setLeftRunId(sa.run_id);
                      setRightRunId(sb.run_id);
                    }
                  }}
                  className={`px-3 py-1 rounded-md font-medium transition ${
                    leftRun.agent_type === "small_after"
                      ? "bg-surface text-ink-primary shadow-sm border border-hairline"
                      : "text-ink-muted hover:text-ink-primary"
                  }`}
                >
                  Small Model (After vs Before)
                </button>
              </div>
            )}
          </div>
        </div>

        {scenarioMeta && (
          <div className="p-3.5 rounded-lg bg-canvas border border-hairline text-xs text-ink-secondary flex flex-wrap items-center justify-between gap-2">
            <div>
              <span className="font-semibold text-ink-primary">What changes mid-run (Step 4): </span>
              {scenarioMeta.disruptionSummary}
            </div>
          </div>
        )}
      </div>

      {/* 2 Clean Score Comparison Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-surface border border-hairline rounded-xl p-5 flex items-center justify-between gap-4">
          <div>
            <div className="text-xs font-medium text-emerald-700">Left Agent (Benchmark Target)</div>
            <select
              value={leftRun.run_id}
              onChange={(e) => setLeftRunId(e.target.value)}
              className="mt-1 bg-transparent text-base font-semibold text-ink-primary focus:outline-none cursor-pointer"
            >
              {seedRuns.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {formatAgentDisplayName(r.agent_name, r.agent_type)}
                </option>
              ))}
            </select>
            <div className="text-xs text-ink-muted mt-1">
              {leftRun.verification.valid_order_count}/4 tasks passed · {leftFrame.simCallsUsed}/
              {leftFrame.simBudgetTotal} simulation calls used
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-semibold font-mono text-emerald-600 tabular-nums">
              {leftRun.verification.final_score.toFixed(2)}
            </div>
            <div className="text-xs text-ink-muted">Final Score</div>
          </div>
        </div>

        <div className="bg-surface border border-hairline rounded-xl p-5 flex items-center justify-between gap-4">
          <div>
            <div className="text-xs font-medium text-rose-600">Right Agent (Evaluated Run)</div>
            <select
              value={rightRun.run_id}
              onChange={(e) => setRightRunId(e.target.value)}
              className="mt-1 bg-transparent text-base font-semibold text-ink-primary focus:outline-none cursor-pointer"
            >
              {seedRuns.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {formatAgentDisplayName(r.agent_name, r.agent_type)} ({r.run_id.slice(-2)})
                </option>
              ))}
            </select>
            <div className="text-xs text-ink-muted mt-1">
              {rightRun.verification.valid_order_count}/4 tasks passed · {rightFrame.simCallsUsed}/
              {rightFrame.simBudgetTotal} simulation calls used
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-semibold font-mono text-rose-600 tabular-nums">
              {rightRun.verification.final_score.toFixed(2)}
            </div>
            <div className="text-xs text-ink-muted">Final Score</div>
          </div>
        </div>
      </div>

      {/* Step Replay Scrubber */}
      <div className="bg-surface border border-hairline rounded-xl p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="text-xs text-ink-secondary">
            Step <strong className="text-ink-primary">{clampedStep + 1}</strong> of {maxSteps}
            {clampedStep === shockIdx && (
              <span className="ml-2 px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 font-medium">
                Step-4 Disruption Fires Here
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => {
                setIsPlaying(false);
                setScrubStep(0);
              }}
              className="p-1.5 rounded-lg border border-hairline hover:bg-canvas text-ink-secondary"
              title="Reset to Step 1"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => {
                setIsPlaying(false);
                setScrubStep((s) => Math.max(0, s - 1));
              }}
              className="p-1.5 rounded-lg border border-hairline hover:bg-canvas text-ink-secondary"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => {
                if (clampedStep >= maxSteps - 1) setScrubStep(0);
                setIsPlaying((p) => !p);
              }}
              className="px-3 py-1.5 rounded-lg bg-ink-primary hover:bg-slate-800 text-white text-xs font-medium flex items-center gap-1.5"
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {isPlaying ? "Pause" : "Replay"}
            </button>
            <button
              type="button"
              onClick={() => {
                setIsPlaying(false);
                setScrubStep((s) => Math.min(maxSteps - 1, s + 1));
              }}
              className="p-1.5 rounded-lg border border-hairline hover:bg-canvas text-ink-secondary"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => {
                setIsPlaying(false);
                setScrubStep(shockIdx);
              }}
              className="px-3 py-1.5 rounded-lg border border-indigo-200 bg-indigo-50 text-indigo-700 hover:bg-indigo-100 text-xs font-medium flex items-center gap-1"
            >
              <Zap className="w-3.5 h-3.5" />
              Step-4 Shock
            </button>
            <button
              type="button"
              onClick={() => {
                setIsPlaying(false);
                setScrubStep(maxSteps - 1);
              }}
              className="px-3 py-1.5 rounded-lg border border-hairline hover:bg-canvas text-ink-secondary text-xs font-medium flex items-center gap-1"
            >
              <FastForward className="w-3.5 h-3.5" />
              Final
            </button>
          </div>
        </div>

        <input
          type="range"
          min={0}
          max={Math.max(0, maxSteps - 1)}
          value={clampedStep}
          onChange={(e) => {
            setIsPlaying(false);
            setScrubStep(Number(e.target.value));
          }}
          className="w-full accent-indigo-600 h-1.5 bg-hairline rounded-lg cursor-pointer"
        />
      </div>

      {/* Clean 4-Task Outcome Table */}
      <div className="bg-surface border border-hairline rounded-xl overflow-hidden">
        <div className="px-5 py-3.5 border-b border-hairline flex items-center justify-between">
          <h3 className="text-sm font-semibold text-ink-primary">
            Task-by-Task Outcome Comparison
          </h3>
          <span className="text-xs text-ink-muted">Click any row to highlight on timeline</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-canvas border-b border-hairline text-ink-muted font-medium">
                <th className="py-2.5 px-5">Task</th>
                <th className="py-2.5 px-4">{formatAgentDisplayName(leftRun.agent_name, leftRun.agent_type)}</th>
                <th className="py-2.5 px-4">{formatAgentDisplayName(rightRun.agent_name, rightRun.agent_type)}</th>
                <th className="py-2.5 px-5">Why the Right Agent Failed (or Passed)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-hairline">
              {world.orders.map((ord) => {
                const lSnap = leftFrame.orders.find((o) => o.order.order_id === ord.order_id);
                const rSnap = rightFrame.orders.find((o) => o.order.order_id === ord.order_id);
                const isSelected = ord.order_id === activeOrderId;
                return (
                  <tr
                    key={ord.order_id}
                    onClick={() => setSelectedOrderId(ord.order_id)}
                    className={`cursor-pointer transition ${
                      isSelected ? "bg-indigo-50/60" : "hover:bg-canvas/70"
                    }`}
                  >
                    <td className="py-3.5 px-5 align-top">
                      <div className="font-semibold text-ink-primary">
                        {formatOrderLabel(ord.order_id)}
                      </div>
                      <div className="text-ink-muted mt-0.5">
                        {ord.client_name} · SLA Week {ord.sla_week}
                      </div>
                    </td>

                    <td className="py-3.5 px-4 align-top">
                      <div className="flex items-center gap-1.5">
                        {lSnap?.passedAllHardChecks ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium">
                            <CheckCircle2 className="w-3 h-3" /> Passed
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-rose-50 text-rose-700 border border-rose-200 font-medium">
                            <AlertCircle className="w-3 h-3" /> Failed
                          </span>
                        )}
                      </div>
                      {lSnap?.booking && (
                        <div className="text-ink-muted mt-1">
                          {formatBayLabel(lSnap.booking.bay_id)} · {formatConfigTier(lSnap.spec.insulation)} ·{" "}
                          {lSnap.deliveryWeek ? `Week ${lSnap.deliveryWeek}` : "—"}
                        </div>
                      )}
                    </td>

                    <td className="py-3.5 px-4 align-top">
                      <div className="flex items-center gap-1.5">
                        {rSnap?.passedAllHardChecks ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium">
                            <CheckCircle2 className="w-3 h-3" /> Passed
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-rose-50 text-rose-700 border border-rose-200 font-medium">
                            <AlertCircle className="w-3 h-3" /> Failed
                          </span>
                        )}
                      </div>
                      {rSnap?.booking && (
                        <div className="text-ink-muted mt-1">
                          {formatBayLabel(rSnap.booking.bay_id)} · {formatConfigTier(rSnap.spec.insulation)} ·{" "}
                          {rSnap.deliveryWeek ? `Week ${rSnap.deliveryWeek}` : "—"}
                        </div>
                      )}
                    </td>

                    <td className="py-3.5 px-5 align-top max-w-md">
                      {rSnap?.passedAllHardChecks ? (
                        <span className="text-emerald-700">
                          Met safety, route clearance, and Week-{ord.sla_week} deadline.
                        </span>
                      ) : rSnap?.allViolations && rSnap.allViolations.length > 0 ? (
                        <span className="text-rose-700">
                          {humanizeViolationString(rSnap.allViolations[0])}
                        </span>
                      ) : (
                        <span className="text-ink-muted">Not yet scheduled at Step {clampedStep + 1}.</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Side-by-Side 2-Lane Schedule Timelines */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <BayGanttTimeline
          seed={seed}
          frame={leftFrame}
          agentLabel={formatAgentDisplayName(leftRun.agent_name, leftRun.agent_type)}
          accentTone={leftRun.agent_type === "small_after" ? "indigo" : "emerald"}
          selectedOrderId={activeOrderId}
          onSelectOrder={setSelectedOrderId}
          horizonWeeks={world.horizonWeeks}
        />
        <BayGanttTimeline
          seed={seed}
          frame={rightFrame}
          agentLabel={formatAgentDisplayName(rightRun.agent_name, rightRun.agent_type)}
          accentTone="rose"
          selectedOrderId={activeOrderId}
          onSelectOrder={setSelectedOrderId}
          horizonWeeks={world.horizonWeeks}
        />
      </div>

      {/* Optional Collapsible Raw MCP Tool Logs */}
      <div className="bg-surface border border-hairline rounded-xl overflow-hidden">
        <button
          type="button"
          onClick={() => setShowRawLogs((v) => !v)}
          className="w-full px-5 py-3.5 flex items-center justify-between text-left hover:bg-canvas/70 transition"
        >
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-ink-muted" />
            <span className="text-sm font-semibold text-ink-primary">
              Inspect Raw MCP Tool Calls (Step {clampedStep + 1})
            </span>
          </div>
          <span className="text-xs font-medium text-indigo-600">
            {showRawLogs ? "Hide Logs" : "Show Tool Call Logs"}
          </span>
        </button>

        {showRawLogs && (
          <div className="p-4 border-t border-hairline grid grid-cols-1 xl:grid-cols-2 gap-4 bg-canvas">
            {[
              { run: leftRun, stepObj: leftStepObj },
              { run: rightRun, stepObj: rightStepObj },
            ].map(({ run, stepObj }) => (
              <div key={run.run_id} className="bg-surface border border-hairline rounded-lg p-3.5 space-y-2.5">
                <div className="flex items-center justify-between text-xs font-semibold text-ink-primary border-b border-hairline pb-2">
                  <span>{formatAgentDisplayName(run.agent_name, run.agent_type)}</span>
                  <span className="font-mono text-ink-muted">
                    Step {stepObj?.step_index ?? 0}/{run.steps.length}
                  </span>
                </div>
                {stepObj && (
                  <div className="space-y-2 font-mono text-[11px] max-h-56 overflow-y-auto">
                    <div className="text-indigo-700 font-semibold">{stepObj.tool_name}()</div>
                    <pre className="p-2 rounded bg-canvas border border-hairline text-ink-secondary whitespace-pre-wrap break-all">
                      {JSON.stringify(stepObj.arguments, null, 2)}
                    </pre>
                    <pre className="p-2 rounded bg-canvas border border-hairline text-emerald-800 whitespace-pre-wrap break-all">
                      {JSON.stringify(stepObj.observation, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
