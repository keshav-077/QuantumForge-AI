import React from "react";
import { AlertCircle, CheckCircle2, Wrench, Zap } from "lucide-react";
import type { BayId } from "../types/cryoforge";
import type { ReplayedFrameState } from "../lib/trajectoryEngine";
import {
  formatBayLabel,
  formatOrderLabel,
  SCENARIO_PLAIN_META,
} from "../lib/plainEnglishLabels";

export interface BayGanttTimelineProps {
  seed?: string;
  frame: ReplayedFrameState;
  agentLabel: string;
  accentTone: "emerald" | "rose" | "indigo";
  selectedOrderId: string;
  onSelectOrder: (orderId: string) => void;
  horizonWeeks?: number;
}

const BAYS: BayId[] = ["BAY-A", "BAY-B"];

export const BayGanttTimeline: React.FC<BayGanttTimelineProps> = ({
  seed = "S01",
  frame,
  agentLabel,
  accentTone,
  selectedOrderId,
  onSelectOrder,
  horizonWeeks = 14,
}) => {
  const weeks = Array.from({ length: horizonWeeks }, (_, i) => i + 1);
  const pct = (weekOneBased: number) => ((weekOneBased - 1) / horizonWeeks) * 100;
  const spanPct = (durationWeeks: number) => (durationWeeks / horizonWeeks) * 100;
  const scenarioMeta = SCENARIO_PLAIN_META[seed];

  return (
    <div className="bg-surface border border-hairline rounded-xl overflow-hidden flex flex-col">
      {/* Header */}
      <div className="px-4 py-3 border-b border-hairline flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <span className="text-sm font-semibold text-ink-primary">{agentLabel}</span>
          <span
            className={`px-2 py-0.5 rounded-md text-xs font-medium ${
              accentTone === "emerald"
                ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                : accentTone === "indigo"
                  ? "bg-indigo-50 text-indigo-700 border border-indigo-200"
                  : "bg-rose-50 text-rose-700 border border-rose-200"
            }`}
          >
            {frame.orders.filter((o) => o.passedAllHardChecks).length}/4 Valid
          </span>
        </div>

        <div className="flex items-center gap-3 text-[11px] text-ink-muted">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" />
            Passed
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-rose-500 inline-block" />
            Failed
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm outage-pattern border border-rose-300 inline-block" />
            Outage
          </span>
        </div>
      </div>

      {/* Disruption Strip */}
      <div
        className={`px-4 py-2 border-b border-hairline text-xs flex items-center gap-2 ${
          frame.disruptionActive ? "bg-indigo-50/70 text-indigo-900" : "bg-canvas text-ink-muted"
        }`}
      >
        <Zap
          className={`w-3.5 h-3.5 shrink-0 ${
            frame.disruptionActive ? "text-indigo-600" : "text-ink-muted"
          }`}
        />
        <span className="truncate">
          {frame.disruptionActive
            ? `Step-4 Disruption active: ${scenarioMeta?.disruptionSummary ?? frame.disruption?.headline ?? ""}`
            : `Pre-disruption plan (disruption hits at Step ${frame.shockStepNumber ?? 4})`}
        </span>
      </div>

      {/* Grid */}
      <div className="p-4 space-y-4 overflow-x-auto">
        <div className="min-w-[520px]">
          <div className="flex items-center mb-2">
            <div className="w-20 shrink-0 text-[11px] font-medium text-ink-muted">Lane</div>
            <div
              className="flex-1 grid border-l border-hairline"
              style={{ gridTemplateColumns: `repeat(${horizonWeeks}, minmax(0, 1fr))` }}
            >
              {weeks.map((w) => (
                <div
                  key={w}
                  className="text-center text-[10px] font-mono text-ink-muted border-r border-hairline/70 py-0.5"
                >
                  W{w}
                </div>
              ))}
            </div>
            <div className="w-28 shrink-0 pl-3 text-right text-[11px] font-medium text-ink-muted">
              Delivery
            </div>
          </div>

          <div className="space-y-2 mb-4">
            {BAYS.map((bayId) => {
              const bayBlackouts = frame.activeBlackouts.filter((b) => b.bay_id === bayId);
              const bayOrders = frame.orders.filter((o) => o.booking?.bay_id === bayId);
              const tareCap = frame.bayMaxTareKg[bayId];

              return (
                <div key={bayId} className="flex items-center">
                  <div className="w-20 shrink-0 pr-2">
                    <div className="text-xs font-medium text-ink-primary">
                      {formatBayLabel(bayId)}
                    </div>
                    {tareCap !== undefined && (
                      <div className="text-[10px] text-rose-600">
                        Max {Math.round(tareCap).toLocaleString()}kg
                      </div>
                    )}
                  </div>

                  <div className="flex-1 relative h-9 bg-canvas border border-hairline rounded-lg overflow-hidden">
                    <div
                      className="absolute inset-0 grid pointer-events-none"
                      style={{ gridTemplateColumns: `repeat(${horizonWeeks}, minmax(0, 1fr))` }}
                    >
                      {weeks.map((w) => (
                        <div key={w} className="border-r border-hairline/50 h-full" />
                      ))}
                    </div>

                    {bayBlackouts.map((blk, idx) => (
                      <div
                        key={`${blk.bay_id}-${blk.start_week}-${idx}`}
                        style={{
                          left: `${pct(blk.start_week)}%`,
                          width: `${spanPct(blk.end_week - blk.start_week)}%`,
                        }}
                        title={`${formatBayLabel(blk.bay_id)} Outage [W${blk.start_week}–W${blk.end_week})`}
                        className="absolute inset-y-0 outage-pattern border-x border-rose-300 flex items-center justify-center z-10"
                      >
                        <span className="text-[10px] font-medium text-rose-700 flex items-center gap-0.5">
                          <Wrench className="w-2.5 h-2.5" /> Outage
                        </span>
                      </div>
                    ))}

                    {bayOrders.map((snap) => {
                      const b = snap.booking!;
                      const isSelected = snap.order.order_id === selectedOrderId;
                      const isBad = !snap.passedAllHardChecks;

                      return (
                        <button
                          key={snap.order.order_id}
                          type="button"
                          onClick={() => onSelectOrder(snap.order.order_id)}
                          style={{
                            left: `${pct(b.bay_start_week)}%`,
                            width: `${spanPct(b.fab_weeks)}%`,
                          }}
                          className={`absolute top-1 bottom-1 rounded-md px-2 flex items-center justify-between text-xs font-medium transition z-20 ${
                            isBad
                              ? "bg-rose-100 border border-rose-300 text-rose-900"
                              : "bg-emerald-100 border border-emerald-300 text-emerald-900"
                          } ${isSelected ? "ring-2 ring-indigo-500" : "hover:brightness-95"}`}
                        >
                          <span className="truncate">{formatOrderLabel(snap.order.order_id)}</span>
                          {b.post_shock_revised && (
                            <span className="ml-1 text-[10px] font-semibold text-indigo-700 shrink-0">
                              Rev
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>

                  <div className="w-28 shrink-0 pl-3 text-right text-xs text-ink-muted">
                    {bayOrders.length} booked
                  </div>
                </div>
              );
            })}
          </div>

          {/* Per-Task SLA Status Rows */}
          <div className="pt-3 border-t border-hairline space-y-1.5">
            {frame.orders.map((snap) => {
              const b = snap.booking;
              const isSelected = snap.order.order_id === selectedOrderId;
              const slaW = snap.order.sla_week;
              const delW = snap.deliveryWeek;
              const fitW = snap.fittingsReadyWeek;
              const fabEnd = snap.fabEndWeek;

              return (
                <div
                  key={snap.order.order_id}
                  onClick={() => onSelectOrder(snap.order.order_id)}
                  className={`flex items-center cursor-pointer rounded-lg px-1 py-1 transition ${
                    isSelected ? "bg-indigo-50/80" : "hover:bg-subtle/60"
                  }`}
                >
                  <div className="w-20 shrink-0 pr-2 flex items-center gap-1.5">
                    {snap.passedAllHardChecks ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                    ) : (
                      <AlertCircle className="w-3.5 h-3.5 text-rose-600 shrink-0" />
                    )}
                    <span className="text-xs font-medium text-ink-primary">
                      {formatOrderLabel(snap.order.order_id)}
                    </span>
                  </div>

                  <div className="flex-1 relative h-5 bg-canvas border border-hairline rounded overflow-hidden">
                    <div
                      className="absolute inset-0 grid pointer-events-none"
                      style={{ gridTemplateColumns: `repeat(${horizonWeeks}, minmax(0, 1fr))` }}
                    >
                      {weeks.map((w) => (
                        <div key={w} className="border-r border-hairline/40 h-full" />
                      ))}
                    </div>

                    {b && fitW !== null && (
                      <div
                        style={{
                          left: `${pct(b.order_release_week)}%`,
                          width: `${spanPct(Math.max(1, fitW - b.order_release_week))}%`,
                        }}
                        className="absolute top-1 h-1 bg-indigo-300 rounded-full"
                      />
                    )}

                    {b && delW !== null && fabEnd !== null && fitW !== null && (
                      <div
                        style={{
                          left: `${pct(Math.max(fabEnd, fitW))}%`,
                          width: `${spanPct(Math.max(1, delW - Math.max(fabEnd, fitW)))}%`,
                        }}
                        className={`absolute bottom-1 h-1.5 rounded-sm ${
                          snap.slaPass ? "bg-emerald-500" : "bg-rose-500"
                        }`}
                      />
                    )}

                    <div
                      style={{ left: `${Math.min(99, pct(slaW + 1) - 0.5)}%` }}
                      title={`SLA Deadline: Week ${slaW}`}
                      className="absolute inset-y-0 w-0.5 bg-indigo-500 z-20"
                    />
                  </div>

                  <div className="w-28 shrink-0 pl-3 text-right text-xs font-mono tabular-nums">
                    {delW !== null ? (
                      <span className={snap.slaPass ? "text-emerald-700 font-semibold" : "text-rose-600 font-semibold"}>
                        W{delW} / SLA W{slaW}
                      </span>
                    ) : (
                      <span className="text-ink-muted">— / SLA W{slaW}</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
