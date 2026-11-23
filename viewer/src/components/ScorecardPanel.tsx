import React, { useState } from "react";
import { CheckCircle2, Lock, ShieldCheck, Terminal, TrendingUp } from "lucide-react";
import type { ScorecardData } from "../types/cryoforge";
import { humanizeViolationString } from "../lib/plainEnglishLabels";

export interface ScorecardPanelProps {
  scorecard: ScorecardData;
}

const PROGRESS_CURVE_POINTS = [
  { orders: 0, label: "0 Valid", score: 0.0 },
  { orders: 1, label: "1 Valid (Stop-Early Cap)", score: 0.08 },
  { orders: 2, label: "2 Valid (Frontier Avg)", score: 0.34 },
  { orders: 3, label: "3 Valid", score: 0.587 },
  { orders: 4, label: "4 Valid + Bonus", score: 0.949 },
];

export const ScorecardPanel: React.FC<ScorecardPanelProps> = ({ scorecard }) => {
  const [selectedExploitId, setSelectedExploitId] = useState<string>(
    scorecard.exploits[0]?.exploit_id ?? ""
  );
  const selectedExploit =
    scorecard.exploits.find((e) => e.exploit_id === selectedExploitId) ?? scorecard.exploits[0];

  return (
    <div className="space-y-6">
      {/* Top Row */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <div className="lg:col-span-7 bg-surface border border-hairline rounded-xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-2">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-600" />
                <h2 className="text-base font-semibold text-ink-primary">
                  Anti-Cherry-Picking Reward Curve
                </h2>
              </div>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                Strictly Monotonic
              </span>
            </div>
            <p className="text-sm text-ink-secondary mb-5 leading-relaxed">
              Doing nothing scores <strong>0.00</strong>. Solving only 1 easy task and quitting before Step 4 is capped at{" "}
              <strong className="text-rose-600">0.08</strong>. Efficiency bonuses only unlock once all{" "}
              <strong className="text-emerald-700">4/4 tasks</strong> pass post-disruption verification.
            </p>
          </div>

          <div className="bg-canvas border border-hairline rounded-xl p-4">
            <svg viewBox="0 0 640 215" className="w-full h-auto select-none">
              {[0, 0.25, 0.5, 0.75, 1.0].map((val) => {
                const y = 175 - val * 145;
                return (
                  <g key={val}>
                    <line x1="56" y1={y} x2="605" y2={y} stroke="#E5E7EB" strokeWidth="1" />
                    <text
                      x="48"
                      y={y + 4}
                      textAnchor="end"
                      fill="#6B7280"
                      fontSize="10"
                      fontFamily="JetBrains Mono, monospace"
                    >
                      {val.toFixed(2)}
                    </text>
                  </g>
                );
              })}

              <line
                x1="56"
                y1={175 - 0.08 * 145}
                x2="605"
                y2={175 - 0.08 * 145}
                stroke="#E11D48"
                strokeWidth="1.5"
                strokeDasharray="6 4"
              />
              <text
                x="598"
                y={175 - 0.08 * 145 - 6}
                textAnchor="end"
                fill="#E11D48"
                fontSize="10"
                fontWeight="600"
                fontFamily="Inter, sans-serif"
              >
                Stop-Early & Exploit Ceiling (0.08)
              </text>

              <polyline
                fill="none"
                stroke="#4F46E5"
                strokeWidth="3"
                points={PROGRESS_CURVE_POINTS.map((pt, idx) => {
                  const x = 90 + idx * 122;
                  const y = 175 - pt.score * 145;
                  return `${x},${y}`;
                }).join(" ")}
              />

              {PROGRESS_CURVE_POINTS.map((pt, idx) => {
                const x = 90 + idx * 122;
                const y = 175 - pt.score * 145;
                const fill =
                  pt.orders === 4 ? "#059669" : pt.orders <= 1 ? "#E11D48" : "#4F46E5";
                return (
                  <g key={pt.orders}>
                    <circle cx={x} cy={y} r="5.5" fill="#FFFFFF" stroke={fill} strokeWidth="3" />
                    <text
                      x={x}
                      y={y - 12}
                      textAnchor="middle"
                      fill={fill}
                      fontSize="11"
                      fontWeight="700"
                      fontFamily="JetBrains Mono, monospace"
                    >
                      {pt.score.toFixed(2)}
                    </text>
                    <text
                      x={x}
                      y="196"
                      textAnchor="middle"
                      fill="#374151"
                      fontSize="11"
                      fontWeight="500"
                      fontFamily="Inter, sans-serif"
                    >
                      {pt.orders}/4 Tasks
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        <div className="lg:col-span-5 bg-surface border border-hairline rounded-xl p-6 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Lock className="w-4 h-4 text-indigo-600" />
              <h2 className="text-base font-semibold text-ink-primary">
                4 Server Safeguards Against Cheating
              </h2>
            </div>
            <div className="space-y-3 mt-4 text-xs">
              {[
                {
                  title: "1. Isolated Sandbox",
                  desc: "Agents interact strictly through MCP tool calls and cannot read or modify scenario definitions or verifier logic.",
                },
                {
                  title: "2. Anti-Oracle Booking",
                  desc: "Scheduling a task never leaks hidden constraint pass/fail booleans—preventing free brute-force search.",
                },
                {
                  title: "3. Terminal Lock",
                  desc: "Unfinalized episodes score 0.00. Once submitted, state is locked and signed with HMAC-SHA256.",
                },
                {
                  title: "4. Full Post-Shock Verification",
                  desc: "Final scores recompute every constraint from scratch under the Step-4 modified rules.",
                },
              ].map((s) => (
                <div key={s.title} className="p-3 rounded-lg bg-canvas border border-hairline">
                  <div className="font-semibold text-ink-primary">{s.title}</div>
                  <div className="text-ink-secondary mt-0.5 leading-relaxed">{s.desc}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2.5 pt-2 text-center">
            <div className="p-3 rounded-lg bg-canvas border border-hairline">
              <div className="text-[11px] text-ink-muted">Do-Nothing</div>
              <div className="text-lg font-mono font-semibold text-emerald-600">
                {scorecard.rubric.do_nothing_score.toFixed(2)}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-canvas border border-hairline">
              <div className="text-[11px] text-ink-muted">Stop-Early Cap</div>
              <div className="text-lg font-mono font-semibold text-emerald-600">
                {scorecard.rubric.stop_early_score.toFixed(2)}
              </div>
            </div>
            <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200">
              <div className="text-[11px] text-emerald-700">Blocked</div>
              <div className="text-lg font-mono font-semibold text-emerald-700">
                {scorecard.rubric.exploits_blocked}/{scorecard.rubric.exploits_total}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 8 Blocked Exploit Probes */}
      <div className="bg-surface border border-hairline rounded-xl overflow-hidden">
        <div className="px-6 py-4 border-b border-hairline flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <h3 className="text-sm font-semibold text-ink-primary">
              8 Automated Red-Team Exploit Tests
            </h3>
          </div>
          <span className="text-xs text-emerald-700 font-medium">
            All {scorecard.rubric.exploits_blocked}/{scorecard.rubric.exploits_total} exploits capped ≤ 0.08
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 divide-y lg:divide-y-0 lg:divide-x divide-hairline">
          <div className="lg:col-span-7 divide-y divide-hairline">
            {scorecard.exploits.map((exp, idx) => {
              const active = exp.exploit_id === selectedExploit?.exploit_id;
              return (
                <button
                  key={exp.exploit_id}
                  type="button"
                  onClick={() => setSelectedExploitId(exp.exploit_id)}
                  className={`w-full text-left px-6 py-3.5 transition flex items-center justify-between gap-3 ${
                    active ? "bg-indigo-50/70" : "hover:bg-canvas"
                  }`}
                >
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-ink-primary truncate">
                      {idx + 1}. {exp.name}
                    </div>
                    <div className="text-xs text-ink-muted truncate mt-0.5">{exp.category}</div>
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <span className="text-xs font-mono text-ink-secondary tabular-nums">
                      Score {exp.observed_score.toFixed(2)}
                    </span>
                    <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" /> Blocked
                    </span>
                  </div>
                </button>
              );
            })}
          </div>

          {selectedExploit && (
            <div className="lg:col-span-5 p-6 bg-canvas/50 space-y-4">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-indigo-700">{selectedExploit.category}</span>
                <span className="font-mono text-emerald-700 font-semibold">
                  Observed: {selectedExploit.observed_score.toFixed(2)}
                </span>
              </div>

              <h4 className="text-base font-semibold text-ink-primary">{selectedExploit.name}</h4>

              <div className="p-3.5 rounded-lg bg-rose-50/70 border border-rose-200 text-xs space-y-1">
                <div className="font-semibold text-rose-800">Exploit Attempt</div>
                <p className="text-rose-950/80 leading-relaxed">
                  {humanizeViolationString(selectedExploit.attack_vector)}
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-emerald-50/70 border border-emerald-200 text-xs space-y-1">
                <div className="font-semibold text-emerald-800">Server Defense</div>
                <p className="text-emerald-950/80 leading-relaxed">
                  {humanizeViolationString(selectedExploit.defense_mechanism)}
                </p>
              </div>

              <div className="p-3.5 rounded-lg bg-surface border border-hairline">
                <div className="flex items-center gap-1.5 text-xs font-medium text-ink-secondary mb-1.5">
                  <Terminal className="w-3.5 h-3.5" /> Test Trace
                </div>
                <div className="space-y-1 font-mono text-[11px] text-ink-secondary">
                  {selectedExploit.sample_tool_trace.map((line, i) => (
                    <div key={i} className="break-all">
                      $ {humanizeViolationString(line)}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
