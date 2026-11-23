import React, { useRef, useState } from "react";
import { ArrowRight, Check, Copy, FileUp, Sparkles, Terminal } from "lucide-react";
import type { ScorecardData } from "../types/cryoforge";
import {
  formatSeedLabel,
  humanizeViolationString,
  SCENARIO_PLAIN_META,
} from "../lib/plainEnglishLabels";

export interface TiebreakerPanelProps {
  scorecard: ScorecardData;
  onInspectHeldOutSeed: (seed: string) => void;
  onUploadJsonFile: (file: File) => void;
}

const REPRO_COMMANDS = [
  {
    label: "1. Verify All Recorded Results & Launch Dashboard (~25s, Offline)",
    cmd: "python scripts/run_eval.py --verify-recorded --serve",
  },
  {
    label: "2. Run Unit & Anti-Cheat Tests",
    cmd: "pytest -q",
  },
  {
    label: "3. Full Live LLM Evaluation (Requires API Key)",
    cmd: "python scripts/run_eval.py --all --env-file .env --frontier-provider gemini --frontier-model gemini-3.8-flash --runs 3",
  },
];

export const TiebreakerPanel: React.FC<TiebreakerPanelProps> = ({
  scorecard,
  onInspectHeldOutSeed,
  onUploadJsonFile,
}) => {
  const tb = scorecard.tiebreaker;
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const copyCommand = (cmd: string, idx: number) => {
    navigator.clipboard?.writeText(cmd).catch(() => {});
    setCopiedIdx(idx);
    window.setTimeout(() => setCopiedIdx(null), 1800);
  };

  return (
    <div className="space-y-6">
      {/* Small Model Summary Card */}
      <div className="bg-surface border border-hairline rounded-xl p-6 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div className="space-y-2 max-w-2xl">
          <div className="inline-flex items-center gap-1.5 text-xs font-medium text-indigo-700 bg-indigo-50 border border-indigo-200 px-2.5 py-0.5 rounded-full">
            <Sparkles className="w-3.5 h-3.5" /> Small Model Tiebreaker · Zero Holdout Leakage
          </div>
          <h2 className="text-lg font-semibold text-ink-primary">
            Held-Out Scenarios 6–8: Small Model Before vs After Training
          </h2>
          <p className="text-sm text-ink-secondary leading-relaxed">
            A lightweight model (<strong>{tb.model_family}</strong>) learns strictly from Scenarios 1–5 trajectories and is evaluated zero-shot on completely unseen Scenarios 6–8—improving from{" "}
            <strong className="text-rose-600">{tb.before_mean.toFixed(2)}</strong> to{" "}
            <strong className="text-emerald-700">{tb.after_mean.toFixed(2)}</strong>.
          </p>
        </div>

        <div className="grid grid-cols-3 gap-3 shrink-0 text-center">
          <div className="p-4 rounded-xl bg-canvas border border-hairline min-w-[110px]">
            <div className="text-xs text-ink-muted">Before</div>
            <div className="text-2xl font-mono font-semibold text-rose-600 mt-1 tabular-nums">
              {tb.before_mean.toFixed(2)}
            </div>
          </div>
          <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 min-w-[110px]">
            <div className="text-xs text-emerald-700">After</div>
            <div className="text-2xl font-mono font-semibold text-emerald-700 mt-1 tabular-nums">
              {tb.after_mean.toFixed(2)}
            </div>
          </div>
          <div className="p-4 rounded-xl bg-indigo-50 border border-indigo-200 min-w-[110px]">
            <div className="text-xs text-indigo-700">Gain</div>
            <div className="text-2xl font-mono font-semibold text-indigo-700 mt-1 tabular-nums">
              +{Math.round(tb.relative_gain_pct)}%
            </div>
          </div>
        </div>
      </div>

      {/* 3 Held-Out Scenarios */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {tb.per_seed.map((item) => {
          const meta = SCENARIO_PLAIN_META[item.seed];
          return (
            <div
              key={item.seed}
              className="bg-surface border border-hairline rounded-xl p-5 flex flex-col justify-between space-y-4"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-indigo-700">{formatSeedLabel(item.seed)}</span>
                  <span className="font-mono font-semibold text-emerald-700">
                    +{item.delta.toFixed(2)}
                  </span>
                </div>

                <h3 className="text-sm font-semibold text-ink-primary leading-snug">
                  {meta?.title ?? item.scenario_title}
                </h3>

                <div className="space-y-2 pt-1 text-xs">
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-ink-muted">Before ({item.before_valid_orders}/4)</span>
                      <span className="font-mono font-semibold text-rose-600">
                        {item.before_score.toFixed(2)}
                      </span>
                    </div>
                    <div className="h-2 bg-canvas rounded-full overflow-hidden border border-hairline">
                      <div
                        className="h-full bg-rose-500 rounded-full"
                        style={{ width: `${Math.max(4, item.before_score * 100)}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-emerald-700 font-medium">
                        After ({item.after_valid_orders}/4)
                      </span>
                      <span className="font-mono font-semibold text-emerald-700">
                        {item.after_score.toFixed(2)}
                      </span>
                    </div>
                    <div className="h-2 bg-canvas rounded-full overflow-hidden border border-hairline">
                      <div
                        className="h-full bg-emerald-500 rounded-full"
                        style={{ width: `${Math.max(4, item.after_score * 100)}%` }}
                      />
                    </div>
                  </div>
                </div>

                <p className="text-xs text-ink-secondary leading-relaxed">
                  {humanizeViolationString(item.key_learned_adaptation)}
                </p>
              </div>

              <button
                type="button"
                onClick={() => onInspectHeldOutSeed(item.seed)}
                className="w-full py-2 px-3 rounded-lg border border-hairline hover:bg-canvas text-xs font-medium text-ink-primary flex items-center justify-center gap-1.5 transition"
              >
                Compare {formatSeedLabel(item.seed)} <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          );
        })}
      </div>

      {/* Judge Quick-Start Guide + Optional Custom JSON Loader */}
      <div className="bg-surface border border-hairline rounded-xl p-6 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-ink-primary" />
            <h3 className="text-base font-semibold text-ink-primary">
              One-Command Judge Verification
            </h3>
          </div>

          <div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onUploadJsonFile(f);
                e.target.value = "";
              }}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-3 py-1.5 rounded-lg border border-hairline hover:bg-canvas text-xs font-medium text-ink-secondary flex items-center gap-1.5"
            >
              <FileUp className="w-3.5 h-3.5" /> Load custom run JSON
            </button>
          </div>
        </div>

        <div className="space-y-2.5">
          {REPRO_COMMANDS.map((item, idx) => (
            <div
              key={item.cmd}
              className="p-3.5 rounded-lg bg-canvas border border-hairline flex flex-wrap items-center justify-between gap-3"
            >
              <div className="min-w-0">
                <div className="text-xs font-medium text-ink-secondary mb-1">{item.label}</div>
                <code className="text-xs font-mono text-ink-primary break-all">{item.cmd}</code>
              </div>
              <button
                type="button"
                onClick={() => copyCommand(item.cmd, idx)}
                className="px-3 py-1.5 rounded-lg bg-surface hover:bg-subtle border border-hairline text-xs font-medium text-ink-primary flex items-center gap-1.5 shrink-0"
              >
                {copiedIdx === idx ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-600" /> Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5 text-ink-muted" /> Copy
                  </>
                )}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
