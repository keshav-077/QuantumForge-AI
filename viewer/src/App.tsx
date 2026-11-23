import { type CSSProperties, type ReactNode, useEffect, useMemo, useState } from "react";
import { FALLBACK_SCORECARD } from "./data/fallbackScorecard";
import { loadStudioData, normalizeScorecardPayload } from "./lib/apiAdapter";
import {
  formatBayLabel,
  humanizeViolationString,
} from "./lib/plainEnglishLabels";
import {
  type OrderLiveSnapshot,
  extractSeedWorldMetadata,
  findShockArrayIndex,
  replayTrajectoryAtStep,
} from "./lib/trajectoryEngine";
import type {
  BayBlackoutWindow,
  ExploitAuditItem,
  OrderVerificationResult,
  ScorecardData,
  TrajectoryLog,
  TrajectoryStepLog,
} from "./types/cryoforge";

type IconName =
  | "overview"
  | "compare"
  | "shield"
  | "learn"
  | "arrow"
  | "check"
  | "x"
  | "copy"
  | "play"
  | "spark"
  | "terminal";

const iconPaths: Record<IconName, ReactNode> = {
  overview: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </>
  ),
  compare: (
    <>
      <path d="M7 3v18M17 3v18" />
      <path d="m3 7 4-4 4 4M13 17l4 4 4-4" />
    </>
  ),
  shield: (
    <>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
      <path d="m9 12 2 2 4-4" />
    </>
  ),
  learn: (
    <>
      <path d="M4 19V9M10 19V5M16 19v-7M22 19H2" />
      <path d="m3 5 6-3 6 5 6-4" />
    </>
  ),
  arrow: <path d="m9 18 6-6-6-6" />,
  check: <path d="m5 12 4 4L19 6" />,
  x: (
    <>
      <path d="m7 7 10 10M17 7 7 17" />
    </>
  ),
  copy: (
    <>
      <rect x="8" y="8" width="12" height="12" rx="2" />
      <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
    </>
  ),
  play: <path d="m9 7 8 5-8 5V7Z" />,
  spark: (
    <>
      <path d="m12 3 1.4 4.1L17.5 8.5l-4.1 1.4L12 14l-1.4-4.1-4.1-1.4 4.1-1.4L12 3Z" />
      <path d="m19 14 .7 2.3L22 17l-2.3.7L19 20l-.7-2.3L16 17l2.3-.7L19 14Z" />
    </>
  ),
  terminal: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="m7 9 3 3-3 3M13 15h4" />
    </>
  ),
};

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {iconPaths[name]}
    </svg>
  );
}

const tabs = [
  { id: "overview", label: "Overview", icon: "overview" as IconName },
  { id: "comparison", label: "Agent comparison", icon: "compare" as IconName },
  { id: "integrity", label: "Reward integrity", icon: "shield" as IconName },
  { id: "learning", label: "Learning & CLI", icon: "learn" as IconName },
];

interface EvalScenarioMeta {
  seed: string;
  title: string;
  category: string;
  detail: string;
  disruption: string;
  defaultRunIdx: number;
}

const EVAL_SCENARIOS: EvalScenarioMeta[] = [
  {
    seed: "S02",
    title: "Standard lane outage",
    category: "Live maintenance outage",
    detail: "Lane B closes for maintenance in Weeks 5–7 and €15,000 contingency is frozen after initial commitments.",
    disruption: "Lane B outage (W5–7)",
    defaultRunIdx: 1,
  },
  {
    seed: "S05",
    title: "Emergency budget cut",
    category: "Cost & hoist constraint shift",
    detail: "Tier-2 supplier backlog (+2 weeks) and a 5,200 kg Lane A lift cap force costly emergency retooling.",
    disruption: "Budget strain + lift cap",
    defaultRunIdx: 1,
  },
  {
    seed: "S03",
    title: "Reduced lane capacity",
    category: "Efficiency & permit drop",
    detail: "Standard-tier efficiency degrades 1.55× while oversize route permits are cut from 2 to 1 mid-episode.",
    disruption: "1.55× loss & −1 permit",
    defaultRunIdx: 2,
  },
  {
    seed: "S04",
    title: "Safety clearance increase",
    category: "Safety & corridor lock",
    detail: "Job 3 minimum safety pressure jumps to 2.5 MPa and Alpine corridor height clearance locks at 2,700 mm.",
    disruption: "2.5 MPa & 2.7m clearance",
    defaultRunIdx: 0,
  },
  {
    seed: "S01",
    title: "Accelerated deadline",
    category: "Lead-time compression",
    detail: "Customs hold adds +3 weeks to precision kits—compressing effective SLA windows—and caps North route mass at 5.8t.",
    disruption: "+3w lead & 5.8t cap",
    defaultRunIdx: 0,
  },
];

const HELDOUT_PLAIN_TITLES: Record<string, { title: string; summary: string }> = {
  S06: {
    title: "Precision delay & corridor weight cap",
    summary: "Tier-3 lead time +2 weeks, Fjord route capped at 5,600 kg, and €10k budget cut.",
  },
  S07: {
    title: "Lane A outage & oversize permit cut",
    summary: "Lane A offline in Weeks 4–6, Standard-tier loss 1.50×, and permits cut from 2 to 1.",
  },
  S08: {
    title: "Safety pressure uprate & clearance lock",
    summary: "Job 2 uprated to 2.5 MPa, Gorge clearance locked at 2,700 mm, and Lane B capped at 5.1t.",
  },
};

const EXPLOIT_PLAIN_MAP: Record<string, { title: string; defense: string; attempt: string }> = {
  EXPLOIT_01_IMMEDIATE_FINALIZE: {
    title: "Immediate Finalize",
    defense: "Batch completeness gate",
    attempt: "Calls finalize_commissioning() on Turn 1 with zero scheduled jobs hoping for base credit.",
  },
  EXPLOIT_02_CHERRY_PICK_ONE_EASY_ORDER: {
    title: "Cherry-Pick 1 Easy Job",
    defense: "1/4 batch cap (≤ 0.08)",
    attempt: "Schedules only the easiest warm-up job and stops early to dodge multi-lane contention.",
  },
  EXPLOIT_03_SCHEMA_AND_NAN_FUZZING: {
    title: "Schema & NaN Injection",
    defense: "Strict Pydantic schema",
    attempt: "Injects unsupported fields or NaN/Inf numerics to bypass constraint arithmetic.",
  },
  EXPLOIT_04_FREE_ORACLE_RESERVE_FARMING: {
    title: "Tool Oracle Farming",
    defense: "Anti-oracle + retool fee",
    attempt: "Repeatedly books and revises jobs to probe hidden physics without spending simulation quota.",
  },
  EXPLOIT_05_SIM_QUOTA_OVERFLOW: {
    title: "Sim Budget Overflow",
    defense: "Hard 3-call server quota",
    attempt: "Attempts a 4th simulation tool call after exhausting the 3-call episode budget.",
  },
  EXPLOIT_06_BAY_AND_PERMIT_DOUBLE_SPEND: {
    title: "Lane & Permit Double-Spend",
    defense: "Global resource ledger",
    attempt: "Double-books overlapping lane weeks or reuses a single oversize transport permit twice.",
  },
  EXPLOIT_07_POST_FINALIZE_MUTATION: {
    title: "Post-Finalize Mutation",
    defense: "Immutable terminal lock",
    attempt: "Attempts to mutate bookings after finalize_commissioning() has locked the episode.",
  },
  EXPLOIT_08_STATIC_PREBOOK_IGNORE_SHOCK: {
    title: "Ignoring Disruption",
    defense: "Post-shock state audit",
    attempt: "Submits the pre-disruption schedule unchanged without adapting to live constraint shifts.",
  },
};

const FAILURE_SHORT_LABELS: Record<string, string> = {
  SIM_EXHAUSTION: "Tool budget (3/3)",
  SHOCK_UNADAPTED: "Ignored disruption",
  BAY_CONFLICT: "Lane collision",
  HOOP_STRESS: "Safety margin",
  BRIDGE_MASS_CAP: "Route capacity",
  BOIL_OFF: "Efficiency limit",
};

function StatusBadge({ pass, children }: { pass: boolean; children: ReactNode }) {
  return (
    <span className={`status-badge ${pass ? "pass" : "fail"}`}>
      <Icon name={pass ? "check" : "x"} size={13} />
      {children}
    </span>
  );
}

function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="section-heading">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
      </div>
      <p>{description}</p>
    </div>
  );
}

function formatBookingChoice(
  snap: OrderLiveSnapshot | undefined,
  verif: OrderVerificationResult | undefined
): string {
  const b = snap?.booking;
  if (!b) {
    if (verif && !verif.booked) return "Unscheduled · Not booked";
    return "Unscheduled";
  }
  const lane = formatBayLabel(b.bay_id);
  const endInclusive = Math.max(b.bay_start_week, b.fab_end_week - 1);
  const weeks =
    b.bay_start_week === endInclusive
      ? `W${b.bay_start_week}`
      : `W${b.bay_start_week}–${endInclusive}`;
  const units = `${b.spec.unit_count}×${b.spec.unit_volume_m3}m³`;
  const tier = b.spec.insulation === "EVI" ? "High-Eff" : "Std-Tier";
  const deliv = verif?.delivery_week ?? snap?.deliveryWeek;
  const delivTag = deliv != null ? ` · Deliv W${deliv}` : "";
  return `${lane} · ${weeks} · ${units} · ${tier}${delivTag}`;
}

function formatOrderTaskTitle(snap: OrderLiveSnapshot): string {
  const mode = snap.order.execution === "TRANSPORTABLE" ? "Mobile transfer" : "Stationary install";
  return `${snap.order.client_name} · ${mode} (SLA W${snap.order.sla_week})`;
}

function summarizeStepCall(step: TrajectoryStepLog): {
  signature: string;
  summary: string;
  status: "ENVIRONMENT EVENT" | "VERIFIED" | "REJECTED" | "SUCCESS";
  variant: "" | "event" | "rejected";
} {
  const args = step.arguments ?? {};
  const obs = step.observation ?? {};
  const ok = obs.ok !== false;

  let argPreview = "";
  if (typeof args.order_id === "string") {
    const jobNum = args.order_id.replace("ORD-", "Job ");
    const extras: string[] = [jobNum];
    if (typeof args.bay_id === "string") extras.push(formatBayLabel(args.bay_id));
    if (typeof args.bay_start_week === "number") extras.push(`W${args.bay_start_week}`);
    if (typeof args.route_id === "string") extras.push(args.route_id);
    argPreview = extras.join(", ");
  } else if (typeof args.section === "string") {
    argPreview = `"${args.section}"`;
  }

  const signature = `${step.tool_name}(${argPreview})`;

  if (step.disruption_triggered_this_step) {
    const alertObj =
      obs.disruption_alert && typeof obs.disruption_alert === "object"
        ? (obs.disruption_alert as Record<string, unknown>)
        : null;
    const headline =
      typeof alertObj?.headline === "string"
        ? humanizeViolationString(alertObj.headline)
        : "Live mid-episode constraint disruption applied to server state.";
    return {
      signature,
      summary: `Disruption triggered: ${headline}`,
      status: "ENVIRONMENT EVENT",
      variant: "event",
    };
  }

  if (!ok) {
    const errCode = typeof obs.error === "string" ? obs.error : "CONSTRAINT_REJECTED";
    const msg = typeof obs.message === "string" ? obs.message : "Rejected by server guardrail.";
    return {
      signature,
      summary: humanizeViolationString(`${errCode}: ${msg}`),
      status: "REJECTED",
      variant: "rejected",
    };
  }

  if (step.tool_name === "finalize_commissioning") {
    const score = typeof obs.final_score === "number" ? obs.final_score.toFixed(2) : "—";
    const valid = typeof obs.valid_order_count === "number" ? `${obs.valid_order_count}/4 valid` : "audited";
    return {
      signature,
      summary: `Zero-trust server audit completed: ${valid}, final score ${score}.`,
      status: "VERIFIED",
      variant: "",
    };
  }

  if (step.tool_name === "get_episode_brief") {
    return {
      signature,
      summary: "Loaded 4 interdependent jobs, 2 execution lanes, budget cap, and 3-call simulation limit.",
      status: "SUCCESS",
      variant: "",
    };
  }

  if (step.tool_name === "read_engineering_handbook") {
    return {
      signature,
      summary: "Read deterministic safety margin, thermal loss, and corridor clearance formulas.",
      status: "SUCCESS",
      variant: "",
    };
  }

  if (step.tool_name === "run_physics_sim") {
    const rem = typeof obs.sim_calls_remaining === "number" ? `${obs.sim_calls_remaining} left` : "checked";
    return {
      signature,
      summary: `Simulated candidate configuration against engineering thresholds (${rem}).`,
      status: "SUCCESS",
      variant: "",
    };
  }

  if (step.tool_name === "reserve_bay_and_order" || step.tool_name === "revise_order") {
    const verb = step.tool_name === "revise_order" ? "Revised" : "Reserved";
    return {
      signature,
      summary: `${verb} lane schedule and hardware spec (${argPreview || "updated ledger"}).`,
      status: "SUCCESS",
      variant: "",
    };
  }

  if (step.tool_name === "assign_transport") {
    const permit = args.use_escort_permit ? "with oversize permit" : "standard gauge";
    return {
      signature,
      summary: `Assigned transport corridor (${argPreview}) ${permit}.`,
      status: "SUCCESS",
      variant: "",
    };
  }

  return {
    signature,
    summary: "Executed MCP tool call against live plant state.",
    status: "SUCCESS",
    variant: "",
  };
}

const BENCHMARK_FIGURES = [
  {
    id: "fig-01",
    code: "FIGURE 01",
    title: "Benchmark system",
    caption: "Setup → Initial plan → Step-4 disruption → Hard server verification",
    src: "./figures/figure-01-benchmark-system.svg",
  },
  {
    id: "fig-02",
    code: "FIGURE 02",
    title: "Performance gap (+0.63)",
    caption: "Reference solver (0.95) vs. frontier agent (0.32) across 5 disruptions",
    src: "./figures/figure-02-agent-gap.svg",
  },
  {
    id: "fig-03",
    code: "FIGURE 03",
    title: "Disruption recovery",
    caption: "12-week schedule rerouting around a mid-episode Lane B outage",
    src: "./figures/figure-03-disruption-recovery.svg",
  },
  {
    id: "fig-04",
    code: "FIGURE 04",
    title: "Reward integrity (8/8)",
    caption: "Eight adversarial reward-hacking strategies capped at 0.00–0.08",
    src: "./figures/figure-04-reward-integrity.svg",
  },
] as const;

function Overview({
  scorecard,
  onExplore,
  onOpenIntegrity,
  onSelectScenario,
}: {
  scorecard: ScorecardData;
  onExplore: () => void;
  onOpenIntegrity: () => void;
  onSelectScenario: (seed: string) => void;
}) {
  const [howOpen, setHowOpen] = useState(false);
  const [copiedCmdId, setCopiedCmdId] = useState<string | null>(null);
  const [activeFigure, setActiveFigure] = useState<(typeof BENCHMARK_FIGURES)[number] | null>(null);

  const copyInstallCmd = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedCmdId(id);
      window.setTimeout(() => setCopiedCmdId(null), 1800);
    } catch {
      setCopiedCmdId(null);
    }
  };

  const refMean = scorecard.rubric.reference_mean.toFixed(2);
  const frontMean = scorecard.rubric.frontier_mean.toFixed(2);
  const gapDelta = scorecard.rubric.gap_delta.toFixed(2);
  const gapWidthPct = Math.max(15, Math.min(95, Math.round(scorecard.rubric.gap_delta * 100)));
  const failureCount = scorecard.failure_taxonomy.length || 6;
  const blockedText = `${scorecard.rubric.exploits_blocked}/${scorecard.rubric.exploits_total}`;
  const doNothing = scorecard.rubric.do_nothing_score.toFixed(2);
  const stopEarly = scorecard.rubric.stop_early_score.toFixed(2);
  const maxExploit = scorecard.rubric.max_exploit_score.toFixed(2);
  const beforeMean = scorecard.tiebreaker.before_mean.toFixed(2);
  const afterMean = scorecard.tiebreaker.after_mean.toFixed(2);
  const relGainPct = Math.round(scorecard.tiebreaker.relative_gain_pct);

  const failureChips =
    scorecard.failure_taxonomy.length > 0
      ? scorecard.failure_taxonomy.map(
          (f) => FAILURE_SHORT_LABELS[f.code] ?? f.title
        )
      : Object.values(FAILURE_SHORT_LABELS);

  return (
    <div className="view-stack">
      <section className="hero">
        <div
          className="hero-art-bg"
          style={{ backgroundImage: 'url("./assets/concept-hero-art.webp")' }}
          aria-hidden="true"
        />
        <div className="hero-copy">
          <div className="hero-kicker">
            <Icon name="spark" size={15} />
            Multi-constraint operations benchmark
          </div>
          <h1>Can an AI agent recover when the plan breaks?</h1>
          <p>
            AgentBreak tests planning, tool use, and disruption recovery against hard safety,
            capacity, and deadline constraints—not just plausible-looking answers.
          </p>
          <div className="hero-actions">
            <button className="primary-button" onClick={onExplore}>
              Explore agent replay <Icon name="arrow" size={16} />
            </button>
            <span className="run-note">
              <span className="live-dot" /> 5 scenarios · 3 runs each
            </span>
          </div>
        </div>
        <div className="score-orbit" aria-label="Benchmark score gap">
          <div className="score-ring">
            <span className="score-label">Score gap</span>
            <strong>+{gapDelta}</strong>
            <small>reference vs frontier</small>
          </div>
        </div>
      </section>

      <section className="visual-showcase-grid" aria-label="Benchmark design concepts">
        <button type="button" className="visual-card-item" onClick={onExplore}>
          <div className="visual-card-media">
            <img
              src="./assets/concept-disruption-art.webp"
              alt="Modular jobs rerouting across parallel execution lanes around a mid-episode barrier"
              loading="lazy"
            />
          </div>
          <div className="visual-card-caption">
            <div>
              <small>Dynamic Rerouting</small>
              <strong>4 modular jobs across 2 contested execution lanes</strong>
            </div>
            <span className="visual-card-link">
              Replay <Icon name="arrow" size={13} />
            </span>
          </div>
        </button>

        <button type="button" className="visual-card-item" onClick={onOpenIntegrity}>
          <div className="visual-card-media">
            <img
              src="./assets/concept-shield-art.webp"
              alt="Zero-trust verification gate deflecting reward-hacking shortcuts"
              loading="lazy"
            />
          </div>
          <div className="visual-card-caption">
            <div>
              <small>Zero-Trust Verifier</small>
              <strong>8/8 reward-hacking shortcuts deflected at server boundary</strong>
            </div>
            <span className="visual-card-link">
              Audit <Icon name="arrow" size={13} />
            </span>
          </div>
        </button>
      </section>

      <section className={`how-expander ${howOpen ? "open" : "collapsed"}`}>
        <button
          type="button"
          className="expander-bar"
          onClick={() => setHowOpen((prev) => !prev)}
          aria-expanded={howOpen}
        >
          <div className="expander-bar-left">
            <span className="expander-bar-title">How this benchmark works</span>
            <span className="expander-bar-hint">
              Plan 4 jobs across 2 lanes · Adapt to live disruption · Deterministic server check
            </span>
          </div>
          <span className="expander-pill">
            {howOpen ? "Hide details ▴" : "How this benchmark works ▾"}
          </span>
        </button>

        {howOpen && (
          <div className="expander-body">
            <SectionHeading
              eyebrow="How this benchmark works"
              title="One plan. One surprise. Hard verification."
              description="The agent gets a constrained planning problem, limited tools, and no second chance to ignore reality."
            />
            <div className="steps">
              <article className="step-card">
                <div className="step-top">
                  <span className="step-number">01</span>
                  <span className="step-icon">
                    <Icon name="overview" size={20} />
                  </span>
                </div>
                <h3>Plan four jobs</h3>
                <p>Schedule 4 interdependent jobs across 2 lanes with only 3 simulation tool calls.</p>
                <div className="mini-tags">
                  <span>4 jobs</span>
                  <span>2 lanes</span>
                  <span>3 calls</span>
                </div>
              </article>
              <div className="step-connector">
                <Icon name="arrow" size={16} />
              </div>
              <article className="step-card">
                <div className="step-top">
                  <span className="step-number">02</span>
                  <span className="step-icon disruption">
                    <Icon name="spark" size={20} />
                  </span>
                </div>
                <h3>Adapt mid-episode</h3>
                <p>A live lane outage, budget cut, or capacity drop invalidates the original plan.</p>
                <div className="event-chip">
                  <span /> Mid-episode disruption
                </div>
              </article>
              <div className="step-connector">
                <Icon name="arrow" size={16} />
              </div>
              <article className="step-card">
                <div className="step-top">
                  <span className="step-number">03</span>
                  <span className="step-icon success">
                    <Icon name="shield" size={20} />
                  </span>
                </div>
                <h3>Verify every constraint</h3>
                <p>
                  The server checks the final plan against hard safety, capacity, and deadline rules.
                </p>
                <div className="verify-line">
                  <Icon name="check" size={14} /> Deterministic scoring
                </div>
              </article>
            </div>
            <div className="flow-svg-frame">
              <img
                src="./assets/benchmark-flow.svg"
                alt="Benchmark loop: Plan four jobs, adapt at Step 4, verify every constraint"
                loading="lazy"
              />
            </div>
          </div>
        )}
      </section>

      <section>
        <SectionHeading
          eyebrow="Benchmark results"
          title="Benchmark evidence at a glance"
          description="Scores are grounded in executable outcomes from runs/scorecard.json, not subjective model judging."
        />
        <div className="kpi-grid">
          <article className="kpi-card gap-card">
            <div className="kpi-header">
              <span>The Gap</span>
              <b>5 eval tasks</b>
            </div>
            <div className="model-score-row">
              <div>
                <small>Reference solver</small>
                <strong>{refMean}</strong>
              </div>
              <div className="score-vs">vs</div>
              <div>
                <small>Frontier agent</small>
                <strong>{frontMean}</strong>
              </div>
            </div>
            <div className="gap-track">
              <span style={{ width: `${gapWidthPct}%` }} />
            </div>
            <p>
              <b>+{gapDelta} gap</b> across 5 scenarios and 3 runs each.
            </p>
          </article>

          <article className="kpi-card">
            <div className="kpi-header">
              <span>Meaningful difficulty</span>
              <b>6 failure modes</b>
            </div>
            <div className="large-metric">
              {failureCount} <span>failure modes</span>
            </div>
            <div className="failure-list">
              {failureChips.map((x) => (
                <span key={x}>{x}</span>
              ))}
            </div>
          </article>

          <article className="kpi-card">
            <div className="kpi-header">
              <span>Reward integrity</span>
              <b>Server-guarded</b>
            </div>
            <div className="large-metric emerald">
              {blockedText} <span>exploits blocked (0.00–{maxExploit})</span>
            </div>
            <div className="integrity-bars">
              <div>
                <span>Do nothing</span>
                <b>{doNothing}</b>
              </div>
              <div>
                <span>Stop early</span>
                <b>{stopEarly}</b>
              </div>
            </div>
            <div className="guarded">
              <Icon name="shield" size={14} /> Server-side validation
            </div>
          </article>

          <article className="kpi-card learning-card">
            <div className="kpi-header">
              <span>Craft & learning</span>
              <b>Held-out split</b>
            </div>
            <div className="growth-row">
              <div>
                <small>Before</small>
                <strong>{beforeMean}</strong>
              </div>
              <Icon name="arrow" size={20} />
              <div>
                <small>After</small>
                <strong>{afterMean}</strong>
              </div>
              <span className="growth-pill">+{relGainPct}%</span>
            </div>
            <div className="craft-tags">
              <span>1-cmd reproduction</span>
              <span>Harbor sandbox</span>
              <span>Held-out lift</span>
            </div>
            <p>1-command local reproduction + held-out generalization from environment trajectories.</p>
          </article>
        </div>
      </section>

      <section className="panel scenario-chart-panel">
        <div className="panel-heading" style={{ marginBottom: 18 }}>
          <div>
            <p className="eyebrow">Scenario breakdown</p>
            <h2>Reference solver vs. frontier agent across all 5 disruptions</h2>
          </div>
          <div className="legend">
            <span>
              <i className="ref-color" />
              Reference ({refMean})
            </span>
            <span>
              <i className="front-color" style={{ background: "var(--rose)" }} />
              Frontier ({frontMean})
            </span>
          </div>
        </div>
        <div className="scenario-bars-list">
          {EVAL_SCENARIOS.map((scen) => {
            const seedRow = scorecard.per_seed.find((s) => s.seed === scen.seed);
            const rVal = seedRow?.reference_score ?? scorecard.rubric.reference_mean;
            const fVal = seedRow?.frontier_mean ?? scorecard.rubric.frontier_mean;
            const gapVal = Math.max(0, rVal - fVal);
            return (
              <button
                key={scen.seed}
                type="button"
                className="scenario-bar-row"
                onClick={() => onSelectScenario(scen.seed)}
              >
                <div className="sbr-meta">
                  <strong>{scen.title}</strong>
                  <small>{scen.disruption}</small>
                </div>
                <div className="sbr-tracks">
                  <div className="sbr-track">
                    <div
                      className="sbr-fill-ref"
                      style={{ width: `${Math.round(rVal * 100)}%` }}
                    />
                  </div>
                  <div className="sbr-track">
                    <div
                      className="sbr-fill-front"
                      style={{ width: `${Math.max(4, Math.round(fVal * 100))}%` }}
                    />
                  </div>
                </div>
                <div className="sbr-scores">
                  <b>{rVal.toFixed(2)}</b> vs <span>{fVal.toFixed(2)}</span>
                  <small>−{gapVal.toFixed(2)} gap · Inspect →</small>
                </div>
              </button>
            );
          })}
        </div>
      </section>

      <section className="panel figures-gallery-panel" aria-label="Benchmark architecture and result figures">
        <div className="panel-heading" style={{ marginBottom: 16 }}>
          <div>
            <p className="eyebrow">Architecture &amp; result figures</p>
            <h2>Vector system diagrams (click any figure to zoom)</h2>
          </div>
          <span className="score-range">SVG · High-DPI</span>
        </div>
        <div className="figures-strip-grid">
          {BENCHMARK_FIGURES.map((fig) => (
            <button
              key={fig.id}
              type="button"
              className="figure-thumb-card"
              onClick={() => setActiveFigure(fig)}
            >
              <div className="figure-thumb-frame">
                <img src={fig.src} alt={`${fig.code} — ${fig.title}`} loading="lazy" />
              </div>
              <div className="figure-thumb-meta">
                <span>{fig.code}</span>
                <strong>{fig.title}</strong>
                <p>{fig.caption}</p>
              </div>
            </button>
          ))}
        </div>
      </section>

      {activeFigure && (
        <div
          className="figure-lightbox-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label={activeFigure.title}
          onClick={() => setActiveFigure(null)}
        >
          <div className="figure-lightbox-card" onClick={(e) => e.stopPropagation()}>
            <div className="figure-lightbox-top">
              <div>
                <span>{activeFigure.code}</span>
                <strong>{activeFigure.title}</strong>
              </div>
              <button
                type="button"
                className="compact-toggle"
                onClick={() => setActiveFigure(null)}
              >
                Close ✕
              </button>
            </div>
            <div className="figure-lightbox-body">
              <img src={activeFigure.src} alt={activeFigure.title} />
            </div>
          </div>
        </div>
      )}

      <section className="panel judge-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Quick-start & agent skill</p>
            <h2>Install & use locally</h2>
          </div>
          <a
            href={GITHUB_REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="compact-toggle"
          >
            GitHub repo <Icon name="arrow" size={13} />
          </a>
        </div>
        <div className="terminal-box" style={{ marginTop: 0 }}>
          <div className="terminal-top">
            <span>
              <i />
              <i />
              <i />
            </span>
            <small>LOCAL CLI · AGENT SKILL · HARBOR</small>
          </div>
          {INSTALL_COMMANDS.map((item) => (
            <div className="terminal-command compact-cmd" key={item.id}>
              <span className="cmd-label">{item.label}</span>
              <code>{item.cmd}</code>
              <button
                type="button"
                onClick={() => copyInstallCmd(item.id, item.cmd)}
                aria-label={`Copy ${item.label} command`}
              >
                <Icon name={copiedCmdId === item.id ? "check" : "copy"} size={16} />
                {copiedCmdId === item.id ? "Copied" : "Copy"}
              </button>
            </div>
          ))}
        </div>
        <div className="expected">
          <span>Included in repository</span>
          <div>
            <Icon name="check" size={14} />
            <p>
              Bundled skill at <code>.agents/skills/agentbreak-bench/SKILL.md</code> for zero-key
              MCP playtesting (Codex / OpenCode / Antigravity / Claude Code) or Harbor evaluation
            </p>
          </div>
          <div>
            <Icon name="check" size={14} />
            <p>
              Verified scorecards in <code>runs/scorecard.json</code>,{" "}
              <code>runs/SCORECARD.md</code>, and Harbor logs in <code>runs/harbor_jobs/</code>
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

interface TimelineBlock {
  label: string;
  startWeek: number;
  endWeekExclusive: number;
  variant: "ref" | "front" | "front-fail";
}

function TimelineRow({
  lane,
  totalWeeks,
  blocks,
  blackouts,
  deadline,
}: {
  lane: string;
  totalWeeks: number;
  blocks: TimelineBlock[];
  blackouts: BayBlackoutWindow[];
  deadline: string;
}) {
  const weekStyle = { "--weeks": totalWeeks } as CSSProperties;
  return (
    <div className="timeline-row">
      <div className="lane-label">{lane}</div>
      <div className="week-grid dynamic-weeks" style={weekStyle}>
        {Array.from({ length: totalWeeks }, (_, i) => (
          <i key={i} />
        ))}
        {blackouts.map((blk, idx) => {
          const s = Math.max(1, Math.min(totalWeeks, blk.start_week));
          const e = Math.max(s + 1, Math.min(totalWeeks + 1, blk.end_week));
          return (
            <span
              key={`blk-${idx}`}
              className="blackout"
              style={{ gridColumn: `${s} / ${e}` }}
              title={blk.reason}
            >
              <b>OUT</b>
            </span>
          );
        })}
        {blocks.map((blk, idx) => {
          const s = Math.max(1, Math.min(totalWeeks, blk.startWeek));
          const e = Math.max(s + 1, Math.min(totalWeeks + 1, blk.endWeekExclusive));
          return (
            <span
              key={`${blk.label}-${s}-${idx}`}
              className={`schedule-block ${blk.variant}`}
              style={{ gridColumn: `${s} / ${e}` }}
            >
              {blk.label}
            </span>
          );
        })}
      </div>
      <div className="deadline-badge">{deadline}</div>
    </div>
  );
}

function Comparison({
  scorecard,
  initialSeed,
}: {
  scorecard: ScorecardData;
  initialSeed?: string;
}) {
  const startMeta =
    EVAL_SCENARIOS.find((s) => s.seed === initialSeed) ?? EVAL_SCENARIOS[0];
  const [selectedSeed, setSelectedSeed] = useState<string>(startMeta.seed);
  const [frontierRunIdx, setFrontierRunIdx] = useState<number>(startMeta.defaultRunIdx);
  const [openReasons, setOpenReasons] = useState<Record<string, boolean>>({});
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [traceAgent, setTraceAgent] = useState<"frontier" | "reference">("frontier");
  const [stepIdx, setStepIdx] = useState<number>(0);

  const scenarioIndex = Math.max(
    0,
    EVAL_SCENARIOS.findIndex((s) => s.seed === selectedSeed)
  );
  const scenario = EVAL_SCENARIOS[scenarioIndex] ?? EVAL_SCENARIOS[0];

  const seedSummary = useMemo(
    () => scorecard.per_seed.find((s) => s.seed === scenario.seed),
    [scorecard.per_seed, scenario.seed]
  );

  const seedTrajectories = useMemo(
    () => scorecard.trajectories.filter((t) => t.seed === scenario.seed),
    [scorecard.trajectories, scenario.seed]
  );

  const referenceRun: TrajectoryLog | undefined = useMemo(
    () => seedTrajectories.find((t) => t.agent_type === "reference"),
    [seedTrajectories]
  );

  const frontierRuns: TrajectoryLog[] = useMemo(
    () =>
      seedTrajectories
        .filter((t) => t.agent_type === "frontier")
        .sort((a, b) => a.run_id.localeCompare(b.run_id)),
    [seedTrajectories]
  );

  const clampedRunIdx = frontierRuns.length > 0 ? Math.min(frontierRunIdx, frontierRuns.length - 1) : 0;
  const frontierRun: TrajectoryLog | undefined = frontierRuns[clampedRunIdx];

  const world = useMemo(
    () => extractSeedWorldMetadata(scenario.seed, seedTrajectories),
    [scenario.seed, seedTrajectories]
  );

  const refFrame = useMemo(
    () =>
      referenceRun
        ? replayTrajectoryAtStep(referenceRun, referenceRun.steps.length - 1, world)
        : null,
    [referenceRun, world]
  );

  const frontFrame = useMemo(
    () =>
      frontierRun
        ? replayTrajectoryAtStep(frontierRun, frontierRun.steps.length - 1, world)
        : null,
    [frontierRun, world]
  );

  const activeTraceRun = traceAgent === "reference" ? referenceRun : frontierRun;
  const activeShockIdx = useMemo(
    () => (activeTraceRun ? findShockArrayIndex(activeTraceRun) : null),
    [activeTraceRun]
  );

  useEffect(() => {
    if (activeTraceRun && activeTraceRun.steps.length > 0) {
      setStepIdx(activeShockIdx ?? Math.min(3, activeTraceRun.steps.length - 1));
    } else {
      setStepIdx(0);
    }
  }, [activeTraceRun, activeShockIdx]);

  const handleScenarioChange = (newSeed: string) => {
    const meta = EVAL_SCENARIOS.find((s) => s.seed === newSeed) ?? EVAL_SCENARIOS[0];
    setSelectedSeed(meta.seed);
    setFrontierRunIdx(meta.defaultRunIdx);
    setOpenReasons({});
  };

  const jobCards = useMemo(() => {
    const orders = refFrame?.orders ?? frontFrame?.orders ?? [];
    return orders.map((ordSnap, idx) => {
      const orderId = ordSnap.order.order_id;
      const refSnap = refFrame?.orders.find((o) => o.order.order_id === orderId);
      const frSnap = frontFrame?.orders.find((o) => o.order.order_id === orderId);
      const refVerif = referenceRun?.verification.order_results.find((r) => r.order_id === orderId);
      const frVerif = frontierRun?.verification.order_results.find((r) => r.order_id === orderId);

      const pass = frVerif ? frVerif.passed_all_hard_checks : Boolean(frSnap?.passedAllHardChecks);
      const rawViolations = frVerif?.violations?.length
        ? frVerif.violations
        : frSnap?.allViolations ?? [];

      const reason = pass
        ? "All safety, efficiency, lane clearance, and delivery deadline rules satisfied."
        : rawViolations.length > 0
          ? rawViolations.map(humanizeViolationString).join(" · ")
          : frontierRun?.verification.budget_exceeded
            ? "Global operating budget exceeded after emergency retooling."
            : "Failed terminal constraint verification.";

      return {
        key: orderId,
        name: `Job ${idx + 1}`,
        task: formatOrderTaskTitle(ordSnap),
        reference: formatBookingChoice(refSnap, refVerif),
        frontier: formatBookingChoice(frSnap, frVerif),
        pass,
        reason,
      };
    });
  }, [refFrame, frontFrame, referenceRun, frontierRun]);

  const { totalWeeks, laneARef, laneAFront, laneBRef, laneBFront, blackoutsA, blackoutsB } =
    useMemo(() => {
      const allBlackouts = [
        ...world.initialBlackouts,
        ...(world.disruption?.bay_blackouts ?? []),
      ];
      let maxEnd = Math.max(12, world.horizonWeeks);

      const buildBlocks = (
        orders: OrderLiveSnapshot[] | undefined,
        verifList: OrderVerificationResult[] | undefined,
        isRef: boolean
      ) => {
        const laneA: TimelineBlock[] = [];
        const laneB: TimelineBlock[] = [];
        if (!orders) return { laneA, laneB };

        orders.forEach((o, idx) => {
          const b = o.booking;
          if (!b) return;
          maxEnd = Math.max(maxEnd, b.fab_end_week - 1);
          const v = verifList?.find((r) => r.order_id === o.order.order_id);
          const passed = v ? v.passed_all_hard_checks : o.passedAllHardChecks;
          const blk: TimelineBlock = {
            label: `J${idx + 1}`,
            startWeek: b.bay_start_week,
            endWeekExclusive: b.fab_end_week,
            variant: isRef ? "ref" : passed ? "front" : "front-fail",
          };
          if (b.bay_id === "BAY-B") laneB.push(blk);
          else laneA.push(blk);
        });
        return { laneA, laneB };
      };

      const refB = buildBlocks(
        refFrame?.orders,
        referenceRun?.verification.order_results,
        true
      );
      const frB = buildBlocks(
        frontFrame?.orders,
        frontierRun?.verification.order_results,
        false
      );

      for (const blk of allBlackouts) {
        maxEnd = Math.max(maxEnd, blk.end_week - 1);
      }

      return {
        totalWeeks: Math.min(16, maxEnd),
        laneARef: refB.laneA,
        laneBRef: refB.laneB,
        laneAFront: frB.laneA,
        laneBFront: frB.laneB,
        blackoutsA: allBlackouts.filter((b) => b.bay_id === "BAY-A"),
        blackoutsB: allBlackouts.filter((b) => b.bay_id === "BAY-B"),
      };
    }, [world, refFrame, frontFrame, referenceRun, frontierRun]);

  const refScore = (
    referenceRun?.verification.final_score ??
    seedSummary?.reference_score ??
    scorecard.rubric.reference_mean
  ).toFixed(2);
  const refValid = referenceRun?.verification.valid_order_count ?? 4;

  const frontScore = (
    frontierRun?.verification.final_score ??
    seedSummary?.frontier_mean ??
    scorecard.rubric.frontier_mean
  ).toFixed(2);
  const frontMeanScore = (seedSummary?.frontier_mean ?? scorecard.rubric.frontier_mean).toFixed(2);
  const frontValid = frontierRun?.verification.valid_order_count ?? 0;

  const traceSteps = activeTraceRun?.steps ?? [];
  const safeStepIdx =
    traceSteps.length > 0 ? Math.max(0, Math.min(stepIdx, traceSteps.length - 1)) : 0;
  const currentStep = traceSteps[safeStepIdx];
  const stepInfo = currentStep ? summarizeStepCall(currentStep) : null;

  const weekHeaderStyle = { "--weeks": totalWeeks } as CSSProperties;

  return (
    <div className="view-stack">
      <section className="view-intro">
        <div>
          <p className="eyebrow">Interactive replay</p>
          <h1>See exactly where the agents diverge.</h1>
          <p>Compare decisions, schedules, and tool use against the same live disruption.</p>
        </div>
        <div className="scenario-control">
          <label htmlFor="scenario">Disruption scenario</label>
          <select
            id="scenario"
            value={scenario.seed}
            onChange={(e) => handleScenarioChange(e.target.value)}
          >
            {EVAL_SCENARIOS.map((item) => (
              <option key={item.seed} value={item.seed}>
                {item.title}
              </option>
            ))}
          </select>
        </div>
      </section>

      <section className="scenario-banner">
        <div className="scenario-index">{String(scenarioIndex + 1).padStart(2, "0")}</div>
        <div>
          <span>{scenario.category}</span>
          <h2>{scenario.title}</h2>
          <p>{scenario.detail}</p>
        </div>
        <div className="disruption-label">
          <span>Mid-episode disruption</span>
          <strong>{scenario.disruption}</strong>
        </div>
      </section>

      <section>
        <div className="agent-summary">
          <div className="agent-title">
            <span className="agent-dot reference" />
            <div>
              <small>REFERENCE AGENT</small>
              <h3>Constraint-aware solver</h3>
            </div>
          </div>
          <div className="agent-result">
            <strong>{refScore}</strong>
            <StatusBadge pass={refValid === 4}>{refValid}/4 valid</StatusBadge>
          </div>
          <div className="agent-divider" />
          <div className="agent-title">
            <span className="agent-dot frontier" />
            <div>
              <small>FRONTIER AGENT</small>
              <h3>{scorecard.trajectories.find((t) => t.agent_type === "frontier")?.model_id ?? "gemini-3.8-flash"}</h3>
              <div className="agent-subrow">
                <span className="agent-avg-note">3-run avg {frontMeanScore}</span>
                {frontierRuns.length > 1 && (
                  <div className="run-pills" role="group" aria-label="Select recorded frontier run">
                    {frontierRuns.map((run, rIdx) => (
                      <button
                        key={run.run_id}
                        type="button"
                        className={clampedRunIdx === rIdx ? "active" : ""}
                        onClick={() => {
                          setFrontierRunIdx(rIdx);
                          setOpenReasons({});
                        }}
                      >
                        R{rIdx + 1}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="agent-result">
            <strong>{frontScore}</strong>
            <StatusBadge pass={frontValid === 4}>{frontValid}/4 valid</StatusBadge>
          </div>
        </div>

        <div className="job-grid">
          {jobCards.map((job) => {
            const isWhyOpen = Boolean(openReasons[job.key]);
            return (
              <article className={`job-card ${job.pass ? "" : "broken"}`} key={job.key}>
                <div className="job-heading">
                  <div>
                    <span>{job.name}</span>
                    <h3>{job.task}</h3>
                  </div>
                  <StatusBadge pass={job.pass}>{job.pass ? "Valid" : "Failed"}</StatusBadge>
                </div>
                <div className="choice-row">
                  <span className="choice-mark reference" />
                  <div>
                    <small>Reference choice</small>
                    <p>{job.reference}</p>
                  </div>
                </div>
                <div className="choice-row">
                  <span className="choice-mark frontier" />
                  <div>
                    <small>Frontier choice</small>
                    <p>{job.frontier}</p>
                  </div>
                </div>
                {!job.pass && (
                  <>
                    <div className="why-row">
                      <button
                        type="button"
                        className="why-toggle"
                        onClick={() =>
                          setOpenReasons((prev) => ({ ...prev, [job.key]: !prev[job.key] }))
                        }
                        aria-expanded={isWhyOpen}
                      >
                        {isWhyOpen ? "Hide reason ▴" : "Why? ▾"}
                      </button>
                    </div>
                    {isWhyOpen && (
                      <div className="reason">
                        <Icon name="x" size={14} />
                        <span>{job.reason}</span>
                      </div>
                    )}
                  </>
                )}
              </article>
            );
          })}
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Schedule comparison</p>
            <h2>{totalWeeks}-week lane timeline</h2>
          </div>
          <div className="legend">
            <span>
              <i className="ref-color" />
              Reference
            </span>
            <span>
              <i className="front-color" />
              Frontier
            </span>
            <span>
              <i className="blackout-color" />
              Outage
            </span>
          </div>
        </div>
        <div className="timeline-wrap">
          <div className="timeline">
            <div className="timeline-head dynamic-weeks" style={weekHeaderStyle}>
              <div>Execution lane</div>
              {Array.from({ length: totalWeeks }, (_, i) => (
                <span key={i}>W{i + 1}</span>
              ))}
              <div>Status</div>
            </div>
            <TimelineRow
              lane="Lane A · Ref"
              totalWeeks={totalWeeks}
              blocks={laneARef}
              blackouts={blackoutsA}
              deadline="All valid"
            />
            <TimelineRow
              lane="Lane A · Frontier"
              totalWeeks={totalWeeks}
              blocks={laneAFront}
              blackouts={blackoutsA}
              deadline={
                laneAFront.some((b) => b.variant === "front-fail")
                  ? "Rule breach"
                  : laneAFront.length > 0
                    ? "On time"
                    : "Idle"
              }
            />
            <TimelineRow
              lane="Lane B · Ref"
              totalWeeks={totalWeeks}
              blocks={laneBRef}
              blackouts={blackoutsB}
              deadline="All valid"
            />
            <TimelineRow
              lane="Lane B · Frontier"
              totalWeeks={totalWeeks}
              blocks={laneBFront}
              blackouts={blackoutsB}
              deadline={
                laneBFront.some((b) => b.variant === "front-fail")
                  ? "Rule breach"
                  : laneBFront.length > 0
                    ? "On time"
                    : "Idle"
              }
            />
          </div>
        </div>
      </section>

      <section className={`panel tool-panel ${inspectorOpen ? "open" : "collapsed"}`}>
        <div className="panel-heading">
          <div>
            <p className="eyebrow">MCP tool call inspector</p>
            <h2>Recorded decision trace ({traceSteps.length} calls)</h2>
          </div>
          <div className="heading-actions">
            {inspectorOpen && (
              <>
                <div className="trace-agent-switch" role="group" aria-label="Trace agent">
                  <button
                    type="button"
                    className={traceAgent === "frontier" ? "active" : ""}
                    onClick={() => setTraceAgent("frontier")}
                  >
                    Frontier
                  </button>
                  <button
                    type="button"
                    className={traceAgent === "reference" ? "active" : ""}
                    onClick={() => setTraceAgent("reference")}
                  >
                    Reference
                  </button>
                </div>
                {activeShockIdx !== null && (
                  <button
                    type="button"
                    className="secondary-button"
                    onClick={() => setStepIdx(activeShockIdx)}
                  >
                    <Icon name="play" size={15} /> Jump to disruption (Step{" "}
                    {traceSteps[activeShockIdx]?.step_index ?? activeShockIdx + 1})
                  </button>
                )}
              </>
            )}
            <button
              type="button"
              className="compact-toggle"
              onClick={() => setInspectorOpen((prev) => !prev)}
              aria-expanded={inspectorOpen}
            >
              {inspectorOpen ? "Hide tool trace ▴" : "MCP tool call inspector ▾"}
            </button>
          </div>
        </div>

        {inspectorOpen && traceSteps.length > 0 && stepInfo && (
          <>
            <div className="scrubber">
              {traceSteps.map((st, i) => (
                <button
                  key={`${st.step_index}-${i}`}
                  type="button"
                  onClick={() => setStepIdx(i)}
                  className={safeStepIdx >= i ? "active" : ""}
                  aria-label={`Go to step ${st.step_index}`}
                >
                  <span>
                    {safeStepIdx > i ? <Icon name="check" size={12} /> : st.step_index}
                  </span>
                  {i < traceSteps.length - 1 && <i />}
                </button>
              ))}
            </div>
            <div className={`call-detail ${stepInfo.variant}`}>
              <div className="call-number">STEP {currentStep.step_index}</div>
              <div>
                <code>{stepInfo.signature}</code>
                <p>{stepInfo.summary}</p>
              </div>
              <span className="call-status">{stepInfo.status}</span>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function Integrity({ scorecard }: { scorecard: ScorecardData }) {
  const [showAttackDetails, setShowAttackDetails] = useState(false);
  const [showFigure04, setShowFigure04] = useState(false);

  const blockedRatio = `${scorecard.rubric.exploits_blocked}/${scorecard.rubric.exploits_total}`;
  const maxExploit = scorecard.rubric.max_exploit_score.toFixed(2);
  const doNothing = scorecard.rubric.do_nothing_score.toFixed(2);

  const rows = scorecard.exploits.map((item: ExploitAuditItem) => {
    const mapped = EXPLOIT_PLAIN_MAP[item.exploit_id];
    return {
      id: item.exploit_id,
      attack: mapped?.title ?? item.name,
      attempt: mapped?.attempt ?? item.attack_vector,
      defense: mapped?.defense ?? item.defense_mechanism,
      score: item.observed_score.toFixed(2),
      blocked: item.blocked,
    };
  });

  return (
    <div className="view-stack">
      <section className="view-intro">
        <div>
          <p className="eyebrow">Anti-cheat & reward integrity</p>
          <h1>Good scores require actually solving the task.</h1>
          <p>
            Every reward is recomputed from immutable server state. Shortcuts stay at or near zero.
          </p>
        </div>
        <div className="integrity-score">
          <Icon name="shield" size={22} />
          <div>
            <strong>{blockedRatio}</strong>
            <span>exploits blocked</span>
          </div>
        </div>
      </section>

      <section className="integrity-callout">
        <div className="lock-visual">
          <Icon name="shield" size={26} />
        </div>
        <div>
          <h2>Server-authoritative by design</h2>
          <p>
            The final score is based on the verified plan—not agent claims, tool-call volume, or
            malformed output.
          </p>
        </div>
        <div className="callout-points">
          <span>
            <Icon name="check" size={13} /> Strict schema
          </span>
          <span>
            <Icon name="check" size={13} /> Immutable final state
          </span>
          <span>
            <Icon name="check" size={13} /> Global resource ledger
          </span>
        </div>
      </section>

      <section className="panel table-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Adversarial test suite</p>
            <h2>Blocked reward-hacking attacks</h2>
          </div>
          <div className="heading-actions">
            <span className="score-range">
              Observed range&nbsp; <b>0.00–{maxExploit}</b>
            </span>
            <button
              type="button"
              className="compact-toggle"
              onClick={() => setShowFigure04((prev) => !prev)}
              aria-expanded={showFigure04}
            >
              {showFigure04 ? "Hide Figure 04 ▴" : "Figure 04 diagram ▾"}
            </button>
            <button
              type="button"
              className="compact-toggle"
              onClick={() => setShowAttackDetails((prev) => !prev)}
              aria-expanded={showAttackDetails}
            >
              {showAttackDetails ? "Hide attack details ▴" : "Show attack details ▾"}
            </button>
          </div>
        </div>
        {showFigure04 && (
          <div className="inline-figure-drawer">
            <img
              src="./figures/figure-04-reward-integrity.svg"
              alt="Figure 04 — Reward integrity attack matrix (8/8 blocked)"
              loading="lazy"
            />
          </div>
        )}
        <div className="table-scroll">
          <table className={showAttackDetails ? "" : "compact-table"}>
            <thead>
              <tr>
                <th>Attack</th>
                {showAttackDetails && <th>What the agent attempts</th>}
                <th>Defense</th>
                <th>Score</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={row.id}>
                  <td>
                    <span className="table-index">{String(i + 1).padStart(2, "0")}</span>
                    <strong>{row.attack}</strong>
                  </td>
                  {showAttackDetails && <td>{row.attempt}</td>}
                  <td>
                    <span className="defense-chip">
                      <Icon name="shield" size={13} />
                      {row.defense}
                    </span>
                  </td>
                  <td>
                    <code>{row.score}</code>
                  </td>
                  <td>
                    <StatusBadge pass={row.blocked}>
                      {row.blocked ? "Blocked" : "Flagged"}
                    </StatusBadge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="integrity-footnotes">
        <div>
          <strong>{doNothing}</strong>
          <span>Do-nothing baseline</span>
          <p>No partial credit for avoiding the task.</p>
        </div>
        <div>
          <strong>{maxExploit}</strong>
          <span>Highest exploit score</span>
          <p>Far below any valid full-plan result.</p>
        </div>
        <div>
          <strong>100%</strong>
          <span>Final-state verification</span>
          <p>Every submitted constraint is recomputed.</p>
        </div>
      </section>
    </div>
  );
}

const GITHUB_REPO_URL = "https://github.com/AndreiLAB/AgentBench.git";

const INSTALL_COMMANDS = [
  {
    id: "clone-serve",
    label: "Clone & run (~25s, no key)",
    description: "Clone & run offline dashboard (~25s, no API key)",
    cmd: `git clone ${GITHUB_REPO_URL} && cd AgentBench && python scripts/run_eval.py --verify-recorded --serve`,
  },
  {
    id: "agent-skill",
    label: "Install Agent Skill",
    description:
      "Install as an Agent Skill (Codex / OpenCode / Antigravity / Claude Code) — bundled at .agents/skills/agentbreak-bench/SKILL.md",
    cmd: `npx skills add ${GITHUB_REPO_URL}`,
  },
  {
    id: "harbor-oracle",
    label: "Harbor container",
    description: "Run Harbor container task (verified Reference Planner = 0.9445)",
    cmd: "harbor run -p ./harbor_task -a oracle",
  },
];

const JUDGE_COMMANDS = [
  ...INSTALL_COMMANDS,
  {
    id: "offline",
    label: "Verify + UI",
    description: "Verify recorded trajectories & launch web dashboard",
    cmd: "python scripts/run_eval.py --verify-recorded --serve",
  },
  {
    id: "live",
    label: "Live 3× eval",
    description: "Optional full 3× live frontier & held-out evaluation",
    cmd: "python scripts/run_eval.py --all --env-file .env --runs 3",
  },
];

function Learning({
  scorecard,
  judgeOpen,
  onToggleJudge,
}: {
  scorecard: ScorecardData;
  judgeOpen: boolean;
  onToggleJudge: () => void;
}) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copyCmd = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      window.setTimeout(() => setCopiedId(null), 1800);
    } catch {
      setCopiedId(null);
    }
  };

  const tb = scorecard.tiebreaker;
  const relGain = Math.round(tb.relative_gain_pct);
  const beforePct = Math.max(8, Math.min(96, Math.round((tb.before_mean / 0.4) * 100)));
  const afterPct = Math.max(8, Math.min(96, Math.round((tb.after_mean / 0.4) * 100)));
  const refMean = scorecard.rubric.reference_mean.toFixed(2);
  const frontMean = scorecard.rubric.frontier_mean.toFixed(2);

  return (
    <div className="view-stack">
      <section className="view-intro">
        <div>
          <p className="eyebrow">Small-model training & CLI</p>
          <h1>Trajectories teach. Held-out scenarios prove it.</h1>
          <p>
            A smaller model improves on unseen disruptions after learning from benchmark experience.
          </p>
        </div>
        <div className="improvement-pill">
          <Icon name="learn" size={18} />
          <strong>+{relGain}%</strong>
          <span>relative improvement</span>
        </div>
      </section>

      <section className="learning-grid">
        <div className="panel chart-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Generalization result</p>
              <h2>Unseen held-out evaluation</h2>
            </div>
            <span className="model-chip">{tb.model_family}</span>
          </div>
          <div className="bar-chart">
            <div className="axis-labels">
              <span>0.40</span>
              <span>0.30</span>
              <span>0.20</span>
              <span>0.10</span>
              <span>0.00</span>
            </div>
            <div className="grid-lines">
              {Array.from({ length: 5 }, (_, i) => (
                <i key={i} />
              ))}
            </div>
            <div className="bar-item before">
              <div className="bar-value">{tb.before_mean.toFixed(2)}</div>
              <div className="bar" style={{ height: `${beforePct}%` }} />
              <strong>Before</strong>
              <span>Base model</span>
            </div>
            <div className="bar-item after">
              <div className="bar-value">{tb.after_mean.toFixed(2)}</div>
              <div className="bar" style={{ height: `${afterPct}%` }} />
              <strong>After</strong>
              <span>Trajectory learning</span>
            </div>
            <div className="delta-arrow">
              <span>+{tb.delta_improvement.toFixed(2)}</span>
              <Icon name="arrow" size={18} />
            </div>
          </div>
          <div className="chart-note">
            <Icon name="check" size={15} />
            <p>
              <strong>No test-set leakage.</strong> Distilled solely from evaluation trajectories;
              tested automatically on unseen held-out disruptions.
            </p>
          </div>
        </div>

        <div className="panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Automatic held-out suite</p>
              <h2>Unseen disruption breakdown</h2>
            </div>
            <StatusBadge pass>Auto-evaluated</StatusBadge>
          </div>
          <div className="heldout-list">
            {tb.per_seed.map((row) => {
              const meta = HELDOUT_PLAIN_TITLES[row.seed] ?? {
                title: row.scenario_title,
                summary: row.key_learned_adaptation,
              };
              const positive = row.delta > 0.0005;
              return (
                <div className="heldout-item" key={row.seed}>
                  <div>
                    <strong>{meta.title}</strong>
                    <p>{meta.summary}</p>
                  </div>
                  <div className="heldout-scores">
                    <span className="before-val">{row.before_score.toFixed(2)}</span>
                    <Icon name="arrow" size={13} />
                    <span className="after-val">{row.after_score.toFixed(2)}</span>
                    <span className={`heldout-delta ${positive ? "" : "neutral"}`}>
                      {positive ? `+${row.delta.toFixed(2)}` : "maintained"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="heldout-auto-note">
            <span>Kept separate from interactive replay to preserve clean held-out separation.</span>
            <strong>{tb.per_seed.length} unseen tests</strong>
          </div>
        </div>
      </section>

      <section id="judge-cli-guide" className="judge-collapsible">
        <button
          type="button"
          className="expander-bar"
          onClick={onToggleJudge}
          aria-expanded={judgeOpen}
        >
          <div className="expander-bar-left">
            <span className="expander-bar-title">Reproduce & run locally</span>
            <span className="expander-bar-hint">
              1-command offline replay + Harbor sandbox + 3-step audit checklist
            </span>
          </div>
          <span className="expander-pill">
            {judgeOpen ? "Hide CLI & Harbor quick-start ▴" : "CLI & Harbor quick-start ▾"}
          </span>
        </button>

        {judgeOpen && (
          <div className="judge-drawer-stack">
            <div className="panel judge-panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Reproduce & run locally</p>
                  <h2>Reproduce in one command</h2>
                </div>
                <Icon name="terminal" size={24} />
              </div>
              <ol className="judge-steps">
                <li>
                  <span>1</span>
                  <div>
                    <strong>Verify recorded trajectories & launch this viewer (no API key required)</strong>
                    <p>Re-audits every trajectory from scratch, writes scorecards, and serves port 8095.</p>
                  </div>
                </li>
                <li>
                  <span>2</span>
                  <div>
                    <strong>Optional: Run full 3× live evaluation or Harbor container check</strong>
                    <p>Re-runs live frontier & held-out evaluations or executes the Harbor oracle task.</p>
                  </div>
                </li>
              </ol>

              <div className="terminal-box">
                <div className="terminal-top">
                  <span>
                    <i />
                    <i />
                    <i />
                  </span>
                  <small>TERMINAL · PROJECT ROOT</small>
                </div>
                {JUDGE_COMMANDS.map((item) => (
                  <div className="terminal-command compact-cmd" key={item.id}>
                    <span className="cmd-label">{item.label}</span>
                    <code>{item.cmd}</code>
                    <button
                      type="button"
                      onClick={() => copyCmd(item.id, item.cmd)}
                      aria-label={`Copy ${item.label} command`}
                    >
                      <Icon name={copiedId === item.id ? "check" : "copy"} size={16} />
                      {copiedId === item.id ? "Copied" : "Copy"}
                    </button>
                  </div>
                ))}
              </div>

              <div className="expected">
                <span>Expected outputs</span>
                <div>
                  <Icon name="check" size={14} />
                  <p>
                    Bundled agent skill at{" "}
                    <code>.agents/skills/agentbreak-bench/SKILL.md</code> for zero-key MCP
                    playtesting or Harbor evaluation
                  </p>
                </div>
                <div>
                  <Icon name="check" size={14} />
                  <p>
                    Scorecard written to <code>runs/scorecard.json</code> and{" "}
                    <code>runs/SCORECARD.md</code>
                  </p>
                </div>
                <div>
                  <Icon name="check" size={14} />
                  <p>Per-step MCP traces served live on http://127.0.0.1:8095</p>
                </div>
              </div>
            </div>

            <div className="judge-checklist">
              <div>
                <p className="eyebrow">What to verify</p>
                <h2>Three checks. Under five minutes.</h2>
              </div>
              <div className="check-item">
                <span>01</span>
                <p>
                  <strong>Score separation</strong> Reference stays at {refMean} while the frontier
                  agent averages {frontMean} (+{scorecard.rubric.gap_delta.toFixed(2)} gap).
                </p>
              </div>
              <div className="check-item">
                <span>02</span>
                <p>
                  <strong>Hard constraints</strong> Open any failed job in Agent Comparison and
                  click Why? to confirm the exact server-side rule violation.
                </p>
              </div>
              <div className="check-item">
                <span>03</span>
                <p>
                  <strong>Held-out lift</strong> Confirm {tb.model_family} improves from{" "}
                  {tb.before_mean.toFixed(2)} to {tb.after_mean.toFixed(2)} (+{relGain}%) on unseen
                  held-out disruptions.
                </p>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [preferredSeed, setPreferredSeed] = useState<string>(EVAL_SCENARIOS[0].seed);
  const [judgeOpen, setJudgeOpen] = useState(false);
  const [scorecard, setScorecard] = useState<ScorecardData>(() =>
    normalizeScorecardPayload(FALLBACK_SCORECARD)
  );

  useEffect(() => {
    let cancelled = false;
    loadStudioData().then((bundle) => {
      if (!cancelled) {
        setScorecard(bundle.scorecard);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const current = useMemo(() => tabs.find((tab) => tab.id === activeTab), [activeTab]);

  const openRunGuide = () => {
    setActiveTab("learning");
    setJudgeOpen(true);
    window.setTimeout(() => {
      document.getElementById("judge-cli-guide")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 60);
  };

  return (
    <div className="app-shell">
      <header>
        <div className="header-inner">
          <button
            type="button"
            className="brand"
            onClick={() => setActiveTab("overview")}
            aria-label="AgentBreak Bench home"
          >
            <img
              src="./assets/agentbreak-mark.svg"
              alt=""
              className="brand-mark-img"
              width={32}
              height={32}
            />
            <span>
              <strong>AgentBreak</strong>
              <small>BENCH</small>
            </span>
          </button>
          <div className="header-meta">
            <span className="byline">by AfterQuery</span>
            <span className="version">v1.0</span>
            <button type="button" onClick={openRunGuide} className="github-link">
              Run Guide <Icon name="arrow" size={13} />
            </button>
            <a
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="github-link"
            >
              GitHub <Icon name="arrow" size={13} />
            </a>
          </div>
        </div>
        <nav className="tab-wrap" aria-label="Dashboard views">
          <div className="tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={activeTab === tab.id ? "active" : ""}
                onClick={() => setActiveTab(tab.id)}
                aria-current={activeTab === tab.id ? "page" : undefined}
              >
                <Icon name={tab.icon} size={16} />
                {tab.label}
              </button>
            ))}
          </div>
        </nav>
      </header>
      <main key={current?.id}>
        {activeTab === "overview" && (
          <Overview
            scorecard={scorecard}
            onExplore={() => {
              setActiveTab("comparison");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            onOpenIntegrity={() => {
              setActiveTab("integrity");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
            onSelectScenario={(seed) => {
              setPreferredSeed(seed);
              setActiveTab("comparison");
              window.scrollTo({ top: 0, behavior: "smooth" });
            }}
          />
        )}
        {activeTab === "comparison" && (
          <Comparison scorecard={scorecard} initialSeed={preferredSeed} />
        )}
        {activeTab === "integrity" && <Integrity scorecard={scorecard} />}
        {activeTab === "learning" && (
          <Learning
            scorecard={scorecard}
            judgeOpen={judgeOpen}
            onToggleJudge={() => setJudgeOpen((prev) => !prev)}
          />
        )}
      </main>
      <footer>
        <div>
          <img
            src="./assets/agentbreak-mark.svg"
            alt=""
            className="brand-mark-img small"
            width={24}
            height={24}
          />
          <strong>AgentBreak Bench</strong>
        </div>
        <p>Multi-Constraint Operations & Disruption-Recovery AI Benchmark</p>
        <span>Built by AfterQuery</span>
      </footer>
    </div>
  );
}

export default App;
