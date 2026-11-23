# CryoForge RL — Master Evaluation & Anti-Exploit Scorecard

- **Generated At:** `2026-11-14T14:19:36.087350+00:00`
- **Environment:** `CryoForge-RL v1.0.0`
- **Evaluated Frontier Agent:** `gemini-3.8-flash` (3× repeated runs over MCP)

## 1. Core AfterQuery Rubric Metrics

| Metric | Observed Value | Target / Invariant | Status |
|---|---:|---:|:---:|
| **Reference Solver Mean (`S01–S05`)** | **`0.9491`** | `>= 0.9400` (Public MCP tools only) | PASS |
| **Frontier Agent 3× Mean ± Std (`S01–S05`)** | **`0.3162 ± 0.2496`** | Consistent below Reference | PASS |
| **The Gap (`Reference - Frontier`)** | **`+0.6329`** | Measurable, reproducible gap | PASS |
| **Do-Nothing Baseline (`S01–S05`)** | **`0.0000`** | `== 0.0000` | PASS |
| **Stop-Early Baseline (`S01–S05`)** | **`0.0800`** | `<= 0.1000` (`0.08` batch gate) | PASS |
| **Adversarial Exploit Audit** | **`8/8` Blocked** | Max exploit score `0.0800 <= 0.15` | PASS |
| **Tiebreaker (`S06–S08` Held-Out Small Model)** | **`0.1400 → 0.3400` (`+0.2000`)** | `After > Before` (`Zero S06–S08 Leakage`) | PASS |

## 2. Per-Seed Reproducibility Breakdown (`S01–S05` Eval)

| Seed | Scenario | Reference | Frontier 3× Runs | Frontier Mean ± Std | Gap Δ |
|---|---|---:|---|---:|---:|
| `S01` | Nordic Cryo Expansion — Customs Hold, Perlite Humidity & Fjord Bridge Downgrade | `0.9445` | `0.340`, `0.340`, `0.340` | `0.3400 ± 0.0000` | `+0.6045` |
| `S02` | Alpine LNG & Oxygen Hub — Bay-B Furnace Outage & Contingency Freeze | `0.9483` | `0.587`, `0.080`, `0.080` | `0.2488 ± 0.2388` | `+0.6995` |
| `S03` | Semiconductor Fab Cryo Batch — Perlite Contamination & Permit Cut | `0.9514` | `0.340`, `0.080`, `0.068` | `0.1627 ± 0.1255` | `+0.7887` |
| `S04` | Trans-Alpine Propulsion Test — ORD-3 Pressure Uprate & Tunnel Height Cap | `0.9480` | `0.340`, `0.068`, `0.892` | `0.4334 ± 0.3429` | `+0.5146` |
| `S05` | Heavy Cryo Industrial Hub — EU_STD Valve Delay & Bay-A Crane Derating | `0.9533` | `0.587`, `0.016`, `0.587` | `0.3963 ± 0.2689` | `+0.5570` |

## 3. Held-Out Tiebreaker Breakdown (`S06–S08` Unseen Seeds)

- **Small Model:** `gemini-3.5-flash-lite`
- **Training Source:** `S01, S02, S03, S04, S05` ONLY (`0` holdout leakage verified)

| Held-Out Seed | Scenario | Base Small Model (`Before`) | Trained / Experience Policy (`After`) | Improvement Δ |
|---|---|---:|---:|---:|
| `S06` | Quantum & Fusion Extreme Cryo — LHe/LH2 Precision Delay & Fjord Cap | `0.3400` (2/4) | `0.3400` (2/4) | **`+0.0000`** |
| `S07` | Mediterranean Bunkering & Medical Cryo — Bay-A Outage, PVI Spike & Permit Cut | `0.0800` (1/4) | `0.3400` (2/4) | **`+0.2600`** |
| `S08` | Continental Hydrogen & Helium Corridor — Pressure Uprate, Crane Cap & Valve Delay | `0.0000` (0/4) | `0.3400` (2/4) | **`+0.3400`** |

## 4. Reward Integrity — 8 Adversarial Exploit Checks

| Exploit ID | Attack Vector | Observed Score | Status |
|---|---|---:|:---:|
| `EXPLOIT_01_IMMEDIATE_FINALIZE` | Calls finalize_commissioning() immediately with zero booked orders hoping for base completion credit. | `0.0000` | BLOCKED |
| `EXPLOIT_02_CHERRY_PICK_ONE_EASY_ORDER` | Solves warm-up order ORD-1 validly and finalizes early to avoid coupled shock/bay contention on ORD-2..4. | `0.0800` | BLOCKED |
| `EXPLOIT_03_SCHEMA_INJECTION_EXTRA_FIELDS` | Injects extra fields (score=1.0, override_safety=true), negative volumes, bool-as-int, null bytes, and NaN/Inf. | `0.0000` | BLOCKED |
| `EXPLOIT_04_ORACLE_FARMING_VIA_RESERVE` | Attempts to bypass 3-call run_physics_sim cap by probing reserve_bay_and_order and revise_order in a loop. | `0.0000` | BLOCKED |
| `EXPLOIT_05_SIM_BUDGET_OVERFLOW` | Attempts >3 calls to run_physics_sim to brute-force vessel wall thicknesses. | `0.0000` | BLOCKED |
| `EXPLOIT_06_RESOURCE_DOUBLE_SPEND` | Books colliding fabrication weeks on BAY-A and claims multiple scarce escort permits beyond quota. | `0.0000` | BLOCKED |
| `EXPLOIT_07_POST_FINALIZE_MUTATION` | Calls finalize_commissioning() then attempts to revise orders, run sims, or re-finalize. | `0.0000` | BLOCKED |
| `EXPLOIT_08_IGNORE_MID_EPISODE_SHOCK` | Pre-books naive pre-shock specs across all 4 orders hoping pre-booked vessels bypass Step-4 customs/bridge shocks. | `0.0800` | BLOCKED |
