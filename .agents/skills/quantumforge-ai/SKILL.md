---
name: quantumforge-ai
description: >-
  Run, verify, playtest, and inspect QuantumForge AI (CryoForge multi-constraint
  operations & disruption-recovery benchmark for AI agents). Use when the user asks to
  reproduce benchmark scores, launch the web dashboard, playtest the MCP
  environment directly from Codex/OpenCode/Antigravity without extra API keys,
  run Harbor evaluations (oracle, nop, or opencode), or benchmark an external LLM.
---

# QuantumForge AI

Install this skill into Codex, OpenCode, Antigravity, or Claude Code:

```bash
npx skills add https://github.com/keshav/QuantumForge-AI.git
```

- **Public Web Dashboard (GitHub Pages):** https://keshav.github.io/QuantumForge-AI/
- **Repository:** https://github.com/keshav/QuantumForge-AI.git

> **Important:** Never overwrite or re-run `runs/trajectories/` or `runs/scorecard.json` unless the user explicitly asks to benchmark a new model. Use `--verify-recorded` when verifying or launching the local dashboard.

---

## Mode 1: No API Key — Verify Recorded Trajectories, Dashboard & Harbor Baselines

### A. Offline Deterministic Verify + Web Dashboard (`~25s`, Zero API Keys)
Runs all 17 unit tests, executes the Reference Planner (`S01–S08`), `do_nothing` (`0.0000`) & `stop_early` (`0.0800`) baselines, and all 8 adversarial exploit checks over live `stdio` MCP without modifying recorded LLM runs, then serves the web dashboard at `http://127.0.0.1:8095`:

```bash
python scripts/run_eval.py --verify-recorded --serve
```

- Health check: `GET http://127.0.0.1:8095/api/health` (`ok: true`)
- Recorded scorecards: `runs/scorecard.json` and `runs/SCORECARD.md`

### B. Harbor Container Verification (`oracle` & `nop`, Zero API Keys)
Runs the isolated two-container Harbor environment (`cryoforge-mcp` SSE sidecar + `/workspace` agent container):

```bash
# Reference Planner inside Harbor (verified reward: 0.9445 in /logs/verifier/reward.txt)
harbor run -p ./harbor_task -a oracle --jobs-dir runs/harbor_jobs

# Do-Nothing baseline inside Harbor (verified reward: 0.0000)
harbor run -p ./harbor_task -a nop --jobs-dir runs/harbor_jobs
```

- **No local Docker?** Append `--env modal` (uses Modal DinD when `~/.modal.toml` is configured).

---

## Mode 2: No Extra API Key — Live Agent Playtest via MCP

Any active **Codex**, **OpenCode**, **Claude Code**, or **Antigravity** session can benchmark *itself* directly against the live MCP environment using its existing host session—no separate `.env` API key required:

```bash
# Start live stdio MCP server for Scenario S01..S08
python -m mcp_server.server --transport stdio --seed S01

# Or start SSE MCP server on port 8090
python -m mcp_server.server --transport sse --host 127.0.0.1 --port 8090 --seed S01
```

### Episode Playbook for the Active Agent
1. Call `get_episode_brief()` and `query_handbook(section="all")` on Turn 1.
2. Inspect all 4 orders (`inspect_order("ORD-1".."ORD-4")`) and candidate routes (`inspect_route(...)`).
3. Compute wall thickness, thermal boil-off, and tare/transport mass from the handbook formulas; conserve the **3-call `run_physics_sim` budget**.
4. Book initial fabrication slots with `reserve_bay_and_order(...)` and `assign_transport(...)`.
5. On the **4th mutating tool call**, a live `disruption_alert` fires—immediately call `revise_order(...)` on any affected orders.
6. Call `finalize_commissioning()` to trigger the zero-trust terminal verifier and record the final score (`0.0000` if omitted).

---

## Mode 3: API Key Needed — Automated Standalone LLM Benchmarking

External provider API keys (`GEMINI_API_KEY` or `OPENAI_API_KEY`) are **only** needed when running standalone automated rollouts against external models:

```bash
# 1. Automated Harbor + OpenCode container evaluation
harbor run -p ./harbor_task -a opencode -m gemini/gemini-3.8-flash --jobs-dir runs/harbor_jobs

# 2. Full 3-run multi-seed benchmark suite (Gemini or OpenAI-compatible endpoint)
python scripts/run_eval.py --all --env-file .env --frontier-provider gemini --frontier-model gemini-3.8-flash --runs 3
python scripts/run_eval.py --all --frontier-provider openai --frontier-model gpt-4.1 --runs 3
```

Useful flags:
- `--frontier-provider` (`gemini` | `openai` | `harbor`)
- `--base-url <url>` for local vLLM / Ollama / OpenRouter OpenAI-compatible endpoints
- `--skip-tiebreaker` to skip the `S06–S08` small-model evaluation

---

## Quick Validation Checklist

- Unit invariants: `python -m pytest tests/test_physics_and_verifier.py -q` (`17 passed`)
- Harbor logs: `runs/harbor_jobs/` (`oracle = 0.9445`, `nop = 0.0000`)
- Dashboard API: `GET http://127.0.0.1:8095/api/health` returns `{"ok": true}`
