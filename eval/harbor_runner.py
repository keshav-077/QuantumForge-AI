"""Multi-Provider Frontier Agent Benchmark Runner over MCP (`gemini`, `openai`, `harbor`).

Connects frontier LLMs strictly to the CryoForge MCP server (`stdio` or `sse`) using the
exact fair task prompt in `harbor_task/instruction.md` (zero hints or hidden tricks) across
`S01–S05` (supports `--runs 3` for 3x reproducibility):

- `provider="gemini"`: Uses `google-genai` v3.x `client.interactions.create(...)` with
  interaction-scoped `tools` and `system_instruction` passed on every turn alongside
  `previous_interaction_id`, handling multi-call turns via `function_result` step inputs.
  Default model: `gemini-4-argon` (also supports `gemini-3.8-flash` and Gemma 4).
- `provider="openai"`: Uses `openai.OpenAI(api_key=..., base_url=...)` Chat Completions
  tool-calling loop (compatible with OpenAI GPT/o-series, OpenRouter, vLLM, Ollama,
  Anthropic OpenAI-compat, and AfterQuery OpenCode endpoints).
- `provider="harbor"`: Invokes the `harbor run` CLI against `harbor_task/` and parses
  `/logs/verifier/reward.txt` + `report.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from typing import Any

from eval.mcp_client_helper import (
    DEFAULT_TRAJECTORIES_DIR,
    PROJECT_ROOT,
    CryoForgeMCPClient,
    load_persisted_trajectory,
    open_cryoforge_mcp_session,
)

logger = logging.getLogger("cryoforge.harbor_runner")

DEFAULT_EVAL_SEEDS: tuple[str, ...] = ("S01", "S02", "S03", "S04", "S05")
INSTRUCTION_PATH: Path = PROJECT_ROOT / "harbor_task" / "instruction.md"

FAIR_SYSTEM_INSTRUCTION: str = (
    "You are an autonomous Cryogenic Plant Commissioning Engineer connected to the "
    "`cryoforge` MCP server. Use only the provided MCP tools to inspect the episode brief, "
    "handbook equations, customer orders, and transport routes; design, schedule, and route "
    "all 4 orders within factory constraints; adapt to any mid-episode disruptions; and "
    "call `finalize_commissioning()` before completing your work."
)

_SECRET_ENV_KEYS: tuple[str, ...] = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "MODAL_TOKEN_SECRET",
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"sk-[0-9A-Za-z_\-]{16,}"),
    re.compile(r"Bearer\s+[0-9A-Za-z_\-\.]{16,}"),
)


def load_env_file(
    env_file: str | Path | None = None,
    *,
    override: bool = True,
) -> dict[str, str]:
    """Load `KEY=VALUE` environment variables safely from an external `.env` file."""
    loaded: dict[str, str] = {}
    candidates: list[tuple[Path, bool]] = []
    default_repo_env = PROJECT_ROOT / ".env"
    if default_repo_env.is_file():
        candidates.append((default_repo_env, False))
    if env_file:
        explicit_path = Path(env_file).expanduser().resolve()
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Environment file not found: {explicit_path}")
        candidates.append((explicit_path, override))

    for path, should_override in candidates:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            if not key or not val:
                continue
            if should_override or not os.environ.get(key):
                os.environ[key] = val
                loaded[key] = val
    return loaded


def redact_secrets(text: str) -> str:
    """Scrub API keys and bearer tokens from any exception or log string."""
    if not text:
        return ""
    scrubbed = str(text)
    for env_key in _SECRET_ENV_KEYS:
        secret_val = os.environ.get(env_key, "")
        if secret_val and len(secret_val) >= 6:
            scrubbed = scrubbed.replace(secret_val, f"<REDACTED:{env_key}>")
    for pat in _SECRET_PATTERNS:
        scrubbed = pat.sub("<REDACTED_TOKEN>", scrubbed)
    return scrubbed


def normalize_gemini_model(model: str) -> str:
    """Enforce current Gemini 3.x / 4.x model IDs and replace deprecated 1.5/2.x IDs."""
    cleaned = (model or "gemini-4-argon").strip()
    lowered = cleaned.lower()
    if lowered.startswith(("gemini-2.", "gemini-1.")):
        logger.warning(
            "Model '%s' is legacy/deprecated; substituting 'gemini-3.8-flash'.",
            cleaned,
        )
        return "gemini-3.8-flash"
    return cleaned


def load_fair_task_prompt(instruction_path: Path = INSTRUCTION_PATH) -> str:
    """Load the exact benchmark prompt from `harbor_task/instruction.md`."""
    if instruction_path.is_file():
        return instruction_path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"Benchmark instruction prompt missing at {instruction_path}")


def _extract_gemini_function_calls(interaction: Any) -> list[dict[str, Any]]:
    """Extract all `function_call` steps from a `google-genai` v3.x `Interaction` object."""
    calls: list[dict[str, Any]] = []
    steps = getattr(interaction, "steps", None) or []
    for idx, step in enumerate(steps):
        step_type = (
            step.get("type")
            if isinstance(step, dict)
            else getattr(step, "type", None)
        )
        if step_type != "function_call":
            continue

        if isinstance(step, dict):
            call_id = str(step.get("id") or f"call_{idx}")
            fn_name = str(step.get("name") or "")
            raw_args = step.get("arguments") or {}
        else:
            call_id = str(getattr(step, "id", None) or f"call_{idx}")
            fn_name = str(getattr(step, "name", None) or "")
            raw_args = getattr(step, "arguments", None) or {}

        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {"_raw_unparsed_arguments": raw_args}
        elif isinstance(raw_args, dict):
            parsed_args = dict(raw_args)
        else:
            parsed_args = {}

        calls.append(
            {
                "id": call_id,
                "name": fn_name,
                "arguments": parsed_args,
            }
        )
    return calls


async def _run_gemini_agent_loop(
    mcp_client: CryoForgeMCPClient,
    *,
    model: str,
    task_prompt: str,
    system_instruction: str = FAIR_SYSTEM_INSTRUCTION,
    max_llm_turns: int = 45,
    request_timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Run `google-genai` v3.x `client.interactions.create` tool-calling loop over MCP."""
    from google import genai

    resolved_model = normalize_gemini_model(model)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Pass --env-file <path> or export GEMINI_API_KEY."
        )

    client = genai.Client(api_key=api_key, http_options={"timeout": int(request_timeout_s * 1000)})
    gemini_tools = await mcp_client.get_gemini_tools()
    thinking_level = os.environ.get("CRYOFORGE_THINKING_LEVEL", "low")

    prev_interaction_id: str | None = None
    next_input: Any = task_prompt
    llm_turns = 0
    total_tokens = 0
    final_output_text = ""

    while llm_turns < max_llm_turns:
        llm_turns += 1
        create_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "input": next_input,
            "system_instruction": system_instruction,
            "tools": gemini_tools,
            "timeout": request_timeout_s,
        }
        if resolved_model.lower().startswith(("gemini-3.5", "gemini-3.8", "gemini-4")):
            create_kwargs["generation_config"] = {
                "max_output_tokens": 8192,
                "thinking_level": thinking_level,
            }
        else:
            create_kwargs["generation_config"] = {"max_output_tokens": 4096}
        if prev_interaction_id is not None:
            create_kwargs["previous_interaction_id"] = prev_interaction_id

        interaction: Any = None
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                interaction = await asyncio.to_thread(
                    client.interactions.create,
                    **create_kwargs,
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                err_text = redact_secrets(str(exc)).lower()
                if attempt < 2 and any(
                    code in err_text for code in ("429", "503", "rate", "overloaded", "timeout")
                ):
                    await asyncio.sleep(2.0 * (2**attempt))
                    continue
                break

        if last_exc is not None or interaction is None:
            safe_err = redact_secrets(str(last_exc))
            return {
                "provider_error": safe_err,
                "llm_turns": llm_turns,
                "total_tokens": total_tokens,
                "model": resolved_model,
            }

        prev_interaction_id = getattr(interaction, "id", None)
        usage = getattr(interaction, "usage", None)
        if usage is not None:
            turn_tokens = getattr(usage, "total_tokens", None)
            if isinstance(turn_tokens, int):
                total_tokens += turn_tokens

        out_text = getattr(interaction, "output_text", None)
        if isinstance(out_text, str) and out_text.strip():
            final_output_text = out_text.strip()

        fn_calls = _extract_gemini_function_calls(interaction)
        if not fn_calls:
            break

        fn_results: list[dict[str, Any]] = []
        episode_terminated = False
        for call in fn_calls:
            obs = await mcp_client.call_tool(call["name"], call["arguments"])
            fn_results.append(
                {
                    "type": "function_result",
                    "call_id": call["id"],
                    "name": call["name"],
                    "result": obs,
                }
            )
            if obs.get("finalized") is True or obs.get("error") in (
                "MAX_STEPS_EXCEEDED",
                "EPISODE_ALREADY_FINALIZED",
            ):
                episode_terminated = True

        if episode_terminated:
            break
        next_input = fn_results

    return {
        "provider_error": None,
        "llm_turns": llm_turns,
        "total_tokens": total_tokens,
        "model": resolved_model,
        "final_output_text": final_output_text,
    }


async def _run_openai_agent_loop(
    mcp_client: CryoForgeMCPClient,
    *,
    model: str,
    task_prompt: str,
    base_url: str | None = None,
    api_key: str | None = None,
    system_instruction: str = FAIR_SYSTEM_INSTRUCTION,
    max_llm_turns: int = 45,
    request_timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Run OpenAI-compatible Chat Completions tool-calling loop over MCP."""
    from openai import OpenAI

    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
    resolved_api_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or ("EMPTY" if resolved_base_url else None)
    )
    if not resolved_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Pass --env-file <path>, --base-url for local vLLM/Ollama, or export OPENAI_API_KEY."
        )

    oai_client = OpenAI(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        timeout=request_timeout_s,
    )
    openai_tools = await mcp_client.get_openai_tools()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": task_prompt},
    ]
    llm_turns = 0
    total_tokens = 0
    final_output_text = ""

    while llm_turns < max_llm_turns:
        llm_turns += 1
        completion: Any = None
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                completion = await asyncio.to_thread(
                    oai_client.chat.completions.create,
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    tools=openai_tools,  # type: ignore[arg-type]
                )
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                err_text = redact_secrets(str(exc)).lower()
                if attempt < 2 and any(
                    code in err_text for code in ("429", "503", "rate", "overloaded", "timeout")
                ):
                    await asyncio.sleep(2.0 * (2**attempt))
                    continue
                break

        if last_exc is not None or completion is None:
            return {
                "provider_error": redact_secrets(str(last_exc)),
                "llm_turns": llm_turns,
                "total_tokens": total_tokens,
                "model": model,
            }

        usage = getattr(completion, "usage", None)
        if usage is not None:
            t_tok = getattr(usage, "total_tokens", None)
            if isinstance(t_tok, int):
                total_tokens += t_tok

        choice_msg = completion.choices[0].message
        if choice_msg.content:
            final_output_text = str(choice_msg.content).strip()

        tool_calls = getattr(choice_msg, "tool_calls", None) or []
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": choice_msg.content or "",
        }
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            break

        episode_terminated = False
        for tc in tool_calls:
            fn_name = tc.function.name
            raw_args_str = tc.function.arguments or "{}"
            try:
                parsed_args = json.loads(raw_args_str)
                if not isinstance(parsed_args, dict):
                    parsed_args = {}
                obs = await mcp_client.call_tool(fn_name, parsed_args)
            except json.JSONDecodeError as exc:
                obs = {
                    "ok": False,
                    "error": "INVALID_TOOL_JSON",
                    "tool": fn_name,
                    "message": f"Malformed JSON tool arguments: {exc}",
                }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs),
                }
            )
            if obs.get("finalized") is True or obs.get("error") in (
                "MAX_STEPS_EXCEEDED",
                "EPISODE_ALREADY_FINALIZED",
            ):
                episode_terminated = True

        if episode_terminated:
            break

    return {
        "provider_error": None,
        "llm_turns": llm_turns,
        "total_tokens": total_tokens,
        "model": model,
        "final_output_text": final_output_text,
    }


def _run_harbor_cli_episode(
    *,
    seed: str,
    model: str,
    run_id: str,
    harbor_agent: str = "opencode",
    log_dir: str | Path | None = DEFAULT_TRAJECTORIES_DIR,
) -> dict[str, Any]:
    """Invoke `harbor run` CLI against `harbor_task/` and parse `reward.txt` + `report.json`."""
    harbor_bin = os.environ.get("HARBOR_BIN", "harbor")
    task_dir = PROJECT_ROOT / "harbor_task"

    with tempfile.TemporaryDirectory(prefix=f"harbor_{seed}_") as jobs_tmp:
        jobs_path = Path(jobs_tmp)
        cmd = [
            harbor_bin,
            "run",
            "--path",
            str(task_dir),
            "--agent",
            harbor_agent,
            "--model",
            model,
            "--jobs-dir",
            str(jobs_path),
        ]
        child_env = dict(os.environ)
        child_env["CRYOFORGE_SEED"] = seed
        child_env["CRYOFORGE_RUN_ID"] = run_id

        t0 = time.monotonic()
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed_s = round(time.monotonic() - t0, 2)

        reward_files = list(jobs_path.rglob("verifier/reward.txt")) + list(
            jobs_path.rglob("reward.txt")
        )
        report_files = list(jobs_path.rglob("verifier/report.json")) + list(
            jobs_path.rglob("report.json")
        )

        reward_score = 0.0
        if reward_files:
            raw_reward = reward_files[0].read_text(encoding="utf-8").strip()
            try:
                reward_score = float(raw_reward.splitlines()[0].strip())
            except (ValueError, IndexError):
                reward_score = 0.0

        report_payload: dict[str, Any] = {}
        if report_files:
            try:
                report_payload = json.loads(report_files[0].read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report_payload = {}

        verification = report_payload.get("verification") or report_payload
        trajectory = report_payload.get("trajectory")
        if trajectory and log_dir:
            out_dir = Path(log_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{run_id}.json").write_text(
                json.dumps(trajectory, indent=2), encoding="utf-8"
            )

        return {
            "seed": seed,
            "run_id": run_id,
            "provider": "harbor",
            "model": model,
            "elapsed_seconds": elapsed_s,
            "return_code": proc.returncode,
            "provider_error": (
                redact_secrets(proc.stderr[-800:]) if proc.returncode != 0 else None
            ),
            "final_score": float(verification.get("final_score", reward_score)),
            "finalized": bool(verification.get("finalized", False)),
            "valid_order_count": int(verification.get("valid_order_count", 0)),
            "verification": verification,
        }


async def run_single_frontier_episode(
    *,
    seed: str = "S01",
    provider: str = "gemini",
    model: str | None = None,
    run_index: int = 1,
    mcp_url: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    agent_type: str = "frontier",
    agent_name: str | None = None,
    log_dir: str | Path | None = DEFAULT_TRAJECTORIES_DIR,
    harbor_agent: str = "opencode",
    max_llm_turns: int = 45,
) -> dict[str, Any]:
    """Run one benchmark episode for `(seed, run_index)` on `provider` over MCP."""
    norm_provider = provider.strip().lower()
    if norm_provider not in ("gemini", "openai", "harbor"):
        raise ValueError(
            f"Unsupported provider '{provider}'. Choose from: gemini, openai, harbor."
        )

    if model is None:
        if norm_provider == "gemini":
            resolved_model = os.environ.get("CRYOFORGE_FRONTIER_MODEL", "gemini-4-argon")
        else:
            resolved_model = os.environ.get("CRYOFORGE_FRONTIER_MODEL", "gpt-4.1")
    else:
        resolved_model = model

    if norm_provider == "gemini":
        resolved_model = normalize_gemini_model(resolved_model)

    safe_model_slug = re.sub(r"[^A-Za-z0-9_\-]+", "-", resolved_model).strip("-")
    run_id = f"{agent_type}_{safe_model_slug}_{seed}_r{run_index}"
    display_name = agent_name or f"Frontier ({resolved_model})"

    if norm_provider == "harbor":
        return await asyncio.to_thread(
            _run_harbor_cli_episode,
            seed=seed,
            model=resolved_model,
            run_id=run_id,
            harbor_agent=harbor_agent,
            log_dir=log_dir,
        )

    task_prompt = load_fair_task_prompt()
    temp_ctx = tempfile.TemporaryDirectory() if (log_dir is None and not mcp_url) else None
    effective_log_dir = Path(temp_ctx.name) if temp_ctx is not None else log_dir

    t0 = time.monotonic()
    try:
        async with open_cryoforge_mcp_session(
            seed=seed,
            mcp_url=mcp_url,
            run_id=run_id,
            agent_name=display_name,
            agent_type=agent_type,
            model_id=resolved_model,
            log_dir=effective_log_dir,
        ) as mcp_client:
            if norm_provider == "gemini":
                loop_meta = await _run_gemini_agent_loop(
                    mcp_client,
                    model=resolved_model,
                    task_prompt=task_prompt,
                    max_llm_turns=max_llm_turns,
                )
            else:
                loop_meta = await _run_openai_agent_loop(
                    mcp_client,
                    model=resolved_model,
                    task_prompt=task_prompt,
                    base_url=base_url,
                    api_key=api_key,
                    max_llm_turns=max_llm_turns,
                )

            sse_grade = mcp_client.fetch_sse_grade() if mcp_url else None

        elapsed_s = round(time.monotonic() - t0, 2)
        persisted = (
            load_persisted_trajectory(run_id, effective_log_dir)
            if not mcp_url
            else None
        )
        verification = (sse_grade or {}).get("verification") or (persisted or {}).get(
            "verification"
        )
        if verification is None:
            raise RuntimeError(
                f"Missing server verification report after {norm_provider} episode {run_id}."
            )

        return {
            "seed": seed,
            "run_index": run_index,
            "run_id": run_id,
            "provider": norm_provider,
            "model": loop_meta.get("model", resolved_model),
            "elapsed_seconds": elapsed_s,
            "llm_turns": loop_meta.get("llm_turns", 0),
            "total_tokens": loop_meta.get("total_tokens", 0),
            "provider_error": loop_meta.get("provider_error"),
            "finalized": bool(verification["finalized"]),
            "final_score": float(verification["final_score"]),
            "valid_order_count": int(verification["valid_order_count"]),
            "step_count": int(verification["step_count"]),
            "sim_calls_used": int(verification["sim_calls_used"]),
            "failure_modes": list(verification.get("failure_modes", [])),
            "primary_failure_categories": list(
                verification.get("primary_failure_categories", [])
            ),
            "verification": verification,
        }
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


async def run_frontier_benchmark(
    *,
    provider: str = "gemini",
    model: str | None = None,
    seeds: tuple[str, ...] = DEFAULT_EVAL_SEEDS,
    runs: int = 3,
    mcp_url: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    agent_type: str = "frontier",
    agent_name: str | None = None,
    log_dir: str | Path | None = DEFAULT_TRAJECTORIES_DIR,
    harbor_agent: str = "opencode",
    max_llm_turns: int = 45,
) -> dict[str, Any]:
    """Run `seeds x runs` multi-episode evaluation and compute reproducibility & failure stats."""
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        for run_idx in range(1, max(1, runs) + 1):
            ep = await run_single_frontier_episode(
                seed=seed,
                provider=provider,
                model=model,
                run_index=run_idx,
                mcp_url=mcp_url,
                base_url=base_url,
                api_key=api_key,
                agent_type=agent_type,
                agent_name=agent_name,
                log_dir=log_dir,
                harbor_agent=harbor_agent,
                max_llm_turns=max_llm_turns,
            )
            episodes.append(ep)

    all_scores = [ep["final_score"] for ep in episodes]
    mean_score = round(sum(all_scores) / max(1, len(all_scores)), 4)
    std_score = (
        round(statistics.pstdev(all_scores), 4) if len(all_scores) > 1 else 0.0
    )

    per_seed_summary: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        s_eps = [ep for ep in episodes if ep["seed"] == seed]
        s_scores = [ep["final_score"] for ep in s_eps]
        per_seed_summary[seed] = {
            "runs": len(s_eps),
            "mean_score": round(sum(s_scores) / max(1, len(s_scores)), 4),
            "std_score": round(statistics.pstdev(s_scores), 4) if len(s_scores) > 1 else 0.0,
            "min_score": min(s_scores) if s_scores else 0.0,
            "max_score": max(s_scores) if s_scores else 0.0,
            "scores": s_scores,
        }

    failure_category_counts: dict[str, int] = {}
    for ep in episodes:
        for cat in ep.get("primary_failure_categories", []):
            failure_category_counts[cat] = failure_category_counts.get(cat, 0) + 1

    resolved_model_name = episodes[0]["model"] if episodes else (model or "gemini-4-argon")
    return {
        "provider": provider,
        "model": resolved_model_name,
        "agent_type": agent_type,
        "seeds": list(seeds),
        "runs_per_seed": runs,
        "total_episodes": len(episodes),
        "mean_score": mean_score,
        "std_score": std_score,
        "min_score": min(all_scores) if all_scores else 0.0,
        "max_score": max(all_scores) if all_scores else 0.0,
        "full_batch_pass_rate": round(
            sum(1 for ep in episodes if ep["valid_order_count"] == 4)
            / max(1, len(episodes)),
            4,
        ),
        "finalization_rate": round(
            sum(1 for ep in episodes if ep["finalized"]) / max(1, len(episodes)),
            4,
        ),
        "failure_category_counts": failure_category_counts,
        "per_seed": per_seed_summary,
        "episodes": episodes,
    }


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.harbor_runner",
        description="Run CryoForge RL Frontier Agent Benchmark over MCP (gemini, openai, harbor).",
    )
    parser.add_argument(
        "--provider",
        choices=("gemini", "openai", "harbor"),
        default=os.environ.get("CRYOFORGE_FRONTIER_PROVIDER", "gemini"),
        help="Evaluation backend (`gemini`, `openai`, or `harbor`).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID (default: gemini-4-argon for gemini, gpt-4.1 for openai).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=list(DEFAULT_EVAL_SEEDS),
        help="Scenario seeds to evaluate (default: S01..S05).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Independent evaluation runs per seed for reproducibility (default: 3).",
    )
    parser.add_argument(
        "--mcp-url",
        default=None,
        help="Optional SSE MCP endpoint URL (e.g. http://127.0.0.1:8090/sse).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible API base URL (OpenRouter, vLLM, Ollama, etc.).",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to external .env file with API keys (loaded at runtime without copying into repo).",
    )
    parser.add_argument(
        "--agent-type",
        default="frontier",
        help="Trajectory agent_type tag (`frontier`, `small_before`, `small_after`).",
    )
    parser.add_argument(
        "--harbor-agent",
        default="opencode",
        help="Harbor CLI agent harness name when --provider=harbor (default: opencode).",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_TRAJECTORIES_DIR),
        help="Directory to persist TrajectoryLog JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the benchmark summary JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_cli(argv)
    load_env_file(args.env_file, override=True)

    summary = asyncio.run(
        run_frontier_benchmark(
            provider=args.provider,
            model=args.model,
            seeds=tuple(args.seeds),
            runs=args.runs,
            mcp_url=args.mcp_url,
            base_url=args.base_url,
            agent_type=args.agent_type,
            log_dir=args.log_dir,
            harbor_agent=args.harbor_agent,
        )
    )
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
