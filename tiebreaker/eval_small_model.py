"""CryoForge RL Tiebreaker Evaluator — Small-Model BEFORE vs AFTER on Unseen Seeds S06-S08.

Evaluates a small model (default: open-weight 4B-active `gemma-4-26b-a4b-it` via
`google-genai` v3.3.0 `client.interactions.create`, also supporting `gemini-3.5-flash-lite`,
OpenAI-compatible vLLM/Ollama/LoRA endpoints for `Qwen/Qwen2.5-3B-Instruct`, and local
PEFT LoRA checkpoints) strictly on held-out unseen seeds `S06, S07, S08` over real MCP
(`stdio` or `sse`).

Runs two fair conditions per held-out seed:
1. `BEFORE` (`agent_type="small_before"`):
   Standard task instruction (`harbor_task/instruction.md`) + all 10 MCP tools.
2. `AFTER` (`agent_type="small_after"`):
   Uses either a fine-tuned LoRA small-model endpoint (`--after-model` / `--after-base-url`
   / `--after-lora-path`) OR the `S01`–`S05` trajectory-distilled experience policy
   (`runs/tiebreaker/experience_playbook.json`, verified zero `S06`–`S08` leakage).

Writes reproducible results to `runs/tiebreaker/tiebreaker_summary.json` and per-episode
`TrajectoryLog` JSON files to `runs/trajectories/`.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from env_core.contracts import (
    EVAL_SEEDS,
    MAX_EPISODE_STEPS,
    TIEBREAKER_SEEDS,
    TrajectoryLog,
)
from tiebreaker import (
    ALLOWED_TRAIN_SEEDS,
    FORBIDDEN_HOLDOUT_SEEDS,
    REPO_ROOT,
    TIEBREAKER_RUNS_DIR,
    TRAJECTORIES_DIR,
)
from tiebreaker.collect_trajectories import (
    DEFAULT_SYSTEM_PROMPT,
    load_task_instruction,
    verify_zero_holdout_leakage_in_text,
)
from tiebreaker.train_sft import compile_experience_policy


# ---------------------------------------------------------------------------
# Environment File Loader (Never Logs or Stores Secrets)
# ---------------------------------------------------------------------------


def load_env_file(env_file: Path | None) -> None:
    """Load key-value pairs from `--env-file` (or `.env` if present) into `os.environ`."""
    candidates = [env_file] if env_file is not None else [REPO_ROOT / ".env"]
    for path in candidates:
        if path is None or not path.exists():
            continue
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
            val = val.strip().strip("'").strip('"')
            if key and val and not os.environ.get(key):
                os.environ[key] = val


def detect_provider(model_id: str, explicit_provider: str, base_url: str | None) -> str:
    if explicit_provider != "auto":
        return explicit_provider
    if base_url:
        return "openai"
    lower = model_id.lower()
    if lower.startswith("gemini-") or lower.startswith("gemma-"):
        return "gemini"
    return "openai"


# ---------------------------------------------------------------------------
# Schema Sanitization & Resilient Multi-Format JSON Tool-Call Fallback Parser
# ---------------------------------------------------------------------------


def _resolve_json_schema_refs(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            if ref_name in defs:
                return _resolve_json_schema_refs(copy.deepcopy(defs[ref_name]), defs)
        cleaned: dict[str, Any] = {}
        for k, v in node.items():
            if k in ("$defs", "definitions", "$schema", "title", "additionalProperties"):
                continue
            # Flatten nullable anyOf: [{"type": "string"}, {"type": "null"}] -> {"type": "string"}
            if k == "anyOf" and isinstance(v, list):
                non_null = [item for item in v if not (isinstance(item, dict) and item.get("type") == "null")]
                if len(non_null) == 1 and isinstance(non_null[0], dict):
                    resolved = _resolve_json_schema_refs(non_null[0], defs)
                    if isinstance(resolved, dict):
                        cleaned.update(resolved)
                        continue
            cleaned[k] = _resolve_json_schema_refs(v, defs)
        return cleaned
    if isinstance(node, list):
        return [_resolve_json_schema_refs(item, defs) for item in node]
    return node


def sanitize_schema_for_gemini(input_schema: dict[str, Any] | None) -> dict[str, Any]:
    """Convert MCP `inputSchema` into a clean JSON Schema subset for `google-genai` Interactions."""
    if not isinstance(input_schema, dict):
        return {"type": "object", "properties": {}}
    defs = input_schema.get("$defs") or input_schema.get("definitions") or {}
    cleaned = _resolve_json_schema_refs(input_schema, defs if isinstance(defs, dict) else {})
    if not isinstance(cleaned, dict):
        return {"type": "object", "properties": {}}
    cleaned.setdefault("type", "object")
    cleaned.setdefault("properties", {})
    return cleaned


def _normalize_candidate_tool_obj(obj: Any, valid_tool_names: set[str]) -> list[dict[str, Any]]:
    """Extract normalized `[{"id": ..., "name": ..., "arguments": {...}}]` from parsed JSON."""
    extracted: list[dict[str, Any]] = []
    if isinstance(obj, list):
        for item in obj:
            extracted.extend(_normalize_candidate_tool_obj(item, valid_tool_names))
        return extracted

    if not isinstance(obj, dict):
        return extracted

    if isinstance(obj.get("tool_calls"), list):
        for tc in obj["tool_calls"]:
            extracted.extend(_normalize_candidate_tool_obj(tc, valid_tool_names))
        return extracted

    if isinstance(obj.get("function"), dict):
        fn = obj["function"]
        name = fn.get("name")
        raw_args = fn.get("arguments", {})
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args if isinstance(raw_args, dict) else {})
        if isinstance(name, str) and name in valid_tool_names:
            extracted.append({"name": name, "arguments": args})
        return extracted

    name = obj.get("name") or obj.get("tool") or obj.get("tool_name") or obj.get("function")
    if isinstance(name, str) and name in valid_tool_names:
        raw_args = obj.get("arguments")
        if raw_args is None:
            raw_args = obj.get("parameters") if obj.get("parameters") is not None else obj.get("args", {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        extracted.append({"name": name, "arguments": args})
    return extracted


def extract_text_tool_calls(text: str | None, valid_tool_names: set[str]) -> list[dict[str, Any]]:
    """Fallback parser when an open-weight small model emits JSON tool calls inside plain text."""
    if not text:
        return []
    calls: list[dict[str, Any]] = []

    # 1. Explicit <tool_call>...</tool_call> blocks (Qwen2.5 / Hermes / Gemma convention)
    for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL):
        try:
            calls.extend(_normalize_candidate_tool_obj(json.loads(match.group(1)), valid_tool_names))
        except json.JSONDecodeError:
            continue
    if calls:
        return [
            {"id": f"txt_call_{i:02d}", "name": c["name"], "arguments": c["arguments"]}
            for i, c in enumerate(calls, start=1)
        ]

    # 2. Fenced ```json ... ``` or ```tool_call ... ``` code blocks
    for match in re.finditer(r"```(?:json|tool_call|tool)?\s*([\{\[].*?[\}\]])\s*```", text, flags=re.DOTALL):
        try:
            calls.extend(_normalize_candidate_tool_obj(json.loads(match.group(1)), valid_tool_names))
        except json.JSONDecodeError:
            continue
    if calls:
        return [
            {"id": f"txt_call_{i:02d}", "name": c["name"], "arguments": c["arguments"]}
            for i, c in enumerate(calls, start=1)
        ]

    # 3. Balanced JSON object scan anywhere in output_text
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        brace = text.find("{", idx)
        if brace == -1:
            break
        try:
            parsed, end_idx = decoder.raw_decode(text, brace)
            found = _normalize_candidate_tool_obj(parsed, valid_tool_names)
            if found:
                calls.extend(found)
                idx = end_idx
                continue
        except json.JSONDecodeError:
            pass
        idx = brace + 1

    return [
        {"id": f"txt_call_{i:02d}", "name": c["name"], "arguments": c["arguments"]}
        for i, c in enumerate(calls, start=1)
    ]


def build_text_tool_protocol_instructions(mcp_tools: list[dict[str, Any]]) -> str:
    """Explicit JSON tool-calling contract used when an endpoint lacks native `tools=[...]`."""
    compact_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        }
        for t in mcp_tools
    ]
    return (
        "\n\n### TOOL CALLING FORMAT (MANDATORY)\n"
        "You have access to the following 10 MCP tools:\n"
        f"{json.dumps(compact_tools, ensure_ascii=False)}\n\n"
        "To call one or more tools on your turn, output brief engineering reasoning followed by a "
        "JSON block in this exact format:\n"
        "```json\n"
        '{"tool_calls": [{"name": "get_episode_brief", "arguments": {}}]}\n'
        "```\n"
        "Wait for the tool results after each turn. Always finish by calling `finalize_commissioning`."
    )


# ---------------------------------------------------------------------------
# MCP Session Helpers (`stdio` & `sse`) + Deterministic Trajectory Export
# ---------------------------------------------------------------------------


def _detect_stdio_server_cli_args(
    *,
    seed: str,
    run_id: str,
    agent_name: str,
    agent_type: str,
    model_id: str,
    trajectories_dir: Path,
) -> list[str]:
    """Inspect `mcp_server/server.py` flags dynamically so CLI args match 100% alongside env vars."""
    server_py = REPO_ROOT / "mcp_server" / "server.py"
    args: list[str] = ["-m", "mcp_server.server"]
    if not server_py.exists():
        return args
    src = server_py.read_text(encoding="utf-8", errors="ignore")
    flag_pairs = [
        ("--seed", seed),
        ("--run-id", run_id),
        ("--agent-name", agent_name),
        ("--agent-type", agent_type),
        ("--model-id", model_id),
        ("--log-dir", str(trajectories_dir)),
    ]
    for flag, val in flag_pairs:
        if f'"{flag}"' in src or f"'{flag}'" in src:
            args.extend([flag, val])
    if ('"--transport"' in src or "'--transport'" in src) and "stdio" in src:
        args.extend(["--transport", "stdio"])
    return args


def _parse_mcp_call_result(call_result: Any) -> dict[str, Any]:
    """Extract JSON dictionary observation from an MCP `CallToolResult` (v2 & v1 safe)."""
    structured = getattr(call_result, "structured_content", None)
    if structured is None:
        structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured

    is_err = bool(
        getattr(call_result, "is_error", False)
        or getattr(call_result, "isError", False)
    )
    content_blocks = getattr(call_result, "content", None) or []
    texts: list[str] = []
    for block in content_blocks:
        txt = getattr(block, "text", None)
        if isinstance(txt, str) and txt.strip():
            texts.append(txt.strip())

    for txt in texts:
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, dict):
                return parsed
            return {"ok": not is_err, "result": parsed}
        except json.JSONDecodeError:
            continue

    joined = "\n".join(texts)
    return {"ok": not is_err, "error": joined if is_err else None, "raw_text": joined}


def _persist_or_load_trajectory_log(
    *,
    seed: str,
    run_id: str,
    agent_name: str,
    agent_type: str,
    model_id: str,
    trajectories_dir: Path,
    recorded_calls: list[tuple[str, dict[str, Any]]],
) -> tuple[TrajectoryLog, Path]:
    """Load server-exported `TrajectoryLog` or deterministically reconstruct via `CryoForgeEngine`."""
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    target_path = trajectories_dir / f"{run_id}.json"

    if target_path.exists():
        try:
            raw = json.loads(target_path.read_text(encoding="utf-8"))
            raw["agent_name"] = agent_name
            raw["agent_type"] = agent_type
            raw["model_id"] = model_id
            tlog = TrajectoryLog.model_validate(raw)
            target_path.write_text(tlog.model_dump_json(indent=2) + "\n", encoding="utf-8")
            return tlog, target_path
        except Exception:
            pass

    # Deterministic replay through CryoForgeEngine to obtain exact server-side VerificationReport
    try:
        from env_core import CryoForgeEngine  # type: ignore[attr-defined]
    except ImportError:
        from env_core.engine import CryoForgeEngine  # type: ignore[no-redef]

    engine = CryoForgeEngine(
        seed=seed,
        run_id=run_id,
        agent_name=agent_name,
        agent_type=agent_type,
        model_id=model_id,
        log_dir=trajectories_dir,
    )
    for tool_name, tool_args in recorded_calls:
        method = getattr(engine, tool_name, None)
        if callable(method):
            try:
                method(**tool_args)
            except TypeError:
                # Ignore malformed hallucinated kwargs if MCP server had already rejected them
                pass

    tlog = engine.export_trajectory_log()
    if not target_path.exists():
        target_path.write_text(tlog.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return tlog, target_path


# ---------------------------------------------------------------------------
# Model Drivers: Gemini Interactions v3.3.0, OpenAI/vLLM, and Local PEFT LoRA
# ---------------------------------------------------------------------------


async def _run_gemini_interactions_loop(
    *,
    session: ClientSession,
    mcp_tools: list[dict[str, Any]],
    model_id: str,
    system_prompt: str,
    user_instruction: str,
    max_turns: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Drive `gemma-4-26b-a4b-it` or `gemini-3.5-flash-lite` via `google-genai` v3.3.0 `interactions.create`."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Pass `--env-file .env` or export GEMINI_API_KEY."
        )
    client = genai.Client(api_key=api_key, http_options={"timeout": 90_000})

    valid_tool_names = {t["name"] for t in mcp_tools}
    gemini_tools = [
        {
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": sanitize_schema_for_gemini(t["parameters"]),
        }
        for t in mcp_tools
    ]

    is_gemma = model_id.lower().startswith("gemma-")
    use_stateless_steps = is_gemma
    use_native_tools = True
    allow_system_instruction = True
    prev_interaction_id: str | None = None
    recorded_calls: list[tuple[str, dict[str, Any]]] = []
    finalized = False
    empty_turn_nudges = 0

    history_steps: list[dict[str, Any]] = [
        {"type": "user_input", "content": [{"type": "text", "text": user_instruction}]}
    ]
    next_input: Any = user_instruction

    async def _call_interactions_with_fallback(turn_input: Any) -> Any:
        nonlocal use_stateless_steps, use_native_tools, allow_system_instruction, prev_interaction_id
        backoff_s = 1.5
        for attempt in range(5):
            effective_input: Any = history_steps if use_stateless_steps else turn_input
            kwargs: dict[str, Any] = {
                "model": model_id,
                "input": effective_input,
            }
            if not use_stateless_steps and prev_interaction_id is not None:
                kwargs["previous_interaction_id"] = prev_interaction_id
            if allow_system_instruction:
                kwargs["system_instruction"] = system_prompt
            elif prev_interaction_id is None and isinstance(effective_input, str):
                kwargs["input"] = f"{system_prompt}\n\n{effective_input}"

            if use_native_tools:
                kwargs["tools"] = gemini_tools
            elif prev_interaction_id is None and isinstance(kwargs["input"], str):
                kwargs["input"] = kwargs["input"] + build_text_tool_protocol_instructions(mcp_tools)

            if not is_gemma and model_id.lower().startswith(("gemini-3.5", "gemini-3.8", "gemini-4")):
                kwargs["generation_config"] = {
                    "max_output_tokens": 8192,
                    "thinking_level": "low",
                    "temperature": 0.0,
                }
            elif is_gemma:
                kwargs["generation_config"] = {"max_output_tokens": 16384, "temperature": 0.0}
            else:
                kwargs["generation_config"] = {"max_output_tokens": 4096, "temperature": 0.0}

            try:
                return await asyncio.to_thread(client.interactions.create, **kwargs)
            except Exception as exc:
                err_str = str(exc)
                err_lower = err_str.lower()

                # Automatic fallback 0: server-side previous_interaction_id empty-text replay bug -> switch to stateless steps
                if not use_stateless_steps and "missing text in content of type text" in err_lower:
                    use_stateless_steps = True
                    prev_interaction_id = None
                    continue

                # Automatic fallback 1: Gemma / open-weight endpoint rejects system_instruction
                if allow_system_instruction and (
                    "developer instruction" in err_lower or "system_instruction" in err_lower
                ):
                    allow_system_instruction = False
                    continue

                # Automatic fallback 2: Gemma / open-weight endpoint rejects native tools=[...]
                if use_native_tools and any(
                    tok in err_lower
                    for tok in ("function calling is not enabled", "tool", "function_declarations", "unsupported")
                ) and ("400" in err_lower or "invalid_argument" in err_lower):
                    use_native_tools = False
                    prev_interaction_id = None
                    continue

                # Transient rate-limit / server retry
                if any(code in err_lower for code in ("429", "503", "resource_exhausted", "unavailable", "timeout")) and attempt < 4:
                    await asyncio.sleep(backoff_s)
                    backoff_s *= 2.0
                    continue
                raise

        raise RuntimeError(f"Exceeded retry budget calling Gemini Interactions API for {model_id}")

    for _turn in range(max_turns):
        if finalized or len(recorded_calls) >= MAX_EPISODE_STEPS:
            break

        interaction = await _call_interactions_with_fallback(next_input)
        prev_interaction_id = getattr(interaction, "id", None)

        native_calls: list[dict[str, Any]] = []
        for idx, step in enumerate(getattr(interaction, "steps", None) or []):
            if getattr(step, "type", None) == "function_call":
                raw_args = getattr(step, "arguments", None) or {}
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed_args = {}
                elif isinstance(raw_args, dict):
                    parsed_args = dict(raw_args)
                else:
                    parsed_args = {}
                native_calls.append(
                    {
                        "id": str(getattr(step, "id", None) or f"call_{len(recorded_calls)+idx+1}"),
                        "name": str(getattr(step, "name", None) or ""),
                        "arguments": parsed_args,
                    }
                )

        out_text = (getattr(interaction, "output_text", None) or "").strip()

        if native_calls:
            if out_text:
                history_steps.append(
                    {"type": "model_output", "content": [{"type": "text", "text": out_text}]}
                )
            fn_result_steps: list[dict[str, Any]] = []
            for fc in native_calls:
                if len(recorded_calls) >= MAX_EPISODE_STEPS or finalized:
                    break
                tname = fc["name"]
                targs = fc["arguments"]
                history_steps.append(
                    {
                        "type": "function_call",
                        "id": fc["id"],
                        "name": tname,
                        "arguments": targs,
                    }
                )
                recorded_calls.append((tname, targs))
                mcp_res = await session.call_tool(tname, targs)
                obs = _parse_mcp_call_result(mcp_res)
                if (
                    obs.get("finalized") is True
                    or obs.get("error") in ("MAX_STEPS_EXCEEDED", "EPISODE_ALREADY_FINALIZED")
                ):
                    finalized = True
                res_step = {
                    "type": "function_result",
                    "name": tname,
                    "call_id": fc["id"],
                    "result": obs,
                }
                fn_result_steps.append(res_step)
                history_steps.append(res_step)
            next_input = fn_result_steps
            continue

        # Fallback: check whether model emitted JSON / <tool_call> in output_text
        parsed_text_calls = extract_text_tool_calls(out_text, valid_tool_names)
        if parsed_text_calls:
            if out_text:
                history_steps.append(
                    {"type": "model_output", "content": [{"type": "text", "text": out_text}]}
                )
            obs_lines: list[str] = []
            for fc in parsed_text_calls:
                if len(recorded_calls) >= MAX_EPISODE_STEPS or finalized:
                    break
                tname = fc["name"]
                targs = fc["arguments"]
                recorded_calls.append((tname, targs))
                mcp_res = await session.call_tool(tname, targs)
                obs = _parse_mcp_call_result(mcp_res)
                if (
                    obs.get("finalized") is True
                    or obs.get("error") in ("MAX_STEPS_EXCEEDED", "EPISODE_ALREADY_FINALIZED")
                ):
                    finalized = True
                obs_lines.append(f"Tool `{tname}` observation:\n{json.dumps(obs, ensure_ascii=False)}")
            nudge_msg = "\n\n".join(obs_lines) + "\n\nProceed with the next MCP tool call."
            history_steps.append(
                {"type": "user_input", "content": [{"type": "text", "text": nudge_msg}]}
            )
            next_input = nudge_msg
            continue

        if "FINAL_DONE" in out_text or empty_turn_nudges >= 2:
            break
        if out_text:
            history_steps.append(
                {"type": "model_output", "content": [{"type": "text", "text": out_text}]}
            )
        empty_turn_nudges += 1
        if use_native_tools:
            next_input = (
                "Continue calling the MCP tools to book all 4 orders, assign their routes, "
                "and call finalize_commissioning(). If completely finished, reply FINAL_DONE."
            )
        else:
            next_input = (
                "Output your next MCP tool call inside a ```json {\"tool_calls\": [...]} ``` block, "
                "or call `finalize_commissioning` when ready."
            )
        history_steps.append(
            {"type": "user_input", "content": [{"type": "text", "text": next_input}]}
        )

    return recorded_calls


async def _run_openai_compatible_loop(
    *,
    session: ClientSession,
    mcp_tools: list[dict[str, Any]],
    model_id: str,
    base_url: str | None,
    system_prompt: str,
    user_instruction: str,
    max_turns: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Drive `Qwen/Qwen2.5-3B-Instruct` or fine-tuned LoRA on OpenAI-compatible vLLM/Ollama."""
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY") or ("local-vllm" if base_url else "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY (or `--base-url` for local vLLM/Ollama) is required for provider=openai.")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    valid_tool_names = {t["name"] for t in mcp_tools}
    oai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in mcp_tools
    ]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instruction},
    ]
    recorded_calls: list[tuple[str, dict[str, Any]]] = []
    finalized = False
    use_tools_param = True
    empty_nudges = 0

    for _turn in range(max_turns):
        if finalized or len(recorded_calls) >= MAX_EPISODE_STEPS:
            break

        req_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        if use_tools_param:
            req_kwargs["tools"] = oai_tools

        try:
            resp = await client.chat.completions.create(**req_kwargs)
        except Exception as exc:
            if use_tools_param and ("tool" in str(exc).lower() or "400" in str(exc).lower()):
                use_tools_param = False
                messages[0]["content"] = system_prompt + build_text_tool_protocol_instructions(mcp_tools)
                continue
            raise

        msg = resp.choices[0].message
        if msg.tool_calls:
            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            messages.append(assistant_entry)
            for tc in msg.tool_calls:
                if len(recorded_calls) >= MAX_EPISODE_STEPS or finalized:
                    break
                tname = tc.function.name
                try:
                    targs = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    targs = {}
                recorded_calls.append((tname, targs))
                mcp_res = await session.call_tool(tname, targs)
                obs = _parse_mcp_call_result(mcp_res)
                if tname == "finalize_commissioning" and obs.get("ok") is True:
                    finalized = True
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tname,
                        "content": json.dumps(obs, ensure_ascii=False),
                    }
                )
            continue

        # Fallback: parse <tool_call>...</tool_call> or JSON blocks emitted in content
        text_content = msg.content or ""
        parsed_calls = extract_text_tool_calls(text_content, valid_tool_names)
        messages.append({"role": "assistant", "content": text_content})
        if parsed_calls:
            feedback: list[str] = []
            for fc in parsed_calls:
                if len(recorded_calls) >= MAX_EPISODE_STEPS or finalized:
                    break
                tname = fc["name"]
                targs = fc["arguments"]
                recorded_calls.append((tname, targs))
                mcp_res = await session.call_tool(tname, targs)
                obs = _parse_mcp_call_result(mcp_res)
                if tname == "finalize_commissioning" and obs.get("ok") is True:
                    finalized = True
                feedback.append(f"<tool_response>\n{json.dumps({'name': tname, 'observation': obs}, ensure_ascii=False)}\n</tool_response>")
            messages.append({"role": "user", "content": "\n".join(feedback)})
            continue

        if "FINAL_DONE" in text_content or empty_nudges >= 2:
            break
        empty_nudges += 1
        messages.append(
            {
                "role": "user",
                "content": "Continue executing MCP tools to commission all 4 orders, then call finalize_commissioning().",
            }
        )

    return recorded_calls


class _LocalPeftModelCache:
    """Lazy singleton loader for local HuggingFace + PEFT LoRA checkpoint inference."""

    _loaded: dict[tuple[str, str], tuple[Any, Any]] = {}

    @classmethod
    def get(cls, base_model: str, adapter_path: Path) -> tuple[Any, Any]:
        key = (base_model, str(adapter_path.resolve()))
        if key not in cls._loaded:
            import torch  # type: ignore[import-not-found]
            from peft import PeftModel  # type: ignore[import-not-found]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

            tok = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            peft_model = PeftModel.from_pretrained(base, str(adapter_path))
            peft_model.eval()
            cls._loaded[key] = (tok, peft_model)
        return cls._loaded[key]


async def _run_local_peft_loop(
    *,
    session: ClientSession,
    mcp_tools: list[dict[str, Any]],
    base_model: str,
    adapter_path: Path,
    system_prompt: str,
    user_instruction: str,
    max_turns: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Run local fine-tuned PEFT LoRA adapter (`cryoforge-qwen25-3b-lora`) over real MCP."""
    import torch  # type: ignore[import-not-found]

    tokenizer, model = _LocalPeftModelCache.get(base_model, adapter_path)
    valid_tool_names = {t["name"] for t in mcp_tools}
    oai_tools = [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in mcp_tools
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instruction},
    ]
    recorded_calls: list[tuple[str, dict[str, Any]]] = []
    finalized = False

    for _turn in range(max_turns):
        if finalized or len(recorded_calls) >= MAX_EPISODE_STEPS:
            break
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=oai_tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        parsed = extract_text_tool_calls(gen_text, valid_tool_names)
        if not parsed:
            break
        tc_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": fc["id"], "type": "function", "function": {"name": fc["name"], "arguments": fc["arguments"]}}
                for fc in parsed
            ],
        }
        messages.append(tc_msg)
        for fc in parsed:
            if len(recorded_calls) >= MAX_EPISODE_STEPS or finalized:
                break
            recorded_calls.append((fc["name"], fc["arguments"]))
            mcp_res = await session.call_tool(fc["name"], fc["arguments"])
            obs = _parse_mcp_call_result(mcp_res)
            if fc["name"] == "finalize_commissioning" and obs.get("ok") is True:
                finalized = True
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": fc["id"],
                    "name": fc["name"],
                    "content": json.dumps(obs, ensure_ascii=False),
                }
            )
    return recorded_calls


# ---------------------------------------------------------------------------
# Single MCP Episode Runner & Full S06-S08 BEFORE vs AFTER Evaluation
# ---------------------------------------------------------------------------


async def evaluate_single_episode_over_mcp(
    *,
    seed: str,
    condition: str,
    provider: str,
    model_id: str,
    base_url: str | None,
    lora_adapter_path: Path | None,
    system_prompt: str,
    user_instruction: str,
    transport: str,
    sse_url: str | None,
    max_turns: int,
    trajectories_dir: Path,
) -> tuple[TrajectoryLog, Path]:
    """Run one full episode (`small_before` or `small_after`) over real MCP (`stdio` or `sse`)."""
    agent_type = "small_before" if condition == "before" else "small_after"
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id).strip("-")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{agent_type}_{seed.lower()}_{safe_model}_{ts}"
    agent_name = f"tiebreaker_{condition}_{safe_model}"

    async def _execute_with_session(session: ClientSession) -> list[tuple[str, dict[str, Any]]]:
        await session.initialize()
        listed = await session.list_tools()
        mcp_tools = []
        for t in listed.tools:
            raw_schema = getattr(t, "input_schema", None)
            if raw_schema is None:
                raw_schema = getattr(t, "inputSchema", None)
            mcp_tools.append(
                {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": raw_schema if isinstance(raw_schema, dict) else {"type": "object", "properties": {}},
                }
            )
        if lora_adapter_path is not None:
            return await _run_local_peft_loop(
                session=session,
                mcp_tools=mcp_tools,
                base_model=model_id,
                adapter_path=lora_adapter_path,
                system_prompt=system_prompt,
                user_instruction=user_instruction,
                max_turns=max_turns,
            )
        if provider == "gemini":
            return await _run_gemini_interactions_loop(
                session=session,
                mcp_tools=mcp_tools,
                model_id=model_id,
                system_prompt=system_prompt,
                user_instruction=user_instruction,
                max_turns=max_turns,
            )
        return await _run_openai_compatible_loop(
            session=session,
            mcp_tools=mcp_tools,
            model_id=model_id,
            base_url=base_url,
            system_prompt=system_prompt,
            user_instruction=user_instruction,
            max_turns=max_turns,
        )

    if transport == "stdio":
        child_env = dict(os.environ)
        existing_pypath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{existing_pypath}" if existing_pypath else str(REPO_ROOT)
        )
        child_env.update(
            {
                "CRYOFORGE_SEED": seed,
                "SEED": seed,
                "CRYOFORGE_RUN_ID": run_id,
                "RUN_ID": run_id,
                "CRYOFORGE_AGENT_NAME": agent_name,
                "AGENT_NAME": agent_name,
                "CRYOFORGE_AGENT_TYPE": agent_type,
                "AGENT_TYPE": agent_type,
                "CRYOFORGE_MODEL_ID": model_id,
                "MODEL_ID": model_id,
                "CRYOFORGE_LOG_DIR": str(trajectories_dir),
                "LOG_DIR": str(trajectories_dir),
                "TRAJECTORY_LOG_DIR": str(trajectories_dir),
            }
        )
        server_args = _detect_stdio_server_cli_args(
            seed=seed,
            run_id=run_id,
            agent_name=agent_name,
            agent_type=agent_type,
            model_id=model_id,
            trajectories_dir=trajectories_dir,
        )
        params = StdioServerParameters(
            command=sys.executable,
            args=server_args,
            env=child_env,
            cwd=str(REPO_ROOT),
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                recorded_calls = await _execute_with_session(session)
    elif transport == "sse":
        if not sse_url:
            raise ValueError("--sse-url is required when --transport=sse.")
        async with sse_client(sse_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                recorded_calls = await _execute_with_session(session)
    else:
        raise ValueError(f"Unsupported MCP transport: {transport!r}")

    return _persist_or_load_trajectory_log(
        seed=seed,
        run_id=run_id,
        agent_name=agent_name,
        agent_type=agent_type,
        model_id=model_id,
        trajectories_dir=trajectories_dir,
        recorded_calls=recorded_calls,
    )


def _summarize_episode(tlog: TrajectoryLog, path: Path) -> dict[str, Any]:
    v = tlog.verification
    return {
        "run_id": tlog.run_id,
        "trajectory_path": str(path),
        "agent_type": tlog.agent_type,
        "model_id": tlog.model_id,
        "final_score": v.final_score,
        "valid_order_count": v.valid_order_count,
        "total_order_count": v.total_order_count,
        "finalized": v.finalized,
        "batch_gate_cap_applied": v.batch_gate_cap_applied,
        "budget_exceeded": v.budget_exceeded,
        "permit_quota_exceeded": v.permit_quota_exceeded,
        "total_budget_eur": v.total_budget_eur,
        "total_spent_eur": v.total_spent_eur,
        "sim_calls_used": v.sim_calls_used,
        "sim_budget_total": v.sim_budget_total,
        "step_count": v.step_count,
        "disruption_triggered": v.disruption_triggered,
        "failure_modes": v.failure_modes,
        "primary_failure_categories": v.primary_failure_categories,
    }


def load_verified_experience_playbook(playbook_path: Path) -> dict[str, Any]:
    """Load `experience_playbook.json` (auto-compiling from `S01`–`S05` if absent) and verify zero leakage."""
    if not playbook_path.exists():
        compile_experience_policy(output_dir=playbook_path.parent)
    raw_text = playbook_path.read_text(encoding="utf-8")
    verify_zero_holdout_leakage_in_text(raw_text, str(playbook_path))
    playbook = json.loads(raw_text)
    for seed in playbook.get("train_seeds_included", []):
        if seed in FORBIDDEN_HOLDOUT_SEEDS or seed not in ALLOWED_TRAIN_SEEDS:
            raise ValueError(f"Playbook seed provenance check failed on seed={seed!r}!")
    return playbook


async def run_tiebreaker_evaluation(
    *,
    model: str = "gemini-3.5-flash-lite",
    provider: str = "auto",
    base_url: str | None = None,
    after_model: str | None = None,
    after_base_url: str | None = None,
    after_lora_path: Path | None = None,
    playbook_path: Path = TIEBREAKER_RUNS_DIR / "experience_playbook.json",
    seeds: list[str] | None = None,
    transport: str = "stdio",
    sse_url: str | None = None,
    max_turns: int = 36,
    output_path: Path = TIEBREAKER_RUNS_DIR / "tiebreaker_summary.json",
    trajectories_dir: Path = TRAJECTORIES_DIR,
    only: str = "both",
    allow_dev_seeds: bool = False,
) -> dict[str, Any]:
    """Evaluate `BEFORE` vs `AFTER` small model across held-out unseen seeds `S06, S07, S08`."""
    eval_seeds = list(seeds) if seeds else list(TIEBREAKER_SEEDS)
    if not allow_dev_seeds:
        overlap = set(eval_seeds) & set(EVAL_SEEDS)
        if overlap:
            raise ValueError(
                f"Tiebreaker evaluation MUST run on held-out seeds {list(TIEBREAKER_SEEDS)} only; "
                f"got training seeds {sorted(overlap)}."
            )

    base_provider = detect_provider(model, provider, base_url)
    eff_after_model = after_model or model
    eff_after_base_url = after_base_url or base_url
    after_provider = detect_provider(eff_after_model, provider, eff_after_base_url)

    task_instruction = load_task_instruction()
    using_finetuned_weights = bool(after_model or after_base_url or after_lora_path)
    playbook = load_verified_experience_playbook(playbook_path)
    compiled_policy = str(playbook["compiled_policy_prompt"])
    verify_zero_holdout_leakage_in_text(compiled_policy, "compiled_policy_prompt")

    before_sys = DEFAULT_SYSTEM_PROMPT
    if using_finetuned_weights and after_lora_path is not None:
        after_sys = DEFAULT_SYSTEM_PROMPT
        after_mode_desc = f"qlora_peft_adapter:{after_lora_path.name}"
    elif using_finetuned_weights:
        after_sys = DEFAULT_SYSTEM_PROMPT
        after_mode_desc = f"finetuned_lora_endpoint:{eff_after_model}"
    else:
        after_sys = f"{DEFAULT_SYSTEM_PROMPT}\n\n{compiled_policy}"
        after_mode_desc = f"s01_s05_experience_distilled:{playbook_path.name}"

    per_seed_entries: list[dict[str, Any]] = []
    per_seed_map: dict[str, dict[str, Any]] = {}
    before_cat_counts: Counter[str] = Counter()
    after_cat_counts: Counter[str] = Counter()
    before_mode_counts: Counter[str] = Counter()
    after_mode_counts: Counter[str] = Counter()

    sem = asyncio.Semaphore(4)

    async def _run_cond(seed: str, cond: str) -> tuple[str, str, TrajectoryLog, Path]:
        async with sem:
            is_before = cond == "before"
            tlog, tpath = await evaluate_single_episode_over_mcp(
                seed=seed,
                condition=cond,
                provider=base_provider if is_before else after_provider,
                model_id=model if is_before else eff_after_model,
                base_url=base_url if is_before else eff_after_base_url,
                lora_adapter_path=None if is_before else after_lora_path,
                system_prompt=before_sys if is_before else after_sys,
                user_instruction=task_instruction,
                transport=transport,
                sse_url=sse_url,
                max_turns=max_turns,
                trajectories_dir=trajectories_dir,
            )
            print(
                f"  [tiebreaker] {seed} {cond.upper():6s} -> score={tlog.verification.final_score:.4f} "
                f"(valid={tlog.verification.valid_order_count}/4, steps={tlog.verification.step_count})",
                flush=True,
            )
            return seed, cond, tlog, tpath

    tasks = []
    for seed in eval_seeds:
        if only in ("both", "before"):
            tasks.append(_run_cond(seed, "before"))
        if only in ("both", "after"):
            tasks.append(_run_cond(seed, "after"))

    completed_runs = await asyncio.gather(*tasks)
    by_seed_cond: dict[tuple[str, str], tuple[TrajectoryLog, Path]] = {
        (s, c): (tlog, tpath) for s, c, tlog, tpath in completed_runs
    }

    for seed in eval_seeds:
        seed_rec: dict[str, Any] = {"seed": seed}
        before_summary: dict[str, Any] | None = None
        after_summary: dict[str, Any] | None = None

        if (seed, "before") in by_seed_cond:
            b_log, b_path = by_seed_cond[(seed, "before")]
            before_summary = _summarize_episode(b_log, b_path)
            seed_rec["before"] = before_summary
            for c in b_log.verification.primary_failure_categories:
                before_cat_counts[c] += 1
            if not b_log.verification.finalized:
                before_cat_counts["UNFINALIZED"] += 1
            for m in b_log.verification.failure_modes:
                before_mode_counts[m] += 1

        if (seed, "after") in by_seed_cond:
            a_log, a_path = by_seed_cond[(seed, "after")]
            after_summary = _summarize_episode(a_log, a_path)
            seed_rec["after"] = after_summary
            for c in a_log.verification.primary_failure_categories:
                after_cat_counts[c] += 1
            if not a_log.verification.finalized:
                after_cat_counts["UNFINALIZED"] += 1
            for m in a_log.verification.failure_modes:
                after_mode_counts[m] += 1

        if before_summary is not None and after_summary is not None:
            seed_rec["score_delta"] = round(
                after_summary["final_score"] - before_summary["final_score"], 4
            )
            seed_rec["valid_orders_delta"] = (
                after_summary["valid_order_count"] - before_summary["valid_order_count"]
            )
            seed_rec["resolved_failure_categories"] = sorted(
                set(before_summary["primary_failure_categories"])
                - set(after_summary["primary_failure_categories"])
            )

        per_seed_entries.append(seed_rec)
        per_seed_map[seed] = seed_rec

    before_scores = [e["before"]["final_score"] for e in per_seed_entries if "before" in e]
    after_scores = [e["after"]["final_score"] for e in per_seed_entries if "after" in e]
    before_mean = round(sum(before_scores) / max(1, len(before_scores)), 4) if before_scores else 0.0
    after_mean = round(sum(after_scores) / max(1, len(after_scores)), 4) if after_scores else 0.0
    mean_delta = round(after_mean - before_mean, 4)
    rel_pct = round(100.0 * mean_delta / max(0.01, before_mean), 2) if before_scores and after_scores else 0.0

    all_cats = sorted(set(before_cat_counts) | set(after_cat_counts))
    category_deltas: dict[str, dict[str, int | float]] = {}
    for cat in all_cats:
        b_cnt = before_cat_counts.get(cat, 0)
        a_cnt = after_cat_counts.get(cat, 0)
        red = b_cnt - a_cnt
        category_deltas[cat] = {
            "before": b_cnt,
            "after": a_cnt,
            "reduction": red,
            "reduction_pct": round(100.0 * red / b_cnt, 1) if b_cnt > 0 else 0.0,
        }

    aggregate = {
        "before_mean_score": before_mean,
        "after_mean_score": after_mean,
        "mean_score_delta": mean_delta,
        "relative_improvement_pct": rel_pct,
        "before_valid_orders_total": sum(e["before"]["valid_order_count"] for e in per_seed_entries if "before" in e),
        "after_valid_orders_total": sum(e["after"]["valid_order_count"] for e in per_seed_entries if "after" in e),
        "total_orders_evaluated": 4 * len(eval_seeds),
        "before_full_batch_pass_seeds": sum(
            1 for e in per_seed_entries if "before" in e and e["before"]["valid_order_count"] == 4
        ),
        "after_full_batch_pass_seeds": sum(
            1 for e in per_seed_entries if "after" in e and e["after"]["valid_order_count"] == 4
        ),
        "before_mean_sim_calls": round(
            sum(e["before"]["sim_calls_used"] for e in per_seed_entries if "before" in e) / max(1, len(before_scores)),
            2,
        ) if before_scores else 0.0,
        "after_mean_sim_calls": round(
            sum(e["after"]["sim_calls_used"] for e in per_seed_entries if "after" in e) / max(1, len(after_scores)),
            2,
        ) if after_scores else 0.0,
        "improved_seed_count": sum(1 for e in per_seed_entries if e.get("score_delta", 0.0) > 0.0),
        "tiebreaker_passed": bool(after_mean > before_mean and len(after_scores) > 0),
    }

    summary_payload = {
        "benchmark": "CryoForge RL — Small-Model Experience Improvement Tiebreaker",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_seeds_used": list(EVAL_SEEDS),
        "holdout_eval_seeds": eval_seeds,
        "zero_holdout_leakage_verified": True,
        "mcp_transport": transport,
        "before_model": {
            "provider": base_provider,
            "model_id": model,
            "mode": "base_zero_shot",
            "agent_type": "small_before",
        },
        "after_model": {
            "provider": "local_peft" if after_lora_path else after_provider,
            "model_id": eff_after_model,
            "mode": after_mode_desc,
            "agent_type": "small_after",
            "experience_playbook_sha256": playbook.get("sft_dataset_sha256", ""),
        },
        "before_mean_score": before_mean,
        "after_mean_score": after_mean,
        "mean_score_delta": mean_delta,
        "aggregate": aggregate,
        "per_seed": per_seed_entries,
        "per_seed_map": per_seed_map,
        "failure_reduction_breakdown": {
            "before_failure_category_counts": dict(before_cat_counts),
            "after_failure_category_counts": dict(after_cat_counts),
            "category_deltas": category_deltas,
            "before_failure_mode_counts": dict(before_mode_counts),
            "after_failure_mode_counts": dict(after_mode_counts),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary_payload


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a small model BEFORE vs AFTER S01-S05 experience/SFT on held-out seeds S06-S08 over real MCP."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Small base model ID (default: CRYOFORGE_SMALL_MODEL or 'gemma-4-26b-a4b-it').",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["auto", "gemini", "openai"],
        default="auto",
        help="Model API provider ('gemini' uses google-genai v3.3.0 interactions.create; 'openai' uses OpenAI/vLLM).",
    )
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible base URL for base small model.")
    parser.add_argument("--after-model", type=str, default=None, help="Optional fine-tuned LoRA model ID for AFTER condition.")
    parser.add_argument("--after-base-url", type=str, default=None, help="Optional OpenAI-compatible base URL for fine-tuned LoRA endpoint.")
    parser.add_argument("--after-lora-path", type=Path, default=None, help="Optional local PEFT LoRA checkpoint path for direct GPU inference.")
    parser.add_argument(
        "--playbook",
        type=Path,
        default=TIEBREAKER_RUNS_DIR / "experience_playbook.json",
        help="Path to S01-S05 distilled experience playbook JSON.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=list(TIEBREAKER_SEEDS),
        help="Held-out evaluation seeds (default: S06 S07 S08).",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Real MCP wire transport ('stdio' spawns mcp_server.server; 'sse' connects to --sse-url).",
    )
    parser.add_argument("--sse-url", type=str, default=None, help="MCP SSE URL when --transport=sse.")
    parser.add_argument("--max-turns", type=int, default=38, help="Max LLM turns per episode.")
    parser.add_argument(
        "--output",
        type=Path,
        default=TIEBREAKER_RUNS_DIR / "tiebreaker_summary.json",
        help="Output JSON path for reproducible BEFORE vs AFTER comparison.",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=TRAJECTORIES_DIR,
        help="Directory where TrajectoryLog JSON files (small_before / small_after) are saved.",
    )
    parser.add_argument(
        "--only",
        choices=["both", "before", "after"],
        default="both",
        help="Run both conditions or only before/after.",
    )
    parser.add_argument("--env-file", type=Path, default=None, help="Path to .env file containing API credentials.")
    parser.add_argument("--allow-dev-seeds", action="store_true", help="Allow running on dev seeds for smoke debugging.")
    args = parser.parse_args(argv)

    load_env_file(args.env_file)
    resolved_model = args.model or os.environ.get("CRYOFORGE_SMALL_MODEL") or "gemini-3.5-flash-lite"
    resolved_base_url = args.base_url or (os.environ.get("OPENAI_BASE_URL") or None)

    summary = asyncio.run(
        run_tiebreaker_evaluation(
            model=resolved_model,
            provider=args.provider,
            base_url=resolved_base_url,
            after_model=args.after_model,
            after_base_url=args.after_base_url,
            after_lora_path=args.after_lora_path,
            playbook_path=args.playbook,
            seeds=list(args.seeds),
            transport=args.transport,
            sse_url=args.sse_url,
            max_turns=args.max_turns,
            output_path=args.output,
            trajectories_dir=args.trajectories_dir,
            only=args.only,
            allow_dev_seeds=args.allow_dev_seeds,
        )
    )
    print(json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
