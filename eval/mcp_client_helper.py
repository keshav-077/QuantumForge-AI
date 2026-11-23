"""Async MCP client helper for CryoForge RL (`stdio` and `sse` transports).

Provides a clean async context manager wrapping `mcp.ClientSession` over either
`mcp.client.stdio.stdio_client` (`python -m mcp_server.server --seed ...`) or
`mcp.client.sse.sse_client` (`--mcp-url`), normalizes tool outputs in `mcp>=2.0`
and `mcp 1.x` into plain Python `dict`s, and converts MCP tool schemas cleanly
for both Google GenAI Interactions API (`google-genai` v3.x) and OpenAI tool-calling.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
import urllib.parse
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_TRAJECTORIES_DIR: Path = PROJECT_ROOT / "runs" / "trajectories"


def parse_tool_result(result: Any) -> dict[str, Any]:
    """Normalize an MCP `CallToolResult` (v1.x or v2.x) into a Python `dict`.

    Handles:
    - `structured_content` (mcp 2.x) / `structuredContent` (mcp 1.x)
    - `content[i].text` containing serialized JSON objects or arrays
    - `is_error=True` plain-text `ToolError` messages from MCP argument validation
    """
    if isinstance(result, dict):
        return copy.deepcopy(result)

    is_error = bool(
        getattr(result, "is_error", getattr(result, "isError", False))
    )

    structured = getattr(result, "structured_content", None)
    if structured is None:
        structured = getattr(result, "structuredContent", None)

    if isinstance(structured, dict):
        payload = copy.deepcopy(structured)
        if (
            "result" in payload
            and len(payload) == 1
            and isinstance(payload["result"], dict)
        ):
            payload = copy.deepcopy(payload["result"])
        if is_error and "ok" not in payload:
            payload["ok"] = False
        return payload

    content_blocks = getattr(result, "content", None) or []
    text_fragments: list[str] = []

    for block in content_blocks:
        text_val: str | None = None
        if isinstance(block, dict):
            raw = block.get("text")
            if isinstance(raw, str):
                text_val = raw
        else:
            raw = getattr(block, "text", None)
            if isinstance(raw, str):
                text_val = raw

        if text_val is None:
            continue
        stripped = text_val.strip()
        if not stripped:
            continue
        text_fragments.append(stripped)
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                if is_error and "ok" not in parsed:
                    parsed["ok"] = False
                return parsed
            return {
                "ok": not is_error,
                "is_error": is_error,
                "data": parsed,
            }
        except json.JSONDecodeError:
            continue

    combined_text = "\n".join(text_fragments)
    if is_error:
        return {
            "ok": False,
            "error": "MCP_TOOL_ERROR",
            "is_error": True,
            "message": combined_text or "MCP tool execution returned an error.",
        }
    if combined_text:
        return {
            "ok": True,
            "message": combined_text,
        }
    return {"ok": not is_error}


def clean_json_schema_for_llm(
    schema: dict[str, Any] | None,
    *,
    is_required: bool = True,
) -> dict[str, Any]:
    """Sanitize Pydantic/MCP JSON Schema for universal Gemini 3.x & OpenAI tool calling.

    Strips redundant `title` metadata and simplifies optional nullable `anyOf` unions
    (`[{"type": T}, {"type": "null"}]`) so strict function-calling validators across
    Gemini Interactions API, OpenAI, OpenRouter, vLLM, and Ollama accept every schema.
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    cleaned: dict[str, Any] = {}

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        non_null_branches = [
            b
            for b in schema["anyOf"]
            if not (isinstance(b, dict) and b.get("type") == "null")
        ]
        has_null = len(non_null_branches) < len(schema["anyOf"])
        if len(non_null_branches) == 1 and isinstance(non_null_branches[0], dict):
            merged = dict(non_null_branches[0])
            for k in ("description", "default", "enum"):
                if k in schema and k not in merged:
                    merged[k] = schema[k]
            if has_null and not is_required and merged.get("default") is None:
                merged.pop("default", None)
            return clean_json_schema_for_llm(merged, is_required=is_required)

    for key, val in schema.items():
        if key == "title":
            continue
        if key == "default" and val is None and not is_required:
            continue
        if key == "properties" and isinstance(val, dict):
            req_set = set(schema.get("required") or [])
            cleaned["properties"] = {
                prop_name: clean_json_schema_for_llm(
                    prop_schema,
                    is_required=(prop_name in req_set),
                )
                for prop_name, prop_schema in val.items()
            }
        elif key == "items" and isinstance(val, dict):
            cleaned["items"] = clean_json_schema_for_llm(val, is_required=True)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(val, list):
            cleaned[key] = [
                clean_json_schema_for_llm(item, is_required=is_required)
                if isinstance(item, dict)
                else copy.deepcopy(item)
                for item in val
            ]
        else:
            cleaned[key] = copy.deepcopy(val)

    if "properties" in cleaned and "type" not in cleaned:
        cleaned["type"] = "object"
    return cleaned


def extract_mcp_tool_definition(tool: Any) -> dict[str, Any]:
    """Extract `{name, description, input_schema}` from an `mcp.types.Tool` (v1 or v2)."""
    if isinstance(tool, dict):
        name = str(tool["name"])
        description = str(tool.get("description") or "")
        raw_schema = (
            tool.get("input_schema")
            or tool.get("inputSchema")
            or {"type": "object", "properties": {}}
        )
    else:
        name = str(getattr(tool, "name"))
        description = str(getattr(tool, "description", "") or "")
        raw_schema = (
            getattr(tool, "input_schema", None)
            or getattr(tool, "inputSchema", None)
            or {"type": "object", "properties": {}}
        )

    cleaned_schema = clean_json_schema_for_llm(raw_schema, is_required=True)
    if "type" not in cleaned_schema:
        cleaned_schema["type"] = "object"
    if "properties" not in cleaned_schema:
        cleaned_schema["properties"] = {}

    return {
        "name": name,
        "description": description,
        "input_schema": cleaned_schema,
    }


def mcp_tools_to_gemini_declarations(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tools into `google-genai` v3.x `interactions.create(tools=...)` declarations."""
    declarations: list[dict[str, Any]] = []
    for raw_tool in tools:
        defn = extract_mcp_tool_definition(raw_tool)
        declarations.append(
            {
                "type": "function",
                "name": defn["name"],
                "description": defn["description"],
                "parameters": defn["input_schema"],
            }
        )
    return declarations


def mcp_tools_to_openai_declarations(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP tools into OpenAI Chat Completions `tools=[{"type": "function", ...}]`."""
    declarations: list[dict[str, Any]] = []
    for raw_tool in tools:
        defn = extract_mcp_tool_definition(raw_tool)
        declarations.append(
            {
                "type": "function",
                "function": {
                    "name": defn["name"],
                    "description": defn["description"],
                    "parameters": defn["input_schema"],
                },
            }
        )
    return declarations


class CryoForgeMCPClient:
    """High-level async client wrapping `mcp.ClientSession` over `stdio` or `sse`."""

    def __init__(
        self,
        session: ClientSession,
        *,
        seed: str = "S01",
        run_id: str | None = None,
        log_dir: Path | None = None,
        mcp_url: str | None = None,
        verifier_token: str | None = None,
    ) -> None:
        self.session = session
        self.seed = seed
        self.run_id = run_id
        self.log_dir = log_dir
        self.mcp_url = mcp_url
        self.verifier_token = verifier_token
        self.call_history: list[dict[str, Any]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        """List all tools registered on the connected MCP server."""
        resp = await self.session.list_tools()
        raw_tools = getattr(resp, "tools", []) or []
        return [extract_mcp_tool_definition(t) for t in raw_tools]

    async def get_gemini_tools(self) -> list[dict[str, Any]]:
        """Return MCP tools formatted for `google-genai` v3.x `client.interactions.create`."""
        resp = await self.session.list_tools()
        return mcp_tools_to_gemini_declarations(list(getattr(resp, "tools", []) or []))

    async def get_openai_tools(self) -> list[dict[str, Any]]:
        """Return MCP tools formatted for OpenAI Chat Completions `tools=`."""
        resp = await self.session.list_tools()
        return mcp_tools_to_openai_declarations(list(getattr(resp, "tools", []) or []))

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke an MCP tool by name and parse the response into a Python `dict`."""
        payload = arguments if arguments is not None else {}
        try:
            raw_result = await self.session.call_tool(name, payload)
            parsed = parse_tool_result(raw_result)
        except Exception as exc:
            parsed = {
                "ok": False,
                "error": "MCP_TRANSPORT_ERROR",
                "tool": name,
                "message": str(exc),
            }

        self.call_history.append(
            {
                "tool_name": name,
                "arguments": copy.deepcopy(payload) if isinstance(payload, dict) else payload,
                "response": copy.deepcopy(parsed),
            }
        )
        return parsed

    def fetch_sse_grade(self, timeout: float = 10.0) -> dict[str, Any] | None:
        """Fetch `/internal/grade` from an SSE sidecar server when `verifier_token` is configured."""
        if not self.mcp_url or not self.verifier_token:
            return None
        parsed = urllib.parse.urlparse(self.mcp_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        grade_url = f"{base_url}/internal/grade"
        resp = httpx.get(
            grade_url,
            headers={"X-Verifier-Token": self.verifier_token},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None


def load_persisted_trajectory(
    run_id: str | None,
    log_dir: str | Path | None,
    *,
    wait_seconds: float = 1.5,
) -> dict[str, Any] | None:
    """Read the persisted `TrajectoryLog` JSON written by `mcp_server.server` after run completion."""
    if not run_id or not log_dir:
        return None
    base = Path(log_dir)
    candidates = [
        base / f"{run_id}.json",
        base / "trajectories" / f"{run_id}.json",
    ]
    deadline = time.monotonic() + max(0.05, wait_seconds)
    while True:
        for candidate in candidates:
            if candidate.is_file():
                try:
                    raw = candidate.read_text(encoding="utf-8")
                    if raw.strip():
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            return parsed
                except (OSError, json.JSONDecodeError):
                    pass
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    return None


@asynccontextmanager
async def open_cryoforge_mcp_session(
    *,
    seed: str = "S01",
    mcp_url: str | None = None,
    run_id: str | None = None,
    agent_name: str = "agent",
    agent_type: str = "frontier",
    model_id: str = "unknown",
    log_dir: str | Path | None = DEFAULT_TRAJECTORIES_DIR,
    python_executable: str | None = None,
    verifier_token: str | None = None,
    extra_env: dict[str, str] | None = None,
    sse_headers: dict[str, Any] | None = None,
    sse_read_timeout: float = 300.0,
) -> AsyncIterator[CryoForgeMCPClient]:
    """Open an initialized `CryoForgeMCPClient` session over `stdio` or `sse`.

    - When `mcp_url` is `None`: spawns `python -m mcp_server.server --seed <seed> ...` via `stdio_client`.
    - When `mcp_url` is set: connects to the remote/sidecar SSE endpoint via `sse_client`.
    """
    effective_run_id = run_id or f"{agent_type}_{seed}_{uuid.uuid4().hex[:8]}"
    resolved_log_dir = Path(log_dir).resolve() if log_dir is not None else None
    if resolved_log_dir is not None:
        resolved_log_dir.mkdir(parents=True, exist_ok=True)

    effective_verifier_token = (
        verifier_token
        if verifier_token is not None
        else os.environ.get("CRYOFORGE_VERIFIER_TOKEN") or None
    )

    async with AsyncExitStack() as stack:
        if mcp_url:
            streams = await stack.enter_async_context(
                sse_client(
                    url=mcp_url,
                    headers=sse_headers,
                    timeout=10.0,
                    sse_read_timeout=sse_read_timeout,
                )
            )
            read_stream, write_stream = streams[0], streams[1]
        else:
            py_bin = python_executable or sys.executable
            cmd_args = [
                "-m",
                "mcp_server.server",
                "--transport",
                "stdio",
                "--seed",
                str(seed),
                "--run-id",
                effective_run_id,
                "--agent-name",
                str(agent_name),
                "--agent-type",
                str(agent_type),
                "--model-id",
                str(model_id),
            ]
            if resolved_log_dir is not None:
                cmd_args.extend(["--log-dir", str(resolved_log_dir)])

            child_env = dict(os.environ)
            existing_py_path = child_env.get("PYTHONPATH", "")
            root_str = str(PROJECT_ROOT)
            if existing_py_path:
                child_env["PYTHONPATH"] = f"{root_str}{os.pathsep}{existing_py_path}"
            else:
                child_env["PYTHONPATH"] = root_str
            child_env["PYTHONIOENCODING"] = "utf-8"
            if extra_env:
                child_env.update(extra_env)

            server_params = StdioServerParameters(
                command=py_bin,
                args=cmd_args,
                env=child_env,
                cwd=str(PROJECT_ROOT),
            )
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(server_params)
            )

        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        yield CryoForgeMCPClient(
            session,
            seed=seed,
            run_id=effective_run_id,
            log_dir=resolved_log_dir,
            mcp_url=mcp_url,
            verifier_token=effective_verifier_token,
        )
