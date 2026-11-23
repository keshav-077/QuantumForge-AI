"""Harbor-compatible CryoForge RL MCP server (`stdio` and isolated `sse` sidecar modes).

Supports both MCP Python SDK v2.x (`from mcp.server.mcpserver import MCPServer`)
and MCP Python SDK v1.x (`from mcp.server.fastmcp import FastMCP as MCPServer`),
enforces strict `StrictModel` input validation before delegating to `CryoForgeEngine`,
and exposes a token-protected `/internal/grade` endpoint in SSE mode for zero-trust
Harbor sidecar verification.
"""

from __future__ import annotations

import argparse
import hmac
import inspect
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - fallback for mcp 1.x images
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef]

from env_core.contracts import MAX_EPISODE_STEPS
from mcp_server.schemas import validate_tool_arguments

logger = logging.getLogger("cryoforge.mcp_server")

ALLOWED_AGENT_TYPES: frozenset[str] = frozenset(
    {
        "reference",
        "frontier",
        "do_nothing",
        "stop_early",
        "small_before",
        "small_after",
        "exploit",
    }
)

SERVER_INSTRUCTIONS: str = (
    "CryoForge RL Commissioning & Supply-Shock MCP Server. "
    "Inspect the episode brief, engineering handbook, all 4 customer orders, and transport routes; "
    "design compliant cryogenic vessels within simulation, manufacturing bay, transport clearance, "
    "escort permit, SLA, and factory budget constraints; adapt to any mid-episode disruptions; "
    "and call `finalize_commissioning()` when all 4 orders are ready."
)


def _normalize_agent_type(agent_type: str) -> str:
    cleaned = (agent_type or "frontier").strip()
    if cleaned in ALLOWED_AGENT_TYPES:
        return cleaned
    return "frontier"


def _instantiate_engine(
    *,
    seed: str,
    run_id: str | None,
    agent_name: str,
    agent_type: str,
    model_id: str,
    log_dir: str | Path | None,
) -> Any:
    from env_core.engine import CryoForgeEngine

    return CryoForgeEngine(
        seed=seed,
        run_id=run_id,
        agent_name=agent_name,
        agent_type=_normalize_agent_type(agent_type),
        model_id=model_id,
        log_dir=log_dir,
    )


def _persist_trajectory_safely(engine: Any, log_dir: str | Path | None) -> Path | None:
    """Ensure the authoritative `TrajectoryLog` JSON is flushed to disk when `log_dir` is set."""
    if not log_dir or not hasattr(engine, "export_trajectory_log"):
        return None
    try:
        traj = engine.export_trajectory_log()
        base = Path(log_dir)
        direct_candidate = base / f"{traj.run_id}.json"
        nested_candidate = base / "trajectories" / f"{traj.run_id}.json"

        if nested_candidate.exists():
            target = nested_candidate
        elif direct_candidate.exists():
            target = direct_candidate
        elif base.name == "trajectories":
            target = direct_candidate
        elif (base / "trajectories").is_dir():
            target = nested_candidate
        else:
            target = direct_candidate

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(".json.tmp")
        tmp_path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(target)
        return target
    except Exception as exc:  # pragma: no cover - defensive logging to stderr only
        logger.warning("Failed to persist trajectory log: %s", exc)
        return None


def _build_mcp_instance(name: str = "cryoforge") -> MCPServer:
    init_params = inspect.signature(MCPServer.__init__).parameters
    kwargs: dict[str, Any] = {}
    if "instructions" in init_params:
        kwargs["instructions"] = SERVER_INSTRUCTIONS
    return MCPServer(name, **kwargs)


def create_cryoforge_mcp_server(
    *,
    seed: str = "S01",
    run_id: str | None = None,
    agent_name: str = "agent",
    agent_type: str = "frontier",
    model_id: str = "unknown",
    log_dir: str | Path | None = None,
    engine: Any | None = None,
) -> tuple[MCPServer, Any]:
    """Create a configured `MCPServer` with all 10 CryoForge tools bound to a single episode engine."""
    active_engine = (
        engine
        if engine is not None
        else _instantiate_engine(
            seed=seed,
            run_id=run_id,
            agent_name=agent_name,
            agent_type=agent_type,
            model_id=model_id,
            log_dir=log_dir,
        )
    )
    mcp = _build_mcp_instance("cryoforge")
    lock = threading.RLock()
    validation_rejections = 0
    max_validation_rejections = MAX_EPISODE_STEPS * 3

    def _invoke(tool_name: str, raw_args: dict[str, Any]) -> dict[str, Any]:
        nonlocal validation_rejections
        with lock:
            validated, err = validate_tool_arguments(tool_name, raw_args)
            if err is not None:
                validation_rejections += 1
                if validation_rejections > max_validation_rejections:
                    return {
                        "ok": False,
                        "error": "MAX_VALIDATION_ERRORS_EXCEEDED",
                        "tool": tool_name,
                        "message": "Too many malformed tool requests in this episode.",
                    }
                return err

            assert validated is not None
            method = getattr(active_engine, tool_name)
            try:
                result = method(**validated)
            except (ValidationError, ValueError) as exc:
                return {
                    "ok": False,
                    "error": "INVALID_SPEC_PARAMETER",
                    "tool": tool_name,
                    "message": str(exc),
                }
            except Exception as exc:  # pragma: no cover - never leak internal tracebacks
                logger.error("Unhandled engine error in %s: %s", tool_name, exc)
                return {
                    "ok": False,
                    "error": "ENGINE_EXECUTION_ERROR",
                    "tool": tool_name,
                    "message": "Tool execution failed due to an internal validation or state error.",
                }

            if tool_name == "finalize_commissioning" and log_dir:
                _persist_trajectory_safely(active_engine, log_dir)

            if not isinstance(result, dict):
                return {"ok": True, "data": result}
            return result

    @mcp.tool()
    def get_episode_brief() -> dict[str, Any]:
        """Return the episode commissioning brief, initial budget, horizon weeks, quotas, bay IDs, order IDs, and route IDs."""
        return _invoke("get_episode_brief", {})

    @mcp.tool()
    def query_handbook(section: str = "all") -> dict[str, Any]:
        """Query the CryoForge engineering handbook (`overview`, `physics`, `materials_and_insulation`, `transport_and_routes`, `manufacturing_and_bays`, `disruptions_and_revisions`, `scoring`, or `all`)."""
        return _invoke("query_handbook", {"section": section})

    @mcp.tool()
    def inspect_order(order_id: str) -> dict[str, Any]:
        """Inspect full technical requirements, cryogenic medium, pressure, boil-off cap, site limits, route, and SLA for a customer order."""
        return _invoke("inspect_order", {"order_id": order_id})

    @mcp.tool()
    def inspect_route(route_id: str) -> dict[str, Any]:
        """Inspect heavy-haul transport clearances, mass limits, transit duration, freight pricing, and escort permit rules for a route."""
        return _invoke("inspect_route", {"route_id": route_id})

    @mcp.tool()
    def get_factory_state() -> dict[str, Any]:
        """Return current factory bookings, bay schedule & blackouts, crane caps, remaining budget, permit usage, remaining physics sim quota, step count, and active disruption alerts."""
        return _invoke("get_factory_state", {})

    @mcp.tool()
    def run_physics_sim(
        order_id: str,
        unit_count: int,
        unit_volume_m3: float,
        design_pressure_mpa: float,
        wall_thickness_mm: float,
        insulation: str,
        fittings: str,
    ) -> dict[str, Any]:
        """Run deterministic cryogenic physics, hoop-stress, thermal boil-off, material hazard, site-clearance, fabrication time, and cost simulation (consumes 1 simulation budget call)."""
        return _invoke(
            "run_physics_sim",
            {
                "order_id": order_id,
                "unit_count": unit_count,
                "unit_volume_m3": unit_volume_m3,
                "design_pressure_mpa": design_pressure_mpa,
                "wall_thickness_mm": wall_thickness_mm,
                "insulation": insulation,
                "fittings": fittings,
            },
        )

    @mcp.tool()
    def reserve_bay_and_order(
        order_id: str,
        bay_id: str,
        bay_start_week: int,
        unit_count: int,
        unit_volume_m3: float,
        design_pressure_mpa: float,
        wall_thickness_mm: float,
        insulation: str,
        fittings: str,
    ) -> dict[str, Any]:
        """Book a fabrication bay (`BAY-A` or `BAY-B`) starting at `bay_start_week` with a candidate vessel design specification for `order_id`."""
        return _invoke(
            "reserve_bay_and_order",
            {
                "order_id": order_id,
                "bay_id": bay_id,
                "bay_start_week": bay_start_week,
                "unit_count": unit_count,
                "unit_volume_m3": unit_volume_m3,
                "design_pressure_mpa": design_pressure_mpa,
                "wall_thickness_mm": wall_thickness_mm,
                "insulation": insulation,
                "fittings": fittings,
            },
        )

    @mcp.tool()
    def assign_transport(
        order_id: str,
        route_id: str,
        use_escort_permit: bool = False,
    ) -> dict[str, Any]:
        """Assign heavy-haul transport on `route_id` for a booked order, optionally requesting an oversized escort permit."""
        return _invoke(
            "assign_transport",
            {
                "order_id": order_id,
                "route_id": route_id,
                "use_escort_permit": use_escort_permit,
            },
        )

    @mcp.tool()
    def revise_order(
        order_id: str,
        bay_id: str,
        bay_start_week: int,
        unit_count: int,
        unit_volume_m3: float,
        design_pressure_mpa: float,
        wall_thickness_mm: float,
        insulation: str,
        fittings: str,
        route_id: str | None = None,
        use_escort_permit: bool | None = None,
    ) -> dict[str, Any]:
        """Revise an existing order's bay schedule, vessel design specification, and optionally its transport route/permit assignment (incurs retooling fee)."""
        return _invoke(
            "revise_order",
            {
                "order_id": order_id,
                "bay_id": bay_id,
                "bay_start_week": bay_start_week,
                "unit_count": unit_count,
                "unit_volume_m3": unit_volume_m3,
                "design_pressure_mpa": design_pressure_mpa,
                "wall_thickness_mm": wall_thickness_mm,
                "insulation": insulation,
                "fittings": fittings,
                "route_id": route_id,
                "use_escort_permit": use_escort_permit,
            },
        )

    @mcp.tool()
    def finalize_commissioning() -> dict[str, Any]:
        """Lock all bookings and transport assignments and submit the 4-order plant commissioning batch for final verification."""
        return _invoke("finalize_commissioning", {})

    return mcp, active_engine


def create_sse_asgi_app(
    mcp: MCPServer,
    engine: Any,
    *,
    seed: str = "S01",
    host: str = "0.0.0.0",
    verifier_token: str = "",
    log_dir: str | Path | None = None,
) -> Any:
    """Create the SSE Starlette ASGI app with `/health` and zero-trust `/internal/grade` routes."""
    sse_sig = inspect.signature(mcp.sse_app).parameters
    sse_kwargs: dict[str, Any] = {}
    if "host" in sse_sig:
        sse_kwargs["host"] = host
    if "transport_security" in sse_sig:
        try:
            from mcp.server.transport_security import TransportSecuritySettings

            sse_kwargs["transport_security"] = TransportSecuritySettings(
                enable_dns_rebinding_protection=False
            )
        except ImportError:  # pragma: no cover
            pass

    app = mcp.sse_app(**sse_kwargs)

    async def health_endpoint(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "service": "cryoforge-mcp",
                "transport": "sse",
                "seed": seed,
            }
        )

    async def internal_grade_endpoint(request: Request) -> JSONResponse:
        expected = (verifier_token or "").strip()
        provided = (request.headers.get("x-verifier-token") or "").strip()
        if not expected or not provided or not hmac.compare_digest(provided, expected):
            return JSONResponse(
                {
                    "ok": False,
                    "error": "FORBIDDEN_VERIFIER_TOKEN",
                    "message": "Valid X-Verifier-Token header is required for server-side grading.",
                },
                status_code=403,
            )

        _persist_trajectory_safely(engine, log_dir)
        traj = engine.export_trajectory_log()
        traj_dict = traj.model_dump(mode="json")
        verification_dict = traj.verification.model_dump(mode="json")
        return JSONResponse(
            {
                "ok": True,
                "seed": traj.seed,
                "run_id": traj.run_id,
                "finalized": traj.verification.finalized,
                "final_score": traj.verification.final_score,
                "verification": verification_dict,
                "trajectory": traj_dict,
            }
        )

    # Prepend `/health` and `/internal/grade` before any catch-all Mount routes
    app.routes.insert(0, Route("/internal/grade", endpoint=internal_grade_endpoint, methods=["GET"]))
    app.routes.insert(0, Route("/health", endpoint=health_endpoint, methods=["GET"]))
    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mcp_server.server",
        description="CryoForge RL MCP Server (stdio and Harbor SSE sidecar modes).",
    )
    parser.add_argument(
        "--seed",
        default=os.environ.get("CRYOFORGE_SEED", "S01"),
        help="Episode scenario seed ID (e.g. S01..S08). Env: CRYOFORGE_SEED.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default=os.environ.get("CRYOFORGE_TRANSPORT", "stdio"),
        help="MCP transport protocol (`stdio` or `sse`). Env: CRYOFORGE_TRANSPORT.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("CRYOFORGE_HOST", "0.0.0.0"),
        help="Bind host when running with `--transport sse`. Env: CRYOFORGE_HOST.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CRYOFORGE_MCP_PORT", "8090")),
        help="Bind port when running with `--transport sse`. Env: CRYOFORGE_MCP_PORT.",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("CRYOFORGE_RUN_ID") or None,
        help="Optional deterministic trajectory run identifier. Env: CRYOFORGE_RUN_ID.",
    )
    parser.add_argument(
        "--agent-name",
        default=os.environ.get("CRYOFORGE_AGENT_NAME", "agent"),
        help="Agent display name recorded in TrajectoryLog. Env: CRYOFORGE_AGENT_NAME.",
    )
    parser.add_argument(
        "--agent-type",
        default=os.environ.get("CRYOFORGE_AGENT_TYPE", "frontier"),
        help="Agent category (`reference`, `frontier`, `do_nothing`, `stop_early`, `small_before`, `small_after`, `exploit`).",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("CRYOFORGE_MODEL_ID", "unknown"),
        help="Model identifier recorded in TrajectoryLog. Env: CRYOFORGE_MODEL_ID.",
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("CRYOFORGE_LOG_DIR") or None,
        help="Directory where TrajectoryLog JSON is persisted. Env: CRYOFORGE_LOG_DIR.",
    )
    parser.add_argument(
        "--verifier-token",
        default=os.environ.get("CRYOFORGE_VERIFIER_TOKEN", ""),
        help="Shared secret protecting GET /internal/grade in SSE mode. Env: CRYOFORGE_VERIFIER_TOKEN.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parse_args(argv)
    mcp, engine = create_cryoforge_mcp_server(
        seed=args.seed,
        run_id=args.run_id,
        agent_name=args.agent_name,
        agent_type=args.agent_type,
        model_id=args.model_id,
        log_dir=args.log_dir,
    )

    try:
        if args.transport == "stdio":
            mcp.run(transport="stdio")
        elif args.transport == "sse":
            import uvicorn

            asgi_app = create_sse_asgi_app(
                mcp,
                engine,
                seed=args.seed,
                host=args.host,
                verifier_token=args.verifier_token,
                log_dir=args.log_dir,
            )
            uvicorn.run(asgi_app, host=args.host, port=args.port, log_level="warning")
        else:
            run_sig = inspect.signature(mcp.run).parameters
            if "kwargs" in run_sig or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in run_sig.values()
            ):
                mcp.run(transport="streamable-http", host=args.host, port=args.port)
            else:  # pragma: no cover
                mcp.run(transport="streamable-http")
    finally:
        if args.log_dir:
            _persist_trajectory_safely(engine, args.log_dir)


if __name__ == "__main__":
    main()
