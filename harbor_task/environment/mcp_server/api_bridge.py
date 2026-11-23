"""FastAPI Replay & Live Evaluation Bridge for CryoForge RL (`mcp_server/api_bridge.py`).

Endpoints:
- `GET /api/health` — service status, trajectory count, scorecard & viewer build availability.
- `GET /api/scorecard` — master evaluation summary (`runs/scorecard.json` or `viewer/public/scorecard.json`).
- `GET /api/runs` — lists recorded `TrajectoryLog` summaries from `runs/trajectories/*.json`.
- `GET /api/runs/{run_id}` — returns full `TrajectoryLog` JSON (with computed blueprint geometry per booked order).
- `GET /api/scenarios` — public order/route summaries across seeds + exact SVG blueprint geometry catalog.
- `GET /api/scenarios/{seed}` — single-seed public scenario summary + SVG geometry catalog.
- `GET /api/blueprint` — on-demand millimeter vessel envelope & mass computation for SVG blueprints.
- Static SPA serving for `viewer/dist` when built.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env_core.contracts import (
    ALL_SEEDS,
    BayId,
    EVAL_SEEDS,
    ExecutionType,
    FITTINGS_PROPS,
    FittingsGrade,
    INSULATION_PROPS,
    InsulationType,
    MAX_DESIGN_PRESSURE_MPA,
    MAX_EPISODE_STEPS,
    MAX_WALL_THICKNESS_MM,
    MEDIUM_PROPS,
    MIN_DESIGN_PRESSURE_MPA,
    MIN_WALL_THICKNESS_MM,
    SIM_BUDGET_DEFAULT,
    STANDARD_VOLUMES_M3,
    TIEBREAKER_SEEDS,
    UNIT_COUNTS_ALLOWED,
    VesselDesignSpec,
)
from env_core.physics import compute_dimensions, compute_masses_and_area

SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(getattr(obj, k)) for k in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _normalize_entity_list(collection: Any, id_key: str) -> list[dict[str, Any]]:
    raw = _to_jsonable(collection)
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        items: list[dict[str, Any]] = []
        for key, val in raw.items():
            if isinstance(val, dict):
                entry = dict(val)
                entry.setdefault(id_key, str(key))
                items.append(entry)
        return items
    return []


def build_geometry_catalog() -> dict[str, Any]:
    """Precompute exact millimeter dimensions for all standard catalogue shells for SVG rendering."""
    by_exec: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    flat_items: list[dict[str, Any]] = []

    for execution in ExecutionType:
        by_exec[execution.value] = {}
        for insulation in InsulationType:
            by_exec[execution.value][insulation.value] = {}
            for vol in STANDARD_VOLUMES_M3:
                dims = compute_dimensions(vol, execution, insulation)
                sample_spec = VesselDesignSpec(
                    unit_count=1,
                    unit_volume_m3=vol,
                    design_pressure_mpa=1.6,
                    wall_thickness_mm=8.0,
                    insulation=insulation,
                    fittings=FittingsGrade.EU_STD,
                )
                area_m2, ref_tare_kg, ref_transport_kg = compute_masses_and_area(
                    sample_spec, execution, dims
                )
                payload = {
                    "unit_volume_m3": vol,
                    "execution": execution.value,
                    "insulation": insulation.value,
                    "annulus_mm": INSULATION_PROPS[insulation]["annulus_mm"],
                    "dimensions_mm": dims.model_dump(mode="json"),
                    "inner_surface_area_m2": round(area_m2, 2),
                    "reference_8mm_tare_kg": ref_tare_kg,
                    "reference_8mm_transport_kg": ref_transport_kg,
                }
                vol_key = f"{vol:g}"
                by_exec[execution.value][insulation.value][vol_key] = payload
                flat_items.append(payload)

    return {
        "by_execution_and_insulation": by_exec,
        "items": flat_items,
        "catalogue_constants": {
            "standard_volumes_m3": list(STANDARD_VOLUMES_M3),
            "unit_counts_allowed": list(UNIT_COUNTS_ALLOWED),
            "bays": [b.value for b in BayId],
            "wall_thickness_mm_range": [MIN_WALL_THICKNESS_MM, MAX_WALL_THICKNESS_MM],
            "design_pressure_mpa_range": [MIN_DESIGN_PRESSURE_MPA, MAX_DESIGN_PRESSURE_MPA],
            "sim_budget_default": SIM_BUDGET_DEFAULT,
            "max_episode_steps": MAX_EPISODE_STEPS,
            "inter_unit_footprint_gap_mm": 400,
            "insulation_props": _to_jsonable(INSULATION_PROPS),
            "fittings_props": _to_jsonable(FITTINGS_PROPS),
            "medium_props": {
                k.value: {
                    pk: ( list(pv) if isinstance(pv, tuple) else pv )
                    for pk, pv in v.items()
                }
                for k, v in MEDIUM_PROPS.items()
            },
        },
    }


def _load_public_scenario(seed: str) -> dict[str, Any]:
    try:
        from env_core import get_scenario  # type: ignore[attr-defined]
    except ImportError:
        from env_core.scenarios import get_scenario  # type: ignore[no-redef]

    raw_scen = get_scenario(seed)
    data = _to_jsonable(raw_scen)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected scenario structure for seed {seed}")

    orders = _normalize_entity_list(data.get("orders", []), "order_id")
    routes = _normalize_entity_list(data.get("routes", []), "route_id")

    budget_eur = (
        data.get("initial_budget_eur")
        or data.get("budget_eur")
        or data.get("total_budget_eur")
        or 0.0
    )
    horizon_weeks = data.get("horizon_weeks") or data.get("max_weeks") or 16
    escort_permits = (
        data.get("initial_escort_permits")
        if "initial_escort_permits" in data
        else data.get("escort_permits_total", data.get("escort_permits", 1))
    )
    initial_blackouts = _to_jsonable(
        data.get("initial_bay_blackouts", data.get("bay_blackouts", []))
    )

    disruption_raw = data.get("disruption") or {}
    disruption_summary = (
        {
            "disruption_id": disruption_raw.get("disruption_id", ""),
            "title": disruption_raw.get("title", ""),
            "headline": disruption_raw.get("headline", ""),
        }
        if isinstance(disruption_raw, dict) and disruption_raw
        else None
    )

    return {
        "seed": data.get("seed", seed),
        "tier": "tiebreaker" if seed in TIEBREAKER_SEEDS else "eval",
        "title": data.get("title", data.get("name", f"CryoForge Scenario {seed}")),
        "description": data.get("description", data.get("brief", "")),
        "horizon_weeks": horizon_weeks,
        "total_budget_eur": budget_eur,
        "initial_budget_eur": budget_eur,
        "escort_permits_total": escort_permits,
        "bays": [b.value for b in BayId],
        "initial_bay_blackouts": initial_blackouts if isinstance(initial_blackouts, list) else [],
        "disruption_summary": disruption_summary,
        "orders": orders,
        "orders_by_id": {o["order_id"]: o for o in orders if "order_id" in o},
        "routes": routes,
        "routes_by_id": {r["route_id"]: r for r in routes if "route_id" in r},
    }


def _summarize_trajectory_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    ver = payload.get("verification") or {}
    steps = payload.get("steps") or []
    return {
        "run_id": payload.get("run_id", path.stem),
        "seed": payload.get("seed", ver.get("seed", "unknown")),
        "agent_name": payload.get("agent_name", "unknown"),
        "agent_type": payload.get("agent_type", "frontier"),
        "model_id": payload.get("model_id", "unknown"),
        "started_at": payload.get("started_at", ""),
        "completed_at": payload.get("completed_at", ""),
        "step_count": ver.get("step_count", len(steps)),
        "finalized": bool(ver.get("finalized", False)),
        "final_score": float(ver.get("final_score", 0.0)),
        "valid_order_count": int(ver.get("valid_order_count", 0)),
        "total_order_count": int(ver.get("total_order_count", 4)),
        "disruption_triggered": bool(ver.get("disruption_triggered", False)),
        "budget_exceeded": bool(ver.get("budget_exceeded", False)),
        "permit_quota_exceeded": bool(ver.get("permit_quota_exceeded", False)),
        "total_budget_eur": float(ver.get("total_budget_eur", 0.0)),
        "total_spent_eur": float(ver.get("total_spent_eur", 0.0)),
        "sim_calls_used": int(ver.get("sim_calls_used", 0)),
        "sim_budget_total": int(ver.get("sim_budget_total", SIM_BUDGET_DEFAULT)),
        "failure_modes": list(ver.get("failure_modes", [])),
        "primary_failure_categories": list(ver.get("primary_failure_categories", [])),
    }


def create_replay_app(
    *,
    repo_root: str | Path | None = None,
    runs_dir: str | Path | None = None,
    viewer_dist_dir: str | Path | None = None,
) -> FastAPI:
    """Create the FastAPI replay & live evaluation server."""
    root = Path(repo_root).resolve() if repo_root else _default_repo_root()
    active_runs_dir = Path(runs_dir).resolve() if runs_dir else (root / "runs")
    trajectories_dir = active_runs_dir / "trajectories"
    dist_dir = Path(viewer_dist_dir).resolve() if viewer_dist_dir else (root / "viewer" / "dist")
    geometry_cache = build_geometry_catalog()

    app = FastAPI(
        title="CryoForge RL — Replay & Evaluation API",
        version="1.0.0",
        description="Read-only trajectory replay, scorecard, and cryogenic vessel blueprint API.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
    )

    def _find_scorecard_path() -> Path | None:
        candidates = [
            active_runs_dir / "scorecard.json",
            root / "viewer" / "public" / "scorecard.json",
            dist_dir / "scorecard.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _list_trajectory_paths() -> list[Path]:
        if not trajectories_dir.is_dir():
            return []
        return sorted(
            (p for p in trajectories_dir.glob("*.json") if p.is_file()),
            key=lambda p: (p.stat().st_mtime, p.name),
            reverse=True,
        )

    @app.get("/api/health")
    async def api_health() -> dict[str, Any]:
        sc_path = _find_scorecard_path()
        traj_paths = _list_trajectory_paths()
        return {
            "ok": True,
            "service": "cryoforge-replay-api",
            "version": "1.0.0",
            "has_scorecard": sc_path is not None,
            "scorecard_source": str(sc_path.relative_to(root)) if sc_path and sc_path.is_relative_to(root) else (str(sc_path) if sc_path else None),
            "trajectory_count": len(traj_paths),
            "viewer_built": (dist_dir / "index.html").is_file(),
            "eval_seeds": list(EVAL_SEEDS),
            "tiebreaker_seeds": list(TIEBREAKER_SEEDS),
        }

    @app.get("/api/scorecard")
    async def api_scorecard() -> dict[str, Any]:
        sc_path = _find_scorecard_path()
        if sc_path is not None:
            try:
                return json.loads(sc_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse scorecard JSON: {exc}",
                ) from exc

        # Graceful live fallback synthesized from available trajectory files
        runs = [
            summary
            for p in _list_trajectory_paths()
            if (summary := _summarize_trajectory_file(p)) is not None
        ]
        return {
            "ok": True,
            "status": "pending_master_scorecard",
            "generated_at": None,
            "eval_seeds": list(EVAL_SEEDS),
            "tiebreaker_seeds": list(TIEBREAKER_SEEDS),
            "runs": runs,
            "summary_by_agent": {},
        }

    @app.get("/api/runs")
    async def api_runs(
        seed: str | None = Query(default=None, description="Optional seed filter (e.g. S01)."),
        agent_type: str | None = Query(default=None, description="Optional agent_type filter."),
    ) -> dict[str, Any]:
        summaries: list[dict[str, Any]] = []
        for path in _list_trajectory_paths():
            summary = _summarize_trajectory_file(path)
            if summary is None:
                continue
            if seed and summary.get("seed") != seed:
                continue
            if agent_type and summary.get("agent_type") != agent_type:
                continue
            summaries.append(summary)
        return {
            "ok": True,
            "count": len(summaries),
            "runs": summaries,
        }

    @app.get("/api/runs/{run_id}")
    async def api_run_detail(run_id: str) -> dict[str, Any]:
        clean_id = run_id[:-5] if run_id.endswith(".json") else run_id
        if not clean_id or not SAFE_RUN_ID_RE.match(clean_id) or ".." in clean_id:
            raise HTTPException(status_code=400, detail="Invalid run_id format.")

        target = (trajectories_dir / f"{clean_id}.json").resolve()
        if not target.is_relative_to(trajectories_dir.resolve()) or not target.is_file():
            raise HTTPException(status_code=404, detail=f"Trajectory '{clean_id}' not found.")

        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Corrupt trajectory JSON: {exc}") from exc

        return payload

    @app.get("/api/scenarios")
    async def api_scenarios(
        include_tiebreaker: bool = Query(default=True, description="Include tiebreaker seeds S06..S08."),
    ) -> dict[str, Any]:
        target_seeds = ALL_SEEDS if include_tiebreaker else EVAL_SEEDS
        scenarios_list: list[dict[str, Any]] = []
        scenarios_by_seed: dict[str, dict[str, Any]] = {}
        load_errors: dict[str, str] = {}

        for seed in target_seeds:
            try:
                scen = _load_public_scenario(seed)
                scenarios_list.append(scen)
                scenarios_by_seed[seed] = scen
            except Exception as exc:
                load_errors[seed] = str(exc)

        return {
            "ok": len(load_errors) == 0,
            "seeds": list(target_seeds),
            "eval_seeds": list(EVAL_SEEDS),
            "tiebreaker_seeds": list(TIEBREAKER_SEEDS),
            "scenarios": scenarios_list,
            "scenarios_by_seed": scenarios_by_seed,
            "geometry": geometry_cache,
            **({"errors": load_errors} if load_errors else {}),
        }

    @app.get("/api/scenarios/{seed}")
    async def api_scenario_detail(seed: str) -> dict[str, Any]:
        clean_seed = seed.strip().upper()
        if clean_seed not in ALL_SEEDS:
            raise HTTPException(status_code=404, detail=f"Unknown scenario seed '{seed}'.")
        try:
            scen = _load_public_scenario(clean_seed)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Scenario module unavailable: {exc}") from exc
        return {
            "ok": True,
            "scenario": scen,
            "geometry": geometry_cache,
        }

    @app.get("/api/blueprint")
    async def api_blueprint(
        unit_volume_m3: float = Query(default=10.0, gt=0.0, le=100.0),
        execution: ExecutionType = Query(default=ExecutionType.STATIONARY),
        insulation: InsulationType = Query(default=InsulationType.EVI),
        unit_count: int = Query(default=1, ge=1, le=3),
        wall_thickness_mm: float = Query(default=8.0, ge=MIN_WALL_THICKNESS_MM, le=MAX_WALL_THICKNESS_MM),
        design_pressure_mpa: float = Query(default=1.6, ge=MIN_DESIGN_PRESSURE_MPA, le=MAX_DESIGN_PRESSURE_MPA),
        fittings: FittingsGrade = Query(default=FittingsGrade.EU_STD),
    ) -> dict[str, Any]:
        dims = compute_dimensions(unit_volume_m3, execution, insulation)
        spec = VesselDesignSpec(
            unit_count=unit_count,
            unit_volume_m3=unit_volume_m3,
            design_pressure_mpa=design_pressure_mpa,
            wall_thickness_mm=wall_thickness_mm,
            insulation=insulation,
            fittings=fittings,
        )
        area_m2, unit_tare_kg, unit_transport_kg = compute_masses_and_area(spec, execution, dims)
        bank_footprint_mm = unit_count * dims.installed_width_mm + (unit_count - 1) * 400
        return {
            "ok": True,
            "spec": spec.model_dump(mode="json"),
            "execution": execution.value,
            "annulus_mm": INSULATION_PROPS[insulation]["annulus_mm"],
            "dimensions_mm": dims.model_dump(mode="json"),
            "bank_footprint_mm": bank_footprint_mm,
            "inner_surface_area_m2": round(area_m2, 2),
            "unit_tare_mass_kg": unit_tare_kg,
            "unit_transport_mass_kg": unit_transport_kg,
        }

    # Mount static Vite SPA assets if `viewer/dist` has been built
    assets_dir = dist_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="viewer-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_viewer_spa(full_path: str) -> Any:
        if full_path.startswith("api/") or full_path == "api":
            raise HTTPException(status_code=404, detail="API endpoint not found.")

        index_html = dist_dir / "index.html"
        if not index_html.is_file():
            return JSONResponse(
                {
                    "ok": True,
                    "service": "cryoforge-replay-api",
                    "viewer_built": False,
                    "message": "Frontend SPA (`viewer/dist`) is not built yet. Build with `cd viewer && npm run build`, or query `/api/*` directly.",
                    "endpoints": [
                        "/api/health",
                        "/api/scorecard",
                        "/api/runs",
                        "/api/scenarios",
                        "/api/blueprint",
                    ],
                }
            )

        if full_path:
            candidate = (dist_dir / full_path).resolve()
            if candidate.is_relative_to(dist_dir) and candidate.is_file():
                return FileResponse(str(candidate))

        return FileResponse(str(index_html))

    return app


app = create_replay_app()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mcp_server.api_bridge",
        description="Run the CryoForge RL FastAPI Replay & Evaluation Dashboard server.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("CRYOFORGE_API_HOST", "127.0.0.1"),
        help="Bind host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("CRYOFORGE_PORT", "8095")),
        help="Bind port (default: CRYOFORGE_PORT or 8095).",
    )
    parser.add_argument(
        "--runs-dir",
        default=os.environ.get("CRYOFORGE_RUNS_DIR") or None,
        help="Optional custom runs directory containing scorecard.json and trajectories/.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = parse_args(argv)
    server_app = create_replay_app(runs_dir=args.runs_dir) if args.runs_dir else app
    uvicorn.run(server_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
