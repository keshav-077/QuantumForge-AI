"""Sanity-Check Baselines (`do_nothing`, `do_nothing_finalize`, `stop_early`) over real MCP.

Verifies anti-reward-hacking scoring invariants end-to-end over live MCP sessions:
1. `run_do_nothing(seed)`: connects over MCP, reads `get_episode_brief()`, takes zero
   commissioning actions, and disconnects -> `final_score == 0.0000` (`unfinalized`).
2. `run_do_nothing_finalize(seed)`: calls `finalize_commissioning()` on turn 1 with
   zero orders booked -> `final_score == 0.0000`.
3. `run_stop_early(seed)`: solves ONLY `ORD-1` validly and calls `finalize_commissioning()`
   early -> `valid_order_count == 1`, capped at `0.0800` (`<= 0.10`) by the batch completion gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import tempfile
from typing import Any

from eval.mcp_client_helper import (
    DEFAULT_TRAJECTORIES_DIR,
    load_persisted_trajectory,
    open_cryoforge_mcp_session,
)
from eval.reference_solver import (
    HandbookModel,
    generate_pareto_candidates_for_order,
)

DEFAULT_EVAL_SEEDS: tuple[str, ...] = ("S01", "S02", "S03", "S04", "S05")


async def run_do_nothing(
    seed: str = "S01",
    *,
    mcp_url: str | None = None,
    run_id: str | None = None,
    log_dir: str | Path | None = DEFAULT_TRAJECTORIES_DIR,
    call_brief: bool = True,
) -> dict[str, Any]:
    """Connect to MCP, optionally call `get_episode_brief()`, take zero actions, verify `0.0000`."""
    effective_run_id = run_id or f"do_nothing_{seed}"
    temp_ctx = tempfile.TemporaryDirectory() if (log_dir is None and not mcp_url) else None
    effective_log_dir = Path(temp_ctx.name) if temp_ctx is not None else log_dir

    try:
        async with open_cryoforge_mcp_session(
            seed=seed,
            mcp_url=mcp_url,
            run_id=effective_run_id,
            agent_name="Do-Nothing Baseline",
            agent_type="do_nothing",
            model_id="baseline-do-nothing",
            log_dir=effective_log_dir,
        ) as client:
            if call_brief:
                await client.call_tool("get_episode_brief")
            sse_grade = client.fetch_sse_grade() if mcp_url else None

        persisted = (
            load_persisted_trajectory(effective_run_id, effective_log_dir)
            if not mcp_url
            else None
        )
        verification = (sse_grade or {}).get("verification") or (persisted or {}).get(
            "verification"
        )
        if verification is None:
            raise RuntimeError(
                f"Could not retrieve server verification report for run_do_nothing({seed})."
            )

        score = float(verification["final_score"])
        if abs(score) > 1e-9:
            raise AssertionError(f"Expected do_nothing score == 0.0, got {score}")

        return {
            "baseline": "do_nothing",
            "seed": seed,
            "run_id": effective_run_id,
            "finalized": bool(verification["finalized"]),
            "final_score": score,
            "valid_order_count": int(verification["valid_order_count"]),
            "verification": verification,
        }
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


async def run_do_nothing_finalize(
    seed: str = "S01",
    *,
    mcp_url: str | None = None,
    run_id: str | None = None,
    log_dir: str | Path | None = DEFAULT_TRAJECTORIES_DIR,
) -> dict[str, Any]:
    """Call `finalize_commissioning()` immediately with 0 orders booked, verify `0.0000`."""
    effective_run_id = run_id or f"do_nothing_finalize_{seed}"
    async with open_cryoforge_mcp_session(
        seed=seed,
        mcp_url=mcp_url,
        run_id=effective_run_id,
        agent_name="Do-Nothing Immediate Finalize Baseline",
        agent_type="do_nothing",
        model_id="baseline-immediate-finalize",
        log_dir=log_dir,
    ) as client:
        fin = await client.call_tool("finalize_commissioning")
        if not fin.get("ok"):
            raise RuntimeError(f"finalize_commissioning() failed: {fin}")

    verification = fin["verification"]
    score = float(fin["final_score"])
    if abs(score) > 1e-9:
        raise AssertionError(f"Expected do_nothing_finalize score == 0.0, got {score}")

    return {
        "baseline": "do_nothing_finalize",
        "seed": seed,
        "run_id": effective_run_id,
        "finalized": bool(fin["finalized"]),
        "final_score": score,
        "valid_order_count": int(fin["valid_order_count"]),
        "verification": verification,
    }


async def run_stop_early(
    seed: str = "S01",
    *,
    mcp_url: str | None = None,
    run_id: str | None = None,
    log_dir: str | Path | None = DEFAULT_TRAJECTORIES_DIR,
) -> dict[str, Any]:
    """Book and route ONLY `ORD-1` validly, finalize early, and verify batch gate `<= 0.08` (`<= 0.10`)."""
    effective_run_id = run_id or f"stop_early_{seed}"
    async with open_cryoforge_mcp_session(
        seed=seed,
        mcp_url=mcp_url,
        run_id=effective_run_id,
        agent_name="Stop-Early Single-Order Baseline",
        agent_type="stop_early",
        model_id="baseline-stop-early",
        log_dir=log_dir,
    ) as client:
        hb_resp = await client.call_tool("query_handbook", {"section": "all"})
        hb = HandbookModel.from_handbook_response(hb_resp)

        ord_resp = await client.call_tool("inspect_order", {"order_id": "ORD-1"})
        if not ord_resp.get("ok"):
            raise RuntimeError(f"inspect_order('ORD-1') failed: {ord_resp}")
        order = ord_resp["order"]
        route_id = str(order["route_id"])

        rt_resp = await client.call_tool("inspect_route", {"route_id": route_id})
        route = rt_resp["route"]
        fstate = await client.call_tool("get_factory_state")

        cands = generate_pareto_candidates_for_order(
            hb,
            order,
            route,
            pvi_k_multiplier=float(fstate.get("active_pvi_k_multiplier", 1.0)),
            fittings_lead_deltas={
                str(k): int(v) for k, v in (fstate.get("active_fittings_lead_deltas") or {}).items()
            },
            bay_max_unit_tare_kg={
                str(k): float(v) for k, v in (fstate.get("bay_max_unit_tare_kg") or {}).items()
            },
        )
        # Filter for candidates that also respect escort_permits_total and SLA at start week 1
        permits_avail = int(fstate["escort_permits_total"])
        sla_cap = min(int(order["sla_week"]), int(fstate["horizon_weeks"]))
        valid_cands = [
            c
            for c in cands
            if (not c.use_escort_permit or permits_avail >= 1)
            and (
                max(1 + c.fab_weeks, 1 + c.fittings_lead_weeks)
                + hb.assembly_commission_weeks
                + int(route["transit_weeks"])
                <= sla_cap
            )
        ]
        if not valid_cands:
            raise RuntimeError(f"No valid single-order candidate found for ORD-1 on {seed}.")

        # Prefer EVI in single-order stop_early so post-shock PVI moisture doesn't zero out ORD-1
        valid_cands.sort(key=lambda c: (c.insulation != "EVI", c.base_order_cost_eur))
        chosen = valid_cands[0]
        chosen_bay = chosen.eligible_bays[0]

        book_res = await client.call_tool(
            "reserve_bay_and_order",
            {
                "order_id": "ORD-1",
                "bay_id": chosen_bay,
                "bay_start_week": 1,
                "unit_count": chosen.unit_count,
                "unit_volume_m3": chosen.unit_volume_m3,
                "design_pressure_mpa": chosen.design_pressure_mpa,
                "wall_thickness_mm": chosen.wall_thickness_mm,
                "insulation": chosen.insulation,
                "fittings": chosen.fittings,
            },
        )
        if not book_res.get("ok"):
            raise RuntimeError(f"Stop-early reserve_bay_and_order failed: {book_res}")

        tr_res = await client.call_tool(
            "assign_transport",
            {
                "order_id": "ORD-1",
                "route_id": route_id,
                "use_escort_permit": chosen.use_escort_permit,
            },
        )
        if not tr_res.get("ok"):
            raise RuntimeError(f"Stop-early assign_transport failed: {tr_res}")

        fin = await client.call_tool("finalize_commissioning")
        if not fin.get("ok"):
            raise RuntimeError(f"Stop-early finalize_commissioning failed: {fin}")

    verification = fin["verification"]
    score = float(fin["final_score"])
    if score > 0.10 + 1e-9:
        raise AssertionError(
            f"Expected stop_early score <= 0.10 (0.08 batch gate), got {score}"
        )

    return {
        "baseline": "stop_early",
        "seed": seed,
        "run_id": effective_run_id,
        "finalized": bool(fin["finalized"]),
        "final_score": score,
        "valid_order_count": int(fin["valid_order_count"]),
        "batch_gate_cap_applied": bool(verification.get("batch_gate_cap_applied", False)),
        "verification": verification,
    }


async def run_all_baselines(
    seeds: tuple[str, ...] = DEFAULT_EVAL_SEEDS,
    *,
    log_dir: str | Path | None = DEFAULT_TRAJECTORIES_DIR,
) -> dict[str, Any]:
    """Run all three baseline checks across `seeds` and summarize results."""
    do_nothing_runs: list[dict[str, Any]] = []
    do_nothing_fin_runs: list[dict[str, Any]] = []
    stop_early_runs: list[dict[str, Any]] = []

    for seed in seeds:
        do_nothing_runs.append(await run_do_nothing(seed=seed, log_dir=log_dir))
        do_nothing_fin_runs.append(await run_do_nothing_finalize(seed=seed, log_dir=log_dir))
        stop_early_runs.append(await run_stop_early(seed=seed, log_dir=log_dir))

    return {
        "all_passed": True,
        "seeds": list(seeds),
        "do_nothing": {
            "mean_score": round(
                sum(r["final_score"] for r in do_nothing_runs) / max(1, len(do_nothing_runs)), 4
            ),
            "episodes": do_nothing_runs,
        },
        "do_nothing_finalize": {
            "mean_score": round(
                sum(r["final_score"] for r in do_nothing_fin_runs)
                / max(1, len(do_nothing_fin_runs)),
                4,
            ),
            "episodes": do_nothing_fin_runs,
        },
        "stop_early": {
            "mean_score": round(
                sum(r["final_score"] for r in stop_early_runs) / max(1, len(stop_early_runs)), 4
            ),
            "max_score": max((r["final_score"] for r in stop_early_runs), default=0.0),
            "episodes": stop_early_runs,
        },
    }


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m eval.baselines",
        description="Run CryoForge RL sanity-check baselines over live MCP.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=list(DEFAULT_EVAL_SEEDS),
        help="Seeds to evaluate (default: S01..S05).",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_TRAJECTORIES_DIR),
        help="Directory to persist TrajectoryLog JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save baseline summary JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_cli(argv)
    summary = asyncio.run(
        run_all_baselines(seeds=tuple(args.seeds), log_dir=args.log_dir)
    )
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
