# Frozen `CryoForgeEngine` & Trajectory Contract

All subagents MUST adhere strictly to these signatures and file boundaries.

## 1. File Ownership Boundaries (Zero Overlap)
- **Frozen (DO NOT EDIT):** `env_core/contracts.py`, `env_core/physics.py`, `env_core/ENGINE_API_SPEC.md`, `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`
- **Subagent 1 (`env_core/` + `tests/`):**
  - `env_core/handbook.py`
  - `env_core/scenarios.py`
  - `env_core/engine.py`
  - `env_core/verifier.py`
  - `env_core/__init__.py` (re-export `CryoForgeEngine`, `evaluate_episode`, `get_scenario`)
  - `tests/test_physics_and_verifier.py`
- **Subagent 2 (`mcp_server/` + `harbor_task/`):**
  - `mcp_server/__init__.py`
  - `mcp_server/schemas.py`
  - `mcp_server/server.py`
  - `mcp_server/api_bridge.py`
  - `harbor_task/task.toml`
  - `harbor_task/instruction.md`
  - `harbor_task/environment/Dockerfile`
  - `harbor_task/environment/docker-compose.yaml`
  - `harbor_task/environment/mcp_servers.toml`
  - `harbor_task/solution/solve.sh`
  - `harbor_task/tests/test.sh`
- **Subagent 3 (`eval/`):**
  - `eval/__init__.py`
  - `eval/mcp_client_helper.py`
  - `eval/reference_solver.py`
  - `eval/baselines.py`
  - `eval/exploit_tests.py`
  - `eval/harbor_runner.py`
- **Subagent 4 (`tiebreaker/`):**
  - `tiebreaker/__init__.py`
  - `tiebreaker/collect_trajectories.py`
  - `tiebreaker/train_sft.py`
  - `tiebreaker/eval_small_model.py`
- **Subagent 5 (`viewer/`):**
  - Everything inside `viewer/` (`package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/**`)

---

## 2. Exact `CryoForgeEngine` Interface (`env_core/engine.py`)

```python
class CryoForgeEngine:
    def __init__(
        self,
        seed: str = "S01",
        *,
        run_id: str | None = None,
        agent_name: str = "agent",
        agent_type: str = "frontier",
        model_id: str = "unknown",
        log_dir: str | Path | None = None,
    ) -> None: ...

    # 10 Public Agent Tool Methods (each logs a TrajectoryStepLog and returns a plain JSON-serializable dict):
    def get_episode_brief(self) -> dict[str, Any]: ...
    def query_handbook(self, section: str = "all") -> dict[str, Any]: ...
    def inspect_order(self, order_id: str) -> dict[str, Any]: ...
    def inspect_route(self, route_id: str) -> dict[str, Any]: ...
    def get_factory_state(self) -> dict[str, Any]: ...

    def run_physics_sim(
        self,
        order_id: str,
        unit_count: int,
        unit_volume_m3: float,
        design_pressure_mpa: float,
        wall_thickness_mm: float,
        insulation: str,
        fittings: str,
    ) -> dict[str, Any]: ...

    def reserve_bay_and_order(
        self,
        order_id: str,
        bay_id: str,
        bay_start_week: int,
        unit_count: int,
        unit_volume_m3: float,
        design_pressure_mpa: float,
        wall_thickness_mm: float,
        insulation: str,
        fittings: str,
    ) -> dict[str, Any]: ...

    def assign_transport(
        self,
        order_id: str,
        route_id: str,
        use_escort_permit: bool = False,
    ) -> dict[str, Any]: ...

    def revise_order(
        self,
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
    ) -> dict[str, Any]: ...

    def finalize_commissioning(self) -> dict[str, Any]: ...

    # Harness-only helper (NEVER exposed as an MCP tool to the agent!):
    def export_trajectory_log(self) -> TrajectoryLog: ...
```

### Exact Behavioral & Security Invariants
1. **No `reset` tool exposed to the agent.** If `self.finalized` is `True`, all mutating methods (`run_physics_sim`, `reserve_bay_and_order`, `assign_transport`, `revise_order`, `finalize_commissioning`) MUST return `{"ok": False, "error": "EPISODE_ALREADY_FINALIZED", ...}` without modifying state.
2. **Step Limit:** Every tool call increments `step_count`. If `step_count > MAX_EPISODE_STEPS` (42), return `{"ok": False, "error": "MAX_STEPS_EXCEEDED"}`.
3. **Simulation Budget (`SIM_BUDGET_DEFAULT = 8`):** `run_physics_sim` decrements `sim_calls_remaining` by 1 and returns `{"ok": True, "sim_calls_remaining": ..., "physics": PhysicsEvaluation.model_dump()}`. If `sim_calls_remaining <= 0`, return `{"ok": False, "error": "SIM_BUDGET_EXHAUSTED", "sim_calls_remaining": 0}`.
4. **Anti-Oracle Guarantee on `reserve_bay_and_order` & `revise_order`:**
   - Validate catalogue bounds (`validate_catalogue_bounds`), bay existence, bay non-overlap against other booked orders AND active `bay_blackouts`, and `bay_max_unit_tare_kg` crane limits if active.
   - Return `{"ok": True, "booking": ..., "budget_remaining_eur": ..., "disruption_alert": ...}`.
   - **NEVER** return `hoop_stress_mpa`, `boil_off_pct_day`, `stress_pass`, or `boil_off_pass` in `reserve_bay_and_order` or `revise_order`!
5. **Disruption Trigger:**
   - Fires automatically once `len(booked_orders) >= DISRUPTION_TRIGGER_BOOKINGS` (2) OR `mutation_count >= DISRUPTION_TRIGGER_MUTATIONS` (4), whichever comes first.
   - Applies scenario's `DisruptionEvent` to active orders, routes, bay blackouts, crane caps, PVI multiplier, fittings lead deltas, permit quota, and factory budget.
   - Sets `disruption_triggered = True` and returns `"disruption_alert": DisruptionEvent.model_dump()` in the triggering tool response AND in `get_factory_state()`.
   - When `disruption_triggered` fires, `order_release_week` for any SUBSEQUENT `reserve_bay_and_order` or `revise_order` that changes `fittings` or `unit_count` is `current_factory_week = 2` (whereas initial bookings before shock had `order_release_week = 1`), and `fittings_ready_week = order_release_week + effective_fittings_lead_weeks`.
   - Delivery week formula (documented verbatim in `handbook.py` and checked in `verifier.py`):
     `delivery_week = max(bay_start_week + fab_weeks, order_release_week + effective_fittings_lead_weeks) + ASSEMBLY_COMMISSION_WEEKS + route.transit_weeks` (where `ASSEMBLY_COMMISSION_WEEKS = 1`).
6. **Revision Retooling Fee:**
   - If `disruption_triggered` is `True` and `booking.post_shock_revised` is `False`: fee = `POST_SHOCK_FIRST_REVISION_FEE_EUR` (`1500.0`), and set `post_shock_revised = True`.
   - Otherwise: fee = `STANDARD_REVISION_FEE_EUR` (`4500.0`).
   - Retooling fees accumulate in `retooling_fees_eur` and count toward `total_spent_eur`.
7. **Scoring Formula (`env_core/verifier.py`, documented verbatim in `handbook.py`):**
   - If `not finalized`: `final_score` is multiplied by `0.0` (`unfinalized = 0.0000`! Agents MUST call `finalize_commissioning()` to commission the plant; harness inspection of unfinalized runs sees exact `order_results` diagnostics plus `final_score = 0.0`).
   - An order passes all hard checks (`passed_all_hard_checks = True`) iff:
     1. Order is booked AND has `assigned_route_id == order.route_id` (must deliver to the client's designated route!).
     2. `evaluate_design_physics` under active post-shock world state (`pvi_k_multiplier`, `fittings_lead_delta`, updated `min_pressure_mpa`) has `all_engineering_pass == True`.
     3. No bay overlap with any other booked order on the same bay AND no overlap with active `bay_blackouts` on that bay, AND `unit_tare_mass_kg <= bay_max_unit_tare_kg` (if active on that bay).
     4. Transport envelope check on `route`:
        - `unit_transport_mass_kg <= route.max_unit_transport_mass_kg`
        - If `not use_escort_permit`: `shipping_width_mm <= route.std_max_width_mm` and `shipping_height_mm <= route.std_max_height_mm`
        - If `use_escort_permit`: `route.escort_permit_allowed == True` AND `shipping_width_mm <= route.permit_max_width_mm` and `shipping_height_mm <= route.permit_max_height_mm` AND total permits used across batch `<= effective_escort_permits_total` (if permit quota is exceeded, ALL orders claiming `use_escort_permit=True` fail `PERMIT_QUOTA_EXCEEDED`!).
     5. Schedule check: `delivery_week <= order.sla_week` AND `delivery_week <= episode.horizon_weeks`.
   - Base progress credit (`0.85` max): sum of `order.priority_weight` (`0.25` each across 4 orders -> `0.25 * 0.85 = 0.2125` per valid order).
   - Efficiency & margin bonuses (`0.15` max, awarded ONLY when `valid_order_count == 4` and `not budget_exceeded`):
     - `budget_efficiency_bonus` (`0.07` max): `0.07 * min(1.0, max(0.0, (effective_budget_eur - total_spent_eur) / (0.18 * effective_budget_eur)))`
     - `schedule_margin_bonus` (`0.04` max): `0.04 * min(1.0, avg_slack_weeks / 2.0)`
     - `sim_efficiency_bonus` (`0.04` max): `0.04 * ((sim_budget_total - sim_calls_used) / sim_budget_total)`
   - **Batch Completion Gate (Anti-Cherry-Picking):**
     - `0` valid orders -> `0.0000`
     - `1` valid order -> `min(raw_score, 0.0800)` (`stop_early` / single-order cherry-picking strictly `<= 0.08`!)
     - `2` valid orders -> `raw_score * 0.80` (`~0.34`)
     - `3` valid orders -> `raw_score * 0.92` (`~0.586`)
     - `4` valid orders -> full `raw_score + bonuses` (`0.85` to `1.00`)
   - **Budget Breach Penalty:**
     - `total_spent_eur = sum(manufacturing_cost_eur + unit_count * base_freight_eur_per_unit + (permit_fee_eur if use_escort_permit else 0)) + retooling_fees_eur`
     - If `total_spent_eur > effective_budget_eur`: `budget_exceeded = True`, multiply score by `0.20` and cap at `0.15`!

## 3. MCP Tool Names (`mcp_server/server.py`)
Exact 10 tool names matching `CryoForgeEngine` methods 1-to-1:
1. `get_episode_brief`
2. `query_handbook`
3. `inspect_order`
4. `inspect_route`
5. `get_factory_state`
6. `run_physics_sim`
7. `reserve_bay_and_order`
8. `assign_transport`
9. `revise_order`
10. `finalize_commissioning`

## 4. Scorecard & Runs Artifact Layout (`runs/`)
- Individual trajectories saved to `runs/trajectories/<run_id>.json` (`TrajectoryLog` JSON).
- Master evaluation summary saved to `runs/scorecard.json` and `runs/SCORECARD.md` (and bundled into `viewer/public/scorecard.json` + `viewer/src/data/fallbackScorecard.ts`).
