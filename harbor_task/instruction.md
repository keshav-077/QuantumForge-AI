# CryoForge RL — Cryogenic Plant Commissioning & Supply-Shock Task

You are the **Lead Cryogenic Vessel & Logistics Commissioning Engineer** for a specialized industrial cryogenic equipment plant. Your assignment is to design, schedule on fabrication bays, route via heavy-haul transport, and commission **all 4 customer cryogenic tank orders** in the active production batch.

All factory state, customer contracts, route clearances, physics simulations, and commissioning actions are managed exclusively through the **`cryoforge` MCP server** (`http://cryoforge-mcp:8090/sse`). Local files inside `/workspace` are scratch space only and are **not** inspected by the grader.

---

## Available `cryoforge` MCP Tools

1. `get_episode_brief()` — Returns the batch overview, initial factory budget (EUR), planning horizon (weeks), simulation call quota, escort permit quota, fabrication bay IDs (`BAY-A`, `BAY-B`), customer order IDs, and transport route IDs.
2. `query_handbook(section="all")` — Queries the authoritative CryoForge Engineering Handbook. Available sections: `"overview"`, `"physics"`, `"materials_and_insulation"`, `"transport_and_routes"`, `"manufacturing_and_bays"`, `"disruptions_and_revisions"`, `"scoring"`, or `"all"`.
3. `inspect_order(order_id)` — Returns exact customer specifications: working cryogenic medium (`LIN`, `LOX`, `LAR`, `LCO2`, `LNG`, `LH2`, `LHe`), execution type (`STATIONARY` vertical GHK vs `TRANSPORTABLE` horizontal CTK), required deliverable product mass (`kg`), minimum design pressure (`MPa`), maximum daily thermal boil-off rate (`%/day`), contractual `sla_week`, designated `route_id`, and installed site height/footprint limits (`mm`).
4. `inspect_route(route_id)` — Returns transit duration (`weeks`), standard vs escort-permit shipping width and height limits (`mm`), maximum per-unit transport mass (`kg`), base freight rate per unit (`EUR`), escort permit availability, and permit fee (`EUR`).
5. `get_factory_state()` — Returns active bay bookings, bay blackout windows, crane tare limits, cumulative expenditure vs remaining budget, escort permits used vs available, remaining `run_physics_sim` calls, current step count, and any active mid-episode disruption alert.
6. `run_physics_sim(order_id, unit_count, unit_volume_m3, design_pressure_mpa, wall_thickness_mm, insulation, fittings)` — Runs full deterministic mechanical hoop-stress, pressure, material hazard compatibility, daily boil-off, site-clearance, fabrication duration, and manufacturing cost checks for a candidate design. **Consumes 1 call from your finite simulation budget.**
7. `reserve_bay_and_order(order_id, bay_id, bay_start_week, unit_count, unit_volume_m3, design_pressure_mpa, wall_thickness_mm, insulation, fittings)` — Books a fabrication slot on `BAY-A` or `BAY-B` starting at `bay_start_week`. Note: booking validates catalogue bounds, bay non-overlap, blackout windows, and crane tare caps, **not** hoop stress or thermal boil-off pass/fail.
8. `assign_transport(order_id, route_id, use_escort_permit=False)` — Assigns heavy-haul transport on the order's designated `route_id`, optionally claiming an oversized escort permit from the batch quota.
9. `revise_order(order_id, bay_id, bay_start_week, unit_count, unit_volume_m3, design_pressure_mpa, wall_thickness_mm, insulation, fittings, route_id=None, use_escort_permit=None)` — Revises an already-booked order's bay schedule, vessel design specification, and optionally its transport assignment (incurs a retooling fee; the first post-disruption revision on an order receives a reduced fee).
10. `finalize_commissioning()` — Locks the entire 4-order batch and submits the plant commissioning plan for final grading.

---

## Mandatory Engineering & Operational Requirements

1. **Consult the Handbook & Contracts First**:
   - Call `get_episode_brief()` and `query_handbook(section="all")` to review exact shell geometry, hoop-stress, boil-off, cryogenic hazard rules (e.g., `LH2`, `LHe`, high-pressure `LOX`), fabrication lead-time formulas, delivery-week rules, and scoring rules.
   - Inspect all 4 customer orders (`inspect_order`) and their designated routes (`inspect_route`).
2. **Satisfy Every Hard Constraint Across All 4 Orders**:
   - **Deliverable Capacity**: Usable liquid mass across `unit_count` (`1`, `2`, or `3` modular units of a standard catalogue volume `1, 2, 3, 5, 8, 10, 15, 20, 25, 30 m3` at 95% fill fraction) must meet `required_product_mass_kg`.
   - **Structural & Cryogenic Safety**: `design_pressure_mpa >= min_pressure_mpa`, hoop stress within allowable medium-adjusted stress, material/insulation/fittings hazard rules satisfied, and `boil_off_pct_day <= max_boil_off_pct_day`.
   - **Site & Transport Envelope**: Installed height and modular bank footprint must fit client site clearances; shipping width, shipping height, and per-unit transport mass must fit the order's designated `route_id` (using an escort permit only when allowed and within total batch permit quota).
   - **Bay Schedule & Delivery SLA**: Fabrication intervals `[bay_start_week, bay_start_week + fab_weeks)` on the same bay cannot overlap each other or any `bay_blackouts`, unit tare mass must respect any bay crane limit, and `delivery_week` must not exceed `order.sla_week` or the episode horizon.
   - **Batch Budget & Completion**: Total manufacturing, freight, permit, and retooling spend across all orders must stay within the effective factory budget. Partial batch completion (cherry-picking 1–3 orders) is heavily penalized.
3. **Adapt to Mid-Episode Supply or Shop-Floor Disruptions**:
   - During commissioning, a supply-chain, shop-floor, or logistics shock may trigger (reported via `disruption_alert` and reflected in `get_factory_state()`, `inspect_order()`, and `inspect_route()`).
   - Whenever a disruption fires, audit **all previously booked orders** against updated pressures, SLAs, insulation multipliers, fittings lead deltas, bay blackouts, crane caps, permit quotas, route limits, and total budget. Use `revise_order(...)` and `assign_transport(...)` to restore full compliance before finalizing.
4. **Respect Step & Simulation Quotas and Finalize**:
   - Keep total tool calls within the 42-step episode limit, and conserve `run_physics_sim` calls where possible (unused simulation calls earn an efficiency bonus when all 4 orders pass within budget).
   - **You MUST call `finalize_commissioning()` before finishing.** Any unfinalized episode receives a score of `0.0000`.
