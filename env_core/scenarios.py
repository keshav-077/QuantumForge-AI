"""Deterministic multi-constraint commissioning scenarios (S01..S08) for CryoForge RL.

Includes 5 evaluation seeds (S01..S05) and 3 held-out small-model tiebreaker seeds (S06..S08).
Every seed features 4 orders (ORD-1 warm-up + ORD-2..ORD-4 coupled geometry, stress, thermal
boil-off, hazard compatibility, permit quota, bay crane/blackout, and mid-episode supply shock).
"""

from __future__ import annotations

from env_core.contracts import (
    ALL_SEEDS,
    BayBlackoutWindow,
    BayId,
    CustomerOrder,
    DisruptionEvent,
    EVAL_SEEDS,
    ExecutionType,
    FittingsGrade,
    Medium,
    StrictModel,
    TIEBREAKER_SEEDS,
    TransportRoute,
)


class ScenarioSpec(StrictModel):
    seed: str
    title: str
    description: str
    horizon_weeks: int
    total_budget_eur: float
    escort_permits_total: int
    orders: list[CustomerOrder]
    routes: list[TransportRoute]
    disruption: DisruptionEvent


_SCENARIO_REGISTRY: dict[str, ScenarioSpec] = {
    "S01": ScenarioSpec(
        seed="S01",
        title="Nordic Cryo Expansion — Customs Hold, Perlite Humidity & Fjord Bridge Downgrade",
        description=(
            "Commission 4 cryogenic orders across Central, Baltic Port, and Northern Fjord "
            "corridors while adapting to a mid-batch DE_PRECISION cryo-valve customs hold (+3w), "
            "Nordic maritime perlite humidity (1.25x PVI k_eff), and an emergency R-NORTH bridge mass cap (5,800 kg)."
        ),
        horizon_weeks=12,
        total_budget_eur=377500.0,
        escort_permits_total=1,
        routes=[
            TransportRoute(
                route_id="R-CENTRAL",
                name="Central Autobahn Corridor",
                transit_weeks=1,
                std_max_width_mm=2500,
                std_max_height_mm=3800,
                permit_max_width_mm=3100,
                permit_max_height_mm=4350,
                escort_permit_allowed=True,
                max_unit_transport_mass_kg=16500.0,
                base_freight_eur_per_unit=2400.0,
                permit_fee_eur=3800.0,
                notes="Arterial highway serving central industrial accounts.",
            ),
            TransportRoute(
                route_id="R-NORTH",
                name="Northern Fjord Bridge Route",
                transit_weeks=2,
                std_max_width_mm=2650,
                std_max_height_mm=3800,
                permit_max_width_mm=3100,
                permit_max_height_mm=4350,
                escort_permit_allowed=True,
                max_unit_transport_mass_kg=12000.0,
                base_freight_eur_per_unit=3100.0,
                permit_fee_eur=4400.0,
                notes="Northern industrial highway crossing coastal fjord bridges.",
            ),
            TransportRoute(
                route_id="R-PORT",
                name="Baltic Heavy Port Corridor",
                transit_weeks=2,
                std_max_width_mm=2500,
                std_max_height_mm=3800,
                permit_max_width_mm=3150,
                permit_max_height_mm=4400,
                escort_permit_allowed=True,
                max_unit_transport_mass_kg=17500.0,
                base_freight_eur_per_unit=3000.0,
                permit_fee_eur=4200.0,
                notes="Port access corridor serving coastal terminal facilities.",
            ),
        ],
        orders=[
            CustomerOrder(
                order_id="ORD-1",
                client_name="VattenCryo GmbH",
                medium=Medium.LIN,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=7200.0,
                min_pressure_mpa=1.4,
                max_boil_off_pct_day=0.50,
                sla_week=7,
                route_id="R-CENTRAL",
                max_site_height_mm=6500,
                max_site_footprint_mm=4500,
                notes="Stationary process nitrogen buffer receiver.",
            ),
            CustomerOrder(
                order_id="ORD-2",
                client_name="Nordic Steelworks AS",
                medium=Medium.LAR,
                execution=ExecutionType.TRANSPORTABLE,
                required_product_mass_kg=18500.0,
                min_pressure_mpa=1.8,
                max_boil_off_pct_day=0.22,
                sla_week=11,
                route_id="R-NORTH",
                max_site_height_mm=3800,
                max_site_footprint_mm=5200,
                notes="Mobile liquid argon supply skid for converter refining.",
            ),
            CustomerOrder(
                order_id="ORD-3",
                client_name="HyDrive Mobility Lab",
                medium=Medium.LH2,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=650.0,
                min_pressure_mpa=1.6,
                max_boil_off_pct_day=0.72,
                sla_week=10,
                route_id="R-CENTRAL",
                max_site_height_mm=5800,
                max_site_footprint_mm=5000,
                notes="Stationary liquid hydrogen refueling test storage vessel.",
            ),
            CustomerOrder(
                order_id="ORD-4",
                client_name="Baltic OxyMed Terminal",
                medium=Medium.LOX,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=20500.0,
                min_pressure_mpa=2.4,
                max_boil_off_pct_day=0.12,
                sla_week=12,
                route_id="R-PORT",
                max_site_height_mm=6000,
                max_site_footprint_mm=3400,
                notes="Bulk liquid oxygen terminal storage receiver.",
            ),
        ],
        disruption=DisruptionEvent(
            disruption_id="DIS-S01",
            title="DE_PRECISION Customs Hold (+3w), Perlite Moisture (1.25x) & R-NORTH Mass Cap (5,800 kg)",
            headline="Customs adds +3w to DE_PRECISION; coastal humidity raises PVI k_eff 1.25x; R-NORTH bridge mass capped at 5,800 kg.",
            details=(
                "DE_PRECISION base lead increases by +3 weeks (to 9 weeks). Maritime humidity raises "
                "pvi_k_multiplier to 1.25 across PVI shells. Structural bearing cracks on R-NORTH "
                "reduce max_unit_transport_mass_kg from 12,000 kg to 5,800 kg; heavy single units must split."
            ),
            fittings_lead_deltas={FittingsGrade.DE_PRECISION: 3},
            pvi_k_multiplier=1.25,
            route_mass_cap_overrides={"R-NORTH": 5800.0},
        ),
    ),
    "S02": ScenarioSpec(
        seed="S02",
        title="Alpine LNG & Oxygen Hub — Bay-B Furnace Outage & Contingency Freeze",
        description=(
            "Schedule high-pressure LNG, LOX, LCO2, and indoor LIN vessels when BAY-B suffers a "
            "weeks [5, 8) vacuum-furnace maintenance blackout and finance freezes 15,000 EUR."
        ),
        horizon_weeks=13,
        total_budget_eur=357500.0,
        escort_permits_total=1,
        routes=[
            TransportRoute(
                route_id="R-WEST",
                name="Rhine Industrial Arterial",
                transit_weeks=1,
                max_unit_transport_mass_kg=16500.0,
                base_freight_eur_per_unit=2500.0,
                permit_fee_eur=3900.0,
                notes="Primary Rhine industrial highway.",
            ),
            TransportRoute(
                route_id="R-SOUTH",
                name="Bavarian Heavy Corridor",
                transit_weeks=2,
                max_unit_transport_mass_kg=15500.0,
                base_freight_eur_per_unit=2900.0,
                permit_fee_eur=4200.0,
                notes="Southern corridor serving regional power generation sites.",
            ),
        ],
        orders=[
            CustomerOrder(
                order_id="ORD-1",
                client_name="RheinBrew Carbonics",
                medium=Medium.LCO2,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=8500.0,
                min_pressure_mpa=2.0,
                max_boil_off_pct_day=0.16,
                sla_week=7,
                route_id="R-WEST",
                max_site_height_mm=6200,
                max_site_footprint_mm=3500,
                notes="Stationary carbon dioxide liquefaction receiver.",
            ),
            CustomerOrder(
                order_id="ORD-2",
                client_name="AlpenPeak LNG Power",
                medium=Medium.LNG,
                execution=ExecutionType.TRANSPORTABLE,
                required_product_mass_kg=5800.0,
                min_pressure_mpa=2.8,
                max_boil_off_pct_day=0.12,
                sla_week=12,
                route_id="R-SOUTH",
                max_site_height_mm=3600,
                max_site_footprint_mm=3200,
                notes="Mobile liquefied natural gas peaking fuel skid.",
            ),
            CustomerOrder(
                order_id="ORD-3",
                client_name="SuedOxy Klinikum",
                medium=Medium.LOX,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=15500.0,
                min_pressure_mpa=2.3,
                max_boil_off_pct_day=0.11,
                sla_week=9,
                route_id="R-WEST",
                max_site_height_mm=5400,
                max_site_footprint_mm=3400,
                notes="Hospital central oxygen supply storage vessel.",
            ),
            CustomerOrder(
                order_id="ORD-4",
                client_name="Freiburg CryoPhotonics",
                medium=Medium.LIN,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=11000.0,
                min_pressure_mpa=1.5,
                max_boil_off_pct_day=0.24,
                sla_week=13,
                route_id="R-WEST",
                max_site_height_mm=4600,
                max_site_footprint_mm=4800,
                notes="Indoor photonics facility nitrogen supply installation.",
            ),
        ],
        disruption=DisruptionEvent(
            disruption_id="DIS-S02",
            title="BAY-B Vacuum Furnace Blackout [5, 8) & -15,000 EUR Contingency Freeze",
            headline="BAY-B vacuum pumpdown furnace offline during weeks [5, 8); treasury freezes 15,000 EUR.",
            details=(
                "No fabrication booking on BAY-B may overlap weeks [5, 8). Any long-duration job "
                "spanning week 5 on BAY-B must be relocated or rescheduled around the blackout window, "
                "and effective factory budget drops by 15,000 EUR."
            ),
            budget_delta_eur=-15000.0,
            bay_blackouts=[
                BayBlackoutWindow(
                    bay_id=BayId.BAY_B,
                    start_week=5,
                    end_week=8,
                    reason="Emergency turbo-molecular vacuum furnace rebuild",
                )
            ],
        ),
    ),
    "S03": ScenarioSpec(
        seed="S03",
        title="Semiconductor Fab Cryo Batch — Perlite Contamination & Permit Cut",
        description=(
            "Deliver semiconductor wafer-fab cryogenic vessels when moisture-contaminated perlite "
            "degrades PVI conductivity by 1.55x and escort permit quota drops from 2 to 1."
        ),
        horizon_weeks=12,
        total_budget_eur=353500.0,
        escort_permits_total=2,
        routes=[
            TransportRoute(
                route_id="R-EAST",
                name="Silicon Saxony Expressway",
                transit_weeks=1,
                max_unit_transport_mass_kg=16500.0,
                base_freight_eur_per_unit=2600.0,
                permit_fee_eur=4000.0,
                notes="Expressway serving semiconductor manufacturing park.",
            ),
            TransportRoute(
                route_id="R-METRO",
                name="Dresden Fab Ring Road",
                transit_weeks=1,
                max_unit_transport_mass_kg=14500.0,
                base_freight_eur_per_unit=2300.0,
                permit_fee_eur=3900.0,
                notes="Metropolitan ring corridor serving cleanroom gas yards.",
            ),
        ],
        orders=[
            CustomerOrder(
                order_id="ORD-1",
                client_name="ElbeLitho Cleanroom",
                medium=Medium.LCO2,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=9200.0,
                min_pressure_mpa=1.8,
                max_boil_off_pct_day=0.20,
                sla_week=7,
                route_id="R-EAST",
                max_site_height_mm=6200,
                max_site_footprint_mm=3200,
                notes="Cleanroom process chiller LCO2 storage tank.",
            ),
            CustomerOrder(
                order_id="ORD-2",
                client_name="SaxonyWafer Fab 4",
                medium=Medium.LIN,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=14800.0,
                min_pressure_mpa=1.5,
                max_boil_off_pct_day=0.35,
                sla_week=11,
                route_id="R-EAST",
                max_site_height_mm=6500,
                max_site_footprint_mm=5000,
                notes="Wafer-fab nitrogen purge utility storage installation.",
            ),
            CustomerOrder(
                order_id="ORD-3",
                client_name="PlasmaEtch Gases",
                medium=Medium.LAR,
                execution=ExecutionType.TRANSPORTABLE,
                required_product_mass_kg=10000.0,
                min_pressure_mpa=1.8,
                max_boil_off_pct_day=0.42,
                sla_week=8,
                route_id="R-METRO",
                max_site_height_mm=3200,
                max_site_footprint_mm=3000,
                notes="Mobile liquid argon delivery skid for plasma etch bay.",
            ),
            CustomerOrder(
                order_id="ORD-4",
                client_name="OxideGate EpiWorks",
                medium=Medium.LOX,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=25500.0,
                min_pressure_mpa=2.2,
                max_boil_off_pct_day=0.10,
                sla_week=12,
                route_id="R-EAST",
                max_site_height_mm=7000,
                max_site_footprint_mm=3200,
                notes="Thermal oxidation liquid oxygen yard storage vessel.",
            ),
        ],
        disruption=DisruptionEvent(
            disruption_id="DIS-S03",
            title="Perlite Batch Moisture Contamination (1.55x k_eff) & Escort Permit Cut (-1)",
            headline="Contaminated perlite batch spikes PVI k_eff by 1.55x; police staffing cuts escort permits from 2 to 1.",
            details=(
                "All PVI vessels experience pvi_k_multiplier = 1.55 during thermal acceptance testing. "
                "Batch escort_permits_total decreases by 1 (only 1 permit available across all 4 orders)."
            ),
            pvi_k_multiplier=1.55,
            escort_permits_delta=-1,
        ),
    ),
    "S04": ScenarioSpec(
        seed="S04",
        title="Trans-Alpine Propulsion Test — ORD-3 Pressure Uprate & Tunnel Height Cap",
        description=(
            "Commission aerospace propulsion vessels when client ORD-3 issues a +0.7 MPa LOX "
            "pressure uprate (crossing 2.0 MPa) and R-ALPINE tunnel repairs cap height at 2,700 mm."
        ),
        horizon_weeks=11,
        total_budget_eur=347500.0,
        escort_permits_total=1,
        routes=[
            TransportRoute(
                route_id="R-VALLEY",
                name="Inn Valley Heavy Link",
                transit_weeks=1,
                max_unit_transport_mass_kg=16500.0,
                base_freight_eur_per_unit=2500.0,
                permit_fee_eur=4000.0,
                notes="Valley arterial corridor serving aerospace propulsion campus.",
            ),
            TransportRoute(
                route_id="R-ALPINE",
                name="Brenner Alpine Tunnel Route",
                transit_weeks=2,
                max_unit_transport_mass_kg=15000.0,
                base_freight_eur_per_unit=3200.0,
                permit_fee_eur=4600.0,
                notes="Trans-alpine mountain highway serving substation sites.",
            ),
        ],
        orders=[
            CustomerOrder(
                order_id="ORD-1",
                client_name="Tyrol Alloys",
                medium=Medium.LAR,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=9800.0,
                min_pressure_mpa=1.5,
                max_boil_off_pct_day=0.36,
                sla_week=6,
                route_id="R-VALLEY",
                max_site_height_mm=5500,
                max_site_footprint_mm=3200,
                notes="Vacuum-metallurgy liquid argon storage receiver.",
            ),
            CustomerOrder(
                order_id="ORD-2",
                client_name="Helvetia CryoGrid",
                medium=Medium.LIN,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=11200.0,
                min_pressure_mpa=1.6,
                max_boil_off_pct_day=0.24,
                sla_week=9,
                route_id="R-ALPINE",
                max_site_height_mm=5600,
                max_site_footprint_mm=4800,
                notes="Superconducting cable cooling nitrogen station.",
            ),
            CustomerOrder(
                order_id="ORD-3",
                client_name="AetherSpace Test Stand B",
                medium=Medium.LOX,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=15200.0,
                min_pressure_mpa=1.8,
                max_boil_off_pct_day=0.11,
                sla_week=10,
                route_id="R-VALLEY",
                max_site_height_mm=5600,
                max_site_footprint_mm=3400,
                notes="Propulsion test stand liquid oxygen run receiver.",
            ),
            CustomerOrder(
                order_id="ORD-4",
                client_name="AetherSpace LH2 Stand A",
                medium=Medium.LH2,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=1250.0,
                min_pressure_mpa=1.8,
                max_boil_off_pct_day=0.62,
                sla_week=11,
                route_id="R-VALLEY",
                max_site_height_mm=6000,
                max_site_footprint_mm=3400,
                notes="Upper-stage test stand liquid hydrogen storage vessel.",
            ),
        ],
        disruption=DisruptionEvent(
            disruption_id="DIS-S04",
            title="ORD-3 +0.7 MPa LOX Uprate & R-ALPINE 2,700 mm Tunnel Height Cap",
            headline="ORD-3 min_pressure_mpa uprated from 1.8 to 2.5 MPa; R-ALPINE tunnel scaffolding caps height at 2,700 mm and suspends permits.",
            details=(
                "ORD-3 min_pressure_mpa increases to 2.5 MPa, requiring higher design pressure, thicker "
                "shell wall, and oxygen-clean EU_STD/DE_PRECISION fittings (>2.0 MPa LOX rule). "
                "R-ALPINE std_max_height_mm drops to 2,700 mm with escort_permit_allowed = False "
                "(15m3 shipping height 2,750 mm no longer clears the tunnel)."
            ),
            order_min_pressure_overrides={"ORD-3": 2.5},
            route_std_height_overrides={"R-ALPINE": 2700},
            route_permit_allowed_overrides={"R-ALPINE": False},
        ),
    ),
    "S05": ScenarioSpec(
        seed="S05",
        title="Heavy Cryo Industrial Hub — EU_STD Valve Delay & Bay-A Crane Derating",
        description=(
            "Balance heavy LOX, LIN, and LH2 vessels across fabrication bays when EU_STD valves "
            "slip by +2 weeks and BAY-A's overhead crane is derated to 5,200 kg unit tare mass."
        ),
        horizon_weeks=11,
        total_budget_eur=294500.0,
        escort_permits_total=1,
        routes=[
            TransportRoute(
                route_id="R-HARBOR",
                name="North Sea Harbor Spine",
                transit_weeks=1,
                max_unit_transport_mass_kg=17000.0,
                base_freight_eur_per_unit=2600.0,
                permit_fee_eur=4100.0,
                notes="Harbor industrial haul road serving maritime accounts.",
            ),
            TransportRoute(
                route_id="R-INLAND",
                name="Ruhr Heavy Logistics Corridor",
                transit_weeks=2,
                max_unit_transport_mass_kg=16000.0,
                base_freight_eur_per_unit=2900.0,
                permit_fee_eur=4200.0,
                notes="Inland highway corridor serving metallurgical plants.",
            ),
        ],
        orders=[
            CustomerOrder(
                order_id="ORD-1",
                client_name="Hanseatic FoodFreeze",
                medium=Medium.LIN,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=7100.0,
                min_pressure_mpa=1.3,
                max_boil_off_pct_day=0.48,
                sla_week=7,
                route_id="R-HARBOR",
                max_site_height_mm=6200,
                max_site_footprint_mm=3200,
                notes="Industrial food-freezing nitrogen receiver.",
            ),
            CustomerOrder(
                order_id="ORD-2",
                client_name="RuhrStahl Oxygen Plant",
                medium=Medium.LOX,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=20800.0,
                min_pressure_mpa=2.2,
                max_boil_off_pct_day=0.10,
                sla_week=11,
                route_id="R-HARBOR",
                max_site_height_mm=6000,
                max_site_footprint_mm=3400,
                notes="Blast-furnace oxygen enrichment storage receiver.",
            ),
            CustomerOrder(
                order_id="ORD-3",
                client_name="Westfalen Precision Annealing",
                medium=Medium.LIN,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=3600.0,
                min_pressure_mpa=1.5,
                max_boil_off_pct_day=0.25,
                sla_week=7,
                route_id="R-INLAND",
                max_site_height_mm=5000,
                max_site_footprint_mm=3000,
                notes="Bright-annealing furnace nitrogen storage vessel.",
            ),
            CustomerOrder(
                order_id="ORD-4",
                client_name="Bremen LH2 Marine Bunkering",
                medium=Medium.LH2,
                execution=ExecutionType.TRANSPORTABLE,
                required_product_mass_kg=620.0,
                min_pressure_mpa=1.8,
                max_boil_off_pct_day=1.10,
                sla_week=11,
                route_id="R-HARBOR",
                max_site_height_mm=3200,
                max_site_footprint_mm=5000,
                notes="Mobile liquid hydrogen marine bunkering skid.",
            ),
        ],
        disruption=DisruptionEvent(
            disruption_id="DIS-S05",
            title="EU_STD Valve Delay (+2w) & BAY-A Crane Derating (5,200 kg Tare)",
            headline="Valve foundry backlog adds +2 weeks to EU_STD kits; BAY-A crane hoist derated to 5,200 kg unit tare.",
            details=(
                "EU_STD effective lead rises from 4 to 6 weeks. BAY-A max unit_tare_mass_kg is capped "
                "at 5,200 kg, requiring heavy 20m3+ vessels to run on BAY-B while lighter shells occupy BAY-A."
            ),
            fittings_lead_deltas={FittingsGrade.EU_STD: 2},
            bay_max_unit_tare_kg={BayId.BAY_A: 5200.0},
        ),
    ),
    "S06": ScenarioSpec(
        seed="S06",
        title="Quantum & Fusion Extreme Cryo — LHe/LH2 Precision Delay & Fjord Cap",
        description=(
            "Held-out tiebreaker combining 4.2K LHe superconducting cryostat, 20K LH2 injector, "
            "and mobile LAR detector bank under DE_PRECISION +2w delay, R-FJORD 5,600 kg cap, and -10k EUR cut."
        ),
        horizon_weeks=12,
        total_budget_eur=371000.0,
        escort_permits_total=1,
        routes=[
            TransportRoute(
                route_id="R-CAMPUS",
                name="Science Campus Ring",
                transit_weeks=1,
                max_unit_transport_mass_kg=16500.0,
                base_freight_eur_per_unit=2400.0,
                permit_fee_eur=3900.0,
                notes="Direct ring road to quantum and fusion laboratories.",
            ),
            TransportRoute(
                route_id="R-FJORD",
                name="Coastal Research Viaduct",
                transit_weeks=2,
                std_max_width_mm=2650,
                max_unit_transport_mass_kg=13000.0,
                base_freight_eur_per_unit=3100.0,
                permit_fee_eur=4400.0,
                notes="Coastal highway viaduct serving marine research laboratories.",
            ),
        ],
        orders=[
            CustomerOrder(
                order_id="ORD-1",
                client_name="Nordic Synchrotron",
                medium=Medium.LIN,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=7300.0,
                min_pressure_mpa=1.4,
                max_boil_off_pct_day=0.46,
                sla_week=7,
                route_id="R-CAMPUS",
                max_site_height_mm=6200,
                max_site_footprint_mm=3200,
                notes="Beamline thermal-shield liquid nitrogen supply tank.",
            ),
            CustomerOrder(
                order_id="ORD-2",
                client_name="Fjord Neutrino Array",
                medium=Medium.LAR,
                execution=ExecutionType.TRANSPORTABLE,
                required_product_mass_kg=18000.0,
                min_pressure_mpa=1.7,
                max_boil_off_pct_day=0.20,
                sla_week=12,
                route_id="R-FJORD",
                max_site_height_mm=3600,
                max_site_footprint_mm=5000,
                notes="Mobile high-purity liquid argon detector fill skid.",
            ),
            CustomerOrder(
                order_id="ORD-3",
                client_name="QubitLab Superconducting Hub",
                medium=Medium.LHe,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=1100.0,
                min_pressure_mpa=1.2,
                max_boil_off_pct_day=9.00,
                sla_week=11,
                route_id="R-CAMPUS",
                max_site_height_mm=5500,
                max_site_footprint_mm=3200,
                notes="Dilution-refrigerator central liquid helium storage Dewar.",
            ),
            CustomerOrder(
                order_id="ORD-4",
                client_name="Stellarator Fueling Testbed",
                medium=Medium.LH2,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=1300.0,
                min_pressure_mpa=2.0,
                max_boil_off_pct_day=0.62,
                sla_week=11,
                route_id="R-CAMPUS",
                max_site_height_mm=6000,
                max_site_footprint_mm=3400,
                notes="Pellet-injector liquid hydrogen testbed storage tank.",
            ),
        ],
        disruption=DisruptionEvent(
            disruption_id="DIS-S06",
            title="DE_PRECISION +2w Delay, R-FJORD 5,600 kg Cap & -10,000 EUR Grant Holdback",
            headline="DE_PRECISION cryo-valves delayed +2 weeks; R-FJORD viaduct capped at 5,600 kg per unit; -10,000 EUR budget cut.",
            details=(
                "DE_PRECISION lead rises from 6 to 8 weeks. R-FJORD max_unit_transport_mass_kg drops "
                "to 5,600 kg. Effective plant budget drops by 10,000 EUR."
            ),
            fittings_lead_deltas={FittingsGrade.DE_PRECISION: 2},
            route_mass_cap_overrides={"R-FJORD": 5600.0},
            budget_delta_eur=-10000.0,
        ),
    ),
    "S07": ScenarioSpec(
        seed="S07",
        title="Mediterranean Bunkering & Medical Cryo — Bay-A Outage, PVI Spike & Permit Cut",
        description=(
            "Held-out tiebreaker combining BAY-A radiographic outage [4, 7), PVI 1.50x humidity "
            "degradation, and escort permit reduction from 2 to 1."
        ),
        horizon_weeks=13,
        total_budget_eur=405000.0,
        escort_permits_total=2,
        routes=[
            TransportRoute(
                route_id="R-COAST",
                name="Ligurian Coastal Highway",
                transit_weeks=1,
                max_unit_transport_mass_kg=16500.0,
                base_freight_eur_per_unit=2600.0,
                permit_fee_eur=4100.0,
                notes="Coastal highway with escort permit checkpoints.",
            ),
            TransportRoute(
                route_id="R-PORT-MED",
                name="Genoa Bunkering Pier",
                transit_weeks=2,
                max_unit_transport_mass_kg=17000.0,
                base_freight_eur_per_unit=3000.0,
                permit_fee_eur=4300.0,
                notes="Marine LNG terminal access corridor.",
            ),
        ],
        orders=[
            CustomerOrder(
                order_id="ORD-1",
                client_name="MedGas Bottling",
                medium=Medium.LCO2,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=9000.0,
                min_pressure_mpa=1.8,
                max_boil_off_pct_day=0.18,
                sla_week=7,
                route_id="R-COAST",
                max_site_height_mm=6200,
                max_site_footprint_mm=3200,
                notes="Beverage-grade liquid carbon dioxide storage tank.",
            ),
            CustomerOrder(
                order_id="ORD-2",
                client_name="Genoa Ferry Bunkering",
                medium=Medium.LNG,
                execution=ExecutionType.TRANSPORTABLE,
                required_product_mass_kg=6000.0,
                min_pressure_mpa=2.6,
                max_boil_off_pct_day=0.26,
                sla_week=10,
                route_id="R-PORT-MED",
                max_site_height_mm=3600,
                max_site_footprint_mm=5600,
                notes="Quayside liquefied natural gas ferry bunkering skid.",
            ),
            CustomerOrder(
                order_id="ORD-3",
                client_name="Lombardy Hospital Network",
                medium=Medium.LOX,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=21000.0,
                min_pressure_mpa=2.3,
                max_boil_off_pct_day=0.09,
                sla_week=13,
                route_id="R-COAST",
                max_site_height_mm=6000,
                max_site_footprint_mm=3400,
                notes="Regional hospital reserve liquid oxygen receiver.",
            ),
            CustomerOrder(
                order_id="ORD-4",
                client_name="Turin AeroHeatTreat",
                medium=Medium.LIN,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=14500.0,
                min_pressure_mpa=1.5,
                max_boil_off_pct_day=0.34,
                sla_week=13,
                route_id="R-COAST",
                max_site_height_mm=5600,
                max_site_footprint_mm=4800,
                notes="Aerospace heat-treatment nitrogen quench supply station.",
            ),
        ],
        disruption=DisruptionEvent(
            disruption_id="DIS-S07",
            title="BAY-A Outage [4, 7), PVI 1.50x Degradation & Escort Permit Cut (-1)",
            headline="BAY-A X-ray bunker offline weeks [4, 7); humid perlite raises PVI k_eff 1.50x; permits cut from 2 to 1.",
            details=(
                "No BAY-A booking may overlap [4, 7). PVI conductivity increases by 1.50x. "
                "Available escort permits drop from 2 to 1."
            ),
            pvi_k_multiplier=1.50,
            escort_permits_delta=-1,
            bay_blackouts=[
                BayBlackoutWindow(
                    bay_id=BayId.BAY_A,
                    start_week=4,
                    end_week=7,
                    reason="Radiographic weld inspection bunker recalibration",
                )
            ],
        ),
    ),
    "S08": ScenarioSpec(
        seed="S08",
        title="Continental Hydrogen & Helium Corridor — Pressure Uprate, Crane Cap & Valve Delay",
        description=(
            "Held-out tiebreaker combining ORD-2 LOX +0.8 MPa uprate, R-GORGE 2,700 mm gallery height "
            "lock, BAY-B 5,100 kg crane limit, and EU_STD +2w valve delay."
        ),
        horizon_weeks=12,
        total_budget_eur=385500.0,
        escort_permits_total=1,
        routes=[
            TransportRoute(
                route_id="R-MAIN",
                name="Main-Danube Heavy Trunk",
                transit_weeks=1,
                max_unit_transport_mass_kg=17000.0,
                base_freight_eur_per_unit=2500.0,
                permit_fee_eur=4000.0,
                notes="Heavy industrial arterial route.",
            ),
            TransportRoute(
                route_id="R-GORGE",
                name="Ardennes Gorge Route",
                transit_weeks=2,
                max_unit_transport_mass_kg=14000.0,
                base_freight_eur_per_unit=3100.0,
                permit_fee_eur=4500.0,
                notes="Mountain gorge corridor with rockfall galleries.",
            ),
        ],
        orders=[
            CustomerOrder(
                order_id="ORD-1",
                client_name="Danube MetalGas",
                medium=Medium.LAR,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=12500.0,
                min_pressure_mpa=1.5,
                max_boil_off_pct_day=0.34,
                sla_week=7,
                route_id="R-MAIN",
                max_site_height_mm=6200,
                max_site_footprint_mm=3200,
                notes="Regional industrial liquid argon utility station.",
            ),
            CustomerOrder(
                order_id="ORD-2",
                client_name="Ardennes Glassworks",
                medium=Medium.LOX,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=15000.0,
                min_pressure_mpa=1.7,
                max_boil_off_pct_day=0.11,
                sla_week=12,
                route_id="R-GORGE",
                max_site_height_mm=5400,
                max_site_footprint_mm=4600,
                notes="Oxy-fuel glass melting furnace oxygen receiver.",
            ),
            CustomerOrder(
                order_id="ORD-3",
                client_name="Rhine Hydrogen Bus Depot",
                medium=Medium.LH2,
                execution=ExecutionType.TRANSPORTABLE,
                required_product_mass_kg=950.0,
                min_pressure_mpa=1.8,
                max_boil_off_pct_day=0.95,
                sla_week=11,
                route_id="R-MAIN",
                max_site_height_mm=3200,
                max_site_footprint_mm=5000,
                notes="Municipal transit depot liquid hydrogen refueling skid.",
            ),
            CustomerOrder(
                order_id="ORD-4",
                client_name="CERN-North Magnet Cryo",
                medium=Medium.LHe,
                execution=ExecutionType.STATIONARY,
                required_product_mass_kg=1700.0,
                min_pressure_mpa=1.3,
                max_boil_off_pct_day=7.60,
                sla_week=11,
                route_id="R-MAIN",
                max_site_height_mm=5600,
                max_site_footprint_mm=3200,
                notes="Superconducting magnet test facility liquid helium cryostat.",
            ),
        ],
        disruption=DisruptionEvent(
            disruption_id="DIS-S08",
            title="ORD-2 +0.8 MPa Uprate, R-GORGE 2,700mm Cap, BAY-B 5,100kg Crane Cap & EU_STD +2w",
            headline="ORD-2 uprated to 2.5 MPa; R-GORGE gallery caps height at 2,700 mm (no permits); BAY-B crane capped at 5,100 kg; EU_STD +2w.",
            details=(
                "ORD-2 min_pressure_mpa rises to 2.5 MPa (>2.0 MPa LOX rule). R-GORGE std_max_height_mm "
                "drops to 2,700 mm with escort_permit_allowed = False. BAY-B max unit tare is capped at "
                "5,100 kg, and EU_STD valve kits incur a +2 week delay."
            ),
            order_min_pressure_overrides={"ORD-2": 2.5},
            route_std_height_overrides={"R-GORGE": 2700},
            route_permit_allowed_overrides={"R-GORGE": False},
            bay_max_unit_tare_kg={BayId.BAY_B: 5100.0},
            fittings_lead_deltas={FittingsGrade.EU_STD: 2},
        ),
    ),
}


def get_scenario(seed: str) -> ScenarioSpec:
    """Return a fresh deep-copied ScenarioSpec for `seed` in S01..S08."""
    if not isinstance(seed, str):
        raise ValueError(f"Scenario seed must be a string, got {type(seed).__name__}")
    normalized = seed.strip().upper()
    if normalized not in _SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario seed '{seed}'. Valid seeds: {list(ALL_SEEDS)}")
    return _SCENARIO_REGISTRY[normalized].model_copy(deep=True)


def list_scenarios(include_tiebreaker: bool = True) -> list[ScenarioSpec]:
    """Return ScenarioSpec list for eval seeds (and optionally held-out tiebreaker seeds)."""
    seeds = ALL_SEEDS if include_tiebreaker else EVAL_SEEDS
    return [get_scenario(s) for s in seeds]


__all__ = [
    "ALL_SEEDS",
    "EVAL_SEEDS",
    "TIEBREAKER_SEEDS",
    "ScenarioSpec",
    "get_scenario",
    "list_scenarios",
]
