#!/usr/bin/env python3
"""Write LaTeX macros and table rows from the generated benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
PAPER = PROJECT / "manuscript"


def _define(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _integer(value: float | int) -> str:
    return f"{int(round(float(value))):,}"


def _one(value: float | int) -> str:
    return f"{float(value):,.1f}"


def _escape(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%")


def _write_table_rows(path: Path, rows: list[str], *, close_rule: bool = True) -> None:
    """Write rows and, when requested, the closing rule in one alignment input."""
    lines = [f"{row}%" for row in rows]
    if close_rule:
        lines.append(r"\bottomrule")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    case = json.loads((PROJECT / "cases" / "schweinfurt.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "model_manifest_four_sector.json").read_text(encoding="utf-8"))
    inventory = pd.read_csv(OUT / "network_inventory.csv").set_index("sector")
    metrics = pd.read_csv(OUT / "morphology_metrics.csv")
    anchors = pd.read_csv(OUT / "aggregate_anchor_checks.csv")
    corridors = pd.read_csv(OUT / "common_corridors.csv")
    interfaces = pd.read_csv(OUT / "coupling_interfaces.csv")
    buildings = pd.read_csv(OUT / "building_sector_demands.csv")
    zones = pd.read_csv(OUT / "demand_zones.csv")
    benchmark_metrics = pd.read_csv(OUT / "benchmark_case_metrics.csv")
    benchmark_manifest = json.loads((OUT / "benchmark_cases" / "manifest.json").read_text(encoding="utf-8"))
    wastewater_facilities = pd.read_csv(OUT / "wastewater_facility_inventory.csv").set_index("category")
    sewer_sensitivity_matrix = pd.read_csv(
        OUT / "sensitivity" / "wastewater_terrain_grade_sensitivity.csv"
    )

    bidirectional = manifest["solver_readiness"]["bidirectional_cosimulation"]
    final_states = bidirectional["final_sector_states"]
    power = final_states["electricity"]
    water = final_states["drinking_water"]
    heat = final_states["district_heating"]
    swmm_coupled = final_states["wastewater"]
    water_native = manifest["solver_readiness"]["epanet_wntr"]
    heat_native = manifest["solver_readiness"]["pandapipes_heat_supply"]
    swmm = manifest["solver_readiness"]["swmm"]
    runtime = manifest["runtime_seconds"]
    heat_meta = manifest["heat_calibration"]
    shared_by_count = corridors.groupby("sector_count")["length_km"].sum()
    relation_counts = interfaces["relation"].value_counts()
    benchmark_runtime = sum(item["runtime_seconds"] for item in benchmark_manifest.values())
    uncoupled_loading = float(manifest["solver_readiness"]["pandapower_uncoupled"]["maximum_line_loading_percent"])
    coupled_headroom = 100.0 - float(power["maximum_line_loading_percent"])
    uncoupled_headroom = 100.0 - uncoupled_loading
    loading_delta = float(power["maximum_line_loading_percent"]) - uncoupled_loading
    history = pd.read_csv(OUT / "cosimulation" / "bidirectional_convergence_history.csv")
    final_iteration = history.iloc[-1]
    water_pipes = pd.read_csv(OUT / "water_pipes.csv")
    wastewater_links = pd.read_csv(OUT / "wastewater_conduits.csv")
    leak_demo = json.loads(
        (OUT / "hydraulic_leak_coupling" / "hydraulic_leak_manifest.json").read_text(encoding="utf-8")
    )
    leak_response = leak_demo["response"]
    electrical_demo = json.loads(
        (
            OUT
            / "electrical_supply_disturbance"
            / "electrical_supply_disturbance_manifest.json"
        ).read_text(encoding="utf-8")
    )
    electrical_response = electrical_demo["response"]
    electrical_baseline = electrical_demo["baseline"]
    electrical_worst = electrical_demo["worst_disturbed_state"]
    sewer_sensitivity = json.loads(
        (
            OUT
            / "sensitivity"
            / "wastewater_terrain_grade_sensitivity_manifest.json"
        ).read_text(encoding="utf-8")
    )
    leak_baseline = leak_demo["baseline"]
    leak_onset = leak_demo["leak_onset"]
    leak_final = leak_demo["final_leak_state"]
    service_root = OUT / "integrated_service_resolved"
    service_manifest = json.loads(
        (service_root / "integrated_generation_manifest.json").read_text(encoding="utf-8")
    )
    service_native = service_manifest["native_solver_results"]
    service_counts = service_manifest["counts"]
    service_audit = service_manifest["coupling_interface_audit"]
    service_closure = service_manifest["coupled_nominal_interface_closure"]
    service_power_links = pd.read_csv(service_root / "electricity_links.csv")
    service_water_links = pd.read_csv(service_root / "drinking_water_links.csv")
    service_waste_links = pd.read_csv(service_root / "wastewater_links.csv")
    service_heat_corridors = pd.read_csv(service_root / "district_heating_corridors.csv")

    macros = [
        "% Generated by scripts/write_results_tex.py; do not edit manually.",
        _define("CaseBuildings", _integer(len(buildings))),
        _define("DemandZones", _integer(len(zones))),
        _define("BenchmarkCaseCount", _integer(len(benchmark_manifest))),
        _define("BenchmarkRuntime", f"{benchmark_runtime:.1f}"),
        _define("PowerNodes", _integer(inventory.loc["electricity", "nodes"])),
        _define("PowerLinks", _integer(inventory.loc["electricity", "links"])),
        _define("PowerOperatingLinks", _integer(inventory.loc["electricity", "operating_links"])),
        _define("PowerRoute", _one(inventory.loc["electricity", "route_length_km"])),
        _define("WaterNodes", _integer(inventory.loc["drinking_water", "nodes"])),
        _define("WaterLinks", _integer(inventory.loc["drinking_water", "links"])),
        _define("WaterRoute", _one(inventory.loc["drinking_water", "route_length_km"])),
        _define("WaterCycles", _integer(inventory.loc["drinking_water", "cycle_rank_asset"])),
        _define("WasteNodes", _integer(inventory.loc["wastewater", "nodes"])),
        _define("WasteLinks", _integer(inventory.loc["wastewater", "links"])),
        _define("WasteRoute", _one(inventory.loc["wastewater", "route_length_km"])),
        _define("HeatNodes", _integer(inventory.loc["district_heating", "nodes"])),
        _define("HeatLinks", _integer(inventory.loc["district_heating", "links"])),
        _define("HeatRoute", _one(inventory.loc["district_heating", "route_length_km"])),
        _define("HeatMappedRoute", _one(heat_meta["mapped_route_length_km"])),
        _define("HeatLaterals", _one(heat_meta["equivalent_service_laterals_km"])),
        _define("HeatPhysicalPipe", _one(heat_meta["physical_pipe_length_supply_return_km"])),
        _define("HeatSubstations", _integer(heat_meta["consumer_substations"])),
        _define("HeatContracts", _integer(heat_meta["represented_sales_contracts"])),
        _define("CouplingInterfaces", _integer(len(interfaces))),
        _define("WaterWasteMappings", _integer(relation_counts.get("delivered_water_to_sanitary_inflow", 0))),
        _define("ElectricSupplyInterfaces", _integer(relation_counts.get("electric_supply", 0))),
        _define("HeatPumpInterfaces", _integer(relation_counts.get("electric_heat_pump", 0))),
        _define("SharedSegments", _integer(len(corridors))),
        _define("SharedRoute", _one(corridors["length_km"].sum())),
        _define("ThreeSectorRoute", _one(shared_by_count.get(3, 0.0))),
        _define("FourSectorRoute", _one(shared_by_count.get(4, 0.0))),
        _define("PowerMinVoltage", f"{power['minimum_voltage_pu']:.3f}"),
        _define("PowerMaxVoltage", f"{power['maximum_voltage_pu']:.3f}"),
        _define("PowerMaxLoading", f"{power['maximum_line_loading_percent']:.1f}"),
        _define("PowerMaxTransformerLoading", f"{power['maximum_transformer_loading_percent']:.1f}"),
        _define("PowerCouplingLoad", f"{power['coupling_active_power_mw']:.3f}"),
        _define(
            "PowerCouplingPeakShare",
            f"{100.0 * power['coupling_active_power_mw'] / case['official_anchors']['electricity_lv_peak_mw']:.1f}",
        ),
        _define(
            "PowerCouplingMeanShare",
            f"{100.0 * power['coupling_active_power_mw'] / (case['official_anchors']['electricity_lv_annual_mwh'] / 8760.0):.1f}",
        ),
        _define("PowerUncoupledLoading", f"{uncoupled_loading:.1f}"),
        _define("PowerCouplingLoadingDelta", f"{loading_delta:.2f}"),
        _define("PowerCoupledHeadroom", f"{coupled_headroom:.1f}"),
        _define("PowerHeadroomConsumed", f"{100.0 * loading_delta / uncoupled_headroom:.1f}"),
        _define("CouplingIterations", _integer(bidirectional["iterations"])),
        _define("CouplingRepairs", _integer(len(bidirectional.get("repairs_applied", [])))),
        _define("CouplingVoltageResidual", f"{final_iteration['voltage_residual_pu']:.2e}"),
        _define("CouplingPowerResidual", f"{final_iteration['interface_power_l1_relative_residual']:.2e}"),
        _define("CoupledWaterPumpPower", f"{water['pump_electrical_power_mw']:.3f}"),
        _define("CoupledWaterMaxPressure", f"{water['maximum_pressure_m']:.1f}"),
        _define("CoupledWaterDelivered", f"{100.0 * water['delivered_water_fraction']:.1f}"),
        _define("CoupledWastePumpPower", f"{swmm_coupled['total_pump_power_mw']:.4f}"),
        _define("CoupledWasteContinuity", f"{abs(swmm_coupled['flow_routing_continuity_error_percent']):.3f}"),
        _define("CoupledHeatPumpPower", f"{heat['pump_electrical_power_mw']:.3f}"),
        _define("CoupledHeatReturnTemperature", f"{heat['return_temperature_c']:.1f}"),
        _define("CoupledHeatLoss", f"{heat['distribution_heat_loss_mw']:.2f}"),
        _define("WaterMinPressure", f"{water['minimum_pressure_m']:.1f}"),
        _define("WaterMaxVelocity", f"{water['maximum_velocity_m_s']:.2f}"),
        _define("WaterPeakMinPressure", f"{water_native['peak_minimum_pressure_m']:.1f}"),
        _define("WaterLowVelocityShare", f"{100.0 * water_native['fraction_links_below_0_05_m_s']:.1f}"),
        _define("WaterAgePninetyfive", f"{water_native['water_age_p95_hours_at_48h']:.1f}"),
        _define("HeatMinPressure", f"{heat['minimum_pressure_bar']:.1f}"),
        _define("HeatMaxVelocity", f"{heat['maximum_velocity_m_s']:.2f}"),
        _define("WasteContinuityError", f"{abs(swmm.get('flow_routing_continuity_error_percent', float('nan'))):.3f}"),
        _define("WasteFloodingLoss", f"{swmm.get('flooding_loss_million_liter', float('nan')):.3f}"),
        _define("WasteNonConverging", f"{swmm.get('steps_not_converging_percent', float('nan')):.2f}"),
        _define("WasteRoutingDuration", f"{int(swmm.get('routing_duration_hours', 0))}"),
        _define("CoupledWasteRoutingDuration", f"{int(swmm_coupled.get('duration_hours', 0))}"),
        _define("LeakDiameter", f"{leak_demo['scenario']['equivalent_orifice_diameter_mm']:.0f}"),
        _define("LeakStartMinute", f"{int(leak_demo['scenario']['leak_start_minute'])}"),
        _define("LeakFinalFlow", f"{leak_response['final_leak_flow_l_s']:.1f}"),
        _define("LeakPumpFlowIncrease", f"{leak_response['pump_flow_change_percent']:.1f}"),
        _define("LeakPumpPowerBaseline", f"{1000.0 * leak_baseline['pump_electrical_power_mw']:.1f}"),
        _define("LeakPumpPowerFinal", f"{1000.0 * leak_final['pump_electrical_power_mw']:.1f}"),
        _define("LeakPumpPowerIncrease", f"{leak_response['pump_power_change_percent']:.1f}"),
        _define("LeakBusVoltageBaseline", f"{leak_baseline['pump_bus_voltage_pu']:.6f}"),
        _define("LeakBusVoltageFinal", f"{leak_final['pump_bus_voltage_pu']:.6f}"),
        _define("LeakBusVoltageChange", f"{1e4 * leak_response['pump_bus_voltage_change_pu']:.2f}"),
        _define("LeakLineLoadingBaseline", f"{leak_baseline['maximum_line_loading_percent']:.2f}"),
        _define("LeakLineLoadingFinal", f"{leak_final['maximum_line_loading_percent']:.2f}"),
        _define("LeakLineLoadingChange", f"{leak_response['maximum_line_loading_change_percentage_points']:.2f}"),
        _define("LeakCriticalPressureBaseline", f"{leak_baseline['critical_node_pressure_m']:.1f}"),
        _define("LeakCriticalPressureOnset", f"{leak_onset['critical_node_pressure_m']:.1f}"),
        _define("LeakCriticalPressureFinal", f"{leak_final['critical_node_pressure_m']:.1f}"),
        _define("LeakControllerCommandFinal", f"{leak_final['pump_speed_command_pu']:.2f}"),
        _define(
            "ElectricalFaultFactor",
            f"{electrical_demo['scenario']['fault_service_voltage_factor']:.3f}",
        ),
        _define(
            "ElectricalFaultStartMinute",
            f"{int(electrical_demo['scenario']['fault_start_minute'])}",
        ),
        _define(
            "ElectricalFaultClearMinute",
            f"{int(electrical_demo['scenario']['fault_clear_start_minute'])}",
        ),
        _define(
            "ElectricalRecoveryMinute",
            f"{int(electrical_demo['scenario']['voltage_recovery_complete_minute'])}",
        ),
        _define(
            "ElectricalFaultTerminalVoltage",
            f"{electrical_response['minimum_drive_terminal_voltage_pu']:.3f}",
        ),
        _define(
            "ElectricalFaultPumpSpeed",
            f"{electrical_response['minimum_pump_speed_pu']:.3f}",
        ),
        _define(
            "ElectricalFaultPressure",
            f"{electrical_response['minimum_critical_pressure_m']:.1f}",
        ),
        _define(
            "ElectricalFaultDelivered",
            f"{100.0 * electrical_response['minimum_delivered_water_fraction']:.1f}",
        ),
        _define(
            "ElectricalFaultPumpPowerBaseline",
            f"{1000.0 * electrical_baseline['pump_electrical_power_mw']:.1f}",
        ),
        _define(
            "ElectricalFaultPumpPowerWorst",
            f"{1000.0 * electrical_worst['pump_electrical_power_mw']:.1f}",
        ),
        _define(
            "ElectricalFaultPumpPowerChange",
            f"{electrical_response['pump_power_change_percent_at_worst_pressure']:.1f}",
        ),
        _define(
            "ElectricalFaultLineLoadingBaseline",
            f"{electrical_baseline['maximum_line_loading_percent']:.2f}",
        ),
        _define(
            "ElectricalFaultLineLoadingWorst",
            f"{electrical_worst['maximum_line_loading_percent']:.2f}",
        ),
        _define(
            "ElectricalFaultLineLoadingChange",
            f"{electrical_response['line_loading_change_percentage_points_at_worst_pressure']:.2f}",
        ),
        _define(
            "SewerSensitivityLiftMinimum",
            _integer(sewer_sensitivity["ranges"]["lift_edge_count"][0]),
        ),
        _define(
            "SewerSensitivityLiftMaximum",
            _integer(sewer_sensitivity["ranges"]["lift_edge_count"][1]),
        ),
        _define(
            "SewerSensitivityForceMinimum",
            f"{sewer_sensitivity['ranges']['force_main_length_km'][0]:.2f}",
        ),
        _define(
            "SewerSensitivityForceMaximum",
            f"{sewer_sensitivity['ranges']['force_main_length_km'][1]:.2f}",
        ),
        _define("WasteLiftNodes", _integer(wastewater_facilities.loc["derived_lift_station_nodes", "count"])),
        _define("WasteOverflowNodes", _integer(wastewater_facilities.loc["overflow_basin_equivalents", "count"])),
        _define("WasteFacilityColocations", _integer(wastewater_facilities.loc["co_located_lift_and_overflow_nodes", "count"])),
        _define("WasteUniqueFacilities", _integer(wastewater_facilities.loc["unique_facility_nodes", "count"])),
        _define("WaterMinimumDiameterShare", f"{100.0 * (water_pipes['diameter_mm'] == water_pipes['diameter_mm'].min()).mean():.1f}"),
        _define("WasteMinimumDiameterShare", f"{100.0 * (wastewater_links['diameter_mm'] == wastewater_links['diameter_mm'].min()).mean():.1f}"),
        _define("BaseRuntime", f"{runtime.get('base_generation', 0.0):.1f}"),
        _define("FrameworkRuntime", f"{runtime.get('four_sector_generation', 0.0):.1f}"),
        _define("FullRuntime", f"{sum(runtime.values()) + benchmark_runtime:.1f}"),
        _define("ServiceEvidenceBuildings", _integer(service_counts["building_evidence_objects"])),
        _define("ServiceEligibleBuildings", _integer(service_counts["service_eligible_buildings"])),
        _define("ServicePowerNodes", _integer(service_native["electricity"]["buses"])),
        _define("ServicePowerLinks", _integer(service_native["electricity"]["lines"])),
        _define("ServicePowerTransformers", _integer(service_native["electricity"]["transformers"])),
        _define("ServicePowerRoute", _one(service_power_links["length_km"].sum())),
        _define("ServiceWaterNodes", _integer(len(pd.read_csv(service_root / "drinking_water_nodes.csv")))),
        _define("ServiceWaterLinks", _integer(len(service_water_links))),
        _define("ServiceWaterRoute", _one(service_water_links["length_km"].sum())),
        _define("ServiceWaterCycles", _integer(service_native["drinking_water"]["cycle_rank"])),
        _define("ServiceWasteNodes", _integer(len(pd.read_csv(service_root / "wastewater_nodes.csv")))),
        _define("ServiceWasteLinks", _integer(len(service_waste_links))),
        _define("ServiceWasteRoute", _one(service_waste_links["length_km"].sum())),
        _define("ServiceWasteNetworkLifts", _integer(service_native["wastewater"]["lift_stations"])),
        _define("ServiceWastePropertyLifts", _integer(service_native["wastewater"]["property_lift_stations"])),
        _define("ServiceHeatNodes", _integer(len(pd.read_csv(service_root / "district_heating_nodes.csv")))),
        _define("ServiceHeatLinks", _integer(len(service_heat_corridors))),
        _define("ServiceHeatRoute", _one(service_heat_corridors["length_km"].sum())),
        _define("ServiceHeatPhysicalPipe", _one(service_native["district_heating"]["paired_physical_pipe_length_km"])),
        _define("ServiceHeatSubstations", _integer(service_native["district_heating"]["consumer_substations"])),
        _define("ServiceCouplingInterfaces", _integer(service_counts["coupling_interfaces"])),
        _define("ServiceBuildingMappings", _integer(service_audit["building_mass_interfaces"])),
        _define("ServiceFacilityInterfaces", _integer(service_audit["electrically_driven_facilities"])),
        _define("ServiceFacilityPower", f"{service_audit['facility_nominal_power_mw']:.3f}"),
        _define("ServiceSharedSegments", _integer(service_counts["shared_route_segments"])),
        _define("ServiceClosureIterations", _integer(service_closure["iterations"])),
        _define("ServiceClosureMinVoltage", f"{service_closure['minimum_system_voltage_pu']:.3f}"),
        _define("ServiceClosureMaxLine", f"{service_closure['maximum_line_loading_percent']:.1f}"),
        _define("ServiceClosureMaxTrafo", f"{service_closure['maximum_transformer_loading_percent']:.1f}"),
    ]
    (PAPER / "results_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    service_rows = [
        r"Electricity & \ServicePowerNodes & \ServicePowerLinks & \ServicePowerRoute & one connected MV/LV graph & \ServicePowerTransformers{} transformers + \ServiceEligibleBuildings{} services \\",
        r"Drinking water & \ServiceWaterNodes & \ServiceWaterLinks & \ServiceWaterRoute & cycle rank \ServiceWaterCycles & \ServiceEligibleBuildings{} service laterals \\",
        r"Wastewater & \ServiceWasteNodes & \ServiceWasteLinks & \ServiceWasteRoute & directed single-outfall graph & \ServiceEligibleBuildings{} property connections \\",
        r"District heating & \ServiceHeatNodes & \ServiceHeatLinks & \ServiceHeatRoute & source-reachable route forest & \ServiceHeatSubstations{} consumer substations \\",
    ]
    _write_table_rows(PAPER / "generated_service_inventory_rows.tex", service_rows)

    inventory_labels = {
        "electricity": "Electricity",
        "drinking_water": "Drinking water",
        "wastewater": "Wastewater",
        "district_heating": "District heating",
    }
    inventory_rows = []
    for sector, row in inventory.iterrows():
        demand = _one(row.annual_demand / 1000.0)
        demand_unit = "GWh y$^{-1}$" if row.demand_unit == "MWh/y" else "10$^3$ m$^3$ y$^{-1}$"
        inventory_rows.append(
            f"{inventory_labels[sector]} & {_integer(row.nodes)} & {_integer(row.links)} & {_one(row.route_length_km)} & "
            f"{_integer(row.cycle_rank_asset)} / {_integer(row.cycle_rank_operating)} & {_integer(row.facilities)} & "
            f"{row.size_median:g} / {row.size_p90:g} / {row.size_max:g} & {demand} {demand_unit} \\\\"
        )
    _write_table_rows(PAPER / "generated_inventory_rows.tex", inventory_rows)

    morphology_rows = []
    labels = {"dense": "Dense urban", "suburban": "Semi-urban", "peripheral": "Peripheral", "industrial": "Industrial/mixed"}
    for morphology in ["dense", "suburban", "peripheral", "industrial"]:
        subset = metrics[metrics["morphology"] == morphology].set_index("sector")
        routes = "/".join(_one(subset.loc[s, "route_length_km"]) for s in ["electricity", "drinking_water", "wastewater", "district_heating"])
        max_routes = "/".join(_one(subset.loc[s, "maximum_service_route_km"]) for s in ["electricity", "drinking_water", "wastewater", "district_heating"])
        morphology_rows.append(
            f"{labels[morphology]} & {_integer(subset.iloc[0].buildings)} & {_integer(subset.iloc[0].building_density_km2)} & "
            f"{routes} & {max_routes} & {_integer(subset.loc['drinking_water', 'cycle_rank'])} & "
            f"{_integer(subset.loc['district_heating', 'district_heat_contracts'])} \\\\"
        )
    _write_table_rows(PAPER / "generated_morphology_rows.tex", morphology_rows, close_rule=False)

    benchmark_summary_rows = []
    benchmark_sector_rows = []
    case_order = ["dense_urban", "semi_urban", "rural_peripheral"]
    case_labels = {
        "dense_urban": "Schweinfurt Innenstadt (dense)",
        "semi_urban": "Bergl/Gartenstadt (semi-urban)",
        "rural_peripheral": "Dittelbrunn (rural/peripheral)",
    }
    sector_labels = {
        "electricity": "Electricity",
        "drinking_water": "Drinking water",
        "wastewater": "Wastewater",
        "district_heating": "District heating",
    }
    for case_id in case_order:
        subset = benchmark_metrics[benchmark_metrics["case_id"] == case_id].set_index("sector")
        case_manifest = benchmark_manifest[case_id]
        route_lengths = [_one(subset.loc[sector, "route_length_km"]) for sector in sector_labels]
        solvers = case_manifest["solver_readiness"]
        ready = all(item["passed"] for item in case_manifest["acceptance_screening"].values())
        benchmark_summary_rows.append(
            f"{case_labels[case_id]} & {_integer(subset.iloc[0].buildings)} & {_integer(subset.iloc[0].demand_zones)} & "
            f"{_integer(subset.iloc[0].building_density_per_km2)} & {' & '.join(route_lengths)} & "
            f"{_integer(case_manifest['counts']['electricity_transformers'])} & {_integer(case_manifest['counts']['district_heating_plants'])} & "
            f"{_integer(subset.loc['drinking_water', 'cycle_rank'])} & {'pass' if ready else 'fail'} & {case_manifest['runtime_seconds']:.1f} \\\\"
        )
        for sector in sector_labels:
            row = subset.loc[sector]
            demand = _one(row.allocated_demand / 1000.0)
            demand_unit = "GWh y$^{-1}$" if row.demand_unit == "MWh/y" else "10$^3$ m$^3$ y$^{-1}$"
            benchmark_sector_rows.append(
                f"{case_labels[case_id]} & {sector_labels[sector]} & {_integer(row.nodes)} & {_integer(row.links)} & "
                f"{_one(row.route_length_km)} & {_integer(row.cycle_rank)} & {_one(row.average_service_route_km)} / {_one(row.maximum_service_route_km)} & "
                f"{row.equipment_size_median:g} / {row.equipment_size_p90:g} / {row.equipment_size_max:g} & {demand} {demand_unit} \\\\"
            )
    _write_table_rows(PAPER / "generated_benchmark_summary_rows.tex", benchmark_summary_rows)
    _write_table_rows(PAPER / "generated_benchmark_sector_rows.tex", benchmark_sector_rows)

    topology_distribution_rows = []
    for case_id in case_order:
        subset = benchmark_metrics[benchmark_metrics["case_id"] == case_id].set_index("sector")
        for sector in sector_labels:
            row = subset.loc[sector]
            topology_distribution_rows.append(
                f"{case_labels[case_id]} & {sector_labels[sector]} & {row.mean_degree:.2f} & "
                f"{100.0 * row.branch_fraction:.1f} / {100.0 * row.leaf_fraction:.1f} & "
                f"{row.edge_length_p50_km:.3f} / {row.edge_length_p90_km:.3f} & "
                f"{row.route_stretch_p50:.2f} / {row.route_stretch_p90:.2f} & "
                f"{row.network_length_per_1000_buildings_km:.2f} & {row.buildings_per_source_mean:.0f} \\\\"
            )
    _write_table_rows(PAPER / "generated_topology_distribution_rows.tex", topology_distribution_rows)

    sensitivity_rows = []
    grades = sorted(sewer_sensitivity_matrix["minimum_gravity_grade"].unique())
    for relief_scale in sorted(sewer_sensitivity_matrix["relief_scale"].unique()):
        subset = sewer_sensitivity_matrix[
            sewer_sensitivity_matrix["relief_scale"] == relief_scale
        ].set_index("minimum_gravity_grade")
        cells = [
            f"{int(subset.loc[grade, 'lift_edge_count'])} / "
            f"{float(subset.loc[grade, 'force_main_length_km']):.2f}"
            for grade in grades
        ]
        sensitivity_rows.append(f"{relief_scale:.2f} & " + " & ".join(cells) + r" \\")
    _write_table_rows(PAPER / "generated_sewer_sensitivity_rows.tex", sensitivity_rows)

    anchor_names = {
        "Electricity LV annual energy": "Electricity annual energy",
        "Electricity LV coincident peak": "Electricity coincident peak",
        "MS/LV withdrawal locations": "MS/LV withdrawal locations",
        "MS/LV installed capacity": "MS/LV installed capacity",
        "Mapped synthetic MV network": "Synthetic MV route",
        "Direct-retail drinking-water allocation": "Direct-retail water allocation",
        "Drinking-water route": "Drinking-water route",
        "Wastewater route inventory": "Wastewater route inventory",
        "Wastewater force mains": "Wastewater force mains",
        "Wastewater pump stations": "Wastewater pump stations",
        "Wastewater overflow basins": "Wastewater overflow basins",
        "District-heat generated route": "District-heat route",
        "District-heat represented sales contracts": "District-heat sales contracts",
        "District-heat design peak assumption": "District-heat design peak",
        "District-heat annual sales": "District-heat annual sales",
    }
    anchor_rows = []
    for row in anchors.itertuples():
        if row.metric not in anchor_names:
            continue
        if row.comparison_tolerance_basis == "relative_percent":
            tolerance = rf"$\pm${row.comparison_tolerance:g}\%"
        elif row.comparison_tolerance_basis == "absolute_count":
            tolerance = rf"$\pm${row.comparison_tolerance:g} count"
        else:
            tolerance = "--"
        anchor_rows.append(
            f"{anchor_names[row.metric]} & {_one(row.generated)} & {_one(row.reference)} & {_escape(row.unit)} & "
            f"{row.relative_error_percent:+.1f} & {row.evidence_role.replace('_', ' ')} & {tolerance} & {str(row.status).replace('_', ' ')} \\\\"
        )
    _write_table_rows(PAPER / "generated_anchor_rows.tex", anchor_rows)

    print("Wrote generated LaTeX result macros and rows")


if __name__ == "__main__":
    main()
