#!/usr/bin/env python3
"""Re-inspect Urban4 native exports and refresh the acceptance manifest.

This module deliberately reads the files consumed by the native solvers.  It
therefore catches stale summaries and parser errors instead of trusting values
cached by the generation routine.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .framework import _screen_solver_results, load_case


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def inspect_electricity(output: Path) -> dict[str, Any]:
    nodes = pd.read_csv(output / "electricity_nodes.csv")
    links = pd.read_csv(output / "electricity_links.csv")
    transformers = pd.read_csv(output / "electricity_transformers.csv")
    graph = nx.Graph()
    graph.add_nodes_from(nodes.bus_id.astype(str))
    graph.add_edges_from(
        links.loc[~links.normally_open.astype(bool), ["from_node", "to_node"]]
        .astype(str).itertuples(index=False, name=None)
    )
    graph.add_edges_from(
        transformers[["hv_bus", "lv_bus"]].astype(str).itertuples(index=False, name=None)
    )
    finite_voltages = pd.to_numeric(nodes.result_vm_pu, errors="coerce").dropna()
    line_loading = pd.to_numeric(links.result_loading_percent, errors="coerce").dropna()
    transformer_loading = pd.to_numeric(
        transformers.result_loading_percent, errors="coerce"
    ).dropna()
    source = str(nodes.loc[nodes.node_type.eq("mv_source"), "bus_id"].iloc[0])
    service_nodes = nodes.loc[
        nodes.node_type.isin(["building_service", "mv_customer"]), "bus_id"
    ].astype(str)
    supplied_component = nx.node_connected_component(graph, source)
    result = {
        "converged": bool(len(finite_voltages) == len(nodes)),
        "minimum_voltage_pu": float(finite_voltages.min()),
        "maximum_voltage_pu": float(finite_voltages.max()),
        "maximum_line_loading_percent": float(line_loading.max()),
        "maximum_transformer_loading_percent": float(transformer_loading.max()),
        "connected_components": int(nx.number_connected_components(graph)),
        "all_service_nodes_reachable_from_source": bool(
            set(service_nodes).issubset(supplied_component)
        ),
        "buses": int(len(nodes)),
        "lines": int(len(links)),
        "transformers": int(len(transformers)),
    }
    _write_json(output / "electricity_solver_results.json", result)
    return result


def inspect_water(output: Path) -> dict[str, Any]:
    import wntr

    path = output / "drinking_water_epanet.inp"
    services = pd.read_csv(output / "drinking_water_service_connections.csv")
    service_ids = services.service_node.astype(str).tolist()

    normal = wntr.network.WaterNetworkModel(str(path))
    pump_curve = normal.get_curve("W_PUMP_CURVE")
    normal_result = wntr.sim.EpanetSimulator(normal).run_sim()
    normal_pressure = normal_result.node["pressure"].iloc[-1].reindex(service_ids)
    normal_velocity = normal_result.link["velocity"].iloc[-1].drop(
        labels=["W_MAIN_PUMP"], errors="ignore"
    ).abs()

    peak = wntr.network.WaterNetworkModel(str(path))
    peak.options.hydraulic.demand_multiplier = 2.0
    peak_result = wntr.sim.EpanetSimulator(peak).run_sim()
    peak_pressure = peak_result.node["pressure"].iloc[-1].reindex(service_ids)
    peak_velocity = peak_result.link["velocity"].iloc[-1].drop(
        labels=["W_MAIN_PUMP"], errors="ignore"
    ).abs()

    nodes = pd.read_csv(output / "drinking_water_nodes.csv")
    links = pd.read_csv(output / "drinking_water_links.csv")
    graph = nx.Graph()
    graph.add_nodes_from(nodes.node_id.astype(str))
    graph.add_edges_from(
        links[["from_node", "to_node"]].astype(str).itertuples(index=False, name=None)
    )
    result = {
        "created": True,
        "converged": bool(
            np.isfinite(normal_pressure).all() and np.isfinite(peak_pressure).all()
        ),
        "building_service_nodes": int(len(service_ids)),
        "minimum_pressure_m": float(normal_pressure.min()),
        "maximum_pressure_m": float(normal_pressure.max()),
        "maximum_velocity_m_s": float(normal_velocity.max()),
        "peak_minimum_pressure_m": float(peak_pressure.min()),
        "peak_maximum_velocity_m_s": float(peak_velocity.max()),
        "source_head_rise_m": float(pump_curve.points[1][1]),
        "connected_components": int(nx.number_connected_components(graph)),
        "cycle_rank": int(
            graph.number_of_edges() - graph.number_of_nodes()
            + nx.number_connected_components(graph)
        ),
    }
    _write_json(output / "drinking_water_solver_results.json", result)
    return result


def _swmm_metric(text: str, label: str, default: float = math.inf) -> float:
    patterns = {
        "continuity": r"Continuity Error \(%\)\s+\.+\s+([-+]?\d+(?:\.\d+)?)",
        "nonconverging": r"% of Steps Not Converging\s*:\s*([-+]?\d+(?:\.\d+)?)",
        "flooding": r"Flooding Loss \.{3,}\s+[-+]?\d+(?:\.\d+)?\s+([-+]?\d+(?:\.\d+)?)",
    }
    match = re.search(patterns[label], text)
    return float(match.group(1)) if match else default


def inspect_wastewater(output: Path) -> dict[str, Any]:
    report = (output / "wastewater_swmm.rpt").read_text(
        encoding="utf-8", errors="ignore"
    )
    sensitivity_path = output / "wastewater_swmm_timestep15.rpt"
    sensitivity = sensitivity_path.read_text(
        encoding="utf-8", errors="ignore"
    ) if sensitivity_path.exists() else ""
    sensitivity_metrics = {
        "flow_routing_continuity_error_percent": _swmm_metric(
            sensitivity, "continuity"
        ),
        "steps_not_converging_percent": _swmm_metric(
            sensitivity, "nonconverging"
        ),
        "flooding_loss_million_liter": _swmm_metric(sensitivity, "flooding"),
    }
    nodes = pd.read_csv(output / "wastewater_nodes.csv")
    links = pd.read_csv(output / "wastewater_links.csv")
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.node_id.astype(str))
    graph.add_edges_from(
        links[["from_node", "to_node"]].astype(str).itertuples(index=False, name=None)
    )
    outfalls = nodes.loc[nodes.node_type.eq("outfall"), "node_id"].astype(str).tolist()
    properties = nodes.loc[
        nodes.node_type.isin(
            ["building_property_connection", "property_lift_station"]
        ), "node_id"
    ].astype(str)
    reaching_outfall = (
        nx.ancestors(graph, outfalls[0]) | {outfalls[0]}
        if len(outfalls) == 1 else set()
    )
    all_reach = bool(len(outfalls) == 1 and set(properties).issubset(reaching_outfall))
    result = {
        "created": True,
        "engine_run_passed": bool("ERROR " not in report),
        "building_property_connections": int(len(properties)),
        "lift_stations": int(nodes.node_type.eq("lift_station").sum()),
        "property_lift_stations": int(
            nodes.node_type.eq("property_lift_station").sum()
        ),
        "flow_routing_continuity_error_percent": _swmm_metric(
            report, "continuity"
        ),
        "steps_not_converging_percent": _swmm_metric(
            report, "nonconverging"
        ),
        "flooding_loss_million_liter": _swmm_metric(report, "flooding"),
        "time_step_sensitivity_15s": sensitivity_metrics,
        "time_step_sensitivity_passed": bool(
            sensitivity
            and "ERROR " not in sensitivity
            and abs(sensitivity_metrics["flow_routing_continuity_error_percent"]) <= 1.0
            and sensitivity_metrics["steps_not_converging_percent"] <= 0.1
            and sensitivity_metrics["flooding_loss_million_liter"] <= 0.001
        ),
        "routing_method": "KINWAVE",
        "all_property_nodes_reach_single_outfall": all_reach,
    }
    _write_json(output / "wastewater_solver_results.json", result)
    return result


def inspect_heat(output: Path) -> dict[str, Any]:
    import pandapipes as ppipe

    net = ppipe.from_json(str(output / "district_heating_pandapipes.json"))
    ppipe.pipeflow(net, mode="hydraulics", max_iter_hyd=100)
    nodes = pd.read_csv(output / "district_heating_nodes.csv")
    corridors = pd.read_csv(output / "district_heating_corridors.csv")
    services = pd.read_csv(output / "district_heating_service_connections.csv")
    config = load_case()
    heat_config = config["district_heating"]
    route_length_km = float(
        corridors.loc[corridors.link_type.eq("route"), "length_km"].sum()
    )
    service_length_km = float(
        corridors.loc[corridors.link_type.eq("building_service"), "length_km"].sum()
    )
    one_way_network_length_km = route_length_km + service_length_km
    published_length_km = float(heat_config["route_length_target_km"])
    length_error = abs(one_way_network_length_km - published_length_km) / max(
        published_length_km, 1e-9
    )
    length_tolerance = float(
        heat_config.get("route_length_tolerance_fraction", 0.10)
    )
    graph = nx.Graph()
    graph.add_nodes_from(nodes.node_id.astype(str))
    graph.add_edges_from(
        corridors[["from_node", "to_node"]].astype(str).itertuples(index=False, name=None)
    )
    plant_nodes = set(nodes.loc[nodes.node_type.eq("plant"), "node_id"].astype(str))
    consumer_nodes = set(
        nodes.loc[nodes.node_type.eq("consumer_substation"), "node_id"].astype(str)
    )
    components = [set(component) for component in nx.connected_components(graph)]
    components_without_plant = sum(not (component & plant_nodes) for component in components)
    consumers_reachable = all(
        not (component & consumer_nodes) or bool(component & plant_nodes)
        for component in components
    )
    return_pressure_bar = float(
        config["bidirectional_coupling"]["heat_return_pressure_bar"]
    )
    annual_heat_loss_mwh = float(25.0 * route_length_km * 1000.0 * 8760.0 / 1e6)
    annual_heat_demand_mwh = float(services.annual_heat_mwh.sum())
    result = {
        "created": True,
        "converged": bool(net.converged),
        "consumer_substations": int(len(services)),
        "minimum_pressure_bar": float(net.res_junction.p_bar.min()),
        "return_pressure_reference_bar": return_pressure_bar,
        "minimum_differential_pressure_bar": float(
            net.res_junction.p_bar.min() - return_pressure_bar
        ),
        "maximum_velocity_m_s": float(net.res_pipe.v_mean_m_per_s.abs().max()),
        "route_trench_length_km": route_length_km,
        "service_connection_length_km": service_length_km,
        "one_way_network_length_km": one_way_network_length_km,
        "published_network_length_km": published_length_km,
        "network_length_scope": heat_config.get(
            "route_length_comparison_scope", "route_plus_service_one_way"
        ),
        "inventory_length_relative_error": length_error,
        "inventory_length_tolerance_fraction": length_tolerance,
        "inventory_length_within_tolerance": bool(length_error <= length_tolerance),
        "paired_physical_pipe_length_km": float(2.0 * corridors.length_km.sum()),
        "annual_heat_loss_mwh": annual_heat_loss_mwh,
        "annual_heat_loss_fraction": annual_heat_loss_mwh
        / max(annual_heat_demand_mwh, 1e-9),
        "connected_components": int(len(components)),
        "plant_boundaries": int(len(plant_nodes)),
        "components_without_plant": int(components_without_plant),
        "all_consumer_substations_reachable_from_a_plant": bool(consumers_reachable),
        "model_scope": "supply hydraulic acceptance proxy; paired supply/return topology exported explicitly",
    }
    _write_json(output / "district_heating_solver_results.json", result)
    return result


def refresh_manifest(output: Path) -> dict[str, Any]:
    config = load_case()
    solvers = {
        "electricity": inspect_electricity(output),
        "drinking_water": inspect_water(output),
        "wastewater": inspect_wastewater(output),
        "district_heating": inspect_heat(output),
    }
    screening = {
        sector: _screen_solver_results(result, sector, config)
        for sector, result in solvers.items()
    }
    manifest_path = output / "integrated_generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["native_solver_results"] = solvers
    manifest["native_acceptance_screening"] = screening
    manifest["all_native_screens_passed"] = bool(
        all(result["passed"] for result in screening.values())
    )
    interfaces_path = output / "coupling_interfaces.csv"
    corridors_path = output / "shared_corridor_ledger.csv"
    if interfaces_path.exists():
        interfaces = pd.read_csv(interfaces_path)
        manifest.setdefault("counts", {})["coupling_interfaces"] = int(len(interfaces))
        manifest["counts"]["electrically_driven_facilities"] = int(
            interfaces.relation.eq("electrically_driven_facility").sum()
        )
    if corridors_path.exists():
        corridors = pd.read_csv(corridors_path)
        manifest.setdefault("counts", {})["shared_route_segments"] = int(
            corridors.is_shared_corridor.astype(bool).sum()
        )
    interface_audit_path = output / "coupling_interface_audit.json"
    if interface_audit_path.exists():
        manifest["coupling_interface_audit"] = json.loads(
            interface_audit_path.read_text(encoding="utf-8")
        )
    corridor_summary_path = output / "shared_corridor_summary.json"
    if corridor_summary_path.exists():
        manifest["shared_corridor_summary"] = json.loads(
            corridor_summary_path.read_text(encoding="utf-8")
        )
    coupled_path = output / "coupled_nominal_interface_manifest.json"
    if coupled_path.exists():
        coupled = json.loads(coupled_path.read_text(encoding="utf-8"))
        manifest["coupled_nominal_interface_closure"] = coupled
        manifest["final_export_gate_passed"] = bool(
            manifest["all_native_screens_passed"] and coupled.get("accepted", False)
        )
    _write_json(manifest_path, manifest)
    return manifest
