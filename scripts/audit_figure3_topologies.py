#!/usr/bin/env python3
"""Audit the data represented in Urban4 Figure 3.

This script reads accepted CSV exports only.  It does not regenerate, resize,
or solve any network.  The checks make the visual claims in Figure 3 explicit:
building heterogeneity, common-zone coverage, graph reachability, and graph
structure for the dense-urban fixed-policy case.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def use_group(value: str) -> str:
    text = str(value).lower()
    if any(token in text for token in ("house", "residential", "apart", "terrace", "detached", "dorm")):
        return "residential"
    if any(token in text for token in ("commercial", "retail", "office", "school", "hospital", "public")):
        return "commercial_or_public"
    if any(token in text for token in ("industrial", "warehouse")):
        return "industrial_or_warehouse"
    if any(token in text for token in ("roof", "garage", "shed", "carport")):
        return "auxiliary"
    return "other_or_unclassified"


def undirected_stats(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> dict:
    node_set = set(nodes)
    edge_list = list(edges)
    dangling = sorted({value for edge in edge_list for value in edge if value not in node_set})
    adjacency = {node: set() for node in node_set}
    for left, right in edge_list:
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    components: list[set[str]] = []
    unseen = set(node_set)
    while unseen:
        start = next(iter(unseen))
        reached = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current] - reached:
                reached.add(neighbour)
                queue.append(neighbour)
        components.append(reached)
        unseen -= reached
    return {
        "nodes": len(node_set),
        "edges": len(edge_list),
        "components": len(components),
        "cycle_rank": len(edge_list) - len(node_set) + len(components),
        "dangling_endpoint_count": len(dangling),
        "component_nodes": components,
        "adjacency": adjacency,
    }


def reachable(adjacency: dict[str, set[str]], sources: Iterable[str]) -> set[str]:
    reached = set(sources)
    queue = deque(reached)
    while queue:
        current = queue.popleft()
        for neighbour in adjacency[current] - reached:
            reached.add(neighbour)
            queue.append(neighbour)
    return reached


def directed_reaches_target(edges: Iterable[tuple[str, str]], starts: Iterable[str], targets: set[str]) -> tuple[int, bool]:
    adjacency: dict[str, list[str]] = {}
    indegree: Counter[str] = Counter()
    nodes: set[str] = set()
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        indegree[right] += 1
        indegree.setdefault(left, 0)
        nodes.update((left, right))

    queue = deque(node for node in nodes if indegree[node] == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for neighbour in adjacency.get(current, []):
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                queue.append(neighbour)
    acyclic = visited == len(nodes)

    successful = 0
    for start in starts:
        seen = {start}
        work = [start]
        found = start in targets
        while work and not found:
            current = work.pop()
            for neighbour in adjacency.get(current, []):
                if neighbour in targets:
                    found = True
                    break
                if neighbour not in seen:
                    seen.add(neighbour)
                    work.append(neighbour)
        successful += int(found)
    return successful, acyclic


def nearest_zone_coverage(zones: pd.DataFrame, nodes: pd.DataFrame, node_type: str) -> dict:
    selected = nodes[nodes["node_type"].eq(node_type)]
    # Local equirectangular distance is sufficient for a sub-city audit.
    lat0 = math.radians(float(zones["lat"].mean()))
    zone_xy = np.column_stack((zones["lon"].to_numpy() * math.cos(lat0), zones["lat"].to_numpy()))
    node_xy = np.column_stack((selected["lon"].to_numpy() * math.cos(lat0), selected["lat"].to_numpy()))
    separations_m = []
    for point in zone_xy:
        squared = np.sum((node_xy - point) ** 2, axis=1)
        separations_m.append(float(np.sqrt(np.min(squared)) * 111_200.0))
    return {
        "zone_count": int(len(zones)),
        "matching_node_count": int(len(selected)),
        "zones_with_matching_node_within_1m": int(np.sum(np.asarray(separations_m) <= 1.0)),
        "maximum_nearest_separation_m": max(separations_m, default=float("nan")),
    }


def node_alignment_to_zones(zones: pd.DataFrame, nodes: pd.DataFrame, node_type: str) -> dict:
    selected = nodes[nodes["node_type"].eq(node_type)]
    lat0 = math.radians(float(zones["lat"].mean()))
    zone_xy = np.column_stack((zones["lon"].to_numpy() * math.cos(lat0), zones["lat"].to_numpy()))
    node_xy = np.column_stack((selected["lon"].to_numpy() * math.cos(lat0), selected["lat"].to_numpy()))
    separations_m = []
    for point in node_xy:
        squared = np.sum((zone_xy - point) ** 2, axis=1)
        separations_m.append(float(np.sqrt(np.min(squared)) * 111_200.0))
    return {
        "selected_node_count": int(len(selected)),
        "selected_nodes_aligned_to_zone_within_1m": int(np.sum(np.asarray(separations_m) <= 1.0)),
        "maximum_nearest_zone_separation_m": max(separations_m, default=float("nan")),
    }


def numeric_summary(frame: pd.DataFrame, column: str) -> dict:
    values = frame[column].astype(float)
    return {
        "minimum": float(values.min()),
        "median": float(values.median()),
        "maximum": float(values.max()),
        "unique_values": int(values.nunique()),
    }


def legacy_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    case = args.case.resolve()

    buildings = pd.read_csv(case / "common_building_ledger.csv")
    zones = pd.read_csv(case / "common_demand_zones.csv")
    electric_nodes = pd.read_csv(case / "electricity_nodes.csv")
    electric_links = pd.read_csv(case / "electricity_links.csv")
    transformers = pd.read_csv(case / "electricity_transformers.csv")
    water_nodes = pd.read_csv(case / "drinking_water_nodes.csv")
    water_links = pd.read_csv(case / "drinking_water_links.csv")
    sewer_nodes = pd.read_csv(case / "wastewater_nodes.csv")
    sewer_links = pd.read_csv(case / "wastewater_links.csv")
    heat_nodes = pd.read_csv(case / "district_heating_nodes.csv")
    heat_links = pd.read_csv(case / "district_heating_links.csv")

    building_groups = buildings["building_type"].map(use_group)
    electric_edges = list(electric_links[["from_node", "to_node"]].itertuples(index=False, name=None))
    electric_edges += list(transformers[["hv_bus", "lv_bus"]].itertuples(index=False, name=None))
    e_stats = undirected_stats(electric_nodes["bus_id"], electric_edges)
    e_roots = set(electric_nodes.loc[electric_nodes["node_type"].eq("upstream_boundary"), "bus_id"])
    e_loads = set(electric_nodes.loc[electric_nodes["node_type"].isin(["load", "transformer_load"]), "bus_id"])
    e_reached = reachable(e_stats["adjacency"], e_roots)
    e_component_root_counts = [len(component & e_roots) for component in e_stats["component_nodes"]]

    w_edges = list(water_links[["from_node", "to_node"]].itertuples(index=False, name=None))
    w_stats = undirected_stats(water_nodes["node_id"], w_edges)
    w_sources = set(water_nodes.loc[water_nodes["node_type"].eq("source"), "node_id"])
    w_demands = set(water_nodes.loc[water_nodes["node_type"].eq("demand"), "node_id"])
    w_reached = reachable(w_stats["adjacency"], w_sources)

    s_edges = list(sewer_links[["from_node", "to_node"]].itertuples(index=False, name=None))
    s_inflows = set(sewer_nodes.loc[sewer_nodes["node_type"].eq("inflow"), "node_id"])
    s_outfalls = set(sewer_nodes.loc[sewer_nodes["node_type"].eq("outfall"), "node_id"])
    s_success, s_acyclic = directed_reaches_target(s_edges, s_inflows, s_outfalls)
    s_stats = undirected_stats(sewer_nodes["node_id"], s_edges)

    h_edges = list(heat_links[["from_node", "to_node"]].itertuples(index=False, name=None))
    h_stats = undirected_stats(heat_nodes["node_id"], h_edges)
    h_plants = set(heat_nodes.loc[heat_nodes["node_type"].eq("plant"), "node_id"])
    h_consumers = set(heat_nodes.loc[heat_nodes["node_type"].eq("consumer_substation"), "node_id"])
    h_reached = reachable(h_stats["adjacency"], h_plants)
    h_component_plant_counts = [len(component & h_plants) for component in h_stats["component_nodes"]]

    report = {
        "scope": "accepted dense-urban fixed-policy CSV exports; no model regeneration or solver rerun",
        "buildings": {
            "count": int(len(buildings)),
            "unique_ids": int(buildings["building_id"].nunique()),
            "raw_building_type_count": int(buildings["building_type"].nunique()),
            "top_raw_types": {str(k): int(v) for k, v in buildings["building_type"].fillna("missing").value_counts().head(15).items()},
            "use_groups": {str(k): int(v) for k, v in building_groups.value_counts().items()},
            "geometry_evidence_classes": {str(k): int(v) for k, v in buildings["geometry_evidence_class"].value_counts().items()},
            "footprint_m2": numeric_summary(buildings, "footprint_m2"),
            "levels": numeric_summary(buildings, "levels"),
            "floor_area_m2": numeric_summary(buildings, "floor_area_m2"),
            "electricity_peak_kw": numeric_summary(buildings, "electricity_peak_kw"),
            "water_m3_year": numeric_summary(buildings, "water_m3_year"),
            "heat_candidate_mwh_year": numeric_summary(buildings, "heat_candidate_mwh_year"),
        },
        "common_zones": {
            "count": int(len(zones)),
            "total_building_count": int(zones["building_count"].sum()),
            "building_count_conserved": bool(int(zones["building_count"].sum()) == len(buildings)),
            "demand_sum_relative_errors": {
                "electricity_peak": float(abs(zones["electricity_peak_mw"].sum() * 1000.0 - buildings["electricity_peak_kw"].sum()) / buildings["electricity_peak_kw"].sum()),
                "electricity_annual": float(abs(zones["electricity_mwh_year"].sum() - buildings["electricity_mwh_year"].sum()) / buildings["electricity_mwh_year"].sum()),
                "water_annual": float(abs(zones["water_m3_year"].sum() - buildings["water_m3_year"].sum()) / buildings["water_m3_year"].sum()),
                "wastewater_annual": float(abs(zones["wastewater_m3_year"].sum() - buildings["wastewater_sanitary_m3_year"].sum()) / buildings["wastewater_sanitary_m3_year"].sum()),
                "heat_annual": float(abs(zones["heat_mwh_year"].sum() - buildings["heat_candidate_mwh_year"].sum()) / buildings["heat_candidate_mwh_year"].sum()),
            },
        },
        "electricity": {
            "node_types": {str(k): int(v) for k, v in electric_nodes["node_type"].value_counts().items()},
            "line_edges": int(len(electric_links)),
            "transformer_edges": int(len(transformers)),
            "graph_components": e_stats["components"],
            "operating_cycle_rank": e_stats["cycle_rank"],
            "dangling_endpoint_count": e_stats["dangling_endpoint_count"],
            "load_nodes": len(e_loads),
            "loads_reachable_from_upstream_boundary": len(e_loads & e_reached),
            "one_upstream_boundary_per_component": bool(all(value == 1 for value in e_component_root_counts)),
            "interpretation": "complete zone-level LV radial forest rooted at independent 20-kV boundary buses; not a synthesized interconnected MV feeder graph",
        },
        "drinking_water": {
            "node_types": {str(k): int(v) for k, v in water_nodes["node_type"].value_counts().items()},
            "graph_components": w_stats["components"],
            "cycle_rank": w_stats["cycle_rank"],
            "dangling_endpoint_count": w_stats["dangling_endpoint_count"],
            "demand_nodes_reachable_from_source": len(w_demands & w_reached),
            "zone_coordinate_coverage": nearest_zone_coverage(zones, water_nodes, "demand"),
        },
        "wastewater": {
            "node_types": {str(k): int(v) for k, v in sewer_nodes["node_type"].value_counts().items()},
            "underlying_components": s_stats["components"],
            "underlying_cycle_rank": s_stats["cycle_rank"],
            "dangling_endpoint_count": s_stats["dangling_endpoint_count"],
            "directed_acyclic": s_acyclic,
            "inflows_reaching_outfall": s_success,
            "zone_coordinate_coverage": nearest_zone_coverage(zones, sewer_nodes, "inflow"),
        },
        "district_heating": {
            "node_types": {str(k): int(v) for k, v in heat_nodes["node_type"].value_counts().items()},
            "route_links": int(len(heat_links)),
            "graph_components": h_stats["components"],
            "cycle_rank": h_stats["cycle_rank"],
            "dangling_endpoint_count": h_stats["dangling_endpoint_count"],
            "consumer_substations_reachable_from_plant": len(h_consumers & h_reached),
            "one_plant_per_component": bool(all(value == 1 for value in h_component_plant_counts)),
            "selected_consumer_zone_alignment": node_alignment_to_zones(zones, heat_nodes, "consumer_substation"),
            "selected_zone_fraction": float(len(h_consumers) / len(zones)),
            "interpretation": "intentional partial heat-service topology under f_H=0.25, not missing all-sector demand-zone coverage",
        },
    }

    checks = {
        "unique_building_ids": buildings["building_id"].nunique() == len(buildings),
        "zone_building_count_conserved": int(zones["building_count"].sum()) == len(buildings),
        "all_zone_demands_conserved": all(
            value <= 1e-12 for value in report["common_zones"]["demand_sum_relative_errors"].values()
        ),
        "electricity_all_zone_loads_reachable": len(e_loads) == len(zones) == len(e_loads & e_reached),
        "electricity_radial_forest": e_stats["cycle_rank"] == 0,
        "electricity_one_boundary_per_component": all(value == 1 for value in e_component_root_counts),
        "water_all_zone_demands_reachable": len(w_demands) == len(zones) == len(w_demands & w_reached),
        "water_connected_and_looped": w_stats["components"] == 1 and w_stats["cycle_rank"] > 0,
        "wastewater_all_inflows_reach_outfall": s_success == len(zones) == len(s_inflows),
        "wastewater_directed_acyclic": s_acyclic,
        "heat_all_selected_consumers_reachable": len(h_consumers) == len(h_consumers & h_reached),
        "heat_three_radial_plant_trees": h_stats["components"] == 3 and h_stats["cycle_rank"] == 0 and all(value == 1 for value in h_component_plant_counts),
        "no_dangling_graph_endpoints": all(
            stats["dangling_endpoint_count"] == 0
            for stats in (e_stats, w_stats, s_stats, h_stats)
        ),
    }
    report["audit_checks"] = {key: bool(value) for key, value in checks.items()}
    report["all_audit_checks_passed"] = bool(all(checks.values()))

    # Remove internal graph objects before serialization (already excluded above).
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["all_audit_checks_passed"]:
        raise SystemExit("Figure 3 topology audit failed; inspect audit_checks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    case = args.case.resolve()

    buildings = pd.read_csv(case / "common_building_ledger.csv")
    zones = pd.read_csv(case / "common_demand_zones.csv")
    electric_nodes = pd.read_csv(case / "electricity_nodes.csv")
    electric_links = pd.read_csv(case / "electricity_links.csv")
    transformers = pd.read_csv(case / "electricity_transformers.csv")
    water_nodes = pd.read_csv(case / "drinking_water_nodes.csv")
    water_links = pd.read_csv(case / "drinking_water_links.csv")
    sewer_nodes = pd.read_csv(case / "wastewater_nodes.csv")
    sewer_links = pd.read_csv(case / "wastewater_links.csv")
    heat_nodes = pd.read_csv(case / "district_heating_nodes.csv")
    heat_links = pd.read_csv(case / "district_heating_corridors.csv")
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))

    service_buildings = set(
        buildings.loc[buildings.service_eligible.astype(bool), "building_id"].astype(str)
    )
    building_groups = buildings["building_type"].map(use_group)

    electric_edges = list(
        electric_links[["from_node", "to_node"]].astype(str)
        .itertuples(index=False, name=None)
    ) + list(
        transformers[["hv_bus", "lv_bus"]].astype(str)
        .itertuples(index=False, name=None)
    )
    e_stats = undirected_stats(electric_nodes.bus_id.astype(str), electric_edges)
    e_sources = set(
        electric_nodes.loc[electric_nodes.node_type.eq("mv_source"), "bus_id"].astype(str)
    )
    e_services = set(
        electric_nodes.loc[
            electric_nodes.node_type.isin(["building_service", "mv_customer"]),
            "bus_id",
        ].astype(str)
    )
    e_reached = reachable(e_stats["adjacency"], e_sources)
    electric_service_buildings = set(
        electric_nodes.loc[
            electric_nodes.node_type.isin(["building_service", "mv_customer"]),
            "building_id",
        ].dropna().astype(str)
    )

    water_edges = list(
        water_links[["from_node", "to_node"]].astype(str)
        .itertuples(index=False, name=None)
    )
    w_stats = undirected_stats(water_nodes.node_id.astype(str), water_edges)
    w_sources = set(
        water_nodes.loc[water_nodes.node_type.eq("reservoir"), "node_id"].astype(str)
    )
    w_services = set(
        water_nodes.loc[water_nodes.node_type.eq("building_service"), "node_id"].astype(str)
    )
    w_reached = reachable(w_stats["adjacency"], w_sources)
    water_service_buildings = set(
        water_nodes.loc[
            water_nodes.node_type.eq("building_service"), "building_id"
        ].dropna().astype(str)
    )

    sewer_edges = list(
        sewer_links[["from_node", "to_node"]].astype(str)
        .itertuples(index=False, name=None)
    )
    sewer_starts = set(
        sewer_nodes.loc[
            sewer_nodes.node_type.isin(
                ["building_property_connection", "property_lift_station"]
            ), "node_id",
        ].astype(str)
    )
    sewer_outfalls = set(
        sewer_nodes.loc[sewer_nodes.node_type.eq("outfall"), "node_id"].astype(str)
    )
    sewer_success, sewer_acyclic = directed_reaches_target(
        sewer_edges, sewer_starts, sewer_outfalls
    )
    s_stats = undirected_stats(sewer_nodes.node_id.astype(str), sewer_edges)
    sewer_service_buildings = set(
        sewer_nodes.loc[
            sewer_nodes.node_type.isin(
                ["building_property_connection", "property_lift_station"]
            ), "building_id",
        ].dropna().astype(str)
    )

    heat_edges = list(
        heat_links[["from_node", "to_node"]].astype(str)
        .itertuples(index=False, name=None)
    )
    h_stats = undirected_stats(heat_nodes.node_id.astype(str), heat_edges)
    heat_plants = set(
        heat_nodes.loc[heat_nodes.node_type.eq("plant"), "node_id"].astype(str)
    )
    heat_services = set(
        heat_nodes.loc[
            heat_nodes.node_type.eq("consumer_substation"), "node_id"
        ].astype(str)
    )
    heat_reached = reachable(h_stats["adjacency"], heat_plants)

    relative_errors = {
        "electricity_peak": float(abs(zones.electricity_peak_mw.sum() * 1000.0 - buildings.electricity_peak_kw.sum()) / buildings.electricity_peak_kw.sum()),
        "electricity_annual": float(abs(zones.electricity_mwh_year.sum() - buildings.electricity_mwh_year.sum()) / buildings.electricity_mwh_year.sum()),
        "water_annual": float(abs(zones.water_m3_year.sum() - buildings.water_m3_year.sum()) / buildings.water_m3_year.sum()),
        "wastewater_annual": float(abs(zones.wastewater_m3_year.sum() - buildings.wastewater_sanitary_m3_year.sum()) / buildings.wastewater_sanitary_m3_year.sum()),
        "heat_annual": float(abs(zones.heat_mwh_year.sum() - buildings.heat_candidate_mwh_year.sum()) / buildings.heat_candidate_mwh_year.sum()),
    }
    checks = {
        "unique_building_ids": buildings.building_id.nunique() == len(buildings),
        "zone_building_count_conserved": int(zones.building_count.sum()) == len(buildings),
        "all_zone_demands_conserved": all(value <= 1e-12 for value in relative_errors.values()),
        "electricity_one_connected_graph": e_stats["components"] == 1,
        "electricity_operating_graph_radial": e_stats["cycle_rank"] == 0,
        "electricity_one_upstream_source": len(e_sources) == 1,
        "electricity_all_services_supplied": e_services.issubset(e_reached),
        "electricity_building_identity_complete": electric_service_buildings == service_buildings,
        "water_connected_and_looped": w_stats["components"] == 1 and w_stats["cycle_rank"] > 0,
        "water_all_services_source_connected": w_services.issubset(w_reached),
        "water_building_identity_complete": water_service_buildings == service_buildings,
        "wastewater_directed_acyclic": sewer_acyclic,
        "wastewater_all_properties_reach_outfall": sewer_success == len(sewer_starts),
        "wastewater_building_identity_complete": sewer_service_buildings == service_buildings,
        "heat_all_substations_reach_a_plant": heat_services.issubset(heat_reached),
        "all_native_solver_screens_pass": all(
            item["passed"] for item in manifest["acceptance_screening"].values()
        ),
        "no_dangling_graph_endpoints": all(
            stats["dangling_endpoint_count"] == 0
            for stats in (e_stats, w_stats, s_stats, h_stats)
        ),
    }
    report = {
        "scope": "accepted service-resolved dense-urban CSV exports; no regeneration or solver rerun",
        "buildings": {
            "evidence_objects": int(len(buildings)),
            "service_eligible": int(len(service_buildings)),
            "raw_building_type_count": int(buildings.building_type.nunique()),
            "use_groups": {
                str(key): int(value) for key, value in building_groups.value_counts().items()
            },
            "floor_area_m2": numeric_summary(buildings, "floor_area_m2"),
        },
        "common_zones": {
            "count": int(len(zones)),
            "demand_sum_relative_errors": relative_errors,
        },
        "electricity": {
            "buses": int(len(electric_nodes)),
            "lines": int(len(electric_links)),
            "transformers": int(len(transformers)),
            "service_buses": int(len(e_services)),
            "components_with_transformers": e_stats["components"],
            "operating_cycle_rank": e_stats["cycle_rank"],
            "minimum_voltage_pu": manifest["solver_readiness"]["pandapower"]["minimum_voltage_pu"],
        },
        "drinking_water": {
            "nodes": int(len(water_nodes)),
            "links": int(len(water_links)),
            "building_laterals": int(len(w_services)),
            "components": w_stats["components"],
            "cycle_rank": w_stats["cycle_rank"],
        },
        "wastewater": {
            "nodes": int(len(sewer_nodes)),
            "links": int(len(sewer_links)),
            "property_connections": int(len(sewer_starts)),
            "inflows_reaching_outfall": int(sewer_success),
            "directed_acyclic": bool(sewer_acyclic),
        },
        "district_heating": {
            "nodes": int(len(heat_nodes)),
            "corridors": int(len(heat_links)),
            "plants": int(len(heat_plants)),
            "consumer_substations": int(len(heat_services)),
            "components": h_stats["components"],
            "cycle_rank": h_stats["cycle_rank"],
        },
        "audit_checks": {key: bool(value) for key, value in checks.items()},
        "all_audit_checks_passed": bool(all(checks.values())),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["all_audit_checks_passed"]:
        raise SystemExit("Figure 3 topology audit failed; inspect audit_checks")


if __name__ == "__main__":
    main()
