#!/usr/bin/env python3
"""Deterministic municipality-scale native network construction and repair.

This module contains the release-oriented algorithms that are deliberately
stricter than the exploratory candidates in :mod:`principle_water_sewer`.
Every generated network follows the same state machine:

``evidence -> sector partition -> route -> catalogue size -> native solve ->
diagnose -> bounded repair -> native re-solve -> release gate``.

The first implementation is the drinking-water branch.  Pressure zones are
created *before* routing and each zone has an explicit head-controlled
hydraulic boundary.  A failed native run is therefore an intermediate design
state, never the released municipality-scale result.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from . import schweinfurt_base as base
from .principle_water_sewer import (
    _hazen_williams_loss_m,
    _nearest_named_facility,
    _weighted_group_access_nodes,
)
from .service_topologies import (
    HEAT_DN,
    LV_CABLES,
    MV_CABLES,
    WATER_MAIN_DN,
    WATER_SERVICE_DN,
    attach_buildings,
    geometry,
    round_up,
)


@dataclass(frozen=True)
class MunicipalWaterConfig:
    """Fixed water-network construction and release policy."""

    access_cell_size_km: float = 0.12
    pressure_zone_height_m: float = 20.0
    zone_route_excursion_penalty: float = 50.0
    target_main_length_km: float = 341.0
    length_tolerance_fraction: float = 0.10
    target_design_velocity_m_s: float = 1.20
    maximum_native_velocity_m_s: float = 2.00
    minimum_release_pressure_m: float = 27.5
    maximum_release_pressure_m: float = 70.0
    target_peak_minimum_pressure_m: float = 35.0
    required_pressure_m: float = 20.0
    average_demand_multiplier: float = 1.0
    peak_demand_multiplier: float = 2.62
    maximum_head_repairs: int = 4
    maximum_pipe_repairs: int = 60
    maximum_second_feeds: int = 6
    maximum_second_feed_new_length_km: float = 3.0
    maximum_total_second_feed_new_length_km: float = 10.0
    source_name: str = "Wasserwerk Schweinfurt"


@dataclass(frozen=True)
class MunicipalWastewaterConfig:
    """Fixed station-scale SWMM construction and release policy."""

    gravity_main_length_km: float = 235.19111298231218
    force_main_length_km: float = 13.102332402191186
    pump_station_count: int = 13
    minimum_equivalent_gravity_grade: float = 0.00265
    minimum_cover_m: float = 1.5
    maximum_cover_m: float = 8.0
    maximum_wet_well_depth_m: float = 15.0
    wet_well_area_m2: float = 25.0
    maximum_continuity_error_percent: float = 1.0
    maximum_nonconverging_steps_percent: float = 0.1
    maximum_flooding_loss_percent: float = 0.001
    initial_duration_hours: int = 24
    maximum_duration_hours: int = 48
    initial_routing_step_seconds: int = 60
    minimum_routing_step_seconds: int = 15
    maximum_repairs: int = 8


def _edge_key(u: tuple[float, float], v: tuple[float, float]) -> frozenset:
    return frozenset((u, v))


def _zone_weight(
    zone: int,
    minimum_elevation_m: float,
    height_m: float,
    penalty: float,
):
    low = minimum_elevation_m + zone * height_m
    high = minimum_elevation_m + (zone + 1) * height_m

    def weight(u: tuple[float, float], v: tuple[float, float], data: dict[str, Any]) -> float:
        midpoint = 0.5 * (base.proxy_elevation_m(u) + base.proxy_elevation_m(v))
        outside = midpoint < low - 1e-9 or midpoint >= high + 1e-9
        return float(data["length_km"]) * (penalty if outside else 1.0)

    return weight


def _route_zone(
    road: nx.Graph,
    root: tuple[float, float],
    targets: Iterable[tuple[float, float]],
    weight,
) -> tuple[nx.Graph, dict[tuple[float, float], list[tuple[float, float]]]]:
    _, paths = nx.single_source_dijkstra(road, root, weight=weight)
    graph = nx.Graph()
    graph.add_node(root)
    retained: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for target in sorted(set(targets)):
        if target not in paths:
            raise RuntimeError(f"Water access node {target!r} is not reachable from zone root")
        path = paths[target]
        retained[target] = path
        base.add_path_edges(graph, road, path)
    return graph, retained


def _add_zone_second_feeds(
    road: nx.Graph,
    zone_graphs: dict[int, nx.Graph],
    zone_roots: dict[int, tuple[float, float]],
    access: pd.DataFrame,
    minimum_elevation_m: float,
    config: MunicipalWaterConfig,
) -> pd.DataFrame:
    """Add a small, bounded set of alternate paths to high-duty access nodes."""
    rows: list[dict[str, Any]] = []
    total_new = 0.0
    upper_inventory = config.target_main_length_km * (1.0 + config.length_tolerance_fraction)
    for zone in sorted(zone_graphs):
        graph = zone_graphs[zone]
        root = zone_roots[zone]
        ranked = access[access.pressure_zone_index.eq(zone)].sort_values(
            ["critical_buildings", "annual_m3", "access_id"],
            ascending=[False, False, True],
        )
        accepted_in_zone = 0
        zone_weight = _zone_weight(
            zone,
            minimum_elevation_m,
            config.pressure_zone_height_m,
            config.zone_route_excursion_penalty,
        )

        def alternate_weight(u, v, data):
            value = zone_weight(u, v, data)
            if graph.has_edge(u, v):
                value *= 200.0
            return value

        for rank, row in enumerate(ranked.itertuples(index=False), 1):
            if len(rows) >= config.maximum_second_feeds or accepted_in_zone >= 2:
                break
            try:
                alternate = nx.shortest_path(road, root, row.access_node, weight=alternate_weight)
            except nx.NetworkXNoPath:
                continue
            new_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
            new_length = 0.0
            for u, v in zip(alternate[:-1], alternate[1:]):
                if not graph.has_edge(u, v):
                    new_edges.append((u, v))
                    new_length += float(road[u][v]["length_km"])
            if new_length < 0.10 or new_length > config.maximum_second_feed_new_length_km:
                continue
            current_length = sum(float(d["length_km"]) for g in zone_graphs.values() for _, _, d in g.edges(data=True))
            if current_length + new_length > upper_inventory + 1e-9:
                continue
            if total_new + new_length > config.maximum_total_second_feed_new_length_km + 1e-9:
                continue
            for u, v in zip(alternate[:-1], alternate[1:]):
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v, **dict(road[u][v]))
            total_new += new_length
            accepted_in_zone += 1
            rows.append(
                {
                    "pressure_zone_id": f"WZ{zone + 1}",
                    "candidate_rank": rank,
                    "access_id": row.access_id,
                    "new_length_km": new_length,
                    "path_length_km": sum(float(road[u][v]["length_km"]) for u, v in zip(alternate[:-1], alternate[1:])),
                    "reason": "bounded alternate feed to a high-duty access node",
                }
            )
    return pd.DataFrame(rows)


def _tree_flows(
    graph: nx.Graph,
    root: tuple[float, float],
    peak_at: dict[tuple[float, float], float],
) -> tuple[dict[frozenset, float], dict[tuple[float, float], tuple[float, float]]]:
    paths = nx.single_source_dijkstra_path(graph, root, weight="length_km")
    parent: dict[tuple[float, float], tuple[float, float]] = {}
    for node, path in paths.items():
        if node != root:
            parent[node] = path[-2]
    order = sorted(paths, key=lambda node: len(paths[node]), reverse=True)
    subtree = {node: float(peak_at.get(node, 0.0)) for node in graph}
    for node in order:
        if node in parent:
            subtree[parent[node]] += subtree[node]
    flows = {
        _edge_key(node, ancestor): max(subtree[node], 2e-4)
        for node, ancestor in parent.items()
    }
    return flows, parent


def _write_water_inp(
    path: Path,
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    boundary_heads: dict[str, float],
    demand_multiplier: float,
    config: MunicipalWaterConfig,
) -> None:
    junctions = nodes[~nodes.node_type.eq("zone_boundary")]
    reservoirs = nodes[nodes.node_type.eq("zone_boundary")]
    lines = [
        "[TITLE]",
        ";; Urban4 municipality-scale pressure-zone distribution model",
        "",
        "[OPTIONS]",
        "UNITS LPS",
        "HEADLOSS H-W",
        "DEMAND MODEL DDA",
        f"DEMAND MULTIPLIER {demand_multiplier:.6f}",
        f"REQUIRED PRESSURE {config.required_pressure_m:.3f}",
        "",
        "[TIMES]",
        "DURATION 0:00",
        "HYDRAULIC TIMESTEP 1:00",
        "REPORT TIMESTEP 1:00",
        "",
        "[JUNCTIONS]",
        ";;ID Elevation Demand",
    ]
    for row in junctions.itertuples():
        lines.append(f"{row.node_id} {float(row.elevation_m):.4f} {float(row.demand_m3s) * 1000.0:.9f}")
    lines += ["", "[RESERVOIRS]", ";;ID Head"]
    for row in reservoirs.itertuples():
        lines.append(f"{row.node_id} {float(boundary_heads[row.pressure_zone_id]):.4f}")
    lines += ["", "[PIPES]", ";;ID Node1 Node2 Length Diameter Roughness MinorLoss Status"]
    for row in links.itertuples():
        lines.append(
            f"{row.link_id} {row.from_node} {row.to_node} "
            f"{max(float(row.length_km) * 1000.0, 0.5):.3f} "
            f"{float(row.diameter_mm):.1f} {float(row.roughness):.1f} 0 OPEN"
        )
    lines += ["", "[COORDINATES]", ";;Node X Y"]
    for row in nodes.itertuples():
        x, y = base.xy_km((float(row.lon), float(row.lat)))
        lines.append(f"{row.node_id} {x * 1000.0:.3f} {y * 1000.0:.3f}")
    lines += ["", "[REPORT]", "STATUS NO", "SUMMARY YES", "NODES NONE", "LINKS NONE", "", "[END]"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_water_native(
    input_path: Path,
    service_nodes: pd.DataFrame,
    links: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    try:
        import wntr
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("WNTR/EPANET is required for municipality-scale water release") from exc
    model = wntr.network.WaterNetworkModel(str(input_path))
    result = wntr.sim.EpanetSimulator(model).run_sim()
    pressure = result.node["pressure"].iloc[-1]
    demand = result.node["demand"].iloc[-1]
    velocity = result.link["velocity"].iloc[-1]
    service = service_nodes[["node_id", "building_id", "pressure_zone_id", "demand_m3s"]].copy()
    service["pressure_m"] = service.node_id.map(pressure)
    service["delivered_m3s"] = service.node_id.map(demand)
    link_result = links[["link_id", "link_type", "pressure_zone_id", "diameter_mm", "length_km"]].copy()
    link_result["velocity_m_s"] = link_result.link_id.map(velocity)
    return service, link_result, {
        "minimum_service_pressure_m": float(service.pressure_m.min()),
        "maximum_service_pressure_m": float(service.pressure_m.max()),
        "maximum_link_velocity_m_s": float(link_result.velocity_m_s.max()),
        "minimum_delivered_demand_fraction": float(
            (service.delivered_m3s / service.demand_m3s.replace(0.0, np.nan)).dropna().min()
        ),
    }


def _next_catalogue_dn(current: float) -> float | None:
    catalogue = [float(value) for value in WATER_MAIN_DN]
    for value in catalogue:
        if value > float(current) + 1e-9:
            return value
    return None


def generate_accepted_municipal_water_network(
    case_dir: Path,
    road: nx.Graph,
    buildings: pd.DataFrame,
    *,
    wholesale_delivery_m3_year: float = 0.0,
    config: MunicipalWaterConfig = MunicipalWaterConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build, natively solve, repair, and release an accepted zonal network."""
    case_dir.mkdir(parents=True, exist_ok=True)
    attached = attach_buildings(buildings[buildings.water_connected.astype(bool)], road)
    attached, access = _weighted_group_access_nodes(
        attached,
        cell_size_km=config.access_cell_size_km,
        elevation_band_m=config.pressure_zone_height_m,
        demand_column="water_m3_year",
    )
    # Resolve rare snap collisions deterministically: the access node owns one
    # pressure zone and every attached building inherits that same zone.
    access_zone = access.set_index("access_node").pressure_zone_index.to_dict()
    attached["pressure_zone_index"] = attached.access_node.map(access_zone).astype(int)
    minimum_elevation = float(attached.ground_elevation_m.min())
    source = _nearest_named_facility(road, kinds={"water_works"}, name=config.source_name)

    zone_graphs: dict[int, nx.Graph] = {}
    zone_roots: dict[int, tuple[float, float]] = {}
    for zone in sorted(access.pressure_zone_index.unique()):
        subset = access[access.pressure_zone_index.eq(zone)]
        root = min(subset.access_node, key=lambda node: (base.distance_km(source, node), node))
        zone_roots[int(zone)] = root
        weight = _zone_weight(
            int(zone), minimum_elevation, config.pressure_zone_height_m,
            config.zone_route_excursion_penalty,
        )
        graph, _ = _route_zone(road, root, subset.access_node, weight)
        zone_graphs[int(zone)] = graph
    second_feeds = _add_zone_second_feeds(
        road, zone_graphs, zone_roots, access, minimum_elevation, config
    )

    node_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    service_rows: list[dict[str, Any]] = []
    main_lookup: dict[tuple[int, tuple[float, float]], str] = {}
    root_main_id: dict[int, str] = {}
    main_paths: dict[int, nx.Graph] = {}
    link_id_by_edge: dict[tuple[int, frozenset], str] = {}
    link_index = 1

    peak_at_by_zone: dict[int, defaultdict] = {
        zone: defaultdict(float) for zone in zone_graphs
    }
    for row in attached.itertuples():
        peak_at_by_zone[int(row.pressure_zone_index)][row.access_node] += float(row.water_service_peak_lps) / 1000.0

    for zone in sorted(zone_graphs):
        zone_id = f"WZ{zone + 1}"
        graph = zone_graphs[zone]
        protected = {zone_roots[zone], *access.loc[access.pressure_zone_index.eq(zone), "access_node"].tolist()}
        compressed = base.compress_undirected(graph, protected)
        main_paths[zone] = compressed
        flows, parent = _tree_flows(compressed, zone_roots[zone], peak_at_by_zone[zone])
        reservoir_id = f"{zone_id}_BOUNDARY"
        node_rows.append({
            "node_id": reservoir_id,
            "node_type": "zone_boundary",
            "building_id": "",
            "lon": zone_roots[zone][0],
            "lat": zone_roots[zone][1],
            "elevation_m": base.proxy_elevation_m(zone_roots[zone]),
            "pressure_zone_id": zone_id,
            "demand_m3s": 0.0,
            "annual_m3": 0.0,
        })
        for index, node in enumerate(sorted(compressed.nodes), 1):
            identifier = f"{zone_id}_M{index:05d}"
            main_lookup[(zone, node)] = identifier
            if node == zone_roots[zone]:
                root_main_id[zone] = identifier
            node_rows.append({
                "node_id": identifier,
                "node_type": "zone_root_junction" if node == zone_roots[zone] else "demand_access_junction" if node in protected else "main_junction",
                "building_id": "",
                "lon": node[0],
                "lat": node[1],
                "elevation_m": base.proxy_elevation_m(node),
                "pressure_zone_id": zone_id,
                "demand_m3s": 0.0,
                "annual_m3": 0.0,
            })
        for u, v, data in sorted(compressed.edges(data=True), key=lambda item: (item[0], item[1])):
            key = _edge_key(u, v)
            design_flow = flows.get(key, max(0.20 * (peak_at_by_zone[zone][u] + peak_at_by_zone[zone][v]), 2e-4))
            required = math.sqrt(4.0 * design_flow / (math.pi * config.target_design_velocity_m_s)) * 1000.0
            diameter = round_up(max(80.0, required), WATER_MAIN_DN)
            link_id = f"W_P{link_index:06d}"
            link_index += 1
            link_id_by_edge[(zone, key)] = link_id
            link_rows.append({
                "link_id": link_id,
                "from_node": main_lookup[(zone, u)],
                "to_node": main_lookup[(zone, v)],
                "link_type": "distribution_main",
                "building_id": "",
                "pressure_zone_id": zone_id,
                "length_km": float(data["length_km"]),
                "diameter_mm": diameter,
                "roughness": 130.0,
                "design_flow_m3s": design_flow,
                "geometry_json": json.dumps(geometry(data, u, v), separators=(",", ":")),
            })
        link_rows.append({
            "link_id": f"{zone_id}_CONTROLLED_BOUNDARY",
            "from_node": reservoir_id,
            "to_node": root_main_id[zone],
            "link_type": "controlled_zone_boundary",
            "building_id": "",
            "pressure_zone_id": zone_id,
            "length_km": 0.001,
            "diameter_mm": 600.0,
            "roughness": 130.0,
            "design_flow_m3s": float(attached.loc[attached.pressure_zone_index.eq(zone), "water_m3_year"].sum()) / (365.0 * 86400.0),
            "geometry_json": json.dumps([list(zone_roots[zone]), list(zone_roots[zone])], separators=(",", ":")),
        })

    for row in attached.sort_values("building_id").itertuples():
        zone = int(row.pressure_zone_index)
        zone_id = f"WZ{zone + 1}"
        suffix = re.sub(r"[^A-Za-z0-9_]", "_", str(row.building_id))
        node_id = f"W_B_{suffix}"
        link_id = f"W_SC_{suffix}"
        demand = float(row.water_m3_year) / (365.0 * 86400.0)
        peak = max(float(row.water_service_peak_lps) / 1000.0, 1e-6)
        required = math.sqrt(4.0 * peak / (math.pi * 1.5)) * 1000.0
        minimum_dn = 40.0 if row.service_class in {"commercial", "public", "industrial"} else 25.0
        diameter = max(minimum_dn, round_up(required, WATER_SERVICE_DN))
        main_id = main_lookup[(zone, row.access_node)]
        length_km = max(base.distance_km((float(row.lon), float(row.lat)), row.access_node), 0.001)
        node_rows.append({
            "node_id": node_id,
            "node_type": "building_service",
            "building_id": row.building_id,
            "lon": row.lon,
            "lat": row.lat,
            "elevation_m": base.proxy_elevation_m((float(row.lon), float(row.lat))),
            "pressure_zone_id": zone_id,
            "demand_m3s": demand,
            "annual_m3": float(row.water_m3_year),
        })
        link_rows.append({
            "link_id": link_id,
            "from_node": main_id,
            "to_node": node_id,
            "link_type": "building_service",
            "building_id": row.building_id,
            "pressure_zone_id": zone_id,
            "length_km": length_km,
            "diameter_mm": diameter,
            "roughness": 130.0,
            "design_flow_m3s": peak,
            "geometry_json": json.dumps([[row.access_node[0], row.access_node[1]], [row.lon, row.lat]], separators=(",", ":")),
        })
        service_rows.append({
            "service_id": link_id,
            "building_id": row.building_id,
            "node_id": node_id,
            "main_node": main_id,
            "pressure_zone_id": zone_id,
            "length_km": length_km,
            "diameter_mm": diameter,
            "annual_m3": float(row.water_m3_year),
            "peak_flow_lps": float(row.water_service_peak_lps),
            "demand_m3s": demand,
        })

    nodes = pd.DataFrame(node_rows)
    links = pd.DataFrame(link_rows)
    services = pd.DataFrame(service_rows)
    main_length = float(links.loc[links.link_type.eq("distribution_main"), "length_km"].sum())
    relative_error = (main_length - config.target_main_length_km) / config.target_main_length_km
    if abs(relative_error) > config.length_tolerance_fraction + 1e-12:
        raise RuntimeError(
            f"Zonal water main inventory {main_length:.3f} km is outside the fixed "
            f"{config.length_tolerance_fraction:.0%} release tolerance"
        )

    boundary_heads = {
        f"WZ{zone + 1}": base.proxy_elevation_m(zone_roots[zone]) + 50.0
        for zone in zone_graphs
    }
    repair_rows: list[dict[str, Any]] = []
    average_service = peak_service = average_links = peak_links = None
    average_metrics = peak_metrics = None

    def solve_pair(iteration: int):
        average_path = case_dir / "drinking_water_municipal_average.inp"
        peak_path = case_dir / "drinking_water_municipal_peak.inp"
        _write_water_inp(average_path, nodes, links, boundary_heads, config.average_demand_multiplier, config)
        _write_water_inp(peak_path, nodes, links, boundary_heads, config.peak_demand_multiplier, config)
        a_s, a_l, a_m = _run_water_native(average_path, services.rename(columns={"node_id": "node_id"}), links)
        p_s, p_l, p_m = _run_water_native(peak_path, services.rename(columns={"node_id": "node_id"}), links)
        return a_s, a_l, a_m, p_s, p_l, p_m

    pipe_repairs = 0
    head_repairs = 0
    pressure_design_margin_m = 0.01
    for head_attempt in range(config.maximum_head_repairs + config.maximum_pipe_repairs + 1):
        average_service, average_links, average_metrics, peak_service, peak_links, peak_metrics = solve_pair(head_attempt)
        changed = False
        for zone_id in sorted(boundary_heads):
            a_zone = average_service[average_service.pressure_zone_id.eq(zone_id)]
            p_zone = peak_service[peak_service.pressure_zone_id.eq(zone_id)]
            lower = (
                config.minimum_release_pressure_m + pressure_design_margin_m
                - float(p_zone.pressure_m.min())
            )
            upper = (
                config.maximum_release_pressure_m - pressure_design_margin_m
                - float(a_zone.pressure_m.max())
            )
            target = config.target_peak_minimum_pressure_m - float(p_zone.pressure_m.min())
            if lower <= upper + 1e-8:
                delta = min(max(target, lower), upper)
                if abs(delta) > 0.02:
                    if head_repairs >= config.maximum_head_repairs * len(boundary_heads):
                        raise RuntimeError("Municipal water head-repair budget exhausted")
                    before = boundary_heads[zone_id]
                    boundary_heads[zone_id] += delta
                    head_repairs += 1
                    repair_rows.append({
                        "iteration": len(repair_rows) + 1,
                        "sector": "drinking_water",
                        "failed_metric": "zone pressure window",
                        "diagnosis": f"{zone_id}: peak minimum {p_zone.pressure_m.min():.3f} m; average maximum {a_zone.pressure_m.max():.3f} m",
                        "repair_action": "adjust head-controlled zone boundary setpoint",
                        "asset_id": f"{zone_id}_BOUNDARY",
                        "before_value": before,
                        "after_value": boundary_heads[zone_id],
                        "threshold_unchanged": True,
                    })
                    changed = True
            else:
                # A head shift cannot close this zone because its hydraulic
                # pressure span is too wide.  Upsize the largest-loss main on
                # the route to the peak critical service before trying again.
                critical = p_zone.loc[p_zone.pressure_m.idxmin()]
                service_row = services.loc[services.node_id.eq(critical.node_id)].iloc[0]
                zone = int(zone_id.replace("WZ", "")) - 1
                main_node_id = str(service_row.main_node)
                inverse = {identifier: node for (z, node), identifier in main_lookup.items() if z == zone}
                target_node = inverse[main_node_id]
                route = nx.shortest_path(main_paths[zone], zone_roots[zone], target_node, weight="length_km")
                candidates: list[tuple[float, str]] = []
                for u, v in zip(route[:-1], route[1:]):
                    link_id = link_id_by_edge[(zone, _edge_key(u, v))]
                    pipe = links.loc[links.link_id.eq(link_id)].iloc[0]
                    loss = _hazen_williams_loss_m(
                        float(pipe.length_km) * 1000.0,
                        float(pipe.design_flow_m3s),
                        float(pipe.diameter_mm) / 1000.0,
                    )
                    candidates.append((loss, link_id))
                candidates.sort(reverse=True)
                upgraded = False
                for _, link_id in candidates:
                    if pipe_repairs >= config.maximum_pipe_repairs:
                        raise RuntimeError("Municipal water pipe-repair budget exhausted")
                    index = links.index[links.link_id.eq(link_id)][0]
                    next_dn = _next_catalogue_dn(float(links.at[index, "diameter_mm"]))
                    if next_dn is None:
                        continue
                    before = float(links.at[index, "diameter_mm"])
                    links.at[index, "diameter_mm"] = next_dn
                    pipe_repairs += 1
                    repair_rows.append({
                        "iteration": len(repair_rows) + 1,
                        "sector": "drinking_water",
                        "failed_metric": "infeasible zone pressure span",
                        "diagnosis": f"{zone_id}: required head shift interval [{lower:.3f}, {upper:.3f}] m is empty",
                        "repair_action": "upsize highest-loss pipe on critical path by one catalogue class",
                        "asset_id": link_id,
                        "before_value": before,
                        "after_value": next_dn,
                        "threshold_unchanged": True,
                    })
                    upgraded = True
                    changed = True
                    break
                if not upgraded:
                    raise RuntimeError(f"No admissible pipe repair remains for {zone_id}")
        if not changed:
            break
    assert average_service is not None and peak_service is not None
    # Final rerun after the last accepted repair.
    average_service, average_links, average_metrics, peak_service, peak_links, peak_metrics = solve_pair(len(repair_rows) + 1)
    pressure_pass = (
        float(peak_service.pressure_m.min()) >= config.minimum_release_pressure_m - 1e-6
        and float(average_service.pressure_m.max()) <= config.maximum_release_pressure_m + 1e-6
    )
    velocity_pass = float(peak_links.velocity_m_s.max()) <= config.maximum_native_velocity_m_s + 1e-6
    if not pressure_pass or not velocity_pass:
        raise RuntimeError(
            "Municipal water release gate failed after bounded repair: "
            f"pressure_pass={pressure_pass}, velocity_pass={velocity_pass}, "
            f"peak_min={peak_service.pressure_m.min():.3f}, "
            f"average_max={average_service.pressure_m.max():.3f}, "
            f"peak_velocity={peak_links.velocity_m_s.max():.3f}"
        )

    zone_rows: list[dict[str, Any]] = []
    for zone in sorted(zone_graphs):
        zone_id = f"WZ{zone + 1}"
        subset = attached[attached.pressure_zone_index.eq(zone)]
        average_flow_lps = float(subset.water_m3_year.sum()) / (365.0 * 86400.0) * 1000.0
        if zone == min(zone_graphs):
            average_flow_lps += float(wholesale_delivery_m3_year) / (365.0 * 86400.0) * 1000.0
        zone_rows.append({
            "pressure_zone_id": zone_id,
            "boundary_id": f"{zone_id}_BOUNDARY",
            "boundary_role": "head-controlled VFD booster/storage hydraulic boundary; synthetic location",
            "lon": zone_roots[zone][0],
            "lat": zone_roots[zone][1],
            "ground_elevation_m": base.proxy_elevation_m(zone_roots[zone]),
            "selected_boundary_head_m": boundary_heads[zone_id],
            "selected_boundary_pressure_m": boundary_heads[zone_id] - base.proxy_elevation_m(zone_roots[zone]),
            "registered_buildings": int(len(subset)),
            "average_boundary_flow_lps": average_flow_lps,
            "design_peak_flow_lps": config.peak_demand_multiplier * average_flow_lps,
            "average_minimum_service_pressure_m": float(average_service.loc[average_service.pressure_zone_id.eq(zone_id), "pressure_m"].min()),
            "average_maximum_service_pressure_m": float(average_service.loc[average_service.pressure_zone_id.eq(zone_id), "pressure_m"].max()),
            "peak_minimum_service_pressure_m": float(peak_service.loc[peak_service.pressure_zone_id.eq(zone_id), "pressure_m"].min()),
            "peak_maximum_service_pressure_m": float(peak_service.loc[peak_service.pressure_zone_id.eq(zone_id), "pressure_m"].max()),
        })
    zones = pd.DataFrame(zone_rows)
    nodes.to_csv(case_dir / "drinking_water_nodes.csv", index=False)
    links.to_csv(case_dir / "drinking_water_links.csv", index=False)
    services.to_csv(case_dir / "drinking_water_service_connections.csv", index=False)
    zones.to_csv(case_dir / "drinking_water_zone_boundaries.csv", index=False)
    second_feeds.to_csv(case_dir / "drinking_water_second_feed_audit.csv", index=False)
    pd.DataFrame(repair_rows).to_csv(case_dir / "drinking_water_repair_log.csv", index=False)
    average_service.to_csv(case_dir / "drinking_water_native_average_service_results.csv", index=False)
    peak_service.to_csv(case_dir / "drinking_water_native_peak_service_results.csv", index=False)
    average_links.to_csv(case_dir / "drinking_water_native_average_link_results.csv", index=False)
    peak_links.to_csv(case_dir / "drinking_water_native_peak_link_results.csv", index=False)
    manifest = {
        "status": "accepted",
        "release_gate_passed": True,
        "method": "zone-first elevation partition, zone-constrained road routing, catalogue sizing, EPANET solve, bounded repair",
        "registered_buildings": int(len(attached)),
        "pressure_zones": int(len(zone_graphs)),
        "main_length_km": main_length,
        "published_main_length_km": config.target_main_length_km,
        "main_length_relative_error": relative_error,
        "cycle_rank": int(sum(g.number_of_edges() - g.number_of_nodes() + 1 for g in zone_graphs.values())),
        "critical_second_feed_paths": int(len(second_feeds)),
        "native_solver": "EPANET through WNTR",
        "native_solver_executed": True,
        "average_minimum_service_pressure_m": float(average_service.pressure_m.min()),
        "average_maximum_service_pressure_m": float(average_service.pressure_m.max()),
        "peak_minimum_service_pressure_m": float(peak_service.pressure_m.min()),
        "peak_maximum_service_pressure_m": float(peak_service.pressure_m.max()),
        "peak_maximum_velocity_m_s": float(peak_links.velocity_m_s.max()),
        "minimum_delivered_demand_fraction": min(
            average_metrics["minimum_delivered_demand_fraction"],
            peak_metrics["minimum_delivered_demand_fraction"],
        ),
        "repair_actions": int(len(repair_rows)),
        "thresholds_relaxed": False,
        "configuration": asdict(config),
        "elevation_evidence": base.terrain_evidence(),
    }
    (case_dir / "drinking_water_native_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return nodes, links, services, manifest


def _hours_to_swmm_clock(hours: int) -> tuple[str, str]:
    days, remainder = divmod(int(hours), 24)
    return f"07/{1 + days:02d}/2026", f"{remainder:02d}:00:00"


def _write_station_swmm(
    path: Path,
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    pumps: pd.DataFrame,
    *,
    duration_hours: int,
    routing_step_seconds: int,
    wet_well_area_m2: float,
) -> None:
    end_date, end_time = _hours_to_swmm_clock(duration_hours)
    lines = [
        "[TITLE]",
        ";; Urban4 municipality-scale wastewater station model",
        ";; 815 sanitary subcatchments -> 13 wet wells -> 13 force mains -> treatment works",
        "",
        "[OPTIONS]",
        "FLOW_UNITS CMS",
        "FLOW_ROUTING DYNWAVE",
        "FORCE_MAIN_EQUATION H-W",
        "START_DATE 07/01/2026",
        "START_TIME 00:00:00",
        "REPORT_START_DATE 07/01/2026",
        "REPORT_START_TIME 00:00:00",
        f"END_DATE {end_date}",
        f"END_TIME {end_time}",
        f"ROUTING_STEP 00:00:{routing_step_seconds:02d}",
        "VARIABLE_STEP 0.75",
        "MINIMUM_STEP 0.1",
        "MAX_TRIALS 100",
        "HEAD_TOLERANCE 0.005",
        "SURCHARGE_METHOD SLOT",
        "",
        "[JUNCTIONS]",
        ";;Name Elevation MaxDepth InitDepth SurDepth Aponded",
    ]
    junctions = nodes[nodes.node_type.isin(["subcatchment_terminal", "force_main_discharge"])]
    for row in junctions.itertuples():
        lines.append(f"{row.node_id} {row.invert_elevation_m:.3f} {row.max_depth_m:.3f} 0 0 0")
    lines += ["", "[STORAGE]", ";;Name Elev MaxDepth InitDepth Shape A1 A2 A0 SurDepth Fevap"]
    for row in nodes[nodes.node_type.eq("station_wet_well")].itertuples():
        lines.append(
            f"{row.node_id} {row.invert_elevation_m:.3f} {row.max_depth_m:.3f} "
            f"0 FUNCTIONAL 0 0 {wet_well_area_m2:.3f} 0 0"
        )
    lines += ["", "[OUTFALLS]", ";;Name Elevation Type StageData Gated"]
    for row in nodes[nodes.node_type.eq("treatment_outfall")].itertuples():
        lines.append(f"{row.node_id} {row.invert_elevation_m:.3f} FREE NO")
    lines += ["", "[CONDUITS]", ";;Name From To Length Roughness InOffset OutOffset InitFlow MaxFlow"]
    for row in links.itertuples():
        # Closed circular links use SWMM's conduit roughness field.  Pump TDH
        # and the separate engineering table retain the H--W C=120 screen.
        roughness = 0.013
        lines.append(
            f"{row.link_id} {row.from_node} {row.to_node} {row.length_km * 1000.0:.3f} "
            f"{roughness:.4f} 0 0 0 0"
        )
    lines += ["", "[PUMPS]", ";;Name From To Curve Status Startup Shutoff"]
    for row in pumps.itertuples():
        lines.append(f"{row.pump_id} {row.from_node} {row.to_node} {row.curve_id} ON 0.25 0.10")
    lines += ["", "[XSECTIONS]", ";;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels"]
    for row in links.itertuples():
        # SWMM represents a closed pressure conduit with a circular section;
        # FORCE_MAIN_EQUATION selects its full-flow friction formulation.
        lines.append(f"{row.link_id} CIRCULAR {row.diameter_mm / 1000.0:.3f} 0 0 0 1")
    lines += ["", "[CURVES]", ";;Name Type X Y"]
    for row in pumps.itertuples():
        # SWMM type-3 curves use head on the x-axis and flow on the y-axis.
        lines.extend(
            [
                f"{row.curve_id} PUMP3 0.0000 {1.50 * row.design_flow_m3s:.8f}",
                f"{row.curve_id} {row.design_head_m:.4f} {row.design_flow_m3s:.8f}",
                f"{row.curve_id} {1.20 * row.design_head_m:.4f} 0.00000000",
            ]
        )
    lines += ["", "[DWF]", ";;Node Constituent Baseline Patterns"]
    for row in nodes[nodes.dry_inflow_m3s.gt(0)].itertuples():
        lines.append(f"{row.node_id} FLOW {row.dry_inflow_m3s:.9f}")
    lines += ["", "[COORDINATES]", ";;Node X Y"]
    for row in nodes.itertuples():
        x, y = base.xy_km((float(row.lon), float(row.lat)))
        lines.append(f"{row.node_id} {x * 1000.0:.3f} {y * 1000.0:.3f}")
    lines += ["", "[REPORT]", "INPUT NO", "CONTROLS NO", "SUBCATCHMENTS NONE", "NODES NONE", "LINKS NONE", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_swmm_report(report: str) -> dict[str, Any]:
    def metric(pattern: str, default: float = math.inf) -> float:
        match = re.search(pattern, report)
        return float(match.group(1)) if match else default

    return {
        "continuity_error_percent": metric(r"Continuity Error \(%\)\s+\.{2,}\s+([-+]?\d+(?:\.\d+)?)"),
        "nonconverging_steps_percent": metric(r"% of Steps Not Converging\s*:\s*([-+]?\d+(?:\.\d+)?)"),
        "flooding_loss_percent": metric(r"Flooding Loss\s+\.{2,}\s+[-+]?\d+(?:\.\d+)?\s+([-+]?\d+(?:\.\d+)?)"),
        "errors": sorted(set(re.findall(r"ERROR\s+\d+[^\n]*", report))),
    }


def generate_accepted_municipal_wastewater_station_model(
    case_dir: Path,
    road: nx.Graph,
    structural_dir: Path,
    *,
    config: MunicipalWastewaterConfig = MunicipalWastewaterConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build and release a 13-station SWMM model from the structural candidate.

    The 6,179-manhole geometry remains the spatial design representation.  For
    municipality-scale native execution, its 815 terminal catchments are
    aggregated into hydraulically equivalent gravity collectors at the 13
    published-count station compounds.  Thus SWMM contains exactly 13 wet
    wells, 13 active pump objects and 13 force mains while preserving the full
    sanitary inflow and the calibrated gravity/force-main inventories.  The
    equipment inventory adds one identical standby pump at each station,
    giving 26 installed physical pump units.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    terminals = pd.read_csv(structural_dir / "wastewater_subcatchments.csv")
    station_seed = pd.read_csv(structural_dir / "wastewater_lift_station_compounds.csv")
    if len(station_seed) != config.pump_station_count:
        raise RuntimeError(
            f"Expected {config.pump_station_count} station compounds, found {len(station_seed)}"
        )
    outfall = _nearest_named_facility(
        road, kinds={"wastewater_plant"}, name="Kläranlage Schweinfurt"
    )
    road_index = base.NodeIndex(road.nodes)
    station_nodes = [road_index.nearest((float(r.lon), float(r.lat))) for r in station_seed.itertuples()]
    route_distances = [
        nx.single_source_dijkstra_path_length(road, node, weight="length_km")
        for node in station_nodes
    ]
    assignments: list[int] = []
    route_lengths: list[float] = []
    for row in terminals.itertuples():
        node = road_index.nearest((float(row.lon), float(row.lat)))
        candidates = [(float(route_distances[index].get(node, math.inf)), index) for index in range(len(station_nodes))]
        distance, station_index = min(candidates)
        if not math.isfinite(distance):
            distance = base.distance_km(node, station_nodes[station_index])
        assignments.append(station_index)
        route_lengths.append(max(distance, 0.05))
    terminals = terminals.copy()
    terminals["station_index"] = assignments
    terminals["unscaled_route_km"] = route_lengths
    # Preserve the calibrated full-city gravity inventory exactly at the
    # station-scale hydraulic-equivalent layer.
    scale = config.gravity_main_length_km / float(terminals.unscaled_route_km.sum())
    terminals["equivalent_gravity_length_km"] = terminals.unscaled_route_km * scale

    weights = np.sqrt(station_seed.transition_count.clip(lower=1).astype(float))
    station_seed = station_seed.copy()
    station_seed["force_main_length_km"] = config.force_main_length_km * weights / weights.sum()
    node_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    pump_rows: list[dict[str, Any]] = []
    outfall_ground = base.proxy_elevation_m(outfall)
    outfall_invert = outfall_ground - 3.0
    node_rows.append({
        "node_id": "S0_OUTFALL", "node_type": "treatment_outfall",
        "lon": outfall[0], "lat": outfall[1], "ground_elevation_m": outfall_ground,
        "invert_elevation_m": outfall_invert, "max_depth_m": 0.0,
        "dry_inflow_m3s": 0.0, "station_id": "S0",
    })
    node_rows.append({
        "node_id": "S0_HEADER", "node_type": "force_main_discharge",
        "lon": outfall[0], "lat": outfall[1] + 2e-6,
        "ground_elevation_m": outfall_ground,
        "invert_elevation_m": outfall_invert + 0.01, "max_depth_m": 20.0,
        "dry_inflow_m3s": 0.0, "station_id": "S0",
    })
    station_inverts: dict[int, float] = {}
    for index, station in enumerate(station_seed.itertuples()):
        group = terminals[terminals.station_index.eq(index)]
        station_ground = base.proxy_elevation_m(station_nodes[index])
        required_invert = station_ground - 4.0
        if len(group):
            shallow_limits = (
                group.ground_elevation_m
                - config.minimum_cover_m
                - config.minimum_equivalent_gravity_grade * group.equivalent_gravity_length_km * 1000.0
            )
            required_invert = min(required_invert, float(shallow_limits.min()))
        if station_ground - required_invert > config.maximum_wet_well_depth_m + 1e-9:
            raise RuntimeError(
                f"Station {station.station_id} requires a {station_ground - required_invert:.2f} m wet well; "
                f"the fixed maximum is {config.maximum_wet_well_depth_m:.2f} m"
            )
        station_inverts[index] = required_invert
        station_id = f"S{index + 1:02d}"
        discharge_id = f"{station_id}_DISCHARGE"
        wet_well_id = f"{station_id}_WETWELL"
        node_rows.extend(
            [
                {
                    "node_id": wet_well_id, "node_type": "station_wet_well",
                    "lon": station_nodes[index][0], "lat": station_nodes[index][1],
                    "ground_elevation_m": station_ground, "invert_elevation_m": required_invert,
                    "max_depth_m": station_ground - required_invert,
                    "dry_inflow_m3s": 0.0, "station_id": station.station_id,
                },
                {
                    "node_id": discharge_id, "node_type": "force_main_discharge",
                    "lon": outfall[0], "lat": outfall[1] + (index + 1) * 1e-6,
                    "ground_elevation_m": outfall_ground, "invert_elevation_m": outfall_invert,
                    "max_depth_m": 20.0, "dry_inflow_m3s": 0.0, "station_id": station.station_id,
                },
            ]
        )
        q_peak = max(float(group.peak_flow_lps.sum()) / 1000.0, 0.0048)
        required_dn = math.sqrt(4.0 * q_peak / (math.pi * 1.0)) * 1000.0
        force_dn = round_up(max(100.0, required_dn), [100, 125, 150, 200, 250, 300, 350, 400, 500])
        force_length = float(station.force_main_length_km)
        velocity = q_peak / (math.pi * (force_dn / 1000.0) ** 2 / 4.0)
        friction = _hazen_williams_loss_m(force_length * 1000.0, q_peak, force_dn / 1000.0, 120.0)
        static = max(0.0, outfall_invert - required_invert)
        head = max(6.0, static + friction + 2.0)
        pump_rows.append({
            "pump_id": f"P_{station_id}", "from_node": wet_well_id,
            "to_node": discharge_id, "curve_id": f"PC_{station_id}",
            "station_id": station.station_id, "design_flow_m3s": q_peak,
            "design_head_m": head, "force_main_dn_mm": force_dn,
            "force_main_velocity_m_s": velocity,
        })
        link_rows.append({
            "link_id": f"FM_{station_id}", "from_node": discharge_id,
            "to_node": "S0_HEADER", "link_type": "force_main",
            "station_id": station.station_id, "length_km": force_length,
            "diameter_mm": force_dn, "design_flow_m3s": q_peak,
            "slope": 0.0,
        })

    link_rows.append({
        "link_id": "S0_TREATMENT_INLET", "from_node": "S0_HEADER",
        "to_node": "S0_OUTFALL", "link_type": "treatment_inlet",
        "station_id": "S0", "length_km": 0.005,
        "diameter_mm": 1200.0,
        "design_flow_m3s": float(terminals.peak_flow_lps.sum()) / 1000.0,
        "slope": 0.002,
    })

    for terminal_index, row in enumerate(terminals.itertuples(), 1):
        station_index = int(row.station_index)
        station_id = f"S{station_index + 1:02d}"
        wet_well_id = f"{station_id}_WETWELL"
        length_km = float(row.equivalent_gravity_length_km)
        invert = station_inverts[station_index] + config.minimum_equivalent_gravity_grade * length_km * 1000.0
        invert = max(invert, float(row.ground_elevation_m) - config.maximum_cover_m)
        cover = float(row.ground_elevation_m) - invert
        if cover < config.minimum_cover_m - 1e-8:
            raise RuntimeError(f"Equivalent collector cover gate failed for {row.subcatchment_id}")
        terminal_id = f"S_SC_{terminal_index:04d}"
        node_rows.append({
            "node_id": terminal_id, "node_type": "subcatchment_terminal",
            "lon": row.lon, "lat": row.lat, "ground_elevation_m": row.ground_elevation_m,
            "invert_elevation_m": invert, "max_depth_m": cover,
            "dry_inflow_m3s": float(row.annual_m3) / (365.0 * 86400.0),
            "station_id": station_seed.iloc[station_index].station_id,
        })
        q_peak = max(float(row.peak_flow_lps) / 1000.0, 0.0002)
        diameter_m = (
            q_peak * 0.013 * 4.0 ** (5.0 / 3.0)
            / (math.pi * math.sqrt(config.minimum_equivalent_gravity_grade))
        ) ** (3.0 / 8.0)
        diameter = round_up(max(200.0, diameter_m * 1000.0), [200, 250, 300, 350, 400, 500, 600, 800, 1000, 1200])
        link_rows.append({
            "link_id": f"GC_{terminal_index:04d}", "from_node": terminal_id,
            "to_node": wet_well_id, "link_type": "equivalent_gravity_collector",
            "station_id": station_seed.iloc[station_index].station_id,
            "length_km": length_km, "diameter_mm": diameter,
            "design_flow_m3s": q_peak, "slope": config.minimum_equivalent_gravity_grade,
        })

    nodes = pd.DataFrame(node_rows)
    links = pd.DataFrame(link_rows)
    pumps = pd.DataFrame(pump_rows)
    if abs(links.loc[links.link_type.eq("equivalent_gravity_collector"), "length_km"].sum() - config.gravity_main_length_km) > 1e-8:
        raise RuntimeError("Equivalent gravity inventory closure failed")
    if abs(links.loc[links.link_type.eq("force_main"), "length_km"].sum() - config.force_main_length_km) > 1e-8:
        raise RuntimeError("Force-main inventory closure failed")

    duration = config.initial_duration_hours
    timestep = config.initial_routing_step_seconds
    wet_well_area = config.wet_well_area_m2
    repair_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    input_path = case_dir / "wastewater_station_municipal.inp"
    report_path = case_dir / "wastewater_station_municipal.rpt"
    binary_path = case_dir / "wastewater_station_municipal.out"
    try:
        from swmm.toolkit import solver as swmm_solver
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("SWMM toolkit is required for municipality-scale wastewater release") from exc
    for attempt in range(config.maximum_repairs + 1):
        _write_station_swmm(
            input_path, nodes, links, pumps,
            duration_hours=duration, routing_step_seconds=timestep,
            wet_well_area_m2=wet_well_area,
        )
        swmm_solver.swmm_run(str(input_path), str(report_path), str(binary_path))
        report = report_path.read_text(encoding="utf-8", errors="ignore")
        metrics = _parse_swmm_report(report)
        pass_gate = (
            not metrics["errors"]
            and abs(metrics["continuity_error_percent"]) <= config.maximum_continuity_error_percent
            and metrics["nonconverging_steps_percent"] <= config.maximum_nonconverging_steps_percent
            and metrics["flooding_loss_percent"] <= config.maximum_flooding_loss_percent
        )
        if pass_gate:
            break
        if metrics["errors"]:
            raise RuntimeError(f"SWMM station model errors: {metrics['errors']}")
        if abs(metrics["continuity_error_percent"]) > config.maximum_continuity_error_percent and duration < config.maximum_duration_hours:
            before = duration
            duration = config.maximum_duration_hours
            action = "extend initialization horizon without changing topology or thresholds"
            asset = "whole_model"
            after = duration
        elif metrics["nonconverging_steps_percent"] > config.maximum_nonconverging_steps_percent and timestep > config.minimum_routing_step_seconds:
            before = timestep
            timestep = max(config.minimum_routing_step_seconds, timestep // 2)
            action = "halve SWMM routing step"
            asset = "whole_model"
            after = timestep
        elif metrics["flooding_loss_percent"] > config.maximum_flooding_loss_percent and wet_well_area < 100.0:
            before = wet_well_area
            wet_well_area *= 2.0
            action = "advance all station wet wells to the next bounded storage class"
            asset = "S01-S13"
            after = wet_well_area
        else:
            raise RuntimeError(f"No admissible SWMM repair remains: {metrics}")
        repair_rows.append({
            "iteration": len(repair_rows) + 1,
            "sector": "wastewater",
            "failed_metric": json.dumps(metrics, sort_keys=True),
            "repair_action": action,
            "asset_id": asset,
            "before_value": before,
            "after_value": after,
            "threshold_unchanged": True,
        })
    else:
        raise RuntimeError("SWMM station-model repair budget exhausted")

    nodes.to_csv(case_dir / "wastewater_station_native_nodes.csv", index=False)
    links.to_csv(case_dir / "wastewater_station_native_links.csv", index=False)
    pumps.to_csv(case_dir / "wastewater_station_native_pumps.csv", index=False)
    terminals.to_csv(case_dir / "wastewater_station_subcatchment_mapping.csv", index=False)
    pd.DataFrame(repair_rows).to_csv(case_dir / "wastewater_station_repair_log.csv", index=False)
    manifest = {
        "status": "accepted",
        "release_gate_passed": True,
        "method": "815 evidence-conditioned sanitary subcatchments aggregated to 13 wet-well/active-pump/force-main hydraulic systems with duty/standby equipment inventory",
        "native_solver": "EPA SWMM through swmm-toolkit",
        "native_solver_executed": True,
        "subcatchment_terminals": int(len(terminals)),
        "station_wet_wells": int(nodes.node_type.eq("station_wet_well").sum()),
        "active_swmm_pump_objects": int(len(pumps)),
        "installed_pump_units": int(2 * len(pumps)),
        "standby_pump_units": int(len(pumps)),
        "force_mains": int(links.link_type.eq("force_main").sum()),
        "gravity_equivalent_length_km": float(links.loc[links.link_type.eq("equivalent_gravity_collector"), "length_km"].sum()),
        "force_main_length_km": float(links.loc[links.link_type.eq("force_main"), "length_km"].sum()),
        "sanitary_inflow_m3_year": float(terminals.annual_m3.sum()),
        "continuity_error_percent": metrics["continuity_error_percent"],
        "nonconverging_steps_percent": metrics["nonconverging_steps_percent"],
        "flooding_loss_percent": metrics["flooding_loss_percent"],
        "simulation_duration_hours": duration,
        "routing_step_seconds": timestep,
        "wet_well_area_m2": wet_well_area,
        "repair_actions": int(len(repair_rows)),
        "thresholds_relaxed": False,
        "structural_layer": {
            "manhole_graph": "outputs/water_sewer_principle_candidate/wastewater_nodes.csv",
            "collector_graph": "outputs/water_sewer_principle_candidate/wastewater_links.csv",
            "relationship": "station-scale SWMM hydraulic equivalent of the retained manhole-first spatial design",
        },
        "configuration": asdict(config),
        "elevation_evidence": base.terrain_evidence(),
    }
    (case_dir / "wastewater_station_native_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return nodes, links, pumps, manifest


def generate_accepted_municipal_heat_network(
    case_dir: Path,
    corridors: pd.DataFrame,
    services: pd.DataFrame,
    sources: pd.DataFrame,
    *,
    slack_pressure_bar: float = 10.0,
    maximum_velocity_m_s: float = 1.5,
    maximum_pressure_gradient_pa_m: float = 100.0,
    maximum_source_pump_lift_bar: float = 10.0,
    maximum_repair_rounds: int = 6,
    maximum_pipe_repairs: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute and, if needed, catalogue-repair the municipal heat network.

    The pandapipes model is a one-way supply hydraulic proxy.  Absolute slack
    pressure is only a numerical reference; release is based on convergence,
    velocity, pressure-gradient and source-pump differential-pressure screens.
    """
    try:
        import pandapipes as pp
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("pandapipes is required for municipality-scale heat release") from exc
    frame = corridors.copy()
    repairs: list[dict[str, Any]] = []
    final_net = None
    final_result = None
    for attempt in range(maximum_repair_rounds + 1):
        coordinates: set[tuple[float, float]] = set()
        for row in frame.itertuples():
            coordinates.add((float(row.from_lon), float(row.from_lat)))
            coordinates.add((float(row.to_lon), float(row.to_lat)))
        ordered_coordinates = sorted(coordinates)
        net = pp.create_empty_network(fluid="water")
        mapping = {
            point: pp.create_junction(
                net,
                pn_bar=25.0,
                tfluid_k=363.15,
                height_m=base.proxy_elevation_m(point),
                name=f"H_J{index:05d}",
                geodata=point,
            )
            for index, point in enumerate(ordered_coordinates, 1)
        }
        for row in frame.itertuples():
            pp.create_pipe_from_parameters(
                net,
                mapping[(float(row.from_lon), float(row.from_lat))],
                mapping[(float(row.to_lon), float(row.to_lat))],
                length_km=max(float(row.length_km), 0.001),
                inner_diameter_mm=float(row.diameter_mm),
                k_mm=0.1,
                name=str(row.corridor_id),
            )
        for row in sources.itertuples():
            point = min(
                ordered_coordinates,
                key=lambda candidate: base.distance_km(candidate, (float(row.lon), float(row.lat))),
            )
            pp.create_ext_grid(
                net, mapping[point], p_bar=slack_pressure_bar,
                t_k=363.15, name=str(row.map_id), type="pt",
            )
        aggregated = services.groupby(["road_lon", "road_lat"], sort=True).design_mass_flow_kg_s.sum()
        for index, ((lon, lat), mass_flow) in enumerate(aggregated.items(), 1):
            point = (float(lon), float(lat))
            if point not in mapping:
                point = min(ordered_coordinates, key=lambda candidate: base.distance_km(candidate, point))
            pp.create_sink(net, mapping[point], mdot_kg_per_s=float(mass_flow), name=f"H_LOAD_{index:04d}")
        pp.pipeflow(net, mode="hydraulics", max_iter_hyd=100)
        result = frame[["corridor_id", "diameter_mm", "length_km"]].copy().reset_index(drop=True)
        result["native_velocity_m_s"] = net.res_pipe.v_mean_m_per_s.abs().to_numpy()
        diameter_m = result.diameter_mm / 1000.0
        density = 985.0
        result["catalogue_pressure_gradient_pa_m"] = (
            0.02 * density * result.native_velocity_m_s**2 / (2.0 * diameter_m)
        )
        velocity_ok = float(result.native_velocity_m_s.max()) <= maximum_velocity_m_s + 1e-9
        gradient_ok = float(result.catalogue_pressure_gradient_pa_m.max()) <= maximum_pressure_gradient_pa_m + 1e-9
        if velocity_ok and gradient_ok:
            final_net = net
            final_result = result
            break
        if attempt >= maximum_repair_rounds:
            raise RuntimeError("Municipal heat repair-round budget exhausted")
        violation_mask = (
            result.native_velocity_m_s.gt(maximum_velocity_m_s + 1e-9)
            | result.catalogue_pressure_gradient_pa_m.gt(maximum_pressure_gradient_pa_m + 1e-9)
        )
        violating = result[violation_mask].sort_values("corridor_id")
        if len(repairs) + len(violating) > maximum_pipe_repairs:
            raise RuntimeError("Municipal heat pipe-repair budget exhausted")
        for row in violating.itertuples():
            corridor_id = str(row.corridor_id)
            frame_index = frame.index[frame.corridor_id.eq(corridor_id)][0]
            before = float(frame.at[frame_index, "diameter_mm"])
            candidates = [float(dn) for dn in HEAT_DN if float(dn) > before + 1e-9]
            if not candidates:
                raise RuntimeError(f"No larger district-heating catalogue diameter remains for {corridor_id}")
            after = candidates[0]
            frame.at[frame_index, "diameter_mm"] = after
            repairs.append({
                "iteration": len(repairs) + 1,
                "repair_round": attempt + 1,
                "sector": "district_heating",
                "failed_metric": "native velocity or catalogue pressure gradient",
                "diagnosis": f"velocity={row.native_velocity_m_s:.6f} m/s; gradient={row.catalogue_pressure_gradient_pa_m:.6f} Pa/m",
                "repair_action": "upsize violating corridor by one catalogue class",
                "asset_id": corridor_id,
                "before_value": before,
                "after_value": after,
                "threshold_unchanged": True,
            })
    assert final_net is not None and final_result is not None
    maximum_screening_dp = float(sources.screening_system_dp_bar.max())
    source_screen_pass = maximum_screening_dp <= maximum_source_pump_lift_bar + 1e-9
    if not source_screen_pass:
        raise RuntimeError("Municipal district-heating source-pump differential-pressure gate failed")
    route_length_km = float(frame.length_km.sum())
    annual_heat_loss_mwh = 25.0 * route_length_km * 1000.0 * 8760.0 / 1e6
    annual_heat_mwh = float(services.annual_heat_mwh.sum())
    manifest = {
        "status": "accepted",
        "release_gate_passed": True,
        "method": "two documented injection boundaries, route-aware catalogue network, pandapipes supply solve, bounded diameter repair",
        "native_solver": "pandapipes",
        "native_solver_executed": True,
        "converged": bool(final_net.converged),
        "consumer_substations": int(len(services)),
        "maximum_native_velocity_m_s": float(final_result.native_velocity_m_s.max()),
        "maximum_catalogue_pressure_gradient_pa_m": float(final_result.catalogue_pressure_gradient_pa_m.max()),
        "maximum_source_screening_differential_pressure_bar": maximum_screening_dp,
        "maximum_source_to_junction_pressure_drop_bar": float(
            slack_pressure_bar - final_net.res_junction.p_bar.min()
        ),
        "slack_pressure_bar_not_interpreted_as_municipal_absolute_pressure": slack_pressure_bar,
        "annual_heat_loss_mwh": annual_heat_loss_mwh,
        "annual_heat_loss_fraction": annual_heat_loss_mwh / max(annual_heat_mwh, 1e-9),
        "route_trench_length_km": route_length_km,
        "repair_actions": int(len(repairs)),
        "thresholds_relaxed": False,
        "release_limits": {
            "maximum_velocity_m_s": maximum_velocity_m_s,
            "maximum_pressure_gradient_pa_m": maximum_pressure_gradient_pa_m,
            "maximum_source_pump_lift_bar": maximum_source_pump_lift_bar,
            "maximum_repair_rounds": maximum_repair_rounds,
            "maximum_pipe_repairs": maximum_pipe_repairs,
        },
    }
    frame = frame.merge(
        final_result[["corridor_id", "native_velocity_m_s", "catalogue_pressure_gradient_pa_m"]],
        on="corridor_id", how="left",
    )
    frame.to_csv(case_dir / "district_heating_corridors_municipal.csv", index=False)
    final_result.to_csv(case_dir / "district_heating_native_pipe_results.csv", index=False)
    pd.DataFrame(repairs).to_csv(case_dir / "district_heating_repair_log.csv", index=False)
    pp.to_json(final_net, str(case_dir / "district_heating_municipal_pandapipes.json"))
    (case_dir / "district_heating_native_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return frame, manifest


def generate_accepted_municipal_electricity_network(
    case_dir: Path,
    base_network_json: Path,
    transformers: pd.DataFrame,
    links: pd.DataFrame,
    *,
    minimum_voltage_pu: float = 0.90,
    maximum_voltage_pu: float = 1.10,
    maximum_lv_path_drop_pu: float = 0.055,
    maximum_loading_percent: float = 100.0,
    maximum_parallel_feeder_circuits: int = 8,
    maximum_repair_rounds: int = 8,
    maximum_line_repairs: int = 500,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the municipal transformer portfolio and release an AC solution."""
    try:
        import pandapower as pp
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("pandapower is required for municipality-scale electricity release") from exc
    net = pp.from_json(str(base_network_json))
    transformer_capacity = transformers.set_index("transformer_id").municipal_selected_capacity_mva
    net.trafo["sn_mva"] = [float(transformer_capacity.loc[name]) for name in net.trafo.name]
    frame = links.copy()
    frame_index = frame.set_index("line_id", drop=False)
    repairs: list[dict[str, Any]] = []
    final_drop = math.inf

    def lv_graph_and_drops():
        graph = nx.Graph()
        edge_index: dict[frozenset, int] = {}
        for line_index, row in net.line[net.line.in_service.astype(bool)].iterrows():
            u, v = int(row.from_bus), int(row.to_bus)
            if float(net.bus.vn_kv.at[u]) < 1.0 and float(net.bus.vn_kv.at[v]) < 1.0:
                graph.add_edge(u, v)
                edge_index[frozenset((u, v))] = int(line_index)
        rows = []
        for transformer in net.trafo[net.trafo.in_service.astype(bool)].itertuples():
            root = int(transformer.lv_bus)
            if root not in graph:
                continue
            component = nx.node_connected_component(graph, root)
            target = int(net.res_bus.vm_pu.loc[list(component)].idxmin())
            rows.append({
                "transformer_id": transformer.name,
                "root": root,
                "target": target,
                "drop_pu": float(net.res_bus.vm_pu.at[root] - net.res_bus.vm_pu.at[target]),
            })
        return graph, edge_index, pd.DataFrame(rows)

    for repair_round in range(maximum_repair_rounds + 1):
        pp.runpp(net, algorithm="nr", max_iteration=100, numba=False)
        graph, edge_index, drops = lv_graph_and_drops()
        final_drop = float(drops.drop_pu.max())
        voltage_ok = (
            float(net.res_bus.vm_pu.min()) >= minimum_voltage_pu - 1e-9
            and float(net.res_bus.vm_pu.max()) <= maximum_voltage_pu + 1e-9
        )
        loading_ok = (
            float(net.res_line.loading_percent.max()) <= maximum_loading_percent + 1e-9
            and float(net.res_trafo.loading_percent.max()) <= maximum_loading_percent + 1e-9
        )
        drop_ok = final_drop <= maximum_lv_path_drop_pu + 1e-9
        if voltage_ok and loading_ok and drop_ok:
            break
        if repair_round >= maximum_repair_rounds:
            raise RuntimeError("Municipal electricity repair-round budget exhausted")
        limiting_lines: set[int] = set()
        for row in drops[drops.drop_pu.gt(maximum_lv_path_drop_pu + 1e-9)].itertuples():
            path = nx.shortest_path(graph, int(row.root), int(row.target))
            path_lines = [edge_index[frozenset((u, v))] for u, v in zip(path[:-1], path[1:])]
            limiting_lines.add(max(
                path_lines,
                key=lambda index: abs(
                    float(net.res_bus.vm_pu.at[int(net.line.from_bus.at[index])])
                    - float(net.res_bus.vm_pu.at[int(net.line.to_bus.at[index])])
                ),
            ))
        if float(net.res_bus.vm_pu.min()) < minimum_voltage_pu - 1e-9:
            target = int(net.res_bus.vm_pu.idxmin())
            for row in drops.itertuples():
                if target in nx.node_connected_component(graph, int(row.root)):
                    path = nx.shortest_path(graph, int(row.root), target)
                    path_lines = [edge_index[frozenset((u, v))] for u, v in zip(path[:-1], path[1:])]
                    limiting_lines.add(max(
                        path_lines,
                        key=lambda index: abs(
                            float(net.res_bus.vm_pu.at[int(net.line.from_bus.at[index])])
                            - float(net.res_bus.vm_pu.at[int(net.line.to_bus.at[index])])
                        ),
                    ))
                    break
        overloaded = net.res_line.index[net.res_line.loading_percent.gt(maximum_loading_percent + 1e-9)]
        limiting_lines.update(int(index) for index in overloaded)
        if not limiting_lines:
            raise RuntimeError("No admissible municipal electricity repair target found")
        if len(repairs) + len(limiting_lines) > maximum_line_repairs:
            raise RuntimeError("Municipal electricity line-repair budget exhausted")
        for line_index in sorted(limiting_lines, key=lambda index: str(net.line.name.at[index])):
            line_id = str(net.line.name.at[line_index])
            record = frame_index.loc[line_id]
            level = str(record.voltage_level)
            catalogue = LV_CABLES if level == "LV" else MV_CABLES
            current_size = float(record.cable_size_mm2)
            next_items = [item for item in catalogue if float(item[0]) > current_size + 1e-9]
            if next_items:
                size, ampacity_ka, resistance = next_items[0]
                before = current_size
                after = float(size)
                action = "advance limiting line to next conductor class"
                net.line.at[line_index, "r_ohm_per_km"] = float(resistance)
                net.line.at[line_index, "max_i_ka"] = float(ampacity_ka)
                frame_index.at[line_id, "cable_size_mm2"] = float(size)
                frame_index.at[line_id, "r_ohm_per_km"] = float(resistance)
                frame_index.at[line_id, "max_i_ka"] = float(ampacity_ka)
            else:
                current_parallel = int(net.line.parallel.at[line_index])
                if current_parallel >= maximum_parallel_feeder_circuits:
                    raise RuntimeError(f"No admissible cable repair remains for {line_id}")
                before = current_parallel
                after = current_parallel + 1
                action = "add one separately protected parallel feeder circuit"
                net.line.at[line_index, "parallel"] = after
                frame_index.at[line_id, "parallel_circuits"] = after
            repairs.append({
                "iteration": len(repairs) + 1,
                "repair_round": repair_round + 1,
                "sector": "electricity",
                "failed_metric": "voltage, LV path drop, or loading",
                "repair_action": action,
                "asset_id": line_id,
                "before_value": before,
                "after_value": after,
                "threshold_unchanged": True,
            })
    else:
        raise RuntimeError("Municipal electricity repair failed")

    pp.runpp(net, algorithm="nr", max_iteration=100, numba=False)
    _, _, drops = lv_graph_and_drops()
    final_drop = float(drops.drop_pu.max())
    final_links = frame_index.reset_index(drop=True)
    loading_by_name = pd.Series(net.res_line.loading_percent.to_numpy(), index=net.line.name)
    final_links["result_loading_percent"] = final_links.line_id.map(loading_by_name)
    final_links["municipal_circuit_groups"] = np.where(
        final_links.voltage_level.eq("LV"),
        np.ceil(final_links.parallel_circuits / 2).astype(int), 1,
    )
    final_links["conductors_per_group"] = np.where(
        final_links.voltage_level.eq("LV"),
        np.minimum(final_links.parallel_circuits, 2), final_links.parallel_circuits,
    )
    manifest = {
        "status": "accepted",
        "release_gate_passed": True,
        "method": "municipal transformer portfolio, pandapower AC solve, bounded conductor/parallel-circuit repair",
        "native_solver": "pandapower",
        "native_solver_executed": True,
        "converged": bool(net.converged),
        "minimum_voltage_pu": float(net.res_bus.vm_pu.min()),
        "maximum_voltage_pu": float(net.res_bus.vm_pu.max()),
        "maximum_line_loading_percent": float(net.res_line.loading_percent.max()),
        "maximum_transformer_loading_percent": float(net.res_trafo.loading_percent.max()),
        "maximum_lv_path_drop_pu": final_drop,
        "repair_actions": int(len(repairs)),
        "thresholds_relaxed": False,
        "release_limits": {
            "minimum_voltage_pu": minimum_voltage_pu,
            "maximum_voltage_pu": maximum_voltage_pu,
            "maximum_lv_path_drop_pu": maximum_lv_path_drop_pu,
            "maximum_loading_percent": maximum_loading_percent,
            "maximum_parallel_feeder_circuits": maximum_parallel_feeder_circuits,
            "maximum_repair_rounds": maximum_repair_rounds,
            "maximum_line_repairs": maximum_line_repairs,
        },
    }
    final_links.to_csv(case_dir / "electricity_links_municipal.csv", index=False)
    pd.DataFrame(repairs).to_csv(case_dir / "electricity_repair_log.csv", index=False)
    pp.to_json(net, str(case_dir / "electricity_municipal_pandapower.json"))
    (case_dir / "electricity_native_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return final_links, manifest
