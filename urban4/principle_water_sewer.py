#!/usr/bin/env python3
"""Principle-aligned drinking-water and wastewater topology candidates.

The original Urban4 service-resolved fallback attached every building directly
as a main-network terminal on one common road graph.  That representation was
solver-ready, but it made water and sewer main footprints almost identical by
construction.  This module retains the immutable building ledger while moving
main-network decisions to sector-appropriate aggregation layers:

* drinking water: elevation-bounded pressure zones, demand access nodes,
  backbone routing, and explicit second-feed paths for high-demand access nodes;
* wastewater: terrain-aware subcatchment outlets, manhole-first collection,
  hard grade/cover feasibility, and explicit lift/force-main segments.

Published route/manhole inventories are declared calibration inputs for the
full-city candidate.  They are not reused as independent validation metrics.
This module exports the reproducible intermediate spatial layers.  The final
municipality-scale release is constructed and executed by
``urban4.municipal_native``: its water model is solved in EPANET/WNTR and its
wastewater layer is consolidated into exactly 13 station--force-main systems
and solved in SWMM.  Thus an intermediate structural export is never presented
as the framework's released network.
"""

from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from . import schweinfurt_base as base
from .service_topologies import (
    SEWER_MAIN_DN,
    WATER_MAIN_DN,
    WATER_SERVICE_DN,
    attach_buildings,
    geometry,
    round_up,
)


@dataclass(frozen=True)
class WaterCandidateConfig:
    access_cell_size_km: float = 0.12
    pressure_zone_height_m: float = 20.0
    target_main_length_km: float = 341.0
    length_tolerance_fraction: float = 0.10
    critical_access_candidates: int = 180
    path_penalty_factor: float = 200.0
    maximum_new_second_feed_km: float = 8.0
    minimum_new_second_feed_km: float = 0.10
    target_velocity_m_s: float = 1.2
    maximum_velocity_m_s: float = 2.0
    source_head_rise_m: float = 105.21012163162229
    source_name: str = "Wasserwerk Schweinfurt"


@dataclass(frozen=True)
class SewerCandidateConfig:
    subcatchment_cell_size_km: float = 0.21
    target_total_main_length_km: float = 249.0
    target_force_main_length_km: float = 13.0
    target_manhole_count: int = 6200
    target_pump_station_count: int = 13
    length_tolerance_fraction: float = 0.10
    force_main_length_tolerance_fraction: float = 0.15
    manhole_spacing_candidates_m: tuple[float, ...] = (
        40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0
    )
    terrain_rise_penalty: float = 0.45
    minimum_cover_m: float = 1.5
    maximum_cover_m: float = 8.0
    # The grade is selected from this declared structural sweep so that the
    # generated force-main scale is consistent with the published 13 km.
    # It is not presented as a surveyed local design standard.
    gravity_grade_candidates: tuple[float, ...] = (
        0.0015, 0.00175, 0.0020, 0.00225, 0.0025, 0.0026,
        0.00265, 0.0027, 0.00275, 0.00285, 0.0029, 0.0030
    )
    manning_n: float = 0.013
    outfall_name: str = "Kläranlage Schweinfurt"


def _path_edges(path: Iterable[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    points = list(path)
    return list(zip(points[:-1], points[1:]))


def _edge_key(u: tuple[float, float], v: tuple[float, float]) -> frozenset[tuple[float, float]]:
    return frozenset((u, v))


def _polyline_length_km(points: list[tuple[float, float]]) -> float:
    return sum(base.distance_km(a, b) for a, b in zip(points[:-1], points[1:]))


def _nearest_named_facility(
    road: nx.Graph,
    *,
    kinds: set[str],
    name: str,
) -> tuple[float, float]:
    facilities = base.named_facilities(base.load_elements("infrastructure.json"), kinds)
    exact = [item for item in facilities if item["name"] == name]
    if not exact:
        raise ValueError(f"Required mapped facility {name!r} not found")
    return base.NodeIndex(road.nodes).nearest(exact[0]["point"])


def _weighted_group_access_nodes(
    attached: pd.DataFrame,
    *,
    cell_size_km: float,
    elevation_band_m: float,
    demand_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return building-to-access mapping and aggregated water access nodes."""
    frame = attached.copy()
    xy = np.asarray([base.xy_km(node) for node in frame["road_node"]])
    frame["x_km"] = xy[:, 0]
    frame["y_km"] = xy[:, 1]
    frame["ground_elevation_m"] = frame["road_node"].map(base.proxy_elevation_m)
    minimum_elevation = float(frame["ground_elevation_m"].min())
    frame["pressure_zone_index"] = np.floor(
        (frame["ground_elevation_m"] - minimum_elevation) / elevation_band_m
    ).astype(int)
    frame["access_cell_x"] = np.floor(frame["x_km"] / cell_size_km).astype(int)
    frame["access_cell_y"] = np.floor(frame["y_km"] / cell_size_km).astype(int)

    rows: list[dict[str, Any]] = []
    mapping: dict[tuple[int, int, int], tuple[float, float]] = {}
    group_columns = ["access_cell_x", "access_cell_y", "pressure_zone_index"]
    for key, group in frame.groupby(group_columns, sort=True):
        weights = group[demand_column].astype(float).clip(lower=1e-9).to_numpy()
        centre_x = float(np.average(group["x_km"], weights=weights))
        centre_y = float(np.average(group["y_km"], weights=weights))
        candidates = list(dict.fromkeys(group["road_node"]))
        access = min(
            candidates,
            key=lambda node: (
                (base.xy_km(node)[0] - centre_x) ** 2
                + (base.xy_km(node)[1] - centre_y) ** 2
            ),
        )
        mapping[tuple(int(value) for value in key)] = access
        rows.append(
            {
                "access_node": access,
                "pressure_zone_index": int(key[2]),
                "access_cell_x": int(key[0]),
                "access_cell_y": int(key[1]),
                "building_count": int(len(group)),
                "annual_m3": float(group["water_m3_year"].sum()),
                "peak_flow_lps": float(group["water_service_peak_lps"].sum()),
                "critical_buildings": int(
                    group["service_class"].isin(["public", "industrial"]).sum()
                ),
                "lon": float(access[0]),
                "lat": float(access[1]),
                "ground_elevation_m": float(base.proxy_elevation_m(access)),
            }
        )
    frame["access_node"] = [
        mapping[(int(row.access_cell_x), int(row.access_cell_y), int(row.pressure_zone_index))]
        for row in frame.itertuples()
    ]

    access = pd.DataFrame(rows)
    # Road snapping can collapse neighbouring groups.  Merge those groups while
    # retaining the lowest pressure-zone index as the physical elevation band.
    access = (
        access.groupby("access_node", as_index=False)
        .agg(
            pressure_zone_index=("pressure_zone_index", "min"),
            building_count=("building_count", "sum"),
            annual_m3=("annual_m3", "sum"),
            peak_flow_lps=("peak_flow_lps", "sum"),
            critical_buildings=("critical_buildings", "sum"),
            lon=("lon", "first"),
            lat=("lat", "first"),
            ground_elevation_m=("ground_elevation_m", "first"),
        )
        .sort_values(["pressure_zone_index", "annual_m3"], ascending=[True, False])
        .reset_index(drop=True)
    )
    access["access_id"] = [f"W_A{index:04d}" for index in range(1, len(access) + 1)]
    return frame, access


def _source_tree_union(
    road: nx.Graph,
    source: tuple[float, float],
    targets: Iterable[tuple[float, float]],
) -> tuple[nx.Graph, dict[tuple[float, float], list[tuple[float, float]]]]:
    _, paths = nx.single_source_dijkstra(road, source, weight="length_km")
    graph = nx.Graph()
    graph.add_node(source)
    retained_paths: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for target in targets:
        if target not in paths:
            continue
        path = paths[target]
        retained_paths[target] = path
        base.add_path_edges(graph, road, path)
    return graph, retained_paths


def _add_critical_second_feeds(
    road: nx.Graph,
    model: nx.Graph,
    source: tuple[float, float],
    source_paths: dict[tuple[float, float], list[tuple[float, float]]],
    access: pd.DataFrame,
    config: WaterCandidateConfig,
) -> tuple[nx.Graph, pd.DataFrame]:
    """Add alternate paths to high-demand access nodes until the scale gate closes."""
    graph = model.copy()
    current_length = sum(float(data["length_km"]) for _, _, data in graph.edges(data=True))
    target = float(config.target_main_length_km)
    # Keep the coordinate tuple as a scalar Python object.  A pandas index of
    # two-tuples is silently promoted to a MultiIndex, which makes ``.loc``
    # ambiguous.  Ranking rows directly is deterministic and avoids that trap.
    ranked_records = list(
        access.sort_values(
            ["critical_buildings", "annual_m3", "access_id"],
            ascending=[False, False, True],
        ).itertuples(index=False)
    )
    access_lookup = {row.access_node: row for row in ranked_records}
    ranked = [row.access_node for row in ranked_records]
    rows: list[dict[str, Any]] = []
    for rank, target_node in enumerate(ranked[: config.critical_access_candidates], 1):
        if current_length >= target * 0.999:
            break
        first_path = source_paths.get(target_node)
        if not first_path:
            continue
        first_edges = {_edge_key(u, v) for u, v in _path_edges(first_path)}

        def penalized_weight(u: tuple[float, float], v: tuple[float, float], data: dict[str, Any]) -> float:
            factor = config.path_penalty_factor if _edge_key(u, v) in first_edges else 1.0
            return float(data["length_km"]) * factor

        try:
            alternate = nx.shortest_path(road, source, target_node, weight=penalized_weight)
        except nx.NetworkXNoPath:
            continue
        new_edges = [(u, v) for u, v in _path_edges(alternate) if not graph.has_edge(u, v)]
        new_length = sum(float(road[u][v]["length_km"]) for u, v in new_edges)
        if new_length < config.minimum_new_second_feed_km:
            continue
        if new_length > config.maximum_new_second_feed_km:
            continue
        proposed = current_length + new_length
        # Allow a final small overshoot only when it is closer to the official
        # inventory than the current candidate.
        if proposed > target * 1.01 and abs(target - proposed) >= abs(target - current_length):
            continue
        before_cycles = graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)
        base.add_path_edges(graph, road, alternate)
        after_cycles = graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)
        if after_cycles <= before_cycles:
            # Defensive rollback is unnecessary because no new edge was added
            # when cycle rank did not increase, but record nothing.
            continue
        current_length = proposed
        rows.append(
            {
                "selection_rank": rank,
                "access_node": target_node,
                "access_id": access_lookup[target_node].access_id,
                "annual_m3": float(access_lookup[target_node].annual_m3),
                "critical_buildings": int(access_lookup[target_node].critical_buildings),
                "new_route_km": new_length,
                "cumulative_main_length_km": current_length,
                "cycle_rank_after": after_cycles,
            }
        )
    return graph, pd.DataFrame(rows)


def _tree_parent_map(
    graph: nx.Graph,
    source: tuple[float, float],
) -> tuple[dict[tuple[float, float], tuple[float, float]], list[tuple[float, float]]]:
    predecessor, _ = nx.dijkstra_predecessor_and_distance(graph, source, weight="length_km")
    parent: dict[tuple[float, float], tuple[float, float]] = {}
    for node, values in predecessor.items():
        if node == source or not values:
            continue
        parent[node] = values[0]
    order = list(nx.bfs_tree(nx.Graph([(p, n) for n, p in parent.items()]), source).nodes)
    return parent, order


def _hazen_williams_loss_m(length_m: float, flow_m3s: float, diameter_m: float, c: float = 130.0) -> float:
    if flow_m3s <= 0 or diameter_m <= 0:
        return 0.0
    return 10.67 * length_m * flow_m3s ** 1.852 / (c ** 1.852 * diameter_m ** 4.8704)


def generate_pressure_zone_water_candidate(
    case_dir: Path,
    road: nx.Graph,
    buildings: pd.DataFrame,
    *,
    config: WaterCandidateConfig = WaterCandidateConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate the pressure-zone/access-node drinking-water candidate."""
    case_dir.mkdir(parents=True, exist_ok=True)
    attached = attach_buildings(buildings[buildings["water_connected"].astype(bool)], road)
    attached, access = _weighted_group_access_nodes(
        attached,
        cell_size_km=config.access_cell_size_km,
        elevation_band_m=config.pressure_zone_height_m,
        demand_column="water_m3_year",
    )
    source = _nearest_named_facility(road, kinds={"water_works"}, name=config.source_name)
    base_graph, source_paths = _source_tree_union(road, source, access["access_node"])
    graph, second_feeds = _add_critical_second_feeds(
        road, base_graph, source, source_paths, access, config
    )
    main_length = sum(float(data["length_km"]) for _, _, data in graph.edges(data=True))
    cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)

    protected = {source, *access["access_node"].tolist()}
    compressed = base.compress_undirected(graph, protected)
    main_index = base.NodeIndex(compressed.nodes)
    attached["main_node"] = [main_index.nearest((row.lon, row.lat)) for row in attached.itertuples()]
    attached["main_connection_length_km"] = [
        base.distance_km((row.lon, row.lat), row.main_node) for row in attached.itertuples()
    ]

    annual_at: defaultdict[tuple[float, float], float] = defaultdict(float)
    peak_at: defaultdict[tuple[float, float], float] = defaultdict(float)
    building_count_at: defaultdict[tuple[float, float], int] = defaultdict(int)
    for row in attached.itertuples():
        annual_at[row.main_node] += float(row.water_m3_year)
        peak_at[row.main_node] += float(row.water_service_peak_lps) / 1000.0
        building_count_at[row.main_node] += 1

    parent, order = _tree_parent_map(compressed, source)
    subtree = {node: float(peak_at[node]) for node in compressed}
    for node in reversed(order):
        if node in parent:
            subtree[parent[node]] += subtree[node]
    tree_edges = {_edge_key(node, ancestor) for node, ancestor in parent.items()}

    access_nodes = set(access["access_node"])
    zone_by_access = access.set_index("access_node")["pressure_zone_index"].to_dict()
    pressure_zone_by_node: dict[tuple[float, float], int] = {}
    access_index = base.NodeIndex(access_nodes)
    for node in compressed:
        nearest = access_index.nearest(node)
        pressure_zone_by_node[node] = int(zone_by_access[nearest])
    # The first node in each elevation band encountered from the source is the
    # synthetic zone entry.  This is a topology marker, not a recovered valve.
    zone_entry: dict[int, tuple[float, float]] = {}
    distances = nx.single_source_dijkstra_path_length(compressed, source, weight="length_km")
    for zone in sorted(set(pressure_zone_by_node.values())):
        candidates = [node for node, value in pressure_zone_by_node.items() if value == zone]
        zone_entry[zone] = min(candidates, key=lambda node: distances.get(node, math.inf))

    node_id = {node: f"W_M{index:05d}" for index, node in enumerate(compressed.nodes, 1)}
    node_rows: list[dict[str, Any]] = [
        {
            "node_id": "W_RES",
            "node_type": "reservoir",
            "building_id": "",
            "lon": source[0],
            "lat": source[1],
            "elevation_m": base.proxy_elevation_m(source),
            "pressure_zone_id": "SOURCE",
            "demand_m3s": 0.0,
            "annual_m3": 0.0,
            "building_count": 0,
        }
    ]
    for node, identifier in node_id.items():
        zone = pressure_zone_by_node[node]
        if node == source:
            node_type = "source_outlet"
        elif node == zone_entry[zone]:
            node_type = "pressure_zone_entry"
        elif node in access_nodes:
            node_type = "demand_access_junction"
        else:
            node_type = "main_junction"
        node_rows.append(
            {
                "node_id": identifier,
                "node_type": node_type,
                "building_id": "",
                "lon": node[0],
                "lat": node[1],
                "elevation_m": base.proxy_elevation_m(node),
                "pressure_zone_id": f"WZ{zone + 1}",
                "demand_m3s": 0.0,
                "annual_m3": annual_at[node],
                "building_count": building_count_at[node],
            }
        )

    pipe_rows: list[dict[str, Any]] = []
    edge_design: dict[frozenset[tuple[float, float]], float] = {}
    maximum_velocity = 0.0
    for index, (u, v, data) in enumerate(compressed.edges(data=True), 1):
        key = _edge_key(u, v)
        if key in tree_edges:
            downstream = u if parent.get(u) == v else v
            design_flow = max(float(subtree[downstream]), 2e-4)
        else:
            design_flow = max(0.20 * (peak_at[u] + peak_at[v]), 2e-4)
        required_diameter_m = math.sqrt(
            4.0 * design_flow / (math.pi * config.target_velocity_m_s)
        )
        dn = round_up(required_diameter_m * 1000.0, WATER_MAIN_DN)
        velocity = 4.0 * design_flow / (math.pi * (dn / 1000.0) ** 2)
        maximum_velocity = max(maximum_velocity, velocity)
        edge_design[key] = design_flow
        pipe_rows.append(
            {
                "link_id": f"W_P{index:05d}",
                "from_node": node_id[u],
                "to_node": node_id[v],
                "link_type": "distribution_main",
                "building_id": "",
                "length_km": float(data["length_km"]),
                "diameter_mm": dn,
                "roughness": 130.0,
                "design_flow_m3s": design_flow,
                "pressure_zone_from": f"WZ{pressure_zone_by_node[u] + 1}",
                "pressure_zone_to": f"WZ{pressure_zone_by_node[v] + 1}",
                "geometry_json": json.dumps(geometry(data, u, v), separators=(",", ":")),
            }
        )

    service_rows: list[dict[str, Any]] = []
    for row in attached.itertuples():
        suffix = str(row.building_id).replace("OSM_W", "").replace("INFDB_", "I")
        service_node = f"W_B_{suffix}"
        service_id = f"W_SC_{suffix}"
        demand = float(row.water_m3_year) / (365.0 * 86400.0)
        peak = max(float(row.water_service_peak_lps) / 1000.0, 1e-6)
        required_mm = math.sqrt(4.0 * peak / (math.pi * 1.5)) * 1000.0
        minimum_dn = 40 if row.service_class in {"commercial", "public", "industrial"} else 25
        dn = max(minimum_dn, round_up(required_mm, WATER_SERVICE_DN))
        main_identifier = node_id[row.main_node]
        node_rows.append(
            {
                "node_id": service_node,
                "node_type": "building_service",
                "building_id": row.building_id,
                "lon": row.lon,
                "lat": row.lat,
                "elevation_m": base.proxy_elevation_m((row.lon, row.lat)),
                "pressure_zone_id": f"WZ{pressure_zone_by_node[row.main_node] + 1}",
                "demand_m3s": demand,
                "annual_m3": float(row.water_m3_year),
                "building_count": 1,
            }
        )
        length_km = max(float(row.main_connection_length_km), 0.001)
        pipe_rows.append(
            {
                "link_id": service_id,
                "from_node": main_identifier,
                "to_node": service_node,
                "link_type": "building_service",
                "building_id": row.building_id,
                "length_km": length_km,
                "diameter_mm": dn,
                "roughness": 130.0,
                "design_flow_m3s": peak,
                "pressure_zone_from": f"WZ{pressure_zone_by_node[row.main_node] + 1}",
                "pressure_zone_to": f"WZ{pressure_zone_by_node[row.main_node] + 1}",
                "geometry_json": json.dumps(
                    [[row.main_node[0], row.main_node[1]], [row.lon, row.lat]],
                    separators=(",", ":"),
                ),
            }
        )
        service_rows.append(
            {
                "service_id": service_id,
                "building_id": row.building_id,
                "service_node": service_node,
                "main_node": main_identifier,
                "access_node": str(row.access_node),
                "pressure_zone_id": f"WZ{pressure_zone_by_node[row.main_node] + 1}",
                "service_class": row.service_class,
                "length_km": length_km,
                "diameter_mm": dn,
                "annual_m3": float(row.water_m3_year),
                "peak_flow_lps": float(row.water_service_peak_lps),
                "connection_evidence_class": row.connection_evidence_class,
            }
        )
    pipe_rows.append(
        {
            "link_id": "W_MAIN_PUMP",
            "from_node": "W_RES",
            "to_node": node_id[source],
            "link_type": "pump",
            "building_id": "",
            "length_km": 0.0,
            "diameter_mm": 0,
            "roughness": 0.0,
            "design_flow_m3s": float(attached["water_m3_year"].sum()) / (365.0 * 86400.0),
            "pressure_zone_from": "SOURCE",
            "pressure_zone_to": f"WZ{pressure_zone_by_node[source] + 1}",
            "geometry_json": json.dumps([[source[0], source[1]], [source[0], source[1]]]),
        }
    )

    nodes = pd.DataFrame(node_rows)
    links = pd.DataFrame(pipe_rows)
    services = pd.DataFrame(service_rows)

    # Transparent tree-based hydraulic proxy.  It supports design auditing but
    # is not labelled as an EPANET/WNTR pass.
    total_head_at_source = base.proxy_elevation_m(source) + config.source_head_rise_m
    head = {source: total_head_at_source}
    for node in order[1:]:
        ancestor = parent[node]
        data = compressed[ancestor][node]
        key = _edge_key(ancestor, node)
        design = edge_design[key]
        # Recover selected DN from the matching row.
        match = next(
            row for row in pipe_rows
            if row["link_type"] == "distribution_main"
            and {row["from_node"], row["to_node"]} == {node_id[ancestor], node_id[node]}
        )
        head_loss = _hazen_williams_loss_m(
            float(data["length_km"]) * 1000.0,
            design,
            float(match["diameter_mm"]) / 1000.0,
        )
        head[node] = head[ancestor] - head_loss
    proxy_pressures = {
        node: head.get(node, math.nan) - base.proxy_elevation_m(node)
        for node in compressed
    }
    finite_pressures = [value for value in proxy_pressures.values() if math.isfinite(value)]

    # Apply the same bounded source-head repair used by the native water branch:
    # pipes are sized first; source head is then advanced only by the remaining
    # system-wide pressure deficit.  If the upper pressure screen would be
    # exceeded, the candidate is left for explicit zone-control repair instead
    # of silently relaxing a threshold.
    head_adjustment_m = max(0.0, 20.0 - min(finite_pressures))
    if max(finite_pressures) + head_adjustment_m <= 110.0 + 1e-9:
        effective_source_head_rise_m = config.source_head_rise_m + head_adjustment_m
        finite_pressures = [value + head_adjustment_m for value in finite_pressures]
        proxy_pressures = {
            node: value + head_adjustment_m
            for node, value in proxy_pressures.items()
        }
        pressure_repair = "bounded source-head increase after pipe sizing"
    else:
        effective_source_head_rise_m = config.source_head_rise_m
        pressure_repair = "explicit pressure-zone control required before native acceptance"

    # Native input file is produced even when WNTR is not installed.
    _write_epanet_candidate(
        case_dir / "drinking_water_pressure_zone_candidate.inp",
        nodes,
        links,
        source_head_rise_m=effective_source_head_rise_m,
    )

    main_relative_error = (main_length - config.target_main_length_km) / config.target_main_length_km
    solver = {
        "topology_created": True,
        "native_model_created": True,
        "native_solver_executed": False,
        "native_solver_status": (
            "intermediate spatial layer only; final EPANET/WNTR execution is "
            "performed by urban4.municipal_native"
        ),
        "method": "pressure-zone access-node backbone with critical second-feed paths",
        "source": config.source_name,
        "source_coordinate": list(source),
        "registered_buildings": int(len(attached)),
        "access_nodes": int(len(access)),
        "pressure_zones": int(access["pressure_zone_index"].nunique()),
        "critical_second_feed_paths": int(len(second_feeds)),
        "cycle_rank": int(cycle_rank),
        "main_length_km": float(main_length),
        "published_main_length_km": float(config.target_main_length_km),
        "main_length_relative_error": float(main_relative_error),
        "inventory_gate_passed": bool(abs(main_relative_error) <= config.length_tolerance_fraction),
        "maximum_design_velocity_m_s": float(maximum_velocity),
        "proxy_minimum_pressure_m": float(min(finite_pressures)),
        "proxy_maximum_pressure_m": float(max(finite_pressures)),
        "proxy_pressure_screen_passed": bool(
            min(finite_pressures) >= 20.0 and max(finite_pressures) <= 110.0
        ),
        "source_head_rise_m": float(effective_source_head_rise_m),
        "source_head_adjustment_m": float(head_adjustment_m),
        "pressure_repair": pressure_repair,
        "elevation_evidence": base.terrain_evidence(),
        "configuration": asdict(config),
    }

    nodes.to_csv(case_dir / "drinking_water_nodes.csv", index=False)
    links.to_csv(case_dir / "drinking_water_links.csv", index=False)
    services.to_csv(case_dir / "drinking_water_service_connections.csv", index=False)
    access.drop(columns=["access_node"]).to_csv(case_dir / "drinking_water_access_nodes.csv", index=False)
    second_feeds.drop(columns=["access_node"], errors="ignore").to_csv(
        case_dir / "drinking_water_second_feed_audit.csv", index=False
    )
    return nodes, links, services, solver


def _write_epanet_candidate(
    path: Path,
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    *,
    source_head_rise_m: float,
) -> None:
    reservoir = nodes[nodes["node_id"].eq("W_RES")].iloc[0]
    lines = [
        "[TITLE]",
        ";; Urban4 pressure-zone/access-node drinking-water candidate",
        "",
        "[OPTIONS]",
        "UNITS LPS",
        "HEADLOSS H-W",
        "DEMAND MODEL PDA",
        "MINIMUM PRESSURE 0",
        "REQUIRED PRESSURE 20",
        "",
        "[JUNCTIONS]",
        ";;ID Elevation Demand Pattern",
    ]
    for row in nodes[~nodes["node_id"].eq("W_RES")].itertuples():
        lines.append(
            f"{row.node_id} {float(row.elevation_m):.3f} {float(row.demand_m3s) * 1000.0:.8f}"
        )
    lines += [
        "",
        "[RESERVOIRS]",
        ";;ID Head Pattern",
        f"W_RES {float(reservoir.elevation_m):.3f}",
        "",
        "[PIPES]",
        ";;ID Node1 Node2 Length Diameter Roughness MinorLoss Status",
    ]
    for row in links[~links["link_type"].isin(["pump"])].itertuples():
        lines.append(
            f"{row.link_id} {row.from_node} {row.to_node} "
            f"{max(float(row.length_km) * 1000.0, 0.5):.3f} "
            f"{float(row.diameter_mm):.1f} {float(row.roughness):.1f} 0 OPEN"
        )
    total_flow_lps = float(nodes["demand_m3s"].sum()) * 1000.0
    lines += [
        "",
        "[PUMPS]",
        ";;ID Node1 Node2 Parameters",
        "W_MAIN_PUMP W_RES " + str(links.loc[links.link_type.eq("pump"), "to_node"].iloc[0]) + " HEAD W_PUMP_CURVE",
        "",
        "[CURVES]",
        ";;ID X-Value Y-Value",
        f"W_PUMP_CURVE 0 {1.20 * source_head_rise_m:.3f}",
        f"W_PUMP_CURVE {max(total_flow_lps, 0.001):.6f} {source_head_rise_m:.3f}",
        f"W_PUMP_CURVE {max(3.0 * total_flow_lps, 0.003):.6f} {0.85 * source_head_rise_m:.3f}",
        "",
        "[COORDINATES]",
        ";;Node X Y",
    ]
    for row in nodes.itertuples():
        x, y = base.xy_km((row.lon, row.lat))
        lines.append(f"{row.node_id} {x * 1000.0:.3f} {y * 1000.0:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _subcatchment_terminals(
    attached: pd.DataFrame,
    *,
    cell_size_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = attached.copy()
    xy = np.asarray([base.xy_km(node) for node in frame["road_node"]])
    frame["x_km"] = xy[:, 0]
    frame["y_km"] = xy[:, 1]
    frame["ground_elevation_m"] = frame["road_node"].map(base.proxy_elevation_m)
    frame["subcatchment_x"] = np.floor(frame["x_km"] / cell_size_km).astype(int)
    frame["subcatchment_y"] = np.floor(frame["y_km"] / cell_size_km).astype(int)
    mapping: dict[tuple[int, int], tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(["subcatchment_x", "subcatchment_y"], sort=True):
        weights = group["wastewater_sanitary_m3_year"].astype(float).clip(lower=1e-9)
        centre_x = float(np.average(group["x_km"], weights=weights))
        centre_y = float(np.average(group["y_km"], weights=weights))
        candidates = list(dict.fromkeys(group["road_node"]))
        terminal = min(
            candidates,
            key=lambda node: (
                base.proxy_elevation_m(node),
                0.05
                * (
                    (base.xy_km(node)[0] - centre_x) ** 2
                    + (base.xy_km(node)[1] - centre_y) ** 2
                ),
            ),
        )
        mapping[(int(key[0]), int(key[1]))] = terminal
        rows.append(
            {
                "terminal_node": terminal,
                "subcatchment_x": int(key[0]),
                "subcatchment_y": int(key[1]),
                "building_count": int(len(group)),
                "annual_m3": float(group["wastewater_sanitary_m3_year"].sum()),
                "peak_flow_lps": float(group["wastewater_service_peak_lps"].sum()),
                "lon": terminal[0],
                "lat": terminal[1],
                "ground_elevation_m": base.proxy_elevation_m(terminal),
            }
        )
    frame["subcatchment_terminal"] = [
        mapping[(int(row.subcatchment_x), int(row.subcatchment_y))]
        for row in frame.itertuples()
    ]
    terminals = (
        pd.DataFrame(rows)
        .groupby("terminal_node", as_index=False)
        .agg(
            building_count=("building_count", "sum"),
            annual_m3=("annual_m3", "sum"),
            peak_flow_lps=("peak_flow_lps", "sum"),
            lon=("lon", "first"),
            lat=("lat", "first"),
            ground_elevation_m=("ground_elevation_m", "first"),
        )
        .sort_values("annual_m3", ascending=False)
        .reset_index(drop=True)
    )
    terminals["subcatchment_id"] = [f"SC{index:04d}" for index in range(1, len(terminals) + 1)]
    return frame, terminals


def _directed_routes_to_outfall(
    road: nx.Graph,
    terminals: Iterable[tuple[float, float]],
    outfall: tuple[float, float],
    *,
    terrain_rise_penalty: float,
) -> nx.DiGraph:
    elevations = {node: base.proxy_elevation_m(node) for node in road}
    minimum = min(elevations.values())
    span = max(max(elevations.values()) - minimum, 1.0)
    directed = nx.DiGraph()
    for u, v, data in road.edges(data=True):
        for a, b in ((u, v), (v, u)):
            rise = max(0.0, elevations[b] - elevations[a]) / span
            directed.add_edge(
                a,
                b,
                length_km=float(data["length_km"]),
                cost=float(data["length_km"]) * (1.0 + terrain_rise_penalty * rise),
            )
    reverse = directed.reverse(copy=False)
    route_distance = {outfall: 0.0}
    next_to_outfall: dict[tuple[float, float], tuple[float, float]] = {}
    queue: list[tuple[float, tuple[float, float]]] = [(0.0, outfall)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance > route_distance.get(node, math.inf) + 1e-12:
            continue
        for neighbour, data in reverse[node].items():
            candidate = current_distance + float(data.get("cost", 1.0))
            if candidate < route_distance.get(neighbour, math.inf) - 1e-12:
                route_distance[neighbour] = candidate
                next_to_outfall[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    union = nx.DiGraph()
    union.add_node(outfall)
    for terminal in terminals:
        union.add_node(terminal)
        if terminal not in route_distance:
            union.add_edge(
                terminal,
                outfall,
                length_km=base.distance_km(terminal, outfall),
                geometry=[terminal, outfall],
                confidence_class="D",
            )
            continue
        current = terminal
        while current != outfall:
            downstream = next_to_outfall[current]
            data = road[current][downstream]
            union.add_edge(
                current,
                downstream,
                length_km=float(data["length_km"]),
                geometry=geometry(data, current, downstream),
                confidence_class="B/C",
            )
            current = downstream
    return union


def _assign_compressed_inverts(
    graph: nx.DiGraph,
    outfall: tuple[float, float],
    config: SewerCandidateConfig,
) -> tuple[dict[tuple[float, float], float], set[tuple[tuple[float, float], tuple[float, float]]], dict[tuple[tuple[float, float], tuple[float, float]], float]]:
    ground = {node: base.proxy_elevation_m(node) for node in graph}
    inverts = {outfall: ground[outfall] - 3.0}
    pump_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    severity: dict[tuple[tuple[float, float], tuple[float, float]], float] = {}
    for downstream in reversed(list(nx.topological_sort(graph))):
        if downstream not in inverts:
            continue
        for upstream in graph.predecessors(downstream):
            length_m = max(float(graph[upstream][downstream]["length_km"]) * 1000.0, 1.0)
            desired = inverts[downstream] + config.gravity_grade_candidates[0] * length_m
            shallow_limit = ground[upstream] - config.minimum_cover_m
            severity[(upstream, downstream)] = desired - shallow_limit
            if desired > shallow_limit:
                inverts[upstream] = ground[upstream] - 2.0
                pump_edges.add((upstream, downstream))
            else:
                inverts[upstream] = max(desired, ground[upstream] - config.maximum_cover_m)
    return inverts, pump_edges, severity


def _interpolate_point(a: tuple[float, float], b: tuple[float, float], fraction: float) -> tuple[float, float]:
    return (
        round(a[0] + fraction * (b[0] - a[0]), 9),
        round(a[1] + fraction * (b[1] - a[1]), 9),
    )


def _cut_polyline(
    points: list[tuple[float, float]],
    max_segment_km: float,
) -> list[list[tuple[float, float]]]:
    """Split a polyline by interpolating cut points at the requested spacing."""
    if len(points) < 2:
        return []
    edge_lengths = [base.distance_km(a, b) for a, b in zip(points[:-1], points[1:])]
    total = sum(edge_lengths)
    if total <= max_segment_km:
        return [points]
    cut_distances = list(np.arange(max_segment_km, total, max_segment_km))
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [points[0]]
    travelled = 0.0
    cut_index = 0
    for a, b, length in zip(points[:-1], points[1:], edge_lengths):
        local_start = travelled
        local_end = travelled + length
        while cut_index < len(cut_distances) and cut_distances[cut_index] < local_end - 1e-12:
            fraction = (cut_distances[cut_index] - local_start) / max(length, 1e-12)
            cut = _interpolate_point(a, b, fraction)
            if cut != current[-1]:
                current.append(cut)
            segments.append(current)
            current = [cut]
            cut_index += 1
        if b != current[-1]:
            current.append(b)
        travelled = local_end
    if len(current) > 1:
        segments.append(current)
    return segments


def _split_manhole_graph(
    compressed: nx.DiGraph,
    inverts: dict[tuple[float, float], float],
    pump_edges: set[tuple[tuple[float, float], tuple[float, float]]],
    spacing_m: float,
) -> tuple[nx.DiGraph, dict[tuple[float, float], float], set[tuple[tuple[float, float], tuple[float, float]]]]:
    graph = nx.DiGraph()
    split_inverts: dict[tuple[float, float], float] = {}
    split_force_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for u, v, data in compressed.edges(data=True):
        points = [(float(x), float(y)) for x, y in data["geometry"]]
        is_force = (u, v) in pump_edges
        pieces = [points] if is_force else _cut_polyline(points, spacing_m / 1000.0)
        total_length = max(_polyline_length_km(points), 1e-12)
        accumulated = 0.0
        for piece in pieces:
            a = piece[0]
            b = piece[-1]
            piece_length = _polyline_length_km(piece)
            start_fraction = accumulated / total_length
            end_fraction = (accumulated + piece_length) / total_length
            split_inverts.setdefault(
                a, inverts[u] + start_fraction * (inverts[v] - inverts[u])
            )
            split_inverts[b] = inverts[u] + end_fraction * (inverts[v] - inverts[u])
            graph.add_edge(
                a,
                b,
                length_km=piece_length,
                geometry=piece,
                compressed_parent=(u, v),
                link_type="force_main" if is_force else "gravity_main",
            )
            if is_force:
                split_force_edges.add((a, b))
            accumulated += piece_length
    return graph, split_inverts, split_force_edges


def _select_manhole_spacing(
    compressed: nx.DiGraph,
    inverts: dict[tuple[float, float], float],
    pump_edges: set[tuple[tuple[float, float], tuple[float, float]]],
    config: SewerCandidateConfig,
) -> tuple[float, nx.DiGraph, dict[tuple[float, float], float], set[tuple[tuple[float, float], tuple[float, float]]], pd.DataFrame]:
    trials: list[dict[str, Any]] = []
    best: tuple[float, nx.DiGraph, dict, set] | None = None
    best_error = math.inf
    for spacing in config.manhole_spacing_candidates_m:
        graph, split_inverts, force_edges = _split_manhole_graph(
            compressed, inverts, pump_edges, spacing
        )
        error = abs(graph.number_of_nodes() - config.target_manhole_count)
        trials.append(
            {
                "maximum_spacing_m": spacing,
                "generated_manhole_nodes": graph.number_of_nodes(),
                "absolute_count_error": error,
            }
        )
        if error < best_error:
            best_error = error
            best = (spacing, graph, split_inverts, force_edges)
    assert best is not None
    return best[0], best[1], best[2], best[3], pd.DataFrame(trials)



def _split_all_manhole_graph(
    compressed: nx.DiGraph,
    spacing_m: float,
) -> nx.DiGraph:
    """Insert manholes without deciding gravity/force-main status first.

    Geometry is split for the SWMM representation, while the complete parent
    corridor geometry is retained on every piece.  The latter lets the
    water--sewer audit compare street-corridor use rather than being distorted
    by different internal node segmentation.
    """
    graph = nx.DiGraph()
    for u, v, data in compressed.edges(data=True):
        parent_geometry = [(float(x), float(y)) for x, y in data["geometry"]]
        for piece in _cut_polyline(parent_geometry, spacing_m / 1000.0):
            a, b = piece[0], piece[-1]
            graph.add_edge(
                a,
                b,
                length_km=_polyline_length_km(piece),
                geometry=piece,
                parent_edge=(u, v),
                parent_geometry=parent_geometry,
            )
    return graph


def _select_manhole_spacing_untyped(
    compressed: nx.DiGraph,
    config: SewerCandidateConfig,
) -> tuple[float, nx.DiGraph, pd.DataFrame]:
    trials: list[dict[str, Any]] = []
    best: tuple[float, nx.DiGraph] | None = None
    best_error = math.inf
    for spacing in config.manhole_spacing_candidates_m:
        graph = _split_all_manhole_graph(compressed, spacing)
        error = abs(graph.number_of_nodes() - config.target_manhole_count)
        trials.append(
            {
                "maximum_spacing_m": spacing,
                "generated_manhole_nodes": graph.number_of_nodes(),
                "absolute_count_error": error,
            }
        )
        if error < best_error:
            best_error = error
            best = (spacing, graph)
    assert best is not None
    return best[0], best[1], pd.DataFrame(trials)


def _assign_manhole_inverts(
    model: nx.DiGraph,
    outfall: tuple[float, float],
    *,
    minimum_grade: float,
    minimum_cover_m: float,
    maximum_cover_m: float,
) -> tuple[
    dict[tuple[float, float], float],
    set[tuple[tuple[float, float], tuple[float, float]]],
    dict[tuple[tuple[float, float], tuple[float, float]], float],
]:
    """Assign cover-bounded inverts on the actual manhole graph.

    A gravity edge must descend at the declared minimum grade while its
    upstream invert remains between minimum and maximum cover.  When that is
    impossible, the edge is explicitly classified as a force-main transition.
    This avoids the interpolation artefact that previously created thousands
    of cover violations between compressed endpoints.
    """
    if not nx.is_directed_acyclic_graph(model):
        raise ValueError("Manhole collection graph must be acyclic")
    ground = {node: base.proxy_elevation_m(node) for node in model}
    inverts: dict[tuple[float, float], float] = {
        outfall: ground[outfall] - min(maximum_cover_m, max(minimum_cover_m, 3.0))
    }
    force_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    severity: dict[tuple[tuple[float, float], tuple[float, float]], float] = {}
    for downstream in reversed(list(nx.topological_sort(model))):
        if downstream not in inverts:
            continue
        for upstream in model.predecessors(downstream):
            length_m = max(float(model[upstream][downstream]["length_km"]) * 1000.0, 1.0)
            desired = inverts[downstream] + minimum_grade * length_m
            shallow_limit = ground[upstream] - minimum_cover_m
            deep_limit = ground[upstream] - maximum_cover_m
            severity[(upstream, downstream)] = max(0.0, desired - shallow_limit)
            if desired > shallow_limit:
                # A lift/force-main transition resets the upstream wet-well
                # invert while retaining the declared cover envelope.
                inverts[upstream] = ground[upstream] - min(
                    maximum_cover_m, max(minimum_cover_m, 2.0)
                )
                force_edges.add((upstream, downstream))
            else:
                # Raising a too-deep candidate to the maximum-cover limit only
                # increases the downstream slope and therefore preserves the
                # minimum-grade requirement.
                inverts[upstream] = max(desired, deep_limit)
    return inverts, force_edges, severity


def _select_grade_to_force_inventory(
    model: nx.DiGraph,
    outfall: tuple[float, float],
    config: SewerCandidateConfig,
) -> tuple[
    float,
    dict[tuple[float, float], float],
    set[tuple[tuple[float, float], tuple[float, float]]],
    dict[tuple[tuple[float, float], tuple[float, float]], float],
    pd.DataFrame,
]:
    """Select a declared grade candidate against the published force-main scale."""
    rows: list[dict[str, Any]] = []
    best: tuple[float, dict, set, dict] | None = None
    best_key: tuple[float, float] | None = None
    for grade in config.gravity_grade_candidates:
        inverts, force_edges, severity = _assign_manhole_inverts(
            model,
            outfall,
            minimum_grade=float(grade),
            minimum_cover_m=config.minimum_cover_m,
            maximum_cover_m=config.maximum_cover_m,
        )
        force_length = sum(float(model[u][v]["length_km"]) for u, v in force_edges)
        relative_error = (
            force_length - config.target_force_main_length_km
        ) / config.target_force_main_length_km
        rows.append(
            {
                "minimum_gravity_grade": float(grade),
                "force_transition_edges": len(force_edges),
                "force_main_length_km": force_length,
                "target_force_main_length_km": config.target_force_main_length_km,
                "relative_error": relative_error,
            }
        )
        key = (abs(relative_error), float(grade))
        if best_key is None or key < best_key:
            best_key = key
            best = (float(grade), inverts, force_edges, severity)
    assert best is not None
    return best[0], best[1], best[2], best[3], pd.DataFrame(rows)


def _weighted_station_clusters(
    force_edges: set[tuple[tuple[float, float], tuple[float, float]]],
    severity: dict[tuple[tuple[float, float], tuple[float, float]], float],
    station_count: int,
) -> tuple[dict[tuple[tuple[float, float], tuple[float, float]], str], pd.DataFrame]:
    """Group force transitions into the published number of station compounds.

    The count is an aggregate calibration input because installed coordinates
    are unavailable.  Clustering is performed in metric coordinates and does
    not alter the collection routes or their hydraulic classification.
    """
    ordered = sorted(force_edges, key=lambda edge: (edge[0], edge[1]))
    if not ordered:
        return {}, pd.DataFrame(columns=["station_id", "lon", "lat", "transition_count"])
    k = min(max(1, int(station_count)), len(ordered))
    points = np.asarray([base.xy_km(edge[0]) for edge in ordered], dtype=float)
    weights = np.asarray(
        [max(float(severity.get(edge, 0.0)), 0.01) for edge in ordered], dtype=float
    )

    # Deterministic farthest-first initialization, followed by weighted Lloyd
    # iterations.  This avoids a new package dependency and keeps reruns exact.
    first = int(np.argmax(weights))
    centers = [points[first]]
    while len(centers) < k:
        distance = np.min(
            np.stack([np.sum((points - center) ** 2, axis=1) for center in centers]),
            axis=0,
        )
        score = distance * weights
        centers.append(points[int(np.argmax(score))])
    centers_array = np.asarray(centers, dtype=float)
    labels = np.zeros(len(points), dtype=int)
    for _ in range(60):
        distances = np.stack(
            [np.sum((points - center) ** 2, axis=1) for center in centers_array],
            axis=1,
        )
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers_array.copy()
        for index in range(k):
            mask = new_labels == index
            if mask.any():
                new_centers[index] = np.average(
                    points[mask], axis=0, weights=weights[mask]
                )
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers_array):
            labels = new_labels
            centers_array = new_centers
            break
        labels = new_labels
        centers_array = new_centers

    edge_station: dict[tuple[tuple[float, float], tuple[float, float]], str] = {}
    rows: list[dict[str, Any]] = []
    for index in range(k):
        member_indices = np.flatnonzero(labels == index)
        if not len(member_indices):
            continue
        station_id = f"S_LS{index + 1:02d}"
        centre = centers_array[index]
        representative_index = int(
            member_indices[
                np.argmin(np.sum((points[member_indices] - centre) ** 2, axis=1))
            ]
        )
        representative = ordered[representative_index][0]
        for member_index in member_indices:
            edge_station[ordered[int(member_index)]] = station_id
        rows.append(
            {
                "station_id": station_id,
                "lon": representative[0],
                "lat": representative[1],
                "transition_count": int(len(member_indices)),
                "maximum_transition_severity_m": float(
                    max(severity.get(ordered[int(i)], 0.0) for i in member_indices)
                ),
            }
        )
    return edge_station, pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)


def generate_manhole_first_wastewater_candidate(
    case_dir: Path,
    road: nx.Graph,
    buildings: pd.DataFrame,
    *,
    config: SewerCandidateConfig = SewerCandidateConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate the subcatchment/manhole-first wastewater candidate."""
    case_dir.mkdir(parents=True, exist_ok=True)
    attached = attach_buildings(buildings[buildings["wastewater_connected"].astype(bool)], road)
    attached, terminals = _subcatchment_terminals(
        attached, cell_size_km=config.subcatchment_cell_size_km
    )
    outfall = _nearest_named_facility(
        road, kinds={"wastewater_plant"}, name=config.outfall_name
    )
    union = _directed_routes_to_outfall(
        road,
        terminals["terminal_node"],
        outfall,
        terrain_rise_penalty=config.terrain_rise_penalty,
    )
    protected = {outfall, *terminals["terminal_node"].tolist()}
    compressed = base.compress_directed(union, protected)
    spacing_m, model, spacing_trials = _select_manhole_spacing_untyped(
        compressed, config
    )
    (
        selected_grade,
        split_inverts,
        force_edges,
        severity,
        grade_trials,
    ) = _select_grade_to_force_inventory(model, outfall, config)
    edge_station, station_compounds = _weighted_station_clusters(
        force_edges, severity, config.target_pump_station_count
    )

    main_index = base.NodeIndex(model.nodes)
    attached["main_node"] = [main_index.nearest((row.lon, row.lat)) for row in attached.itertuples()]
    attached["property_connection_length_km"] = [
        base.distance_km((row.lon, row.lat), row.main_node) for row in attached.itertuples()
    ]

    dry_at: defaultdict[tuple[float, float], float] = defaultdict(float)
    peak_at: defaultdict[tuple[float, float], float] = defaultdict(float)
    annual_at: defaultdict[tuple[float, float], float] = defaultdict(float)
    building_count_at: defaultdict[tuple[float, float], int] = defaultdict(int)
    for row in attached.itertuples():
        dry_at[row.main_node] += float(row.wastewater_sanitary_m3_year) / (365.0 * 86400.0)
        peak_at[row.main_node] += float(row.wastewater_service_peak_lps) / 1000.0
        annual_at[row.main_node] += float(row.wastewater_sanitary_m3_year)
        building_count_at[row.main_node] += 1

    # Every node has one downstream route because all terminals use one reverse
    # shortest-path tree.  Accumulate design flow in topological order.
    flow = {node: float(peak_at[node]) for node in model}
    for node in nx.topological_sort(model):
        successors = list(model.successors(node))
        if successors:
            flow[successors[0]] += flow[node]

    force_transition_nodes = {u for u, _ in force_edges}
    station_node_to_id = {
        (float(row.lon), float(row.lat)): str(row.station_id)
        for row in station_compounds.itertuples()
    }
    lift_nodes = set(station_node_to_id)
    node_id = {node: f"S_M{index:05d}" for index, node in enumerate(model.nodes, 1)}
    terminal_nodes = set(terminals["terminal_node"])
    node_rows: list[dict[str, Any]] = []
    cover_violations = 0
    for node, identifier in node_id.items():
        ground = base.proxy_elevation_m(node)
        invert = float(split_inverts[node])
        cover = ground - invert
        if cover < config.minimum_cover_m - 1e-6 or cover > config.maximum_cover_m + 1e-6:
            cover_violations += 1
        if node == outfall:
            node_type = "outfall"
        elif node in lift_nodes:
            node_type = "lift_station"
        elif node in force_transition_nodes:
            node_type = "force_transition_manhole"
        elif node in terminal_nodes:
            node_type = "subcatchment_terminal_manhole"
        else:
            node_type = "collection_manhole"
        node_rows.append(
            {
                "node_id": identifier,
                "node_type": node_type,
                "building_id": "",
                "lon": node[0],
                "lat": node[1],
                "ground_elevation_m": ground,
                "invert_elevation_m": invert,
                "cover_depth_m": cover,
                "dry_inflow_m3s": dry_at[node],
                "annual_m3": annual_at[node],
                "building_count": building_count_at[node],
                "lift_station_id": station_node_to_id.get(node, ""),
            }
        )

    link_rows: list[dict[str, Any]] = []
    minimum_gravity_slope = math.inf
    force_main_length = 0.0
    gravity_length = 0.0
    for index, (u, v, data) in enumerate(model.edges(data=True), 1):
        length_km = float(data["length_km"])
        length_m = max(length_km * 1000.0, 1.0)
        link_type = "force_main" if (u, v) in force_edges else "gravity_main"
        design_flow = max(float(flow[u]), 0.0002)
        if link_type == "force_main":
            slope = 0.0
            target_velocity = 1.0
            required = math.sqrt(4.0 * design_flow / (math.pi * target_velocity)) * 1000.0
            dn = round_up(max(200.0, required), SEWER_MAIN_DN)
            force_main_length += length_km
            pump_design = design_flow
            pump_head = max(2.0, split_inverts[v] - split_inverts[u] + 3.0)
        else:
            slope = max(
                selected_grade,
                (split_inverts[u] - split_inverts[v]) / length_m,
            )
            minimum_gravity_slope = min(minimum_gravity_slope, slope)
            diameter_m = (
                design_flow
                * config.manning_n
                * 4.0 ** (5.0 / 3.0)
                / (math.pi * math.sqrt(slope))
            ) ** (3.0 / 8.0)
            dn = round_up(max(200.0, math.ceil(diameter_m * 1000.0)), SEWER_MAIN_DN)
            gravity_length += length_km
            pump_design = 0.0
            pump_head = 0.0
        link_rows.append(
            {
                "link_id": f"S_C{index:05d}",
                "from_node": node_id[u],
                "to_node": node_id[v],
                "link_type": link_type,
                "building_id": "",
                "length_km": length_km,
                "diameter_mm": dn,
                "slope": slope,
                "manning_n": config.manning_n,
                "design_flow_m3s": design_flow,
                "pump_design_flow_m3s": pump_design,
                "pump_head_m": pump_head,
                "lift_station_id": edge_station.get((u, v), ""),
                "geometry_json": json.dumps(data["geometry"], separators=(",", ":")),
                "corridor_geometry_json": json.dumps(
                    data.get("parent_geometry", data["geometry"]), separators=(",", ":")
                ),
            }
        )

    service_rows: list[dict[str, Any]] = []
    for row in attached.itertuples():
        suffix = str(row.building_id).replace("OSM_W", "").replace("INFDB_", "I")
        property_node = f"S_B_{suffix}"
        service_id = f"S_SC_{suffix}"
        main_identifier = node_id[row.main_node]
        ground = base.proxy_elevation_m((row.lon, row.lat))
        main_invert = float(split_inverts[row.main_node])
        length_km = max(float(row.property_connection_length_km), 0.001)
        required_invert = main_invert + 0.005 * length_km * 1000.0
        needs_property_lift = required_invert > ground - 0.50
        property_invert = ground - 2.0 if needs_property_lift else max(required_invert, ground - 4.0)
        node_rows.append(
            {
                "node_id": property_node,
                "node_type": "property_lift_station" if needs_property_lift else "building_property_connection",
                "building_id": row.building_id,
                "lon": row.lon,
                "lat": row.lat,
                "ground_elevation_m": ground,
                "invert_elevation_m": property_invert,
                "cover_depth_m": ground - property_invert,
                "dry_inflow_m3s": float(row.wastewater_sanitary_m3_year) / (365.0 * 86400.0),
                "annual_m3": float(row.wastewater_sanitary_m3_year),
                "building_count": 1,
            }
        )
        service_dn = 200 if row.service_class in {"industrial", "public"} else 150
        peak = max(float(row.wastewater_service_peak_lps) / 1000.0, 1e-6)
        link_rows.append(
            {
                "link_id": service_id,
                "from_node": property_node,
                "to_node": main_identifier,
                "link_type": "property_force_main" if needs_property_lift else "property_lateral",
                "building_id": row.building_id,
                "length_km": length_km,
                "diameter_mm": service_dn,
                "slope": 0.005,
                "manning_n": config.manning_n,
                "design_flow_m3s": peak,
                "pump_design_flow_m3s": peak if needs_property_lift else 0.0,
                "pump_head_m": max(2.0, main_invert - property_invert + 3.0) if needs_property_lift else 0.0,
                "lift_station_id": "PRIVATE_PROPERTY" if needs_property_lift else "",
                "corridor_geometry_json": "",
                "geometry_json": json.dumps(
                    [[row.lon, row.lat], [row.main_node[0], row.main_node[1]]],
                    separators=(",", ":"),
                ),
            }
        )
        service_rows.append(
            {
                "service_id": service_id,
                "building_id": row.building_id,
                "property_node": property_node,
                "main_node": main_identifier,
                "subcatchment_terminal": str(row.subcatchment_terminal),
                "service_class": row.service_class,
                "length_km": length_km,
                "diameter_mm": service_dn,
                "annual_m3": float(row.wastewater_sanitary_m3_year),
                "peak_flow_lps": float(row.wastewater_service_peak_lps),
                "connection_mode": "property_lift" if needs_property_lift else "gravity_lateral",
                "connection_evidence_class": row.connection_evidence_class,
            }
        )

    nodes = pd.DataFrame(node_rows)
    links = pd.DataFrame(link_rows)
    services = pd.DataFrame(service_rows)
    # Do not emit a misleading SWMM file.  The structural algorithm can mark
    # several grade-infeasible transition segments within one spatial lift
    # station compound.  Those transitions must first be consolidated into
    # explicit wet wells, pumps, storage, and force-main systems using measured
    # elevation before a defensible SWMM model can be written.
    stale_swmm = case_dir / "wastewater_manhole_candidate.inp"
    if stale_swmm.exists():
        stale_swmm.unlink()

    main_length = gravity_length + force_main_length
    relative_error = (main_length - config.target_total_main_length_km) / config.target_total_main_length_km
    solver = {
        "topology_created": True,
        "native_model_created": False,
        "native_solver_executed": False,
        "native_solver_status": (
            "intermediate spatial layer only; consolidation into 13 hydraulic "
            "station/force-main systems and final SWMM execution are performed "
            "by urban4.municipal_native"
        ),
        "method": "terrain-aware subcatchment terminals with manhole-first directed collection",
        "outfall": config.outfall_name,
        "outfall_coordinate": list(outfall),
        "registered_buildings": int(len(attached)),
        "subcatchment_terminals": int(len(terminals)),
        "main_manhole_nodes": int(model.number_of_nodes()),
        "published_manhole_count": int(config.target_manhole_count),
        "manhole_count_error": int(model.number_of_nodes() - config.target_manhole_count),
        "selected_maximum_manhole_spacing_m": float(spacing_m),
        "main_length_km": float(main_length),
        "gravity_main_length_km": float(gravity_length),
        "force_main_length_km": float(force_main_length),
        "published_force_main_length_km": float(config.target_force_main_length_km),
        "force_main_length_relative_error": float(
            (force_main_length - config.target_force_main_length_km)
            / config.target_force_main_length_km
        ),
        "force_main_inventory_gate_passed": bool(
            abs(force_main_length - config.target_force_main_length_km)
            / config.target_force_main_length_km
            <= config.force_main_length_tolerance_fraction
        ),
        "published_total_collection_length_km": float(config.target_total_main_length_km),
        "main_length_relative_error": float(relative_error),
        "inventory_gate_passed": bool(abs(relative_error) <= config.length_tolerance_fraction),
        "lift_station_compounds": int(len(station_compounds)),
        "force_transition_edges": int(len(force_edges)),
        "published_pump_station_count": int(config.target_pump_station_count),
        "pump_station_count_difference": int(len(station_compounds) - config.target_pump_station_count),
        "selected_minimum_gravity_grade": float(selected_grade),
        "minimum_gravity_slope": float(minimum_gravity_slope),
        "cover_violations_after_interpolation": int(cover_violations),
        "directed_acyclic": bool(nx.is_directed_acyclic_graph(model)),
        "all_nodes_reach_outfall": bool(
            all(node == outfall or nx.has_path(model, node, outfall) for node in model)
        ),
        "property_lift_stations": int(nodes.node_type.eq("property_lift_station").sum()),
        "elevation_evidence": base.terrain_evidence(),
        "configuration": asdict(config),
    }

    nodes.to_csv(case_dir / "wastewater_nodes.csv", index=False)
    links.to_csv(case_dir / "wastewater_links.csv", index=False)
    services.to_csv(case_dir / "wastewater_service_connections.csv", index=False)
    terminals.drop(columns=["terminal_node"]).to_csv(case_dir / "wastewater_subcatchments.csv", index=False)
    spacing_trials.to_csv(case_dir / "wastewater_manhole_spacing_sweep.csv", index=False)
    grade_trials.to_csv(case_dir / "wastewater_grade_force_main_sweep.csv", index=False)
    station_compounds.to_csv(case_dir / "wastewater_lift_station_compounds.csv", index=False)
    pd.DataFrame(
        [
            {
                "from_node": str(u),
                "to_node": str(v),
                "lift_station_id": edge_station.get((u, v), ""),
                "infeasibility_m": float(severity.get((u, v), 0.0)),
                "transition_length_km": float(model[u][v]["length_km"]),
            }
            for u, v in sorted(force_edges)
        ]
    ).to_csv(case_dir / "wastewater_lift_audit.csv", index=False)
    return nodes, links, services, solver


def _canonical_road_segment(a: Iterable[float], b: Iterable[float], digits: int = 7) -> tuple:
    p = (round(float(a[0]), digits), round(float(a[1]), digits))
    q = (round(float(b[0]), digits), round(float(b[1]), digits))
    return (p, q) if p <= q else (q, p)


def _segment_map(frame: pd.DataFrame, allowed_types: set[str]) -> dict[tuple, float]:
    segments: dict[tuple, float] = {}
    use_corridor = "corridor_geometry_json" in frame.columns
    for row in frame[frame.link_type.isin(allowed_types)].itertuples():
        raw = getattr(row, "corridor_geometry_json", "") if use_corridor else ""
        if not isinstance(raw, str) or not raw.strip():
            raw = row.geometry_json
        points = json.loads(raw)
        for a, b in zip(points[:-1], points[1:]):
            key = _canonical_road_segment(a, b)
            segments[key] = base.distance_km(key[0], key[1])
    return segments


def audit_water_sewer_candidate(
    output_dir: Path,
    water_links: pd.DataFrame,
    sewer_links: pd.DataFrame,
    *,
    baseline_summary_path: Path | None = None,
) -> dict[str, Any]:
    water_segments = _segment_map(water_links, {"distribution_main"})
    sewer_segments = _segment_map(sewer_links, {"gravity_main", "force_main"})
    water_keys = set(water_segments)
    sewer_keys = set(sewer_segments)
    shared = water_keys & sewer_keys
    water_length = sum(water_segments.values())
    sewer_length = sum(sewer_segments.values())
    shared_length = sum(min(water_segments[key], sewer_segments[key]) for key in shared)
    water_only = sum(water_segments[key] for key in water_keys - sewer_keys)
    sewer_only = sum(sewer_segments[key] for key in sewer_keys - water_keys)
    union_length = shared_length + water_only + sewer_only
    metrics: dict[str, Any] = {
        "water_unique_main_geometry_km": water_length,
        "sewer_unique_main_geometry_km": sewer_length,
        "exact_shared_geometry_km": shared_length,
        "water_only_geometry_km": water_only,
        "sewer_only_geometry_km": sewer_only,
        "water_shared_percent": 100.0 * shared_length / water_length,
        "sewer_shared_percent": 100.0 * shared_length / sewer_length,
        "length_weighted_jaccard": shared_length / union_length,
    }
    if baseline_summary_path and baseline_summary_path.exists():
        baseline = pd.read_csv(baseline_summary_path).set_index("metric")["value"]
        metrics.update(
            {
                "baseline_water_shared_percent": float(
                    baseline["water_main_length_shared_percent"]
                ),
                "baseline_sewer_shared_percent": float(
                    baseline["sewer_main_length_shared_percent"]
                ),
                "baseline_length_weighted_jaccard": float(
                    baseline["length_weighted_jaccard"]
                ),
            }
        )
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).to_csv(output_dir / "water_wastewater_overlap_before_after.csv", index=False)
    features = []
    for key in water_keys | sewer_keys:
        if key in shared:
            kind = "shared"
        elif key in water_keys:
            kind = "water_only"
        else:
            kind = "sewer_only"
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "class": kind,
                    "length_km": base.distance_km(key[0], key[1]),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(key[0]), list(key[1])],
                },
            }
        )
    (output_dir / "water_wastewater_candidate_segments.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}) + "\n",
        encoding="utf-8",
    )
    return metrics


def run_principle_aligned_candidates(
    project: Path,
    *,
    output_name: str = "water_sewer_principle_candidate",
) -> dict[str, Any]:
    output = project / "outputs" / output_name
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((project / "cases" / "schweinfurt.json").read_text(encoding="utf-8"))
    buildings = pd.read_csv(
        project / "outputs" / "integrated_service_resolved" / "common_building_service_ledger.csv"
    )
    road = base.build_road_graph(base.load_elements("local_roads.json"))
    water_config = WaterCandidateConfig(
        target_main_length_km=float(config["official_anchors"]["drinking_water_route_km"]),
        source_head_rise_m=float(config["bidirectional_coupling"]["water_design_head_rise_m"]),
    )
    sewer_config = SewerCandidateConfig(
        target_total_main_length_km=float(
            config["official_anchors"]["wastewater_combined_km"]
            + config["official_anchors"]["wastewater_sanitary_km"]
            + config["official_anchors"]["wastewater_storm_km"]
            + config["official_anchors"]["wastewater_force_main_km"]
        ),
        target_force_main_length_km=float(
            config["official_anchors"]["wastewater_force_main_km"]
        ),
        target_manhole_count=int(config["official_anchors"]["wastewater_manhole_count"]),
        target_pump_station_count=int(config["official_anchors"]["wastewater_pump_stations"]),
    )
    water_nodes, water_links, water_services, water_state = generate_pressure_zone_water_candidate(
        output, road, buildings, config=water_config
    )
    sewer_nodes, sewer_links, sewer_services, sewer_state = generate_manhole_first_wastewater_candidate(
        output, road, buildings, config=sewer_config
    )
    baseline_summary = (
        project
        / "outputs"
        / "water_wastewater_overlap"
        / "water_wastewater_overlap_summary.csv"
    )
    overlap = audit_water_sewer_candidate(
        output,
        water_links,
        sewer_links,
        baseline_summary_path=baseline_summary,
    )
    manifest = {
        "status": (
            "intermediate principle-aligned spatial layers; final municipality-scale "
            "native construction and acceptance are performed by urban4.municipal_native"
        ),
        "evidence_boundary": {
            "water_route_km": "calibration input",
            "wastewater_total_collection_km": "calibration input; includes combined, sanitary, storm and force-main inventories",
            "wastewater_gravity_and_force_main_lengths": "calibration inputs",
            "wastewater_manhole_count": "calibration input",
            "wastewater_pump_stations": "calibration input for spatial station compounds",
            "installed_edge_geometry": "unavailable and not reconstructed",
        },
        "drinking_water": water_state,
        "wastewater": sewer_state,
        "overlap": overlap,
        "counts": {
            "water_nodes": len(water_nodes),
            "water_links": len(water_links),
            "water_services": len(water_services),
            "wastewater_nodes": len(sewer_nodes),
            "wastewater_links": len(sewer_links),
            "wastewater_services": len(sewer_services),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_principle_aligned_candidates(project_root), indent=2))
