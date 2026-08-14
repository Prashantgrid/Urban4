#!/usr/bin/env python3
"""Building/service-resolved sector-native topology generators for Urban4.

The common demand-zone ledger remains a planning and reporting object.  The
physical graphs generated here terminate at explicit building service points:
electric meters, water laterals, wastewater property connections and selected
district-heat substations.  Every terminal retains the immutable building ID.
"""

from __future__ import annotations

import json
import heapq
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

from . import schweinfurt_base as base
from .route_aware_heat import RouteAwareHeatConfig, select_route_aware_heat_connections


LV_CABLES = [
    (16, 0.076, 1.910),
    (25, 0.096, 1.200),
    (35, 0.115, 0.868),
    (70, 0.179, 0.443),
    (95, 0.216, 0.320),
    (150, 0.282, 0.206),
    (240, 0.368, 0.125),
]
MV_CABLES = [
    (95, 0.250, 0.320),
    (150, 0.320, 0.206),
    (185, 0.360, 0.164),
    (240, 0.420, 0.125),
    (300, 0.480, 0.100),
    (400, 0.560, 0.077),
]
TRANSFORMER_MVA = [0.25, 0.40, 0.63, 0.80, 1.00]
WATER_MAIN_DN = [80, 100, 125, 150, 200, 250, 300, 400, 500, 600]
WATER_SERVICE_DN = [25, 32, 40, 50, 63, 80]
SEWER_MAIN_DN = [200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1600, 2000]
HEAT_DN = [25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800]


def round_up(value: float, library: Iterable[int]) -> int:
    values = list(library)
    return int(next((item for item in values if item >= value), values[-1]))


def choose_heat_dn(
    mass_flow_kg_s: float,
    *,
    maximum_velocity_m_s: float = 1.5,
    maximum_pressure_gradient_pa_m: float = 100.0,
) -> int:
    """Select the smallest heat pipe satisfying velocity and friction screens."""
    density = 985.0
    darcy_friction = 0.02
    for dn in HEAT_DN:
        diameter = dn / 1000.0
        velocity = mass_flow_kg_s / (density * math.pi * diameter**2 / 4.0)
        pressure_gradient = darcy_friction * density * velocity**2 / (2.0 * diameter)
        if velocity <= maximum_velocity_m_s and pressure_gradient <= maximum_pressure_gradient_pa_m:
            return dn
    return HEAT_DN[-1]


def geometry(data: dict[str, Any], u: tuple[float, float], v: tuple[float, float]) -> list[tuple[float, float]]:
    points = list(data.get("geometry", [u, v]))
    if tuple(points[0]) != u:
        points.reverse()
    return [(float(point[0]), float(point[1])) for point in points]


def attach_buildings(buildings: pd.DataFrame, road: nx.Graph) -> pd.DataFrame:
    """Attach every utility customer to an admissible road node."""
    result = buildings[buildings["service_eligible"].astype(bool)].copy().reset_index(drop=True)
    index = base.NodeIndex(road.nodes)
    result["road_node"] = [index.nearest((row.lon, row.lat)) for row in result.itertuples()]
    result["road_lon"] = result["road_node"].map(lambda point: point[0])
    result["road_lat"] = result["road_node"].map(lambda point: point[1])
    result["service_length_km"] = [
        base.distance_km((row.lon, row.lat), row.road_node) for row in result.itertuples()
    ]
    result["connection_evidence_class"] = np.where(
        result["service_length_km"] <= 0.10, "C", "D"
    )
    return result


def select_heat_service_territories(
    eligible: pd.DataFrame,
    *,
    maximum_connections: int,
    cell_size_km: float = 0.25,
    minimum_heat_density_mwh_ha: float = 800.0,
    enforce_connection_count: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select compact district-heat territories before selecting customers.

    Buildings are first aggregated to a regular planning lattice.  Only cells
    meeting the declared annual heat-density screen are retained; queen-
    adjacent cells form candidate service territories.  Customer selection is
    then restricted to those territories.  Consequently, a low-density case
    may yield a small heat island or no district-heating topology at all.

    ``enforce_connection_count`` is reserved for cases with an imposed public
    contract total.  It relaxes the density screen through a declared sequence
    but never selects buildings outside a retained contiguous territory.
    """
    if eligible.empty or maximum_connections <= 0:
        return eligible.iloc[0:0].copy(), {
            "network_present": False,
            "reason": "no eligible buildings or zero connection limit",
            "heat_density_threshold_mwh_ha": minimum_heat_density_mwh_ha,
            "territory_count": 0,
        }
    frame = eligible.copy()
    xy = np.asarray([base.xy_km((row.lon, row.lat)) for row in frame.itertuples()])
    frame["heat_cell_x"] = np.floor(xy[:, 0] / cell_size_km).astype(int)
    frame["heat_cell_y"] = np.floor(xy[:, 1] / cell_size_km).astype(int)
    cell_area_ha = cell_size_km**2 * 100.0
    cells = frame.groupby(["heat_cell_x", "heat_cell_y"], as_index=False).agg(
        candidate_buildings=("building_id", "size"),
        annual_heat_mwh=("heat_candidate_mwh_year", "sum"),
    )
    cells["heat_density_mwh_ha"] = cells.annual_heat_mwh / cell_area_ha
    threshold_sequence = [float(minimum_heat_density_mwh_ha)]
    if enforce_connection_count:
        threshold_sequence.extend(
            value for value in (700.0, 600.0, 500.0) if value < minimum_heat_density_mwh_ha
        )
    selected_cells: set[tuple[int, int]] = set()
    applied_threshold = float(minimum_heat_density_mwh_ha)
    for threshold in threshold_sequence:
        selected_cells = {
            (int(row.heat_cell_x), int(row.heat_cell_y))
            for row in cells[cells.heat_density_mwh_ha.ge(threshold)].itertuples()
        }
        candidate_count = int(
            frame.apply(
                lambda row: (int(row.heat_cell_x), int(row.heat_cell_y)) in selected_cells,
                axis=1,
            ).sum()
        )
        applied_threshold = threshold
        if not enforce_connection_count or candidate_count >= maximum_connections:
            break
    if not selected_cells:
        return frame.iloc[0:0].copy(), {
            "network_present": False,
            "reason": "no planning cell met the heat-density screen",
            "cell_size_km": cell_size_km,
            "heat_density_threshold_mwh_ha": applied_threshold,
            "territory_count": 0,
        }
    cell_graph = nx.Graph()
    cell_graph.add_nodes_from(selected_cells)
    for cell_x, cell_y in selected_cells:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbour = (cell_x + dx, cell_y + dy)
                if neighbour != (cell_x, cell_y) and neighbour in selected_cells:
                    cell_graph.add_edge((cell_x, cell_y), neighbour)
    territory_by_cell: dict[tuple[int, int], int] = {}
    territory_summaries = []
    for territory_id, component in enumerate(nx.connected_components(cell_graph), 1):
        component_set = set(component)
        for cell in component_set:
            territory_by_cell[cell] = territory_id
        component_cells = cells[
            cells.apply(
                lambda row: (int(row.heat_cell_x), int(row.heat_cell_y)) in component_set,
                axis=1,
            )
        ]
        territory_summaries.append({
            "territory_id": territory_id,
            "annual_heat_mwh": float(component_cells.annual_heat_mwh.sum()),
            "mean_heat_density_mwh_ha": float(component_cells.heat_density_mwh_ha.mean()),
        })
    candidate = frame[
        frame.apply(
            lambda row: (int(row.heat_cell_x), int(row.heat_cell_y)) in selected_cells,
            axis=1,
        )
    ].copy()
    candidate["heat_territory_id"] = [
        territory_by_cell[(int(row.heat_cell_x), int(row.heat_cell_y))]
        for row in candidate.itertuples()
    ]
    territory_order = [
        item["territory_id"]
        for item in sorted(
            territory_summaries,
            key=lambda item: (item["annual_heat_mwh"], item["mean_heat_density_mwh_ha"]),
            reverse=True,
        )
    ]
    candidate["territory_rank"] = candidate.heat_territory_id.map(
        {territory_id: rank for rank, territory_id in enumerate(territory_order)}
    )
    target_count = min(maximum_connections, len(candidate))
    territory_heat = candidate.groupby("heat_territory_id").heat_candidate_mwh_year.sum()
    quotas = {
        territory_id: min(
            int((candidate.heat_territory_id == territory_id).sum()),
            max(1, int(round(target_count * heat / territory_heat.sum()))),
        )
        for territory_id, heat in territory_heat.items()
    }
    while sum(quotas.values()) > target_count:
        reducible = [key for key, value in quotas.items() if value > 1]
        if not reducible:
            break
        key = max(reducible, key=lambda item: quotas[item])
        quotas[key] -= 1
    while sum(quotas.values()) < target_count:
        expandable = [
            key for key, value in quotas.items()
            if value < int((candidate.heat_territory_id == key).sum())
        ]
        if not expandable:
            break
        key = max(
            expandable,
            key=lambda item: territory_heat[item] / max(quotas[item], 1),
        )
        quotas[key] += 1
    selected_parts = []
    for territory_id, quota in quotas.items():
        selected_parts.append(
            candidate[candidate.heat_territory_id.eq(territory_id)].nlargest(
                quota, ["heat_candidate_mwh_year", "floor_area_m2"]
            )
        )
    selected = pd.concat(selected_parts, ignore_index=False).copy()
    metadata = {
        "network_present": not selected.empty,
        "cell_size_km": cell_size_km,
        "heat_density_threshold_mwh_ha": applied_threshold,
        "territory_count": int(selected.heat_territory_id.nunique()) if not selected.empty else 0,
        "suitable_cell_count": len(selected_cells),
        "suitable_candidate_count": len(candidate),
        "selected_connection_count": len(selected),
        "connection_limit_enforced": bool(enforce_connection_count),
    }
    return selected, metadata


def _shortest_path_cache(
    road: nx.Graph,
    source: tuple[float, float],
    targets: Iterable[tuple[float, float]],
) -> tuple[dict[tuple[float, float], float], dict[tuple[float, float], tuple[float, float]]]:
    unique_targets = list(dict.fromkeys(targets))
    # Stop Dijkstra as soon as every requested terminal is settled.  Asking
    # NetworkX for paths to every road node creates a large path-list object
    # and is prohibitively expensive for building-resolved cases.
    targets_remaining = set(unique_targets)
    distance = {source: 0.0}
    predecessor: dict[tuple[float, float], tuple[float, float]] = {}
    queue = [(0.0, source)]
    settled: set[tuple[float, float]] = set()
    while queue and targets_remaining:
        current_distance, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        targets_remaining.discard(node)
        for neighbour, data in road[node].items():
            candidate = current_distance + float(data.get("length_km", 1.0))
            if candidate < distance.get(neighbour, math.inf):
                distance[neighbour] = candidate
                predecessor[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    return distance, predecessor


def _union_tree_from_cache(
    road: nx.Graph,
    source: tuple[float, float],
    targets: Iterable[tuple[float, float]],
    distance: dict[tuple[float, float], float],
    predecessor: dict[tuple[float, float], tuple[float, float]],
) -> nx.Graph:
    unique_targets = list(dict.fromkeys(targets))
    union = nx.Graph()
    union.add_node(source)
    for target in unique_targets:
        if target not in distance:
            union.add_edge(
                source, target, length_km=base.distance_km(source, target),
                geometry=[source, target], confidence_class="D",
            )
            continue
        path = [target]
        while path[-1] != source:
            path.append(predecessor[path[-1]])
        base.add_path_edges(union, road, list(reversed(path)))
    tree = nx.minimum_spanning_tree(union, weight="length_km")
    protected = {source, *unique_targets}
    reduced = base.compress_undirected(tree, protected)
    reduced.add_nodes_from(node for node in protected if node in tree)
    return reduced


def _union_tree(
    road: nx.Graph,
    source: tuple[float, float],
    targets: Iterable[tuple[float, float]],
) -> nx.Graph:
    unique_targets = list(dict.fromkeys(targets))
    distance, predecessor = _shortest_path_cache(road, source, unique_targets)
    return _union_tree_from_cache(
        road, source, unique_targets, distance, predecessor
    )


def _weighted_centres(frame: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    xy = np.asarray([base.xy_km((row.road_lon, row.road_lat)) for row in frame.itertuples()])
    if count <= 1:
        return np.asarray([np.average(xy, axis=0, weights=np.maximum(frame["electricity_peak_kw"], 0.01))])
    centres, _ = kmeans2(xy, min(count, len(frame)), minit="++", iter=80, seed=seed)
    return np.asarray(centres)


def _site_assignments(
    customers: pd.DataFrame,
    road: nx.Graph,
    sites: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {site: index for index, site in enumerate(sites)}
    distances = {site: 0.0 for site in sites}
    owner = dict(lookup)
    queue = [(0.0, site) for site in sites]
    heapq.heapify(queue)
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance > distances.get(node, math.inf) + 1e-12:
            continue
        for neighbour, data in road[node].items():
            candidate = current_distance + float(data.get("length_km", 1.0))
            if candidate < distances.get(neighbour, math.inf) - 1e-12:
                distances[neighbour] = candidate
                owner[neighbour] = owner[node]
                heapq.heappush(queue, (candidate, neighbour))
    euclidean = cKDTree(np.asarray([base.xy_km(site) for site in sites]))
    assignment = []
    route_distance = []
    for row in customers.itertuples():
        target = row.road_node
        if target in owner:
            assignment.append(owner[target])
            route_distance.append(float(distances[target]))
        else:
            distance, index = euclidean.query(np.asarray(base.xy_km(target)))
            assignment.append(int(index))
            route_distance.append(float(distance))
    return np.asarray(assignment, dtype=int), np.asarray(route_distance, dtype=float)


def _transformer_sites(
    customers: pd.DataFrame,
    road: nx.Graph,
    maximum_radius_km: float,
    seed: int,
    target_site_count: int | None = None,
    maximum_customers_per_site: int = 300,
) -> tuple[list[tuple[float, float]], pd.DataFrame]:
    lv = customers[customers["electricity_connection_level"].eq("LV")].copy().reset_index(drop=True)
    if lv.empty:
        raise ValueError("No LV-connected buildings in case")
    normal_capacity_mw = 0.63 * 0.80 * 0.96
    initial = max(
        1,
        int(math.ceil(lv["electricity_peak_kw"].sum() / 1000.0 / normal_capacity_mw)),
        int(math.ceil(len(lv) / maximum_customers_per_site)),
    )
    if target_site_count is not None:
        initial = int(target_site_count)
    centres = _weighted_centres(lv, initial, seed)
    road_index = base.NodeIndex(road.nodes)
    sites: list[tuple[float, float]] = []
    for x, y in centres:
        site = road_index.nearest((float(x / 71.55), float(y / 111.20)))
        if site not in sites:
            sites.append(site)
    candidate_nodes = list(dict.fromkeys(lv["road_node"]))
    while len(sites) < initial:
        site = max(
            (node for node in candidate_nodes if node not in sites),
            key=lambda node: min(base.distance_km(node, existing) for existing in sites),
        )
        sites.append(site)

    # Fast Euclidean farthest-point completion enforces load and customer-count
    # limits before the more expensive road-distance verification.  This avoids
    # rebuilding a citywide multi-source shortest-path partition for every new
    # site in a building-resolved case.
    for _ in range(250):
        site_tree = cKDTree(np.asarray([base.xy_km(site) for site in sites]))
        route_distance, assignment = site_tree.query(
            np.asarray([base.xy_km(node) for node in lv.road_node])
        )
        lv["site_index"] = assignment
        lv["transformer_route_km"] = route_distance
        violations: list[tuple[float, int]] = []
        for site_index, group in lv.groupby("site_index"):
            peak_mw = float(group["electricity_peak_kw"].sum() / 1000.0)
            overload = peak_mw / normal_capacity_mw
            too_many = len(group) / maximum_customers_per_site
            too_far = float(group["transformer_route_km"].max()) / maximum_radius_km
            score = max(overload, too_many, too_far)
            if score > 1.0 + 1e-9:
                violations.append((score, int(site_index)))
        if not violations or target_site_count is not None:
            break
        _, worst_site = max(violations)
        group = lv[lv["site_index"].eq(worst_site)]
        candidate = max(
            (row for row in group.itertuples() if row.road_node not in sites),
            key=lambda row: row.transformer_route_km * max(row.electricity_peak_kw, 0.01),
            default=None,
        )
        if candidate is None:
            break
        sites.append(candidate.road_node)
    # Verify actual admissible-corridor distance and add at most twenty repair
    # sites.  The repair itself is recorded through the final site count and
    # maximum customer-route metric.
    for _ in range(21):
        assignment, route_distance = _site_assignments(lv, road, sites)
        lv["site_index"] = assignment
        lv["transformer_route_km"] = route_distance
        violations = []
        for site_index, group in lv.groupby("site_index"):
            peak_mw = float(group.electricity_peak_kw.sum() / 1000.0)
            score = max(
                peak_mw / normal_capacity_mw,
                len(group) / maximum_customers_per_site,
                float(group.transformer_route_km.max()) / maximum_radius_km,
            )
            if score > 1.0 + 1e-9:
                violations.append((score, int(site_index)))
        if not violations or target_site_count is not None:
            break
        _, worst = max(violations)
        group = lv[lv.site_index.eq(worst)]
        candidate = max(
            (row for row in group.itertuples() if row.road_node not in sites),
            key=lambda row: row.transformer_route_km * max(row.electricity_peak_kw, 0.01),
            default=None,
        )
        if candidate is None:
            break
        sites.append(candidate.road_node)
    assignment, route_distance = _site_assignments(lv, road, sites)
    lv["site_index"] = assignment
    lv["transformer_route_km"] = route_distance
    return sites, lv


def _choose_cable(
    current_ka: float,
    length_km: float,
    voltage_kv: float,
    library: list[tuple[int, float, float]],
    drop_limit_pu: float,
    max_parallel: int = 4,
) -> tuple[int, float, float, int, float]:
    sin_phi = math.sqrt(1.0 - 0.96**2)
    feasible = []
    for size, ampacity, resistance in library:
        for parallel in range(1, max_parallel + 1):
            drop = (
                math.sqrt(3.0) * current_ka
                * (resistance * 0.96 + 0.08 * sin_phi)
                * length_km / max(voltage_kv * parallel, 1e-6)
            )
            if current_ka <= 0.80 * ampacity * parallel and drop <= drop_limit_pu:
                feasible.append((size * parallel, size, ampacity, resistance, parallel, drop))
                break
    if feasible:
        _, size, ampacity, resistance, parallel, drop = min(feasible)
        return size, ampacity, resistance, parallel, drop
    size, ampacity, resistance = library[-1]
    uncorrected = (
        math.sqrt(3.0) * current_ka
        * (resistance * 0.96 + 0.08 * sin_phi) * length_km / max(voltage_kv, 1e-6)
    )
    parallel = max(
        1,
        math.ceil(current_ka / (0.80 * ampacity)),
        math.ceil(uncorrected / max(drop_limit_pu, 1e-6)),
    )
    return size, ampacity, resistance, parallel, uncorrected / parallel


def generate_electricity(
    case_dir: Path,
    road: nx.Graph,
    buildings: pd.DataFrame,
    *,
    maximum_radius_km: float,
    seed: int,
    upstream_point: tuple[float, float] | None = None,
    target_site_count: int | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate one connected MV graph and building-resolved radial LV feeders."""
    started = time.perf_counter()
    stage_started = started
    def progress(message: str) -> None:
        nonlocal stage_started
        if verbose:
            now = time.perf_counter()
            print(f"[electricity] {message}: {now-stage_started:.1f} s (total {now-started:.1f} s)", flush=True)
            stage_started = now
    attached = attach_buildings(buildings, road)
    sites, lv = _transformer_sites(
        attached, road, maximum_radius_km, seed, target_site_count=target_site_count
    )
    progress(f"attached {len(attached)} services and allocated {len(sites)} transformer sites")
    if upstream_point is None:
        centroid = (
            float(np.average(lv["road_lon"], weights=np.maximum(lv["electricity_peak_kw"], 0.01))),
            float(np.average(lv["road_lat"], weights=np.maximum(lv["electricity_peak_kw"], 0.01))),
        )
        upstream = min(
            road.nodes,
            key=lambda node: base.distance_km(node, centroid) + 0.20 * (node[0] - min(p[0] for p in road.nodes)) ** 2,
        )
    else:
        upstream = base.NodeIndex(road.nodes).nearest(upstream_point)

    node_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    transformer_rows: list[dict[str, Any]] = []
    service_rows: list[dict[str, Any]] = []
    node_rows.append({
        "bus_id": "E_MV_SOURCE", "node_type": "mv_source", "voltage_kv": 20.0,
        "root": True, "building_id": "", "transformer_id": "", "feeder_id": "",
        "lon": upstream[0], "lat": upstream[1], "electricity_peak_mw": 0.0,
        "annual_electricity_mwh": 0.0, "q_mvar": 0.0,
    })

    # Connected road-routed MV tree.
    mv_tree = _union_tree(road, upstream, sites)
    progress(f"routed connected MV tree with {mv_tree.number_of_edges()} sections")
    mv_node_id: dict[tuple[float, float], str] = {upstream: "E_MV_SOURCE"}
    for site_index, site in enumerate(sites):
        bus_id = f"E_MV_TR{site_index + 1:03d}"
        if site == upstream:
            # The upstream road vertex can also host a transformer.  Reuse the
            # energized source bus as its HV terminal; overwriting this mapping
            # would leave the otherwise valid MV/LV network as an island.
            continue
        mv_node_id[site] = bus_id
        node_rows.append({
            "bus_id": bus_id, "node_type": "transformer_mv_bus", "voltage_kv": 20.0,
            "root": False, "building_id": "", "transformer_id": f"E_TRF{site_index + 1:03d}",
            "feeder_id": "", "lon": site[0], "lat": site[1],
            "electricity_peak_mw": 0.0, "annual_electricity_mwh": 0.0, "q_mvar": 0.0,
        })
    mv_junction = 1
    for node in mv_tree:
        if node not in mv_node_id:
            mv_node_id[node] = f"E_MV_J{mv_junction:04d}"
            node_rows.append({
                "bus_id": mv_node_id[node], "node_type": "mv_junction", "voltage_kv": 20.0,
                "root": False, "building_id": "", "transformer_id": "", "feeder_id": "",
                "lon": node[0], "lat": node[1], "electricity_peak_mw": 0.0,
                "annual_electricity_mwh": 0.0, "q_mvar": 0.0,
            })
            mv_junction += 1
    mv_parent = {child: parent for parent, child in nx.bfs_edges(mv_tree, upstream)}
    mv_direct = {node: 0.0 for node in mv_tree}
    for site_index, site in enumerate(sites):
        mv_direct[site] += float(lv.loc[lv["site_index"].eq(site_index), "electricity_peak_kw"].sum() / 1000.0)
    mv_subtree = dict(mv_direct)
    for node in reversed(list(nx.bfs_tree(mv_tree, upstream).nodes)):
        if node != upstream:
            mv_subtree[mv_parent[node]] += mv_subtree[node]
    line_counter = 1
    for child, parent in mv_parent.items():
        data = mv_tree[parent][child]
        power = max(mv_subtree[child], 0.001)
        current = power / (math.sqrt(3.0) * 20.0 * 0.96)
        size, ampacity, resistance, parallel, drop = _choose_cable(
            current, float(data["length_km"]), 20.0, MV_CABLES, 0.03, max_parallel=3
        )
        line_rows.append({
            "line_id": f"E_MV_L{line_counter:04d}", "from_node": mv_node_id[parent],
            "to_node": mv_node_id[child], "asset_type": "mv_backbone", "voltage_level": "MV",
            "feeder_id": "MV", "length_km": float(data["length_km"]),
            "cable_size_mm2": size, "parallel_circuits": parallel,
            "r_ohm_per_km": resistance, "x_ohm_per_km": 0.08, "max_i_ka": ampacity,
            "normally_open": False, "design_power_mw": power, "design_current_ka": current,
            "design_voltage_drop_pu": drop,
            "geometry_json": json.dumps(geometry(data, parent, child), separators=(",", ":")),
        })
        line_counter += 1

    lv_junction_counter = 1
    lv_line_counter = 1
    for site_index, site in enumerate(sites):
        group = lv[lv["site_index"].eq(site_index)].copy().reset_index(drop=True)
        assigned_peak_mw = float(group["electricity_peak_kw"].sum() / 1000.0)
        required_mva = assigned_peak_mw / (0.80 * 0.96)
        rating = next((value for value in TRANSFORMER_MVA if value >= required_mva), TRANSFORMER_MVA[-1])
        units = max(1, int(math.ceil(required_mva / TRANSFORMER_MVA[-1])))
        unit_rating = rating if units == 1 else TRANSFORMER_MVA[-1]
        portfolio_mva = units * unit_rating
        transformer_id = f"E_TRF{site_index + 1:03d}"
        lv_root = f"E_LV_TR{site_index + 1:03d}"
        node_rows.append({
            "bus_id": lv_root, "node_type": "transformer_lv_bus", "voltage_kv": 0.4,
            "root": False, "building_id": "", "transformer_id": transformer_id,
            "feeder_id": "", "lon": site[0], "lat": site[1],
            "electricity_peak_mw": 0.0, "annual_electricity_mwh": 0.0, "q_mvar": 0.0,
        })
        transformer_rows.append({
            "transformer_id": transformer_id, "hv_bus": mv_node_id[site], "lv_bus": lv_root,
            "unit_count": units, "unit_rating_mva": unit_rating, "sn_mva": portfolio_mva,
            "assigned_peak_mw": assigned_peak_mw, "customer_count": len(group),
            "maximum_customer_route_km": float(group["transformer_route_km"].max()),
            "lon": site[0], "lat": site[1],
        })

        feeder_count = min(10, max(1, int(math.ceil(len(group) / 25.0))))
        if feeder_count == 1:
            feeder_labels = np.zeros(len(group), dtype=int)
        else:
            feeder_xy = np.asarray([base.xy_km(row.road_node) for row in group.itertuples()])
            _, feeder_labels = kmeans2(feeder_xy, feeder_count, minit="++", iter=60, seed=seed + site_index)
        group["feeder_index"] = feeder_labels
        feeder_distance, feeder_predecessor = _shortest_path_cache(
            road, site, group["road_node"]
        )
        for feeder_index, feeder_group in group.groupby("feeder_index"):
            feeder_id = f"E_F{site_index + 1:03d}_{int(feeder_index) + 1:02d}"
            feeder_tree = _union_tree_from_cache(
                road, site, feeder_group["road_node"], feeder_distance, feeder_predecessor
            )
            road_bus: dict[tuple[float, float], str] = {site: lv_root}
            for node in feeder_tree:
                if node == site:
                    continue
                road_bus[node] = f"E_LV_J{lv_junction_counter:06d}"
                node_rows.append({
                    "bus_id": road_bus[node], "node_type": "lv_junction", "voltage_kv": 0.4,
                    "root": False, "building_id": "", "transformer_id": transformer_id,
                    "feeder_id": feeder_id, "lon": node[0], "lat": node[1],
                    "electricity_peak_mw": 0.0, "annual_electricity_mwh": 0.0, "q_mvar": 0.0,
                })
                lv_junction_counter += 1
            parent = {child: ancestor for ancestor, child in nx.bfs_edges(feeder_tree, site)}
            direct = {node: 0.0 for node in feeder_tree}
            for row in feeder_group.itertuples():
                direct[row.road_node] += float(row.electricity_peak_kw / 1000.0)
            subtree = dict(direct)
            for node in reversed(list(nx.bfs_tree(feeder_tree, site).nodes)):
                if node != site:
                    subtree[parent[node]] += subtree[node]
            longest = max(nx.single_source_dijkstra_path_length(feeder_tree, site, weight="length_km").values(), default=0.001)
            for child, ancestor in parent.items():
                data = feeder_tree[ancestor][child]
                power = max(subtree[child], 0.0005)
                current = power / (math.sqrt(3.0) * 0.4 * 0.96)
                drop_budget = max(0.003, 0.035 * float(data["length_km"]) / max(longest, 0.001))
                size, ampacity, resistance, parallel, drop = _choose_cable(
                    current, float(data["length_km"]), 0.4, LV_CABLES[2:], drop_budget, max_parallel=4
                )
                line_rows.append({
                    "line_id": f"E_LV_L{lv_line_counter:06d}", "from_node": road_bus[ancestor],
                    "to_node": road_bus[child], "asset_type": "lv_feeder", "voltage_level": "LV",
                    "feeder_id": feeder_id, "length_km": float(data["length_km"]),
                    "cable_size_mm2": size, "parallel_circuits": parallel,
                    "r_ohm_per_km": resistance, "x_ohm_per_km": 0.08, "max_i_ka": ampacity,
                    "normally_open": False, "design_power_mw": power, "design_current_ka": current,
                    "design_voltage_drop_pu": drop,
                    "geometry_json": json.dumps(geometry(data, ancestor, child), separators=(",", ":")),
                })
                lv_line_counter += 1

            for row in feeder_group.itertuples():
                building_bus = f"E_B_{str(row.building_id).replace('OSM_W', '')}"
                load_mw = float(row.electricity_peak_kw / 1000.0)
                node_rows.append({
                    "bus_id": building_bus, "node_type": "building_service", "voltage_kv": 0.4,
                    "root": False, "building_id": row.building_id, "transformer_id": transformer_id,
                    "feeder_id": feeder_id, "lon": row.lon, "lat": row.lat,
                    "electricity_peak_mw": load_mw, "annual_electricity_mwh": float(row.electricity_mwh_year),
                    "q_mvar": load_mw * math.tan(math.acos(0.96)),
                })
                service_length = max(float(row.service_length_km), 0.003)
                service_current = float(row.electricity_service_peak_kw / 1000.0) / (math.sqrt(3.0) * 0.4 * 0.96)
                size, ampacity, resistance, parallel, drop = _choose_cable(
                    service_current, service_length, 0.4, LV_CABLES, 0.02, max_parallel=2
                )
                service_id = f"E_SC_{str(row.building_id).replace('OSM_W', '')}"
                line_rows.append({
                    "line_id": service_id, "from_node": road_bus[row.road_node], "to_node": building_bus,
                    "asset_type": "building_service", "voltage_level": "LV", "feeder_id": feeder_id,
                    "length_km": service_length, "cable_size_mm2": size,
                    "parallel_circuits": parallel, "r_ohm_per_km": resistance,
                    "x_ohm_per_km": 0.08, "max_i_ka": ampacity, "normally_open": False,
                    "design_power_mw": float(row.electricity_service_peak_kw / 1000.0),
                    "design_current_ka": service_current, "design_voltage_drop_pu": drop,
                    "geometry_json": json.dumps([[row.road_lon, row.road_lat], [row.lon, row.lat]], separators=(",", ":")),
                })
                service_rows.append({
                    "service_id": service_id, "building_id": row.building_id,
                    "building_bus": building_bus, "road_bus": road_bus[row.road_node],
                    "transformer_id": transformer_id, "feeder_id": feeder_id,
                    "service_class": row.service_class, "length_km": service_length,
                    "cable_size_mm2": size, "service_design_peak_kw": row.electricity_service_peak_kw,
                    "diversified_peak_kw": row.electricity_peak_kw,
                    "connection_evidence_class": row.connection_evidence_class,
                })
        if verbose and (site_index + 1) % 10 == 0:
            progress(f"routed LV feeders for {site_index + 1}/{len(sites)} sites")

    # Direct-MV customers above 250 kW keep their own service identity and do
    # not masquerade as ordinary LV building loads.
    mv_customers = attached[attached["electricity_connection_level"].eq("MV")]
    if not mv_customers.empty:
        source_paths = nx.single_source_dijkstra(road, upstream, weight="length_km")[1]
        for row in mv_customers.itertuples():
            bus_id = f"E_MVC_{str(row.building_id).replace('OSM_W', '')}"
            node_rows.append({
                "bus_id": bus_id, "node_type": "mv_customer", "voltage_kv": 20.0,
                "root": False, "building_id": row.building_id, "transformer_id": "",
                "feeder_id": "MV_CUSTOMER", "lon": row.lon, "lat": row.lat,
                "electricity_peak_mw": row.electricity_peak_kw / 1000.0,
                "annual_electricity_mwh": row.electricity_mwh_year,
                "q_mvar": row.electricity_peak_kw / 1000.0 * math.tan(math.acos(0.96)),
            })
            # Connect to the closest node already represented in the MV tree.
            closest = min(mv_node_id, key=lambda node: base.distance_km(node, row.road_node))
            try:
                path = nx.shortest_path(road, closest, row.road_node, weight="length_km")
                route_geometry = [point for u, v in zip(path[:-1], path[1:]) for point in geometry(road[u][v], u, v)]
                length = nx.path_weight(road, path, weight="length_km")
            except nx.NetworkXNoPath:
                route_geometry = [closest, row.road_node]
                length = base.distance_km(closest, row.road_node)
            load_mw = float(row.electricity_peak_kw / 1000.0)
            current = load_mw / (math.sqrt(3.0) * 20.0 * 0.96)
            size, ampacity, resistance, parallel, drop = _choose_cable(
                current, max(length, 0.005), 20.0, MV_CABLES, 0.03, max_parallel=2
            )
            line_rows.append({
                "line_id": f"E_MVC_L{str(row.building_id).replace('OSM_W', '')}",
                "from_node": mv_node_id[closest], "to_node": bus_id,
                "asset_type": "mv_customer_spur", "voltage_level": "MV", "feeder_id": "MV_CUSTOMER",
                "length_km": max(length, 0.005), "cable_size_mm2": size,
                "parallel_circuits": parallel, "r_ohm_per_km": resistance,
                "x_ohm_per_km": 0.08, "max_i_ka": ampacity, "normally_open": False,
                "design_power_mw": load_mw, "design_current_ka": current,
                "design_voltage_drop_pu": drop,
                "geometry_json": json.dumps(route_geometry, separators=(",", ":")),
            })

    nodes = pd.DataFrame(node_rows).drop_duplicates("bus_id", keep="last")
    lines = pd.DataFrame(line_rows)
    transformers = pd.DataFrame(transformer_rows)
    services = pd.DataFrame(service_rows)
    progress(f"materialized {len(nodes)} buses and {len(lines)} line records")
    solver: dict[str, Any] = {"created": False, "converged": False}
    try:
        import pandapower as pp

        net = pp.create_empty_network(sn_mva=100.0)
        created_buses = pp.create_buses(
            net, len(nodes), vn_kv=nodes.voltage_kv.to_numpy(float),
            name=nodes.bus_id.tolist(),
            geodata=list(zip(nodes.lon.to_numpy(float), nodes.lat.to_numpy(float))),
        )
        bus_index = dict(zip(nodes.bus_id, created_buses))
        pp.create_ext_grid(net, bus_index["E_MV_SOURCE"], vm_pu=1.03, name="tile MV source")
        load_nodes = nodes[nodes.electricity_peak_mw.gt(0)].copy()
        if not load_nodes.empty:
            pp.create_loads(
                net, [bus_index[item] for item in load_nodes.bus_id],
                p_mw=load_nodes.electricity_peak_mw.to_numpy(float),
                q_mvar=load_nodes.q_mvar.to_numpy(float),
                name=[f"LD_{item}" for item in load_nodes.bus_id],
            )
        created_lines = pp.create_lines_from_parameters(
            net,
            [bus_index[item] for item in lines.from_node],
            [bus_index[item] for item in lines.to_node],
            length_km=lines.length_km.clip(lower=0.001).to_numpy(float),
            r_ohm_per_km=lines.r_ohm_per_km.to_numpy(float),
            x_ohm_per_km=lines.x_ohm_per_km.to_numpy(float),
            c_nf_per_km=np.where(lines.voltage_level.eq("LV"), 210.0, 250.0),
            max_i_ka=lines.max_i_ka.to_numpy(float),
            parallel=lines.parallel_circuits.clip(lower=1).to_numpy(int),
            name=lines.line_id.tolist(),
        )
        net_line_index = dict(zip(lines.line_id, created_lines))
        created_transformers = pp.create_transformers_from_parameters(
            net,
            [bus_index[item] for item in transformers.hv_bus],
            [bus_index[item] for item in transformers.lv_bus],
            sn_mva=transformers.sn_mva.to_numpy(float),
            vn_hv_kv=np.full(len(transformers), 20.0),
            vn_lv_kv=np.full(len(transformers), 0.4),
            vk_percent=np.full(len(transformers), 6.0),
            vkr_percent=np.full(len(transformers), 0.85),
            pfe_kw=np.maximum(0.6, 1.2 * transformers.sn_mva.to_numpy(float)),
            i0_percent=np.full(len(transformers), 0.2),
            shift_degree=np.zeros(len(transformers)),
            tap_side=["hv"] * len(transformers), tap_neutral=np.zeros(len(transformers), dtype=int),
            tap_min=np.full(len(transformers), -2), tap_max=np.full(len(transformers), 2),
            tap_step_percent=np.full(len(transformers), 2.5), tap_pos=np.full(len(transformers), -1),
            name=transformers.transformer_id.tolist(),
        )
        net_trafo_index = dict(zip(transformers.transformer_id, created_transformers))
        progress("constructed pandapower tables")
        for algorithm in ["nr", "bfsw"]:
            try:
                pp.runpp(
                    net, algorithm=algorithm, numba=True,
                    max_iteration=40, init="flat", calculate_voltage_angles=False,
                    voltage_depend_loads=False,
                )
                if net.converged:
                    break
            except Exception:
                continue
        if not net.converged:
            raise RuntimeError("pandapower power flow did not converge")
        progress("completed pandapower AC solve")
        energized_graph = nx.Graph()
        energized_graph.add_nodes_from(int(index) for index in net.bus.index[net.bus.in_service])
        energized_graph.add_edges_from(
            (int(row.from_bus), int(row.to_bus)) for row in net.line.itertuples() if row.in_service
        )
        energized_graph.add_edges_from(
            (int(row.hv_bus), int(row.lv_bus)) for row in net.trafo.itertuples() if row.in_service
        )
        component_count = nx.number_connected_components(energized_graph)
        if component_count != 1 or net.res_bus.vm_pu.isna().any() or net.res_load.p_mw.isna().any():
            raise RuntimeError(
                f"pandapower solved only a partial topology: {component_count} components"
            )
        pp.to_json(net, str(case_dir / "electricity_pandapower.json"))
        nodes["result_vm_pu"] = [float(net.res_bus.loc[bus_index[bus], "vm_pu"]) for bus in nodes["bus_id"]]
        lines["result_loading_percent"] = [float(net.res_line.loc[net_line_index[line], "loading_percent"]) for line in lines["line_id"]]
        transformers["result_loading_percent"] = [float(net.res_trafo.loc[net_trafo_index[item], "loading_percent"]) for item in transformers["transformer_id"]]
        solver = {
            "created": True, "converged": True, "method": "AC power flow (runpp), not OPF",
            "graph_components": component_count,
            "minimum_voltage_pu": float(net.res_bus.vm_pu.min()),
            "maximum_voltage_pu": float(net.res_bus.vm_pu.max()),
            "maximum_line_loading_percent": float(net.res_line.loading_percent.max()),
            "maximum_transformer_loading_percent": float(net.res_trafo.loading_percent.max()),
            "building_service_buses": int((nodes["node_type"] == "building_service").sum()),
            "mv_customer_buses": int((nodes["node_type"] == "mv_customer").sum()),
        }
    except Exception as exc:
        solver["error"] = f"{type(exc).__name__}: {exc}"

    nodes.to_csv(case_dir / "electricity_nodes.csv", index=False)
    lines.to_csv(case_dir / "electricity_links.csv", index=False)
    transformers.to_csv(case_dir / "electricity_transformers.csv", index=False)
    services.to_csv(case_dir / "electricity_service_connections.csv", index=False)
    return nodes, lines, transformers, services, solver


def _looped_water_graph(
    road: nx.Graph,
    source: tuple[float, float],
    targets: list[tuple[float, float]],
) -> tuple[nx.Graph, nx.Graph]:
    """Return the radial sizing tree and a reproducibly looped main graph."""
    tree = _union_tree(road, source, targets)
    model = tree.copy()
    unique = list(dict.fromkeys(targets))
    if len(unique) < 3:
        return tree, model
    xy = np.asarray([base.xy_km(point) for point in unique])
    neighbours = cKDTree(xy)
    candidate_pairs: set[tuple[int, int]] = set()
    for left, point in enumerate(xy):
        _, indices = neighbours.query(point, k=min(4, len(unique)))
        for right in np.atleast_1d(indices):
            right = int(right)
            if left != right:
                candidate_pairs.add(tuple(sorted((left, right))))
    candidates = []
    target_cycles = min(120, max(3, int(math.ceil(len(unique) / 75.0))))
    ranked_pairs = sorted(
        candidate_pairs,
        key=lambda item: float(np.linalg.norm(xy[item[0]] - xy[item[1]])),
    )
    for left, right in ranked_pairs:
        a, b = unique[left], unique[right]
        if base.distance_km(a, b) > 1.2:
            continue
        try:
            route = nx.shortest_path(road, a, b, weight="length_km")
            road_length = nx.path_weight(road, route, weight="length_km")
            tree_length = nx.shortest_path_length(tree, a, b, weight="length_km")
        except nx.NetworkXNoPath:
            continue
        if road_length <= 1.2 and tree_length >= 1.30 * road_length:
            candidates.append((road_length / tree_length, road_length, route))
        if len(candidates) >= max(4 * target_cycles, 24):
            break
    base_length = sum(float(data["length_km"]) for _, _, data in tree.edges(data=True))
    added_length = 0.0
    for _, length, route in sorted(candidates):
        if added_length + length > 0.12 * max(base_length, 0.001):
            continue
        before = model.number_of_edges() - model.number_of_nodes() + nx.number_connected_components(model)
        base.add_path_edges(model, road, route)
        after = model.number_of_edges() - model.number_of_nodes() + nx.number_connected_components(model)
        if after > before:
            added_length += length
        if after >= target_cycles:
            break
    return tree, model


def generate_water(
    case_dir: Path,
    road: nx.Graph,
    buildings: pd.DataFrame,
    *,
    source_point: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate looped water mains and one explicit service lateral per building."""
    attached = attach_buildings(buildings[buildings["water_connected"].astype(bool)], road)
    if source_point is None:
        median_lat = float(attached["road_lat"].median())
        west = float(attached["road_lon"].quantile(0.02))
        source = min(road.nodes, key=lambda node: abs(node[1] - median_lat) + 0.3 * abs(node[0] - west))
    else:
        source = base.NodeIndex(road.nodes).nearest(source_point)
    targets = list(dict.fromkeys(attached["road_node"]))
    sizing_tree, model = _looped_water_graph(road, source, targets)

    demand_at = defaultdict(float)
    peak_at = defaultdict(float)
    for row in attached.itertuples():
        demand_at[row.road_node] += float(row.water_m3_year / (365.0 * 86400.0))
        peak_at[row.road_node] += float(row.water_service_peak_lps / 1000.0)
    parent = {child: ancestor for ancestor, child in nx.bfs_edges(sizing_tree, source)}
    subtree = {node: float(peak_at[node]) for node in sizing_tree}
    for node in reversed(list(nx.bfs_tree(sizing_tree, source).nodes)):
        if node != source:
            subtree[parent[node]] += subtree[node]
    tree_design: dict[frozenset[tuple[float, float]], float] = {}
    for child, ancestor in parent.items():
        tree_design[frozenset((ancestor, child))] = max(subtree[child], 2e-4)

    road_node_id = {node: f"W_M{index:05d}" for index, node in enumerate(model.nodes, 1)}
    node_rows: list[dict[str, Any]] = [{
        "node_id": "W_RES", "node_type": "reservoir", "building_id": "",
        "lon": source[0], "lat": source[1], "elevation_m": base.proxy_elevation_m(source),
        "demand_m3s": 0.0, "annual_m3": 0.0,
    }]
    for node, node_id in road_node_id.items():
        node_rows.append({
            "node_id": node_id, "node_type": "pump_outlet" if node == source else "main_junction",
            "building_id": "", "lon": node[0], "lat": node[1],
            "elevation_m": base.proxy_elevation_m(node), "demand_m3s": 0.0, "annual_m3": 0.0,
        })

    pipe_rows: list[dict[str, Any]] = []
    main_index = 1
    for u, v, data in model.edges(data=True):
        design = tree_design.get(frozenset((u, v)), max(0.20 * (peak_at[u] + peak_at[v]), 2e-4))
        required_mm = math.sqrt(4.0 * design / math.pi) * 1000.0
        dn = round_up(required_mm, WATER_MAIN_DN)
        pipe_rows.append({
            "link_id": f"W_P{main_index:05d}", "from_node": road_node_id[u], "to_node": road_node_id[v],
            "link_type": "distribution_main", "building_id": "", "length_km": float(data["length_km"]),
            "diameter_mm": dn, "roughness": 130.0, "design_flow_m3s": design,
            "geometry_json": json.dumps(geometry(data, u, v), separators=(",", ":")),
        })
        main_index += 1

    service_rows: list[dict[str, Any]] = []
    for row in attached.itertuples():
        suffix = str(row.building_id).replace("OSM_W", "")
        node_id = f"W_B_{suffix}"
        demand = float(row.water_m3_year / (365.0 * 86400.0))
        service_peak = max(float(row.water_service_peak_lps / 1000.0), 1e-6)
        required_mm = math.sqrt(4.0 * service_peak / (math.pi * 1.5)) * 1000.0
        minimum_dn = 40 if row.service_class in {"commercial", "public", "industrial"} else 25
        dn = max(minimum_dn, round_up(required_mm, WATER_SERVICE_DN))
        service_id = f"W_SC_{suffix}"
        node_rows.append({
            "node_id": node_id, "node_type": "building_service", "building_id": row.building_id,
            "lon": row.lon, "lat": row.lat, "elevation_m": base.proxy_elevation_m((row.lon, row.lat)),
            "demand_m3s": demand, "annual_m3": float(row.water_m3_year),
        })
        pipe_rows.append({
            "link_id": service_id, "from_node": road_node_id[row.road_node], "to_node": node_id,
            "link_type": "building_service", "building_id": row.building_id,
            "length_km": max(float(row.service_length_km), 0.001), "diameter_mm": dn,
            "roughness": 130.0, "design_flow_m3s": service_peak,
            "geometry_json": json.dumps([[row.road_lon, row.road_lat], [row.lon, row.lat]], separators=(",", ":")),
        })
        service_rows.append({
            "service_id": service_id, "building_id": row.building_id,
            "service_node": node_id, "main_node": road_node_id[row.road_node],
            "service_class": row.service_class, "length_km": max(float(row.service_length_km), 0.001),
            "diameter_mm": dn, "annual_m3": row.water_m3_year,
            "peak_flow_lps": row.water_service_peak_lps,
            "connection_evidence_class": row.connection_evidence_class,
        })
    # The pump is an explicit asset/interface, not a disguised high reservoir.
    pipe_rows.append({
        "link_id": "W_MAIN_PUMP", "from_node": "W_RES", "to_node": road_node_id[source],
        "link_type": "pump", "building_id": "", "length_km": 0.0, "diameter_mm": 0,
        "roughness": 0.0, "design_flow_m3s": float(attached["water_m3_year"].sum() / (365.0 * 86400.0)),
        "geometry_json": json.dumps([[source[0], source[1]], [source[0], source[1]]], separators=(",", ":")),
    })

    nodes = pd.DataFrame(node_rows)
    pipes = pd.DataFrame(pipe_rows)
    services = pd.DataFrame(service_rows)
    solver: dict[str, Any] = {"created": False, "converged": False}
    try:
        import wntr

        total_average = float(nodes["demand_m3s"].sum())
        terrain_relief = float(nodes["elevation_m"].max() - nodes["elevation_m"].min())
        head_rise = max(55.0, terrain_relief + 32.0)

        def create_network(multiplier: float, rise: float):
            wn = wntr.network.WaterNetworkModel()
            reservoir = nodes[nodes["node_id"].eq("W_RES")].iloc[0]
            wn.add_reservoir("W_RES", base_head=float(reservoir.elevation_m), coordinates=(reservoir.lon, reservoir.lat))
            for row in nodes[~nodes["node_id"].eq("W_RES")].itertuples():
                wn.add_junction(
                    row.node_id, base_demand=float(row.demand_m3s), elevation=float(row.elevation_m),
                    coordinates=(row.lon, row.lat),
                )
            wn.add_curve(
                "W_PUMP_CURVE", "HEAD",
                [(0.0, 1.20 * rise), (total_average, rise), (3.00 * total_average, 0.85 * rise)],
            )
            wn.add_pump("W_MAIN_PUMP", "W_RES", road_node_id[source], pump_type="HEAD", pump_parameter="W_PUMP_CURVE")
            for row in pipes[pipes["link_type"].ne("pump")].itertuples():
                wn.add_pipe(
                    row.link_id, row.from_node, row.to_node, length=max(row.length_km * 1000.0, 0.5),
                    diameter=row.diameter_mm / 1000.0, roughness=row.roughness,
                )
            wn.options.time.duration = 3600
            wn.options.time.hydraulic_timestep = 3600
            wn.options.hydraulic.demand_multiplier = multiplier
            return wn

        service_ids = services["service_node"].tolist()
        peak_pressure = pd.Series(dtype=float)
        peak_velocity = pd.Series(dtype=float)
        for _ in range(5):
            peak_wn = create_network(base.WATER_PEAK_FACTOR, head_rise)
            peak_result = wntr.sim.EpanetSimulator(peak_wn).run_sim()
            peak_pressure = peak_result.node["pressure"].iloc[-1].reindex(service_ids).dropna()
            peak_velocity = peak_result.link["velocity"].iloc[-1].drop(labels=["W_MAIN_PUMP"], errors="ignore").abs()
            deficit = 20.0 - float(peak_pressure.min())
            if deficit <= 0:
                break
            head_rise += deficit + 1.0
        wn = create_network(1.0, head_rise)
        wntr.network.io.write_inpfile(wn, case_dir / "drinking_water_epanet.inp", units="LPS")
        result = wntr.sim.EpanetSimulator(wn).run_sim()
        pressure = result.node["pressure"].iloc[-1].reindex(service_ids).dropna()
        velocity = result.link["velocity"].iloc[-1].drop(labels=["W_MAIN_PUMP"], errors="ignore").abs()
        solver = {
            "created": True, "converged": bool(np.isfinite(pressure).all()),
            "building_service_nodes": len(service_ids),
            "minimum_pressure_m": float(pressure.min()), "maximum_pressure_m": float(pressure.max()),
            "maximum_velocity_m_s": float(velocity.max()),
            "peak_minimum_pressure_m": float(peak_pressure.min()),
            "peak_maximum_velocity_m_s": float(peak_velocity.max()),
            "source_head_rise_m": head_rise,
            "cycle_rank": int(model.number_of_edges() - model.number_of_nodes() + 1),
        }
    except Exception as exc:
        solver["error"] = f"{type(exc).__name__}: {exc}"

    nodes.to_csv(case_dir / "drinking_water_nodes.csv", index=False)
    pipes.to_csv(case_dir / "drinking_water_links.csv", index=False)
    services.to_csv(case_dir / "drinking_water_service_connections.csv", index=False)
    return nodes, pipes, services, solver


def generate_wastewater(
    case_dir: Path,
    road: nx.Graph,
    buildings: pd.DataFrame,
    *,
    terrain_rise_penalty: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate a directed sewer with one property connection per building."""
    attached = attach_buildings(
        buildings[buildings["wastewater_connected"].astype(bool)], road
    )
    elevations = {node: base.proxy_elevation_m(node) for node in road.nodes}
    minimum = min(elevations.values())
    span = max(max(elevations.values()) - minimum, 1.0)
    outfall = min(road, key=lambda node: elevations[node] + 8.0 * node[1])
    directed = nx.DiGraph()
    for u, v, data in road.edges(data=True):
        for a, b in ((u, v), (v, u)):
            rise = max(0.0, elevations[b] - elevations[a]) / span
            directed.add_edge(
                a, b, length_km=float(data["length_km"]),
                cost=float(data["length_km"]) * (1.0 + terrain_rise_penalty * rise),
            )
    targets = list(dict.fromkeys(attached.road_node))
    union = nx.DiGraph()
    union.add_nodes_from([outfall, *targets])
    # One reverse Dijkstra yields the least-cost next hop to the outfall for
    # every road node.  Running a full shortest-path search once per building
    # attachment is not tractable at city scale.
    reverse = directed.reverse(copy=False)
    route_distance = {outfall: 0.0}
    next_to_outfall: dict[tuple[float, float], tuple[float, float]] = {}
    queue = [(0.0, outfall)]
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
    for target in targets:
        if target not in route_distance:
            union.add_edge(
                target, outfall, length_km=base.distance_km(target, outfall),
                geometry=[target, outfall], confidence_class="D",
            )
            continue
        path = [target]
        while path[-1] != outfall:
            path.append(next_to_outfall[path[-1]])
        for u, v in zip(path[:-1], path[1:]):
            data = road[u][v]
            union.add_edge(
                u, v, length_km=float(data["length_km"]),
                geometry=geometry(data, u, v), confidence_class="B/C",
            )
    model = base.compress_directed(union, {outfall, *targets})
    model.add_nodes_from(node for node in [outfall, *targets] if node in union)
    predecessors = list(model.predecessors(outfall))
    if len(predecessors) > 1:
        terminal = (outfall[0], outfall[1] + 0.000045)
        while terminal in model:
            terminal = (terminal[0], terminal[1] + 0.00001)
        for predecessor in predecessors:
            data = dict(model[predecessor][outfall])
            model.remove_edge(predecessor, outfall)
            model.add_edge(
                predecessor, terminal,
                length_km=max(base.distance_km(predecessor, terminal), 0.005),
                geometry=[predecessor, terminal], confidence_class=data.get("confidence_class", "C"),
            )
        model.add_edge(
            terminal, outfall, length_km=0.005,
            geometry=[terminal, outfall], confidence_class="C/D",
        )

    dry_at = defaultdict(float)
    wet_at = defaultdict(float)
    annual_at = defaultdict(float)
    for row in attached.itertuples():
        dry_at[row.road_node] += float(row.wastewater_sanitary_m3_year / (365.0 * 86400.0))
        wet_at[row.road_node] += float(row.wastewater_service_peak_lps / 1000.0)
        annual_at[row.road_node] += float(row.wastewater_sanitary_m3_year)
    flow = defaultdict(float)
    for target in targets:
        if target not in model or not nx.has_path(model, target, outfall):
            continue
        path = nx.shortest_path(model, target, outfall)
        for u, v in zip(path[:-1], path[1:]):
            flow[(u, v)] += wet_at[target]

    # Cover-bounded invert design; infeasible gravity edges become explicit
    # wet-well/force-main interfaces rather than being hidden by elevation edits.
    inverts = {outfall: elevations.get(outfall, base.proxy_elevation_m(outfall)) - 3.0}
    pump_edges: set[tuple[float, float]] = set()
    for downstream in reversed(list(nx.topological_sort(model))):
        if downstream not in inverts:
            continue
        for upstream in model.predecessors(downstream):
            ground = base.proxy_elevation_m(upstream)
            length_m = float(model[upstream][downstream]["length_km"] * 1000.0)
            desired = inverts[downstream] + 0.0002 * length_m
            if desired > ground - 1.5:
                inverts[upstream] = ground - 2.0
                pump_edges.add((upstream, downstream))
            else:
                inverts[upstream] = max(desired, ground - 8.0)
    lifts = {u for u, _ in pump_edges}
    node_ids = {node: f"S_M{index:05d}" for index, node in enumerate(model, 1)}
    node_rows: list[dict[str, Any]] = []
    for node, node_id in node_ids.items():
        ground = base.proxy_elevation_m(node)
        node_rows.append({
            "node_id": node_id,
            "node_type": "outfall" if node == outfall else "lift_station" if node in lifts else "collection_manhole",
            "building_id": "", "lon": node[0], "lat": node[1],
            "ground_elevation_m": ground, "invert_elevation_m": inverts[node],
            "cover_depth_m": ground - inverts[node], "dry_inflow_m3s": 0.0,
            "annual_m3": annual_at[node],
        })
    link_rows: list[dict[str, Any]] = []
    for index, (u, v, data) in enumerate(model.edges(data=True), 1):
        q = max(float(flow[(u, v)]), 0.0002)
        length_m = max(float(data["length_km"]) * 1000.0, 1.0)
        slope = max(0.0002, (inverts[u] - inverts[v]) / length_m)
        diameter = (q * 0.013 * 4.0 ** (5.0 / 3.0) / (math.pi * math.sqrt(slope))) ** (3.0 / 8.0)
        dn = round_up(max(200, math.ceil(diameter * 1000.0)), SEWER_MAIN_DN)
        force = (u, v) in pump_edges
        link_rows.append({
            "link_id": f"S_C{index:05d}", "from_node": node_ids[u], "to_node": node_ids[v],
            "link_type": "force_main" if force else "gravity_main", "building_id": "",
            "length_km": float(data["length_km"]), "diameter_mm": dn,
            "slope": slope, "manning_n": 0.013, "design_flow_m3s": q,
            "pump_design_flow_m3s": max(q / base.SEWER_PEAK_FACTOR * 1.3, 0.0002) if force else 0.0,
            "pump_head_m": max(2.0, inverts[v] - inverts[u] + 3.0) if force else 0.0,
            "geometry_json": json.dumps(geometry(data, u, v), separators=(",", ":")),
        })

    service_rows: list[dict[str, Any]] = []
    for row in attached.itertuples():
        suffix = str(row.building_id).replace("OSM_W", "").replace("INFDB_", "I")
        property_node = f"S_B_{suffix}"
        service_id = f"S_SC_{suffix}"
        ground = base.proxy_elevation_m((row.lon, row.lat))
        main_invert = float(inverts[row.road_node])
        length_km = max(float(row.service_length_km), 0.001)
        required_gravity_invert = main_invert + 0.005 * length_km * 1000.0
        needs_property_lift = required_gravity_invert > ground - 0.50
        property_invert = (
            ground - 2.0
            if needs_property_lift
            else max(required_gravity_invert, ground - 4.0)
        )
        node_rows.append({
            "node_id": property_node,
            "node_type": "property_lift_station" if needs_property_lift else "building_property_connection",
            "building_id": row.building_id, "lon": row.lon, "lat": row.lat,
            "ground_elevation_m": ground, "invert_elevation_m": property_invert,
            "cover_depth_m": ground - property_invert,
            "dry_inflow_m3s": float(row.wastewater_sanitary_m3_year / (365.0 * 86400.0)),
            "annual_m3": float(row.wastewater_sanitary_m3_year),
        })
        service_dn = 200 if row.service_class in {"industrial", "public"} else 150
        design_flow_m3s = max(float(row.wastewater_service_peak_lps / 1000.0), 1e-6)
        link_rows.append({
            "link_id": service_id, "from_node": property_node, "to_node": node_ids[row.road_node],
            "link_type": "property_force_main" if needs_property_lift else "property_lateral",
            "building_id": row.building_id,
            "length_km": length_km, "diameter_mm": service_dn,
            "slope": 0.005, "manning_n": 0.013,
            "design_flow_m3s": design_flow_m3s,
            "pump_design_flow_m3s": max(design_flow_m3s, 1e-5) if needs_property_lift else 0.0,
            "pump_head_m": max(2.0, main_invert - property_invert + 3.0) if needs_property_lift else 0.0,
            "geometry_json": json.dumps([[row.lon, row.lat], [row.road_lon, row.road_lat]], separators=(",", ":")),
        })
        service_rows.append({
            "service_id": service_id, "building_id": row.building_id,
            "property_node": property_node, "main_node": node_ids[row.road_node],
            "service_class": row.service_class, "length_km": length_km,
            "diameter_mm": service_dn, "annual_m3": row.wastewater_sanitary_m3_year,
            "peak_flow_lps": row.wastewater_service_peak_lps,
            "connection_mode": "property_lift" if needs_property_lift else "gravity_lateral",
            "connection_evidence_class": row.connection_evidence_class,
        })
    nodes = pd.DataFrame(node_rows)
    links = pd.DataFrame(link_rows)
    services = pd.DataFrame(service_rows)

    swmm = [
        "[TITLE]", ";; Urban4 service-resolved normal dry-weather sewer", "",
        "[OPTIONS]", "FLOW_UNITS CMS", "FLOW_ROUTING KINWAVE", "FORCE_MAIN_EQUATION H-W",
        "START_DATE 07/01/2026", "START_TIME 00:00:00", "REPORT_START_DATE 07/01/2026",
        "REPORT_START_TIME 00:00:00", "END_DATE 07/01/2026", "END_TIME 06:00:00",
        "ROUTING_STEP 00:00:30", "VARIABLE_STEP 0.75", "MINIMUM_STEP 0.1", "MAX_TRIALS 100", "",
        "[JUNCTIONS]", ";;Name Elevation MaxDepth InitDepth SurDepth Aponded",
    ]
    lift_ids = set(
        nodes.loc[
            nodes.node_type.isin(["lift_station", "property_lift_station"]),
            "node_id",
        ]
    )
    for row in nodes[~nodes.node_type.eq("outfall") & ~nodes.node_id.isin(lift_ids)].itertuples():
        swmm.append(f"{row.node_id} {row.invert_elevation_m:.3f} {max(1.5,row.cover_depth_m):.3f} 0 0 0")
    swmm += ["", "[STORAGE]", ";;Name Elev MaxDepth InitDepth Shape A1 A2 A0 SurDepth Fevap"]
    for row in nodes[nodes.node_id.isin(lift_ids)].itertuples():
        swmm.append(f"{row.node_id} {row.invert_elevation_m:.3f} {max(2.0,row.cover_depth_m):.3f} 0 FUNCTIONAL 0 0 25 0 0")
    swmm += ["", "[OUTFALLS]", ";;Name Elevation Type StageData Gated"]
    for row in nodes[nodes.node_type.eq("outfall")].itertuples():
        swmm.append(f"{row.node_id} {row.invert_elevation_m:.3f} FREE NO")
    swmm += ["", "[CONDUITS]", ";;Name From To Length Roughness InOffset OutOffset InitFlow MaxFlow"]
    gravity = links[~links.link_type.isin(["force_main", "property_force_main"])]
    for row in gravity.itertuples():
        swmm.append(f"{row.link_id} {row.from_node} {row.to_node} {row.length_km*1000:.2f} {row.manning_n:.3f} 0 0 0 0")
    forces = links[links.link_type.isin(["force_main", "property_force_main"])]
    swmm += ["", "[PUMPS]", ";;Name From To Curve Status Startup Shutoff"]
    for index, row in enumerate(forces.itertuples(), 1):
        swmm.append(f"P_{index:04d} {row.from_node} {row.to_node} PC_{index:04d} ON 0.15 0.05")
    swmm += ["", "[XSECTIONS]", ";;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels"]
    for row in gravity.itertuples():
        swmm.append(f"{row.link_id} CIRCULAR {row.diameter_mm/1000:.3f} 0 0 0 1")
    swmm += ["", "[CURVES]", ";;Name Type X Y"]
    for index, row in enumerate(forces.itertuples(), 1):
        swmm += [f"PC_{index:04d} PUMP4 0 0", f"PC_{index:04d} 0.15 {row.pump_design_flow_m3s:.8f}", f"PC_{index:04d} 2.0 {row.pump_design_flow_m3s:.8f}"]
    swmm += ["", "[DWF]", ";;Node Constituent Baseline Patterns"]
    for row in nodes[nodes.dry_inflow_m3s.gt(0)].itertuples():
        swmm.append(f"{row.node_id} FLOW {row.dry_inflow_m3s:.8f}")
    swmm += ["", "[COORDINATES]", ";;Node X Y"]
    for row in nodes.itertuples():
        x, y = base.xy_km((row.lon, row.lat))
        swmm.append(f"{row.node_id} {x*1000:.2f} {y*1000:.2f}")
    input_path = case_dir / "wastewater_swmm.inp"
    input_path.write_text("\n".join(swmm) + "\n", encoding="utf-8")
    solver: dict[str, Any] = {"created": True, "engine_run_passed": False}
    try:
        from swmm.toolkit import solver as swmm_solver
        report_path = case_dir / "wastewater_swmm.rpt"
        binary_path = case_dir / "wastewater_swmm.out"
        swmm_solver.swmm_run(str(input_path), str(report_path), str(binary_path))
        report = report_path.read_text(encoding="utf-8", errors="ignore")
        def metric(pattern: str, default: float = math.inf) -> float:
            import re
            match = re.search(pattern, report)
            return float(match.group(1)) if match else default
        continuity = metric(
            r"Continuity Error \(%\)\s+\.+\s+([-+]?\d+(?:\.\d+)?)"
        )
        nonconverging = metric(
            r"% of Steps Not Converging\s*:\s*([-+]?\d+(?:\.\d+)?)"
        )
        flooding = metric(
            r"Flooding Loss \.{3,}\s+[-+]?\d+(?:\.\d+)?\s+([-+]?\d+(?:\.\d+)?)"
        )
        routing_duration_hours = 6
        extended_initialisation_window = False
        # Small dry-weather systems can retain a material fraction of the
        # first six hours' inflow while initially empty conduits fill.  If that
        # is the only failed numerical screen, extend the same topology and
        # demand state to 24 h; do not resize assets or relax the threshold.
        if (
            "ERROR " not in report
            and abs(continuity) > 1.0
            and nonconverging <= 0.1
            and flooding <= 0.001
        ):
            input_path.write_text(
                input_path.read_text(encoding="utf-8")
                .replace("END_DATE 07/01/2026", "END_DATE 07/02/2026")
                .replace("END_TIME 06:00:00", "END_TIME 00:00:00"),
                encoding="utf-8",
            )
            swmm_solver.swmm_run(
                str(input_path), str(report_path), str(binary_path)
            )
            report = report_path.read_text(encoding="utf-8", errors="ignore")
            continuity = metric(
                r"Continuity Error \(%\)\s+\.+\s+([-+]?\d+(?:\.\d+)?)"
            )
            nonconverging = metric(
                r"% of Steps Not Converging\s*:\s*([-+]?\d+(?:\.\d+)?)"
            )
            flooding = metric(
                r"Flooding Loss \.{3,}\s+[-+]?\d+(?:\.\d+)?\s+([-+]?\d+(?:\.\d+)?)"
            )
            routing_duration_hours = 24
            extended_initialisation_window = True

        sensitivity_path = case_dir / "wastewater_swmm_timestep15.inp"
        sensitivity_report_path = case_dir / "wastewater_swmm_timestep15.rpt"
        sensitivity_binary_path = case_dir / "wastewater_swmm_timestep15.out"
        sensitivity_path.write_text(
            input_path.read_text(encoding="utf-8").replace(
                "ROUTING_STEP 00:00:30", "ROUTING_STEP 00:00:15"
            ),
            encoding="utf-8",
        )
        swmm_solver.swmm_run(
            str(sensitivity_path), str(sensitivity_report_path),
            str(sensitivity_binary_path),
        )
        sensitivity_report = sensitivity_report_path.read_text(
            encoding="utf-8", errors="ignore"
        )
        import re
        def sensitivity_metric(pattern: str, default: float = math.inf) -> float:
            match = re.search(pattern, sensitivity_report)
            return float(match.group(1)) if match else default
        sensitivity_continuity = sensitivity_metric(
            r"Continuity Error \(%\)\s+\.+\s+([-+]?\d+(?:\.\d+)?)"
        )
        sensitivity_nonconverging = sensitivity_metric(
            r"% of Steps Not Converging\s*:\s*([-+]?\d+(?:\.\d+)?)"
        )
        sensitivity_flooding = sensitivity_metric(
            r"Flooding Loss \.{3,}\s+[-+]?\d+(?:\.\d+)?\s+([-+]?\d+(?:\.\d+)?)"
        )
        solver.update({
            "engine_run_passed": "ERROR " not in report,
            "building_property_connections": len(services),
            "lift_stations": len(lifts),
            "property_lift_stations": int(
                nodes.node_type.eq("property_lift_station").sum()
            ),
            "flow_routing_continuity_error_percent": continuity,
            "steps_not_converging_percent": nonconverging,
            "flooding_loss_million_liter": flooding,
            "time_step_sensitivity_15s": {
                "flow_routing_continuity_error_percent": sensitivity_continuity,
                "steps_not_converging_percent": sensitivity_nonconverging,
                "flooding_loss_million_liter": sensitivity_flooding,
            },
            "time_step_sensitivity_passed": bool(
                "ERROR " not in sensitivity_report
                and abs(sensitivity_continuity) <= 1.0
                and sensitivity_nonconverging <= 0.1
                and sensitivity_flooding <= 0.001
            ),
            "routing_method": "KINWAVE",
            "routing_duration_hours": routing_duration_hours,
            "extended_initialisation_window": extended_initialisation_window,
        })
    except Exception as exc:
        solver["error"] = f"{type(exc).__name__}: {exc}"
    nodes.to_csv(case_dir / "wastewater_nodes.csv", index=False)
    links.to_csv(case_dir / "wastewater_links.csv", index=False)
    services.to_csv(case_dir / "wastewater_service_connections.csv", index=False)
    return nodes, links, services, solver


def generate_district_heat(
    case_dir: Path,
    road: nx.Graph,
    buildings: pd.DataFrame,
    *,
    connection_count: int,
    plant_capacity_mw: float = 30.0,
    seed: int = 20260802,
    annual_heat_target_mwh: float | None = None,
    design_peak_target_mw: float | None = None,
    minimum_heat_density_mwh_ha: float = 800.0,
    heat_density_cell_size_km: float = 0.25,
    enforce_connection_count: bool = False,
    selection_policy: str = "demand_ranked",
    route_aware_demand_exponent: float = 1.0,
    route_aware_spatial_quota_level: str = "cell",
    published_network_length_km: float | None = None,
    network_length_tolerance_fraction: float = 0.10,
    network_length_scope: str = "route_plus_service_one_way",
    return_pressure_reference_bar: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate paired heat routes inside accepted heat-suitable territories.

    ``selection_policy='route_aware'`` replaces purely demand-ranked customer
    truncation with a non-monetary, quota-constrained prize-collecting-forest
    heuristic.  Annual heat is the prize; only newly added corridor length and
    the building service are penalised.  Public contract totals remain imposed
    quotas rather than behavioural or tariff assumptions.

    In the integrated city case, a published aggregate network length can be
    supplied as a calibration constraint.  Urban4 compares that inventory with
    the one-way generated network length, defined explicitly as routed mains
    plus building service connections.  Paired supply/return pipe length is
    exported separately and is never compared with the one-way inventory.
    """
    eligible = attach_buildings(buildings[buildings.heat_eligible.astype(bool)], road)
    selection_audit = pd.DataFrame()
    if selection_policy == "demand_ranked":
        selected, suitability = select_heat_service_territories(
            eligible,
            maximum_connections=min(connection_count, len(eligible)),
            cell_size_km=heat_density_cell_size_km,
            minimum_heat_density_mwh_ha=minimum_heat_density_mwh_ha,
            enforce_connection_count=enforce_connection_count,
        )
        suitability["selection_policy"] = "demand_ranked_within_suitable_territories"
    elif selection_policy == "route_aware":
        # First retain every building in cells that pass the density screen;
        # the route-aware stage then chooses the declared partial connection set.
        candidates, suitability = select_heat_service_territories(
            eligible,
            maximum_connections=len(eligible),
            cell_size_km=heat_density_cell_size_km,
            minimum_heat_density_mwh_ha=minimum_heat_density_mwh_ha,
            enforce_connection_count=False,
        )
        selected, route_metadata, selection_audit = select_route_aware_heat_connections(
            road, candidates, target_count=min(connection_count, len(candidates)),
            config=RouteAwareHeatConfig(
                demand_exponent=route_aware_demand_exponent,
                spatial_quota_level=route_aware_spatial_quota_level,
            ),
        )
        suitability.update(route_metadata)
    else:
        raise ValueError(
            "selection_policy must be 'demand_ranked' or 'route_aware'"
        )
    if selected.empty:
        empty_nodes = pd.DataFrame(columns=[
            "node_id", "node_type", "building_id", "lon", "lat", "elevation_m",
            "annual_heat_mwh", "peak_heat_mw",
        ])
        empty_links = pd.DataFrame(columns=[
            "corridor_id", "from_node", "to_node", "link_type", "building_id",
            "length_km", "diameter_mm", "design_peak_mw", "mass_flow_kg_s",
            "geometry_json",
        ])
        empty_services = pd.DataFrame(columns=[
            "service_id", "building_id", "substation_node", "route_node",
            "heat_source_node", "heat_source_map_id", "service_class", "length_km",
            "annual_heat_mwh", "peak_heat_mw", "connection_evidence_class",
        ])
        empty_sources = pd.DataFrame(columns=[
            "map_id", "source_node", "lon", "lat", "planning_capacity_mw_th",
            "assigned_peak_mw_th", "assigned_annual_heat_mwh", "customer_count",
            "evidence_role",
        ])
        solver = {
            "created": True,
            "converged": True,
            "network_present": False,
            "consumer_substations": 0,
            "minimum_pressure_bar": 25.0,
            "return_pressure_reference_bar": float(return_pressure_reference_bar),
            "minimum_differential_pressure_bar": float(
                25.0 - return_pressure_reference_bar
            ),
            "maximum_velocity_m_s": 0.0,
            "annual_heat_loss_fraction": 0.0,
            "circulation_pump_power_mw": 0.0,
            "one_way_network_length_km": 0.0,
            "published_network_length_km": (
                None if published_network_length_km is None
                else float(published_network_length_km)
            ),
            "network_length_scope": network_length_scope,
            "inventory_length_relative_error": (
                math.nan if published_network_length_km is None else 1.0
            ),
            "inventory_length_tolerance_fraction": float(network_length_tolerance_fraction),
            "inventory_length_within_tolerance": (
                published_network_length_km is None
            ),
            "suitability": suitability,
            "model_scope": "no district-heating territory passed the declared suitability screen",
        }
        empty_nodes.to_csv(case_dir / "district_heating_nodes.csv", index=False)
        empty_links.to_csv(case_dir / "district_heating_corridors.csv", index=False)
        empty_links.to_csv(case_dir / "district_heating_supply_return_pipes.csv", index=False)
        empty_services.to_csv(case_dir / "district_heating_service_connections.csv", index=False)
        empty_sources.to_csv(case_dir / "district_heating_sources.csv", index=False)
        selection_audit.to_csv(case_dir / "district_heating_selection_audit.csv", index=False)
        return empty_nodes, empty_links, empty_services, solver
    selected["district_heat_mwh_year"] = selected.heat_candidate_mwh_year
    selected["district_heat_peak_mw"] = selected.heat_service_design_kw / 1000.0
    if annual_heat_target_mwh is not None:
        selected["district_heat_mwh_year"] *= float(annual_heat_target_mwh) / selected.district_heat_mwh_year.sum()
    if design_peak_target_mw is not None:
        selected["district_heat_peak_mw"] *= float(design_peak_target_mw) / selected.district_heat_peak_mw.sum()
    selected["plant_index"] = -1
    next_plant_index = 0
    for _, territory in selected.groupby("heat_territory_id"):
        territory_plant_count = max(
            1, int(math.ceil(territory.district_heat_peak_mw.sum() / plant_capacity_mw))
        )
        xy = np.asarray([base.xy_km(row.road_node) for row in territory.itertuples()])
        if territory_plant_count == 1:
            labels = np.zeros(len(territory), dtype=int)
        else:
            _, labels = kmeans2(
                xy, min(territory_plant_count, len(territory)), minit="++", iter=60,
                seed=seed + int(territory.heat_territory_id.iloc[0]),
            )
        selected.loc[territory.index, "plant_index"] = labels + next_plant_index
        next_plant_index += territory_plant_count
    selected["plant_index"] = selected.plant_index.astype(int)
    road_index = base.NodeIndex(road.nodes)
    components = []
    plant_nodes = set()
    for plant_index, group in selected.groupby("plant_index"):
        weights = np.maximum(group.district_heat_peak_mw.to_numpy(float), 0.001)
        centre = (
            float(np.average(group.road_lon, weights=weights)),
            float(np.average(group.road_lat, weights=weights)),
        )
        plant = road_index.nearest(centre)
        plant_nodes.add(plant)
        components.append((plant, _union_tree(road, plant, list(group.road_node)), group))
    all_road_nodes = sorted({node for _, tree, _ in components for node in tree})
    road_ids = {node: f"H_M{index:05d}" for index, node in enumerate(all_road_nodes, 1)}
    # Heat-source nodes are synthetic hydraulic/thermal boundaries used to
    # partition suitable territories.  The planning-capacity value is a
    # catchment ceiling, not an observed boiler/CHP/heat-pump rating.
    source_rows = []
    source_for_building: dict[str, str] = {}
    for plant, _, group in components:
        source_node = road_ids[plant]
        source_rows.append({
            "source_node": source_node,
            "lon": float(plant[0]),
            "lat": float(plant[1]),
            "planning_capacity_mw_th": float(plant_capacity_mw),
            "assigned_peak_mw_th": float(group.district_heat_peak_mw.sum()),
            "assigned_annual_heat_mwh": float(group.district_heat_mwh_year.sum()),
            "customer_count": int(len(group)),
            "evidence_role": "synthetic heat-source boundary; not an observed installed plant",
        })
        for building_id in group.building_id.astype(str):
            source_for_building[building_id] = source_node
    sources = pd.DataFrame(source_rows).sort_values(
        ["lat", "lon"], ascending=[False, True]
    ).reset_index(drop=True)
    sources.insert(0, "map_id", [f"H{index}" for index in range(1, len(sources) + 1)])
    source_label_by_node = dict(zip(sources.source_node, sources.map_id))
    node_rows = []
    for node in all_road_nodes:
        node_rows.append({
            "node_id": road_ids[node], "node_type": "plant" if node in plant_nodes else "heat_route_junction",
            "building_id": "", "lon": node[0], "lat": node[1],
            "elevation_m": base.proxy_elevation_m(node), "annual_heat_mwh": 0.0,
            "peak_heat_mw": 0.0,
        })
    corridor_rows = []
    used = set()
    for plant, tree, group in components:
        peak_at = defaultdict(float)
        for row in group.itertuples():
            peak_at[row.road_node] += float(row.district_heat_peak_mw)
        parent = {child: ancestor for ancestor, child in nx.bfs_edges(tree, plant)}
        subtree = {node: peak_at[node] for node in tree}
        for node in reversed(list(nx.bfs_tree(tree, plant).nodes)):
            if node != plant:
                subtree[parent[node]] += subtree[node]
        for child, ancestor in parent.items():
            key = frozenset((ancestor, child))
            if key in used:
                continue
            used.add(key)
            data = tree[ancestor][child]
            peak = max(float(subtree[child]), 0.02)
            mass = peak * 1e6 / (4180.0 * 35.0)
            corridor_rows.append({
                "corridor_id": f"H_R{len(corridor_rows)+1:05d}",
                "from_node": road_ids[ancestor], "to_node": road_ids[child],
                "link_type": "route", "building_id": "", "length_km": float(data["length_km"]),
                "diameter_mm": choose_heat_dn(mass),
                "design_peak_mw": peak, "mass_flow_kg_s": mass,
                "geometry_json": json.dumps(geometry(data, ancestor, child), separators=(",", ":")),
            })
    service_rows = []
    for row in selected.itertuples():
        suffix = str(row.building_id).replace("OSM_W", "").replace("INFDB_", "I")
        node_id = f"H_B_{suffix}"
        node_rows.append({
            "node_id": node_id, "node_type": "consumer_substation", "building_id": row.building_id,
            "lon": row.lon, "lat": row.lat, "elevation_m": base.proxy_elevation_m((row.lon, row.lat)),
            "annual_heat_mwh": float(row.district_heat_mwh_year),
            "peak_heat_mw": float(row.district_heat_peak_mw),
        })
        peak = max(float(row.district_heat_peak_mw), 0.001)
        mass = peak * 1e6 / (4180.0 * 35.0)
        service_id = f"H_SC_{suffix}"
        corridor_rows.append({
            "corridor_id": service_id, "from_node": road_ids[row.road_node], "to_node": node_id,
            "link_type": "building_service", "building_id": row.building_id,
            "length_km": max(float(row.service_length_km), 0.001),
            "diameter_mm": choose_heat_dn(
                mass, maximum_velocity_m_s=1.0,
                maximum_pressure_gradient_pa_m=150.0,
            ),
            "design_peak_mw": peak, "mass_flow_kg_s": mass,
            "geometry_json": json.dumps([[row.road_lon, row.road_lat], [row.lon, row.lat]], separators=(",", ":")),
        })
        service_rows.append({
            "service_id": service_id, "building_id": row.building_id,
            "substation_node": node_id, "route_node": road_ids[row.road_node],
            "heat_source_node": source_for_building[str(row.building_id)],
            "heat_source_map_id": source_label_by_node[source_for_building[str(row.building_id)]],
            "service_class": row.service_class, "length_km": max(float(row.service_length_km), 0.001),
            "annual_heat_mwh": row.district_heat_mwh_year, "peak_heat_mw": peak,
            "connection_evidence_class": row.connection_evidence_class,
            "selection_policy": getattr(row, "heat_selection_policy", suitability.get("selection_policy", "demand_ranked")),
            "incremental_trench_km": getattr(row, "incremental_trench_km", math.nan),
            "route_aware_score_mwh_per_km": getattr(row, "route_aware_score_mwh_per_km", math.nan),
        })
    nodes = pd.DataFrame(node_rows)
    corridors = pd.DataFrame(corridor_rows)
    services = pd.DataFrame(service_rows)
    paired_rows = []
    for row in corridors.itertuples():
        for circuit, from_node, to_node in (("supply", row.from_node, row.to_node), ("return", row.to_node, row.from_node)):
            paired_rows.append({
                "pipe_id": f"{row.corridor_id}_{circuit[0].upper()}", "corridor_id": row.corridor_id,
                "circuit": circuit, "from_node": from_node, "to_node": to_node,
                "link_type": row.link_type, "building_id": row.building_id,
                "length_km": row.length_km, "diameter_mm": row.diameter_mm,
                "design_peak_mw": row.design_peak_mw, "mass_flow_kg_s": row.mass_flow_kg_s,
                "geometry_json": row.geometry_json,
            })
    paired = pd.DataFrame(paired_rows)
    route_mask = corridors.link_type.eq("route")
    service_mask = corridors.link_type.eq("building_service")
    route_trench_length_km = float(corridors.loc[route_mask, "length_km"].sum())
    service_connection_length_km = float(corridors.loc[service_mask, "length_km"].sum())
    one_way_network_length_km = route_trench_length_km + service_connection_length_km
    if network_length_scope != "route_plus_service_one_way":
        raise ValueError(
            "network_length_scope must be 'route_plus_service_one_way'"
        )
    if published_network_length_km is None:
        inventory_length_relative_error = math.nan
        inventory_length_within_tolerance = True
    else:
        target_length = max(float(published_network_length_km), 1e-9)
        inventory_length_relative_error = abs(one_way_network_length_km - target_length) / target_length
        inventory_length_within_tolerance = (
            inventory_length_relative_error <= float(network_length_tolerance_fraction)
        )
    annual_heat_loss_mwh = float(
        25.0 * route_trench_length_km * 1000.0 * 8760.0 / 1e6
    )
    density = 985.0
    velocity_values = []
    gradient_values = []
    for row in corridors.itertuples():
        diameter_m = max(float(row.diameter_mm) / 1000.0, 1e-6)
        velocity = float(row.mass_flow_kg_s) / (density * math.pi * diameter_m**2 / 4.0)
        gradient = 0.02 * density * velocity**2 / (2.0 * diameter_m)
        velocity_values.append(velocity)
        gradient_values.append(gradient)
    structural_metrics = {
        "network_present": True,
        "consumer_substations": len(services),
        "route_trench_length_km": route_trench_length_km,
        "service_connection_length_km": service_connection_length_km,
        "one_way_network_length_km": one_way_network_length_km,
        "published_network_length_km": (
            None if published_network_length_km is None
            else float(published_network_length_km)
        ),
        "network_length_scope": network_length_scope,
        "inventory_length_relative_error": inventory_length_relative_error,
        "inventory_length_tolerance_fraction": float(network_length_tolerance_fraction),
        "inventory_length_within_tolerance": bool(inventory_length_within_tolerance),
        "paired_physical_pipe_length_km": float(2.0 * corridors.length_km.sum()),
        "annual_heat_loss_mwh": annual_heat_loss_mwh,
        "annual_heat_loss_fraction": annual_heat_loss_mwh / max(float(services.annual_heat_mwh.sum()), 1e-9),
        "catalogue_maximum_velocity_m_s": max(velocity_values, default=0.0),
        "catalogue_maximum_pressure_gradient_pa_m": max(gradient_values, default=0.0),
        "selection_policy": suitability.get("selection_policy", selection_policy),
        "suitability": suitability,
    }
    solver: dict[str, Any] = {
        "created": False, "converged": False, **structural_metrics,
        "native_solver_status": "pandapipes not yet executed for this generated candidate",
    }
    try:
        import pandapipes as ppipe
        def solve_native_heat(candidate: pd.DataFrame):
            network = ppipe.create_empty_network(fluid="water")
            mapping = {
                row.node_id: ppipe.create_junction(
                    network, pn_bar=25.0, tfluid_k=363.15,
                    height_m=row.elevation_m, name=row.node_id,
                    geodata=(row.lon, row.lat),
                )
                for row in nodes.itertuples()
            }
            for plant in nodes[nodes.node_type.eq("plant")].itertuples():
                ppipe.create_ext_grid(
                    network, mapping[plant.node_id], p_bar=25.0,
                    t_k=363.15, type="pt", name=str(plant.node_id),
                )
            for row in candidate.itertuples():
                ppipe.create_pipe_from_parameters(
                    network, mapping[row.from_node], mapping[row.to_node],
                    max(row.length_km, 0.001),
                    inner_diameter_mm=row.diameter_mm, k_mm=0.1,
                )
            for row in nodes[nodes.node_type.eq("consumer_substation")].itertuples():
                ppipe.create_sink(
                    network, mapping[row.node_id],
                    max(row.peak_heat_mw * 1e6 / (4180.0 * 35.0), 1e-5),
                )
            ppipe.pipeflow(network, mode="hydraulics", max_iter_hyd=100)
            return network

        # Shared road sections can carry flow from more than one heat territory.
        # Their actual native-solver flow can therefore exceed the preliminary
        # subtree estimate. Advance only violating pipes by one declared DN class
        # and rerun; the velocity threshold itself is never relaxed.
        diameter_resize_iterations = 0
        diameter_upgraded_links = 0
        while True:
            net = solve_native_heat(corridors)
            violations = net.res_pipe.v_mean_m_per_s.abs() > 2.0
            if not bool(violations.any()) or diameter_resize_iterations >= 6:
                break
            changed = 0
            for position in np.flatnonzero(violations.to_numpy()):
                row_index = corridors.index[int(position)]
                current = int(corridors.loc[row_index, "diameter_mm"])
                larger = next((value for value in HEAT_DN if value > current), None)
                if larger is not None:
                    corridors.loc[row_index, "diameter_mm"] = larger
                    changed += 1
            diameter_resize_iterations += 1
            diameter_upgraded_links += changed
            if changed == 0:
                break

        final_diameter = corridors.set_index("corridor_id")["diameter_mm"]
        paired["diameter_mm"] = paired["corridor_id"].map(final_diameter).astype(int)
        ppipe.to_json(net, str(case_dir / "district_heating_pandapipes.json"))
        solver = {
            "created": True, "converged": bool(net.converged),
            "consumer_substations": len(services), "minimum_pressure_bar": float(net.res_junction.p_bar.min()),
            "return_pressure_reference_bar": float(return_pressure_reference_bar),
            "minimum_differential_pressure_bar": float(
                net.res_junction.p_bar.min() - return_pressure_reference_bar
            ),
            "maximum_velocity_m_s": float(net.res_pipe.v_mean_m_per_s.abs().max()),
            "diameter_resize_iterations": diameter_resize_iterations,
            "diameter_upgraded_links": diameter_upgraded_links,
            "paired_physical_pipe_length_km": structural_metrics["paired_physical_pipe_length_km"],
            "route_trench_length_km": route_trench_length_km,
            "service_connection_length_km": service_connection_length_km,
            "one_way_network_length_km": one_way_network_length_km,
            "published_network_length_km": (
                None if published_network_length_km is None
                else float(published_network_length_km)
            ),
            "network_length_scope": network_length_scope,
            "inventory_length_relative_error": inventory_length_relative_error,
            "inventory_length_tolerance_fraction": float(network_length_tolerance_fraction),
            "inventory_length_within_tolerance": bool(inventory_length_within_tolerance),
            "annual_heat_loss_mwh": annual_heat_loss_mwh,
            "annual_heat_loss_fraction": structural_metrics["annual_heat_loss_fraction"],
            "catalogue_maximum_pressure_gradient_pa_m": structural_metrics["catalogue_maximum_pressure_gradient_pa_m"],
            "selection_policy": structural_metrics["selection_policy"],
            "circulation_pump_power_mw": float(
                max(25.0 - float(net.res_junction.p_bar.min()), 0.0) * 1e5
                * (selected.district_heat_peak_mw.sum() * 1e6 / (4180.0 * 35.0) / 985.0)
                / 0.75 / 1e6
            ),
            "suitability": suitability,
            "model_scope": "steady hydraulic screen on the generated supply graph; paired supply/return topology and route-level heat-loss diagnostic exported",
        }
    except Exception as exc:
        solver["error"] = f"{type(exc).__name__}: {exc}"
    nodes.to_csv(case_dir / "district_heating_nodes.csv", index=False)
    corridors.to_csv(case_dir / "district_heating_corridors.csv", index=False)
    paired.to_csv(case_dir / "district_heating_supply_return_pipes.csv", index=False)
    services.to_csv(case_dir / "district_heating_service_connections.csv", index=False)
    sources.to_csv(case_dir / "district_heating_sources.csv", index=False)
    selection_audit.to_csv(case_dir / "district_heating_selection_audit.csv", index=False)
    return nodes, corridors, services, solver
