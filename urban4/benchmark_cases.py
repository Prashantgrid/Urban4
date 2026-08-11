#!/usr/bin/env python3
"""Independent dense, semi-urban and rural four-sector benchmark runs.

Each benchmark clips the common evidence first and then reinitializes demand
clustering, sources, facilities, topology and sizing.  The resulting files are
therefore independent generation runs rather than crops of the Schweinfurt
integrated output.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

from . import schweinfurt_base as base
from .framework import (
    CASE,
    OUT,
    PROJECT,
    _parse_swmm_report_metrics,
    _round_up,
    _screen_solver_results,
    load_case,
)
from .service_topologies import (
    generate_district_heat,
    generate_electricity,
    generate_wastewater,
    generate_water,
)


BENCHMARK_OUT = OUT / "benchmark_cases"
WATER_DN = [80, 100, 125, 150, 200, 250, 300, 400, 500, 600]
SEWER_DN = [200, 250, 300, 400, 500, 600, 800, 1000, 1200]
HEAT_DN = [25, 32, 40, 50, 65, 80, 100, 125, 150, 200, 250, 300, 400, 500, 600]
LV_LIBRARY = [(35, 0.115, 0.868), (70, 0.179, 0.443), (95, 0.216, 0.320), (150, 0.282, 0.206), (240, 0.368, 0.125)]


def _inside(point: tuple[float, float], bbox: list[float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _tile_road_graph(bbox: list[float]) -> nx.Graph:
    complete = base.build_road_graph(base.load_elements("local_roads.json"))
    nodes = [node for node in complete if _inside(node, bbox)]
    graph = complete.subgraph(nodes).copy()
    if not graph:
        raise RuntimeError(f"No road evidence in {bbox}")
    component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(component).copy()


def _cluster_zones(buildings: pd.DataFrame, road: nx.Graph, count: int, seed: int) -> pd.DataFrame:
    index = base.NodeIndex(road.nodes)
    xy = np.asarray([base.xy_km((r.lon, r.lat)) for r in buildings.itertuples()])
    count = min(count, max(3, len(buildings) // 12))
    _, labels = kmeans2(xy, count, minit="++", iter=60, seed=seed)
    rows: dict[tuple[float, float], dict[str, Any]] = {}
    for cluster in range(count):
        members = buildings.iloc[np.where(labels == cluster)[0]]
        if members.empty:
            continue
        weights = np.maximum(members["floor_area_m2"].to_numpy(dtype=float), 20.0)
        centroid = (float(np.average(members["lon"], weights=weights)), float(np.average(members["lat"], weights=weights)))
        snapped = index.nearest(centroid)
        item = rows.setdefault(
            snapped,
            {
                "lon": snapped[0], "lat": snapped[1], "building_count": 0,
                "electricity_peak_mw": 0.0, "electricity_mwh_year": 0.0,
                "water_m3_year": 0.0, "wastewater_m3_year": 0.0,
                "heat_mwh_year": 0.0, "population": 0.0,
            },
        )
        item["building_count"] += int(len(members))
        item["electricity_peak_mw"] += float(members["electricity_peak_kw"].sum() / 1000.0)
        item["electricity_mwh_year"] += float(members["electricity_mwh_year"].sum())
        item["water_m3_year"] += float(members["water_m3_year"].sum())
        item["wastewater_m3_year"] += float(members["wastewater_sanitary_m3_year"].sum())
        item["heat_mwh_year"] += float(members["heat_candidate_mwh_year"].sum())
        item["population"] += float(members["population"].sum())
    zones = pd.DataFrame(rows.values())
    zones.insert(0, "zone_id", [f"Z{i:03d}" for i in range(1, len(zones) + 1)])
    zones["water_q_avg_m3s"] = zones["water_m3_year"] / (365.0 * 86400.0)
    zones["water_q_peak_m3s"] = 2.0 * zones["water_q_avg_m3s"]
    zones["sewer_q_dry_m3s"] = zones["wastewater_m3_year"] / (365.0 * 86400.0)
    zones["sewer_q_wet_m3s"] = 3.6 * zones["sewer_q_dry_m3s"]
    zones["heat_peak_mw"] = zones["heat_mwh_year"] / 1500.0
    return zones


def _union_tree(road: nx.Graph, source: tuple[float, float], targets: list[tuple[float, float]]) -> nx.Graph:
    union = nx.Graph()
    union.add_nodes_from([source, *targets])
    for target in targets:
        try:
            base.add_path_edges(union, road, nx.shortest_path(road, source, target, weight="length_km"))
        except nx.NetworkXNoPath:
            union.add_edge(source, target, length_km=base.distance_km(source, target), geometry=[source, target])
    tree = nx.minimum_spanning_tree(union, weight="length_km")
    model = base.compress_undirected(tree, {source, *targets})
    model.add_nodes_from(node for node in [source, *targets] if node in tree)
    return model


def _geometry(data: dict[str, Any], u: tuple[float, float], v: tuple[float, float]) -> list[tuple[float, float]]:
    geometry = data.get("geometry", [u, v])
    if tuple(geometry[0]) != u:
        geometry = list(reversed(geometry))
    return geometry


def _run_power(
    case_dir: Path,
    road: nx.Graph,
    zones: pd.DataFrame,
    buildings: pd.DataFrame,
    maximum_feeder_radius_km: float | None = None,
    seed: int = 20260802,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    peak = float(zones["electricity_peak_mw"].sum())
    transformer_count = int(np.clip(max(math.ceil(peak / 0.50), math.ceil(len(zones) / 4)), 3, 30))
    xy = np.asarray([base.xy_km((r.lon, r.lat)) for r in buildings.itertuples()])
    centres, _ = kmeans2(xy, transformer_count, minit="++", iter=60, seed=seed)
    road_index = base.NodeIndex(road.nodes)
    sites = []
    for x, y in centres:
        site = road_index.nearest((float(x / 71.55), float(y / 111.20)))
        if site not in sites:
            sites.append(site)
    zone_points = [(float(r.lon), float(r.lat)) for r in zones.itertuples()]
    while len(sites) < transformer_count:
        sites.append(max((point for point in zone_points if point not in sites), key=lambda p: min(base.distance_km(p, s) for s in sites)))
    sites = sites[:transformer_count]
    width_km = max(base.distance_km((buildings["lon"].min(), buildings["lat"].min()), (buildings["lon"].max(), buildings["lat"].min())), 0.1)
    height_km = max(base.distance_km((buildings["lon"].min(), buildings["lat"].min()), (buildings["lon"].min(), buildings["lat"].max())), 0.1)
    building_density = len(buildings) / (width_km * height_km)
    if maximum_feeder_radius_km is None:
        maximum_feeder_radius_km = 0.45 if building_density >= 800 else 0.70 if building_density >= 400 else 1.20
    while len(sites) < len(zone_points):
        distances, _ = nx.multi_source_dijkstra(road, sites, weight="length_km")
        farthest = max(zone_points, key=lambda point: distances.get(point, math.inf))
        if distances.get(farthest, math.inf) <= maximum_feeder_radius_km:
            break
        sites.append(farthest)
    transformer_count = len(sites)
    try:
        _, paths = nx.multi_source_dijkstra(road, sites, weight="length_km")
        lookup = {site: index for index, site in enumerate(sites)}
        zones["transformer_index"] = [lookup[paths[point][0]] for point in zone_points]
    except nx.NetworkXNoPath:
        tree = cKDTree(np.asarray([base.xy_km(site) for site in sites]))
        _, zones["transformer_index"] = tree.query(np.asarray([base.xy_km(point) for point in zone_points]))

    bus_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    transformer_rows = []
    junction_index = 1
    line_index = 1
    standards = [0.25, 0.4, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0]
    for site_index, source in enumerate(sites):
        group = zones[zones["transformer_index"] == site_index]
        assigned_peak = float(group["electricity_peak_mw"].sum())
        required = assigned_peak / 0.80
        size = next((value for value in standards if value >= required), standards[-1])
        hv_bus, lv_bus = f"E_HV{site_index + 1:03d}", f"E_TR{site_index + 1:03d}"
        common = {"lon": source[0], "lat": source[1], "root": False, "annual_electricity_mwh": 0.0, "electricity_peak_mw": 0.0, "q_mvar": 0.0}
        bus_rows.append({"bus_id": hv_bus, "node_type": "upstream_boundary", "voltage_kv": 20.0, **common, "root": True})
        bus_rows.append({"bus_id": lv_bus, "node_type": "transformer", "voltage_kv": 0.4, **common})
        transformer_rows.append({"transformer_id": f"E_TRF{site_index + 1:03d}", "hv_bus": hv_bus, "lv_bus": lv_bus, "sn_mva": size, "assigned_peak_mw": assigned_peak, "lon": source[0], "lat": source[1]})
        targets = [(float(r.lon), float(r.lat)) for r in group.itertuples()]
        model = _union_tree(road, source, targets)
        node_ids = {source: lv_bus}
        zone_at = {(float(r.lon), float(r.lat)): r for r in group.itertuples()}
        for node in model:
            if node == source:
                continue
            if node in zone_at:
                zone = zone_at[node]
                bus_id = f"E_{zone.zone_id}"
                bus_rows.append({"bus_id": bus_id, "node_type": "load", "voltage_kv": 0.4, "root": False, "lon": node[0], "lat": node[1], "annual_electricity_mwh": zone.electricity_mwh_year, "electricity_peak_mw": zone.electricity_peak_mw, "q_mvar": zone.electricity_peak_mw * math.tan(math.acos(0.96))})
            else:
                bus_id = f"E_J{junction_index:04d}"
                junction_index += 1
                bus_rows.append({"bus_id": bus_id, "node_type": "junction", "voltage_kv": 0.4, "root": False, "lon": node[0], "lat": node[1], "annual_electricity_mwh": 0.0, "electricity_peak_mw": 0.0, "q_mvar": 0.0})
            node_ids[node] = bus_id
        if source in zone_at:
            zone = zone_at[source]
            lv_row = next(row for row in bus_rows if row["bus_id"] == lv_bus)
            lv_row.update(node_type="transformer_load", annual_electricity_mwh=zone.electricity_mwh_year, electricity_peak_mw=zone.electricity_peak_mw, q_mvar=zone.electricity_peak_mw * math.tan(math.acos(0.96)))
        parent = {child: ancestor for ancestor, child in nx.bfs_edges(model, source)}
        path_lengths = nx.single_source_dijkstra_path_length(model, source, weight="length_km")
        maximum_path_length = max(path_lengths.values(), default=0.001)
        direct = {node: float(zone_at[node].electricity_peak_mw) if node in zone_at else 0.0 for node in model}
        subtree = dict(direct)
        for node in reversed(list(nx.bfs_tree(model, source).nodes)):
            if node != source:
                subtree[parent[node]] += subtree[node]
        for child, ancestor in parent.items():
            data = model[ancestor][child]
            design = max(subtree[child], 0.001)
            current = design / (math.sqrt(3.0) * 0.4 * 0.96)
            edge_drop_limit = max(
                0.004,
                0.045 * float(data["length_km"]) / max(maximum_path_length, 0.001),
            )
            candidates = []
            for cable in LV_LIBRARY:
                uncorrected_drop = math.sqrt(3.0) * current * cable[2] * float(data["length_km"]) / 0.4
                circuits = max(
                    1,
                    math.ceil(current / (0.80 * cable[1])),
                    math.ceil(uncorrected_drop / edge_drop_limit),
                )
                if circuits <= 16:
                    installed_area = cable[0] * circuits * (1.0 + 0.10 * (circuits - 1))
                    candidates.append((installed_area, cable, circuits))
            if candidates:
                _, cable, circuits = min(candidates, key=lambda item: item[0])
            else:
                cable = LV_LIBRARY[-1]
                circuits = max(1, math.ceil(current / (0.80 * cable[1])), math.ceil(math.sqrt(3.0) * current * cable[2] * float(data["length_km"]) / (0.4 * edge_drop_limit)))
            line_rows.append({"line_id": f"E_L{line_index:04d}", "from_node": node_ids[ancestor], "to_node": node_ids[child], "length_km": float(data["length_km"]), "cable_size_mm2": cable[0], "parallel_circuits": circuits, "r_ohm_per_km": cable[2], "x_ohm_per_km": 0.08, "max_i_ka": cable[1], "normally_open": False, "geometry_json": json.dumps(_geometry(data, ancestor, child), separators=(",", ":"))})
            line_index += 1
    buses = pd.DataFrame(bus_rows).drop_duplicates("bus_id", keep="last")
    lines = pd.DataFrame(line_rows)
    transformers = pd.DataFrame(transformer_rows)
    solver: dict[str, Any] = {"created": False, "converged": False}
    try:
        import pandapower as pp

        net = pp.create_empty_network(sn_mva=100.0)
        mapping = {row.bus_id: pp.create_bus(net, vn_kv=row.voltage_kv, name=row.bus_id, geodata=(row.lon, row.lat)) for row in buses.itertuples()}
        for row in buses[buses["root"]].itertuples():
            pp.create_ext_grid(net, mapping[row.bus_id], vm_pu=1.05)
        for row in buses[buses["electricity_peak_mw"] > 0].itertuples():
            pp.create_load(net, mapping[row.bus_id], p_mw=row.electricity_peak_mw, q_mvar=row.q_mvar)
        for row in transformers.itertuples():
            pp.create_transformer_from_parameters(net, mapping[row.hv_bus], mapping[row.lv_bus], sn_mva=row.sn_mva, vn_hv_kv=20.0, vn_lv_kv=0.4, vk_percent=6.0, vkr_percent=0.85, pfe_kw=1.0, i0_percent=0.2, shift_degree=0.0, tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2, tap_step_percent=2.5, tap_pos=-2)
        for row in lines.itertuples():
            pp.create_line_from_parameters(net, mapping[row.from_node], mapping[row.to_node], max(row.length_km, 0.001), row.r_ohm_per_km, row.x_ohm_per_km, 210.0, row.max_i_ka, parallel=int(row.parallel_circuits))
        for algorithm in ["bfsw", "iwamoto_nr", "nr"]:
            try:
                pp.runpp(net, algorithm=algorithm, numba=False, max_iteration=80)
                if net.converged:
                    break
            except Exception:
                continue
        if not net.converged:
            raise RuntimeError("power flow did not converge")
        pp.to_json(net, str(case_dir / "electricity_pandapower.json"))
        solver = {"created": True, "converged": True, "minimum_voltage_pu": float(net.res_bus.vm_pu.min()), "maximum_voltage_pu": float(net.res_bus.vm_pu.max()), "maximum_line_loading_percent": float(net.res_line.loading_percent.max()), "maximum_transformer_loading_percent": float(net.res_trafo.loading_percent.max())}
    except Exception as exc:
        solver["error"] = f"{type(exc).__name__}: {exc}"
    buses.to_csv(case_dir / "electricity_nodes.csv", index=False)
    lines.to_csv(case_dir / "electricity_links.csv", index=False)
    transformers.to_csv(case_dir / "electricity_transformers.csv", index=False)
    return buses, lines, transformers, solver


def _run_water(case_dir: Path, road: nx.Graph, zones: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    centre_lat = float(zones["lat"].median())
    source = min(road.nodes, key=lambda p: abs(p[1] - centre_lat) + 0.25 * (p[0] - min(n[0] for n in road))**2)
    targets = [(float(r.lon), float(r.lat)) for r in zones.itertuples()]
    model = _union_tree(road, source, targets)
    base_length = sum(d["length_km"] for _, _, d in model.edges(data=True))
    added = 0.0
    target_xy = np.asarray([base.xy_km(point) for point in targets])
    neighbour_tree = cKDTree(target_xy)
    candidate_pairs: set[tuple[int, int]] = set()
    neighbour_count = min(5, len(targets))
    for left, point in enumerate(target_xy):
        _, neighbours = neighbour_tree.query(point, k=neighbour_count)
        for right in np.atleast_1d(neighbours):
            right = int(right)
            if left != right:
                candidate_pairs.add(tuple(sorted((left, right))))
    loop_candidates = []
    for left, right in candidate_pairs:
        a, b = targets[left], targets[right]
        try:
            road_path = nx.shortest_path(road, a, b, weight="length_km")
            road_length = nx.path_weight(road, road_path, weight="length_km")
            tree_length = nx.shortest_path_length(model, a, b, weight="length_km")
        except nx.NetworkXNoPath:
            continue
        if tree_length >= 1.30 * road_length and road_length <= 1.2:
            loop_candidates.append((road_length / tree_length, road_length, road_path))
    for _, length, path in sorted(loop_candidates):
        if added >= 0.15 * base_length:
            break
        base.add_path_edges(model, road, path)
        added += length
    node_ids = {node: f"W_N{i:04d}" for i, node in enumerate(model, 1)}
    zone_at = {(float(r.lon), float(r.lat)): r for r in zones.itertuples()}
    node_rows = []
    for node, node_id in node_ids.items():
        zone = zone_at.get(node)
        node_rows.append({"node_id": node_id, "node_type": "source" if node == source else "demand" if zone else "junction", "lon": node[0], "lat": node[1], "elevation_m": base.proxy_elevation_m(node), "demand_m3s": float(zone.water_q_avg_m3s) if zone else 0.0, "annual_m3": float(zone.water_m3_year) if zone else 0.0})
    q_design = defaultdict(float)
    for target, zone in zone_at.items():
        path = nx.shortest_path(model, source, target, weight="length_km")
        for u, v in zip(path[:-1], path[1:]):
            q_design[frozenset((u, v))] += float(zone.water_q_peak_m3s)
    pipe_rows = []
    for index, (u, v, data) in enumerate(model.edges(data=True), 1):
        q = max(q_design[frozenset((u, v))], 0.0002)
        diameter = math.sqrt(4.0 * q / math.pi)
        dn = _round_up(math.ceil(diameter * 1000.0), WATER_DN)
        pipe_rows.append({"link_id": f"W_P{index:04d}", "from_node": node_ids[u], "to_node": node_ids[v], "length_km": data["length_km"], "diameter_mm": dn, "design_flow_m3s": q, "geometry_json": json.dumps(data.get("geometry", [u, v]), separators=(",", ":"))})
    nodes, pipes = pd.DataFrame(node_rows), pd.DataFrame(pipe_rows)
    solver: dict[str, Any] = {"created": False, "converged": False}
    try:
        import wntr

        wn = wntr.network.WaterNetworkModel()
        source_head = max(nodes["elevation_m"].max() + 45.0, base.proxy_elevation_m(source) + 70.0)
        for row in nodes.itertuples():
            if row.node_type == "source":
                wn.add_reservoir(row.node_id, base_head=source_head, coordinates=(row.lon, row.lat))
            else:
                wn.add_junction(row.node_id, base_demand=row.demand_m3s, elevation=row.elevation_m, coordinates=(row.lon, row.lat))
        for row in pipes.itertuples():
            wn.add_pipe(row.link_id, row.from_node, row.to_node, length=row.length_km * 1000.0, diameter=row.diameter_mm / 1000.0, roughness=130.0)
        wn.options.time.duration = 3600
        wntr.network.io.write_inpfile(wn, case_dir / "drinking_water_epanet.inp", units="LPS")
        result = wntr.sim.EpanetSimulator(wn).run_sim()
        pressure = result.node["pressure"].iloc[0].drop(labels=[node_ids[source]], errors="ignore")
        velocity = result.link["velocity"].iloc[0]
        wn.options.hydraulic.demand_multiplier = 2.0
        peak_result = wntr.sim.EpanetSimulator(wn).run_sim()
        peak_pressure = peak_result.node["pressure"].iloc[0].drop(labels=[node_ids[source]], errors="ignore")
        peak_velocity = peak_result.link["velocity"].iloc[0]
        solver = {
            "created": True, "converged": bool(np.isfinite(pressure).all()),
            "minimum_pressure_m": float(pressure.min()), "maximum_pressure_m": float(pressure.max()),
            "maximum_velocity_m_s": float(velocity.abs().max()),
            "peak_minimum_pressure_m": float(peak_pressure.min()),
            "peak_maximum_velocity_m_s": float(peak_velocity.abs().max()),
        }
    except Exception as exc:
        solver["error"] = f"{type(exc).__name__}: {exc}"
    nodes.to_csv(case_dir / "drinking_water_nodes.csv", index=False)
    pipes.to_csv(case_dir / "drinking_water_links.csv", index=False)
    return nodes, pipes, solver


def _run_wastewater(case_dir: Path, road: nx.Graph, zones: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    elevations = {node: base.proxy_elevation_m(node) for node in road.nodes}
    elevation_min = min(elevations.values())
    elevation_span = max(max(elevations.values()) - elevation_min, 1.0)
    latitudes = [node[1] for node in road.nodes]
    latitude_min = min(latitudes)
    latitude_span = max(max(latitudes) - latitude_min, 1e-9)
    outfall = min(
        road.nodes,
        key=lambda p: (elevations[p] - elevation_min) / elevation_span
        + 0.25 * (p[1] - latitude_min) / latitude_span,
    )
    directed = nx.DiGraph()
    rise_penalty_alpha = float(
        load_case(CASE)["wastewater"]["terrain_rise_penalty_alpha_independent"]
    )
    for u, v, data in road.edges(data=True):
        for a, b in [(u, v), (v, u)]:
            rise = max(0.0, elevations[b] - elevations[a]) / elevation_span
            directed.add_edge(a, b, length_km=data["length_km"], cost=data["length_km"] * (1.0 + rise_penalty_alpha * rise))
    union = nx.DiGraph()
    targets = [(float(r.lon), float(r.lat)) for r in zones.itertuples()]
    union.add_nodes_from([outfall, *targets])
    for target in targets:
        try:
            path = nx.shortest_path(directed, target, outfall, weight="cost")
        except nx.NetworkXNoPath:
            continue
        for u, v in zip(path[:-1], path[1:]):
            union.add_edge(u, v, length_km=road[u][v]["length_km"], geometry=road[u][v].get("geometry", [u, v]))
    model = base.compress_directed(union, {outfall, *targets})
    model.add_nodes_from(node for node in [outfall, *targets] if node in union)
    # SWMM permits only one inlet link at an outfall.  If independently routed
    # branches meet only at the boundary, introduce a short terminal collector
    # manhole so the physical outfall remains a single-link boundary element.
    predecessors = list(model.predecessors(outfall))
    if len(predecessors) > 1:
        terminal = (outfall[0], outfall[1] + 0.000045)
        while terminal in model:
            terminal = (terminal[0], terminal[1] + 0.00001)
        for predecessor in predecessors:
            data = dict(model[predecessor][outfall])
            geometry = _geometry(data, predecessor, outfall)
            geometry[-1] = terminal
            model.remove_edge(predecessor, outfall)
            model.add_edge(
                predecessor, terminal,
                length_km=max(base.distance_km(predecessor, terminal), 0.005),
                geometry=geometry,
            )
        model.add_edge(
            terminal, outfall,
            length_km=max(base.distance_km(terminal, outfall), 0.005),
            geometry=[terminal, outfall],
        )
    zone_at = {(float(r.lon), float(r.lat)): r for r in zones.itertuples()}
    flow = defaultdict(float)
    for target, zone in zone_at.items():
        if target in model and nx.has_path(model, target, outfall):
            path = nx.shortest_path(model, target, outfall)
            for u, v in zip(path[:-1], path[1:]):
                flow[(u, v)] += float(zone.sewer_q_wet_m3s)
    inverts = {outfall: base.proxy_elevation_m(outfall) - 3.0}
    pump_edges: set[tuple[float, float]] = set()
    for downstream in reversed(list(nx.topological_sort(model))):
        if downstream not in inverts:
            continue
        for upstream in model.predecessors(downstream):
            length_m = float(model[upstream][downstream]["length_km"] * 1000.0)
            desired = inverts[downstream] + 0.0002 * length_m
            shallow = base.proxy_elevation_m(upstream) - 1.5
            deep = base.proxy_elevation_m(upstream) - 8.0
            if desired > shallow:
                inverts[upstream] = base.proxy_elevation_m(upstream) - 2.0
                pump_edges.add((upstream, downstream))
            else:
                inverts[upstream] = max(desired, deep)
    lifts = {u for u, _ in pump_edges}
    node_ids = {node: f"S_N{i:04d}" for i, node in enumerate(model, 1)}
    node_rows = []
    for node, node_id in node_ids.items():
        zone = zone_at.get(node)
        ground = base.proxy_elevation_m(node)
        node_rows.append({"node_id": node_id, "node_type": "outfall" if node == outfall else "lift_station" if node in lifts else "inflow" if zone else "manhole", "lon": node[0], "lat": node[1], "ground_elevation_m": ground, "invert_elevation_m": inverts[node], "cover_depth_m": ground - inverts[node], "dry_inflow_m3s": float(zone.sewer_q_dry_m3s) if zone else 0.0, "annual_m3": float(zone.wastewater_m3_year) if zone else 0.0})
    link_rows = []
    for index, (u, v, data) in enumerate(model.edges(data=True), 1):
        q = max(flow[(u, v)], 0.0005)
        slope = max(0.0025, (base.proxy_elevation_m(u) - base.proxy_elevation_m(v)) / max(data["length_km"] * 1000.0, 1.0))
        diameter = (q * 0.013 * 4.0 ** (5.0 / 3.0) / (math.pi * math.sqrt(slope))) ** (3.0 / 8.0)
        dn = _round_up(max(200, math.ceil(diameter * 1000.0)), SEWER_DN)
        link_type = "force_main" if (u, v) in pump_edges else "gravity"
        dry_flow = q / 3.6
        pump_flow = max(1.3 * dry_flow, 0.0002) if link_type == "force_main" else 0.0
        pump_head = max(2.0, inverts[v] - inverts[u] + 3.0) if link_type == "force_main" else 0.0
        link_rows.append({"link_id": f"S_C{index:04d}", "from_node": node_ids[u], "to_node": node_ids[v], "link_type": link_type, "length_km": data["length_km"], "diameter_mm": dn, "slope": slope, "manning_n": 0.013, "design_flow_m3s": q, "dry_design_flow_m3s": dry_flow, "pump_design_flow_m3s": pump_flow, "pump_head_m": pump_head, "geometry_json": json.dumps(data.get("geometry", [u, v]), separators=(",", ":"))})
    nodes, links = pd.DataFrame(node_rows), pd.DataFrame(link_rows)
    swmm = [
        "[TITLE]", ";; Independent morphology benchmark: normal dry-weather condition", "",
        "[OPTIONS]", "FLOW_UNITS CMS", "FLOW_ROUTING DYNWAVE", "FORCE_MAIN_EQUATION H-W",
        "START_DATE 07/01/2026", "START_TIME 00:00:00",
        "REPORT_START_DATE 07/01/2026", "REPORT_START_TIME 00:00:00",
        "END_DATE 07/03/2026", "END_TIME 00:00:00", "ROUTING_STEP 00:00:30",
        "VARIABLE_STEP 0.75", "MINIMUM_STEP 0.1", "MAX_TRIALS 100",
        "HEAD_TOLERANCE 0.005", "SURCHARGE_METHOD SLOT", "",
        "[JUNCTIONS]", ";;Name Elevation MaxDepth InitDepth SurDepth Aponded",
    ]
    pump_node_ids = set(nodes.loc[nodes["node_type"] == "lift_station", "node_id"])
    for row in nodes[(nodes["node_type"] != "outfall") & (~nodes["node_id"].isin(pump_node_ids))].itertuples():
        swmm.append(f"{row.node_id} {row.invert_elevation_m:.3f} {max(2.0, row.cover_depth_m):.3f} 0 0 0")
    swmm += ["", "[STORAGE]", ";;Name Elev MaxDepth InitDepth Shape A1 A2 A0 SurDepth Fevap Psi Ksat IMD"]
    for row in nodes[nodes["node_id"].isin(pump_node_ids)].itertuples():
        swmm.append(f"{row.node_id} {row.invert_elevation_m:.3f} {max(2.0, row.cover_depth_m):.3f} 0 FUNCTIONAL 0 0 25 0 0")
    swmm += ["", "[OUTFALLS]", ";;Name Elevation Type StageData Gated"]
    for row in nodes[nodes["node_type"] == "outfall"].itertuples():
        swmm.append(f"{row.node_id} {row.invert_elevation_m:.3f} FREE NO")
    swmm += ["", "[CONDUITS]", ";;Name From To Length Roughness InOffset OutOffset InitFlow MaxFlow"]
    for row in links[links["link_type"] == "gravity"].itertuples():
        swmm.append(f"{row.link_id} {row.from_node} {row.to_node} {row.length_km*1000:.2f} {row.manning_n:.3f} 0 0 0 0")
    swmm += ["", "[PUMPS]", ";;Name From To Curve Status Startup Shutoff"]
    force_mains = links[links["link_type"] == "force_main"]
    for index, row in enumerate(force_mains.itertuples(), 1):
        swmm.append(f"P_{index:03d} {row.from_node} {row.to_node} PC_{index:03d} ON 0.15 0.05")
    swmm += ["", "[XSECTIONS]", ";;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels"]
    for row in links[links["link_type"] == "gravity"].itertuples():
        swmm.append(f"{row.link_id} CIRCULAR {row.diameter_mm/1000:.3f} 0 0 0 1")
    swmm += ["", "[CURVES]", ";;Name Type X Y"]
    for index, row in enumerate(force_mains.itertuples(), 1):
        swmm += [
            f"PC_{index:03d} PUMP4 0 0",
            f"PC_{index:03d} 0.15 {row.pump_design_flow_m3s:.8f}",
            f"PC_{index:03d} 2.0 {row.pump_design_flow_m3s:.8f}",
        ]
    swmm += ["", "[DWF]", ";;Node Constituent Baseline Patterns"]
    for row in nodes[nodes["dry_inflow_m3s"] > 0].itertuples():
        swmm.append(f"{row.node_id} FLOW {row.dry_inflow_m3s:.8f}")
    swmm += ["", "[COORDINATES]", ";;Node X Y"]
    for row in nodes.itertuples():
        x, y = base.xy_km((row.lon, row.lat))
        swmm.append(f"{row.node_id} {x*1000:.2f} {y*1000:.2f}")
    input_path = case_dir / "wastewater_swmm.inp"
    report_path = case_dir / "wastewater_swmm.rpt"
    binary_path = case_dir / "wastewater_swmm.out"
    input_path.write_text("\n".join(swmm) + "\n", encoding="utf-8")
    required = ["[JUNCTIONS]", "[STORAGE]", "[OUTFALLS]", "[CONDUITS]", "[PUMPS]", "[XSECTIONS]", "[CURVES]", "[DWF]", "[COORDINATES]"]
    solver = {
        "created": True,
        "syntax_gate_passed": all(section in "\n".join(swmm) for section in required),
        "engine_run_passed": False,
    }
    try:
        from swmm.toolkit import solver as swmm_solver

        swmm_solver.swmm_run(str(input_path), str(report_path), str(binary_path))
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
        solver["engine_run_passed"] = "ERROR " not in report_text
        solver["report_file"] = report_path.name
        solver.update(_parse_swmm_report_metrics(report_text))
        sensitivity_input = case_dir / "wastewater_swmm_timestep15.inp"
        sensitivity_report = case_dir / "wastewater_swmm_timestep15.rpt"
        sensitivity_binary = case_dir / "wastewater_swmm_timestep15.out"
        sensitivity_input.write_text("\n".join(swmm).replace("ROUTING_STEP 00:00:30", "ROUTING_STEP 00:00:15") + "\n", encoding="utf-8")
        swmm_solver.swmm_run(str(sensitivity_input), str(sensitivity_report), str(sensitivity_binary))
        sensitivity_text = sensitivity_report.read_text(encoding="utf-8", errors="ignore")
        solver["time_step_sensitivity_15s"] = _parse_swmm_report_metrics(sensitivity_text)
        thresholds = load_case(CASE)["acceptance_screening"]["wastewater"]
        s = solver["time_step_sensitivity_15s"]
        solver["time_step_sensitivity_passed"] = (
            "ERROR " not in sensitivity_text
            and abs(s.get("flow_routing_continuity_error_percent", math.inf)) <= thresholds["maximum_absolute_flow_routing_continuity_error_percent"]
            and s.get("steps_not_converging_percent", math.inf) <= thresholds["maximum_nonconverging_steps_percent"]
            and s.get("flooding_loss_million_liter", math.inf) <= thresholds["maximum_dry_weather_flooding_loss_million_liter"]
        )
    except Exception as exc:
        solver["error"] = f"{type(exc).__name__}: {exc}"
    nodes.to_csv(case_dir / "wastewater_nodes.csv", index=False)
    links.to_csv(case_dir / "wastewater_links.csv", index=False)
    return nodes, links, solver


def _run_heat(
    case_dir: Path,
    road: nx.Graph,
    zones: pd.DataFrame,
    zone_selection_fraction: float,
    plant_capacity_mw: float = 30.0,
    seed: int = 20260802,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    count = max(4, int(math.ceil(len(zones) * zone_selection_fraction)))
    consumers = zones.nlargest(count, "heat_mwh_year").copy()
    plant_count = min(count, max(1, int(math.ceil(consumers["heat_peak_mw"].sum() / plant_capacity_mw))))
    consumer_xy = np.asarray([base.xy_km((r.lon, r.lat)) for r in consumers.itertuples()])
    if plant_count == 1:
        labels = np.zeros(len(consumers), dtype=int)
    else:
        _, labels = kmeans2(consumer_xy, plant_count, minit="++", iter=60, seed=seed)
    consumers["plant_index"] = labels
    road_index = base.NodeIndex(road.nodes)
    used_plants: set[tuple[float, float]] = set()
    components: list[tuple[tuple[float, float], nx.Graph, dict[tuple[float, float], Any]]] = []
    for plant_index in sorted(consumers["plant_index"].unique()):
        group = consumers[consumers["plant_index"] == plant_index]
        weights = np.maximum(group["heat_peak_mw"].to_numpy(dtype=float), 0.01)
        centre = (
            float(np.average(group["lon"], weights=weights)),
            float(np.average(group["lat"], weights=weights)),
        )
        plant = road_index.nearest(centre)
        if plant in used_plants:
            plant = min(
                (node for node in road if node not in used_plants),
                key=lambda node: base.distance_km(node, centre),
            )
        used_plants.add(plant)
        targets = [(float(r.lon), float(r.lat)) for r in group.itertuples()]
        model = _union_tree(road, plant, targets)
        consumer_at = {(float(r.lon), float(r.lat)): r for r in group.itertuples()}
        components.append((plant, model, consumer_at))

    all_nodes = sorted({node for _, model, _ in components for node in model})
    node_ids = {node: f"H_N{i:04d}" for i, node in enumerate(all_nodes, 1)}
    consumer_lookup = {
        (float(row.lon), float(row.lat)): row for row in consumers.itertuples()
    }
    node_rows = []
    for node in all_nodes:
        consumer = consumer_lookup.get(node)
        node_rows.append(
            {
                "node_id": node_ids[node],
                "node_type": "plant" if node in used_plants else "consumer_substation" if consumer else "junction",
                "lon": node[0], "lat": node[1], "elevation_m": base.proxy_elevation_m(node),
                "annual_heat_mwh": float(consumer.heat_mwh_year) if consumer else 0.0,
                "peak_heat_mw": float(consumer.heat_peak_mw) if consumer else 0.0,
                "represented_connections": int(consumer.building_count) if consumer else 0,
            }
        )
    link_rows = []
    used_edges: set[frozenset[tuple[float, float]]] = set()
    link_index = 1
    for plant, model, consumer_at in components:
        parent = {child: ancestor for ancestor, child in nx.bfs_edges(model, plant)}
        direct = {node: float(consumer_at[node].heat_peak_mw) if node in consumer_at else 0.0 for node in model}
        subtree = dict(direct)
        for node in reversed(list(nx.bfs_tree(model, plant).nodes)):
            if node != plant:
                subtree[parent[node]] += subtree[node]
        for child, ancestor in parent.items():
            edge_key = frozenset((ancestor, child))
            if edge_key in used_edges:
                continue
            used_edges.add(edge_key)
            data = model[ancestor][child]
            peak = max(subtree[child], 0.02)
            mass_flow = peak * 1e6 / (4180.0 * 35.0)
            diameter = math.sqrt(4.0 * mass_flow / (985.0 * math.pi * 1.0))
            dn = _round_up(math.ceil(diameter * 1000.0), HEAT_DN)
            link_rows.append({"link_id": f"H_P{link_index:04d}", "from_node": node_ids[ancestor], "to_node": node_ids[child], "length_km": data["length_km"], "diameter_mm": dn, "design_peak_mw": peak, "mass_flow_kg_s": mass_flow, "geometry_json": json.dumps(_geometry(data, ancestor, child), separators=(",", ":"))})
            link_index += 1
    nodes, links = pd.DataFrame(node_rows), pd.DataFrame(link_rows)
    solver: dict[str, Any] = {"created": False, "converged": False}
    try:
        import pandapipes as ppipe

        net = ppipe.create_empty_network(fluid="water")
        mapping = {row.node_id: ppipe.create_junction(net, pn_bar=25.0, tfluid_k=363.15, height_m=row.elevation_m, name=row.node_id, geodata=(row.lon, row.lat)) for row in nodes.itertuples()}
        for plant_id in nodes.loc[nodes["node_type"] == "plant", "node_id"]:
            ppipe.create_ext_grid(net, mapping[plant_id], p_bar=25.0, t_k=363.15, type="pt")
        for row in links.itertuples():
            ppipe.create_pipe_from_parameters(
                net, mapping[row.from_node], mapping[row.to_node],
                length_km=max(row.length_km, 0.005),
                inner_diameter_mm=row.diameter_mm, k_mm=0.1,
            )
        for row in nodes[nodes["peak_heat_mw"] > 0].itertuples():
            ppipe.create_sink(net, mapping[row.node_id], max(row.peak_heat_mw * 1e6 / (4180.0 * 35.0), 1e-4))
        ppipe.pipeflow(net, mode="hydraulics", max_iter_hyd=100)
        ppipe.to_json(net, str(case_dir / "district_heating_pandapipes.json"))
        solver = {"created": True, "converged": bool(net.converged), "minimum_pressure_bar": float(net.res_junction.p_bar.min()), "maximum_velocity_m_s": float(net.res_pipe.v_mean_m_per_s.abs().max())}
    except Exception as exc:
        solver["error"] = f"{type(exc).__name__}: {exc}"
    nodes.to_csv(case_dir / "district_heating_nodes.csv", index=False)
    links.to_csv(case_dir / "district_heating_links.csv", index=False)
    return nodes, links, solver


def _graph_metrics(nodes: pd.DataFrame, links: pd.DataFrame, sector: str) -> dict[str, Any]:
    node_col = "bus_id" if sector == "electricity" else "node_id"
    from_col = "from_node"
    to_col = "to_node"
    graph = nx.Graph()
    for row in links.itertuples():
        graph.add_edge(getattr(row, from_col), getattr(row, to_col), length_km=float(row.length_km))
    if sector == "electricity":
        metric_nodes = nodes
        # Transformer windings are explicit pandapower elements and therefore
        # do not appear in electricity_links.csv.  Add zero-route graph edges
        # here so source-to-building topology metrics traverse the same MV/LV
        # connection that the AC power flow uses.
        mv_candidates = nodes[nodes["voltage_kv"].ge(10.0)]
        for lv in nodes[nodes["node_type"].eq("transformer_lv_bus")].itertuples():
            same = mv_candidates[
                mv_candidates["transformer_id"].astype(str).eq(str(lv.transformer_id))
            ]
            if same.empty:
                same = mv_candidates.assign(
                    _distance=(mv_candidates["lon"] - float(lv.lon)) ** 2
                    + (mv_candidates["lat"] - float(lv.lat)) ** 2
                ).nsmallest(1, "_distance")
            if not same.empty:
                graph.add_edge(str(same.iloc[0][node_col]), str(getattr(lv, node_col)), length_km=0.0)
        sources = nodes.loc[nodes["node_type"].eq("mv_source"), node_col].tolist()
        targets = nodes.loc[nodes["node_type"].isin(["building_service", "mv_customer"]), node_col].tolist()
        facility_count = int(nodes["node_type"].eq("transformer_lv_bus").sum())
    elif sector == "drinking_water":
        metric_nodes = nodes
        sources = nodes.loc[nodes["node_type"] == "reservoir", node_col].tolist()
        targets = nodes.loc[nodes["node_type"] == "building_service", node_col].tolist()
        facility_count = int(nodes["node_type"].isin(["reservoir", "pump_outlet"]).sum())
    elif sector == "wastewater":
        metric_nodes = nodes
        sources = nodes.loc[nodes["node_type"] == "outfall", node_col].tolist()
        targets = nodes.loc[nodes["node_type"].eq("building_property_connection"), node_col].tolist()
        facility_count = int(nodes["node_type"].isin(["outfall", "lift_station"]).sum())
    else:
        metric_nodes = nodes
        sources = nodes.loc[nodes["node_type"] == "plant", node_col].tolist()
        targets = nodes.loc[nodes["node_type"].eq("consumer_substation"), node_col].tolist()
        facility_count = int(nodes["node_type"].eq("plant").sum())
    graph.add_nodes_from(metric_nodes[node_col])
    components = nx.number_connected_components(graph)
    degrees = np.asarray([degree for _, degree in graph.degree], dtype=float)
    valid_sources = [node for node in sources if node in graph]
    if valid_sources:
        distances, paths = nx.multi_source_dijkstra(
            graph, valid_sources, weight="length_km"
        )
    else:
        distances, paths = {}, {}
    service_routes = [float(distances[node]) for node in targets if node in distances]
    coordinates = nodes.set_index(node_col)[["lon", "lat"]].to_dict("index")
    route_stretches = []
    for target in targets:
        if target not in paths or not paths[target]:
            continue
        source = paths[target][0]
        if source not in coordinates or target not in coordinates:
            continue
        straight = base.distance_km(
            (float(coordinates[source]["lon"]), float(coordinates[source]["lat"])),
            (float(coordinates[target]["lon"]), float(coordinates[target]["lat"])),
        )
        network = float(distances[target])
        if straight > 1e-4 and network > 0.0:
            route_stretches.append(network / straight)
    edge_lengths = links["length_km"].to_numpy(dtype=float)
    return {
        "nodes": len(metric_nodes), "links": len(links), "route_length_km": float(links["length_km"].sum()),
        "cycle_rank": graph.number_of_edges() - graph.number_of_nodes() + components,
        "mean_degree": float(np.mean(degrees)) if len(degrees) else 0.0,
        "branch_fraction": float(np.mean(degrees >= 3)) if len(degrees) else 0.0,
        "leaf_fraction": float(np.mean(degrees == 1)) if len(degrees) else 0.0,
        "edge_length_p50_km": float(np.median(edge_lengths)) if len(edge_lengths) else 0.0,
        "edge_length_p90_km": float(np.percentile(edge_lengths, 90)) if len(edge_lengths) else 0.0,
        "average_service_route_km": float(np.mean(service_routes)) if service_routes else 0.0,
        "maximum_service_route_km": float(np.max(service_routes)) if service_routes else 0.0,
        "route_stretch_p50": float(np.median(route_stretches)) if route_stretches else 1.0,
        "route_stretch_p90": float(np.percentile(route_stretches, 90)) if route_stretches else 1.0,
        "source_count": len(sources), "facility_count": facility_count,
    }


def _interfaces(
    case_dir: Path,
    buildings: pd.DataFrame,
    power_nodes: pd.DataFrame,
    water_nodes: pd.DataFrame,
    sewer_nodes: pd.DataFrame,
    heat_nodes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    load_buses = power_nodes[power_nodes["node_type"].isin(["building_service", "mv_customer", "transformer_lv_bus"])]
    water_by_building = water_nodes[water_nodes.building_id.astype(str).ne("")].set_index("building_id")
    sewer_by_building = sewer_nodes[sewer_nodes.building_id.astype(str).ne("")].set_index("building_id")
    for building in buildings[buildings.water_connected.astype(bool) & buildings.wastewater_connected.astype(bool)].itertuples():
        if building.building_id not in water_by_building.index or building.building_id not in sewer_by_building.index:
            continue
        rows.append({
            "building_id": building.building_id,
            "from_sector": "drinking_water", "from_id": water_by_building.loc[building.building_id].node_id,
            "to_sector": "wastewater", "to_id": sewer_by_building.loc[building.building_id].node_id,
            "relation": "delivered_water_to_sanitary_inflow", "capacity_value": 0.82,
            "capacity_unit": "m3/m3",
        })
    pump_outlet = water_nodes[water_nodes.node_type.eq("pump_outlet")].iloc[0]
    facilities = [("drinking_water", pump_outlet.node_id, "water_pump", pump_outlet)]
    for row in sewer_nodes[sewer_nodes["node_type"] == "lift_station"].itertuples():
        facilities.append(("wastewater", row.node_id, "lift_station", row))
    for heat_plant in heat_nodes[heat_nodes["node_type"] == "plant"].itertuples():
        facilities.append(("district_heating", heat_plant.node_id, "heat_network_pump", heat_plant))
    for sector, facility, relation, point_row in facilities:
        nearest = min(load_buses.itertuples(), key=lambda bus: base.distance_km((point_row.lon, point_row.lat), (bus.lon, bus.lat)))
        rows.append({"building_id": "", "from_sector": "electricity", "from_id": nearest.bus_id, "to_sector": sector, "to_id": facility, "relation": relation, "capacity_value": np.nan, "capacity_unit": "MW"})
    frame = pd.DataFrame(rows)
    frame.insert(0, "interface_id", [f"I{i:04d}" for i in range(1, len(frame) + 1)])
    frame.to_csv(case_dir / "coupling_interfaces.csv", index=False)
    return frame


def run_benchmark_cases() -> dict[str, Any]:
    config = load_case(CASE)
    buildings_all = pd.read_csv(OUT / "building_sector_demands.csv")
    BENCHMARK_OUT.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    manifests = {}
    for case_number, case in enumerate(config["benchmark_cases"], 1):
        started = time.perf_counter()
        case_dir = BENCHMARK_OUT / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        bbox = case["bbox"]
        buildings = buildings_all[buildings_all["lon"].between(bbox[0], bbox[2]) & buildings_all["lat"].between(bbox[1], bbox[3])].copy()
        road = _tile_road_graph(bbox)
        road_index = base.NodeIndex(road.nodes)
        buildings["road_distance"] = [base.distance_km((r.lon, r.lat), road_index.nearest((r.lon, r.lat))) for r in buildings.itertuples()]
        buildings = buildings[buildings["road_distance"] <= 0.75].drop(columns="road_distance")
        controlled_buildings_per_zone = 80
        controlled_zone_target = max(12, int(round(len(buildings) / controlled_buildings_per_zone)))
        controlled_seed = config["seed"] + case_number
        zones = _cluster_zones(buildings, road, controlled_zone_target, controlled_seed)
        zones.to_csv(case_dir / "common_demand_zones.csv", index=False)
        buildings.to_csv(case_dir / "common_building_ledger.csv", index=False)
        power_nodes, power_links, transformers, _, power_solver = generate_electricity(
            case_dir, road, buildings, maximum_radius_km=0.70, seed=controlled_seed
        )
        water_nodes, water_links, _, water_solver = generate_water(case_dir, road, buildings)
        sewer_nodes, sewer_links, _, sewer_solver = generate_wastewater(case_dir, road, buildings)
        heat_connection_count = int(round(0.25 * buildings.heat_eligible.astype(bool).sum()))
        heat_nodes, heat_links, _, heat_solver = generate_district_heat(
            case_dir, road, buildings, connection_count=heat_connection_count,
            plant_capacity_mw=30.0, seed=controlled_seed,
            minimum_heat_density_mwh_ha=float(
                config["district_heating"]["minimum_heat_density_mwh_ha"]
            ),
            heat_density_cell_size_km=float(
                config["district_heating"]["heat_density_cell_size_km"]
            ),
            selection_policy=str(
                config["district_heating"].get("customer_selection_policy", "route_aware")
            ),
            route_aware_demand_exponent=float(
                config["district_heating"].get("route_aware_demand_exponent", 1.0)
            ),
            route_aware_spatial_quota_level=str(
                config["district_heating"].get("route_aware_spatial_quota_level", "cell")
            ),
            return_pressure_reference_bar=float(
                config["bidirectional_coupling"]["heat_return_pressure_bar"]
            ),
        )
        interfaces = _interfaces(case_dir, buildings, power_nodes, water_nodes, sewer_nodes, heat_nodes)
        runtime = time.perf_counter() - started
        sector_data = {
            "electricity": (power_nodes, power_links, "cable_size_mm2", float(zones["electricity_mwh_year"].sum()), "MWh/y"),
            "drinking_water": (water_nodes, water_links, "diameter_mm", float(zones["water_m3_year"].sum()), "m3/y"),
            "wastewater": (sewer_nodes, sewer_links, "diameter_mm", float(zones["wastewater_m3_year"].sum()), "m3/y"),
            "district_heating": (heat_nodes, heat_links, "diameter_mm", float(heat_nodes["annual_heat_mwh"].sum()), "MWh/y"),
        }
        for sector, (nodes, links, size_col, demand, demand_unit) in sector_data.items():
            graph_metrics = _graph_metrics(nodes, links, sector)
            graph_metrics["network_length_per_1000_buildings_km"] = (
                graph_metrics["route_length_km"] / len(buildings) * 1000.0
            )
            graph_metrics["buildings_per_source_mean"] = (
                len(buildings) / graph_metrics["source_count"]
                if graph_metrics["source_count"] else 0.0
            )
            heat_contracts = int(heat_nodes["node_type"].eq("consumer_substation").sum())
            metric_rows.append({"experiment_mode": "controlled_morphology", "case_id": case["id"], "label": case["label"], "place_name": case["place_name"], "morphology_label": case["morphology_label"], "morphology": case["morphology"], "sector": sector, **graph_metrics, "buildings": len(buildings), "demand_zones": len(zones), "building_density_per_km2": len(buildings) / (base.distance_km((bbox[0], bbox[1]), (bbox[2], bbox[1])) * base.distance_km((bbox[0], bbox[1]), (bbox[0], bbox[3]))), "allocated_demand": demand, "demand_unit": demand_unit, "equipment_size_median": float(links[size_col].median()), "equipment_size_p90": float(links[size_col].quantile(0.9)), "equipment_size_max": float(links[size_col].max()), "district_heat_contract_equivalents": heat_contracts if sector == "district_heating" else 0, "district_heat_contract_equivalents_per_km2": heat_contracts / (base.distance_km((bbox[0], bbox[1]), (bbox[2], bbox[1])) * base.distance_km((bbox[0], bbox[1]), (bbox[0], bbox[3]))) if sector == "district_heating" else 0.0, "generation_runtime_s": runtime})
        solver_readiness = {
            "pandapower": power_solver,
            "epanet_wntr": water_solver,
            "swmm": sewer_solver,
            "pandapipes_heat": heat_solver,
        }
        screening = {
            "electricity": _screen_solver_results(power_solver, "electricity", config),
            "drinking_water": _screen_solver_results(water_solver, "drinking_water", config),
            "wastewater": _screen_solver_results(sewer_solver, "wastewater", config),
            "district_heating": _screen_solver_results(heat_solver, "district_heating", config),
        }
        manifest = {"case_id": case["id"], "label": case["label"], "place_name": case["place_name"], "morphology_label": case["morphology_label"], "study_role": case["study_role"], "bbox": bbox, "independent_generation": True, "experiment_mode": "controlled_morphology", "controlled_parameters": {"buildings_per_requested_zone": controlled_buildings_per_zone, "maximum_lv_service_radius_km": 0.70, "district_heat_connection_fraction": 0.25, "district_heat_connection_fraction_ceiling": 0.25, "district_heat_selection": "contiguous cells passing the declared annual heat-density screen; no network is an admissible outcome", "district_heat_minimum_density_mwh_ha": float(config["district_heating"]["minimum_heat_density_mwh_ha"]), "district_heat_plant_capacity_mw": 30.0, "seed": controlled_seed}, "generation_sequence": ["clip_common_evidence", "cluster_demands", "place_facilities", "generate_sector_topologies", "size_assets", "create_coupling_interfaces", "export_and_solve", "evaluate_screening_envelope"], "counts": {"buildings": len(buildings), "demand_zones": len(zones), "electricity_transformers": len(transformers), "district_heating_plants": int((heat_nodes["node_type"] == "plant").sum()), "coupling_interfaces": len(interfaces)}, "solver_readiness": solver_readiness, "acceptance_screening": screening, "runtime_seconds": runtime}
        (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests[case["id"]] = manifest
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "benchmark_case_metrics.csv", index=False)
    (BENCHMARK_OUT / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")

    # Experiment B: activate the morphology-specific planning parameters while
    # retaining the same evidence clips.  The resulting delta is kept separate
    # from Experiment A so morphology and policy are not conflated.
    policy_root = OUT / "benchmark_policy_cases"
    policy_root.mkdir(parents=True, exist_ok=True)
    policy_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    for case_number, case in enumerate(config["benchmark_cases"], 1):
        bbox = case["bbox"]
        buildings = buildings_all[
            buildings_all["lon"].between(bbox[0], bbox[2])
            & buildings_all["lat"].between(bbox[1], bbox[3])
        ].copy()
        road = _tile_road_graph(bbox)
        road_index = base.NodeIndex(road.nodes)
        buildings["road_distance"] = [
            base.distance_km((r.lon, r.lat), road_index.nearest((r.lon, r.lat)))
            for r in buildings.itertuples()
        ]
        buildings = buildings[buildings["road_distance"] <= 0.75].drop(columns="road_distance")
        policy_seed = config["seed"] + 100 + case_number
        policy_zones = _cluster_zones(
            buildings, road, int(case["demand_zone_target"]), policy_seed
        )
        case_dir = policy_root / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        power_nodes, power_links, _, _ = _run_power(
            case_dir, road, policy_zones, buildings,
            maximum_feeder_radius_km=None, seed=policy_seed,
        )
        water_nodes, water_links, _ = _run_water(case_dir, road, policy_zones)
        sewer_nodes, sewer_links, _ = _run_wastewater(case_dir, road, policy_zones)
        heat_limit = int(round(
            float(case["district_heat_zone_selection_fraction"])
            * buildings.heat_eligible.astype(bool).sum()
        ))
        heat_nodes, heat_links, _, _ = generate_district_heat(
            case_dir, road, buildings, connection_count=heat_limit,
            plant_capacity_mw=30.0, seed=policy_seed,
            minimum_heat_density_mwh_ha=float(
                config["district_heating"]["minimum_heat_density_mwh_ha"]
            ),
            heat_density_cell_size_km=float(
                config["district_heating"]["heat_density_cell_size_km"]
            ),
            selection_policy=str(
                config["district_heating"].get("customer_selection_policy", "route_aware")
            ),
            route_aware_demand_exponent=float(
                config["district_heating"].get("route_aware_demand_exponent", 1.0)
            ),
            route_aware_spatial_quota_level=str(
                config["district_heating"].get("route_aware_spatial_quota_level", "cell")
            ),
            return_pressure_reference_bar=float(
                config["bidirectional_coupling"]["heat_return_pressure_bar"]
            ),
        )
        for sector, nodes, links in [
            ("electricity", power_nodes, power_links),
            ("drinking_water", water_nodes, water_links),
            ("wastewater", sewer_nodes, sewer_links),
            ("district_heating", heat_nodes, heat_links),
        ]:
            policy_rows.append(
                {
                    "experiment_mode": "morphology_aware_policy",
                    "case_id": case["id"], "place_name": case["place_name"],
                    "morphology_label": case["morphology_label"], "sector": sector,
                    **_graph_metrics(nodes, links, sector),
                    "demand_zones": len(policy_zones),
                    "maximum_lv_radius_policy_km": 0.45 if case["morphology"] == "dense" else 0.70 if case["morphology"] == "suburban" else 1.20,
                    "district_heat_zone_selection_fraction_policy": float(case["district_heat_zone_selection_fraction"]),
                    "district_heat_contract_equivalents": int(heat_nodes["node_type"].eq("consumer_substation").sum()) if sector == "district_heating" else 0,
                }
            )

        # Twenty clustering realizations isolate stochastic robustness at the
        # only random common-evidence stage.  Downstream generators are
        # deterministic conditional on a frozen zone ledger.
        controlled_target = max(12, int(round(len(buildings) / 80)))
        for seed_offset in range(20):
            seed = config["seed"] + 1000 * case_number + seed_offset
            z = _cluster_zones(buildings, road, controlled_target, seed)
            points = np.asarray([base.xy_km((r.lon, r.lat)) for r in z.itertuples()])
            nearest = cKDTree(points).query(points, k=2)[0][:, 1] if len(points) > 1 else np.asarray([0.0])
            seed_rows.append(
                {
                    "case_id": case["id"], "place_name": case["place_name"], "seed": seed,
                    "stochastic_stage": "common_zone_clustering_only",
                    "realized_zones": len(z),
                    "buildings_per_zone_median": float(z["building_count"].median()),
                    "buildings_per_zone_iqr": float(z["building_count"].quantile(0.75) - z["building_count"].quantile(0.25)),
                    "electricity_peak_zone_cv": float(z["electricity_peak_mw"].std() / max(z["electricity_peak_mw"].mean(), 1e-9)),
                    "water_zone_cv": float(z["water_m3_year"].std() / max(z["water_m3_year"].mean(), 1e-9)),
                    "nearest_zone_spacing_median_km": float(np.median(nearest)),
                    "nearest_zone_spacing_p95_km": float(np.percentile(nearest, 95)),
                }
            )

        water_ablation = _cluster_zones(buildings, road, controlled_target, policy_seed + 3000)
        sewer_ablation = _cluster_zones(buildings, road, controlled_target, policy_seed + 6000)
        water_xy = np.asarray([base.xy_km((r.lon, r.lat)) for r in water_ablation.itertuples()])
        sewer_xy = np.asarray([base.xy_km((r.lon, r.lat)) for r in sewer_ablation.itertuples()])
        separation = cKDTree(sewer_xy).query(water_xy)[0]
        ablation_rows.append(
            {
                "case_id": case["id"], "place_name": case["place_name"],
                "common_ledger_exact_zone_interfaces_percent": 100.0,
                "independent_sector_clustering_exact_coincidence_percent": float(np.mean(separation < 1e-9) * 100.0),
                "independent_sector_clustering_mean_interface_offset_km": float(np.mean(separation)),
                "independent_sector_clustering_p95_interface_offset_km": float(np.percentile(separation, 95)),
            }
        )

    policy_metrics = pd.DataFrame(policy_rows)
    policy_metrics.to_csv(OUT / "benchmark_policy_case_metrics.csv", index=False)
    comparison_columns = [
        "nodes", "links", "route_length_km", "cycle_rank", "mean_degree", "branch_fraction", "leaf_fraction",
        "edge_length_p50_km", "edge_length_p90_km", "route_stretch_p50", "route_stretch_p90",
        "average_service_route_km", "maximum_service_route_km", "source_count", "facility_count",
    ]
    controlled = metrics[["case_id", "sector", *comparison_columns]].copy()
    policy = policy_metrics[["case_id", "sector", *comparison_columns]].copy()
    effects = policy.merge(controlled, on=["case_id", "sector"], suffixes=("_policy", "_controlled"))
    for column in comparison_columns:
        denominator = effects[f"{column}_controlled"].replace(0, np.nan)
        effects[f"{column}_policy_delta_percent"] = (
            (effects[f"{column}_policy"] - effects[f"{column}_controlled"]) / denominator * 100.0
        )
    effects.to_csv(OUT / "benchmark_policy_effects.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "clustering_seed_sensitivity.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(OUT / "common_ledger_ablation.csv", index=False)
    return manifests


if __name__ == "__main__":
    print(json.dumps(run_benchmark_cases(), indent=2))
