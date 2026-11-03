#!/usr/bin/env python3
"""Controlled Andermatt DH routing comparison against mapped installed routes.

The cadastral transfer points and local incoming feed are shared inputs.
Installed route geometry is held out from both routing methods. This is a
conditional routing benchmark, not an independent prediction of customers,
heat demand, plant dispatch, pipe sizes, or the upstream transmission line.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import set_precision, union_all
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import linemerge

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urban4 import schweinfurt_base as base
from urban4.route_aware_heat import (
    RouteAwareHeatConfig, _path_to_root, _single_source_predecessor,
    select_route_aware_heat_connections,
)
from urban4.external_validation import symmetric_corridor_coverage


def json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    return [part for child in getattr(geometry, "geoms", ()) for part in line_parts(child)]


def prepare_roads(data: Path, case: dict):
    """Use Urban4's OSM importer, with a common input-defined corridor window."""
    raw = json.loads((data / "osm_snapshot.json").read_text())
    excluded = {"motorway", "motorway_link", "trunk", "trunk_link", "steps",
                "construction", "proposed", "raceway"}
    ways = [w for w in raw["elements"] if w.get("tags", {}).get("highway")
            and w["tags"]["highway"] not in excluded]
    imported = base.build_road_graph(ways)
    project = Transformer.from_crs(4326, 2056, always_xy=True)
    domain = box(*case["domain_bounds_2056"])
    corridor_window = domain.buffer(200.0)
    road = nx.Graph()
    for a, b, attrs in imported.edges(data=True):
        geometry = LineString([project.transform(*a), project.transform(*b)])
        for part in line_parts(geometry.intersection(corridor_window)):
            if part.length < 1e-4:
                continue
            u, v = (tuple(np.round(xy, 3)) for xy in (part.coords[0], part.coords[-1]))
            if u != v:
                road.add_edge(u, v, length_km=part.length / 1000.0,
                              geometry=LineString([u, v]), road_class=attrs.get("road_class"))
    road = road.subgraph(max(nx.connected_components(road), key=len)).copy()
    buildings = []
    for way in raw["elements"]:
        coords = way.get("geometry", [])
        if not way.get("tags", {}).get("building") or len(coords) < 4:
            continue
        try:
            polygon = Polygon([project.transform(p["lon"], p["lat"]) for p in coords])
            if polygon.is_valid and polygon.area >= 20 and domain.covers(polygon.centroid):
                buildings.append({"osm_id": way["id"], "geometry": polygon})
        except (ValueError, TypeError):
            continue
    return road, gpd.GeoDataFrame(buildings, crs=2056), domain


def attach_and_run_urban4(road: nx.Graph, customers, source: Point, case: dict):
    nodes = list(road)
    tree = cKDTree(np.asarray(nodes))
    source_root = nodes[int(tree.query([source.x, source.y])[1])]
    inverse = Transformer.from_crs(2056, 4326, always_xy=True)
    rows = []
    for _, row in customers.iterrows():
        p = row.geometry
        distance, index = tree.query([p.x, p.y])
        attached = nodes[int(index)]
        lon, lat = inverse.transform(*attached)
        rows.append({"building_id": row.customer_id, "road_node": attached,
                     "road_lon": lon, "road_lat": lat,
                     "service_length_km": float(distance / 1000),
                     "heat_candidate_mwh_year": float(row.peak_kw * row.flh / 1000),
                     "heat_territory_id": 0, "heat_cell_x": 0, "heat_cell_y": 0})
    candidates = pd.DataFrame(rows)
    protected = {source_root, *candidates.road_node}
    # Removing unused leaf spurs is exact for both positive-cost formulations.
    graph = road.copy()
    queue = deque(n for n, degree in graph.degree if degree <= 1 and n not in protected)
    while queue:
        n = queue.popleft()
        if n not in graph or n in protected or graph.degree(n) > 1:
            continue
        neighbours = list(graph.neighbors(n))
        graph.remove_node(n)
        queue.extend(v for v in neighbours if graph.degree(v) <= 1 and v not in protected)

    started = time.perf_counter()
    selected, metadata, audit = select_route_aware_heat_connections(
        graph, candidates, target_count=len(customers),
        config=RouteAwareHeatConfig(demand_exponent=0.5), root_by_territory={0: source_root})
    assert len(selected) == len(customers)
    _, predecessor = _single_source_predecessor(graph, source_root, selected.road_node)
    used = set()
    for target in selected.road_node:
        path = _path_to_root(target, source_root, predecessor)
        used.update(tuple(sorted((u, v))) for u, v in zip(path[:-1], path[1:]))
    urban_lines = [graph[u][v]["geometry"] for u, v in sorted(used)]
    assert abs(sum(graph[u][v]["length_km"] for u, v in used)
               - metadata["provisional_forest_trench_km"]) < 1e-9
    elapsed = time.perf_counter() - started

    candidate = nx.MultiGraph()
    for u, v, attrs in graph.edges(data=True):
        candidate.add_edge(u, v, **attrs, original_edges=[tuple(sorted((u, v)))])
    # A source and every consumer have their own port, also when locations coincide.
    candidate.add_node("source", xy=(source.x, source.y), port="source")
    source_line = LineString([source, source_root])
    candidate.add_edge("source", source_root, geometry=source_line,
                       length_km=max(source_line.length, .001) / 1000,
                       original_edges=[], port="source")
    services = []
    for i, (_, customer) in enumerate(customers.iterrows()):
        attached = rows[i]["road_node"]
        name = f"consumer_{i:03d}"
        candidate.add_node(name, xy=(customer.geometry.x, customer.geometry.y), port="consumer", consumer=i)
        line = LineString([customer.geometry, attached])
        candidate.add_edge(name, attached, geometry=line, length_km=max(line.length, .001) / 1000,
                           original_edges=[], port="consumer")
        services.append(line)
    urban_lines.extend([source_line, *services])
    return graph, candidate, urban_lines, metadata | {"runtime_s": elapsed}, audit, candidates


def contract_degree_two(graph: nx.MultiGraph):
    """Preserve alternate corridors and their geometry while suppressing degree-2 nodes."""
    reduced = graph.copy()
    queue = deque(n for n, degree in reduced.degree if degree == 2 and not isinstance(n, str))
    while queue:
        n = queue.popleft()
        if n not in reduced or reduced.degree(n) != 2 or isinstance(n, str):
            continue
        incident = list(reduced.edges(n, keys=True, data=True))
        if len(incident) != 2:
            continue
        _, a, _, first = incident[0]
        _, b, _, second = incident[1]
        reduced.remove_node(n)
        if a == b:
            continue  # A positive-cost loop with no terminal cannot improve a route.
        geometry = linemerge([first["geometry"], second["geometry"]])
        reduced.add_edge(a, b, geometry=geometry,
                         length_km=first["length_km"] + second["length_km"],
                         original_edges=first["original_edges"] + second["original_edges"])
        queue.extend(x for x in (a, b) if not isinstance(x, str) and reduced.degree(x) == 2)
    return reduced


def run_topotherm(candidate: nx.MultiGraph, customers, out: Path, time_limit: int):
    import pyomo.environ as pyo
    import topotherm as tt

    nodes = list(candidate)
    indices = {n: i for i, n in enumerate(nodes)}
    edges = list(candidate.edges(keys=True, data=True))
    ni, ne, nc = len(nodes), len(edges), len(customers)
    a_i = np.zeros((ni, ne))
    for k, (u, v, _, _) in enumerate(edges):
        a_i[indices[u], k], a_i[indices[v], k] = 1., -1.
    a_p = np.zeros((ni, 1)); a_p[indices["source"], 0] = -1.
    a_c = np.zeros((ni, nc))
    for i in range(nc):
        assert candidate.degree(f"consumer_{i:03d}") == 1
        a_c[indices[f"consumer_{i:03d}"], i] = 1.
    matrices = {"a_i": a_i, "a_p": a_p, "a_c": a_c,
                "l_i": np.array([a["length_km"] * 1000 for *_, a in edges]),
                "q_c": customers.peak_kw.to_numpy(float).reshape(-1, 1),
                "flh_sinks": customers.flh.to_numpy(float).reshape(-1, 1),
                "flh_sources": np.array([[1600.]])}
    settings = tt.settings.Settings()
    settings.economics.source_max_power = [float(2 * customers.peak_kw.sum())]
    settings.economics.source_c_inv = [0.]
    settings.economics.heat_price = 0.  # Constant forced-connection revenue omitted.
    started = time.perf_counter()
    model = tt.models.single_timestep.create(matrices, tt.models.sets.create(matrices),
                                             economics=settings.economics, optimization_mode="forced")
    prepare_s = time.perf_counter() - started
    solver = pyo.SolverFactory("highs")
    solver.options.update({"time_limit": time_limit, "mip_rel_gap": .01,
                           "threads": 2, "random_seed": 20261103,
                           "log_file": str(out / "topotherm_solver.log")})
    started = time.perf_counter()
    result = solver.solve(model, tee=True)
    runtime = time.perf_counter() - started
    selected = [k for k in range(ne)
                if pyo.value(model.lambda_["ij", k]) + pyo.value(model.lambda_["ji", k]) > .5]
    chosen = nx.MultiGraph()
    for k in selected:
        u, v, key, attrs = edges[k]
        chosen.add_edge(u, v, key=key, **attrs)
    assert all(nx.has_path(chosen, "source", f"consumer_{i:03d}") for i in range(nc))
    source_supply = pyo.value(model.P_source[0, 0])
    assert source_supply >= customers.peak_kw.sum() - 1e-5
    rows = []
    for k, (u, v, _, attrs) in enumerate(edges):
        rows.append({"edge": k, "length_m": attrs["length_km"] * 1000,
                     "selected": k in selected,
                     "lambda_ij": pyo.value(model.lambda_["ij", k]),
                     "lambda_ji": pyo.value(model.lambda_["ji", k]),
                     "flow_ij_in_kw": pyo.value(model.P["ij", "in", k, 0]),
                     "flow_ji_in_kw": pyo.value(model.P["ji", "in", k, 0])})
    pd.DataFrame(rows).to_csv(out / "topotherm_edge_decisions.csv", index=False)
    info = {"termination": str(result.solver.termination_condition),
            "solver_status": str(result.solver.status), "runtime_s": runtime,
            "prepare_s": prepare_s, "objective_eur_per_year": pyo.value(model.obj),
            "lower_bound": result.problem.lower_bound, "upper_bound": result.problem.upper_bound,
            "source_supply_kw": source_supply, "demand_kw": float(customers.peak_kw.sum()),
            "customers_connected": nc, "candidate_nodes": ni, "candidate_edges": ne,
            "selected_edges": len(selected), "time_limit_s": time_limit, "mip_relative_gap_target": .01,
            "economics": settings.economics.model_dump()}
    json_write(out / "topotherm_run.json", info)
    return [edges[k][3]["geometry"] for k in selected], info


def export_lines(path: Path, layer: str, lines):
    gpd.GeoDataFrame({"edge": range(len(lines))}, geometry=lines, crs=2056).to_file(path, layer=layer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/andermatt_benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/andermatt_topotherm")
    parser.add_argument("--time-limit", type=int, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    case = json.loads((args.data / "case.json").read_text())
    customers = gpd.read_file(args.data / "common_inputs.gpkg", layer="customers")
    source = gpd.read_file(args.data / "common_inputs.gpkg", layer="source").geometry.iloc[0]
    road, buildings, domain = prepare_roads(args.data, case)
    full, candidate, urban_lines, urban_info, audit, attachments = attach_and_run_urban4(road, customers, source, case)
    reduced = contract_degree_two(candidate)
    print(json.dumps({"road_nodes":len(full),"road_edges":full.number_of_edges(),
                      "reduced_nodes":len(reduced),"reduced_edges":reduced.number_of_edges(),
                      "Urban4":urban_info},indent=2),flush=True)
    path = args.output / "routes.gpkg"
    export_lines(path, "urban4", urban_lines)
    export_lines(path, "candidates", [d["geometry"] for *_,d in reduced.edges(data=True)])
    buildings.to_file(path, layer="evaluation_buildings")
    audit.to_csv(args.output / "urban4_selection_audit.csv", index=False)
    attachments.drop(columns="road_node").to_csv(args.output / "shared_attachments.csv", index=False)
    json_write(args.output / "urban4_run.json", urban_info)
    top_lines, top_info = run_topotherm(reduced, customers, args.output, args.time_limit)
    export_lines(path, "topotherm", top_lines)
    json_write(args.output / "run_manifest.json", {
        "case":case,"urban4":urban_info,"topotherm":top_info,
        "candidate_graph_reduction":"non-terminal leaf pruning and degree-2 contraction; alternate parallel corridors retained",
        "shared_inputs":["85 observed transfer points","local incoming feed","uniform 50-kW scenario demands",
                         "same OSM road corridors and service attachments"],
        "reference_route_leakage":"Only the declared inlet location is read from the reference route; no installed edge guides routing",
        "reference_sha256":hashlib.sha256((args.data/'fwa_trasse_linie.geojson').read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
