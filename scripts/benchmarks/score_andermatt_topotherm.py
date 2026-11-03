#!/usr/bin/env python3
"""Score both Andermatt routing tools with one explicit geometry protocol."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import set_precision, union_all
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmarks.run_andermatt_topotherm import line_parts, json_write
from urban4.external_validation import symmetric_corridor_coverage


DEFINITIONS = {
    "length_per_area": "unique one-way mapped route length / common window area (km/km2)",
    "length_per_connection": "unique route length / 85 observed transfer points (m/point)",
    "connection_spacing": "mean nearest-neighbour distance of observed transfer points (m); input control",
    "building_to_network_distance": "mean minimum distance from every eligible OSM building centroid to route union (m)",
    "branch_length": "mean branch length between degree-not-equal-to-2 nodes after common 0.5-m geometry quantisation and noding (m)",
    "terminal_share": "degree-1 nodes / nodes in the degree-2-suppressed inferred route graph",
}


def route_union(lines, domain=None):
    geometry = union_all([g for g in lines if g is not None and not g.is_empty])
    if domain is not None:
        geometry = geometry.intersection(domain)
    return union_all(line_parts(geometry))  # Boundary-only point touches are not routes.


def graph_descriptors(geometry, precision_m: float):
    noded = union_all([set_precision(line, precision_m) for line in line_parts(geometry)])
    graph = nx.MultiGraph()
    for line in line_parts(noded):
        if line.length <= 0:
            continue
        graph.add_edge(tuple(line.coords[0]), tuple(line.coords[-1]), length_m=line.length)
    if not graph:
        raise ValueError("No route geometry for topology scoring")
    critical = {n for n, d in graph.degree if d != 2}
    for component in nx.connected_components(graph):
        if not critical.intersection(component):
            critical.add(min(component))  # Closed all-degree-2 rings have one loop branch.
    visited = set()
    branches = []
    for start in sorted(critical):
        for u, v, key, attrs in graph.edges(start, keys=True, data=True):
            identifier = (frozenset((u, v)), key)
            if identifier in visited:
                continue
            visited.add(identifier)
            length, current = attrs["length_m"], v
            while current not in critical:
                choices = [(a, b, k, d) for a, b, k, d in graph.edges(current, keys=True, data=True)
                           if (frozenset((a, b)), k) not in visited]
                if not choices:
                    break
                a, b, k, d = choices[0]
                visited.add((frozenset((a, b)), k))
                length += d["length_m"]
                current = b
            branches.append(length)
    leaves = sum(degree == 1 for _, degree in graph.degree)
    return {"branch_length": float(np.mean(branches)),
            "branch_length_median": float(np.median(branches)),
            "terminal_share": leaves / len(critical),
            "nodes_noded": len(graph), "nodes_degree2_suppressed": len(critical),
            "leaves": leaves, "branches": len(branches),
            "components": nx.number_connected_components(graph),
            "cycles": graph.number_of_edges() - len(graph) + nx.number_connected_components(graph)}


def geometry_descriptors(geometry, points, building_points, area_m2: float, precision_m=.5):
    xy = np.array([[p.x, p.y] for p in points])
    spacing = cKDTree(xy).query(xy, k=2)[0][:, 1]
    distances = np.array([p.distance(geometry) for p in building_points])
    return {"route_km": geometry.length / 1000,
            "length_per_area": geometry.length / 1000 / (area_m2 / 1e6),
            "length_per_connection": geometry.length / len(points),
            "connection_spacing": float(np.mean(spacing)),
            "connection_spacing_median": float(np.median(spacing)),
            "building_to_network_distance": float(np.mean(distances)),
            "building_to_network_distance_median": float(np.median(distances)),
            "evaluation_buildings": len(building_points),
            **graph_descriptors(geometry, precision_m)}


def weighted_route_distance(reference, target, spacing_m=5.0):
    """Length-weighted directional route distance with equally spaced midpoints."""
    distances, weights = [], []
    for line in line_parts(reference):
        n = max(1, math.ceil(line.length / spacing_m))
        step = line.length / n
        for i in range(n):
            distances.append(line.interpolate((i + .5) * step).distance(target))
            weights.append(step)
    values, weight = np.asarray(distances), np.asarray(weights)
    order = np.argsort(values)
    cumulative = np.cumsum(weight[order]) / np.sum(weight)
    p95 = values[order][min(np.searchsorted(cumulative, .95), len(order) - 1)]
    return {"mean_m": float(np.average(values, weights=weight)), "p95_m": float(p95)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/andermatt_benchmark")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/andermatt_topotherm")
    args = parser.parse_args()
    case = json.loads((args.data / "case.json").read_text())
    domain = box(*case["domain_bounds_2056"])
    customers = gpd.read_file(args.data / "common_inputs.gpkg", layer="customers")
    reference = route_union(gpd.read_file(args.data / "common_inputs.gpkg", layer="installed_reference").geometry, domain)
    buildings = gpd.read_file(args.results / "routes.gpkg", layer="evaluation_buildings")
    points = list(customers.geometry)
    building_points = list(buildings.geometry.centroid)
    geometries = {"Installed": reference}
    for label, layer in [("Urban4", "urban4"), ("Topotherm", "topotherm"), ("Candidate universe", "candidates")]:
        geometries[label] = route_union(gpd.read_file(args.results / "routes.gpkg", layer=layer).geometry, domain)
    descriptors = {label: geometry_descriptors(geometry, points, building_points, domain.area, case["topology_precision_m"])
                   for label, geometry in geometries.items() if label != "Candidate universe"}
    pd.DataFrame(descriptors).T.to_csv(args.results / "absolute_metrics.csv", index_label="network")
    rows = []
    for key, definition in DEFINITIONS.items():
        actual = descriptors["Installed"][key]
        rows.append({"measure":key,"definition":definition,"installed":actual,
                     "urban4":descriptors["Urban4"][key],"topotherm":descriptors["Topotherm"][key],
                     "urban4_to_installed":descriptors["Urban4"][key]/actual,
                     "topotherm_to_installed":descriptors["Topotherm"][key]/actual,
                     "held_out_geometry_diagnostic":key!="connection_spacing",
                     "evidence_family": {"length_per_area":"route_extent", "length_per_connection":"route_extent",
                                         "connection_spacing":"fixed_input_control",
                                         "building_to_network_distance":"building_proximity",
                                         "branch_length":"inferred_2d_topology",
                                         "terminal_share":"inferred_2d_topology"}[key]})
    pd.DataFrame(rows).to_csv(args.results / "table_xii_comparison.csv", index=False)
    spatial = []
    for label in ["Urban4", "Topotherm", "Candidate universe"]:
        geometry = geometries[label]
        for tolerance in [10.,25.,50.]:
            coverage = symmetric_corridor_coverage([reference],[geometry],tolerance)
            spatial.append({"network":label,"tolerance_m":tolerance,
                            "installed_route_covered_pct":coverage.installed_to_generated*100,
                            "generated_route_matching_pct":coverage.generated_to_installed*100,
                            "harmonic_mean_pct":coverage.harmonic_mean*100})
    pd.DataFrame(spatial).to_csv(args.results / "spatial_accuracy.csv",index=False)
    distances = {label:{"generated_to_installed":weighted_route_distance(geometry,reference),
                        "installed_to_generated":weighted_route_distance(reference,geometry)}
                 for label,geometry in geometries.items() if label in {"Urban4","Topotherm"}}
    json_write(args.results / "route_distance_errors.json",distances)
    topology = []
    for precision in [.25,.5,1.]:
        for label,geometry in geometries.items():
            if label=="Candidate universe":continue
            topology.append({"network":label,"precision_m":precision,**graph_descriptors(geometry,precision)})
    pd.DataFrame(topology).to_csv(args.results / "topology_precision_sensitivity.csv",index=False)
    json_write(args.results / "scoring_protocol.json",{
        "crs":"EPSG:2056","route_definition":"unique one-way geometric route, including identical service connectors; common window clipping",
        "table_xii_measures":DEFINITIONS,"connection_locations":"fixed observed inputs; connection spacing is a control",
        "dependent_measures":"Length per area and length per connection are two normalisations of the same route extent; do not count them as independent evidence",
        "topology_precision_m":case["topology_precision_m"],"building_filter":"OSM polygons of at least 20 m2 whose centroid is within the fixed window",
        "distance_sampling_m":5,"corridor_tolerances_m":[10,25,50],
        "calibration":"No installed route length, morphology, overlap, or proximity score is used to select model parameters",
        "interpretation":"Conditional mapped-route reconstruction; the cadastral transfer-point layer is not assumed to enumerate all actual customers"})
    print(pd.DataFrame(rows)[["measure","installed","urban4_to_installed","topotherm_to_installed"]].to_string(index=False))
    print(pd.DataFrame(spatial).to_string(index=False))
    print(json.dumps(distances,indent=2))


if __name__ == "__main__":
    main()
