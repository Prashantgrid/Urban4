"""Independent geometry and graph checks for the Andermatt accuracy benchmark."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, box

from scripts.benchmarks.run_andermatt_topotherm import contract_degree_two
from scripts.benchmarks.score_andermatt_topotherm import (
    geometry_descriptors, graph_descriptors, route_union, weighted_route_distance,
)


def test_geometric_route_length_is_unique_and_clipped() -> None:
    full = LineString([(-50, 0), (150, 0)])
    duplicate = LineString([(0, 0), (100, 0)])
    geometry = route_union([full, duplicate], box(0, -10, 100, 10))
    assert abs(geometry.length - 100) < 1e-9


def test_boundary_touches_do_not_create_zero_length_reference_routes() -> None:
    geometry = route_union([LineString([(-10, -10), (0, 0)]),
                            LineString([(0, 20), (100, 20)])], box(0, 0, 100, 100))
    assert geometry.geom_type == "LineString"
    assert geometry.distance(Point(0, 0)) == 20.


def test_parallel_route_distance_has_known_metric_units() -> None:
    reference = LineString([(0, 0), (100, 0)])
    parallel = LineString([(0, 5), (100, 5)])
    assert weighted_route_distance(parallel, reference)["mean_m"] == 5.
    assert weighted_route_distance(reference, parallel)["p95_m"] == 5.


def test_branch_scoring_suppresses_input_segmentation() -> None:
    whole = route_union([LineString([(0, 0), (100, 0)]),
                         LineString([(0, 0), (-100, 0)]),
                         LineString([(0, 0), (0, 100)])])
    segmented = route_union([LineString([(0, 0), (50, 0)]),
                             LineString([(50, 0), (100, 0)]),
                             LineString([(0, 0), (-100, 0)]),
                             LineString([(0, 0), (0, 100)])])
    for geometry in (whole, segmented):
        result = graph_descriptors(geometry, .5)
        assert result["branches"] == 3
        assert result["branch_length"] == 100.
        assert result["terminal_share"] == .75


def test_area_connection_and_building_metrics_use_one_shared_definition() -> None:
    geometry = LineString([(0, 0), (100, 0)])
    result = geometry_descriptors(geometry, [Point(0, 0), Point(100, 0)],
                                  [Point(20, 10), Point(80, 30)], 1e6)
    assert result["length_per_area"] == .1
    assert result["length_per_connection"] == 50.
    assert result["connection_spacing"] == 100.
    assert result["building_to_network_distance"] == 20.


def test_graph_contraction_retains_parallel_corridors_and_shortest_paths() -> None:
    graph = nx.MultiGraph()
    for u, v in [((0, 0), (1, 0)), ((1, 0), (2, 0)),
                 ((0, 0), (1, 1)), ((1, 1), (2, 0))]:
        line = LineString([u, v])
        graph.add_edge(u, v, geometry=line, length_km=line.length / 1000,
                       original_edges=[tuple(sorted((u, v)))])
    for name, node, point in [("source", (0, 0), (-1, 0)),
                              ("consumer_000", (2, 0), (3, 0))]:
        line = LineString([node, point])
        graph.add_edge(name, node, geometry=line, length_km=line.length / 1000,
                       original_edges=[])
    # Protect the two corridor junctions through a third stub each.
    for name, node, point in [("consumer_001", (0, 0), (0, -1)),
                              ("consumer_002", (2, 0), (2, -1))]:
        line = LineString([node, point])
        graph.add_edge(name, node, geometry=line, length_km=line.length / 1000,
                       original_edges=[])
    reduced = contract_degree_two(graph)
    assert reduced.number_of_edges((0, 0), (2, 0)) == 2
    for name in ("consumer_000", "consumer_001", "consumer_002"):
        a = nx.shortest_path_length(graph, "source", name, weight="length_km")
        b = nx.shortest_path_length(reduced, "source", name, weight="length_km")
        assert abs(a - b) < 1e-12
