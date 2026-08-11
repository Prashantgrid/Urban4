from __future__ import annotations

import networkx as nx
import pandas as pd

from urban4.route_aware_heat import (
    RouteAwareHeatConfig,
    allocate_integer_quotas,
    select_route_aware_heat_connections,
)


def test_integer_quota_respects_total_and_ceilings() -> None:
    q = allocate_integer_quotas({"a": 3.0, "b": 1.0}, 5, {"a": 3, "b": 4})
    assert sum(q.values()) == 5
    assert q["a"] <= 3 and q["b"] <= 4


def test_shared_route_is_rewarded() -> None:
    graph = nx.Graph()
    for i in range(6):
        graph.add_node((float(i), 0.0))
    for i in range(5):
        graph.add_edge((float(i), 0.0), (float(i + 1), 0.0), length_km=0.1)
    candidates = pd.DataFrame([
        {"building_id": "near1", "road_node": (2.0, 0.0), "road_lon": 2.0, "road_lat": 0.0,
         "service_length_km": 0.01, "heat_candidate_mwh_year": 100.0, "heat_territory_id": 1,
         "heat_cell_x": 0, "heat_cell_y": 0},
        {"building_id": "near2", "road_node": (3.0, 0.0), "road_lon": 3.0, "road_lat": 0.0,
         "service_length_km": 0.01, "heat_candidate_mwh_year": 100.0, "heat_territory_id": 1,
         "heat_cell_x": 0, "heat_cell_y": 0},
        {"building_id": "far", "road_node": (5.0, 0.0), "road_lon": 5.0, "road_lat": 0.0,
         "service_length_km": 0.01, "heat_candidate_mwh_year": 180.0, "heat_territory_id": 1,
         "heat_cell_x": 0, "heat_cell_y": 0},
    ])
    selected, meta, audit = select_route_aware_heat_connections(
        graph, candidates, target_count=2,
        config=RouteAwareHeatConfig(spatial_quota_level="territory"),
        root_by_territory={1: (0.0, 0.0)},
    )
    assert set(selected.building_id) == {"near1", "near2"}
    assert meta["selected_connection_count"] == 2
    assert audit.incremental_trench_km.sum() <= 0.3 + 1e-9


def test_cell_quota_preserves_spatial_coverage() -> None:
    graph = nx.Graph()
    for i in range(7):
        graph.add_node((float(i), 0.0))
    for i in range(6):
        graph.add_edge((float(i), 0.0), (float(i + 1), 0.0), length_km=0.1)
    rows = []
    for i, heat in enumerate([100.0, 90.0, 80.0, 70.0]):
        rows.append({"building_id": f"a{i}", "road_node": (float(i + 1), 0.0), "road_lon": float(i + 1), "road_lat": 0.0,
                     "service_length_km": 0.01, "heat_candidate_mwh_year": heat, "heat_territory_id": 1,
                     "heat_cell_x": 0, "heat_cell_y": 0})
    for i, heat in enumerate([30.0, 25.0]):
        rows.append({"building_id": f"b{i}", "road_node": (float(i + 5), 0.0), "road_lon": float(i + 5), "road_lat": 0.0,
                     "service_length_km": 0.01, "heat_candidate_mwh_year": heat, "heat_territory_id": 1,
                     "heat_cell_x": 1, "heat_cell_y": 0})
    selected, _, _ = select_route_aware_heat_connections(graph, pd.DataFrame(rows), target_count=4)
    assert selected[["heat_cell_x", "heat_cell_y"]].drop_duplicates().shape[0] == 2
