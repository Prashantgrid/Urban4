#!/usr/bin/env python3
"""Approximate-Steiner routing baseline for the accepted district-heat customer set.

The baseline holds the selected customers and source assignments fixed.  It
therefore tests only the route-construction heuristic, not customer selection or
source placement.  Each source-centred road subgraph is approximated with the
Mehlhorn Steiner-tree heuristic available in NetworkX.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx
import pandas as pd
from networkx.algorithms.approximation.steinertree import steiner_tree

from urban4 import schweinfurt_base as base


PROJECT = Path(__file__).resolve().parents[1]
CASE = PROJECT / "outputs" / "integrated_service_resolved"
OUTPUT = PROJECT / "outputs" / "heat_steiner_baseline"


def _road_edge_key(u: tuple[float, float], v: tuple[float, float]) -> frozenset:
    return frozenset((u, v))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    road = base.build_road_graph(base.load_elements("local_roads.json"))
    nodes = pd.read_csv(CASE / "district_heating_nodes.csv").set_index("node_id")
    services = pd.read_csv(CASE / "district_heating_service_connections.csv")
    corridors = pd.read_csv(CASE / "district_heating_corridors.csv")

    route_only = corridors[corridors["link_type"].eq("route")]
    urban4_route_km = float(route_only["length_km"].sum())
    retained_edges: set[frozenset] = set()
    rows: list[dict] = []
    started = time.perf_counter()

    for source_id, group in services.groupby("heat_source_node"):
        source = nodes.loc[str(source_id)]
        terminals = {
            (float(source.lon), float(source.lat)),
            *{
                (float(nodes.loc[str(node_id)].lon), float(nodes.loc[str(node_id)].lat))
                for node_id in group["route_node"].astype(str)
            },
        }
        missing = [terminal for terminal in terminals if terminal not in road]
        if missing:
            road_index = base.NodeIndex(road.nodes)
            terminals = {road_index.nearest(terminal) for terminal in terminals}

        source_started = time.perf_counter()
        try:
            tree = steiner_tree(road, terminals, weight="length_km", method="mehlhorn")
            method = "NetworkX Mehlhorn"
        except TypeError:
            tree = steiner_tree(road, terminals, weight="length_km")
            method = "NetworkX Steiner approximation (installed default)"
        length_km = float(
            sum(float(data.get("length_km", 0.0)) for _, _, data in tree.edges(data=True))
        )
        for u, v in tree.edges:
            retained_edges.add(_road_edge_key(u, v))
        rows.append(
            {
                "heat_source_node": str(source_id),
                "customer_count": int(len(group)),
                "terminal_count": int(len(terminals)),
                "steiner_route_km": length_km,
                "runtime_s": time.perf_counter() - source_started,
                "method": method,
            }
        )

    unique_route_km = 0.0
    for u, v, data in road.edges(data=True):
        if _road_edge_key(u, v) in retained_edges:
            unique_route_km += float(data.get("length_km", 0.0))

    runtime_s = time.perf_counter() - started
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "source_level_steiner_baseline.csv", index=False)
    summary = {
        "status": "executed",
        "comparison_scope": (
            "same accepted district-heat customers and source assignments; "
            "routing heuristic only"
        ),
        "urban4_route_km": urban4_route_km,
        "steiner_unique_route_km": unique_route_km,
        "relative_excess_route_percent": (
            100.0 * (urban4_route_km / unique_route_km - 1.0)
            if unique_route_km > 0.0 else None
        ),
        "runtime_s": runtime_s,
        "source_count": int(len(result)),
        "method": sorted(result["method"].unique().tolist()) if len(result) else [],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
