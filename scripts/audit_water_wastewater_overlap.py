#!/usr/bin/env python3
"""Quantify exact corridor co-location of generated water and sewer mains."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Iterable

import networkx as nx
import pandas as pd

from urban4 import schweinfurt_base as base


def canonical_segment(a: Iterable[float], b: Iterable[float], digits: int = 7):
    p = (round(float(a[0]), digits), round(float(a[1]), digits))
    q = (round(float(b[0]), digits), round(float(b[1]), digits))
    return (p, q) if p <= q else (q, p)


def segment_map(frame: pd.DataFrame) -> dict[tuple, float]:
    segments: dict[tuple, float] = {}
    for row in frame.itertuples():
        points = json.loads(row.geometry_json)
        for a, b in zip(points[:-1], points[1:]):
            key = canonical_segment(a, b)
            segments.setdefault(key, base.distance_km(key[0], key[1]))
    return segments


def graph_metrics(frame: pd.DataFrame, directed: bool) -> dict[str, object]:
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from(zip(frame.from_node.astype(str), frame.to_node.astype(str)))
    if directed:
        return {
            "nodes": graph.number_of_nodes(),
            "links": graph.number_of_edges(),
            "weak_components": nx.number_weakly_connected_components(graph),
            "acyclic": nx.is_directed_acyclic_graph(graph),
            "cycle_rank": 0 if nx.is_directed_acyclic_graph(graph) else None,
        }
    components = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    return {
        "nodes": graph.number_of_nodes(),
        "links": graph.number_of_edges(),
        "components": components,
        "acyclic": nx.is_forest(graph),
        "cycle_rank": graph.number_of_edges() - graph.number_of_nodes() + components,
    }


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    source = project / "outputs" / "integrated_service_resolved"
    output = project / "outputs" / "water_wastewater_overlap"
    output.mkdir(parents=True, exist_ok=True)
    water_all = pd.read_csv(source / "drinking_water_links.csv")
    sewer_all = pd.read_csv(source / "wastewater_links.csv")
    water = water_all[water_all.link_type.eq("distribution_main")].copy()
    sewer = sewer_all[sewer_all.link_type.isin(["gravity_main", "force_main"])].copy()
    water_segments = segment_map(water)
    sewer_segments = segment_map(sewer)
    water_keys = set(water_segments)
    sewer_keys = set(sewer_segments)
    shared = water_keys & sewer_keys
    union = water_keys | sewer_keys
    water_length = sum(water_segments.values())
    sewer_length = sum(sewer_segments.values())
    shared_length = sum(min(water_segments[k], sewer_segments[k]) for k in shared)
    water_only = sum(water_segments[k] for k in water_keys - sewer_keys)
    sewer_only = sum(sewer_segments[k] for k in sewer_keys - water_keys)
    union_length = shared_length + water_only + sewer_only
    water_graph = graph_metrics(water, directed=False)
    sewer_graph = graph_metrics(sewer, directed=True)
    rows = [
        {"metric": "water_main_table_length_km", "value": float(water.length_km.sum()), "unit": "km"},
        {"metric": "sewer_main_table_length_km", "value": float(sewer.length_km.sum()), "unit": "km"},
        {"metric": "water_unique_geometry_length_km", "value": water_length, "unit": "km"},
        {"metric": "sewer_unique_geometry_length_km", "value": sewer_length, "unit": "km"},
        {"metric": "exact_shared_geometry_length_km", "value": shared_length, "unit": "km"},
        {"metric": "water_only_geometry_length_km", "value": water_only, "unit": "km"},
        {"metric": "sewer_only_geometry_length_km", "value": sewer_only, "unit": "km"},
        {"metric": "water_main_length_shared_percent", "value": 100.0 * shared_length / water_length, "unit": "%"},
        {"metric": "sewer_main_length_shared_percent", "value": 100.0 * shared_length / sewer_length, "unit": "%"},
        {"metric": "length_weighted_jaccard", "value": shared_length / union_length, "unit": "fraction"},
        {"metric": "water_cycle_rank", "value": water_graph["cycle_rank"], "unit": "count"},
        {"metric": "sewer_is_directed_acyclic", "value": int(bool(sewer_graph["acyclic"])), "unit": "boolean"},
        {"metric": "water_main_link_count", "value": len(water), "unit": "count"},
        {"metric": "sewer_main_link_count", "value": len(sewer), "unit": "count"},
        {"metric": "water_service_lateral_length_km", "value": float(water_all.loc[water_all.link_type.eq("building_service"), "length_km"].sum()), "unit": "km"},
        {"metric": "sewer_property_lateral_length_km", "value": float(sewer_all.loc[sewer_all.link_type.isin(["property_lateral", "property_force_main"]), "length_km"].sum()), "unit": "km"},
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "water_wastewater_overlap_summary.csv", index=False)

    features = []
    for key in union:
        if key in shared:
            kind = "shared"
        elif key in water_keys:
            kind = "water_only"
        else:
            kind = "sewer_only"
        features.append({
            "type": "Feature",
            "properties": {"class": kind, "length_km": base.distance_km(key[0], key[1])},
            "geometry": {"type": "LineString", "coordinates": [list(key[0]), list(key[1])]},
        })
    (output / "water_wastewater_overlap_segments.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "comparison_scope": "one-way public-road-constrained distribution/collection mains; building service laterals excluded from overlap",
        "segment_identity": "undirected geometry segment endpoints rounded to seven decimal places",
        "interpretation": "street co-location is expected, but the approximately 95% exact overlap is a diagnostic of the present shared-road/building-attachment construction and should not be treated as installed-edge validation",
        "water_graph": water_graph,
        "sewer_graph": sewer_graph,
        "metrics": {row["metric"]: row["value"] for row in rows},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
