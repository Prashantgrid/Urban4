#!/usr/bin/env python3
"""Structural sensitivity of sewer lift requirements to terrain and grade.

The generated directed sewer topology, link lengths, and outfall are held
fixed.  Node relief is scaled about the outfall elevation and the same
cover-bounded invert-design rule is repeated for a declared grid of minimum
gravity grades.  The experiment isolates the effect of the Class-C terrain
prior and one non-clustering engineering parameter; it is not a probabilistic
uncertainty analysis or a substitute for a surveyed DEM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
CASE_PATH = PROJECT / "cases" / "schweinfurt.json"
SENSITIVITY_OUT = OUT / "sensitivity"


def _load_case() -> dict[str, Any]:
    return json.loads(CASE_PATH.read_text(encoding="utf-8"))


def _design_lift_edges(
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    *,
    relief_scale: float,
    minimum_grade: float,
    minimum_cover_m: float,
    maximum_cover_m: float,
) -> tuple[set[tuple[str, str]], dict[str, float]]:
    """Repeat the baseline cover-bounded invert design on a fixed graph."""
    graph = nx.DiGraph()
    for row in links.itertuples():
        graph.add_edge(
            str(row.from_node),
            str(row.to_node),
            length_m=float(row.length_m),
        )
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Wastewater sensitivity requires a directed acyclic topology")

    indexed = nodes.set_index("node_id", drop=False)
    outfalls = indexed.index[indexed["node_type"] == "outfall"].tolist()
    if len(outfalls) != 1:
        raise ValueError("Wastewater sensitivity requires exactly one outfall")
    outfall = str(outfalls[0])
    outfall_ground = float(indexed.loc[outfall, "ground_elevation_m"])
    ground = outfall_ground + relief_scale * (
        indexed["ground_elevation_m"].astype(float) - outfall_ground
    )

    inverts: dict[str, float] = {outfall: float(ground.loc[outfall] - 3.0)}
    lift_edges: set[tuple[str, str]] = set()
    for downstream in reversed(list(nx.topological_sort(graph))):
        if downstream not in inverts:
            continue
        for upstream in graph.predecessors(downstream):
            length_m = float(graph[upstream][downstream]["length_m"])
            desired = inverts[downstream] + minimum_grade * length_m
            shallow_limit = float(ground.loc[upstream] - minimum_cover_m)
            deep_limit = float(ground.loc[upstream] - maximum_cover_m)
            if desired > shallow_limit:
                inverts[upstream] = float(ground.loc[upstream] - 2.0)
                lift_edges.add((str(upstream), str(downstream)))
            else:
                inverts[upstream] = max(desired, deep_limit)
    if len(inverts) != len(indexed):
        raise ValueError("Every wastewater node must reach the outfall")
    return lift_edges, inverts


def run_wastewater_terrain_grade_sensitivity() -> dict[str, Any]:
    """Write the declared sensitivity matrix and a machine-readable manifest."""
    config = _load_case()
    wastewater = config["wastewater"]
    design = wastewater["terrain_grade_sensitivity"]
    nodes = pd.read_csv(OUT / "wastewater_nodes.csv")
    links = pd.read_csv(OUT / "wastewater_conduits.csv")
    link_lengths = {
        (str(row.from_node), str(row.to_node)): float(row.length_m)
        for row in links.itertuples()
    }
    rows: list[dict[str, float | int | bool]] = []
    baseline_grade = float(wastewater["minimum_gravity_grade"])
    for relief_scale in design["relief_scale_factors"]:
        for minimum_grade in design["minimum_gravity_grades"]:
            lift_edges, inverts = _design_lift_edges(
                nodes,
                links,
                relief_scale=float(relief_scale),
                minimum_grade=float(minimum_grade),
                minimum_cover_m=float(wastewater["minimum_cover_m"]),
                maximum_cover_m=float(wastewater["maximum_cover_m"]),
            )
            force_main_km = sum(link_lengths[edge] for edge in lift_edges) / 1000.0
            rows.append(
                {
                    "relief_scale": float(relief_scale),
                    "minimum_gravity_grade": float(minimum_grade),
                    "lift_edge_count": len(lift_edges),
                    "force_main_length_km": round(force_main_km, 6),
                    "minimum_invert_elevation_m": round(min(inverts.values()), 3),
                    "maximum_invert_elevation_m": round(max(inverts.values()), 3),
                    "is_baseline": bool(
                        abs(float(relief_scale) - 1.0) < 1e-12
                        and abs(float(minimum_grade) - baseline_grade) < 1e-12
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    baseline = frame[frame["is_baseline"]]
    if len(baseline) != 1:
        raise ValueError("Sensitivity matrix must contain exactly one baseline row")
    accepted_lifts = int((nodes["node_type"] == "pump").sum())
    accepted_force_main_km = float(
        links.loc[links["link_type"] == "force_main", "length_m"].sum() / 1000.0
    )
    baseline_row = baseline.iloc[0]
    if int(baseline_row["lift_edge_count"]) != accepted_lifts:
        raise ValueError("Sensitivity baseline does not reproduce accepted lift-edge count")
    if abs(float(baseline_row["force_main_length_km"]) - accepted_force_main_km) > 1e-5:
        raise ValueError("Sensitivity baseline does not reproduce accepted force-main length")

    SENSITIVITY_OUT.mkdir(parents=True, exist_ok=True)
    csv_path = SENSITIVITY_OUT / "wastewater_terrain_grade_sensitivity.csv"
    frame.to_csv(csv_path, index=False)
    manifest = {
        "experiment": "fixed-topology wastewater terrain-grade structural sensitivity",
        "method": design["method"],
        "interpretation": design["interpretation"],
        "topology_held_fixed": True,
        "node_count": int(len(nodes)),
        "link_count": int(len(links)),
        "baseline": {
            "relief_scale": 1.0,
            "minimum_gravity_grade": baseline_grade,
            "lift_edge_count": accepted_lifts,
            "force_main_length_km": round(accepted_force_main_km, 6),
        },
        "ranges": {
            "lift_edge_count": [int(frame["lift_edge_count"].min()), int(frame["lift_edge_count"].max())],
            "force_main_length_km": [
                round(float(frame["force_main_length_km"].min()), 6),
                round(float(frame["force_main_length_km"].max()), 6),
            ],
        },
        "official_held_out_comparison": {
            "pump_station_count": int(config["official_anchors"]["wastewater_pump_stations"]),
            "force_main_length_km": float(config["official_anchors"]["wastewater_force_main_km"]),
            "role": "context only; not fitted in this sensitivity",
        },
        "outputs": {"matrix_csv": str(csv_path.relative_to(PROJECT))},
    }
    manifest_path = SENSITIVITY_OUT / "wastewater_terrain_grade_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run_wastewater_terrain_grade_sensitivity(), indent=2))
