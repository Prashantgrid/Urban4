#!/usr/bin/env python3
"""Replay the shared graph and independently audit the saved Topotherm solution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmarks.run_andermatt_topotherm import (
    prepare_roads, attach_and_run_urban4, contract_degree_two, json_write,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/andermatt_benchmark")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/andermatt_topotherm")
    args = parser.parse_args()
    data, out = args.data, args.results
    case = json.loads((data / "case.json").read_text())
    customers = gpd.read_file(data / "common_inputs.gpkg", layer="customers")
    source = gpd.read_file(data / "common_inputs.gpkg", layer="source").geometry.iloc[0]
    roads, _, _ = prepare_roads(data, case)
    _, full, _, _, _, _ = attach_and_run_urban4(roads, customers, source, case)
    candidate = contract_degree_two(full)
    edges = list(candidate.edges(keys=True, data=True))
    decisions = pd.read_csv(out / "topotherm_edge_decisions.csv").set_index("edge")
    run = json.loads((out / "topotherm_run.json").read_text())
    lengths = np.array([attrs["length_km"] * 1000 for *_, attrs in edges])
    np.testing.assert_allclose(decisions.length_m, lengths, rtol=1e-12, atol=1e-9)
    original_dist = nx.single_source_dijkstra_path_length(full, "source", weight="length_km")
    reduced_dist = nx.single_source_dijkstra_path_length(candidate, "source", weight="length_km")
    contraction_error = max(abs(original_dist[f"consumer_{i:03d}"] - reduced_dist[f"consumer_{i:03d}"])
                            for i in range(len(customers)))
    assert contraction_error < 1e-10

    balance = {node: 0. for node in candidate}
    selected = nx.MultiGraph()
    max_integrality = 0.
    minimum_outflow = 0.
    maximum_capacity_violation = 0.
    mapping = []
    for i, (u, v, key, attrs) in enumerate(edges):
        row = decisions.loc[i]
        li, lj = float(row.lambda_ij), float(row.lambda_ji)
        pi, pj = float(row.flow_ij_in_kw), float(row.flow_ji_in_kw)
        length = lengths[i]
        # Topotherm 0.6.0's declared default thermal-loss regression.
        oi = (1 - 4.348e-7 * length) * pi - .02189 * length * li
        oj = (1 - 4.348e-7 * length) * pj - .02189 * length * lj
        balance[u] += oj - pi
        balance[v] += oi - pj
        max_integrality = max(max_integrality, abs(li-round(li)), abs(lj-round(lj)))
        minimum_outflow = min(minimum_outflow, oi, oj)
        maximum_capacity_violation = max(maximum_capacity_violation,
                                          pi - 69000 * li, pj - 69000 * lj)
        assert li + lj <= 1 + 1e-7
        if li + lj > .5:
            selected.add_edge(u, v, key=key)
        mapping.append({"edge":i,"from_node":str(u),"to_node":str(v),
                        "original_road_edges":len(attrs["original_edges"]),"length_m":length})
    balance["source"] += run["source_supply_kw"]
    for i, load in enumerate(customers.peak_kw):
        balance[f"consumer_{i:03d}"] -= float(load)
        assert nx.has_path(selected, "source", f"consumer_{i:03d}")
    residual = max(abs(x) for x in balance.values())
    assert residual < 1e-4
    assert max_integrality < 1e-6 and minimum_outflow > -1e-4 and maximum_capacity_violation < 1e-4
    upper, lower = float(run["upper_bound"]), float(run["lower_bound"])
    gap = (upper-lower)/abs(upper)
    assert gap <= .010001
    result = {"all_observed_customers_connected":len(customers),
              "maximum_nodal_balance_residual_kw":residual,
              "maximum_binary_integrality_error":max_integrality,
              "minimum_pipe_outflow_kw":minimum_outflow,
              "maximum_pipe_capacity_violation_kw":maximum_capacity_violation,
              "maximum_contraction_distance_error_km":contraction_error,
              "relative_objective_gap":gap,
              "same_candidate_graph_verified":True,
              "objective_and_regression":"unmodified Topotherm 0.6.0 single-timestep forced-connection defaults; constant revenue omitted"}
    json_write(out / "solution_audit.json",result)
    pd.DataFrame(mapping).to_csv(out / "candidate_edge_mapping.csv",index=False)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
