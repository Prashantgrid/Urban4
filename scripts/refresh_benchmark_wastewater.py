#!/usr/bin/env python3
"""Regenerate and re-screen selected service-resolved benchmark sewers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import pandas as pd

from urban4.benchmark_cases import BENCHMARK_OUT, _tile_road_graph
from urban4.framework import _screen_solver_results, load_case
from urban4.service_topologies import generate_wastewater


def main(case_ids: list[str]) -> None:
    config = load_case()
    definitions = {case["id"]: case for case in config["benchmark_cases"]}
    root_manifest_path = BENCHMARK_OUT / "manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    for case_id in case_ids:
        case = definitions[case_id]
        case_dir = BENCHMARK_OUT / case_id
        buildings = pd.read_csv(case_dir / "common_building_ledger.csv")
        road = _tile_road_graph(case["bbox"])
        _, _, _, solver = generate_wastewater(case_dir, road, buildings)
        manifest_path = case_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["solver_readiness"]["swmm"] = solver
        manifest["acceptance_screening"]["wastewater"] = _screen_solver_results(
            solver, "wastewater", config
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        root_manifest[case_id] = manifest
        print(json.dumps({
            "case_id": case_id,
            "solver": solver,
            "screen": manifest["acceptance_screening"]["wastewater"],
        }, indent=2), flush=True)
    root_manifest_path.write_text(
        json.dumps(root_manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv[1:] or ["semi_urban", "rural_peripheral"])
