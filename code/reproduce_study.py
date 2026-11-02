#!/usr/bin/env python3
"""Reproduce the manuscript sensitivities and figures from retained results.

Use --from-models to run the complete heat-power and water-power studies on
the native model exports created by run_all.py. Outputs are written separately
from the retained reference tables so a rerun cannot overwrite that evidence.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT,
                        help="Repository containing the rebuilt native models")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/manuscript_study")
    parser.add_argument("--from-models", action="store_true",
                        help="Recompute all case studies from rebuilt native models")
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    references = ROOT / "results"
    if output == ROOT or output.is_relative_to(references) or references.is_relative_to(output):
        parser.error("--output must be separate from the retained results directory")

    if args.from_models:
        models = repo / "outputs/municipal_scale_v2.5.0"
        required = ["electricity_municipal_pandapower.json",
                    "district_heating_municipal_pandapipes.json",
                    "drinking_water_municipal_peak.inp",
                    "drinking_water_municipal_average.inp"]
        missing = [str(models / name) for name in required if not (models / name).is_file()]
        if missing:
            parser.error("Run python run_all.py first; missing native models: " + ", ".join(missing))

    heat = output / "results/heat_power_case"
    water = output / "results/water_power_case"
    figures = output / "figures"
    for path in (heat, water, figures):
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    def run(name: str, *options: object) -> None:
        command = [sys.executable, str(CODE / name), *map(str, options)]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    if args.from_models:
        run("run_heat_power_case.py", "--repo", repo, "--out", heat)
        run("run_heat_power_variants.py", "--repo", repo, "--out", heat)
        run("run_water_power_case.py", "--repo", repo, "--out", water)
        run("run_water_reliability.py", "--repo", repo, "--case", water, "--out", water)
        run("run_flood_case.py", "--repo", repo, "--water", water, "--heat", heat,
            "--flood", CODE / "data/LfU_Schweinfurt_HQ100_HQextrem.gpkg", "--out", water)
        run("run_extreme_events.py", "--repo", repo, "--water", water, "--heat", heat,
            "--out", water)
    else:
        for name in ("heat_power_case", "water_power_case"):
            shutil.copytree(references / name, output / "results" / name, dirs_exist_ok=True)

    run("run_base_load_sensitivity.py", "--results", heat)
    run("run_capacity_sensitivity.py", "--results", heat)
    run("make_fig03_design_relations.py", "--out", figures)
    run("make_fig06_seattle.py", "--out", figures)
    if args.from_models:
        run("make_fig04_settlements.py", "--repo", repo, "--out", figures)
        run("make_fig05_schweinfurt.py", "--repo", repo, "--out", figures)
        run("make_fig07_heat_power.py", "--repo", repo, "--results", heat, "--out", figures)
        run("make_fig08_water_power.py", "--repo", repo, "--results", water, "--out", figures)
        run("make_fig09_extreme_events.py", "--repo", repo, "--results", water,
            "--flood", CODE / "data/LfU_Schweinfurt_HQ100_HQextrem.gpkg", "--out", figures)
    else:
        run("make_fig07_heat_power_revised.py", "--results", heat, "--out", figures)
    print(f"Study outputs: {output}")


if __name__ == "__main__":
    main()
