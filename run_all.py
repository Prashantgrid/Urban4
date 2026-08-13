#!/usr/bin/env python3
"""Rebuild Urban4 topology cases, interfaces, figures, tests, and manuscript."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_output_checksums() -> None:
    checksum_file = PROJECT / "outputs" / "SHA256SUMS.csv"
    rows = []
    for path in sorted((PROJECT / "outputs").rglob("*")):
        if path.is_file() and path != checksum_file:
            rows.append((str(path.relative_to(PROJECT)), _sha256(path), path.stat().st_size))
    with checksum_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file", "sha256", "bytes"])
        writer.writerows(rows)


def _run(command: list[str], cwd: Path = PROJECT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-osm", action="store_true", help="Refresh the retained OSM evidence snapshot")
    parser.add_argument(
        "--skip-base", action="store_true",
        help="Reuse the legacy reduced model used only by the two matched response projections",
    )
    parser.add_argument(
        "--skip-service-resolved", action="store_true",
        help="Reuse the accepted building/service-resolved integrated outputs",
    )
    parser.add_argument("--skip-paper", action="store_true", help="Generate models and figures without compiling LaTeX")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/mpl-urban4")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((str(PROJECT), existing_pythonpath))
        if existing_pythonpath
        else str(PROJECT)
    )
    start = time.perf_counter()
    base_runtime = 0.0
    if not args.skip_base:
        command = [sys.executable, "-m", "urban4.schweinfurt_base"]
        if args.refresh_osm:
            command.append("--refresh")
        base_start = time.perf_counter()
        subprocess.run(command, cwd=PROJECT, env=environment, check=True)
        base_runtime = time.perf_counter() - base_start

    # The matched waterworks perturbations use the retained reduced model.
    # They are response projections, not the service-resolved release topology.
    framework_code = "from urban4.framework import run_framework; " f"run_framework(base_runtime={base_runtime!r})"
    subprocess.run([sys.executable, "-c", framework_code], cwd=PROJECT, env=environment, check=True)
    subprocess.run([sys.executable, "-m", "urban4.coupling_scenario"], cwd=PROJECT, env=environment, check=True)

    # Independent, building/service-resolved topology experiments and final city case.
    subprocess.run([sys.executable, "-m", "urban4.benchmark_cases"], cwd=PROJECT, env=environment, check=True)
    if not args.skip_service_resolved:
        subprocess.run(
            [sys.executable, "-m", "urban4.integrated_generation"],
            cwd=PROJECT, env=environment, check=True,
        )
    # Preserve the original shared-corridor construction as the declared
    # comparison baseline before generating the principle-aligned alternatives.
    # The latter records the reduction relative to these metrics in its manifest.
    _run([sys.executable, "scripts/audit_water_wastewater_overlap.py"], env=environment)
    # The municipality-scale release is built from a deterministic structural
    # layer and then executed in all four native solvers.  These are required
    # clean-build steps, not pre-generated optional artefacts.
    _run([sys.executable, "-m", "urban4.principle_water_sewer"], env=environment)
    _run([sys.executable, "scripts/run_integrated_interface_closure.py"], env=environment)
    _run([sys.executable, "scripts/refresh_integrated_acceptance.py"], env=environment)
    _run([sys.executable, "-m", "urban4.interface_response"], env=environment)
    _run([sys.executable, "-m", "urban4.network_contingency_scenarios", "--prescreen"], env=environment)
    _run([sys.executable, "-m", "urban4.network_contingency_scenarios", "--line", "PL044"], env=environment)
    _run(
        [
            sys.executable, "scripts/audit_figure3_topologies.py", "--case",
            "outputs/benchmark_cases/dense_urban", "--output",
            "outputs/figure3_topology_audit.json",
        ],
        env=environment,
    )
    # The generated manuscript tables depend on the independently executable
    # terrain/grade sensitivity.  Keep it in the clean-build path so a fresh
    # checkout cannot fail late with missing sensitivity outputs.
    _run([sys.executable, "-m", "urban4.sensitivity"], env=environment)
    _run([sys.executable, "-m", "urban4.heat_sensitivity"], env=environment)
    # The publication figures use the threshold and policy sweeps in addition
    # to the native heat sensitivity.  Regenerate them here so a fresh checkout
    # does not depend on tables retained from an earlier release bundle.
    _run([sys.executable, "scripts/run_heat_selection_analysis.py"], env=environment)
    _run([sys.executable, "scripts/write_results_tex.py"], env=environment)
    _run([sys.executable, "scripts/write_submission_audits.py"], env=environment)
    # Regenerate the compact v1.5.x inventory tables from the current
    # service-resolved interface states.  Older bundles carried these files as
    # prebuilt outputs, which made a clean checkout fail later in the practical
    # sizing audit and source-inventory tests.
    _run([sys.executable, "scripts/export_source_and_asset_inventory.py"], env=environment)
    # Practical constructability and equipment-size audit.  This does not alter
    # the reproduced numerical baseline; any failed screen requires the
    # declared repair and complete native/coupled rerun.
    _run([sys.executable, "scripts/export_practical_design.py"], env=environment)
    _run(
        [sys.executable, "scripts/make_process_figures.py", "--data-root", ".", "--output", "figures"],
        env=environment,
    )
    _run([sys.executable, "scripts/make_figures.py"], env=environment)
    _run([sys.executable, "scripts/make_two_way_stress_figures.py"], env=environment)
    _run([sys.executable, "scripts/build_municipal_scale_design.py"], env=environment)
    _run([sys.executable, "scripts/make_municipal_scale_figure.py"], env=environment)
    _run([sys.executable, "-m", "pytest", "-q"], env=environment)
    _write_output_checksums()

    if not args.skip_paper:
        # In the submission bundle the journal source is a sibling of this
        # reviewer archive.  Keep paper compilation separate from generated
        # result fragments, which remain under ``code_and_results/manuscript``.
        paper_dir = PROJECT.parent / "manuscript"
        paper_tex = paper_dir / "Urban4_IEEE_Submission_v2.7.0.tex"
        if not paper_tex.exists():
            raise FileNotFoundError(
                f"Journal source not found at {paper_tex}; use --skip-paper "
                "when running the code/results archive on its own."
            )
        output = PROJECT / "output" / "pdf"
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            PROJECT / "figures" / "fig07_schweinfurt_municipal_scale_v2_5_0.pdf",
            paper_dir / "figures" / "Fig07_Schweinfurt_Municipal_Small_Multiples.pdf",
        )
        shutil.copy2(
            PROJECT / "figures" / "Fig09_Electrical_Supply_Disturbance.pdf",
            paper_dir / "Fig09_Electrical_Supply_Disturbance.pdf",
        )
        latex_command = [
            "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
            paper_tex.name,
        ]
        _run(latex_command, cwd=paper_dir, env=environment)
        _run(["bibtex", paper_tex.stem], cwd=paper_dir, env=environment)
        _run(latex_command, cwd=paper_dir, env=environment)
        _run(latex_command, cwd=paper_dir, env=environment)
        shutil.copy2(
            paper_tex.with_suffix(".pdf"),
            output / "Urban4_IEEE_Submission_v2.7.0.pdf",
        )
    print(f"Complete workflow finished in {time.perf_counter() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
