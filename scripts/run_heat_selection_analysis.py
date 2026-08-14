#!/usr/bin/env python3
"""Regenerate district-heating selection ablations and structural diagnostics."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from urban4.framework import load_case
from urban4 import schweinfurt_base as base
from urban4.service_topologies import generate_district_heat


def _run(policy: str, threshold: float, root: Path, buildings: pd.DataFrame, road, config):
    out = root / policy / f"density_{int(threshold):04d}"
    out.mkdir(parents=True, exist_ok=True)
    nodes, corridors, services, solver = generate_district_heat(
        out,
        road,
        buildings,
        connection_count=int(config["district_heating"]["sales_contract_count"]),
        plant_capacity_mw=30.0,
        seed=int(config["seed"]),
        annual_heat_target_mwh=float(config["district_heating"]["heat_sales_mwh_year"]),
        design_peak_target_mw=float(config["district_heating"]["design_peak_mw_assumption"]),
        minimum_heat_density_mwh_ha=float(threshold),
        heat_density_cell_size_km=float(config["district_heating"]["heat_density_cell_size_km"]),
        enforce_connection_count=True,
        selection_policy=policy,
        route_aware_demand_exponent=float(config["district_heating"].get("route_aware_demand_exponent", 1.0)),
        route_aware_spatial_quota_level=str(config["district_heating"].get("route_aware_spatial_quota_level", "cell")),
        published_network_length_km=float(
            config["district_heating"]["route_length_target_km"]
        ),
        network_length_tolerance_fraction=float(
            config["district_heating"].get("route_length_tolerance_fraction", 0.10)
        ),
        network_length_scope=str(
            config["district_heating"].get(
                "route_length_comparison_scope", "route_plus_service_one_way"
            )
        ),
        return_pressure_reference_bar=float(
            config["bidirectional_coupling"]["heat_return_pressure_bar"]
        ),
    )
    (out / "district_heating_solver_results.json").write_text(
        json.dumps(solver, indent=2, default=str) + "\n", encoding="utf-8"
    )
    ledger = buildings.set_index("building_id")
    selected_ids = services.building_id.astype(str).tolist() if not services.empty else []
    unscaled_heat = float(ledger.loc[selected_ids, "heat_candidate_mwh_year"].sum()) if selected_ids else 0.0
    selected_cells = 0
    selected_territories = 0
    if selected_ids:
        # Cell/territory counts are retained in the route-aware audit; for the
        # demand-ranked case, the connection table is sufficient for building IDs.
        audit_path = out / "district_heating_selection_audit.csv"
        if audit_path.exists() and audit_path.stat().st_size > 1:
            audit = pd.read_csv(audit_path)
            if not audit.empty:
                selected_cells = int(audit[["heat_cell_x", "heat_cell_y"]].drop_duplicates().shape[0])
                selected_territories = int(audit.heat_territory_id.nunique())
        if selected_cells == 0:
            selected_cells = int(solver["suitability"].get("suitable_cell_count", 0))
            selected_territories = int(solver["suitability"].get("territory_count", 0))
    return {
        "selection_policy": policy,
        "minimum_heat_density_mwh_ha": float(threshold),
        "suitable_cells": int(solver["suitability"].get("suitable_cell_count", 0)),
        "territories": int(solver["suitability"].get("territory_count", 0)),
        "selected_cells": selected_cells,
        "selected_territories": selected_territories,
        "connections": len(services),
        "plants": int(nodes.node_type.eq("plant").sum()) if not nodes.empty else 0,
        "route_trench_km": float(solver.get("route_trench_length_km", 0.0)),
        "service_connection_km": float(solver.get("service_connection_length_km", 0.0)),
        "one_way_network_km": float(solver.get("one_way_network_length_km", 0.0)),
        "published_network_length_km": float(
            solver.get(
                "published_network_length_km",
                config["district_heating"]["route_length_target_km"],
            )
        ),
        "inventory_length_relative_error": float(
            solver.get("inventory_length_relative_error", math.nan)
        ),
        "inventory_length_within_tolerance": bool(
            solver.get("inventory_length_within_tolerance", False)
        ),
        "paired_pipe_km": float(solver.get("paired_physical_pipe_length_km", 0.0)),
        "annual_heat_loss_mwh": float(solver.get("annual_heat_loss_mwh", 0.0)),
        "annual_heat_loss_fraction": float(solver.get("annual_heat_loss_fraction", 0.0)),
        "catalogue_max_velocity_m_s": float(solver.get("catalogue_maximum_velocity_m_s", 0.0)),
        "catalogue_max_pressure_gradient_pa_m": float(solver.get("catalogue_maximum_pressure_gradient_pa_m", 0.0)),
        "selected_unscaled_heat_mwh": unscaled_heat,
        "delivered_heat_mwh": float(services.annual_heat_mwh.sum()) if not services.empty else 0.0,
        "linear_heat_density_mwh_per_route_km": (
            float(services.annual_heat_mwh.sum()) / float(solver.get("route_trench_length_km", 0.0))
            if float(solver.get("route_trench_length_km", 0.0)) > 0 else 0.0
        ),
        "native_solver_executed": bool(solver.get("created", False)),
        "native_solver_converged": bool(solver.get("converged", False)),
        "native_solver_note": solver.get("native_solver_status", solver.get("error", "")),
        "selected_ids": selected_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=None,
        help="Heat-density thresholds in MWh ha^-1 a^-1. Default: configured full sweep.",
    )
    parser.add_argument(
        "--policies", nargs="+", choices=("demand_ranked", "route_aware"),
        default=("demand_ranked", "route_aware"),
        help="Customer-selection policies to execute.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Optional output directory. Default: outputs/heat_selection_analysis.",
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output_root = args.output_root or (project / "outputs" / "heat_selection_analysis")
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_case()
    buildings = pd.read_csv(project / "outputs" / "building_sector_demands.csv")
    road = base.build_road_graph(base.load_elements("local_roads.json"))
    thresholds = args.thresholds or config["district_heating"].get(
        "heat_suitability_extended_sensitivity_mwh_ha",
        [200.0, 400.0, 600.0, 800.0, 1000.0],
    )
    records = []
    selected_sets = {}
    for policy in args.policies:
        for threshold in thresholds:
            row = _run(policy, float(threshold), output_root, buildings, road, config)
            selected_sets[(policy, float(threshold))] = set(row.pop("selected_ids"))
            records.append(row)
    frame = pd.DataFrame(records)
    frame.to_csv(output_root / "heat_selection_threshold_sweep.csv", index=False)

    baseline_threshold = float(config["district_heating"]["minimum_heat_density_mwh_ha"])
    has_comparison = (
        ("demand_ranked", baseline_threshold) in selected_sets
        and ("route_aware", baseline_threshold) in selected_sets
    )
    if not has_comparison:
        print(frame.to_string(index=False))
        return
    demand = frame[(frame.selection_policy == "demand_ranked") & (frame.minimum_heat_density_mwh_ha == baseline_threshold)].iloc[0]
    route = frame[(frame.selection_policy == "route_aware") & (frame.minimum_heat_density_mwh_ha == baseline_threshold)].iloc[0]
    overlap = selected_sets[("demand_ranked", baseline_threshold)] & selected_sets[("route_aware", baseline_threshold)]
    comparison = pd.DataFrame([
        {
            "metric": "selected_buildings_common_to_both",
            "demand_ranked": len(overlap),
            "route_aware": len(overlap),
            "relative_change_percent": 0.0,
        },
        {
            "metric": "selected_building_overlap_percent",
            "demand_ranked": 100.0 * len(overlap) / max(len(selected_sets[("demand_ranked", baseline_threshold)]), 1),
            "route_aware": 100.0 * len(overlap) / max(len(selected_sets[("route_aware", baseline_threshold)]), 1),
            "relative_change_percent": 0.0,
        },
    ])
    metrics = [
        "route_trench_km", "service_connection_km", "one_way_network_km",
        "inventory_length_relative_error", "paired_pipe_km",
        "annual_heat_loss_mwh", "annual_heat_loss_fraction",
        "selected_unscaled_heat_mwh", "linear_heat_density_mwh_per_route_km",
        "catalogue_max_velocity_m_s", "catalogue_max_pressure_gradient_pa_m",
    ]
    extra = []
    for metric in metrics:
        d = float(demand[metric]); r = float(route[metric])
        extra.append({
            "metric": metric,
            "demand_ranked": d,
            "route_aware": r,
            "relative_change_percent": (r - d) / d * 100.0 if d else 0.0,
        })
    comparison = pd.concat([comparison, pd.DataFrame(extra)], ignore_index=True)
    comparison.to_csv(output_root / "heat_selection_policy_comparison.csv", index=False)
    manifest = {
        "purpose": "district-heating candidate sweep under fixed public contracts, heat sales, and a 52-km one-way network-length calibration target",
        "canonical_threshold_mwh_ha": baseline_threshold,
        "policies": {
            "demand_ranked": "largest annual heat candidates within suitable territories",
            "route_aware": "annual heat prize divided by newly added corridor plus service length, with quotas allocated across suitable cells",
        },
        "native_solver_limitation": "pandapipes is not installed in this execution environment; topology, catalogue velocity/gradient, and heat-loss diagnostics were regenerated, while the retained coupled case remains the native-solver baseline until the revised candidate is rerun in the released environment",
        "files": [
            "heat_selection_threshold_sweep.csv",
            "heat_selection_policy_comparison.csv",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(frame.to_string(index=False))
    print("\nComparison at", baseline_threshold)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
