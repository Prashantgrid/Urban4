#!/usr/bin/env python3
"""District-heating suitability sensitivity for the integrated city ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import schweinfurt_base as base
from .framework import OUT, PROJECT, load_case
from .service_topologies import generate_district_heat


ROOT = OUT / "sensitivity" / "district_heat_suitability"


def run_heat_suitability_sensitivity() -> dict[str, object]:
    config = load_case()
    buildings = pd.read_csv(OUT / "building_sector_demands.csv")
    road = base.build_road_graph(base.load_elements("local_roads.json"))
    rows = []
    for threshold in config["district_heating"]["heat_suitability_sensitivity_mwh_ha"]:
        case_dir = ROOT / f"density_{int(threshold):04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        nodes, corridors, services, solver = generate_district_heat(
            case_dir, road, buildings,
            connection_count=int(config["district_heating"]["sales_contract_count"]),
            plant_capacity_mw=30.0,
            seed=int(config["seed"]),
            annual_heat_target_mwh=float(config["district_heating"]["heat_sales_mwh_year"]),
            design_peak_target_mw=float(config["district_heating"]["design_peak_mw_assumption"]),
            minimum_heat_density_mwh_ha=float(threshold),
            heat_density_cell_size_km=float(config["district_heating"]["heat_density_cell_size_km"]),
            enforce_connection_count=True,
            selection_policy=str(config["district_heating"].get("customer_selection_policy", "route_aware")),
            route_aware_demand_exponent=float(config["district_heating"].get("route_aware_demand_exponent", 1.0)),
            route_aware_spatial_quota_level=str(config["district_heating"].get("route_aware_spatial_quota_level", "cell")),
            return_pressure_reference_bar=float(
                config["bidirectional_coupling"]["heat_return_pressure_bar"]
            ),
        )
        rows.append({
            "minimum_heat_density_mwh_ha": float(threshold),
            "network_present": bool(solver["suitability"]["network_present"]),
            "suitable_cells": int(solver["suitability"].get("suitable_cell_count", 0)),
            "territories": int(solver["suitability"].get("territory_count", 0)),
            "consumer_substations": len(services),
            "plants": int(nodes.node_type.eq("plant").sum()),
            "trench_route_km": float(solver.get("route_trench_length_km", 0.0)),
            "service_connection_km": float(solver.get("service_connection_length_km", 0.0)),
            "annual_heat_loss_fraction": float(solver.get("annual_heat_loss_fraction", 0.0)),
            "circulation_pump_power_mw": float(solver.get("circulation_pump_power_mw", 0.0)),
            "minimum_differential_pressure_bar": float(solver.get("minimum_differential_pressure_bar", 0.0)),
            "maximum_velocity_m_s": float(solver.get("maximum_velocity_m_s", 0.0)),
            "solver_converged": bool(solver.get("converged", False)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "district_heat_suitability_sensitivity.csv", index=False)
    manifest = {
        "method": "rerun district-heating territory selection and native hydraulic screen while varying only the declared minimum annual heat density",
        "fixed_inputs": ["building ledger", "road graph", "contract-equivalent count", "annual heat sales", "equipment catalogue", "seed"],
        "thresholds_mwh_ha": config["district_heating"]["heat_suitability_sensitivity_mwh_ha"],
        "all_solver_runs_converged": bool(summary.solver_converged.all()),
    }
    (OUT / "district_heat_suitability_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(run_heat_suitability_sensitivity(), indent=2))
