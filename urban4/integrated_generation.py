#!/usr/bin/env python3
"""One-command InfDB/pylovo/Urban4 service-resolved generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import schweinfurt_base as base
from .framework import PROJECT, build_building_ledger, load_case, _screen_solver_results
from .infdb_adapter import (
    build_infdb_building_ledger,
    load_infdb_road_graph,
    resolve_evidence_backend,
    write_backend_status,
)
from .pylovo_bridge import _point, merge_pylovo_grids
from .integration_contract import (
    write_coupling_interfaces,
    write_shared_corridor_ledger,
)
from .service_topologies import (
    generate_district_heat,
    generate_electricity,
    generate_wastewater,
    generate_water,
)
from .service_coupling import run_nominal_interface_closure


OUTPUT = PROJECT / "outputs" / "integrated_service_resolved"


def _power_view_from_pylovo(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    buses = frames["buses"].copy()
    coordinates = [_point(value) for value in buses.geo]
    buses["bus_id"] = buses.bus_index.map(lambda value: f"PP_BUS_{int(value)}")
    buses["lon"] = [point[0] for point in coordinates]
    buses["lat"] = [point[1] for point in coordinates]
    buses["node_type"] = "power_bus"
    return buses[["bus_id", "bus_index", "name", "vn_kv", "lon", "lat", "node_type", "origin_grid_id"]]


def run_integrated_generation() -> dict[str, Any]:
    config = load_case()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status = resolve_evidence_backend(config, PROJECT)
    write_backend_status(status, OUTPUT)
    if status.selected == "infdb":
        export_dir = Path(status.infdb_export_dir)
        buildings = build_infdb_building_ledger(export_dir, config)
        road = load_infdb_road_graph(export_dir)
    else:
        buildings = build_building_ledger(config)
        road = base.build_road_graph(base.load_elements("local_roads.json"))
    buildings.to_csv(OUTPUT / "common_building_service_ledger.csv", index=False)

    pylovo_dir = Path(status.infdb_export_dir) / "pylovo_grids"
    pylovo_paths = sorted(pylovo_dir.glob("*.json")) if status.selected == "infdb" else []
    if pylovo_paths:
        _, power_frames, power_solver = merge_pylovo_grids(
            pylovo_paths, road, OUTPUT,
            coincident_peak_mw=float(config["official_anchors"]["electricity_lv_peak_mw"]),
        )
        power_nodes = _power_view_from_pylovo(power_frames)
        electricity_backend = "pylovo LV retained + Urban4 connected MV"
    else:
        power_nodes, _, _, _, power_solver = generate_electricity(
            OUTPUT, road, buildings, maximum_radius_km=0.70,
            seed=int(config["seed"]),
            target_site_count=(
                int(config["official_anchors"]["electricity_mv_lv_withdrawal_locations"])
                if len(buildings) > 10_000 else None
            ),
            verbose=True,
        )
        electricity_backend = "Urban4 building-service fallback; pylovo export unavailable"
    water_nodes, water_links, water_services, water_solver = generate_water(OUTPUT, road, buildings)
    sewer_nodes, sewer_links, sewer_services, sewer_solver = generate_wastewater(OUTPUT, road, buildings)
    heat_nodes, heat_corridors, heat_services, heat_solver = generate_district_heat(
        OUTPUT, road, buildings,
        connection_count=int(config["district_heating"]["sales_contract_count"]),
        plant_capacity_mw=float(config["district_heating"].get(
            "heat_source_planning_capacity_mw_th", 30.0
        )), seed=int(config["seed"]),
        annual_heat_target_mwh=float(config["district_heating"]["heat_sales_mwh_year"]),
        design_peak_target_mw=float(config["district_heating"]["design_peak_mw_assumption"]),
        minimum_heat_density_mwh_ha=float(
            config["district_heating"]["minimum_heat_density_mwh_ha"]
        ),
        heat_density_cell_size_km=float(
            config["district_heating"]["heat_density_cell_size_km"]
        ),
        enforce_connection_count=True,
        selection_policy=str(
            config["district_heating"].get("customer_selection_policy", "route_aware")
        ),
        route_aware_demand_exponent=float(
            config["district_heating"].get("route_aware_demand_exponent", 1.0)
        ),
        route_aware_spatial_quota_level=str(
            config["district_heating"].get("route_aware_spatial_quota_level", "cell")
        ),
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
    interfaces = write_coupling_interfaces(
        OUTPUT, power_nodes, water_nodes, water_links, water_services, water_solver,
        sewer_nodes, sewer_links, sewer_services,
        heat_nodes, heat_corridors, heat_solver,
    )
    shared_corridors = write_shared_corridor_ledger(OUTPUT)
    screening = {
        "electricity": _screen_solver_results(power_solver, "electricity", config),
        "drinking_water": _screen_solver_results(water_solver, "drinking_water", config),
        "wastewater": _screen_solver_results(sewer_solver, "wastewater", config),
        "district_heating": _screen_solver_results(heat_solver, "district_heating", config),
    }
    native_passed = all(value["passed"] for value in screening.values())
    coupled_closure = (
        run_nominal_interface_closure(OUTPUT) if native_passed else {
            "converged": False,
            "skipped": True,
            "reason": "one or more native sector screens failed",
        }
    )
    manifest = {
        "framework": "Urban4: Shared-Ledger Generation of Coupled Urban Infrastructure Networks",
        "evidence_backend": status.selected,
        "evidence_backend_reason": status.reason,
        "electricity_backend": electricity_backend,
        "resolution_contract": "one immutable building ID and explicit sector service point; zones are planning/reporting objects only",
        "counts": {
            "building_evidence_objects": len(buildings),
            "service_eligible_buildings": int(buildings.service_eligible.astype(bool).sum()),
            "water_service_connections": len(water_services),
            "wastewater_property_connections": len(sewer_services),
            "district_heat_consumer_substations": len(heat_services),
            "coupling_interfaces": len(interfaces),
            "electrically_driven_facilities": int(
                interfaces.relation.eq("electrically_driven_facility").sum()
            ),
            "shared_route_segments": int(shared_corridors.is_shared_corridor.sum()),
        },
        "native_solver_results": {
            "electricity": power_solver,
            "drinking_water": water_solver,
            "wastewater": sewer_solver,
            "district_heating": heat_solver,
        },
        "native_acceptance_screening": screening,
        "all_native_screens_passed": native_passed,
        "coupled_nominal_interface_closure": coupled_closure,
        "final_export_gate_passed": bool(
            native_passed and coupled_closure.get("accepted", False)
        ),
    }
    (OUTPUT / "integrated_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    print(json.dumps(run_integrated_generation(), indent=2))
