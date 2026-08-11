#!/usr/bin/env python3
"""Executable cross-sector interface and shared-corridor contracts for Urban4."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import schweinfurt_base as base


def _normalise_geometry(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, str):
        value = json.loads(value)
    return [(float(point[0]), float(point[1])) for point in value]


def _segment_key(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    endpoints = (
        (round(first[0], 7), round(first[1], 7)),
        (round(second[0], 7), round(second[1], 7)),
    )
    return tuple(sorted(endpoints))  # type: ignore[return-value]


def write_shared_corridor_ledger(output: Path) -> pd.DataFrame:
    """Assign a stable identity to every reused physical route segment."""
    inputs = {
        "electricity": ("electricity_links.csv", "line_id"),
        "drinking_water": ("drinking_water_links.csv", "link_id"),
        "wastewater": ("wastewater_links.csv", "link_id"),
        "district_heating": ("district_heating_corridors.csv", "corridor_id"),
    }
    membership: dict[
        tuple[tuple[float, float], tuple[float, float]],
        dict[str, set[str]],
    ] = defaultdict(lambda: defaultdict(set))
    for sector, (filename, id_column) in inputs.items():
        frame = pd.read_csv(output / filename)
        for row in frame.itertuples(index=False):
            try:
                points = _normalise_geometry(getattr(row, "geometry_json"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            for first, second in zip(points[:-1], points[1:]):
                if first == second:
                    continue
                membership[_segment_key(first, second)][sector].add(
                    str(getattr(row, id_column))
                )
    rows = []
    for index, (segment, sectors) in enumerate(sorted(membership.items()), 1):
        first, second = segment
        names = sorted(sectors)
        rows.append({
            "corridor_id": f"U4C_{index:07d}",
            "from_lon": first[0], "from_lat": first[1],
            "to_lon": second[0], "to_lat": second[1],
            "length_m": 1000.0 * base.distance_km(first, second),
            "sector_count": len(names),
            "sectors": "|".join(names),
            "is_shared_corridor": len(names) >= 2,
            "electricity_link_ids": "|".join(sorted(sectors.get("electricity", set()))),
            "drinking_water_link_ids": "|".join(sorted(sectors.get("drinking_water", set()))),
            "wastewater_link_ids": "|".join(sorted(sectors.get("wastewater", set()))),
            "district_heating_link_ids": "|".join(sorted(sectors.get("district_heating", set()))),
            "evidence_role": "generated route colocation; not an installed common-trench claim",
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "shared_corridor_ledger.csv", index=False)
    summary = {
        "physical_route_segments": int(len(frame)),
        "shared_route_segments": int(frame.is_shared_corridor.sum()),
        "shared_route_length_km": float(
            frame.loc[frame.is_shared_corridor, "length_m"].sum() / 1000.0
        ),
        "interpretation": (
            "Exact reuse of a generated road/corridor segment by at least two sectors; "
            "this is a synthetic colocation identity, not observed trench evidence."
        ),
    }
    (output / "shared_corridor_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return frame


def _nearest_compatible_bus(
    point: tuple[float, float], power_nodes: pd.DataFrame, power_mw: float
) -> pd.Series:
    nodes = power_nodes.copy()
    if "voltage_kv" not in nodes and "vn_kv" in nodes:
        nodes["voltage_kv"] = nodes.vn_kv
    if power_mw >= 0.10:
        candidates = nodes[
            nodes.voltage_kv.ge(10.0)
            & nodes.node_type.isin(
                ["mv_source", "mv_junction", "transformer_mv_bus", "power_bus"]
            )
        ]
    elif power_mw >= 0.02:
        # Medium-size three-phase drives connect at the LV switchboard of a
        # transformer site, not at an arbitrary downstream service bus.
        candidates = nodes[
            nodes.voltage_kv.le(1.0)
            & nodes.node_type.eq("transformer_lv_bus")
        ]
    else:
        candidates = nodes[
            nodes.voltage_kv.le(1.0)
            & nodes.node_type.isin(
                ["transformer_lv_bus", "lv_junction", "building_service", "power_bus"]
            )
        ]
    if candidates.empty:
        candidates = nodes
    return min(
        candidates.itertuples(),
        key=lambda bus: base.distance_km(point, (float(bus.lon), float(bus.lat))),
    )


def _facility_interface(
    rows: list[dict[str, Any]],
    *,
    power_nodes: pd.DataFrame,
    sector: str,
    asset_id: str,
    point: tuple[float, float],
    nominal_power_mw: float,
    facility_type: str,
    duty_basis: str,
    reverse_response: str,
) -> None:
    bus = _nearest_compatible_bus(point, power_nodes, nominal_power_mw)
    rows.append({
        "building_id": "",
        "from_sector": "electricity", "from_id": str(bus.bus_id),
        "to_sector": sector, "to_id": asset_id,
        "relation": "electrically_driven_facility",
        "facility_type": facility_type,
        "exchange_variable": "active_power_and_terminal_voltage",
        "nominal_power_mw": float(nominal_power_mw),
        "conversion": np.nan, "unit": "MW, p.u.",
        "directionality": "bidirectional_iterative",
        "connected_bus_type": str(bus.node_type),
        "connected_bus_voltage_kv": float(bus.voltage_kv),
        "connection_distance_km": base.distance_km(
            point, (float(bus.lon), float(bus.lat))
        ),
        "forward_duty_model": duty_basis,
        "reverse_response_model": reverse_response,
        "evidence_role": "generated interface; not an observed utility connection",
    })


def write_coupling_interfaces(
    output: Path,
    power_nodes: pd.DataFrame,
    water_nodes: pd.DataFrame,
    water_links: pd.DataFrame,
    water_services: pd.DataFrame,
    water_solver: dict[str, Any],
    sewer_nodes: pd.DataFrame,
    sewer_links: pd.DataFrame,
    sewer_services: pd.DataFrame,
    heat_nodes: pd.DataFrame,
    heat_corridors: pd.DataFrame,
    heat_solver: dict[str, Any],
) -> pd.DataFrame:
    """Write building mass mappings and capacity-aware facility interfaces."""
    rows: list[dict[str, Any]] = []
    sewer_by_building = sewer_services.set_index("building_id")
    for water in water_services.itertuples():
        if water.building_id not in sewer_by_building.index:
            continue
        sewer = sewer_by_building.loc[water.building_id]
        rows.append({
            "building_id": water.building_id,
            "from_sector": "drinking_water", "from_id": water.service_node,
            "to_sector": "wastewater", "to_id": sewer.property_node,
            "relation": "delivered_water_to_sanitary_inflow",
            "facility_type": "building_mass_mapping",
            "exchange_variable": "volume_flow",
            "nominal_power_mw": np.nan,
            "conversion": 0.82, "unit": "m3/m3",
            "directionality": "directed_mass_transfer_inside_coupled_loop",
            "connected_bus_type": "", "connected_bus_voltage_kv": np.nan,
            "connection_distance_km": 0.0,
            "forward_duty_model": "Q_wastewater = 0.82 Q_delivered_water",
            "reverse_response_model": "wastewater state does not alter delivered-water mass at this interface",
            "evidence_role": "common building identity",
        })

    pump = water_links.loc[water_links.link_type.eq("pump")].iloc[0]
    pump_node = water_nodes.loc[
        water_nodes.node_id.eq(pump.to_node)
    ].iloc[0]
    water_power_mw = (
        998.0 * 9.81 * float(pump.design_flow_m3s)
        * float(water_solver["source_head_rise_m"]) / 0.78 / 1e6
    )
    _facility_interface(
        rows, power_nodes=power_nodes, sector="drinking_water",
        asset_id=str(pump.link_id), point=(pump_node.lon, pump_node.lat),
        nominal_power_mw=water_power_mw, facility_type="waterworks_head_pump",
        duty_basis="P = rho g Q H / (eta_pump eta_motor), eta_combined=0.78",
        reverse_response="terminal voltage -> VFD availability/speed -> pump head and flow",
    )

    force_links = sewer_links[
        sewer_links.link_type.isin(["force_main", "property_force_main"])
    ]
    sewer_coordinates = sewer_nodes.set_index("node_id")[["lon", "lat"]]
    for force in force_links.itertuples():
        location = sewer_coordinates.loc[force.from_node]
        power_mw = (
            998.0 * 9.81 * float(force.pump_design_flow_m3s)
            * float(force.pump_head_m) / 0.65 / 1e6
        )
        _facility_interface(
            rows, power_nodes=power_nodes, sector="wastewater",
            asset_id=f"P_{force.link_id}", point=(location.lon, location.lat),
            nominal_power_mw=max(power_mw, 1e-5),
            facility_type=(
                "property_lift_pump" if force.link_type == "property_force_main"
                else "network_lift_pump"
            ),
            duty_basis="P = rho g Q H / (eta_pump eta_motor), eta_combined=0.65",
            reverse_response="terminal voltage -> pump availability/speed -> wet-well outflow",
        )

    outfall = sewer_nodes[sewer_nodes.node_type.eq("outfall")].iloc[0]
    annual_m3 = float(sewer_services.annual_m3.sum())
    treatment_power_mw = annual_m3 * 0.35 / 8760.0 / 1000.0 * 1.5
    _facility_interface(
        rows, power_nodes=power_nodes, sector="wastewater",
        asset_id="WWTP_INTERFACE", point=(outfall.lon, outfall.lat),
        nominal_power_mw=treatment_power_mw,
        facility_type="wastewater_treatment_interface",
        duty_basis="0.35 kWh/m3 specific electricity and 1.5 operating peak factor",
        reverse_response="terminal voltage -> treatment availability; inflow remains a mass balance state",
    )

    pressure_lift_bar = max(
        1.0, 25.0 - float(heat_solver["minimum_pressure_bar"])
    )
    for plant in heat_nodes[heat_nodes.node_type.eq("plant")].itertuples():
        outgoing = heat_corridors[
            heat_corridors.from_node.eq(plant.node_id)
            & heat_corridors.link_type.eq("route")
        ]
        mass_flow = float(outgoing.mass_flow_kg_s.sum())
        volumetric_flow_m3s = mass_flow / 985.0
        pump_power_mw = (
            pressure_lift_bar * 1e5 * volumetric_flow_m3s / 0.75 / 1e6
        )
        _facility_interface(
            rows, power_nodes=power_nodes, sector="district_heating",
            asset_id=f"HPUMP_{plant.node_id}", point=(plant.lon, plant.lat),
            nominal_power_mw=max(pump_power_mw, 1e-4),
            facility_type="district_heat_circulation_pump",
            duty_basis="P = Delta_p Q / eta, eta_combined=0.75",
            reverse_response="terminal voltage -> circulation-pump speed -> mass flow, pressure and temperature delivery",
        )

    frame = pd.DataFrame(rows)
    frame.insert(
        0, "interface_id",
        [f"U4I_{index:06d}" for index in range(1, len(frame) + 1)],
    )
    frame.to_csv(output / "coupling_interfaces.csv", index=False)
    facility = frame[frame.relation.eq("electrically_driven_facility")]
    audit = {
        "building_mass_interfaces": int(
            frame.relation.eq("delivered_water_to_sanitary_inflow").sum()
        ),
        "electrically_driven_facilities": int(len(facility)),
        "facility_nominal_power_mw": float(facility.nominal_power_mw.sum()),
        "all_facilities_have_power_bus": bool(facility.from_id.notna().all()),
        "all_facilities_have_forward_and_reverse_models": bool(
            facility.forward_duty_model.str.len().gt(0).all()
            and facility.reverse_response_model.str.len().gt(0).all()
        ),
        "building_identity_match_percent": float(
            100.0 * frame.relation.eq("delivered_water_to_sanitary_inflow").sum()
            / max(len(water_services), 1)
        ),
    }
    (output / "coupling_interface_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return frame
