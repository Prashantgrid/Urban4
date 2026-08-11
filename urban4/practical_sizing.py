#!/usr/bin/env python3
"""Practical design envelopes and equipment-size estimates for Urban4.

This module does not claim recovery of installed utility equipment.  It applies
predeclared engineering screening envelopes to the accepted synthetic baseline
and reports three distinct quantities wherever possible:

1. the native-model operating duty;
2. a practical design duty derived from flow/head/path constraints; and
3. the next discrete screening nameplate size.

A practical-screen failure never replaces or relaxes a native solver criterion.
It requests topology, pressure-zone, pump-cycle, feeder, or equipment redesign
and a subsequent native/coupled rerun.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

RHO_WATER = 998.0
G = 9.81
RHO_HEAT = 985.0
CP_WATER = 4180.0

STANDARD_MOTOR_KW = (
    0.75, 0.90, 1.10, 1.30, 1.50, 2.20, 3.00, 4.00, 5.50, 7.50,
    11.0, 15.0, 18.5, 22.0, 30.0, 37.0, 45.0, 55.0, 75.0, 90.0,
    110.0, 132.0, 160.0, 200.0, 250.0, 315.0, 355.0, 400.0, 500.0,
)
STANDARD_THERMAL_SOURCE_MW = (
    1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0,
    20.0, 25.0, 30.0,
)
STANDARD_GRID_CONNECTION_MVA = (40.0, 50.0, 63.0, 80.0, 100.0)
PRACTICAL_TRANSFORMER_MVA = (
    0.25, 0.40, 0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00,
)


@dataclass(frozen=True)
class PumpEstimate:
    design_flow_lps: float
    design_head_m: float
    electrical_duty_kw: float
    selected_motor_kw: float
    installed_units: int
    duty_units_at_peak: int
    installed_nameplate_kw: float
    firm_nameplate_kw: float


def next_size(value: float, catalogue: Sequence[float]) -> float:
    """Return the first catalogue size not smaller than ``value``."""
    for item in catalogue:
        if item + 1e-12 >= value:
            return float(item)
    return float(catalogue[-1])


def pump_power_kw(
    flow_lps: float,
    head_m: float,
    *,
    hydraulic_efficiency: float,
    motor_efficiency: float,
    density_kg_m3: float = RHO_WATER,
) -> float:
    if flow_lps < 0 or head_m < 0:
        raise ValueError("flow and head must be non-negative")
    if not (0 < hydraulic_efficiency <= 1 and 0 < motor_efficiency <= 1):
        raise ValueError("efficiencies must be in (0, 1]")
    q_m3s = flow_lps / 1000.0
    return density_kg_m3 * G * q_m3s * head_m / (
        hydraulic_efficiency * motor_efficiency
    ) / 1000.0


def water_station_estimate(
    *,
    normal_flow_lps: float,
    peak_flow_lps: float,
    head_m: float,
    hydraulic_efficiency: float = 0.72,
    motor_efficiency: float = 0.94,
    design_margin: float = 1.10,
) -> PumpEstimate:
    """Screen a three-unit water station (two duty plus one standby).

    Each identical unit carries the normal flow.  Two units therefore cover the
    two-times-peak case while the third provides the declared standby unit.
    """
    if peak_flow_lps > 2.0 * normal_flow_lps + 1e-9:
        raise ValueError("three-unit screening assumes peak <= 2 x normal flow")
    duty = pump_power_kw(
        normal_flow_lps,
        head_m,
        hydraulic_efficiency=hydraulic_efficiency,
        motor_efficiency=motor_efficiency,
    )
    motor = next_size(duty * design_margin, STANDARD_MOTOR_KW)
    return PumpEstimate(
        design_flow_lps=normal_flow_lps,
        design_head_m=head_m,
        electrical_duty_kw=duty,
        selected_motor_kw=motor,
        installed_units=3,
        duty_units_at_peak=2,
        installed_nameplate_kw=3.0 * motor,
        firm_nameplate_kw=2.0 * motor,
    )


def sewage_lift_estimate(
    *,
    existing_static_head_m: float,
    force_main_dn_mm: float = 100.0,
    target_velocity_m_s: float = 0.80,
    minimum_total_dynamic_head_m: float = 5.0,
    hydraulic_efficiency: float = 0.55,
    motor_efficiency: float = 0.90,
    minimum_raw_sewage_motor_kw: float = 1.30,
) -> PumpEstimate:
    """Estimate an intermittent raw-sewage lift duty.

    The design flow is at least that required to maintain the target cleansing
    velocity in a DN100 force main.  Two identical units provide duty/standby
    firm capacity.  The result is a screening estimate; wet-well cycles and
    manufacturer curves remain required before final selection.
    """
    diameter_m = force_main_dn_mm / 1000.0
    q_m3s = target_velocity_m_s * math.pi * diameter_m**2 / 4.0
    q_lps = q_m3s * 1000.0
    head_m = max(float(existing_static_head_m) + 2.0, minimum_total_dynamic_head_m)
    duty = pump_power_kw(
        q_lps,
        head_m,
        hydraulic_efficiency=hydraulic_efficiency,
        motor_efficiency=motor_efficiency,
    )
    motor = next_size(max(duty * 1.15, minimum_raw_sewage_motor_kw), STANDARD_MOTOR_KW)
    return PumpEstimate(
        design_flow_lps=q_lps,
        design_head_m=head_m,
        electrical_duty_kw=duty,
        selected_motor_kw=motor,
        installed_units=2,
        duty_units_at_peak=1,
        installed_nameplate_kw=2.0 * motor,
        firm_nameplate_kw=motor,
    )


def heat_circulation_estimate(
    *,
    peak_mw_th: float,
    critical_one_way_route_km: float,
    delta_t_k: float = 35.0,
    specific_pressure_loss_pa_m: float = 100.0,
    local_loss_factor: float = 1.30,
    terminal_differential_pressure_bar: float = 0.50,
    maximum_pump_lift_bar: float = 10.0,
    hydraulic_efficiency: float = 0.78,
    motor_efficiency: float = 0.94,
    nameplate_margin: float = 1.15,
) -> dict[str, float]:
    """Estimate a source-catchment circulation pump from its critical path."""
    mass_flow_kg_s = peak_mw_th * 1e6 / (CP_WATER * delta_t_k)
    volume_flow_m3s = mass_flow_kg_s / RHO_HEAT
    pair_length_m = 2.0 * critical_one_way_route_km * 1000.0
    network_dp_bar = (
        pair_length_m * specific_pressure_loss_pa_m * local_loss_factor / 1e5
    )
    lift_bar = min(maximum_pump_lift_bar, terminal_differential_pressure_bar + network_dp_bar)
    head_m = lift_bar * 1e5 / (RHO_HEAT * G)
    duty_kw = pump_power_kw(
        volume_flow_m3s * 1000.0,
        head_m,
        hydraulic_efficiency=hydraulic_efficiency,
        motor_efficiency=motor_efficiency,
        density_kg_m3=RHO_HEAT,
    )
    motor_kw = next_size(duty_kw * nameplate_margin, STANDARD_MOTOR_KW)
    return {
        "mass_flow_kg_s": mass_flow_kg_s,
        "volume_flow_m3h": volume_flow_m3s * 3600.0,
        "critical_one_way_route_km": critical_one_way_route_km,
        "specific_pressure_loss_pa_m": specific_pressure_loss_pa_m,
        "estimated_pump_lift_bar": lift_bar,
        "estimated_pump_head_m": head_m,
        "estimated_electrical_duty_kw": duty_kw,
        "selected_motor_kw": motor_kw,
    }


def heat_source_capacity_class(assigned_peak_mw_th: float, utilization: float = 0.80) -> float:
    """Estimate a thermal boundary capacity class without assigning technology."""
    return next_size(assigned_peak_mw_th / utilization, STANDARD_THERMAL_SOURCE_MW)


def grid_connection_class_mva(
    peak_mw: float,
    *,
    power_factor: float = 0.96,
    planning_loading: float = 0.80,
) -> tuple[float, float, float]:
    apparent_peak = peak_mw / power_factor
    design_minimum = apparent_peak / planning_loading
    selected = next_size(design_minimum, STANDARD_GRID_CONNECTION_MVA)
    return apparent_peak, design_minimum, selected


def _critical_heat_paths(
    corridors: pd.DataFrame,
    services: pd.DataFrame,
    sources: pd.DataFrame,
) -> dict[str, float]:
    graph = nx.Graph()
    route = corridors[corridors["link_type"].eq("route")]
    for row in route.itertuples(index=False):
        graph.add_edge(row.from_node, row.to_node, length_km=float(row.length_km))
    result: dict[str, float] = {}
    for source in sources.itertuples(index=False):
        terminals = services.loc[
            services["heat_source_node"].eq(source.source_node), "route_node"
        ].dropna().unique()
        distances: list[float] = []
        for terminal in terminals:
            try:
                distances.append(
                    float(nx.shortest_path_length(
                        graph, source.source_node, terminal, weight="length_km"
                    ))
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        result[str(source.map_id)] = max(distances) if distances else 0.0
    return result


def practical_envelope_rows() -> list[dict[str, str]]:
    """Return the declared preferred and hard screening envelopes."""
    return [
        {
            "sector": "Electricity",
            "preferred_optimization_band": (
                "0.95-1.05 p.u.; 20-80% line/transformer loading; <=5% cumulative LV drop; "
                "<=4 LV and <=3 MV parallel circuits"
            ),
            "hard_acceptance_or_trigger": (
                "0.90-1.10 p.u.; loading <=100%; a parallel-circuit excess triggers feeder/zone split"
            ),
            "discrete_equipment_rule": (
                "LV 0.6/1-kV Al-equivalent 16-240 mm2; MV 12/20(24)-kV Al-equivalent 95-400 mm2; "
                "transformers 0.25-4.0 MVA in practical screening"
            ),
            "evidence_role": "VDE supply-voltage band plus declared Urban4 planning margins",
        },
        {
            "sector": "Drinking water",
            "preferred_optimization_band": (
                "35-48 m normal pressure; 0.3-1.5 m/s preferred main velocity"
            ),
            "hard_acceptance_or_trigger": (
                "27.5-70 m normal pressure; >=14 m emergency/fire; velocity <=2 m/s; "
                "pressure-zone/PRV repair outside band"
            ),
            "discrete_equipment_rule": (
                "mains DN80-DN600; services DN25-DN80; firm pump capacity with largest unit unavailable"
            ),
            "evidence_role": "public drinking-water design guidance plus declared Urban4 velocity screen",
        },
        {
            "sector": "Wastewater",
            "preferred_optimization_band": (
                "force-main velocity 0.6-1.1 m/s; target 0.8 m/s; gravity cover 1.5-8 m"
            ),
            "hard_acceptance_or_trigger": (
                "force-main velocity <=3 m/s; raw-sewage force main >=DN100 unless justified; "
                "firm capacity at peak instantaneous flow"
            ),
            "discrete_equipment_rule": (
                "gravity mains DN200-DN2000; practical lift selection uses wet-well/on-state duty and duty/standby pumps"
            ),
            "evidence_role": "public sewage-pumping guidance; current average-duty lift proxies are not nameplates",
        },
        {
            "sector": "District heating",
            "preferred_optimization_band": (
                "0.5-1.5 m/s; 45-100 Pa/m; 0.3-0.5 bar terminal differential pressure; PN16 preferred"
            ),
            "hard_acceptance_or_trigger": (
                "velocity <=2 m/s; pressure gradient <=150 Pa/m; pump lift <=10 bar; PN25 hard screen"
            ),
            "discrete_equipment_rule": (
                "paired DN25-DN800; path-based pump duty; next standard motor with 15% margin; "
                "thermal source class at <=80% assigned peak"
            ),
            "evidence_role": "district-heating design literature plus declared Urban4 hard screens",
        },
    ]


def export_practical_design_outputs(code_root: Path, output_dir: Path) -> dict[str, object]:
    """Audit the packaged full-city baseline and export practical estimates."""
    integrated = code_root / "outputs" / "integrated_service_resolved"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((code_root / "cases" / "schweinfurt.json").read_text())
    manifest = json.loads((integrated / "integrated_generation_manifest.json").read_text())

    electricity_links = pd.read_csv(integrated / "electricity_links.csv")
    transformers = pd.read_csv(integrated / "electricity_transformers.csv")
    water_links = pd.read_csv(integrated / "drinking_water_links.csv")
    sewer_links = pd.read_csv(integrated / "wastewater_links.csv")
    heat_corridors = pd.read_csv(integrated / "district_heating_corridors.csv")
    heat_services = pd.read_csv(integrated / "district_heating_service_connections.csv")
    heat_sources = pd.read_csv(integrated / "district_heating_sources.csv")
    published_heat = pd.read_csv(
        code_root / "outputs" / "publication_ready_v1.5.3" / "heat_source_ratings.csv"
    )

    # Electricity audit.
    el = electricity_links.copy()
    el["practical_parallel_limit"] = np.where(el["voltage_level"].eq("LV"), 4, 3)
    el["parallel_limit_pass"] = el["parallel_circuits"] <= el["practical_parallel_limit"]
    el["practical_action"] = np.where(
        el["parallel_limit_pass"], "retain", "split feeder/service zone and rerun"
    )
    cable_summary = (
        el.groupby(["voltage_level", "cable_size_mm2", "parallel_circuits"], as_index=False)
        .agg(
            section_count=("line_id", "size"),
            route_length_km=("length_km", "sum"),
            model_ampacity_a=("max_i_ka", lambda values: float(values.iloc[0]) * 1000.0),
        )
    )
    cable_summary["screening_voltage_class"] = np.where(
        cable_summary["voltage_level"].eq("LV"), "0.6/1 (1.2) kV", "12/20(24) kV"
    )
    cable_summary["screening_conductor_description"] = (
        "aluminium-equivalent single-core planning model; final ampacity requires installation derating"
    )
    cable_summary.to_csv(output_dir / "electrical_cable_transformer_audit.csv", index=False)
    el.loc[~el["parallel_limit_pass"], [
        "line_id", "voltage_level", "feeder_id", "length_km", "cable_size_mm2",
        "parallel_circuits", "practical_parallel_limit", "design_power_mw",
        "design_current_ka", "practical_action",
    ]].to_csv(output_dir / "electrical_parallel_circuit_violations.csv", index=False)

    peak_mw = float(config["official_anchors"]["electricity_lv_peak_mw"])
    apparent_mva, design_mva, connection_class_mva = grid_connection_class_mva(peak_mw)
    trafo_util = transformers["assigned_peak_mw"] / transformers["sn_mva"]
    transformer_screen = transformers[[
        "transformer_id", "unit_count", "unit_rating_mva", "sn_mva",
        "assigned_peak_mw", "customer_count", "maximum_customer_route_km"
    ]].copy()
    transformer_screen["assigned_apparent_peak_mva"] = (
        transformer_screen["assigned_peak_mw"] / 0.96
    )
    transformer_screen["minimum_at_80_percent_mva"] = (
        transformer_screen["assigned_apparent_peak_mva"] / 0.80
    )
    transformer_screen["practical_catalogue_class_mva"] = [
        next_size(value, PRACTICAL_TRANSFORMER_MVA)
        for value in transformer_screen["minimum_at_80_percent_mva"]
    ]
    transformer_screen["model_utilization_fraction"] = trafo_util.to_numpy(float)
    transformer_screen["screening_status"] = np.where(
        transformer_screen["model_utilization_fraction"].between(0.20, 0.80),
        "inside preferred 20-80% loading band",
        np.where(
            transformer_screen["model_utilization_fraction"].le(1.0),
            "inside hard limit but outside preferred band",
            "hard loading limit exceeded",
        ),
    )
    transformer_screen.to_csv(
        output_dir / "electricity_transformer_site_sizing.csv", index=False
    )

    # Water station and practical pressure audit.
    water_native = manifest["native_solver_results"]["drinking_water"]
    normal_flow = float(config["official_anchors"]["drinking_water_annual_m3"]) / (365.0 * 86400.0) * 1000.0
    peak_flow = 2.0 * normal_flow
    water_station = water_station_estimate(
        normal_flow_lps=normal_flow,
        peak_flow_lps=peak_flow,
        head_m=float(water_native["source_head_rise_m"]),
    )
    water_practical_pass = (
        float(water_native["minimum_pressure_m"]) >= 27.5
        and float(water_native["maximum_pressure_m"]) <= 70.0
        and float(water_native["peak_minimum_pressure_m"]) >= 27.5
    )

    # Wastewater practical station estimates.
    network_force = sewer_links[sewer_links["link_type"].eq("force_main")].copy()
    network_force["map_id"] = [f"S{i+2}" for i in range(len(network_force))]
    wastewater_rows: list[dict[str, object]] = []
    for row in network_force.itertuples(index=False):
        estimate = sewage_lift_estimate(existing_static_head_m=float(row.pump_head_m))
        wastewater_rows.append({
            "map_id": row.map_id,
            "model_average_flow_lps": float(row.pump_design_flow_m3s) * 1000.0,
            "model_static_head_m": float(row.pump_head_m),
            "practical_force_main_dn_mm": 100,
            "practical_on_state_flow_lps": estimate.design_flow_lps,
            "practical_total_dynamic_head_m": estimate.design_head_m,
            "estimated_on_state_duty_kw": estimate.electrical_duty_kw,
            "selected_motor_kw_per_unit": estimate.selected_motor_kw,
            "arrangement": "1 duty + 1 standby",
            "installed_nameplate_kw": estimate.installed_nameplate_kw,
            "status": "screening estimate; wet-well cycle and pump curve required",
        })
    wastewater_df = pd.DataFrame(wastewater_rows)
    wastewater_df.to_csv(output_dir / "wastewater_lift_practical_sizing.csv", index=False)

    # Heat path-based practical sizing.
    critical_paths = _critical_heat_paths(heat_corridors, heat_services, heat_sources)
    heat_rows: list[dict[str, object]] = []
    published_by_id = published_heat.set_index("map_id")
    for row in heat_sources.itertuples(index=False):
        design = heat_circulation_estimate(
            peak_mw_th=float(row.assigned_peak_mw_th),
            critical_one_way_route_km=float(critical_paths[str(row.map_id)]),
        )
        low = heat_circulation_estimate(
            peak_mw_th=float(row.assigned_peak_mw_th),
            critical_one_way_route_km=float(critical_paths[str(row.map_id)]),
            specific_pressure_loss_pa_m=45.0,
            local_loss_factor=1.20,
            terminal_differential_pressure_bar=0.30,
        )
        high = heat_circulation_estimate(
            peak_mw_th=float(row.assigned_peak_mw_th),
            critical_one_way_route_km=float(critical_paths[str(row.map_id)]),
            specific_pressure_loss_pa_m=150.0,
            local_loss_factor=1.50,
            terminal_differential_pressure_bar=0.50,
        )
        existing = published_by_id.loc[str(row.map_id)]
        heat_rows.append({
            "map_id": str(row.map_id),
            "assigned_peak_mw_th": float(row.assigned_peak_mw_th),
            "planning_ceiling_mw_th": float(row.planning_capacity_mw_th),
            "estimated_thermal_capacity_class_mw_th": heat_source_capacity_class(
                float(row.assigned_peak_mw_th)
            ),
            "source_technology_input_power": "not inferred; requires technology/COP/efficiency",
            "critical_one_way_route_km": design["critical_one_way_route_km"],
            "volume_flow_m3h": design["volume_flow_m3h"],
            "model_closed_circulation_duty_kw": float(existing["closed_pump_power_kw"]),
            "practical_lower_lift_bar": low["estimated_pump_lift_bar"],
            "practical_design_lift_bar": design["estimated_pump_lift_bar"],
            "practical_upper_lift_bar": high["estimated_pump_lift_bar"],
            "practical_lower_duty_kw": low["estimated_electrical_duty_kw"],
            "practical_design_duty_kw": design["estimated_electrical_duty_kw"],
            "practical_upper_duty_kw": high["estimated_electrical_duty_kw"],
            "selected_motor_kw": design["selected_motor_kw"],
            "status": "path-based screening estimate; pandapipes/coupled rerun required",
        })
    heat_df = pd.DataFrame(heat_rows)
    heat_df.to_csv(output_dir / "heat_source_practical_sizing.csv", index=False)

    # Pipe/cable selected-size summary.
    summaries: list[pd.DataFrame] = []
    for sector, frame, kind_col, size_col in [
        ("electricity", el, "voltage_level", "cable_size_mm2"),
        ("drinking_water", water_links, "link_type", "diameter_mm"),
        ("wastewater", sewer_links, "link_type", "diameter_mm"),
        ("district_heating", heat_corridors, "link_type", "diameter_mm"),
    ]:
        temp = (
            frame.groupby([kind_col, size_col], as_index=False)
            .agg(element_count=(size_col, "size"), route_length_km=("length_km", "sum"))
            .rename(columns={kind_col: "asset_class", size_col: "nominal_size"})
        )
        temp.insert(0, "sector", sector)
        summaries.append(temp)
    pd.concat(summaries, ignore_index=True).to_csv(
        output_dir / "pipe_and_cable_selection_summary.csv", index=False
    )

    # Compact facility and practical-screen table.
    treatment_model_kw = 209.3
    treatment_connection_kw = next_size(treatment_model_kw * 1.15, STANDARD_MOTOR_KW)
    property_lift_motor_kw = 1.50
    heat_model_total_kw = float(heat_df["model_closed_circulation_duty_kw"].sum())
    heat_lower_total_kw = float(heat_df["practical_lower_duty_kw"].sum())
    heat_design_total_kw = float(heat_df["practical_design_duty_kw"].sum())
    heat_upper_total_kw = float(heat_df["practical_upper_duty_kw"].sum())
    heat_motor_total_kw = float(heat_df["selected_motor_kw"].sum())
    heat_source_total_mw_th = float(heat_df["estimated_thermal_capacity_class_mw_th"].sum())

    water_power_lower_kw = pump_power_kw(
        normal_flow, float(water_native["source_head_rise_m"]),
        hydraulic_efficiency=0.85, motor_efficiency=0.97,
    )
    water_power_upper_kw = pump_power_kw(
        normal_flow, float(water_native["source_head_rise_m"]),
        hydraulic_efficiency=0.65, motor_efficiency=0.93,
    )
    sewage_low = sewage_lift_estimate(
        existing_static_head_m=3.55,
        target_velocity_m_s=0.60,
        hydraulic_efficiency=0.65,
        motor_efficiency=0.94,
    )
    sewage_high = sewage_lift_estimate(
        existing_static_head_m=5.0,
        target_velocity_m_s=1.10,
        minimum_total_dynamic_head_m=7.0,
        hydraulic_efficiency=0.45,
        motor_efficiency=0.88,
    )

    pump_detail_rows = [
        {
            "id": "W1", "facility": "drinking-water station",
            "flow_lower_lps": normal_flow, "flow_design_lps": normal_flow,
            "flow_upper_lps": peak_flow, "head_lower_m": float(water_native["source_head_rise_m"]),
            "head_design_m": float(water_native["source_head_rise_m"]),
            "head_upper_m": float(water_native["source_head_rise_m"]),
            "electrical_duty_lower_kw": water_power_lower_kw,
            "electrical_duty_design_kw": water_station.electrical_duty_kw,
            "electrical_duty_upper_kw": water_power_upper_kw,
            "selected_motor_kw": water_station.selected_motor_kw,
            "unit_arrangement": "3 identical: 1 duty normal / 2 duty peak / 1 standby",
            "status": "screening nameplate; pump curves, NPSH and storage required",
        },
        {
            "id": "S1", "facility": "wastewater treatment process",
            "flow_lower_lps": float(config["official_anchors"]["wastewater_dry_weather_m3_day"]) / 86.4,
            "flow_design_lps": float(config["official_anchors"]["wastewater_dry_weather_m3_day"]) / 86.4,
            "flow_upper_lps": float(config["official_anchors"]["wastewater_dry_weather_m3_day"]) / 86.4 * 1.5,
            "head_lower_m": np.nan, "head_design_m": np.nan, "head_upper_m": np.nan,
            "electrical_duty_lower_kw": treatment_model_kw,
            "electrical_duty_design_kw": treatment_model_kw,
            "electrical_duty_upper_kw": treatment_connection_kw,
            "selected_motor_kw": np.nan,
            "unit_arrangement": "aggregate process connection; not one pump",
            "status": "250-kW connection-class estimate",
        },
        {
            "id": "S2-S3", "facility": "network sewage lift (each)",
            "flow_lower_lps": sewage_low.design_flow_lps,
            "flow_design_lps": sewage_lift_estimate(existing_static_head_m=3.55).design_flow_lps,
            "flow_upper_lps": sewage_high.design_flow_lps,
            "head_lower_m": sewage_low.design_head_m,
            "head_design_m": sewage_lift_estimate(existing_static_head_m=3.55).design_head_m,
            "head_upper_m": sewage_high.design_head_m,
            "electrical_duty_lower_kw": sewage_low.electrical_duty_kw,
            "electrical_duty_design_kw": sewage_lift_estimate(existing_static_head_m=3.55).electrical_duty_kw,
            "electrical_duty_upper_kw": sewage_high.electrical_duty_kw,
            "selected_motor_kw": 1.50,
            "unit_arrangement": "1 duty + 1 standby",
            "status": "wet-well cycle and manufacturer curve required",
        },
        {
            "id": "S-property", "facility": "property lift (each, screening)",
            "flow_lower_lps": np.nan, "flow_design_lps": np.nan, "flow_upper_lps": np.nan,
            "head_lower_m": np.nan, "head_design_m": np.nan, "head_upper_m": np.nan,
            "electrical_duty_lower_kw": np.nan, "electrical_duty_design_kw": np.nan,
            "electrical_duty_upper_kw": np.nan, "selected_motor_kw": 1.50,
            "unit_arrangement": "single small lift; standby not inferred",
            "status": "screening floor only; building-specific head/volume required",
        },
    ]
    pd.DataFrame(pump_detail_rows).to_csv(
        output_dir / "water_wastewater_pump_practical_sizing.csv", index=False
    )

    facility_rows = [
        {
            "id_or_sector": "E1 / electricity",
            "model_operating_or_peak": f"{peak_mw:.3f} MW coincident peak; {apparent_mva:.2f} MVA at pf=0.96",
            "practical_lower_upper_or_rule": "0.95-1.05 p.u. preferred; 0.90-1.10 hard; 20-80% preferred loading; <=100% hard",
            "estimated_practical_size": f"minimum planning connection {design_mva:.2f} MVA; next class {connection_class_mva:.0f} MVA",
            "installed_or_selected_arrangement": "external-grid boundary; no local generator inferred",
            "practical_screen_status": (
                f"{int((~el['parallel_limit_pass']).sum())} LV sections exceed four circuits and require feeder split"
            ),
        },
        {
            "id_or_sector": "W1 / drinking water",
            "model_operating_or_peak": f"{normal_flow:.2f} L/s, {water_native['source_head_rise_m']:.2f} m; closed duty 157.2 kW",
            "practical_lower_upper_or_rule": "35-48 m preferred; 27.5-70 m normal hard; >=14 m emergency; firm capacity with largest unit unavailable",
            "estimated_practical_size": (
                f"{water_power_lower_kw:.1f}-{water_power_upper_kw:.1f} kW efficiency range; "
                f"{water_station.electrical_duty_kw:.1f} kW design; {water_station.selected_motor_kw:.0f}-kW motor"
            ),
            "installed_or_selected_arrangement": (
                f"3 x {water_station.selected_motor_kw:.0f} kW (2 duty at peak + 1 standby); "
                f"{water_station.installed_nameplate_kw:.0f} kW installed"
            ),
            "practical_screen_status": (
                "pressure-zone/PRV redesign required" if not water_practical_pass else "passes practical pressure screen"
            ),
        },
        {
            "id_or_sector": "S1-S3 / wastewater",
            "model_operating_or_peak": "S1 209.3-kW process duty; S2-S3 continuous average proxies near 0.01 kW each",
            "practical_lower_upper_or_rule": "DN100 minimum; 0.6-1.1 m/s cleansing range; <=3 m/s; peak instantaneous firm capacity",
            "estimated_practical_size": (
                f"S1 {treatment_connection_kw:.0f}-kW connection class; S2-S3 1.5-kW motors; property lifts 1.5 kW"
            ),
            "installed_or_selected_arrangement": "S2-S3 each 1 duty + 1 standby; property-lift standby not inferred",
            "practical_screen_status": "current average-duty lift proxies fail practical nameplate screen; wet-well rerun required",
        },
        {
            "id_or_sector": "H1-H14 / district heat",
            "model_operating_or_peak": (
                f"58.333 MWth assigned; {heat_model_total_kw:.2f} kW model circulation duty"
            ),
            "practical_lower_upper_or_rule": "45-100 Pa/m preferred; <=150 Pa/m hard; 0.5-1.5 m/s preferred; <=2 m/s; 0.3-0.5 bar terminal dp; PN16 preferred/PN25 hard",
            "estimated_practical_size": (
                f"{heat_lower_total_kw:.1f}-{heat_upper_total_kw:.1f} kW path-envelope; "
                f"{heat_design_total_kw:.1f} kW design; {heat_motor_total_kw:.1f} kW selected motors; "
                f"{heat_source_total_mw_th:.1f} MWth aggregate thermal capacity classes"
            ),
            "installed_or_selected_arrangement": "14 circulation motors; heat-production technology and input power not inferred",
            "practical_screen_status": "absolute pressure and path-based pump sizing require pandapipes/coupled rerun",
        },
    ]
    pd.DataFrame(facility_rows).to_csv(output_dir / "facility_practical_sizing.csv", index=False)
    pd.DataFrame(practical_envelope_rows()).to_csv(
        output_dir / "practical_design_envelopes.csv", index=False
    )

    practical_catalogue_rows = [
        {"sector": "Electricity", "asset": "LV cable", "voltage_or_pressure_class": "0.6/1 (1.2) kV", "ordered_sizes": "16,25,35,70,95,150,240 mm2", "practical_rule": "model ampacity 76-368 A; <=4 parallel circuits; final installation derating required"},
        {"sector": "Electricity", "asset": "MV cable", "voltage_or_pressure_class": "12/20(24) kV", "ordered_sizes": "95,150,185,240,300,400 mm2", "practical_rule": "model ampacity 250-560 A; <=3 parallel circuits; final installation derating required"},
        {"sector": "Electricity", "asset": "MV/LV transformer", "voltage_or_pressure_class": "20/0.4 kV", "ordered_sizes": ",".join(str(value) for value in PRACTICAL_TRANSFORMER_MVA) + " MVA", "practical_rule": "preferred site loading 20-80%; hard <=100%"},
        {"sector": "Drinking water", "asset": "main / service pipe", "voltage_or_pressure_class": "PN16 screening at W1/trunk; zone class after surge study", "ordered_sizes": "mains DN80-DN600; services DN25-DN80", "practical_rule": "35-48 m preferred pressure; 27.5-70 m hard normal; 0.3-1.5 m/s preferred"},
        {"sector": "Drinking water", "asset": "W1 pump motor", "voltage_or_pressure_class": "400-V motor screening", "ordered_sizes": ",".join(str(value) for value in STANDARD_MOTOR_KW) + " kW", "practical_rule": "three identical units; one duty normal, two duty peak, one standby"},
        {"sector": "Wastewater", "asset": "gravity / force-main pipe", "voltage_or_pressure_class": "PN10 force-main screening; verify surge", "ordered_sizes": "gravity DN200-DN2000; raw-sewage force main >=DN100", "practical_rule": "0.6-1.1 m/s force-main velocity; <=3 m/s hard"},
        {"sector": "Wastewater", "asset": "lift pump motor", "voltage_or_pressure_class": "400-V submersible motor screening", "ordered_sizes": ",".join(str(value) for value in STANDARD_MOTOR_KW) + " kW", "practical_rule": "size from wet-well on-state flow; one duty plus one standby"},
        {"sector": "District heating", "asset": "pre-insulated paired pipe", "voltage_or_pressure_class": "PN16 preferred; PN25 hard screen", "ordered_sizes": "DN25-DN800", "practical_rule": "0.5-1.5 m/s and 45-100 Pa/m preferred; <=2 m/s and <=150 Pa/m hard"},
        {"sector": "District heating", "asset": "circulation motor / source class", "voltage_or_pressure_class": "motor nameplate +15%; thermal class at <=80%", "ordered_sizes": "motors 0.75-500 kW; source classes 1-30 MWth", "practical_rule": "source input power remains undefined without technology/COP/efficiency"},
    ]
    pd.DataFrame(practical_catalogue_rows).to_csv(
        output_dir / "practical_equipment_catalogues.csv", index=False
    )

    practical_acceptance_rows = [
        {"sector": "Electricity", "screen": "preferred cable/feeder constructability", "result": "fail", "evidence": f"{int((~el['parallel_limit_pass']).sum())} LV sections exceed four circuits", "required_action": "split feeder/service zones and rerun pandapower/coupling"},
        {"sector": "Drinking water", "screen": "27.5-70 m normal pressure envelope", "result": "pass" if water_practical_pass else "fail", "evidence": f"average {water_native['minimum_pressure_m']:.2f}-{water_native['maximum_pressure_m']:.2f} m; peak minimum {water_native['peak_minimum_pressure_m']:.2f} m", "required_action": "pressure zones, PRVs/booster/storage and WNTR/coupled rerun"},
        {"sector": "Wastewater", "screen": "wet-well on-state pump sizing and force-main cleansing", "result": "fail", "evidence": "accepted baseline uses approximately 0.20 L/s continuous proxies for S2-S3", "required_action": "DN100/0.8 m/s on-state design, wet-well cycle, duty/standby pumps and SWMM/coupled rerun"},
        {"sector": "District heating", "screen": "path-based pressure loss, pump lift and PN class", "result": "fail", "evidence": f"model 98.63 kW versus {heat_design_total_kw:.1f} kW path-based design; reported 23.09-25.22 bar approaches the PN25 hard screen", "required_action": "retune source pressures and pumps, then rerun pandapipes/coupling"},
    ]
    pd.DataFrame(practical_acceptance_rows).to_csv(
        output_dir / "practical_acceptance_summary.csv", index=False
    )

    audit = {
        "version": "1.6.0",
        "interpretation": (
            "Practical screening estimates supplement the accepted numerical benchmark. "
            "They are not installed asset records and are not promoted to the coupled baseline "
            "until native and coupled reruns pass."
        ),
        "electricity": {
            "coincident_peak_mw": peak_mw,
            "apparent_peak_mva_at_pf_0_96": apparent_mva,
            "minimum_planning_connection_mva_at_80_percent": design_mva,
            "selected_grid_connection_class_mva": connection_class_mva,
            "generated_transformer_capacity_mva": float(transformers["sn_mva"].sum()),
            "public_installed_transformer_capacity_mva": float(
                config["official_anchors"]["electricity_mv_lv_installed_capacity_mva"]
            ),
            "transformer_utilization_min_median_max": [
                float((transformers["assigned_peak_mw"] / transformers["sn_mva"]).min()),
                float((transformers["assigned_peak_mw"] / transformers["sn_mva"]).median()),
                float((transformers["assigned_peak_mw"] / transformers["sn_mva"]).max()),
            ],
            "transformer_sites_outside_preferred_20_80_percent": int(
                (~transformer_screen["model_utilization_fraction"].between(0.20, 0.80)).sum()
            ),
            "lv_sections_exceeding_four_parallel_circuits": int(
                ((el["voltage_level"].eq("LV")) & (el["parallel_circuits"] > 4)).sum()
            ),
            "mv_sections_exceeding_three_parallel_circuits": int(
                ((el["voltage_level"].eq("MV")) & (el["parallel_circuits"] > 3)).sum()
            ),
        },
        "drinking_water": {
            "model_closed_duty_kw": 157.2,
            "screened_station": water_station.__dict__,
            "native_pressure_results_m": {
                "minimum_average": float(water_native["minimum_pressure_m"]),
                "maximum_average": float(water_native["maximum_pressure_m"]),
                "minimum_peak": float(water_native["peak_minimum_pressure_m"]),
            },
            "practical_pressure_screen_passed": bool(water_practical_pass),
        },
        "wastewater": {
            "treatment_process_model_duty_kw": treatment_model_kw,
            "treatment_connection_class_kw": treatment_connection_kw,
            "network_lift_count": int(len(wastewater_df)),
            "network_lift_selected_motor_kw_each": 1.50,
            "property_lift_count": 9,
            "property_lift_screening_motor_kw_each": property_lift_motor_kw,
            "status": "wet-well cycling and native/coupled rerun required",
        },
        "district_heating": {
            "model_circulation_duty_kw": heat_model_total_kw,
            "path_based_lower_duty_kw": heat_lower_total_kw,
            "path_based_design_duty_kw": heat_design_total_kw,
            "path_based_upper_duty_kw": heat_upper_total_kw,
            "selected_motor_nameplate_total_kw": heat_motor_total_kw,
            "estimated_thermal_capacity_class_total_mw_th": heat_source_total_mw_th,
            "source_production_input_power": "unknown without technology/COP/efficiency",
            "status": "path-based pressure retuning and native/coupled rerun required",
        },
    }
    (output_dir / "practical_sizing_manifest.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit
