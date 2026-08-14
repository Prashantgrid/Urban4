#!/usr/bin/env python3
"""Quasi-steady bidirectional electricity--water--wastewater--heat co-simulation.

The topology generators and their native acceptance screens run before this
module.  Accepted native models are then staged, coupled through explicit
facility interfaces, and iterated to a fixed point.  Sector solvers return
electrical duties; pandapower returns motor-bus voltages; the voltages update
pump speed/head/flow through a declared drive law; and the sector models are
re-solved.  Only a converged, screened state is copied to ``final_accepted``.

This is a deliberately transparent normal-condition co-simulation adapter.  It
does not claim that the voltage--drive curve is a universal motor model or that
the synthetic networks reproduce confidential utility control systems.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from . import schweinfurt_base as base
from .coupling_repair import (
    declared_policy,
    next_catalogue_value,
    phase_return_for_failure,
    terminal_failure,
    validate_interface_state,
)


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
CASE = PROJECT / "cases" / "schweinfurt.json"
COSIM = OUT / "cosimulation"


LV_CABLE_CATALOGUE = [
    {"area_mm2": 35, "max_i_ka": 0.115, "r_ohm_per_km": 0.868},
    {"area_mm2": 70, "max_i_ka": 0.179, "r_ohm_per_km": 0.443},
    {"area_mm2": 95, "max_i_ka": 0.216, "r_ohm_per_km": 0.320},
    {"area_mm2": 150, "max_i_ka": 0.282, "r_ohm_per_km": 0.206},
    {"area_mm2": 240, "max_i_ka": 0.368, "r_ohm_per_km": 0.125},
]
MV_CABLE_CATALOGUE = [
    {"area_mm2": 95, "max_i_ka": 0.250, "r_ohm_per_km": 0.320},
    {"area_mm2": 150, "max_i_ka": 0.320, "r_ohm_per_km": 0.206},
    {"area_mm2": 185, "max_i_ka": 0.360, "r_ohm_per_km": 0.164},
    {"area_mm2": 240, "max_i_ka": 0.420, "r_ohm_per_km": 0.125},
    {"area_mm2": 300, "max_i_ka": 0.480, "r_ohm_per_km": 0.100},
    {"area_mm2": 400, "max_i_ka": 0.560, "r_ohm_per_km": 0.077},
]
TRANSFORMER_CATALOGUE_MVA = [0.25, 0.40, 0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50]


def _config() -> dict[str, Any]:
    return json.loads(CASE.read_text(encoding="utf-8"))


def _motor_speed(voltage_pu: float, command: float, drive: dict[str, float]) -> float:
    """Map terminal voltage to available pump speed.

    Above the recovery threshold the explicit command is applied, with only a
    configured droop term (zero in the v2.9 evidence-grounded case). Between
    trip and recovery the drive is linearly derated; below trip it is
    unavailable. Pump head and capacity are subsequently changed by the
    affinity laws in the native sector adapters.
    """
    if not math.isfinite(voltage_pu) or voltage_pu < drive["trip_voltage_pu"]:
        return 0.0
    if voltage_pu < drive["recovery_voltage_pu"]:
        span = drive["recovery_voltage_pu"] - drive["trip_voltage_pu"]
        fraction = (voltage_pu - drive["trip_voltage_pu"]) / max(span, 1e-9)
        return float(np.clip(command * drive["minimum_running_speed_pu"] * fraction, 0.0, drive["maximum_speed_pu"]))
    speed = command * (1.0 + drive["voltage_speed_droop"] * (voltage_pu - 1.0))
    return float(np.clip(speed, drive["minimum_running_speed_pu"], drive["maximum_speed_pu"]))


def _active_interface_state(interfaces: pd.DataFrame) -> pd.DataFrame:
    active = interfaces[
        (interfaces["from_sector"] == "electricity")
        & (interfaces["relation"] == "electric_supply")
        & interfaces["capacity_value"].notna()
        & (interfaces["capacity_value"] > 0)
    ].copy()
    active.rename(
        columns={"from_id": "bus_id", "to_sector": "sector", "to_id": "asset_id"},
        inplace=True,
    )
    active["nominal_power_mw"] = active["capacity_value"].astype(float)
    active["power_mw"] = active["nominal_power_mw"]
    active["voltage_pu"] = 1.0
    active["speed_pu"] = 1.0
    active["energized"] = True
    active["feedback_model"] = "constant_auxiliary"
    active.loc[active["asset_id"] == "WAT-WW-01", "feedback_model"] = "WNTR_HEAD_PUMP"
    active.loc[active["asset_id"].str.startswith("WAT-WF-"), "feedback_model"] = "AFFINITY_AUXILIARY_PUMP"
    active.loc[active["asset_id"].str.startswith("PU"), "feedback_model"] = "SWMM_PUMP4"
    active.loc[active["asset_id"] == "SEW-WWTP-01", "feedback_model"] = "FLOW_SENSITIVE_PROCESS"
    active.loc[active["asset_id"] == "DH_PUMP_GKS", "feedback_model"] = "PANDAPIPES_CIRCULATION_PUMP"
    return active[
        [
            "interface_id", "bus_id", "sector", "asset_id", "nominal_power_mw",
            "power_mw", "voltage_pu", "speed_pu", "energized", "feedback_model",
            "confidence_class", "notes",
        ]
    ].reset_index(drop=True)


def _run_water(
    speed: float,
    iteration_dir: Path,
    config: dict[str, Any],
    pipe_diameter_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    import wntr

    nodes = pd.read_csv(OUT / "water_nodes.csv")
    pipes = pd.read_csv(OUT / "water_pipes.csv")
    source = nodes[nodes["node_type"] == "source"].iloc[0]
    source_id = str(source.node_id)
    nominal_flow = float(nodes["water_q_avg_m3s"].sum())
    head_rise = float(config["bidirectional_coupling"]["water_design_head_rise_m"])

    wn = wntr.network.WaterNetworkModel()
    for row in nodes.itertuples():
        wn.add_junction(
            row.node_id,
            base_demand=0.0 if row.node_id == source_id else max(0.0, float(row.water_q_avg_m3s)),
            elevation=float(row.height_m),
            coordinates=(row.lon, row.lat),
        )
    raw_id = "WAT_RAW_SOURCE"
    pump_id = "WAT_MAIN_PUMP"
    wn.add_reservoir(raw_id, base_head=float(source.height_m), coordinates=(source.lon, source.lat))
    wn.add_curve(
        "WAT_MAIN_CURVE",
        "HEAD",
        [(0.0, 1.20 * head_rise), (nominal_flow, head_rise), (1.50 * nominal_flow, 0.55 * head_rise)],
    )
    wn.add_pump(
        pump_id, raw_id, source_id, pump_type="HEAD", pump_parameter="WAT_MAIN_CURVE",
        speed=max(speed, 1e-4), initial_status="OPEN" if speed > 0 else "CLOSED",
    )
    pipe_diameter_overrides = pipe_diameter_overrides or {}
    effective_diameter: dict[str, float] = {}
    for row in pipes.itertuples():
        diameter_mm = float(pipe_diameter_overrides.get(str(row.pipe_id), row.diameter_mm))
        effective_diameter[str(row.pipe_id)] = diameter_mm
        wn.add_pipe(
            row.pipe_id, row.from_node, row.to_node,
            length=max(float(row.length_km) * 1000.0, 0.1),
            diameter=diameter_mm / 1000.0,
            roughness=130.0, minor_loss=float(row.loss_coefficient), initial_status="OPEN",
        )
    wn.options.time.duration = 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.minimum_pressure = 0.0
    wn.options.hydraulic.required_pressure = float(config["acceptance_screening"]["drinking_water"]["minimum_pressure_m"])
    export_path = iteration_dir / "water_epanet_bidirectional.inp"
    wntr.network.io.write_inpfile(wn, export_path, units="LPS")
    simulation = wntr.sim.EpanetSimulator(wn).run_sim()
    pressure = simulation.node["pressure"].iloc[-1].drop(labels=[raw_id])
    velocity = simulation.link["velocity"].iloc[-1].drop(labels=[pump_id]).abs()
    headloss = simulation.link["headloss"].iloc[-1].drop(labels=[pump_id]).abs()
    demand = simulation.node["demand"].iloc[-1].drop(labels=[raw_id, source_id])
    pump_flow = abs(float(simulation.link["flowrate"].iloc[-1][pump_id]))
    pump_head = float(
        simulation.node["head"].iloc[-1][source_id]
        - simulation.node["head"].iloc[-1][raw_id]
    )
    efficiency = (
        float(config["bidirectional_coupling"]["water_pump_hydraulic_efficiency"])
        * float(config["bidirectional_coupling"]["motor_efficiency"])
    )
    power_mw = 1000.0 * 9.81 * pump_flow * max(pump_head, 0.0) / efficiency / 1e6
    delivered_ratio = float(np.clip(float(demand.sum()) / max(nominal_flow, 1e-12), 0.0, 1.0))
    thresholds = config["acceptance_screening"]["drinking_water"]
    checks = {
        "converged": bool(np.isfinite(pressure).all() and np.isfinite(velocity).all()),
        "minimum_pressure": float(pressure.min()) >= thresholds["minimum_pressure_m"],
        "maximum_pressure": float(pressure.max()) <= thresholds["maximum_pressure_m"],
        "maximum_velocity": float(velocity.max()) <= thresholds["maximum_velocity_m_s"],
        "delivered_demand": delivered_ratio >= config["bidirectional_coupling"]["minimum_delivered_water_fraction"],
    }
    velocity_pipe = str(velocity.idxmax())
    headloss_pipe = str(headloss.idxmax())
    return {
        "created": True, "converged": checks["converged"], "checks": checks,
        "passed": all(checks.values()), "pump_speed_pu": speed,
        "pump_flow_m3_s": pump_flow, "pump_head_m": pump_head,
        "pump_electrical_power_mw": power_mw, "delivered_water_fraction": delivered_ratio,
        "minimum_pressure_m": float(pressure.min()), "maximum_pressure_m": float(pressure.max()),
        "maximum_velocity_m_s": float(velocity.max()),
        "limiting_velocity_pipe_id": velocity_pipe,
        "limiting_velocity_pipe_dn_mm": effective_diameter[velocity_pipe],
        "limiting_headloss_pipe_id": headloss_pipe,
        "limiting_headloss_pipe_dn_mm": effective_diameter[headloss_pipe],
        "export": str(export_path.relative_to(PROJECT)),
    }


def _swmm_variant_text(
    base_text: str, water_factor: float, pump_speeds: dict[str, float], hours: int
) -> str:
    section = ""
    output: list[str] = []
    for raw_line in base_text.splitlines():
        line = raw_line
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        elif section == "[OPTIONS]" and stripped.startswith("END_DATE"):
            line = "END_DATE             07/01/2026"
        elif section == "[OPTIONS]" and stripped.startswith("END_TIME"):
            line = f"END_TIME             {hours:02d}:00:00"
        elif section == "[PUMPS]" and stripped and not stripped.startswith(";"):
            fields = stripped.split()
            speed = pump_speeds.get(fields[0], 1.0)
            if len(fields) >= 5 and speed <= 0.0:
                fields[4] = "OFF"
                line = " ".join(fields)
        elif section == "[CURVES]" and stripped and not stripped.startswith(";"):
            fields = stripped.split()
            if fields and re.fullmatch(r"PC\d+", fields[0]):
                pump_id = "PU" + fields[0][2:]
                if len(fields) == 3 and fields[1] != "PUMP4":
                    fields[2] = f"{float(fields[2]) * pump_speeds.get(pump_id, 1.0):.8f}"
                    line = " ".join(fields)
        elif section == "[DWF]" and stripped and not stripped.startswith(";"):
            fields = stripped.split()
            if len(fields) >= 3 and fields[1] == "FLOW":
                fields[2] = f"{float(fields[2]) * water_factor:.8f}"
                line = " ".join(fields)
        output.append(line)
    return "\n".join(output) + "\n"


def _parse_swmm_report(text: str, duration_hours: int) -> dict[str, Any]:
    patterns = {
        "flow_routing_continuity_error_percent": r"Continuity Error \(\%\) \.{3,}\s+([-+]?\d+(?:\.\d+)?)",
        "flooding_loss_million_liter": r"Flooding Loss \.{3,}\s+[-+]?\d+(?:\.\d+)?\s+([-+]?\d+(?:\.\d+)?)",
        "steps_not_converging_percent": r"% of Steps Not Converging\s*:\s*([-+]?\d+(?:\.\d+)?)",
    }
    metrics: dict[str, Any] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        metrics[name] = float(match.group(1)) if match else math.inf
    pump_power: dict[str, float] = {}
    in_summary = False
    for line in text.splitlines():
        if "Pumping Summary" in line:
            in_summary = True
            continue
        if in_summary and re.match(r"^\s*PU\d+\s+", line):
            fields = line.split()
            if len(fields) >= 8:
                pump_power[fields[0]] = float(fields[7]) / max(duration_hours, 1) / 1000.0
        elif in_summary and pump_power and not line.strip():
            break
    metrics["pump_power_mw"] = pump_power
    metrics["total_pump_power_mw"] = float(sum(pump_power.values()))
    metrics["engine_run_passed"] = "ERROR " not in text and bool(pump_power)
    return metrics


def _run_wastewater(
    water_factor: float,
    pump_speeds: dict[str, float],
    iteration_dir: Path,
    config: dict[str, Any],
    duration_hours: int | None = None,
) -> dict[str, Any]:
    from swmm.toolkit import solver as swmm_solver

    duration = int(
        duration_hours
        if duration_hours is not None
        else config["bidirectional_coupling"]["wastewater_iteration_hours"]
    )
    base_text = (OUT / "schweinfurt_proxy.inp").read_text(encoding="utf-8")
    inp = iteration_dir / "wastewater_swmm_bidirectional.inp"
    rpt = iteration_dir / "wastewater_swmm_bidirectional.rpt"
    binary = iteration_dir / "wastewater_swmm_bidirectional.out"
    inp.write_text(_swmm_variant_text(base_text, water_factor, pump_speeds, duration), encoding="utf-8")
    swmm_solver.swmm_run(str(inp), str(rpt), str(binary))
    report_text = rpt.read_text(encoding="utf-8", errors="ignore")
    metrics = _parse_swmm_report(report_text, duration)
    thresholds = config["acceptance_screening"]["wastewater"]
    checks = {
        "engine_run": bool(metrics["engine_run_passed"]),
        "mass_continuity": abs(metrics["flow_routing_continuity_error_percent"])
        <= thresholds["maximum_absolute_flow_routing_continuity_error_percent"],
        "nonconverging_steps": metrics["steps_not_converging_percent"]
        <= thresholds["maximum_nonconverging_steps_percent"],
        "no_flooding": metrics["flooding_loss_million_liter"]
        <= thresholds["maximum_dry_weather_flooding_loss_million_liter"],
    }
    return {
        **metrics, "created": True, "converged": checks["engine_run"], "checks": checks,
        "passed": all(checks.values()), "sanitary_inflow_factor": water_factor,
        "duration_hours": duration, "export": str(inp.relative_to(PROJECT)),
        "report": str(rpt.relative_to(PROJECT)),
    }


def _run_heat(
    speed: float,
    iteration_dir: Path,
    config: dict[str, Any],
    pipe_diameter_overrides: dict[str, float] | None = None,
    lift_factor: float = 1.0,
) -> dict[str, Any]:
    import pandapipes as ppipe

    nodes = pd.read_csv(OUT / "heat_nodes.csv")
    corridors = pd.read_csv(OUT / "heat_corridors.csv")
    plant_rows = nodes[nodes["node_type"] == "plant"]
    if plant_rows.empty:
        raise ValueError("District-heating topology has no plant node")
    plant_node = str(plant_rows.iloc[0].node_id)
    heat = config["district_heating"]
    coupling = config["bidirectional_coupling"]
    supply_k = float(heat["nominal_supply_temperature_c"]) + 273.15
    return_k = float(heat["nominal_return_temperature_c"]) + 273.15
    delta_t = float(heat["nominal_supply_temperature_c"] - heat["nominal_return_temperature_c"])
    return_pressure = float(coupling["heat_return_pressure_bar"])
    lift = float(coupling["heat_design_pump_lift_bar"]) * speed**2 * lift_factor

    net = ppipe.create_empty_network(fluid="water")
    supply: dict[str, int] = {}
    return_side: dict[str, int] = {}
    for row in nodes.itertuples():
        supply[row.node_id] = ppipe.create_junction(
            net, pn_bar=return_pressure + lift, tfluid_k=supply_k,
            height_m=row.elevation_m, name=f"{row.node_id}_SUP", geodata=(row.lon, row.lat),
        )
        return_side[row.node_id] = ppipe.create_junction(
            net, pn_bar=return_pressure, tfluid_k=return_k,
            height_m=row.elevation_m, name=f"{row.node_id}_RET", geodata=(row.lon, row.lat),
        )
    pipe_diameter_overrides = pipe_diameter_overrides or {}
    effective_diameter: dict[str, float] = {}
    for row in corridors.itertuples():
        diameter_mm = float(
            pipe_diameter_overrides.get(str(row.corridor_id), row.diameter_mm)
        )
        effective_diameter[str(row.corridor_id)] = diameter_mm
        kwargs = {
            "length_km": max(float(row.length_km), 0.005),
            "inner_diameter_mm": diameter_mm, "k_mm": 0.1,
            "sections": max(1, int(math.ceil(float(row.length_km) / 0.25))),
        }
        ppipe.create_pipe_from_parameters(
            net, supply[row.from_node], supply[row.to_node], name=f"{row.corridor_id}_SUP", **kwargs,
        )
        ppipe.create_pipe_from_parameters(
            net, return_side[row.to_node], return_side[row.from_node], name=f"{row.corridor_id}_RET", **kwargs,
        )
    delivered_heat_mw = 0.0
    for row in nodes[nodes["node_type"] == "consumer_substation"].itertuples():
        intermediate = ppipe.create_junction(
            net, pn_bar=return_pressure + 0.5 * lift, tfluid_k=0.5 * (supply_k + return_k),
            height_m=row.elevation_m, name=f"{row.node_id}_HX", geodata=(row.lon, row.lat),
        )
        mass_flow = max(1e-4, float(row.peak_heat_mw) * 1e6 / (4180.0 * delta_t))
        ppipe.create_flow_control(
            net, supply[row.node_id], intermediate, controlled_mdot_kg_per_s=mass_flow,
            name=f"FC_{row.node_id}",
        )
        ppipe.create_heat_exchanger(
            net, intermediate, return_side[row.node_id], qext_w=float(row.peak_heat_mw) * 1e6,
            inner_diameter_mm=50.0, loss_coefficient=5.0, name=f"HX_{row.node_id}",
        )
        delivered_heat_mw += float(row.peak_heat_mw)
    ppipe.create_circ_pump_const_pressure(
        net, return_side[plant_node], supply[plant_node],
        p_flow_bar=return_pressure + lift, plift_bar=lift,
        t_flow_k=supply_k, type="pt", name="DH_MAIN_PUMP",
    )
    ppipe.pipeflow(
        net, mode="bidirectional", max_iter_hyd=100, max_iter_therm=100,
        max_iter_bidirect=100, friction_model="colebrook",
    )
    export_path = iteration_dir / "heat_pandapipes_bidirectional.json"
    ppipe.to_json(net, str(export_path))
    pump = net.res_circ_pump_pressure.iloc[0]
    mass_flow = abs(float(pump.mdot_from_kg_per_s))
    power_mw = mass_flow * lift * 1e5 / (
        float(coupling["heat_fluid_density_kg_m3"])
        * float(heat["pump_efficiency"]) * float(heat["motor_efficiency"])
    ) / 1e6
    source_heat_mw = float(pump.qext_w) / 1e6
    return_temperature_c = float(pump.t_from_k) - 273.15
    thresholds = config["acceptance_screening"]["district_heating"]
    minimum_pressure = float(net.res_junction.p_bar.min())
    node_differentials = {
        node_id: (
            float(net.res_junction.loc[supply[node_id], "p_bar"])
            - float(net.res_junction.loc[return_side[node_id], "p_bar"])
        )
        for node_id in supply
    }
    limiting_differential_node = min(node_differentials, key=node_differentials.get)
    minimum_differential_pressure = node_differentials[limiting_differential_node]
    heat_graph = nx.Graph()
    corridor_by_edge: dict[frozenset[str], str] = {}
    for row in corridors.itertuples():
        heat_graph.add_edge(str(row.from_node), str(row.to_node))
        corridor_by_edge[frozenset((str(row.from_node), str(row.to_node)))] = str(
            row.corridor_id
        )
    limiting_path = nx.shortest_path(
        heat_graph, str(plant_node), str(limiting_differential_node)
    )
    differential_path_losses: list[tuple[float, str]] = []
    for from_node, to_node in zip(limiting_path[:-1], limiting_path[1:]):
        corridor_id = corridor_by_edge[frozenset((from_node, to_node))]
        combined_loss = abs(
            float(net.res_junction.loc[supply[from_node], "p_bar"])
            - float(net.res_junction.loc[supply[to_node], "p_bar"])
        ) + abs(
            float(net.res_junction.loc[return_side[from_node], "p_bar"])
            - float(net.res_junction.loc[return_side[to_node], "p_bar"])
        )
        differential_path_losses.append((combined_loss, corridor_id))
    limiting_differential_corridor = max(differential_path_losses)[1]
    maximum_velocity = float(net.res_pipe.v_mean_m_per_s.abs().max())
    distribution_heat_loss_mw = source_heat_mw - delivered_heat_mw
    heat_loss_fraction = distribution_heat_loss_mw / max(source_heat_mw, 1e-9)
    limiting_pipe_index = net.res_pipe.v_mean_m_per_s.abs().idxmax()
    limiting_pipe_name = str(net.pipe.loc[limiting_pipe_index, "name"])
    limiting_corridor = limiting_pipe_name.rsplit("_", 1)[0]
    checks = {
        "converged": bool(net.converged),
        "minimum_pressure": minimum_pressure + 1e-6 >= thresholds["minimum_pressure_bar"],
        "minimum_differential_pressure": minimum_differential_pressure
        >= thresholds["minimum_differential_pressure_bar"],
        "maximum_velocity": maximum_velocity <= thresholds["maximum_velocity_m_s"],
        "return_temperature": coupling["heat_return_temperature_min_c"]
        <= return_temperature_c <= coupling["heat_return_temperature_max_c"],
        "heat_loss": heat_loss_fraction <= thresholds["maximum_annual_heat_loss_fraction"],
        "heat_delivery": abs(delivered_heat_mw - float(heat["design_peak_mw_assumption"]))
        <= coupling["heat_balance_tolerance_mw"],
    }
    return {
        "created": True, "converged": bool(net.converged), "checks": checks,
        "passed": all(checks.values()), "pump_speed_pu": speed,
        "pump_lift_bar": lift, "pump_mass_flow_kg_s": mass_flow,
        "pump_electrical_power_mw": power_mw,
        "delivered_heat_mw": delivered_heat_mw, "source_heat_mw": source_heat_mw,
        "distribution_heat_loss_mw": distribution_heat_loss_mw,
        "heat_loss_fraction": heat_loss_fraction,
        "return_temperature_c": return_temperature_c,
        "minimum_pressure_bar": minimum_pressure,
        "minimum_differential_pressure_bar": minimum_differential_pressure,
        "limiting_differential_node_id": limiting_differential_node,
        "limiting_differential_corridor_id": limiting_differential_corridor,
        "limiting_differential_corridor_dn_mm": effective_diameter[
            limiting_differential_corridor
        ],
        "maximum_velocity_m_s": maximum_velocity,
        "lift_factor": lift_factor,
        "limiting_velocity_corridor_id": limiting_corridor,
        "limiting_velocity_corridor_dn_mm": effective_diameter[limiting_corridor],
        "export": str(export_path.relative_to(PROJECT)),
    }


def _run_power(
    state: pd.DataFrame,
    iteration_dir: Path,
    config: dict[str, Any],
    *,
    write_export: bool = True,
    line_overrides: dict[str, dict[str, float]] | None = None,
    transformer_overrides: dict[str, dict[str, float]] | None = None,
    outaged_lines: set[str] | None = None,
    ext_grid_vm_pu: float | dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    import pandapower as pp

    net = pp.from_json(str(OUT / "power_pandapower.json"))
    line_overrides = line_overrides or {}
    transformer_overrides = transformer_overrides or {}
    outaged_lines = set(outaged_lines or set())
    for index, row in net.line.iterrows():
        if str(row["name"]) in outaged_lines:
            net.line.at[index, "in_service"] = False
        override = line_overrides.get(str(row["name"]))
        if override:
            for field in ["max_i_ka", "r_ohm_per_km", "parallel"]:
                if field in override:
                    net.line.at[index, field] = override[field]
    for index, row in net.trafo.iterrows():
        override = transformer_overrides.get(str(row["name"]))
        if override:
            for field in ["sn_mva", "parallel"]:
                if field in override:
                    net.trafo.at[index, field] = override[field]
    if ext_grid_vm_pu is not None:
        if isinstance(ext_grid_vm_pu, dict):
            for index, row in net.ext_grid.iterrows():
                name = str(row.get("name", ""))
                bus_name = str(net.bus.loc[int(row["bus"]), "name"])
                if name in ext_grid_vm_pu:
                    net.ext_grid.at[index, "vm_pu"] = float(ext_grid_vm_pu[name])
                elif bus_name in ext_grid_vm_pu:
                    net.ext_grid.at[index, "vm_pu"] = float(ext_grid_vm_pu[bus_name])
        else:
            net.ext_grid.loc[:, "vm_pu"] = float(ext_grid_vm_pu)
    bus_by_name = {str(row["name"]): int(index) for index, row in net.bus.iterrows()}
    attached = 0
    for row in state.itertuples():
        if row.bus_id not in bus_by_name or not math.isfinite(float(row.power_mw)) or row.power_mw <= 0:
            continue
        pp.create_load(
            net, bus_by_name[row.bus_id], p_mw=float(row.power_mw),
            q_mvar=float(row.power_mw) * math.tan(math.acos(0.95)),
            name=f"BIDIR_{row.interface_id}",
        )
        attached += 1
    algorithm = ""
    error: Exception | None = None
    for candidate in ["bfsw", "iwamoto_nr", "nr"]:
        try:
            pp.runpp(
                net, algorithm=candidate, calculate_voltage_angles=False,
                numba=False, max_iteration=100, init="results",
            )
            if net.converged:
                algorithm = candidate
                break
        except Exception as exc:  # pragma: no cover - solver fallback
            error = exc
    if not net.converged:
        raise RuntimeError(f"pandapower did not converge: {error}")
    export_path = iteration_dir / "power_pandapower_bidirectional.json"
    if write_export:
        pp.to_json(net, str(export_path))
    voltages = {str(row["name"]): float(net.res_bus.loc[index, "vm_pu"]) for index, row in net.bus.iterrows()}
    thresholds = config["acceptance_screening"]["electricity"]
    limiting_line_index = net.res_line.loading_percent.idxmax()
    limiting_line = net.line.loc[limiting_line_index]
    limiting_trafo_index = net.res_trafo.loading_percent.idxmax()
    limiting_trafo = net.trafo.loc[limiting_trafo_index]
    limiting_line_voltage = float(net.bus.loc[int(limiting_line.from_bus), "vn_kv"])
    metrics = {
        "created": True, "converged": True, "algorithm": algorithm,
        "attached_interface_loads": attached,
        "coupling_active_power_mw": float(state["power_mw"].sum()),
        "minimum_voltage_pu": float(net.res_bus.vm_pu.min()),
        "maximum_voltage_pu": float(net.res_bus.vm_pu.max()),
        "maximum_line_loading_percent": float(net.res_line.loading_percent.max()),
        "maximum_transformer_loading_percent": float(net.res_trafo.loading_percent.max()),
        "limiting_line_id": str(limiting_line["name"]),
        "limiting_line_voltage_kv": limiting_line_voltage,
        "limiting_line_max_i_ka": float(limiting_line.max_i_ka),
        "limiting_line_r_ohm_per_km": float(limiting_line.r_ohm_per_km),
        "limiting_line_parallel": int(limiting_line.parallel),
        "limiting_transformer_id": str(limiting_trafo["name"]),
        "limiting_transformer_sn_mva": float(limiting_trafo.sn_mva),
        "limiting_transformer_parallel": int(limiting_trafo.parallel),
        "outaged_lines": sorted(outaged_lines),
        "external_grid_voltage_override": ext_grid_vm_pu,
        "export": str(export_path.relative_to(PROJECT)) if write_export else None,
    }
    metrics["checks"] = {
        "converged": True,
        "minimum_voltage": metrics["minimum_voltage_pu"] >= thresholds["minimum_voltage_pu"],
        "maximum_voltage": metrics["maximum_voltage_pu"] <= thresholds["maximum_voltage_pu"],
        "line_loading": metrics["maximum_line_loading_percent"] <= thresholds["maximum_line_loading_percent"],
        "transformer_loading": metrics["maximum_transformer_loading_percent"] <= thresholds["maximum_transformer_loading_percent"],
    }
    metrics["passed"] = all(metrics["checks"].values())
    return metrics, voltages


def _update_interface_powers(
    state: pd.DataFrame,
    water: dict[str, Any],
    wastewater: dict[str, Any],
    heat: dict[str, Any],
    relaxation: float,
) -> pd.DataFrame:
    updated = state.copy()
    calculated = updated["nominal_power_mw"].copy()
    for index, row in updated.iterrows():
        asset = str(row.asset_id)
        if asset == "WAT-WW-01":
            calculated.loc[index] = water["pump_electrical_power_mw"]
        elif asset.startswith("WAT-WF-"):
            calculated.loc[index] = row.nominal_power_mw * row.speed_pu**3 if row.energized else 0.0
        elif asset.startswith("WAT-ST-"):
            calculated.loc[index] = row.nominal_power_mw if row.energized else 0.0
        elif asset.startswith("PU"):
            calculated.loc[index] = wastewater["pump_power_mw"].get(asset, 0.0)
        elif asset == "SEW-WWTP-01":
            calculated.loc[index] = row.nominal_power_mw * (0.60 + 0.40 * water["delivered_water_fraction"])
        elif asset == "DH_PUMP_GKS":
            calculated.loc[index] = heat["pump_electrical_power_mw"]
    updated["calculated_power_mw"] = calculated
    updated["power_mw"] = (1.0 - relaxation) * updated["power_mw"] + relaxation * calculated
    return updated


def _state_speeds(
    state: pd.DataFrame,
    voltages: dict[str, float],
    commands: dict[str, float],
    drive: dict[str, float],
) -> pd.DataFrame:
    state = state.copy()
    state["voltage_pu"] = [float(voltages.get(bus, math.nan)) for bus in state["bus_id"]]
    state["energized"] = state["voltage_pu"].notna() & (state["voltage_pu"] >= drive["trip_voltage_pu"])
    state["speed_pu"] = [
        _motor_speed(float(row.voltage_pu), commands.get(str(row.asset_id), 1.0), drive)
        if row.feedback_model != "constant_auxiliary" else (1.0 if row.energized else 0.0)
        for row in state.itertuples()
    ]
    return state


def _repair_record(
    *,
    iteration: int,
    priority: int,
    action: str,
    sector: str,
    asset_id: str,
    old_value: Any,
    new_value: Any,
    reason: str,
    phase: str,
    outcome: str = "applied",
) -> dict[str, Any]:
    return {
        "attempt": iteration,
        "priority": priority,
        "action": action,
        "sector": sector,
        "asset_id": asset_id,
        "old_value": old_value,
        "new_value": new_value,
        "reason": reason,
        "phase": phase,
        "outcome": outcome,
    }


def _next_cable_parameters(power: dict[str, Any]) -> dict[str, float] | None:
    catalogue = (
        LV_CABLE_CATALOGUE
        if float(power["limiting_line_voltage_kv"]) < 1.0
        else MV_CABLE_CATALOGUE
    )
    current_ampacity = float(power["limiting_line_max_i_ka"])
    for item in catalogue:
        if float(item["max_i_ka"]) > current_ampacity + 1e-9:
            return {
                "max_i_ka": float(item["max_i_ka"]),
                "r_ohm_per_km": float(item["r_ohm_per_km"]),
                "parallel": int(power["limiting_line_parallel"]),
                "area_mm2": float(item["area_mm2"]),
            }
    return None


def _select_and_apply_limit_repair(
    *,
    iteration: int,
    last: dict[str, Any],
    commands: dict[str, float],
    water_pipe_overrides: dict[str, float],
    heat_pipe_overrides: dict[str, float],
    line_overrides: dict[str, dict[str, float]],
    transformer_overrides: dict[str, dict[str, float]],
    runtime: dict[str, Any],
    repair_counts: dict[str, int],
    policy: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Apply exactly one declared repair; otherwise request an upstream return.

    The ordering is by repair *intrusiveness*, not by whichever solver happens
    to execute first: numerical protocol, existing operating setpoint, one
    limiting catalogue component, and finally a Phase-II regeneration request.
    """
    water = last["drinking_water"]
    wastewater = last["wastewater"]
    heat = last["district_heating"]
    power = last["electricity"]

    # Priority 2: an unchanged SWMM model may be extended to clear start-up
    # storage effects.  No geometry, diameter, threshold, or inflow is changed.
    if (
        not wastewater["passed"]
        and (
            not wastewater["checks"].get("mass_continuity", True)
            or not wastewater["checks"].get("engine_run", True)
        )
        and int(runtime["wastewater_duration_hours"]) < 24
    ):
        old = int(runtime["wastewater_duration_hours"])
        runtime["wastewater_duration_hours"] = 24
        return _repair_record(
            iteration=iteration, priority=2,
            action="extend_unchanged_swmm_initialization_run",
            sector="wastewater", asset_id="SWMM_MODEL",
            old_value=old, new_value=24,
            reason="SWMM engine/continuity screen failed during the 6 h initialization run",
            phase="Phase III-B",
        ), None

    # Priority 3: use existing bounded controls before changing an asset.
    water_rule = policy["water_speed_command"]
    water_command = float(commands.get("WAT-WW-01", 1.0))
    if (
        not water["passed"]
        and not water["checks"].get("maximum_pressure", True)
        and water_command > float(water_rule["minimum_pu"])
    ):
        new = max(float(water_rule["minimum_pu"]), water_command - float(water_rule["step_pu"]))
        commands["WAT-WW-01"] = new
        return _repair_record(
            iteration=iteration, priority=3,
            action="reduce_water_pump_speed_command",
            sector="drinking_water", asset_id="WAT-WW-01",
            old_value=water_command, new_value=new,
            reason="maximum pressure exceeded the declared screen",
            phase="Phase III-B",
        ), None
    if (
        not water["passed"]
        and (
            not water["checks"].get("minimum_pressure", True)
            or not water["checks"].get("delivered_demand", True)
        )
        and water_command < float(water_rule["maximum_pu"])
    ):
        new = min(float(water_rule["maximum_pu"]), water_command + float(water_rule["step_pu"]))
        commands["WAT-WW-01"] = new
        return _repair_record(
            iteration=iteration, priority=3,
            action="increase_water_pump_speed_command",
            sector="drinking_water", asset_id="WAT-WW-01",
            old_value=water_command, new_value=new,
            reason="minimum pressure or delivered-demand screen failed",
            phase="Phase III-B",
        ), None

    heat_rule = policy["heat_speed_command"]
    heat_command = float(commands.get("DH_PUMP_GKS", 1.0))
    if (
        not heat["passed"]
        and not heat["checks"].get("minimum_pressure", True)
        and heat_command < float(heat_rule["maximum_pu"])
    ):
        new = min(float(heat_rule["maximum_pu"]), heat_command + float(heat_rule["step_pu"]))
        commands["DH_PUMP_GKS"] = new
        return _repair_record(
            iteration=iteration, priority=3,
            action="increase_heat_pump_speed_command",
            sector="district_heating", asset_id="DH_PUMP_GKS",
            old_value=heat_command, new_value=new,
            reason="minimum district-heating pressure screen failed",
            phase="Phase III-B",
        ), None
    lift_rule = policy["heat_lift_factor"]
    if (
        not heat["passed"]
        and not heat["checks"].get("minimum_pressure", True)
        and float(runtime["heat_lift_factor"]) < float(lift_rule["maximum"])
    ):
        old = float(runtime["heat_lift_factor"])
        runtime["heat_lift_factor"] = min(
            float(lift_rule["maximum"]), old + float(lift_rule["step"])
        )
        return _repair_record(
            iteration=iteration, priority=3,
            action="increase_existing_heat_pump_lift_setpoint",
            sector="district_heating", asset_id="DH_MAIN_PUMP",
            old_value=old, new_value=runtime["heat_lift_factor"],
            reason="minimum pressure still failed after the speed-command range was exhausted",
            phase="Phase III-B",
        ), None

    # Priority 4: change one limiting component to the next declared catalogue
    # entry.  A whole network or threshold is never scaled.
    maximum_upgrades = int(policy["maximum_component_upgrades_per_sector"])
    sector_order = list(policy["sector_tie_break_order"])
    for sector in sector_order:
        if repair_counts.get(sector, 0) >= maximum_upgrades:
            continue
        if sector == "electricity" and not power["passed"]:
            if not power["checks"]["line_loading"] or not power["checks"]["minimum_voltage"]:
                target = str(power["limiting_line_id"])
                next_parameters = _next_cable_parameters(power)
                if next_parameters is not None:
                    old = {
                        "max_i_ka": power["limiting_line_max_i_ka"],
                        "r_ohm_per_km": power["limiting_line_r_ohm_per_km"],
                        "parallel": power["limiting_line_parallel"],
                    }
                    line_overrides[target] = {
                        key: next_parameters[key]
                        for key in ["max_i_ka", "r_ohm_per_km", "parallel"]
                    }
                    repair_counts[sector] = repair_counts.get(sector, 0) + 1
                    return _repair_record(
                        iteration=iteration, priority=4,
                        action="upsize_limiting_electric_line_next_catalogue",
                        sector=sector, asset_id=target, old_value=old,
                        new_value=next_parameters,
                        reason="minimum voltage or line-loading screen failed",
                        phase="Phase II-A",
                    ), None
                old_parallel = int(power["limiting_line_parallel"])
                line_overrides[target] = {
                    "max_i_ka": float(power["limiting_line_max_i_ka"]),
                    "r_ohm_per_km": float(power["limiting_line_r_ohm_per_km"]),
                    "parallel": old_parallel + 1,
                }
                repair_counts[sector] = repair_counts.get(sector, 0) + 1
                return _repair_record(
                    iteration=iteration, priority=4,
                    action="add_parallel_circuit_to_limiting_electric_line",
                    sector=sector, asset_id=target, old_value=old_parallel,
                    new_value=old_parallel + 1,
                    reason="largest declared conductor was exhausted",
                    phase="Phase II-A",
                ), None
            if not power["checks"]["transformer_loading"]:
                target = str(power["limiting_transformer_id"])
                old_rating = float(power["limiting_transformer_sn_mva"])
                next_rating = next_catalogue_value(old_rating, TRANSFORMER_CATALOGUE_MVA)
                old_parallel = int(power["limiting_transformer_parallel"])
                if next_rating is not None:
                    transformer_overrides[target] = {
                        "sn_mva": next_rating, "parallel": old_parallel
                    }
                    new_value: Any = next_rating
                    action = "upsize_limiting_transformer_next_catalogue"
                else:
                    transformer_overrides[target] = {
                        "sn_mva": old_rating, "parallel": old_parallel + 1
                    }
                    new_value = old_parallel + 1
                    action = "add_parallel_transformer_unit"
                repair_counts[sector] = repair_counts.get(sector, 0) + 1
                return _repair_record(
                    iteration=iteration, priority=4, action=action,
                    sector=sector, asset_id=target,
                    old_value=old_rating if next_rating is not None else old_parallel,
                    new_value=new_value,
                    reason="transformer-loading screen failed",
                    phase="Phase II-A",
                ), None

        if sector == "drinking_water" and not water["passed"]:
            if not water["checks"]["maximum_velocity"]:
                target = str(water["limiting_velocity_pipe_id"])
                old_dn = float(water["limiting_velocity_pipe_dn_mm"])
            elif not water["checks"]["minimum_pressure"] or not water["checks"]["delivered_demand"]:
                target = str(water["limiting_headloss_pipe_id"])
                old_dn = float(water["limiting_headloss_pipe_dn_mm"])
            else:
                target, old_dn = "", 0.0
            next_dn = next_catalogue_value(old_dn, config["equipment_libraries"]["water_dn_mm"])
            if target and next_dn is not None:
                water_pipe_overrides[target] = next_dn
                repair_counts[sector] = repair_counts.get(sector, 0) + 1
                return _repair_record(
                    iteration=iteration, priority=4,
                    action="upsize_limiting_water_pipe_next_dn",
                    sector=sector, asset_id=target, old_value=old_dn,
                    new_value=next_dn,
                    reason="water pressure/delivery or velocity screen failed after control adjustment",
                    phase="Phase II-B",
                ), None

        if sector == "district_heating" and not heat["passed"]:
            if not heat["checks"]["maximum_velocity"]:
                target = str(heat["limiting_velocity_corridor_id"])
                old_dn = float(heat["limiting_velocity_corridor_dn_mm"])
                reason = "district-heating velocity screen failed"
            elif not heat["checks"].get("minimum_differential_pressure", True):
                target = str(heat["limiting_differential_corridor_id"])
                old_dn = float(heat["limiting_differential_corridor_dn_mm"])
                reason = (
                    "minimum consumer differential-pressure screen failed; "
                    "upsize the largest-loss corridor on the limiting path"
                )
            else:
                target, old_dn, reason = "", 0.0, ""
            if target:
                next_dn = next_catalogue_value(
                    old_dn, config["equipment_libraries"]["district_heat_dn_mm"]
                )
                if next_dn is not None:
                    heat_pipe_overrides[target] = next_dn
                    repair_counts[sector] = repair_counts.get(sector, 0) + 1
                    return _repair_record(
                        iteration=iteration, priority=4,
                        action="upsize_limiting_heat_pipe_next_dn",
                        sector=sector, asset_id=target, old_value=old_dn,
                        new_value=next_dn,
                        reason=reason,
                        phase="Phase II-D",
                    ), None

    failing_sector = next(
        (
            sector for sector in sector_order
            if not last[sector]["passed"]
        ),
        "fixed_point",
    )
    return None, failing_sector


def _copy_final_exports(iteration_dir: Path, state: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    final_dir = OUT / "final_accepted"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)
    mapping = {
        "power_pandapower_bidirectional.json": "power_pandapower_bidirectional.json",
        "water_epanet_bidirectional.inp": "water_epanet_bidirectional.inp",
        "wastewater_swmm_bidirectional.inp": "wastewater_swmm_bidirectional.inp",
        "wastewater_swmm_bidirectional.rpt": "wastewater_swmm_bidirectional.rpt",
        "heat_pandapipes_bidirectional.json": "heat_pandapipes_bidirectional.json",
    }
    for source_name, target_name in mapping.items():
        shutil.copy2(iteration_dir / source_name, final_dir / target_name)
    state.to_csv(final_dir / "bidirectional_interface_states.csv", index=False)
    (final_dir / "bidirectional_acceptance_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return final_dir


def run_bidirectional_cosimulation(
    interfaces: pd.DataFrame,
    native_screening: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Run the integrated fixed-point exchange and enforce the final export gate."""
    config = _config()
    settings = config["bidirectional_coupling"]
    policy = declared_policy(config)
    # A new solve must never inherit iteration files or an accepted export from
    # a previous attempt.  In particular, a failed native/coupled rerun must
    # leave no stale directory that could be mistaken for its own acceptance.
    if COSIM.exists():
        shutil.rmtree(COSIM)
    COSIM.mkdir(parents=True, exist_ok=True)
    prior_final = OUT / "final_accepted"
    if prior_final.exists():
        shutil.rmtree(prior_final)
    native_passed = all(item.get("passed", False) for item in native_screening.values())
    if not native_passed:
        terminal = terminal_failure(
            "one or more native Phase-II sector screens failed before coupling"
        ).to_dict()
        failure = {
            "status": policy["terminal_status"], "accepted": False,
            "native_screening": native_screening,
            "repair_policy": policy,
            "terminal_failure": terminal,
            "final_export_created": False,
        }
        pd.DataFrame([
            _repair_record(
                iteration=0, priority=6,
                action=terminal["action"], sector=terminal["sector"],
                asset_id=terminal["target_id"], old_value="native gate failed",
                new_value="no accepted export", reason=terminal["reason"],
                phase=terminal["phase"], outcome="terminal_failure",
            )
        ]).to_csv(COSIM / "bidirectional_repairs.csv", index=False)
        (COSIM / "bidirectional_coupling_manifest.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        return failure

    state = _active_interface_state(interfaces)
    interface_errors = validate_interface_state(state)
    if interface_errors:
        decision = phase_return_for_failure("interface").to_dict()
        terminal = terminal_failure(
            f"interface contract failed: {', '.join(interface_errors)}", "interface"
        ).to_dict()
        failure = {
            "status": policy["terminal_status"],
            "accepted": False,
            "converged": False,
            "native_screening": native_screening,
            "repair_policy": policy,
            "interface_contract_errors": interface_errors,
            "phase_return_request": decision,
            "terminal_failure": terminal,
            "final_export_created": False,
        }
        pd.DataFrame([
            _repair_record(
                iteration=0, priority=1,
                action=decision["action"], sector=decision["sector"],
                asset_id=decision["target_id"], old_value="invalid interface contract",
                new_value="return to Phase III-A", reason=decision["reason"],
                phase=decision["phase"], outcome="regeneration_required",
            ),
            _repair_record(
                iteration=0, priority=6,
                action=terminal["action"], sector=terminal["sector"],
                asset_id=terminal["target_id"], old_value="invalid candidate",
                new_value="no accepted export", reason=terminal["reason"],
                phase=terminal["phase"], outcome="terminal_failure",
            ),
        ]).to_csv(
            COSIM / "bidirectional_repairs.csv", index=False
        )
        (COSIM / "bidirectional_coupling_manifest.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        return failure
    drive = settings["drive_model"]
    commands = {asset: 1.0 for asset in state["asset_id"]}
    voltages = {bus: 1.0 for bus in state["bus_id"]}
    histories: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    previous_delivery = 1.0
    previous_return_c = float(config["district_heating"]["nominal_return_temperature_c"])
    converged = False
    last: dict[str, Any] = {}
    last_dir: Path | None = None
    relaxation = float(policy["relaxation_schedule"][0]["factor"])
    relaxation_stage = 0
    water_pipe_overrides: dict[str, float] = {}
    heat_pipe_overrides: dict[str, float] = {}
    line_overrides: dict[str, dict[str, float]] = {}
    transformer_overrides: dict[str, dict[str, float]] = {}
    runtime: dict[str, Any] = {
        "wastewater_duration_hours": int(settings["wastewater_iteration_hours"]),
        "heat_lift_factor": float(policy["heat_lift_factor"]["minimum"]),
    }
    repair_counts = {sector: 0 for sector in policy["sector_tie_break_order"]}
    phase_return_request: dict[str, Any] | None = None
    terminal: dict[str, Any] | None = None

    for iteration in range(1, int(settings["maximum_iterations"]) + 1):
        while (
            relaxation_stage + 1 < len(policy["relaxation_schedule"])
            and iteration
            > int(policy["relaxation_schedule"][relaxation_stage]["last_global_iteration"])
        ):
            old_relaxation = relaxation
            relaxation_stage += 1
            relaxation = float(policy["relaxation_schedule"][relaxation_stage]["factor"])
            repairs.append(
                _repair_record(
                    iteration=iteration, priority=2,
                    action="reduce_fixed_point_relaxation",
                    sector="coupled_system", asset_id="FIXED_POINT",
                    old_value=old_relaxation, new_value=relaxation,
                    reason="the preceding declared numerical stage did not reach all residual tolerances",
                    phase="Phase III-B",
                )
            )
        iteration_dir = COSIM / "iterations" / f"iter_{iteration:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        previous_state = state.copy()
        previous_voltages = dict(voltages)
        state = _state_speeds(state, voltages, commands, drive)
        water_speed = float(state.loc[state["asset_id"] == "WAT-WW-01", "speed_pu"].iloc[0])
        heat_speed = float(state.loc[state["asset_id"] == "DH_PUMP_GKS", "speed_pu"].iloc[0])
        sewer_speeds = {
            str(row.asset_id): float(row.speed_pu)
            for row in state[state["asset_id"].str.startswith("PU")].itertuples()
        }
        try:
            water = _run_water(
                water_speed, iteration_dir, config, water_pipe_overrides
            )
            wastewater = _run_wastewater(
                water["delivered_water_fraction"], sewer_speeds, iteration_dir,
                config, int(runtime["wastewater_duration_hours"]),
            )
            heat = _run_heat(
                heat_speed, iteration_dir, config, heat_pipe_overrides,
                float(runtime["heat_lift_factor"]),
            )
            state = _update_interface_powers(state, water, wastewater, heat, relaxation)
            power, voltages = _run_power(
                state, iteration_dir, config,
                line_overrides=line_overrides,
                transformer_overrides=transformer_overrides,
            )
        except Exception as exc:
            last = {"error": f"{type(exc).__name__}: {exc}"}
            histories.append({"iteration": iteration, "solver_error": last["error"], "converged": False})
            last_dir = iteration_dir
            break

        voltage_residual = max(
            abs(voltages.get(bus, math.nan) - previous_voltages.get(bus, 1.0))
            for bus in state["bus_id"] if math.isfinite(voltages.get(bus, math.nan))
        )
        power_abs_residual = float((state["power_mw"] - previous_state["power_mw"]).abs().max())
        power_l1_relative = float(
            (state["power_mw"] - previous_state["power_mw"]).abs().sum()
            / max(float(previous_state["power_mw"].sum()), 1e-9)
        )
        delivery_residual = abs(water["delivered_water_fraction"] - previous_delivery)
        temperature_residual = abs(heat["return_temperature_c"] - previous_return_c)
        state = _state_speeds(state, voltages, commands, drive)
        iteration_passed = all([power["passed"], water["passed"], wastewater["passed"], heat["passed"]])
        residual_passed = all(
            [
                voltage_residual <= settings["voltage_tolerance_pu"],
                power_abs_residual <= settings["interface_power_absolute_tolerance_mw"],
                power_l1_relative <= settings["interface_power_l1_relative_tolerance"],
                delivery_residual <= settings["delivered_water_tolerance"],
                temperature_residual <= settings["return_temperature_tolerance_c"],
            ]
        )
        histories.append(
            {
                "iteration": iteration,
                "total_interface_power_mw": float(state["power_mw"].sum()),
                "minimum_voltage_pu": power["minimum_voltage_pu"],
                "maximum_line_loading_percent": power["maximum_line_loading_percent"],
                "maximum_transformer_loading_percent": power["maximum_transformer_loading_percent"],
                "water_pump_power_mw": water["pump_electrical_power_mw"],
                "water_delivered_fraction": water["delivered_water_fraction"],
                "water_minimum_pressure_m": water["minimum_pressure_m"],
                "wastewater_pump_power_mw": wastewater["total_pump_power_mw"],
                "wastewater_continuity_error_percent": wastewater["flow_routing_continuity_error_percent"],
                "heat_pump_power_mw": heat["pump_electrical_power_mw"],
                "heat_return_temperature_c": heat["return_temperature_c"],
                "heat_minimum_pressure_bar": heat["minimum_pressure_bar"],
                "voltage_residual_pu": voltage_residual,
                "interface_power_absolute_residual_mw": power_abs_residual,
                "interface_power_l1_relative_residual": power_l1_relative,
                "delivered_water_residual": delivery_residual,
                "return_temperature_residual_c": temperature_residual,
                "all_sector_limits_passed": iteration_passed,
                "residuals_passed": residual_passed,
                "converged": bool(iteration >= settings["minimum_iterations"] and iteration_passed and residual_passed),
            }
        )
        previous_delivery = water["delivered_water_fraction"]
        previous_return_c = heat["return_temperature_c"]
        last = {"electricity": power, "drinking_water": water, "wastewater": wastewater, "district_heating": heat}
        last_dir = iteration_dir
        if histories[-1]["converged"]:
            converged = True
            break
        if not iteration_passed:
            if len(repairs) >= int(policy["maximum_total_repairs"]):
                terminal = terminal_failure(
                    "maximum_total_repairs was reached before all coupled limits passed"
                ).to_dict()
                repairs.append(
                    _repair_record(
                        iteration=iteration, priority=6,
                        action=terminal["action"], sector=terminal["sector"],
                        asset_id=terminal["target_id"], old_value="failed state",
                        new_value="no accepted export",
                        reason=terminal["reason"], phase=terminal["phase"],
                        outcome="terminal_failure",
                    )
                )
                break
            action, failing_sector = _select_and_apply_limit_repair(
                iteration=iteration, last=last, commands=commands,
                water_pipe_overrides=water_pipe_overrides,
                heat_pipe_overrides=heat_pipe_overrides,
                line_overrides=line_overrides,
                transformer_overrides=transformer_overrides,
                runtime=runtime, repair_counts=repair_counts,
                policy=policy, config=config,
            )
            if action is not None:
                repairs.append(action)
                histories[-1]["repair_action"] = action["action"]
                histories[-1]["repair_priority"] = action["priority"]
                continue
            phase_return_request = phase_return_for_failure(
                str(failing_sector)
            ).to_dict()
            repairs.append(
                _repair_record(
                    iteration=iteration,
                    priority=int(phase_return_request["priority"]),
                    action=str(phase_return_request["action"]),
                    sector=str(phase_return_request["sector"]),
                    asset_id=str(phase_return_request["target_id"]),
                    old_value="bounded local repair exhausted",
                    new_value="regenerate affected topology branch",
                    reason=str(phase_return_request["reason"]),
                    phase=str(phase_return_request["phase"]),
                    outcome="regeneration_required",
                )
            )
            terminal = terminal_failure(
                "the required topology-level regeneration is outside the frozen Phase-III state loop; the present candidate is rejected",
                str(failing_sector),
            ).to_dict()
            repairs.append(
                _repair_record(
                    iteration=iteration, priority=6,
                    action=terminal["action"], sector=terminal["sector"],
                    asset_id=terminal["target_id"], old_value="failed state",
                    new_value="no accepted export", reason=terminal["reason"],
                    phase=terminal["phase"], outcome="terminal_failure",
                )
            )
            break

    if not converged and terminal is None:
        failure_reason = (
            str(last.get("error"))
            if "error" in last
            else "all declared relaxation stages ended before the fixed-point and engineering gates passed"
        )
        phase_return_request = phase_return_for_failure("fixed_point").to_dict()
        repairs.append(
            _repair_record(
                iteration=len(histories), priority=5,
                action=phase_return_request["action"],
                sector=phase_return_request["sector"],
                asset_id=phase_return_request["target_id"],
                old_value="all numerical stages attempted",
                new_value="no further declared numerical action",
                reason=failure_reason, phase=phase_return_request["phase"],
                outcome="exhausted",
            )
        )
        terminal = terminal_failure(failure_reason, "fixed_point").to_dict()
        repairs.append(
            _repair_record(
                iteration=len(histories), priority=6,
                action=terminal["action"], sector=terminal["sector"],
                asset_id=terminal["target_id"], old_value="failed state",
                new_value="no accepted export", reason=terminal["reason"],
                phase=terminal["phase"], outcome="terminal_failure",
            )
        )

    history = pd.DataFrame(histories)
    history.to_csv(COSIM / "bidirectional_convergence_history.csv", index=False)
    state.to_csv(COSIM / "bidirectional_interface_states.csv", index=False)
    pd.DataFrame(repairs, columns=[
        "attempt", "priority", "action", "sector", "asset_id", "old_value",
        "new_value", "reason", "phase", "outcome",
    ]).to_csv(
        COSIM / "bidirectional_repairs.csv", index=False,
    )
    interface_consistency = {
        "all_active_interfaces_have_bus": bool(state["bus_id"].notna().all()),
        "all_active_interfaces_energized": bool(state["energized"].all()),
        "finite_nonnegative_interface_power": bool(np.isfinite(state["power_mw"]).all() and (state["power_mw"] >= 0).all()),
        "water_to_wastewater_factor_exact": bool(
            last.get("wastewater", {}).get("sanitary_inflow_factor")
            == last.get("drinking_water", {}).get("delivered_water_fraction")
        ),
    }
    accepted = bool(converged and all(interface_consistency.values()))
    manifest = {
        "status": "accepted" if accepted else policy["terminal_status"],
        "accepted": accepted,
        "method": "Gauss--Seidel fixed-point co-simulation with under-relaxed interface powers",
        "coupling_direction": "sector hydraulic/thermal states -> electrical duties -> pandapower voltages/energization -> motor speed/head/flow -> sector states",
        "native_screening": native_screening,
        "settings": settings,
        "repair_policy": policy,
        "iterations": int(len(history)),
        "converged": converged,
        "repairs_applied": repairs,
        "component_overrides": {
            "electric_lines": line_overrides,
            "transformers": transformer_overrides,
            "water_pipes_dn_mm": water_pipe_overrides,
            "heat_pipes_dn_mm": heat_pipe_overrides,
            "wastewater_duration_hours": runtime["wastewater_duration_hours"],
            "heat_lift_factor": runtime["heat_lift_factor"],
            "operating_commands": commands,
        },
        "phase_return_request": phase_return_request,
        "terminal_failure": terminal,
        "final_sector_states": last,
        "interface_consistency": interface_consistency,
        "final_export_created": False,
    }
    if accepted and last_dir is not None:
        final_dir = _copy_final_exports(last_dir, state, manifest)
        manifest["final_export_created"] = True
        manifest["final_export_directory"] = str(final_dir.relative_to(PROJECT))
        (final_dir / "bidirectional_acceptance_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (COSIM / "bidirectional_coupling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
