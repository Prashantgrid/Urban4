#!/usr/bin/env python3
"""Deterministic hydraulic-leak response across water and electricity models.

The accepted integrated topology and interface state are held fixed. A
declared equivalent orifice opens at the drinking-water pump-discharge
junction. At each elapsed-time point, EPANET/WNTR returns leak flow, pressure,
velocity, delivered demand, and pump duty; pandapower returns the supplying-bus
voltage and feeder loading; and the voltage-sensitive drive law updates pump
speed. A transparent first-order pressure controller updates the speed command
between time points.

The calculation is sequential quasi-steady. It demonstrates propagation of
one controlled hydraulic perturbation through an explicit interface and back
to the hydraulic state. It is not a water-hammer, motor-transient, protection,
probability, reliability, or restoration model.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cosimulation import PROJECT, OUT, _config, _motor_speed, _run_power


SCENARIO = OUT / "hydraulic_leak_coupling"
ELECTRICAL_SCENARIO = OUT / "electrical_supply_disturbance"


def _run_water_with_leak(
    speed: float,
    equivalent_diameter_mm: float,
    discharge_coefficient: float,
    config: dict[str, Any],
    *,
    export_path: Path | None = None,
) -> dict[str, Any]:
    """Solve one EPANET pressure-dependent hydraulic state with an emitter leak."""
    import wntr

    nodes = pd.read_csv(OUT / "water_nodes.csv")
    pipes = pd.read_csv(OUT / "water_pipes.csv")
    source = nodes[nodes["node_type"] == "source"].iloc[0]
    source_id = str(source.node_id)
    raw_id = "WAT_RAW_SOURCE"
    pump_id = "WAT_MAIN_PUMP"
    nominal_service_flow = float(nodes["water_q_avg_m3s"].sum())
    head_rise = float(config["bidirectional_coupling"]["water_design_head_rise_m"])

    wn = wntr.network.WaterNetworkModel()
    for row in nodes.itertuples():
        wn.add_junction(
            row.node_id,
            base_demand=0.0 if row.node_id == source_id else max(0.0, float(row.water_q_avg_m3s)),
            elevation=float(row.height_m),
            coordinates=(row.lon, row.lat),
        )
    wn.add_reservoir(raw_id, base_head=float(source.height_m), coordinates=(source.lon, source.lat))
    wn.add_curve(
        "WAT_MAIN_CURVE",
        "HEAD",
        [
            (0.0, 1.20 * head_rise),
            (nominal_service_flow, head_rise),
            (1.50 * nominal_service_flow, 0.55 * head_rise),
        ],
    )
    wn.add_pump(
        pump_id,
        raw_id,
        source_id,
        pump_type="HEAD",
        pump_parameter="WAT_MAIN_CURVE",
        speed=max(float(speed), 1e-4),
        initial_status="OPEN" if speed > 0 else "CLOSED",
    )
    for row in pipes.itertuples():
        wn.add_pipe(
            row.pipe_id,
            row.from_node,
            row.to_node,
            length=max(float(row.length_km) * 1000.0, 0.1),
            diameter=float(row.diameter_mm) / 1000.0,
            roughness=130.0,
            minor_loss=float(row.loss_coefficient),
            initial_status="OPEN",
        )

    diameter_m = max(float(equivalent_diameter_mm), 0.0) / 1000.0
    area_m2 = math.pi * diameter_m**2 / 4.0
    emitter_coefficient = float(discharge_coefficient) * area_m2 * math.sqrt(2.0 * 9.81)
    if emitter_coefficient > 0.0:
        # EPANET emitters use q=C*h^0.5. WNTR stores C in SI units,
        # so Cd*A*sqrt(2g) represents the declared circular opening.
        wn.get_node(source_id).emitter_coefficient = emitter_coefficient

    thresholds = config["acceptance_screening"]["drinking_water"]
    wn.options.time.duration = 0
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.minimum_pressure = 0.0
    wn.options.hydraulic.required_pressure = float(thresholds["minimum_pressure_m"])
    if export_path is not None:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        wntr.network.io.write_inpfile(wn, export_path, units="LPS")

    simulation = wntr.sim.EpanetSimulator(wn).run_sim()
    pressure = simulation.node["pressure"].iloc[-1].drop(labels=[raw_id])
    velocity = simulation.link["velocity"].iloc[-1].drop(labels=[pump_id]).abs()
    demand = simulation.node["demand"].iloc[-1]
    service_demand = demand.drop(labels=[raw_id, source_id]).clip(lower=0.0)
    leak_flow = max(0.0, float(demand[source_id]))
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
    delivered_ratio = float(
        np.clip(float(service_demand.sum()) / max(nominal_service_flow, 1e-12), 0.0, 1.0)
    )
    checks = {
        "converged": bool(np.isfinite(pressure).all() and np.isfinite(velocity).all()),
        "minimum_pressure": float(pressure.min()) >= float(thresholds["minimum_pressure_m"]),
        "maximum_pressure": float(pressure.max()) <= float(thresholds["maximum_pressure_m"]),
        "maximum_velocity": float(velocity.max()) <= float(thresholds["maximum_velocity_m_s"]),
        "delivered_demand": delivered_ratio
        >= float(config["bidirectional_coupling"]["minimum_delivered_water_fraction"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pump_speed_pu": float(speed),
        "pump_flow_m3_s": pump_flow,
        "pump_head_m": pump_head,
        "pump_electrical_power_mw": power_mw,
        "leak_flow_m3_s": leak_flow,
        "delivered_water_fraction": delivered_ratio,
        "discharge_pressure_m": float(pressure[source_id]),
        "minimum_pressure_m": float(pressure.min()),
        "maximum_pressure_m": float(pressure.max()),
        "maximum_velocity_m_s": float(velocity.max()),
        "leak_node_id": source_id,
        "leak_area_m2": area_m2,
        "emitter_coefficient_m3_s_sqrt_m": emitter_coefficient,
    }


def run_hydraulic_leak_demonstration() -> dict[str, Any]:
    """Run the elapsed-time leak response and write the auditable state ledger."""
    config = _config()
    scenario = config["hydraulic_leak_demonstration"]
    coupling = config["bidirectional_coupling"]
    if SCENARIO.exists():
        shutil.rmtree(SCENARIO)
    SCENARIO.mkdir(parents=True)

    accepted_state_path = OUT / "final_accepted" / "bidirectional_interface_states.csv"
    accepted_manifest_path = OUT / "final_accepted" / "bidirectional_acceptance_manifest.json"
    if not accepted_state_path.exists() or not accepted_manifest_path.exists():
        raise FileNotFoundError("Run the accepted bidirectional base case before the leak demonstration")
    accepted_manifest = json.loads(accepted_manifest_path.read_text(encoding="utf-8"))
    if not accepted_manifest.get("accepted", False):
        raise RuntimeError("The leak demonstration requires an accepted bidirectional base case")

    state = pd.read_csv(accepted_state_path)
    main_mask = state["asset_id"] == "WAT-WW-01"
    if int(main_mask.sum()) != 1:
        raise ValueError("Expected exactly one WAT-WW-01 interface")
    main_index = int(state.index[main_mask][0])
    main_bus = str(state.loc[main_index, "bus_id"])
    voltages = {str(row.bus_id): float(row.voltage_pu) for row in state.itertuples()}
    command = float(scenario["initial_speed_command_pu"])
    relaxation = float(coupling["relaxation_factor"])
    time_step = int(scenario["time_step_minutes"])
    times = range(0, int(scenario["simulation_end_minute"]) + time_step, time_step)
    target_discharge_pressure: float | None = None
    rows: list[dict[str, Any]] = []
    last_water: dict[str, Any] | None = None

    power_scratch = SCENARIO / "solver_scratch"
    power_scratch.mkdir(parents=True)
    for elapsed_minute in times:
        leak_active = elapsed_minute >= int(scenario["leak_start_minute"])
        diameter = float(scenario["equivalent_orifice_diameter_mm"]) if leak_active else 0.0
        inner_converged = False
        inner_iterations = 0
        water: dict[str, Any] = {}
        power: dict[str, Any] = {}
        voltage_residual = math.inf
        power_residual = math.inf

        for inner in range(1, int(scenario["maximum_inner_iterations"]) + 1):
            inner_iterations = inner
            previous_voltage = float(voltages.get(main_bus, 1.0))
            previous_power = float(state.loc[main_index, "power_mw"])
            actual_speed = _motor_speed(previous_voltage, command, coupling["drive_model"])
            water = _run_water_with_leak(
                actual_speed,
                diameter,
                float(scenario["discharge_coefficient"]),
                config,
            )
            calculated_power = float(water["pump_electrical_power_mw"])
            state.loc[main_index, "calculated_power_mw"] = calculated_power
            state.loc[main_index, "power_mw"] = (
                (1.0 - relaxation) * previous_power + relaxation * calculated_power
            )
            power, voltages = _run_power(
                state,
                power_scratch,
                config,
                write_export=False,
            )
            voltage_residual = abs(float(voltages[main_bus]) - previous_voltage)
            power_residual = abs(float(state.loc[main_index, "power_mw"]) - previous_power)
            if (
                inner >= int(scenario["minimum_inner_iterations"])
                and voltage_residual <= float(coupling["voltage_tolerance_pu"])
                and power_residual <= float(coupling["interface_power_absolute_tolerance_mw"])
            ):
                inner_converged = True
                break

        if target_discharge_pressure is None:
            target_discharge_pressure = float(water["discharge_pressure_m"])

        final_voltage = float(voltages[main_bus])
        final_speed = _motor_speed(final_voltage, command, coupling["drive_model"])
        step_passed = bool(inner_converged and water["passed"] and power["passed"])
        rows.append(
            {
                "elapsed_minute": elapsed_minute,
                "leak_active": leak_active,
                "equivalent_orifice_diameter_mm": diameter,
                "leak_flow_l_s": 1000.0 * float(water["leak_flow_m3_s"]),
                "pump_flow_l_s": 1000.0 * float(water["pump_flow_m3_s"]),
                "pump_electrical_power_mw": float(water["pump_electrical_power_mw"]),
                "pump_speed_command_pu": command,
                "pump_actual_speed_pu": final_speed,
                "pump_discharge_pressure_m": float(water["discharge_pressure_m"]),
                "critical_node_pressure_m": float(water["minimum_pressure_m"]),
                "maximum_pressure_m": float(water["maximum_pressure_m"]),
                "maximum_water_velocity_m_s": float(water["maximum_velocity_m_s"]),
                "delivered_water_fraction": float(water["delivered_water_fraction"]),
                "pump_bus_id": main_bus,
                "pump_bus_voltage_pu": final_voltage,
                "system_minimum_voltage_pu": float(power["minimum_voltage_pu"]),
                "maximum_line_loading_percent": float(power["maximum_line_loading_percent"]),
                "maximum_transformer_loading_percent": float(power["maximum_transformer_loading_percent"]),
                "inner_iterations": inner_iterations,
                "voltage_residual_pu": voltage_residual,
                "pump_power_residual_mw": power_residual,
                "water_limits_passed": bool(water["passed"]),
                "electricity_limits_passed": bool(power["passed"]),
                "inner_coupling_converged": inner_converged,
                "step_accepted": step_passed,
            }
        )
        if not step_passed:
            raise RuntimeError(f"Hydraulic-leak state at {elapsed_minute} min failed its declared gate")

        # First-order pressure-control update for the next elapsed-time point.
        # The desired speed follows H~n^2; the command is adjusted with a
        # declared time constant and bounded independently of the
        # voltage-dependent actual-speed limit.
        desired_actual_speed = final_speed * math.sqrt(
            target_discharge_pressure / max(float(water["discharge_pressure_m"]), 1e-6)
        )
        voltage_factor = final_speed / max(command, 1e-9)
        desired_command = desired_actual_speed / max(voltage_factor, 1e-9)
        alpha = min(
            1.0,
            float(time_step) / float(scenario["pressure_controller_time_constant_minutes"]),
        )
        command = float(
            np.clip(
                command + alpha * (desired_command - command),
                float(scenario["minimum_speed_command_pu"]),
                float(scenario["maximum_speed_command_pu"]),
            )
        )
        last_water = water

    timeline = pd.DataFrame(rows)
    power_scratch.rmdir()
    timeline.to_csv(SCENARIO / "hydraulic_leak_timeseries.csv", index=False)
    if last_water is None:
        raise RuntimeError("Leak demonstration produced no states")
    _run_water_with_leak(
        float(timeline.iloc[-1]["pump_actual_speed_pu"]),
        float(scenario["equivalent_orifice_diameter_mm"]),
        float(scenario["discharge_coefficient"]),
        config,
        export_path=SCENARIO / "water_epanet_hydraulic_leak_final.inp",
    )

    baseline = timeline[timeline["elapsed_minute"] < int(scenario["leak_start_minute"])].iloc[-1]
    onset = timeline[timeline["elapsed_minute"] == int(scenario["leak_start_minute"])].iloc[0]
    final = timeline.iloc[-1]
    summary = {
        "accepted": bool(timeline["step_accepted"].all()),
        "method": "sequential quasi-steady WNTR--pandapower fixed point at each elapsed-time step",
        "scenario": scenario,
        "leak_node_id": str(last_water["leak_node_id"]),
        "pump_bus_id": main_bus,
        "target_discharge_pressure_m": target_discharge_pressure,
        "baseline": baseline.to_dict(),
        "leak_onset": onset.to_dict(),
        "final_leak_state": final.to_dict(),
        "response": {
            "final_leak_flow_l_s": float(final["leak_flow_l_s"]),
            "pump_flow_change_percent": 100.0
            * (float(final["pump_flow_l_s"]) / float(baseline["pump_flow_l_s"]) - 1.0),
            "pump_power_change_percent": 100.0
            * (
                float(final["pump_electrical_power_mw"])
                / float(baseline["pump_electrical_power_mw"])
                - 1.0
            ),
            "pump_bus_voltage_change_pu": float(final["pump_bus_voltage_pu"])
            - float(baseline["pump_bus_voltage_pu"]),
            "maximum_line_loading_change_percentage_points": float(final["maximum_line_loading_percent"])
            - float(baseline["maximum_line_loading_percent"]),
            "critical_pressure_onset_change_m": float(onset["critical_node_pressure_m"])
            - float(baseline["critical_node_pressure_m"]),
            "critical_pressure_final_change_m": float(final["critical_node_pressure_m"])
            - float(baseline["critical_node_pressure_m"]),
            "controller_command_limit_reached": bool(
                np.isclose(
                    float(final["pump_speed_command_pu"]),
                    float(scenario["maximum_speed_command_pu"]),
                    atol=1e-6,
                )
            ),
        },
        "all_time_steps_accepted": bool(timeline["step_accepted"].all()),
        "timeseries": str((SCENARIO / "hydraulic_leak_timeseries.csv").relative_to(PROJECT)),
        "final_epanet_export": str(
            (SCENARIO / "water_epanet_hydraulic_leak_final.inp").relative_to(PROJECT)
        ),
    }
    (SCENARIO / "hydraulic_leak_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    model_manifest_path = OUT / "model_manifest_four_sector.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    model_manifest.setdefault("solver_readiness", {})["hydraulic_leak_coupling_response"] = summary
    model_manifest_path.write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    return summary


def _service_voltage_factor(elapsed_minute: int, scenario: dict[str, Any]) -> float:
    """Return the declared post-protection service-voltage factor."""
    normal = float(scenario["normal_service_voltage_factor"])
    fault = float(scenario["fault_service_voltage_factor"])
    start = int(scenario["fault_start_minute"])
    clear_start = int(scenario["fault_clear_start_minute"])
    recovered = int(scenario["voltage_recovery_complete_minute"])
    if elapsed_minute < start or elapsed_minute >= recovered:
        return normal
    if elapsed_minute < clear_start:
        return fault
    fraction = (elapsed_minute - clear_start) / max(recovered - clear_start, 1)
    return float(fault + fraction * (normal - fault))


def run_electrical_supply_disturbance_demonstration() -> dict[str, Any]:
    """Propagate a declared electrical-side service depression into water operation.

    The scenario begins after protection has established a depressed but finite
    service voltage. It does not calculate short-circuit current or relay
    dynamics. The pandapower connection-bus voltage is multiplied by the
    declared local service factor before the existing drive law is evaluated.
    WNTR then returns pump duty and water state, and the changed duty is passed
    back to pandapower until the disturbed interface state closes.
    """
    config = _config()
    scenario = config["electrical_supply_disturbance_demonstration"]
    coupling = config["bidirectional_coupling"]
    drive = coupling["drive_model"]
    if ELECTRICAL_SCENARIO.exists():
        shutil.rmtree(ELECTRICAL_SCENARIO)
    ELECTRICAL_SCENARIO.mkdir(parents=True)

    accepted_state_path = OUT / "final_accepted" / "bidirectional_interface_states.csv"
    accepted_manifest_path = OUT / "final_accepted" / "bidirectional_acceptance_manifest.json"
    if not accepted_state_path.exists() or not accepted_manifest_path.exists():
        raise FileNotFoundError("Run the accepted bidirectional base case before the electrical disturbance")
    accepted_manifest = json.loads(accepted_manifest_path.read_text(encoding="utf-8"))
    if not accepted_manifest.get("accepted", False):
        raise RuntimeError("The electrical disturbance requires an accepted bidirectional base case")

    state = pd.read_csv(accepted_state_path)
    affected_asset = str(scenario["affected_interface"])
    affected_mask = state["asset_id"] == affected_asset
    if int(affected_mask.sum()) != 1:
        raise ValueError(f"Expected exactly one {affected_asset} interface")
    affected_index = int(state.index[affected_mask][0])
    affected_bus = str(state.loc[affected_index, "bus_id"])
    grid_voltages = {str(row.bus_id): float(row.voltage_pu) for row in state.itertuples()}
    command = float(scenario["speed_command_pu"])
    relaxation = float(coupling["relaxation_factor"])
    time_step = int(scenario["time_step_minutes"])
    times = range(0, int(scenario["simulation_end_minute"]) + time_step, time_step)
    rows: list[dict[str, Any]] = []
    worst_state: pd.DataFrame | None = None
    worst_speed = math.nan
    worst_pressure = math.inf

    power_scratch = ELECTRICAL_SCENARIO / "solver_scratch"
    power_scratch.mkdir(parents=True)
    for elapsed_minute in times:
        factor = _service_voltage_factor(elapsed_minute, scenario)
        inner_converged = False
        inner_iterations = 0
        water: dict[str, Any] = {}
        power: dict[str, Any] = {}
        voltage_residual = math.inf
        power_residual = math.inf

        for inner in range(1, int(scenario["maximum_inner_iterations"]) + 1):
            inner_iterations = inner
            previous_bus_voltage = float(grid_voltages.get(affected_bus, 1.0))
            previous_power = float(state.loc[affected_index, "power_mw"])
            terminal_voltage = factor * previous_bus_voltage
            pump_speed = _motor_speed(terminal_voltage, command, drive)
            water = _run_water_with_leak(
                pump_speed,
                0.0,
                float(config["hydraulic_leak_demonstration"]["discharge_coefficient"]),
                config,
            )
            calculated_power = float(water["pump_electrical_power_mw"])
            state.loc[affected_index, "calculated_power_mw"] = calculated_power
            state.loc[affected_index, "power_mw"] = (
                (1.0 - relaxation) * previous_power + relaxation * calculated_power
            )
            power, grid_voltages = _run_power(
                state,
                power_scratch,
                config,
                write_export=False,
            )
            voltage_residual = abs(float(grid_voltages[affected_bus]) - previous_bus_voltage)
            power_residual = abs(float(state.loc[affected_index, "power_mw"]) - previous_power)
            if (
                inner >= int(scenario["minimum_inner_iterations"])
                and voltage_residual <= float(coupling["voltage_tolerance_pu"])
                and power_residual <= float(coupling["interface_power_absolute_tolerance_mw"])
            ):
                inner_converged = True
                break

        bus_voltage = float(grid_voltages[affected_bus])
        terminal_voltage = factor * bus_voltage
        pump_speed = _motor_speed(terminal_voltage, command, drive)
        solver_state_retained = bool(inner_converged and water["checks"]["converged"] and power["passed"])
        phase = (
            "normal_pre"
            if elapsed_minute < int(scenario["fault_start_minute"])
            else "fault_depressed"
            if elapsed_minute < int(scenario["fault_clear_start_minute"])
            else "voltage_recovery"
            if elapsed_minute < int(scenario["voltage_recovery_complete_minute"])
            else "normal_post"
        )
        rows.append(
            {
                "elapsed_minute": elapsed_minute,
                "phase": phase,
                "service_voltage_factor": factor,
                "connection_bus_id": affected_bus,
                "connection_bus_voltage_pu": bus_voltage,
                "drive_terminal_voltage_pu": terminal_voltage,
                "pump_speed_command_pu": command,
                "pump_actual_speed_pu": pump_speed,
                "pump_flow_l_s": 1000.0 * float(water["pump_flow_m3_s"]),
                "pump_electrical_power_mw": float(water["pump_electrical_power_mw"]),
                "critical_node_pressure_m": float(water["minimum_pressure_m"]),
                "maximum_pressure_m": float(water["maximum_pressure_m"]),
                "maximum_water_velocity_m_s": float(water["maximum_velocity_m_s"]),
                "delivered_water_fraction": float(water["delivered_water_fraction"]),
                "system_minimum_voltage_pu": float(power["minimum_voltage_pu"]),
                "maximum_line_loading_percent": float(power["maximum_line_loading_percent"]),
                "maximum_transformer_loading_percent": float(power["maximum_transformer_loading_percent"]),
                "inner_iterations": inner_iterations,
                "voltage_residual_pu": voltage_residual,
                "pump_power_residual_mw": power_residual,
                "water_limits_passed": bool(water["passed"]),
                "electricity_limits_passed": bool(power["passed"]),
                "inner_coupling_converged": inner_converged,
                "normal_state_limits_passed": bool(water["passed"] and power["passed"]),
                "disturbed_state_retained": solver_state_retained,
            }
        )
        if not solver_state_retained:
            raise RuntimeError(
                f"Electrical disturbance state at {elapsed_minute} min did not produce a retained solver state"
            )
        if float(water["minimum_pressure_m"]) < worst_pressure:
            worst_pressure = float(water["minimum_pressure_m"])
            worst_state = state.copy()
            worst_speed = pump_speed

    timeline = pd.DataFrame(rows)
    power_scratch.rmdir()
    timeline.to_csv(ELECTRICAL_SCENARIO / "electrical_supply_disturbance_timeseries.csv", index=False)
    if worst_state is None or not math.isfinite(worst_speed):
        raise RuntimeError("Electrical disturbance produced no retained states")
    _run_water_with_leak(
        worst_speed,
        0.0,
        float(config["hydraulic_leak_demonstration"]["discharge_coefficient"]),
        config,
        export_path=ELECTRICAL_SCENARIO / "water_epanet_worst_electrical_disturbance.inp",
    )
    _run_power(
        worst_state,
        ELECTRICAL_SCENARIO,
        config,
        write_export=True,
    )

    baseline = timeline[timeline["phase"] == "normal_pre"].iloc[-1]
    worst = timeline.loc[timeline["critical_node_pressure_m"].idxmin()]
    recovered = timeline[timeline["phase"] == "normal_post"].iloc[-1]
    checks = {
        "all_inner_fixed_points_converged": bool(timeline["inner_coupling_converged"].all()),
        "all_electrical_network_states_solved": bool(timeline["electricity_limits_passed"].all()),
        "all_disturbed_solver_states_retained": bool(timeline["disturbed_state_retained"].all()),
        "pre_event_normal_limits_passed": bool(
            timeline.loc[timeline["phase"] == "normal_pre", "normal_state_limits_passed"].all()
        ),
        "disturbance_produced_water_limit_violation": bool(
            (~timeline.loc[timeline["phase"] == "fault_depressed", "water_limits_passed"]).any()
        ),
        "post_event_normal_limits_recovered": bool(
            timeline.loc[timeline["phase"] == "normal_post", "normal_state_limits_passed"].all()
        ),
    }
    summary = {
        "scenario_complete": all(checks.values()),
        "method": "sequential quasi-steady pandapower--drive--WNTR fixed point at each elapsed-time step",
        "scenario": scenario,
        "connection_bus_id": affected_bus,
        "checks": checks,
        "baseline": baseline.to_dict(),
        "worst_disturbed_state": worst.to_dict(),
        "recovered_state": recovered.to_dict(),
        "response": {
            "minimum_drive_terminal_voltage_pu": float(timeline["drive_terminal_voltage_pu"].min()),
            "minimum_pump_speed_pu": float(timeline["pump_actual_speed_pu"].min()),
            "minimum_critical_pressure_m": float(timeline["critical_node_pressure_m"].min()),
            "minimum_delivered_water_fraction": float(timeline["delivered_water_fraction"].min()),
            "pump_power_change_percent_at_worst_pressure": 100.0
            * (
                float(worst["pump_electrical_power_mw"])
                / float(baseline["pump_electrical_power_mw"])
                - 1.0
            ),
            "line_loading_change_percentage_points_at_worst_pressure": float(
                worst["maximum_line_loading_percent"]
            )
            - float(baseline["maximum_line_loading_percent"]),
            "water_limits_recovered_after_clearance": bool(recovered["water_limits_passed"]),
        },
        "timeseries": str(
            (ELECTRICAL_SCENARIO / "electrical_supply_disturbance_timeseries.csv").relative_to(PROJECT)
        ),
        "worst_epanet_export": str(
            (ELECTRICAL_SCENARIO / "water_epanet_worst_electrical_disturbance.inp").relative_to(PROJECT)
        ),
        "worst_pandapower_export": str(
            (ELECTRICAL_SCENARIO / "power_pandapower_bidirectional.json").relative_to(PROJECT)
        ),
    }
    (ELECTRICAL_SCENARIO / "electrical_supply_disturbance_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    model_manifest_path = OUT / "model_manifest_four_sector.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    model_manifest.setdefault("solver_readiness", {})[
        "electrical_supply_disturbance_response"
    ] = summary
    model_manifest_path.write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    leak_result = run_hydraulic_leak_demonstration()
    electrical_result = run_electrical_supply_disturbance_demonstration()
    print(
        json.dumps(
            {
                "hydraulic_leak": {
                    "accepted": leak_result["accepted"],
                    "response": leak_result["response"],
                },
                "electrical_supply_disturbance": {
                    "scenario_complete": electrical_result["scenario_complete"],
                    "response": electrical_result["response"],
                },
            },
            indent=2,
        )
    )
