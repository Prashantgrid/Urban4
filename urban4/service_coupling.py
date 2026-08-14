#!/usr/bin/env python3
"""Close the generated facility-to-power interfaces on the detailed grid."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .coupling_repair import (
    declared_policy,
    phase_return_for_failure,
    terminal_failure,
    validate_interface_state,
)
from .shared_energy_assets import RELATION as SHARED_ASSET_RELATION, resolve_shared_asset_states


def _drive_speed(
    voltage_pu: np.ndarray, drive: dict[str, float] | None = None
) -> np.ndarray:
    """Apply the same declared drive law used by the disturbance adapters."""
    from .cosimulation import _motor_speed
    if drive is None:
        from .framework import load_case
        drive = load_case()["bidirectional_coupling"]["drive_model"]

    voltage = np.asarray(voltage_pu, dtype=float)
    return np.asarray([_motor_speed(float(value), 1.0, drive) for value in voltage])


def run_nominal_interface_closure(
    output: Path,
    *,
    maximum_iterations: int | None = None,
    relaxation: float | None = None,
    voltage_tolerance_pu: float = 1e-6,
    power_tolerance_mw: float = 1e-5,
    shared_asset_availability: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Iterate sector duty, electrical load and terminal voltage to closure.

    Native water, wastewater and heat results establish the required nominal
    duties.  The electrical solve returns each facility terminal voltage.  A
    VFD ride-through law maps that voltage back to motor speed and cubic pump
    power.  Under normal accepted voltages the VFD holds unit speed; disturbed
    studies use the same interface fields but re-solve the affected native
    sector model.
    """
    import pandapower as pp
    from .framework import load_case

    config = load_case()
    policy = declared_policy(config)
    drive = config["bidirectional_coupling"]["drive_model"]
    if maximum_iterations is None:
        maximum_iterations = int(config["bidirectional_coupling"]["maximum_iterations"])
    fixed_relaxation = relaxation
    net = pp.from_json(str(output / "electricity_pandapower.json"))
    interfaces = pd.read_csv(output / "coupling_interfaces.csv", low_memory=False)
    active = interfaces[
        interfaces.relation.eq("electrically_driven_facility")
    ].copy().reset_index(drop=True)
    shared = interfaces[
        interfaces.relation.eq(SHARED_ASSET_RELATION)
    ].copy().reset_index(drop=True)
    if active.empty:
        raise ValueError("No electrically driven facility interfaces found")
    bus_by_name = pd.Series(net.bus.index.to_numpy(), index=net.bus.name.astype(str))
    missing = sorted(set(active.from_id.astype(str)) - set(bus_by_name.index))
    if missing:
        raise ValueError(f"Interface buses absent from pandapower model: {missing[:5]}")
    active["pandapower_bus"] = active.from_id.astype(str).map(bus_by_name).astype(int)

    shared_states = resolve_shared_asset_states(shared, shared_asset_availability)
    if not shared.empty:
        shared_missing = sorted(set(shared.from_id.astype(str)) - set(bus_by_name.index))
        if shared_missing:
            raise ValueError(
                f"Shared-asset electrical connection points absent from pandapower model: {shared_missing[:5]}"
            )
        shared["pandapower_bus"] = (
            shared.from_id.astype(str).map(bus_by_name).astype(int)
        )
        shared_bus_by_asset = pd.Series(
            shared.pandapower_bus.to_numpy(int),
            index=shared.shared_asset_id.astype(str),
        )
        shared_states["pandapower_bus"] = (
            shared_states.shared_asset_id.astype(str)
            .map(shared_bus_by_asset)
            .astype(int)
        )
        # Positive prescribed active power is generation.  Availability-only
        # assets create no injection, so adding the interface cannot change the
        # present benchmark merely because the physical asset is recorded.
        prescribed = shared_states[
            shared_states.available.astype(bool)
            & shared_states.electrical_injection_mw.notna()
        ]
        if len(prescribed):
            pp.create_sgens(
                net,
                buses=prescribed.pandapower_bus.to_numpy(int),
                p_mw=prescribed.electrical_injection_mw.to_numpy(float),
                q_mvar=np.zeros(len(prescribed), dtype=float),
                name=("U4_SHARED_" + prescribed.shared_asset_id.astype(str)).tolist(),
            )
    contract_view = active.rename(
        columns={
            "from_id": "bus_id", "to_sector": "sector", "to_id": "asset_id",
            "nominal_power_mw": "power_mw",
        }
    )
    interface_errors = validate_interface_state(contract_view)
    if interface_errors:
        phase_return = phase_return_for_failure("interface").to_dict()
        terminal = terminal_failure(
            f"interface contract failed: {', '.join(interface_errors)}", "interface"
        ).to_dict()
        result = {
            "status": policy["terminal_status"],
            "accepted": False,
            "converged": False,
            "repair_policy": policy,
            "interface_contract_errors": interface_errors,
            "phase_return_request": phase_return,
            "terminal_failure": terminal,
        }
        (output / "coupled_nominal_interface_manifest.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    nominal = active.nominal_power_mw.to_numpy(float)
    power = nominal.copy()
    previous_voltage = np.ones(len(active), dtype=float)
    history: list[dict[str, Any]] = []
    final_voltage = previous_voltage.copy()
    final_speed = np.ones(len(active), dtype=float)
    for iteration in range(1, maximum_iterations + 1):
        if fixed_relaxation is None:
            relaxation = float(policy["relaxation_schedule"][-1]["factor"])
            for stage in policy["relaxation_schedule"]:
                if iteration <= int(stage["last_global_iteration"]):
                    relaxation = float(stage["factor"])
                    break
        else:
            relaxation = float(fixed_relaxation)
        if len(net.load) and net.load.name.astype(str).str.startswith("U4_IF_").any():
            net.load.drop(
                net.load.index[net.load.name.astype(str).str.startswith("U4_IF_")],
                inplace=True,
            )
        reactive = power * math.tan(math.acos(0.95))
        pp.create_loads(
            net,
            buses=active.pandapower_bus.to_numpy(int),
            p_mw=power,
            q_mvar=reactive,
            name=("U4_IF_" + active.interface_id.astype(str)).tolist(),
        )
        pp.runpp(
            net, algorithm="nr", init="auto", calculate_voltage_angles=False,
            max_iteration=30, tolerance_mva=1e-8,
        )
        voltage = net.res_bus.vm_pu.reindex(active.pandapower_bus).to_numpy(float)
        speed = _drive_speed(voltage, drive)
        target_power = nominal * speed**3
        updated_power = relaxation * target_power + (1.0 - relaxation) * power
        voltage_residual = float(np.max(np.abs(voltage - previous_voltage)))
        power_residual = float(np.max(np.abs(updated_power - power)))
        history.append({
            "iteration": iteration,
            "minimum_terminal_voltage_pu": float(voltage.min()),
            "total_interface_power_mw": float(updated_power.sum()),
            "maximum_voltage_residual_pu": voltage_residual,
            "maximum_power_residual_mw": power_residual,
        })
        final_voltage, final_speed = voltage, speed
        power = updated_power
        if voltage_residual <= voltage_tolerance_pu and power_residual <= power_tolerance_mw:
            break
        previous_voltage = voltage
    converged = bool(
        history[-1]["maximum_voltage_residual_pu"] <= voltage_tolerance_pu
        and history[-1]["maximum_power_residual_mw"] <= power_tolerance_mw
    )
    active["terminal_voltage_pu"] = final_voltage
    active["drive_speed_pu"] = final_speed
    active["closed_interface_power_mw"] = power
    active["energized"] = final_voltage >= 0.75
    active.to_csv(output / "coupled_facility_interface_states.csv", index=False)
    if len(shared_states):
        shared_states["electrical_connection_voltage_pu"] = (
            net.res_bus.vm_pu.reindex(shared_states.pandapower_bus).to_numpy(float)
        )
        shared_states["electrical_connection_energized"] = (
            shared_states.electrical_connection_voltage_pu >= 0.75
        )
        shared_states["availability_note"] = (
            "Common shared-asset availability is a scenario state; connection voltage is monitored but does not "
            "invent a CHP trip/dispatch rule."
        )
    shared_states.to_csv(output / "shared_energy_asset_states.csv", index=False)
    pd.DataFrame(history).to_csv(
        output / "coupled_nominal_convergence_history.csv", index=False
    )
    pp.to_json(net, str(output / "electricity_pandapower_with_interfaces.json"))
    thresholds = config["acceptance_screening"]["electricity"]
    minimum_system_voltage = float(net.res_bus.vm_pu.min())
    maximum_system_voltage = float(net.res_bus.vm_pu.max())
    maximum_line_loading = float(net.res_line.loading_percent.max())
    maximum_transformer_loading = float(net.res_trafo.loading_percent.max())
    accepted = bool(
        converged
        and active.energized.all()
        and minimum_system_voltage >= thresholds["minimum_voltage_pu"]
        and maximum_system_voltage <= thresholds["maximum_voltage_pu"]
        and maximum_line_loading <= thresholds["maximum_line_loading_percent"]
        and maximum_transformer_loading
        <= thresholds["maximum_transformer_loading_percent"]
    )
    result = {
        "status": "accepted" if accepted else policy["terminal_status"],
        "method": "under-relaxed sector-duty--pandapower--VFD fixed-point closure",
        "scope": "normal accepted native operating point",
        "converged": converged,
        "accepted": accepted,
        "repair_policy": policy,
        "relaxation_schedule_used": (
            policy["relaxation_schedule"]
            if fixed_relaxation is None
            else [{"factor": float(fixed_relaxation), "last_global_iteration": maximum_iterations}]
        ),
        "iterations": int(len(history)),
        "facility_interfaces": int(len(active)),
        "shared_conversion_assets": int(len(shared_states)),
        "shared_asset_electrical_injection_mw": float(
            shared_states.electrical_injection_mw.fillna(0.0).sum()
            if len(shared_states) else 0.0
        ),
        "shared_asset_state_file": "shared_energy_asset_states.csv",
        "shared_asset_policy": (
            "Common availability is explicit. Electrical/thermal operating points are used only when prescribed; "
            "Urban4 does not infer a CHP conversion curve."
        ),
        "nominal_interface_power_mw": float(nominal.sum()),
        "closed_interface_power_mw": float(power.sum()),
        "minimum_terminal_voltage_pu": float(final_voltage.min()),
        "minimum_system_voltage_pu": minimum_system_voltage,
        "maximum_system_voltage_pu": maximum_system_voltage,
        "maximum_line_loading_percent": maximum_line_loading,
        "maximum_transformer_loading_percent": maximum_transformer_loading,
        "all_interfaces_energized": bool(active.energized.all()),
        "native_sector_resolve_rule": (
            "If any drive_speed_pu differs from 1, the affected WNTR, SWMM or "
            "pandapipes model must be re-solved before scenario acceptance."
        ),
    }
    if not accepted:
        failure_code = "fixed_point" if not converged else "electricity"
        result["phase_return_request"] = phase_return_for_failure(failure_code).to_dict()
        result["terminal_failure"] = terminal_failure(
            "normal interface closure exhausted its declared iterations or electrical envelope",
            failure_code,
        ).to_dict()
    else:
        result["phase_return_request"] = None
        result["terminal_failure"] = None
    (output / "coupled_nominal_interface_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
