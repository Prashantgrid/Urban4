#!/usr/bin/env python3
"""Verify the declared voltage-to-drive contract for three powered sectors.

This is an interface response envelope, not a fault, protection, or resilience
scenario.  It exercises one generated drinking-water pump, wastewater lift,
and district-heat circulation-pump record through the same drive law.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cosimulation import _motor_speed
from .framework import load_case


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs" / "integrated_service_resolved"


def run_interface_response_envelope() -> dict[str, object]:
    config = load_case()
    drive = config["bidirectional_coupling"]["drive_model"]
    interfaces = pd.read_csv(OUTPUT / "coupling_interfaces.csv")
    active = interfaces[interfaces.relation.eq("electrically_driven_facility")].copy()
    selectors = {
        "drinking_water": active.to_sector.eq("drinking_water"),
        "wastewater": active.facility_type.eq("network_lift_pump"),
        "district_heating": active.facility_type.eq("district_heat_circulation_pump"),
    }
    voltages = np.round(np.arange(0.75, 1.051, 0.01), 3)
    rows = []
    selected_interfaces = []
    for sector, selector in selectors.items():
        candidates = active[selector]
        if candidates.empty:
            raise ValueError(f"No generated powered interface for {sector}")
        interface = candidates.nlargest(1, "nominal_power_mw").iloc[0]
        selected_interfaces.append(str(interface.interface_id))
        for voltage in voltages:
            speed = _motor_speed(float(voltage), 1.0, drive)
            rows.append({
                "sector": sector,
                "interface_id": interface.interface_id,
                "asset_id": interface.to_id,
                "bus_id": interface.from_id,
                "terminal_voltage_pu": voltage,
                "drive_speed_pu": speed,
                "normalized_flow_or_capacity": speed,
                "normalized_head_or_pressure_capability": speed**2,
                "normalized_electrical_duty": speed**3,
                "nominal_power_mw": float(interface.nominal_power_mw),
                "available": bool(speed > 0.0),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "three_sector_interface_response_envelope.csv", index=False)
    result = {
        "method": "declared steady voltage-to-drive adapter with pump affinity laws",
        "scope": "interface contract verification; not a fault, protection, reliability, or transient study",
        "sectors": list(selectors),
        "interfaces": selected_interfaces,
        "voltage_range_pu": [float(voltages.min()), float(voltages.max())],
        "drive_model": drive,
        "checks": {
            "all_three_sectors_present": frame.sector.nunique() == 3,
            "nominal_voltage_available": bool(frame[frame.terminal_voltage_pu.eq(1.0)].available.all()),
            "below_trip_unavailable": bool(
                (~frame[frame.terminal_voltage_pu.lt(drive["trip_voltage_pu"])].available).all()
            ),
            "affinity_laws_closed": bool(np.allclose(
                frame.normalized_electrical_duty,
                frame.normalized_flow_or_capacity**3,
            )),
        },
    }
    result["passed"] = all(result["checks"].values())
    (OUTPUT / "three_sector_interface_response_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run_interface_response_envelope(), indent=2))
