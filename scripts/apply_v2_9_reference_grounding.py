#!/usr/bin/env python3
"""Apply the evidence updates used for the Urban4 v2.9 rerun.

This script is deliberately narrow.  It changes only inputs for which the
reference audit identified a stronger basis than the v2.8 case:

* drinking-water peak hour factor: DVGW W 410 population curve;
* normal-operation drive response: no continuous voltage-to-speed droop.

It does not replace unknown pump curves or efficiencies with new guessed
values.  Those remain explicitly labelled screening parameters until a real
manufacturer curve or utility record is supplied.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "cases" / "schweinfurt.json"


def dvgw_w410_hourly_peak_factor(population: float) -> float:
    """Population-based hourly peak factor for service areas above 1,000 people.

    The functional curve is the DVGW W 410 dimensioning relation.  The later
    DVGW/TZW W 201712 study found that the W 410 population curves can remain a
    dimensioning aid where utility-specific measured peak data are unavailable,
    while noting that they tend to be conservative for smaller service areas.
    """
    if population <= 1000:
        raise ValueError("The W 410 population curve used here requires >1000 inhabitants")
    return 18.1 * population ** (-0.1682)


def main() -> None:
    cfg = json.loads(CASE.read_text(encoding="utf-8"))
    population = float(cfg["official_anchors"]["served_population"])
    peak_factor = dvgw_w410_hourly_peak_factor(population)

    demand = cfg.setdefault("demand_model", {})
    previous_factor = float(demand.get("drinking_water_peak_factor", math.nan))
    demand["drinking_water_peak_factor"] = peak_factor
    demand["drinking_water_peak_method"] = "DVGW W 410 population-based hourly peak factor"
    demand["drinking_water_peak_basis"] = (
        "DVGW W 410 dimensioning curve; applicability checked against DVGW/TZW project W 201712. "
        "Use measured utility peak-hour data when available."
    )

    coupling = cfg["bidirectional_coupling"]
    drive = coupling["drive_model"]
    previous_droop = float(drive.get("voltage_speed_droop", math.nan))
    drive["voltage_speed_droop"] = 0.0
    drive["normal_operation_contract"] = (
        "Above the declared availability/recovery threshold, an energized variable-speed drive follows the explicit "
        "speed command; normal terminal-voltage variation does not create an invented speed droop."
    )
    drive["publication_scope"] = (
        "The manuscript uses binary electrical availability for the feeder-outage event. The archived service-voltage "
        "depression demonstration is not used as evidence for an equipment-specific undervoltage response."
    )

    practical = cfg.get("practical_design_envelopes", {}).get("drinking_water", {})
    if practical:
        practical["station_configuration"] = (
            "firm-capacity screening at the DVGW W 410 peak-hour demand; selected pump efficiency remains a "
            "screening input until a manufacturer curve is supplied"
        )

    cfg["reference_grounding_v2_9"] = {
        "applied": True,
        "water_peak_factor_previous": previous_factor,
        "water_peak_factor_w410": peak_factor,
        "water_peak_formula": "f_h = 18.1 * E^(-0.1682), E = served population",
        "water_peak_reference": "DVGW W 410; DVGW/TZW W 201712",
        "voltage_speed_droop_previous": previous_droop,
        "voltage_speed_droop_used": 0.0,
        "pump_efficiency_policy": (
            "No new hydraulic pump efficiency is invented. Existing values remain screening inputs and are not "
            "reported as measured or manufacturer-certified efficiencies."
        ),
        "wastewater_grade_policy": (
            "The legacy minimum_gravity_grade is a reduced-model topology/sensitivity setting, not a universal DWA "
            "design criterion. Municipality-scale sewers are sized by Manning capacity on generated slopes and checked "
            "in SWMM; DWA-A 110 is cited for the hydraulic-performance basis."
        ),
    }

    CASE.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cfg["reference_grounding_v2_9"], indent=2))


if __name__ == "__main__":
    main()
