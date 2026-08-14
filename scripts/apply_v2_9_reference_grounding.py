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
    anchors = cfg["official_anchors"]
    population = float(anchors["drinking_water_operator_served_population_approx"])
    exact_peak_factor = dvgw_w410_hourly_peak_factor(population)
    applied_peak_factor = math.ceil(exact_peak_factor * 100.0) / 100.0

    demand = cfg.setdefault("demand_model", {})
    previous_grounding = cfg.get("reference_grounding_v2_9", {})
    previous_factor = float(
        previous_grounding.get(
            "water_peak_factor_previous",
            demand.get("drinking_water_peak_factor", math.nan),
        )
    )
    demand["drinking_water_peak_factor_exact"] = exact_peak_factor
    demand["drinking_water_peak_factor"] = applied_peak_factor
    demand["drinking_water_peak_method"] = "DVGW W 410 population-based hourly peak factor"
    demand["drinking_water_peak_basis"] = (
        "DVGW W 410 dimensioning curve evaluated at the operator-reported water-service population; "
        "applicability checked against DVGW/TZW project W 201712. The exact result is rounded upward "
        "to two decimals for the retained EPANET design-hour check. Use measured utility peak-hour "
        "data when available."
    )

    coupling = cfg["bidirectional_coupling"]
    drive = coupling["drive_model"]
    previous_droop = float(
        previous_grounding.get(
            "voltage_speed_droop_previous",
            drive.get("voltage_speed_droop", math.nan),
        )
    )
    drive["voltage_speed_droop"] = 0.0
    drive["normal_operation_contract"] = (
        "Above the declared availability/recovery threshold, an energized variable-speed drive follows the explicit "
        "speed command; normal terminal-voltage variation does not create an invented speed droop."
    )
    drive["validation_scope"] = (
        "The feeder-outage result uses binary electrical availability. The archived service-voltage "
        "depression demonstration is not used as evidence for an equipment-specific undervoltage response."
    )

    practical = cfg.get("practical_design_envelopes", {}).get("drinking_water", {})
    if practical:
        practical["station_configuration"] = (
            f"four identical pumps: three duty at the {applied_peak_factor:.2f}-times DVGW W 410 "
            "design peak and one standby; selected pump efficiency remains a screening input until "
            "a manufacturer curve is supplied"
        )

    cfg["reference_grounding_v2_9"] = {
        "applied": True,
        "water_peak_factor_previous": previous_factor,
        "water_operator_served_population_approx": int(population),
        "water_peak_factor_w410_exact": exact_peak_factor,
        "water_peak_factor_applied": applied_peak_factor,
        "water_pipe_presizing_factor_applied": applied_peak_factor,
        "water_peak_rounding": "upward to two decimals for the retained native EPANET check",
        "water_factor_consistency_policy": (
            "Use the same applied factor for building-ledger peak flows, catalogue pipe pre-sizing, "
            "and the native EPANET design-hour check. Do not retain pre-sized pipes from an earlier factor."
        ),
        "water_peak_formula": "f_h = 18.1 * E^(-0.1682), E = operator-served water population",
        "water_peak_reference": (
            "DVGW W 410; DVGW/TZW W 201712; Stadtwerke Schweinfurt water service page"
        ),
        "water_peak_reference_urls": {
            "dvgw_w410": "https://www.dvgw-regelwerk.de/technische-regel/w-410/51a6b5",
            "dvgw_w201712": (
                "https://www.dvgw.de/themen/forschung-und-innovation/forschungsprojekte/"
                "dvgw-forschungsprojekt-spitzenverbrauch"
            ),
            "operator_service_area": "https://www.stadtwerke-sw.de/wasser",
        },
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
