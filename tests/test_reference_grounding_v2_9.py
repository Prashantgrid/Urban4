from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.apply_v2_9_reference_grounding import dvgw_w410_hourly_peak_factor


ROOT = Path(__file__).resolve().parents[1]


def test_w410_factor_uses_the_water_operators_service_population() -> None:
    case = json.loads((ROOT / "cases" / "schweinfurt.json").read_text(encoding="utf-8"))
    population = float(
        case["official_anchors"]["drinking_water_operator_served_population_approx"]
    )
    exact = dvgw_w410_hourly_peak_factor(population)
    assert population == 100_000
    assert math.isclose(exact, 2.610228786260013, rel_tol=0.0, abs_tol=1e-12)


def test_native_peak_factor_is_the_conservative_upward_rounding() -> None:
    case = json.loads((ROOT / "cases" / "schweinfurt.json").read_text(encoding="utf-8"))
    grounding = case["reference_grounding_v2_9"]
    exact = float(grounding["water_peak_factor_w410_exact"])
    applied = float(grounding["water_peak_factor_applied"])
    assert applied == math.ceil(exact * 100.0) / 100.0 == 2.62
    assert float(case["demand_model"]["drinking_water_peak_factor"]) == applied


def test_population_boundaries_are_not_conflated() -> None:
    case = json.loads((ROOT / "cases" / "schweinfurt.json").read_text(encoding="utf-8"))
    anchors = case["official_anchors"]
    assert anchors["served_population"] == 68_799
    assert anchors["drinking_water_operator_served_population_approx"] == 100_000
    assert anchors["served_population"] != anchors["drinking_water_operator_served_population_approx"]


def test_water_peak_grounding_retains_machine_readable_primary_sources() -> None:
    case = json.loads((ROOT / "cases" / "schweinfurt.json").read_text(encoding="utf-8"))
    urls = case["reference_grounding_v2_9"]["water_peak_reference_urls"]
    assert urls["dvgw_w410"].startswith("https://www.dvgw-regelwerk.de/")
    assert urls["dvgw_w201712"].startswith("https://www.dvgw.de/")
    assert urls["operator_service_area"] == "https://www.stadtwerke-sw.de/wasser"
