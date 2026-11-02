from __future__ import annotations

import pandas as pd
from shapely.geometry import LineString

from urban4.external_validation import (
    corridor_null_distribution,
    symmetric_corridor_coverage,
)
from urban4.framework import load_case
from urban4.service_topologies import _lv_local_coincident_peak_kw


def test_mv_lv_withdrawals_are_not_transformer_count() -> None:
    cfg = load_case()
    anchors = cfg["official_anchors"]
    assert anchors["electricity_mv_lv_withdrawal_points"] == 105
    assert "not a utility transformer" in cfg["data_semantics"]["electricity_mv_lv_withdrawal_points"].lower()


def test_local_residential_coincidence_is_applied_to_branch_households() -> None:
    frame = pd.DataFrame(
        {
            "service_class": ["residential"],
            "household_count": [30],
            "electricity_peak_kw": [20.0],
        }
    )
    expected = 14.5 * 30 * (0.07 + 0.93 * 30 ** (-0.75))
    assert abs(_lv_local_coincident_peak_kw(frame) - expected) < 1e-9
    assert _lv_local_coincident_peak_kw(frame) > frame["electricity_peak_kw"].sum()


def test_water_pressure_contract_is_single_normal_band_with_event_pdd() -> None:
    cfg = load_case()
    water = cfg["acceptance_screening"]["drinking_water"]
    assert water["minimum_pressure_m"] == 27.5
    assert water["maximum_pressure_m"] == 70.0
    assert water["event_pdd_minimum_pressure_m"] == 0.0
    assert water["event_pdd_required_pressure_m"] == 27.5


def test_symmetric_corridor_metric_reports_both_directions_and_null() -> None:
    installed = [LineString([(0, 0), (100, 0)])]
    generated = [LineString([(0, 5), (80, 5)])]
    roads = [
        LineString([(0, 0), (100, 0)]),
        LineString([(0, 5), (100, 5)]),
        LineString([(0, 20), (100, 20)]),
    ]
    coverage = symmetric_corridor_coverage(installed, generated, 5.0)
    assert 0.79 <= coverage.installed_to_generated <= 0.81
    assert coverage.generated_to_installed == 1.0
    result = corridor_null_distribution(
        installed, generated, roads, 5.0, repetitions=8, seed=7
    )
    assert "installed_to_generated" in result
    assert "generated_to_installed" in result
    assert result["repetitions"] == 8
