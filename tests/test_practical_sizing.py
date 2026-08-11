from __future__ import annotations

import math

from urban4.practical_sizing import (
    heat_circulation_estimate,
    heat_source_capacity_class,
    next_size,
    sewage_lift_estimate,
    water_station_estimate,
)


def test_next_size_rounds_up() -> None:
    assert next_size(1.31, (1.1, 1.3, 1.5)) == 1.5


def test_water_station_has_firm_peak_capacity() -> None:
    result = water_station_estimate(
        normal_flow_lps=130.0,
        peak_flow_lps=260.0,
        head_m=92.93,
    )
    assert result.selected_motor_kw == 200.0
    assert result.installed_units == 3
    assert result.duty_units_at_peak == 2
    assert result.firm_nameplate_kw == 400.0


def test_sewage_lift_uses_cleansing_flow_and_practical_motor_floor() -> None:
    result = sewage_lift_estimate(existing_static_head_m=3.55)
    assert 6.2 < result.design_flow_lps < 6.4
    assert result.selected_motor_kw >= 1.3
    assert result.installed_units == 2


def test_heat_path_increases_pump_size() -> None:
    short = heat_circulation_estimate(peak_mw_th=5.0, critical_one_way_route_km=0.2)
    long = heat_circulation_estimate(peak_mw_th=5.0, critical_one_way_route_km=2.0)
    assert long['estimated_electrical_duty_kw'] > short['estimated_electrical_duty_kw']
    assert long['selected_motor_kw'] >= short['selected_motor_kw']


def test_heat_source_capacity_keeps_20_percent_margin() -> None:
    assert heat_source_capacity_class(19.476) == 25.0
