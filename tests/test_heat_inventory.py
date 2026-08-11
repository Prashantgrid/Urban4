from __future__ import annotations

import pandas as pd
import pytest

from urban4.heat_inventory import (
    HeatInventoryTarget,
    choose_calibrated_candidate,
    evaluate_inventory,
)


def test_accepted_coupled_baseline_uses_comparable_one_way_length() -> None:
    target = HeatInventoryTarget(
        contracts=841,
        annual_heat_mwh=87500.0,
        one_way_network_km=52.0,
        length_tolerance_fraction=0.10,
    )
    result = evaluate_inventory(
        connections=841,
        annual_heat_mwh=87500.0,
        route_trench_km=22.22977420811027,
        service_connection_km=25.795015658326815,
        target=target,
    )
    assert result["one_way_network_km"] == pytest.approx(48.02478986643708)
    assert result["length_error_fraction"] == pytest.approx(0.07644634872236388)
    assert result["passed"]


def test_compact_route_aware_candidate_is_rejected_by_inventory_gate() -> None:
    target = HeatInventoryTarget(841, 87500.0, 52.0, 0.10)
    result = evaluate_inventory(
        connections=841,
        annual_heat_mwh=87500.0,
        route_trench_km=8.676886,
        service_connection_km=18.816869,
        target=target,
    )
    assert not result["length_within_tolerance"]
    assert not result["passed"]


def test_native_inventory_candidate_is_closest() -> None:
    frame = pd.DataFrame(
        [
            {
                "selection_policy": "demand_ranked",
                "minimum_heat_density_mwh_ha": 600.0,
                "connections": 841,
                "delivered_heat_mwh": 87500.0,
                "route_trench_km": 34.81053868961525,
                "service_connection_km": 18.912374376297972,
                "inventory_length_relative_error": abs(53.72291306591322 - 52.0) / 52.0,
                "native_solver_converged": True,
                "coupled_closure_passed": False,
            },
            {
                "selection_policy": "demand_ranked",
                "minimum_heat_density_mwh_ha": 800.0,
                "connections": 841,
                "delivered_heat_mwh": 87500.0,
                "route_trench_km": 22.22977420811027,
                "service_connection_km": 25.795015658326815,
                "inventory_length_relative_error": abs(48.02478986643708 - 52.0) / 52.0,
                "native_solver_converged": True,
                "coupled_closure_passed": True,
            },
        ]
    )
    native = choose_calibrated_candidate(frame)
    coupled = choose_calibrated_candidate(
        frame, require_native_convergence=True, require_coupled_closure=True
    )
    assert native.minimum_heat_density_mwh_ha == 600.0
    assert coupled.minimum_heat_density_mwh_ha == 800.0
