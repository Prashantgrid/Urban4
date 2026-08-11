#!/usr/bin/env python3
"""Inventory calibration utilities for the integrated district-heating case.

The public contract count, annual heat sales, and reported network length are
treated as aggregate inputs.  The route accounting scope is explicit:

    one_way_network_km = route_trench_km + service_connection_km

Paired supply/return pipe length is an export quantity, not the quantity
compared with the published one-way network inventory.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import math
import pandas as pd


@dataclass(frozen=True)
class HeatInventoryTarget:
    contracts: int
    annual_heat_mwh: float
    one_way_network_km: float
    length_tolerance_fraction: float = 0.10

    def validate(self) -> None:
        if self.contracts <= 0:
            raise ValueError("contracts must be positive")
        if self.annual_heat_mwh <= 0:
            raise ValueError("annual_heat_mwh must be positive")
        if self.one_way_network_km <= 0:
            raise ValueError("one_way_network_km must be positive")
        if not 0 <= self.length_tolerance_fraction < 1:
            raise ValueError("length_tolerance_fraction must lie in [0, 1)")


def evaluate_inventory(
    *,
    connections: int,
    annual_heat_mwh: float,
    route_trench_km: float,
    service_connection_km: float,
    target: HeatInventoryTarget,
) -> dict[str, Any]:
    """Evaluate aggregate evidence closure for one generated heat candidate."""
    target.validate()
    one_way_network_km = float(route_trench_km) + float(service_connection_km)
    length_error_fraction = abs(one_way_network_km - target.one_way_network_km) / target.one_way_network_km
    contract_error_fraction = abs(int(connections) - target.contracts) / target.contracts
    heat_error_fraction = abs(float(annual_heat_mwh) - target.annual_heat_mwh) / target.annual_heat_mwh
    return {
        "connections": int(connections),
        "annual_heat_mwh": float(annual_heat_mwh),
        "route_trench_km": float(route_trench_km),
        "service_connection_km": float(service_connection_km),
        "one_way_network_km": one_way_network_km,
        "target": asdict(target),
        "contract_error_fraction": contract_error_fraction,
        "heat_error_fraction": heat_error_fraction,
        "length_error_fraction": length_error_fraction,
        "contracts_close": contract_error_fraction <= 1e-12,
        "annual_heat_closes": heat_error_fraction <= 1e-9,
        "length_within_tolerance": length_error_fraction <= target.length_tolerance_fraction,
        "passed": (
            contract_error_fraction <= 1e-12
            and heat_error_fraction <= 1e-9
            and length_error_fraction <= target.length_tolerance_fraction
        ),
        "comparison_scope": "route_plus_service_one_way",
    }


def choose_calibrated_candidate(
    candidates: pd.DataFrame,
    *,
    require_native_convergence: bool = True,
    require_coupled_closure: bool = False,
) -> pd.Series:
    """Choose the closest admissible candidate without hiding evidence roles.

    Required columns are ``connections``, ``delivered_heat_mwh``,
    ``route_trench_km``, ``service_connection_km``, and
    ``inventory_length_relative_error``.  Native and coupled status may be
    required by the caller.  The function never treats the selected length as
    validation: it is an imposed calibration criterion.
    """
    required = {
        "connections", "delivered_heat_mwh", "route_trench_km",
        "service_connection_km", "inventory_length_relative_error",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise KeyError(f"candidate table missing columns: {missing}")
    eligible = candidates.copy()
    if require_native_convergence:
        if "native_solver_converged" not in eligible:
            raise KeyError("native_solver_converged is required")
        eligible = eligible[eligible.native_solver_converged.astype(bool)]
    if require_coupled_closure:
        if "coupled_closure_passed" not in eligible:
            raise KeyError("coupled_closure_passed is required")
        eligible = eligible[eligible.coupled_closure_passed.astype(bool)]
    if eligible.empty:
        raise ValueError("no candidate satisfies the requested acceptance level")
    order = [
        "inventory_length_relative_error",
        "minimum_heat_density_mwh_ha",
        "selection_policy",
    ]
    order = [column for column in order if column in eligible.columns]
    return eligible.sort_values(order, kind="mergesort").iloc[0]
