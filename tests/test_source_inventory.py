from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "verified_inventory_v1.5.3"


def test_map_ids_are_unique_and_complete() -> None:
    data = pd.read_csv(OUT / "facility_map_ids.csv")
    expected = {"E1", "W1", "S1", "S2", "S3"} | {f"H{i}" for i in range(1, 15)}
    assert set(data["map_id"]) == expected
    assert data["map_id"].is_unique


def test_heat_source_assignments_close_public_portfolio() -> None:
    data = pd.read_csv(OUT / "heat_source_ratings.csv")
    assert list(data["map_id"]) == [f"H{i}" for i in range(1, 15)]
    assert (data["planning_capacity_mw_th"] == 30.0).all()
    assert data["assigned_peak_mw_th"].sum() == pytest.approx(54.6875, abs=1e-5)
    assert data["assigned_annual_heat_mwh"].sum() == pytest.approx(87500.0, abs=1e-3)
    assert int(data["customer_count"].sum()) == 841
    assert data["closed_pump_power_kw"].sum() == pytest.approx(81.795, abs=1e-3)


def test_full_city_scale_export_has_all_sectors() -> None:
    data = pd.read_csv(OUT / "full_city_sector_scale.csv")
    assert set(data["sector"]) == {
        "Electricity", "Drinking water", "Wastewater", "District heating"
    }
    assert data["powered_interface_count"].sum() == 54  # electricity receives all 27; originating sectors total 27
    originating = data[data["sector"] != "Electricity"]
    assert int(originating["powered_interface_count"].sum()) == 27
