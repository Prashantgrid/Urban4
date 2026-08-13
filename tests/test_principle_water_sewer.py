from __future__ import annotations

import json
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs" / "water_sewer_principle_candidate"


@pytest.fixture(scope="module")
def manifest() -> dict:
    path = OUTPUT / "manifest.json"
    assert path.exists(), "Run `python -m urban4.principle_water_sewer` first"
    return json.loads(path.read_text(encoding="utf-8"))


def test_water_candidate_closes_registered_scale_and_structural_screens(manifest: dict) -> None:
    water = manifest["drinking_water"]
    assert water["registered_buildings"] == 22641
    assert water["access_nodes"] == 1909
    assert water["pressure_zones"] == 4
    assert water["cycle_rank"] > 0
    assert water["critical_second_feed_paths"] > 0
    assert water["inventory_gate_passed"]
    assert water["main_length_km"] == pytest.approx(341.0, rel=0.01)
    assert water["maximum_design_velocity_m_s"] <= 2.0
    assert not water["proxy_pressure_screen_passed"]
    assert water["pressure_repair"] == "explicit pressure-zone control required before native acceptance"
    assert water["native_model_created"]
    assert not water["native_solver_executed"]


def test_wastewater_candidate_closes_registered_structural_inventories(manifest: dict) -> None:
    sewer = manifest["wastewater"]
    assert sewer["registered_buildings"] == 22641
    assert sewer["inventory_gate_passed"]
    assert sewer["main_length_km"] == pytest.approx(249.0, rel=0.01)
    assert sewer["force_main_inventory_gate_passed"]
    assert sewer["force_main_length_km"] == pytest.approx(13.0, rel=0.02)
    assert abs(sewer["manhole_count_error"]) <= 50
    assert sewer["lift_station_compounds"] == 13
    assert sewer["cover_violations_after_interpolation"] == 0
    assert sewer["directed_acyclic"]
    assert sewer["all_nodes_reach_outfall"]


def test_wastewater_native_export_is_not_falsely_claimed(manifest: dict) -> None:
    sewer = manifest["wastewater"]
    assert not sewer["native_model_created"]
    assert not sewer["native_solver_executed"]
    assert "intermediate spatial layer only" in sewer["native_solver_status"]
    assert "urban4.municipal_native" in sewer["native_solver_status"]
    assert not (OUTPUT / "wastewater_manhole_candidate.inp").exists()


def test_overlap_reduction_emerges_without_an_overlap_target(manifest: dict) -> None:
    overlap = manifest["overlap"]
    assert overlap["water_shared_percent"] < overlap["baseline_water_shared_percent"]
    assert overlap["sewer_shared_percent"] < overlap["baseline_sewer_shared_percent"]
    assert overlap["length_weighted_jaccard"] < overlap["baseline_length_weighted_jaccard"]
    assert overlap["water_shared_percent"] == pytest.approx(63.9008, abs=0.02)
    assert overlap["sewer_shared_percent"] == pytest.approx(87.6735, abs=0.02)
    assert "overlap" not in manifest["drinking_water"]["configuration"]
    assert "overlap" not in manifest["wastewater"]["configuration"]


def test_building_service_ledgers_remain_complete(manifest: dict) -> None:
    counts = manifest["counts"]
    assert counts["water_services"] == 22641
    assert counts["wastewater_services"] == 22641
