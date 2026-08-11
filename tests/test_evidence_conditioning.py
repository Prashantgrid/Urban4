from pathlib import Path
import json, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"municipal_scale_v2.5.0"

def test_no_stale_generic_dh_temperature_anchor():
    case=json.load(open(ROOT/"cases"/"schweinfurt.json"))
    off=case["official_anchors"]
    assert "district_heat_supply_temperature_c" not in off
    assert "district_heat_return_temperature_c" not in off
    assert off["district_heat_temperature_evidence"]=="published_area_operating_envelopes"

def test_observed_dh_temperature_envelopes_drive_outputs():
    p=pd.read_csv(OUT/"district_heating_temperature_profiles.csv")
    s=pd.read_csv(OUT/"district_heating_services_municipal.csv")
    assert set(p.max_supply_c)=={85.0,110.0,120.0}
    assert set(p.min_supply_c)=={65.0,70.0}
    assert (p.max_return_c==50.0).all()
    assert set(s.design_supply_temperature_c).issubset({85.0,110.0,120.0})
    assert set(s.minimum_supply_temperature_c).issubset({65.0,70.0})
    assert (s.design_return_temperature_c==50.0).all()

def test_dh_sources_are_documented_boundaries_not_fourteen_synthetic_plants():
    h=pd.read_csv(OUT/"district_heating_sources_municipal.csv")
    a=pd.read_csv(OUT/"district_heating_connectivity_audit.csv").iloc[0]
    assert list(h.map_id)==["H1","H2"]
    assert int(a.active_network_components)==1
    assert int(a.source_reachable_substations)==841
    assert float(a.source_reachability_fraction)==1.0

def test_secondary_dh_source_routing_is_coarse_prior_not_exact_fuel_ratio():
    case=json.load(open(ROOT/"cases"/"schweinfurt.json"))
    dh=case["district_heating"]
    assert float(dh["h2_proxy_peak_share_cap"])==0.30
    assert "dispatch" in dh["h2_proxy_peak_share_cap_note"].lower()
    a=pd.read_csv(OUT/"district_heating_connectivity_audit.csv").iloc[0]
    assert 0.25 <= float(a.H2_assigned_peak_share) <= 0.30 + 1e-9


def test_dh_source_station_screening_uses_declared_design_envelope():
    h=pd.read_csv(OUT/"district_heating_sources_municipal.csv")
    assert set(h.map_id)=={"H1","H2"}
    assert (h.screening_pressure_gradient_pa_m==50.0).all()
    assert (h.screening_local_loss_factor==1.3).all()
    assert (h.terminal_dp_bar==0.5).all()
    assert h.pump_lift_screen_pass.all()
    assert dict(zip(h.map_id,h.selected_motor_kw_each))=={"H1":110.0,"H2":45.0}
