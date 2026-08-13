from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'municipal_scale_v2.5.0'

def test_waterworks_is_municipal_scale():
    w=pd.read_csv(OUT/'drinking_water_station_municipal.csv').iloc[0]
    assert int(w.installed_units)==4
    assert float(w.selected_motor_kw)==315.0
    assert float(w.firm_nameplate_kw_with_one_unavailable)==945.0
    assert 170 <= float(w.average_total_flow_lps) <= 185
    assert 460 <= float(w.design_peak_flow_lps) <= 470

def test_wastewater_uses_13_duty_standby_pumpworks():
    s=pd.read_csv(OUT/'wastewater_pump_stations_municipal.csv')
    b=pd.read_csv(ROOT/'outputs'/'building_sector_demands.csv')
    assert len(s)==13
    assert int(s.registered_buildings.sum())==int(b.wastewater_connected.astype(bool).sum())
    assert (s.selected_motor_kw >= 3.0).all()
    assert (s.selected_motor_kw <= 45.0).all()
    assert (s.force_main_velocity_m_s >= 0.6 - 1e-9).all()
    assert (s.active_swmm_pump_objects == 1).all()
    assert (s.installed_pump_units == 2).all()
    assert (s.standby_pump_units == 1).all()
    assert (s.installed_nameplate_kw == 2*s.selected_motor_kw).all()

def test_municipal_water_is_natively_accepted_after_logged_bounded_repair():
    import json
    m=json.loads((OUT/'drinking_water_native_manifest.json').read_text(encoding='utf-8'))
    repairs=pd.read_csv(OUT/'drinking_water_repair_log.csv')
    assert m['status']=='accepted' and m['release_gate_passed']
    assert m['native_solver_executed']
    assert m['peak_minimum_service_pressure_m'] >= 27.5
    assert m['average_maximum_service_pressure_m'] <= 70.0 + 1e-6
    assert m['peak_maximum_velocity_m_s'] <= 2.0
    assert m['minimum_delivered_demand_fraction'] >= 0.999
    assert len(repairs)==m['repair_actions']
    assert repairs.threshold_unchanged.astype(bool).all()

def test_municipal_wastewater_has_exactly_13_executed_station_systems():
    import json
    m=json.loads((OUT/'wastewater_station_native_manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='accepted' and m['release_gate_passed']
    assert m['native_solver_executed']
    assert m['station_wet_wells']==m['active_swmm_pump_objects']==m['force_mains']==13
    assert m['installed_pump_units']==26
    assert m['standby_pump_units']==13
    assert abs(m['continuity_error_percent']) <= 1.0
    assert m['nonconverging_steps_percent'] <= 0.1
    assert m['flooding_loss_percent'] <= 0.001

def test_heat_uses_documented_gks_and_shw_nord_sources_with_real_temperature_classes():
    h=pd.read_csv(OUT/'district_heating_sources_municipal.csv')
    a=pd.read_csv(OUT/'district_heating_connectivity_audit.csv').iloc[0]
    p=pd.read_csv(OUT/'district_heating_temperature_profiles.csv')
    assert set(h.map_id)=={'H1','H2'}
    assert int(a.documented_supply_boundaries)==2
    assert int(a.customer_substations)==841
    assert float(a.source_reachability_fraction)==1.0
    assert int(a.active_network_components)==1
    assert abs(float(a.comparable_network_km)-52.0)/52.0 < 0.10
    assert abs(float(h.assigned_peak_mw_th.sum())-54.6875) < 1e-3
    assert set(p.temperature_class)=={'HT120','MT110','LT85'}
    assert set(p.max_supply_c)=={120.0,110.0,85.0}
    assert (p.max_return_c==50.0).all()

def test_municipal_heat_is_natively_accepted_without_absolute_slack_claim():
    import json
    m=json.loads((OUT/'district_heating_native_manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='accepted' and m['release_gate_passed'] and m['converged']
    assert m['native_solver_executed']
    assert m['maximum_native_velocity_m_s'] <= 1.5
    assert m['maximum_catalogue_pressure_gradient_pa_m'] <= 100.0
    assert m['maximum_source_screening_differential_pressure_bar'] <= 10.0
    assert 'slack_pressure_bar_not_interpreted_as_municipal_absolute_pressure' in m

def test_municipal_manifest_uses_the_frozen_phase_i_ledger():
    import json
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    b=pd.read_csv(ROOT/'outputs'/'building_sector_demands.csv')
    assert manifest['input_building_ledger']=='outputs/building_sector_demands.csv'
    assert int(manifest['service_eligible_buildings'])==int(b.service_eligible.astype(bool).sum())
    assert int(manifest['registered_wastewater_buildings'])==int(b.wastewater_connected.astype(bool).sum())

def test_transformer_portfolio_closes_public_scale():
    e=pd.read_csv(OUT/'electricity_transformers_municipal.csv')
    assert len(e)==105
    total=float(e.municipal_selected_capacity_mva.sum())
    assert abs(total-216.71)/216.71 < 0.005
    assert e.municipal_selected_capacity_mva.min() >= 0.63
    assert e.municipal_selected_capacity_mva.max() <= 4.0

def test_municipal_electricity_is_natively_accepted_after_ac_power_flow():
    import json
    m=json.loads((OUT/'electricity_native_manifest.json').read_text(encoding='utf-8'))
    assert m['status']=='accepted' and m['release_gate_passed'] and m['converged']
    assert m['native_solver_executed']
    assert m['minimum_voltage_pu'] >= 0.90
    assert m['maximum_voltage_pu'] <= 1.10
    assert m['maximum_line_loading_percent'] <= 100.0
    assert m['maximum_transformer_loading_percent'] <= 100.0
    assert m['maximum_lv_path_drop_pu'] <= 0.055
    assert not m['thresholds_relaxed']

def test_released_service_resolved_interface_contract_is_complete():
    service=ROOT/'outputs'/'integrated_service_resolved'
    interfaces=pd.read_csv(service/'coupling_interfaces.csv')
    states=pd.read_csv(service/'coupled_facility_interface_states.csv')
    ledger=pd.read_csv(service/'common_building_service_ledger.csv')
    assert len(ledger)==26434
    assert int(ledger.service_eligible.astype(bool).sum())==22641
    assert int((interfaces.relation=='delivered_water_to_sanitary_inflow').sum())==22641
    assert int((interfaces.relation=='electrically_driven_facility').sum())==27
    assert len(states)==27
    assert states.energized.astype(bool).all()

def test_heat_public_operating_sheets_and_pipe_pressure_classes_are_exported():
    a=pd.read_csv(OUT/'district_heating_published_area_profiles.csv')
    c=pd.read_csv(OUT/'district_heating_corridors_municipal.csv')
    assert set(a.area)=={'B1/B2','F1-Sued','Maintal','V6'}
    assert set(a.max_supply_c)=={120.0,110.0,85.0}
    assert set(a.max_normal_return_c)=={50.0}
    assert a.source_url.str.startswith('https://www.stadtwerke-sw.de/').all()
    assert set(c.required_pressure_class).issubset({'PN10','PN16','PN25'})
    assert (c.required_supply_temperature_c.between(85.0,120.0)).all()
