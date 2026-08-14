#!/usr/bin/env python3
"""Legacy export for the reproduced v1.5.x coupled baseline.

This script intentionally preserves the historical 14-boundary heat representation
needed to reproduce the archived 27-interface numerical baseline. It is NOT the
municipality-scale v1.9.0 asset/source model. For current full-city engineering
exports, run ``build_municipal_scale_design.py`` and use
``outputs/municipal_scale_v2.5.0`` (H1 main GKS + H2 SHW Nord).
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
SERVICE = PROJECT / "outputs" / "integrated_service_resolved"
OUT = PROJECT / "outputs" / "verified_inventory_v1.5.3"
OUT.mkdir(parents=True, exist_ok=True)

cfg = json.loads((PROJECT / "cases" / "schweinfurt.json").read_text())
anchors = cfg["official_anchors"]
interfaces = pd.read_csv(SERVICE / "coupled_facility_interface_states.csv")

# ---------------------------------------------------------------------------
# Facility IDs shown in the full-city map.
# ---------------------------------------------------------------------------
map_rows: list[dict] = []

e_nodes = pd.read_csv(SERVICE / "electricity_nodes.csv")
e_source = e_nodes[e_nodes["root"].astype(bool)].iloc[0]
map_rows.append({
    "map_id": "E1", "sector": "electricity", "facility_type": "external_grid_boundary",
    "internal_id": e_source.bus_id, "lon": e_source.lon, "lat": e_source.lat,
    "evidence_role": "synthetic electrical boundary; not an installed generator rating",
})

w_nodes = pd.read_csv(SERVICE / "drinking_water_nodes.csv")
w_source = w_nodes[w_nodes.node_type.eq("reservoir")].iloc[0]
map_rows.append({
    "map_id": "W1", "sector": "drinking_water", "facility_type": "reservoir_and_head_pump_boundary",
    "internal_id": w_source.node_id, "lon": w_source.lon, "lat": w_source.lat,
    "evidence_role": "generated water-source boundary and pump interface",
})

s_nodes = pd.read_csv(SERVICE / "wastewater_nodes.csv")
outfall = s_nodes[s_nodes.node_type.eq("outfall")].iloc[0]
map_rows.append({
    "map_id": "S1", "sector": "wastewater", "facility_type": "treatment_outfall_interface",
    "internal_id": outfall.node_id, "lon": outfall.lon, "lat": outfall.lat,
    "evidence_role": "generated treatment/outfall interface",
})
lifts = s_nodes[s_nodes.node_type.eq("lift_station")].sort_values(["lat", "lon"], ascending=[False, True])
for index, row in enumerate(lifts.itertuples(), start=2):
    map_rows.append({
        "map_id": f"S{index}", "sector": "wastewater", "facility_type": "network_lift_station",
        "internal_id": row.node_id, "lon": row.lon, "lat": row.lat,
        "evidence_role": "generated network lift station",
    })

h_sources = pd.read_csv(SERVICE / "district_heating_sources.csv")
for row in h_sources.itertuples():
    map_rows.append({
        "map_id": row.map_id, "sector": "district_heating", "facility_type": "synthetic_heat_source_boundary",
        "internal_id": row.source_node, "lon": row.lon, "lat": row.lat,
        "evidence_role": row.evidence_role,
    })
facility_ids = pd.DataFrame(map_rows)
facility_ids.to_csv(OUT / "facility_map_ids.csv", index=False)

# ---------------------------------------------------------------------------
# Heat-source assignments and circulation-pump duties.
# ---------------------------------------------------------------------------
heat_pumps = interfaces[interfaces.facility_type.eq("district_heat_circulation_pump")][
    ["to_id", "nominal_power_mw", "closed_interface_power_mw", "connected_bus_voltage_kv"]
].copy()
heat_pumps["source_node"] = heat_pumps.to_id.str.replace("HPUMP_", "", regex=False)
heat_ratings = h_sources.merge(heat_pumps.drop(columns="to_id"), on="source_node", how="left")
heat_ratings["nominal_pump_power_kw"] = heat_ratings.nominal_power_mw * 1000.0
heat_ratings["closed_pump_power_kw"] = heat_ratings.closed_interface_power_mw * 1000.0
heat_ratings["assigned_annual_heat_gwh"] = heat_ratings.assigned_annual_heat_mwh / 1000.0
heat_ratings.to_csv(OUT / "heat_source_ratings.csv", index=False)

# ---------------------------------------------------------------------------
# Full-city sector scale and selected assets.
# ---------------------------------------------------------------------------
e_tr = pd.read_csv(SERVICE / "electricity_transformers.csv")
e_links = pd.read_csv(SERVICE / "electricity_links.csv")
w_links = pd.read_csv(SERVICE / "drinking_water_links.csv")
s_links = pd.read_csv(SERVICE / "wastewater_links.csv")
h_links = pd.read_csv(SERVICE / "district_heating_corridors.csv")
h_services = pd.read_csv(SERVICE / "district_heating_service_connections.csv")
s_services = pd.read_csv(SERVICE / "wastewater_service_connections.csv")

transformer_distribution = (
    e_tr.groupby(["unit_count", "unit_rating_mva"]).size().reset_index(name="count")
)
transformer_distribution_text = "; ".join(
    f"{int(r.count)}x{int(r.unit_count)}x{r.unit_rating_mva:g} MVA"
    for r in transformer_distribution.itertuples(index=False)
)
water_pump = w_links[w_links.link_type.eq("pump")].iloc[0]
water_if = interfaces[interfaces.facility_type.eq("waterworks_head_pump")].iloc[0]
wwtp_if = interfaces[interfaces.facility_type.eq("wastewater_treatment_interface")].iloc[0]
lift_ifs = interfaces[interfaces.facility_type.isin(["network_lift_pump", "property_lift_pump"])]
heat_ifs = interfaces[interfaces.facility_type.eq("district_heat_circulation_pump")]

water_mean_lps = float(anchors["drinking_water_annual_m3"]) / (365.0 * 86400.0) * 1000.0
water_peak_factor = float(cfg["demand_model"]["drinking_water_peak_factor"])
water_peak_lps = water_peak_factor * water_mean_lps
waste_annual_m3 = float(s_services.annual_m3.sum())
waste_mean_lps = waste_annual_m3 / (365.0 * 86400.0) * 1000.0

sector_rows = [
    {
        "sector": "Electricity",
        "service_and_demand_scale": f"22641 building loads; {anchors['electricity_lv_annual_mwh']/1000:.3f} GWh/a; {anchors['electricity_lv_peak_mw']:.3f} MW coincident peak",
        "source_or_facility_rating": f"E1: 20 kV external-grid boundary; no installed generator rating",
        "selected_asset_scale": f"105 MV/LV sites; {e_tr.sn_mva.sum():.2f} MVA selected transformer capacity ({transformer_distribution_text})",
        "powered_interface_count": int(len(interfaces)),
        "closed_interface_duty_mw": float(interfaces.closed_interface_power_mw.sum()),
    },
    {
        "sector": "Drinking water",
        "service_and_demand_scale": f"22641 services; {anchors['drinking_water_annual_m3']/1e6:.2f} million m3/a; {water_mean_lps:.1f}/{water_peak_lps:.1f} L/s mean/design peak",
        "source_or_facility_rating": f"W1: {water_pump.design_flow_m3s*1000:.2f} L/s head pump; {json.loads((SERVICE/'integrated_generation_manifest.json').read_text())['native_solver_results']['drinking_water']['source_head_rise_m']:.2f} m head rise; {water_if.closed_interface_power_mw*1000:.1f} kW closed duty",
        "selected_asset_scale": f"{w_links.loc[w_links.link_type.eq('distribution_main'),'length_km'].sum():.2f} km mains; {w_links.loc[w_links.link_type.eq('building_service'),'length_km'].sum():.2f} km services",
        "powered_interface_count": 1,
        "closed_interface_duty_mw": float(water_if.closed_interface_power_mw),
    },
    {
        "sector": "Wastewater",
        "service_and_demand_scale": f"22641 sanitary mappings; {waste_annual_m3/1e6:.3f} million m3/a; {waste_mean_lps:.1f} L/s mean sanitary inflow",
        "source_or_facility_rating": f"S1: treatment-process interface {wwtp_if.closed_interface_power_mw*1000:.1f} kW; S2-S3: 0.20 L/s at 3.54-3.57 m; 9 property lifts",
        "selected_asset_scale": f"{s_links.loc[s_links.link_type.eq('gravity_main'),'length_km'].sum():.2f} km gravity mains; {s_links.loc[s_links.link_type.eq('force_main'),'length_km'].sum():.3f} km network force mains",
        "powered_interface_count": int(1 + len(lift_ifs)),
        "closed_interface_duty_mw": float(wwtp_if.closed_interface_power_mw + lift_ifs.closed_interface_power_mw.sum()),
    },
    {
        "sector": "District heating",
        "service_and_demand_scale": f"841 substations; {anchors['district_heat_sales_mwh_year']/1000:.1f} GWh/a; {cfg['district_heating']['design_peak_mw_assumption']:.3f} MW_th design peak",
        "source_or_facility_rating": f"H1-H14: synthetic source boundaries; 30 MW_th planning ceiling each; assigned 0.935-19.476 MW_th",
        "selected_asset_scale": f"{h_links.loc[h_links.link_type.eq('route'),'length_km'].sum():.3f} km route + {h_links.loc[h_links.link_type.eq('building_service'),'length_km'].sum():.3f} km services; {len(h_services)} substations",
        "powered_interface_count": int(len(heat_ifs)),
        "closed_interface_duty_mw": float(heat_ifs.closed_interface_power_mw.sum()),
    },
]
pd.DataFrame(sector_rows).to_csv(OUT / "full_city_sector_scale.csv", index=False)

print(f"Wrote {OUT / 'facility_map_ids.csv'}")
print(f"Wrote {OUT / 'heat_source_ratings.csv'}")
print(f"Wrote {OUT / 'full_city_sector_scale.csv'}")
