#!/usr/bin/env python3
"""Water--power case study, part 3: a river flood as a common-cause event.

Overlays the Bavarian flood-hazard extents (LfU / WWA Bad Kissingen; HQ100,
annual exceedance probability 1 %, and HQextrem, about a 1000-year event) on the
facilities and stations of the accepted Schweinfurt models and propagates the
flood through the coupled models:

* electricity: MV/LV stations inside the extent are switched off (standard
  practice for flooded switchgear); stations cut off behind them are restored
  through the open-ring ties if the AC power flow of the MV equivalent allows it
  (same rule set as run_water_reliability.py);
* drinking water: a pumping station is out if its site is inside the extent or
  its station stays unpowered; each zone then runs on s hours of storage, after
  which the average-day EPANET PDD state gives the population equivalent (PE)
  without water;
* wastewater: flooded or unpowered lift stations stop pumping their sanitary
  inflow; a flooded treatment plant stops treating the whole sanitary flow;
* district heat: a flooded source is unavailable; the remaining capacity follows
  from the pandapipes results of the heat--power case (H2-only fraction).

Sites are treated as out of service when inside the mapped extent; site-level
protection (raised equipment, mobile barriers) is not represented and is tested
as a variant for the waterworks.

Usage: python run_flood_case.py --repo /path/to/Urban4 --flood data/LfU_Schweinfurt_HQ100_HQextrem.gpkg
                                --water results/water_power_case --heat results/heat_power_case --out results/water_power_case
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
from shapely.geometry import Point

_spec = importlib.util.spec_from_file_location("rel", Path(__file__).with_name("run_water_reliability.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

STORAGE_H = 24.0
EVENTS = {"HQ100": 0.01, "HQextrem": 0.001}   # annual exceedance probability (LfU map definitions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--flood", type=Path, required=True)
    ap.add_argument("--water", type=Path, required=True, help="outputs of run_water_power_case.py")
    ap.add_argument("--heat", type=Path, required=True, help="outputs of run_heat_power_case.py")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ms = a.repo / "outputs" / "municipal_scale_v2.5.0"
    summ0 = json.loads((a.water / "water_power_summary.json").read_text())
    heat = json.loads((a.heat / "heat_power_summary.json").read_text())
    fac = pd.read_csv(a.water / "facility_attachments.csv")
    ext = {k: gpd.read_file(a.flood, layer=k).to_crs(4326).union_all() for k in EVENTS}

    net = pp.from_json(str(ms / "electricity_municipal_pandapower.json"))
    geo = R.geo_of(net)
    name_to_bus = pd.Series(net.bus.index, index=net.bus.name)
    fac["bus"] = fac.bus_name.map(name_to_bus).astype(int)

    # ---- network structure (as in run_water_reliability.py)
    g_all = nx.Graph()
    g_all.add_edges_from(zip(net.line.from_bus, net.line.to_bus))
    g_all.add_edges_from(zip(net.trafo.hv_bus, net.trafo.lv_bus))
    root = int(net.ext_grid.bus.iloc[0])
    tree_all = nx.bfs_tree(g_all, root)
    mvb = set(int(b) for b in net.bus.index[net.bus.vn_kv >= 1.0])
    mvl = net.line[net.line.from_bus.isin(mvb)]
    g_mv = nx.Graph()
    for i, r in mvl.iterrows():
        g_mv.add_edge(int(r.from_bus), int(r.to_bus), km=float(r.length_km))
    tree = nx.bfs_tree(g_mv, root)
    for u, v in tree.edges:
        tree[u][v]["km"] = g_mv[u][v]["km"]
    stations = sorted(int(b) for b in net.trafo.hv_bus)
    st_of = {}
    for t, r in net.trafo.iterrows():
        for n in nx.descendants(tree_all, int(r.lv_bus)) | {int(r.lv_bus)}:
            st_of[n] = int(r.hv_bus)
    cust = net.load.bus.map(st_of).value_counts().reindex(stations).fillna(0).astype(int)
    fac["station_mv_bus"] = [b if b in mvb else st_of[b] for b in fac.bus]

    bk = summ0["backup_power_kw"]
    wl = fac[fac.kind.eq("water")].set_index("facility")
    facility_loads = [(wl.at["WZ1", "station_mv_bus"], bk["waterworks_duty"] / 1e3)] + \
                     [(wl.at[z, "station_mv_bus"], bk[z] / 1e3) for z in ["WZ2", "WZ3", "WZ4"]]
    m, lv_drop = R.mv_equivalent(net, facility_loads)
    route = mvl.length_km.to_numpy()
    straight = np.array([R.km(geo.at[int(r.from_bus)], geo.at[int(r.to_bus)]) for r in mvl.itertuples()])
    detour = float(route[straight > 0.02].sum() / straight[straight > 0.02].sum())
    ties, _ = R.build_ties(tree, root, stations, geo, detour)
    rest = R.Restorer(m, tree.edges, ties, R.LV_FLOOR_PU + lv_drop)

    # ---- assets
    ww_nodes = pd.read_csv(ms / "wastewater_nodes.csv")
    wwtp = ww_nodes[ww_nodes.node_type.eq("outfall")].iloc[0]
    src = pd.read_csv(ms / "district_heating_sources_municipal.csv").set_index("map_id")
    h2_bus = int(name_to_bus[heat["pump_attachment_bus"]["H2"]])
    assets = [(r.facility, "water" if r.kind == "water" else "wastewater",
               "waterworks" if r.role == "waterworks" else ("booster station" if r.kind == "water" else "lift station"),
               r.lon, r.lat) for r in fac.itertuples()]
    assets += [("WWTP", "wastewater", "treatment plant", float(wwtp.lon), float(wwtp.lat)),
               ("H1", "heat", "GKS (CHP, main DH source)", float(src.at["H1", "lon"]), float(src.at["H1", "lat"])),
               ("H2", "heat", "Spitzenheizwerk Nord", float(src.at["H2", "lon"]), float(src.at["H2", "lat"])),
               ("SOURCE", "electricity", "HV/MV supply point", *geo.at[root])]
    A = gpd.GeoDataFrame(assets, columns=["asset", "sector", "role", "lon", "lat"],
                         geometry=[Point(x, y) for *_, x, y in assets], crs=4326)
    Am = A.to_crs(25832)
    for k, u in ext.items():
        A[k] = A.geometry.within(u)
        A[f"dist_{k}_m"] = Am.geometry.distance(gpd.GeoSeries([u], crs=4326).to_crs(25832).iloc[0]).round(0)
    S = gpd.GeoDataFrame({"bus": stations}, geometry=[Point(*geo.at[b]) for b in stations], crs=4326)
    bld = pd.read_csv(a.repo / "outputs" / "building_sector_demands.csv")
    bld = bld[bld.service_eligible.astype(bool)]
    B = gpd.GeoDataFrame(bld[["building_id", "population"]], geometry=gpd.points_from_xy(bld.lon, bld.lat), crs=4326)

    ledger = bld
    wnodes = pd.read_csv(ms / "drinking_water_nodes.csv")
    zb = pd.read_csv(ms / "drinking_water_zone_boundaries.csv")
    heads = dict(zip(zb.pressure_zone_id, zb.selected_boundary_head_m))
    ws = R.WaterStates(ms / "drinking_water_municipal_average.inp", wnodes, pd.read_csv(a.repo / "outputs" / "building_sector_demands.csv"), heads)
    ls = pd.read_csv(ms / "wastewater_pump_stations_municipal.csv").set_index("map_id")
    inflow_lps = (ls.annual_sanitary_m3 / (365.0 * 86400.0) * 1000.0).to_dict()
    total_sanitary_lps = float(sum(inflow_lps.values()))

    ww_station = int(wl.at["WZ1", "station_mv_bus"])

    def energize(flooded_st):
        live = set(tree.nodes) - flooded_st
        E = nx.node_connected_component(tree.to_undirected().subgraph(live), root)
        D = live - E
        restored, closed = R.restore(rest, tree, D, E, ties, -1)
        return D, restored, closed, flooded_st | (D - restored)

    results, rows_ts = {}, []
    for ev, prob in EVENTS.items():
        u = ext[ev]
        flooded_st = set(S.bus[S.geometry.within(u)])
        # energized part after the flooded stations are switched off
        D, restored, closed, unpowered = energize(flooded_st)
        unpowered_prot = energize(flooded_st - {ww_station})[3]
        cust_flooded = int(cust.reindex(sorted(flooded_st)).sum())
        cust_cut = int(cust.reindex(sorted(s for s in D if s in cust.index)).sum())
        cust_unrestored = int(cust.reindex(sorted(s for s in (D - restored) if s in cust.index)).sum())
        fl = A.set_index("asset")[ev]
        rec = {"annual_exceedance_probability": prob,
               "stations_flooded": len(flooded_st), "through_stations_flooded": int(sum(tree.to_undirected().degree(s) > 1 for s in flooded_st)),
               "customers_at_flooded_stations": cust_flooded, "customers_cut_off": cust_cut,
               "customers_without_power_after_switching": cust_flooded + cust_unrestored,
               "customers_without_power_waterworks_station_protected":
                   int(cust.reindex(sorted(x for x in unpowered_prot if x in cust.index)).sum()),
               "ties_closed": len(closed),
               "buildings_flooded": int(B.geometry.within(u).sum()), "residents_flooded": float(B.population[B.geometry.within(u)].sum()),
               "assets_flooded": sorted(A.asset[A[ev]].tolist())}
        # water variants
        variants = {"as mapped": dict(protect=set(), backup=set(), unp=unpowered),
                    "waterworks site protected": dict(protect={"WZ1"}, backup=set(), unp=unpowered),
                    "site and supply station protected": dict(protect={"WZ1"}, backup=set(), unp=unpowered_prot),
                    "site protected + waterworks backup": dict(protect={"WZ1"}, backup={"WZ1"}, unp=unpowered)}
        rec["waterworks_supply_station_flooded"] = ww_station in flooded_st
        wres = {}
        for vname, v in variants.items():
            out = set()
            for z in ["WZ1", "WZ2", "WZ3", "WZ4"]:
                site = bool(fl[z]) and z not in v["protect"]
                power = int(wl.at[z, "station_mv_bus"]) in v["unp"] and z not in v["backup"]
                if site or power:
                    out.add(z)
            if "WZ1" in out:
                out = {"WZ1", "WZ2", "WZ3", "WZ4"}
            st_ = ws.state(frozenset(out))
            wres[vname] = {"stations_out": sorted(out), "pe_without_water_after_storage": st_["pe"],
                           "unserved_m3_per_day_after_storage": st_["u_m3h"] * 24.0,
                           "hours_until_loss": STORAGE_H if st_["pe"] > 0 else None}
            for t in np.arange(0, 72.25, 0.25):
                rows_ts.append({"event": ev, "variant": vname, "t_h": t, "pe_without_water": st_["pe"] if t >= STORAGE_H else 0.0})
        rec["water"] = wres
        # wastewater
        ls_out = [f for f in ls.index if bool(fl.get(f, False)) or int(fac.set_index("facility").at[f, "station_mv_bus"]) in unpowered]
        rec["wastewater"] = {"lift_stations_out": ls_out, "unpumped_sanitary_lps": float(sum(inflow_lps[f] for f in ls_out)),
                             "treatment_plant_flooded": bool(fl["WWTP"]),
                             "untreated_sanitary_lps": total_sanitary_lps if bool(fl["WWTP"]) else float(sum(inflow_lps[f] for f in ls_out))}
        # heat
        h1_out = bool(fl["H1"])
        h2_out = bool(fl["H2"]) or h2_bus in unpowered
        if h1_out and h2_out:
            heat_rec = {"sources_out": ["H1", "H2"], "dh_share_at_design": 0.0, "full_supply_down_to_c": None}
        elif h1_out:
            heat_rec = {"sources_out": ["H1"], "dh_share_at_design": heat["h2_only_max_fraction"],
                        "full_supply_down_to_c": heat["h2_only_threshold_c"], "with_dn400_intertie_share_at_design": heat["h2_only_dn400_max_fraction"]}
        else:
            heat_rec = {"sources_out": ["H2"] if h2_out else [], "dh_share_at_design": 1.0, "full_supply_down_to_c": -14.0}
        heat_rec["h2_attachment_powered"] = h2_bus not in unpowered
        rec["heat"] = heat_rec
        results[ev] = rec
        print(ev, json.dumps(rec, indent=1, default=float), flush=True)

    A.drop(columns="geometry").to_csv(a.out / "flood_assets.csv", index=False)
    pd.DataFrame(rows_ts).to_csv(a.out / "flood_water_timeseries.csv", index=False)
    summary = {"source": "Bayerisches Landesamt fuer Umwelt (WWA Bad Kissingen): flood-hazard extents HQ100 and HQextrem (Main, Wern, Marienbach)",
               "storage_h": STORAGE_H, "assumptions": {
                   "sites": "inside the mapped extent -> out of service for the flood duration; no site protection unless stated",
                   "stations": "flooded MV/LV stations switched off; restoration through open-ring ties with AC check",
                   "water_states": "average-day EPANET PDD after storage is spent"},
               "events": results}
    (a.out / "flood_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
