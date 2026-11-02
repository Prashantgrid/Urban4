#!/usr/bin/env python3
"""Water--power case study: single MV-line outages and their water consequences.

Every one of the 407 MV line segments of the accepted municipality-scale
electricity model is taken out of service in turn (radial network, state
before switching or repair).  The de-energized part of the network is found
from the radial tree; pumped water and wastewater facilities attached to it
lose power.  Consequences are evaluated in the native solvers:

* drinking water: EPANET 2.2 (via WNTR) with pressure-dependent demand at the
  design hour.  The waterworks feeds zone WZ1; the booster stations of WZ2--WZ4
  take suction from the waterworks system (no storage is represented).  If the
  waterworks is unpowered, no zone is supplied.  If a booster is unpowered, its
  zone is supplied through a bypass at the waterworks-zone head (upper bound)
  or not at all (lower bound, reported as sensitivity).
* wastewater: the average dry-weather sanitary inflow of the catchments whose
  lift station is unpowered (from the accepted SWMM station model inputs); this
  sewage can no longer be pumped and must be held in the network or overflows.

Mitigation variants: backup generation at the waterworks (M1) and at all four
water stations (M2).

Usage: python run_water_power_case.py --repo /path/to/Urban4 --out results/
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import wntr

P_MIN_M, P_REQ_M, P_EXP = 0.0, 27.5, 0.5     # pressure-dependent demand (lower edge of the normal band)
BYPASS_SOURCE = "WZ1"                           # boosters draw from the waterworks zone


def geo_of(net):
    return net.bus.geo.map(lambda g: json.loads(g)["coordinates"] if isinstance(g, str) else g["coordinates"])


def nearest_bus(geo, candidates, lon, lat):
    xs = np.array([geo.at[b][0] for b in candidates]); ys = np.array([geo.at[b][1] for b in candidates])
    d = ((xs - lon) * math.cos(math.radians(lat))) ** 2 + (ys - lat) ** 2
    i = int(np.argmin(d))
    return int(candidates[i]), float(math.sqrt(d[i]) * 111.2)


# ---------------------------------------------------------------- water
class WaterModel:
    def __init__(self, inp: Path, nodes: pd.DataFrame, ledger: pd.DataFrame, heads: dict):
        self.inp = inp
        self.heads = heads
        svc = nodes[nodes.node_type.eq("building_service")][["node_id", "building_id", "pressure_zone_id"]]
        self.svc = svc.merge(ledger[["building_id", "population"]], on="building_id", how="left").fillna({"population": 0.0})
        self.cache = {}

    def solve(self, out: frozenset, bypass: bool) -> dict:
        key = (out, bypass)
        if key in self.cache:
            return self.cache[key]
        wn = wntr.network.WaterNetworkModel(str(self.inp))
        wn.options.hydraulic.demand_model = "PDD"
        wn.options.hydraulic.minimum_pressure = P_MIN_M
        wn.options.hydraulic.required_pressure = P_REQ_M
        wn.options.hydraulic.pressure_exponent = P_EXP
        wn.options.time.duration = 0
        unsupplied_zones = set()
        if BYPASS_SOURCE in out:                      # waterworks lost -> whole system unsupplied
            unsupplied_zones = {"WZ1", "WZ2", "WZ3", "WZ4"}
        else:
            for z in out:
                if bypass:
                    wn.get_node(f"{z}_BOUNDARY").base_head = min(self.heads[BYPASS_SOURCE], self.heads[z])
                else:
                    unsupplied_zones.add(z)
        for z in unsupplied_zones:
            link = wn.get_link(f"{z}_CONTROLLED_BOUNDARY")
            link.initial_status = wntr.network.LinkStatus.Closed
        demand_ref = self._reference_demand()
        if len(unsupplied_zones) == 4:
            delivered = pd.Series(0.0, index=demand_ref.index)
            pressure = pd.Series(np.nan, index=demand_ref.index)
        else:
            res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(Path(tempfile.mkdtemp()) / "w"))
            delivered = res.node["demand"].iloc[0].reindex(demand_ref.index).fillna(0.0).clip(lower=0.0)
            pressure = res.node["pressure"].iloc[0].reindex(demand_ref.index)
            for z in unsupplied_zones:        # closed zone: no source, force zero delivery
                ids = self.svc.loc[self.svc.pressure_zone_id.eq(z), "node_id"]
                delivered.loc[ids] = 0.0
        frac = (delivered / demand_ref.replace(0, np.nan)).fillna(1.0).clip(0.0, 1.0)
        pop = self.svc.set_index("node_id").population.reindex(demand_ref.index).fillna(0.0)
        out_d = {
            "unserved_lps": float((demand_ref - delivered).clip(lower=0).sum() * 1000.0),
            "unserved_pct": float(100.0 * (demand_ref - delivered).clip(lower=0).sum() / demand_ref.sum()),
            "pop_equiv_without_water": float((pop * (1.0 - frac)).sum()),
            "buildings_below_half": int((frac < 0.5).sum()),
            "min_pressure_supplied_m": float(pressure[frac > 0.0].min()) if (frac > 0).any() else float("nan"),
        }
        self.cache[key] = out_d
        return out_d

    _ref = None

    def _reference_demand(self):
        if self._ref is None:
            wn = wntr.network.WaterNetworkModel(str(self.inp))
            ids = list(self.svc.node_id)
            mult = wn.options.hydraulic.demand_multiplier
            self._ref = pd.Series({i: wn.get_node(i).demand_timeseries_list[0].base_value * mult for i in ids})
        return self._ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ms = a.repo / "outputs" / "municipal_scale_v2.5.0"
    out = a.out
    out.mkdir(parents=True, exist_ok=True)

    net = pp.from_json(str(ms / "electricity_municipal_pandapower.json"))
    geo = geo_of(net)
    g = nx.Graph()
    for i, r in net.line.iterrows():
        g.add_edge(int(r.from_bus), int(r.to_bus))
    for i, r in net.trafo.iterrows():
        g.add_edge(int(r.hv_bus), int(r.lv_bus))
    src = int(net.ext_grid.bus.iloc[0])
    tree = nx.bfs_tree(g, src)
    depth = nx.single_source_shortest_path_length(tree, src)
    load_n = net.load.groupby("bus").size()
    load_p = net.load.groupby("bus").p_mw.sum()

    # facilities and their (synthetic, nearest-compatible) electrical attachment
    zb = pd.read_csv(ms / "drinking_water_zone_boundaries.csv")
    st = pd.read_csv(ms / "drinking_water_station_municipal.csv").iloc[0]
    ls = pd.read_csv(ms / "wastewater_pump_stations_municipal.csv")
    mv_tr = [int(i) for i in net.bus.index[net.bus.name.str.startswith("E_MV_TR")]]
    lv = [int(i) for i in net.bus.index[net.bus.vn_kv < 1.0]]
    fac = []
    for r in zb.itertuples():
        b, d = nearest_bus(geo, mv_tr, r.lon, r.lat)
        fac.append({"facility": r.pressure_zone_id, "kind": "water", "role": "waterworks" if r.pressure_zone_id == "WZ1" else "booster",
                    "lon": r.lon, "lat": r.lat, "bus": b, "bus_name": net.bus.name.at[b], "distance_km": d,
                    "buildings": int(r.registered_buildings), "design_flow_lps": float(r.design_peak_flow_lps)})
    for r in ls.itertuples():
        b, d = nearest_bus(geo, lv, r.lon, r.lat)
        fac.append({"facility": r.map_id, "kind": "wastewater", "role": "lift station", "lon": r.lon, "lat": r.lat,
                    "bus": b, "bus_name": net.bus.name.at[b], "distance_km": d,
                    "buildings": int(r.registered_buildings), "design_flow_lps": float(r.design_pump_flow_lps)})
    fac = pd.DataFrame(fac)
    fac.to_csv(out / "facility_attachments.csv", index=False)
    inflow_lps = dict(zip(ls.map_id, ls.annual_sanitary_m3 / (365.0 * 86400.0) * 1000.0))

    ledger = pd.read_csv(a.repo / "outputs" / "building_sector_demands.csv")
    wnodes = pd.read_csv(ms / "drinking_water_nodes.csv")
    heads = dict(zip(zb.pressure_zone_id, zb.selected_boundary_head_m))
    water = WaterModel(ms / "drinking_water_municipal_peak.inp", wnodes, ledger, heads)
    base_w = water.solve(frozenset(), True)
    print("base water", base_w, flush=True)

    mv_lines = net.line[(net.bus.vn_kv.reindex(net.line.from_bus).to_numpy() >= 1.0)]
    total_loads = int(len(net.load)); total_p = float(net.load.p_mw.sum())
    rows = []
    for idx, r in mv_lines.iterrows():
        u, v = int(r.from_bus), int(r.to_bus)
        child = v if depth.get(v, 1e9) > depth.get(u, 1e9) else u
        dead = nx.descendants(tree, child) | {child}
        dead_fac = fac[fac.bus.isin(dead)]
        w_out = frozenset(dead_fac.loc[dead_fac.kind.eq("water"), "facility"])
        rec = {"line": str(r["name"]), "length_km": float(r.length_km),
               "e_loads_lost": int(load_n.reindex(list(dead)).fillna(0).sum()),
               "e_mw_lost": float(load_p.reindex(list(dead)).fillna(0).sum()),
               "water_stations_out": "+".join(sorted(w_out)),
               "lift_stations_out": "+".join(sorted(dead_fac.loc[dead_fac.kind.eq("wastewater"), "facility"]))}
        for mit, protected in [("M0", set()), ("M1", {"WZ1"}), ("M2", {"WZ1", "WZ2", "WZ3", "WZ4"})]:
            wo = frozenset(w_out - protected)
            wb = water.solve(wo, True)
            rec[f"{mit}_w_pop_equiv"] = wb["pop_equiv_without_water"]
            rec[f"{mit}_w_unserved_pct"] = wb["unserved_pct"]
            if mit == "M0":
                wn_ = water.solve(wo, False)
                rec["M0_w_pop_equiv_no_bypass"] = wn_["pop_equiv_without_water"]
                rec["M0_w_unserved_pct_no_bypass"] = wn_["unserved_pct"]
                rec["M0_w_buildings_below_half"] = wb["buildings_below_half"]
        rec["s_inflow_lps"] = float(sum(inflow_lps[f] for f in dead_fac.loc[dead_fac.kind.eq("wastewater"), "facility"]))
        rows.append(rec)
    res = pd.DataFrame(rows)
    res["e_rank"] = res.e_loads_lost.rank(ascending=False, method="min").astype(int)
    res.to_csv(out / "mv_line_contingencies.csv", index=False)

    # line geometry for the map
    links = pd.read_csv(ms / "electricity_links_municipal.csv")
    res.merge(links[["line_id", "geometry_json"]], left_on="line", right_on="line_id", how="left") \
       .drop(columns=["line_id"]).to_csv(out / "mv_line_contingencies_geo.csv", index=False)

    rho = res[["e_loads_lost", "M0_w_pop_equiv"]].rank().corr().iloc[0, 1]
    crit = lambda col, thr=1000.0: res[res[col] >= thr]
    ww_path = res[res.water_stations_out.str.contains("WZ1")]
    summary = {
        "assumptions": {"pdd": {"p_min_m": P_MIN_M, "p_req_m": P_REQ_M, "exponent": P_EXP},
                        "state": "design hour, radial MV network, before switching or repair",
                        "booster_supply": "boosters take suction from the waterworks zone WZ1; bypass at WZ1 head (upper bound) or no supply (lower bound)",
                        "storage": "none represented", "sewer": "average dry-weather sanitary inflow of catchments whose lift station is unpowered",
                        "attachment": "nearest MV/LV transformer MV bus (water stations), nearest LV bus (lift stations); synthetic attachments"},
        "mv_lines": int(len(res)), "mv_km": float(res.length_km.sum()),
        "total_loads": total_loads, "total_mw": total_p, "population": float(water.svc.population.sum()),
        "base_water": base_w,
        "facilities": fac.drop(columns=["bus"]).to_dict(orient="records"),
        "lines_affecting_water_M0": int((res.M0_w_pop_equiv > 0.5).sum()),
        "lines_affecting_lift_stations": int((res.s_inflow_lps > 0).sum()),
        "max_unpumped_sanitary_inflow_lps": float(res.s_inflow_lps.max()),
        "total_sanitary_inflow_lps": float(sum(inflow_lps.values())),
        "lines_ge_1000_pe_M0": int(len(crit("M0_w_pop_equiv"))), "lines_ge_1000_pe_M1": int(len(crit("M1_w_pop_equiv"))),
        "lines_ge_1000_pe_M2": int(len(crit("M2_w_pop_equiv"))),
        "waterworks_path_lines": int(len(ww_path)), "waterworks_path_km": float(ww_path.length_km.sum()),
        "waterworks_path_loads_min_max": [int(ww_path.e_loads_lost.min()), int(ww_path.e_loads_lost.max())] if len(ww_path) else None,
        "waterworks_path_mw_min_max": [float(ww_path.e_mw_lost.min()), float(ww_path.e_mw_lost.max())] if len(ww_path) else None,
        "spearman_electric_vs_water": float(rho),
        "waterworks_path_electric_ranks": sorted(int(x) for x in ww_path.e_rank),
        "lines_ge_1000_pe_M0_no_bypass": int((res.M0_w_pop_equiv_no_bypass >= 1000).sum()),
        "lines_affecting_water_M1": int((res.M1_w_pop_equiv > 0.5).sum()),
        "lines_affecting_water_M2": int((res.M2_w_pop_equiv > 0.5).sum()),
        "backup_power_kw": {"waterworks_duty": float(st.per_pump_operating_power_kw) * 3.0,
                            **{z: 998 * 9.81 * float(r.design_peak_flow_lps) / 1000 * max(float(r.selected_boundary_head_m) - heads["WZ1"], 0.0)
                               / (0.72 * 0.94) / 1000 for z, r in zb.set_index("pressure_zone_id").iterrows() if z != "WZ1"}},
        "max_pop_equiv_M0": float(res.M0_w_pop_equiv.max()), "max_pop_equiv_M1": float(res.M1_w_pop_equiv.max()),
        "max_pop_equiv_M2": float(res.M2_w_pop_equiv.max()),
        "sum_pop_equiv_km_M0": float((res.M0_w_pop_equiv * res.length_km).sum()),
        "sum_pop_equiv_km_M1": float((res.M1_w_pop_equiv * res.length_km).sum()),
        "sum_pop_equiv_km_M2": float((res.M2_w_pop_equiv * res.length_km).sum()),
        "water_state_cache": {f"{'+'.join(sorted(k[0])) or 'none'}|bypass={k[1]}": v for k, v in water.cache.items()},
    }
    (out / "water_power_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("facilities", "water_state_cache")},
                     indent=1, default=float))
    print(json.dumps(summary["water_state_cache"], indent=1, default=float))


if __name__ == "__main__":
    main()
