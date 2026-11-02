#!/usr/bin/env python3
"""Heat--power case study on the accepted Schweinfurt municipality models.

The study couples the accepted pandapipes district-heating (DH) model and the
accepted pandapower electricity model through the shared building ledger and
the GKS shared-asset record.  Quasi-steady winter operating states are indexed
by the outdoor temperature T_a (1-K steps from +15 to -14 degC).

Scenarios
---------
S0  Reference: DH supplies its 841 connection equivalents from GKS (H1) and
    Spitzenheizwerk Nord (H2).  The electricity model carries the accepted
    coincident LV allocation plus the DH circulation-pump demand at the sources.
S1  Building heat pumps: the same buildings are supplied by air-source heat
    pumps connected at their own LV service node; the DH network is idle.
S2  Shared-CHP outage: the GKS shared asset is unavailable, so H2 alone feeds
    the DH network through the generated H1--H2 intertie.  Heat that H2 cannot
    deliver within the hydraulic screens is covered by direct-electric backup
    heaters at the affected buildings (uniform curtailment).
S3  As S2, but the DN300 segments of the intertie are upsized to DN400.

Electricity states are evaluated twice: (i) an AC power flow in pandapower,
reported where it converges, and (ii) a radial load-summation screen that sums
downstream apparent power along the radial MV/LV tree and compares it with
transformer ratings and cable ampacities.  The screen is defined for every
state and agrees with the AC solution to within 2.4 percentage points in the
accepted base state.

Usage:  python run_heat_power_case.py --repo /path/to/Urban4 --out results/
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
import pandapipes as ppi

# --------------------------------------------------------------------------
# Declared case assumptions (all stated in the manuscript)
# --------------------------------------------------------------------------
T_DESIGN_C = -14.0          # lower end of the operator's published heating curve
T_HEAT_LIMIT_C = 15.0       # heating limit
COP_COEFF = (6.08, -0.0941, 0.000464)   # when2heat ASHP regression (Ruhnau et al. 2019)
COP_CORRECTION = 0.85       # when2heat field-performance correction
LOAD_PF = 0.96              # power factor of added loads (as in the ledger)
PUMP_ETA = 0.78 * 0.94      # hydraulic x motor efficiency of DH circulation pumps
HANDOVER_DP_BAR = 0.5       # upper value of the published 0.3-0.5 bar minimum handover dp
V_HARD_M_S = 2.0            # hard velocity screen
LIFT_MAX_BAR = 10.0         # source-lift screen
RHO = 985.0                 # kg/m3
V_MIN_PU, LOADING_MAX = 0.90, 100.0
PLANNING_UTILISATION = 0.80  # transformer planning utilisation used by Urban4 sizing
TEMPS = np.round(np.arange(15.0, -14.0 - 1e-9, -1.0), 1)


def heat_fraction(t_a: float) -> float:
    return float(np.clip((T_HEAT_LIMIT_C - t_a) / (T_HEAT_LIMIT_C - T_DESIGN_C), 0.0, 1.0))


def cop_ashp(t_a: float, correction: float = COP_CORRECTION) -> float:
    t_sink = 40.0 - t_a                      # radiator sink temperature (when2heat)
    d_t = max(t_sink - t_a, 15.0)
    a0, a1, a2 = COP_COEFF
    return correction * (a0 + a1 * d_t + a2 * d_t ** 2)


# --------------------------------------------------------------------------
# Electricity
# --------------------------------------------------------------------------
class RadialScreen:
    """Downstream apparent-power summation on the radial MV/LV tree."""

    def __init__(self, net):
        g = nx.Graph()
        g.add_edges_from(zip(net.line.from_bus, net.line.to_bus))
        g.add_edges_from(zip(net.trafo.hv_bus, net.trafo.lv_bus))
        if not nx.is_tree(g):
            raise RuntimeError("load-summation screen requires a radial network")
        root = int(net.ext_grid.bus.iloc[0])
        self.order = list(nx.dfs_preorder_nodes(g, root))
        parent = dict(nx.bfs_predecessors(g, root))
        self.pos = {b: i for i, b in enumerate(net.bus.index)}
        self.parent_pos = np.array([self.pos[parent[n]] if n != root else -1 for n in self.order])
        self.order_pos = np.array([self.pos[n] for n in self.order])
        child = np.where(
            [parent.get(t) == f for f, t in zip(net.line.from_bus, net.line.to_bus)],
            net.line.to_bus, net.line.from_bus)
        self.line_child = np.array([self.pos[c] for c in child])
        self.line_vn = net.bus.vn_kv.loc[child].to_numpy()
        self.line_amp = (net.line.max_i_ka * net.line.parallel * net.line.df).to_numpy()
        self.line_len = net.line.length_km.to_numpy()
        self.line_is_lv = self.line_vn < 1.0
        self.trafo_lv = np.array([self.pos[b] for b in net.trafo.lv_bus])
        self.trafo_sn = net.trafo.sn_mva.to_numpy()
        self.n = len(net.bus)

    def evaluate(self, net) -> dict:
        s = np.zeros(self.n, complex)
        np.add.at(s, [self.pos[b] for b in net.load.bus],
                  (net.load.p_mw + 1j * net.load.q_mvar).to_numpy() * net.load.scaling.to_numpy())
        for k in range(len(self.order_pos) - 1, 0, -1):
            s[self.parent_pos[k]] += s[self.order_pos[k]]
        line_pct = 100.0 * np.abs(s[self.line_child]) / (math.sqrt(3) * self.line_vn) / self.line_amp
        trafo_s = np.abs(s[self.trafo_lv])
        trafo_pct = 100.0 * trafo_s / self.trafo_sn
        over_line = line_pct > LOADING_MAX + 1e-9
        req = trafo_s / PLANNING_UTILISATION   # MVA needed at the 80 % planning utilisation
        return {
            "scr_max_trafo_loading_pct": float(trafo_pct.max()),
            "scr_overloaded_trafos": int((trafo_pct > LOADING_MAX + 1e-9).sum()),
            "scr_max_line_loading_pct": float(line_pct.max()),
            "scr_overloaded_lv_km": float(self.line_len[over_line & self.line_is_lv].sum()),
            "scr_overloaded_mv_km": float(self.line_len[over_line & ~self.line_is_lv].sum()),
            "scr_additional_mva_at_80pct": float(np.maximum(req - self.trafo_sn, 0.0).sum()),
            "scr_trafos_needing_upgrade": int((req > self.trafo_sn + 1e-9).sum()),
            "_trafo_pct": trafo_pct,
        }


def ac_state(net) -> dict:
    try:
        pp.runpp(net, algorithm="nr", max_iteration=40, numba=False, init="dc")
        ok = bool(net.converged)
    except Exception:  # non-convergence is a reportable outcome
        ok = False
    if not ok:
        return {"ac_converged": False}
    tr, ln, vm = net.res_trafo.loading_percent, net.res_line.loading_percent, net.res_bus.vm_pu
    return {
        "ac_converged": True,
        "ac_max_trafo_loading_pct": float(tr.max()),
        "ac_overloaded_trafos": int((tr > LOADING_MAX + 1e-9).sum()),
        "ac_max_line_loading_pct": float(ln.max()),
        "ac_overloaded_lines": int((ln > LOADING_MAX + 1e-9).sum()),
        "ac_min_vm_pu": float(vm.min()),
        "ac_undervoltage_buses": int((vm < V_MIN_PU - 1e-9).sum()),
    }


def with_loads(net0, buses, p_mw, tag):
    net = copy.deepcopy(net0)
    buses = np.asarray(buses, dtype=int)
    p_mw = np.asarray(p_mw, dtype=float)
    keep = p_mw > 0
    if keep.any():
        q = p_mw[keep] * math.tan(math.acos(LOAD_PF))
        pp.create_loads(net, buses[keep], p_mw=p_mw[keep], q_mvar=q,
                        name=[f"{tag}_{i:04d}" for i in range(int(keep.sum()))])
    return net


# --------------------------------------------------------------------------
# District heating
# --------------------------------------------------------------------------
def dh_solve(net_h, phi: float, sources: list[str]) -> dict:
    """Solve the DH supply network at load fraction phi with the given sources.

    Junction heights are set to zero because supply and return pipes share a
    trench, so static head cancels in the closed loop; the source differential
    pressure is twice the largest supply-side friction loss plus the handover
    differential pressure.  Customers keep their class design temperature
    difference (the remaining sources operate at maximum supply temperature).
    """
    net = copy.deepcopy(net_h)
    net.junction["height_m"] = 0.0
    net.sink["mdot_kg_per_s"] = net_h.sink.mdot_kg_per_s * max(phi, 1e-4)
    net.ext_grid["in_service"] = net.ext_grid.name.isin(sources)
    try:
        ppi.pipeflow(net, mode="hydraulics", max_iter_hyd=200)
        conv = bool(net.converged)
    except Exception:
        conv = False
    if not conv:
        return {"converged": False, "phi": phi, "feasible": False}
    p_slack = float(net.ext_grid.p_bar.iloc[0])
    dp_fric = float(p_slack - net.res_junction.p_bar.min())
    dp_src = 2.0 * dp_fric + HANDOVER_DP_BAR
    v = net.res_pipe.v_mean_m_per_s.abs()
    flows = {}
    for row in net.ext_grid.itertuples():
        flows[row.name] = float(-net.res_ext_grid.mdot_kg_per_s.at[row.Index]) if row.in_service else 0.0
    pump_kw = {s: m / RHO * dp_src * 1e5 / PUMP_ETA / 1e3 for s, m in flows.items()}
    return {
        "converged": True, "phi": phi, "v_max_m_s": float(v.max()),
        "v_max_pipe": str(net.pipe.name.at[v.idxmax()]), "dp_fric_bar": dp_fric,
        "dp_src_bar": dp_src, "mdot_kg_s": flows, "pump_kw": pump_kw,
        "feasible": bool(v.max() <= V_HARD_M_S + 1e-9 and dp_src <= LIFT_MAX_BAR + 1e-9),
    }


def max_feasible_fraction(net_h, sources: list[str]) -> tuple[float, dict]:
    top = dh_solve(net_h, 1.0, sources)
    if top.get("feasible"):
        return 1.0, top
    lo, hi, best = 0.0, 1.0, None
    while hi - lo > 1e-5:
        mid = 0.5 * (lo + hi)
        res = dh_solve(net_h, mid, sources)
        if res.get("feasible"):
            lo, best = mid, res
        else:
            hi = mid
    return lo, best


def upsize_intertie(net_h, corridors: pd.DataFrame, from_dn: float, to_dn: float):
    net = copy.deepcopy(net_h)
    ids = set(corridors.loc[corridors.edge_role.eq("source_intertie")
                            & corridors.diameter_mm.eq(from_dn), "corridor_id"])
    mask = net.pipe.name.isin(ids)
    net.pipe.loc[mask, "inner_diameter_mm"] = to_dn
    length = float(net.pipe.loc[mask, "length_km"].sum())
    return net, length


def nearest_mv_bus(net_e, lon: float, lat: float) -> int:
    mv = net_e.bus[net_e.bus.vn_kv >= 1.0]
    geo = mv.geo.map(lambda g: json.loads(g)["coordinates"] if isinstance(g, str) else g["coordinates"])
    xs = np.array([c[0] for c in geo]); ys = np.array([c[1] for c in geo])
    d = ((xs - lon) * math.cos(math.radians(lat))) ** 2 + (ys - lat) ** 2
    return int(mv.index[int(np.argmin(d))])


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ms = args.repo / "outputs" / "municipal_scale_v2.5.0"
    integ = args.repo / "outputs" / "integrated_service_resolved"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    net_e0 = pp.from_json(str(ms / "electricity_municipal_pandapower.json"))
    net_h0 = ppi.from_json(str(ms / "district_heating_municipal_pandapipes.json"))
    corridors = pd.read_csv(ms / "district_heating_corridors_municipal.csv")
    dh = pd.read_csv(ms / "district_heating_services_municipal.csv")
    sources = pd.read_csv(ms / "district_heating_sources_municipal.csv").set_index("map_id")
    shared = pd.read_csv(ms / "shared_energy_assets_municipal.csv")
    svc = pd.read_csv(integ / "electricity_service_connections.csv")

    # ---- shared-ledger join: DH customer building -> LV service bus and transformer area
    svc_by_building = svc.set_index("building_id")
    missing = sorted(set(dh.building_id) - set(svc_by_building.index))
    if missing:
        raise RuntimeError(f"{len(missing)} DH buildings lack an electricity service node")
    name_to_idx = pd.Series(net_e0.bus.index, index=net_e0.bus.name)
    dh["lv_bus"] = dh.building_id.map(svc_by_building.building_bus).map(name_to_idx).astype(int)
    dh["transformer_id"] = dh.building_id.map(svc_by_building.transformer_id)

    # ---- DH circulation-pump attachment: GKS electrical port (shared asset) and nearest MV bus for H2
    gks = shared[shared.shared_asset_id.eq("GKS_CHP") & shared.to_id.eq("H1")]
    pump_bus = {"H1": int(name_to_idx[str(gks.from_id.iloc[0])]),
                "H2": nearest_mv_bus(net_e0, float(sources.at["H2", "lon"]), float(sources.at["H2", "lat"]))}

    screen = RadialScreen(net_e0)
    base_ac = ac_state(copy.deepcopy(net_e0))
    base_scr = {k: v for k, v in screen.evaluate(net_e0).items() if not k.startswith("_")}

    # ---- DH hydraulic capability
    net_h400, intertie_upsized_km = upsize_intertie(net_h0, corridors, 300.0, 400.0)
    phi_h2, h2_lim = max_feasible_fraction(net_h0, ["H2"])
    phi_h2_400, h2_lim_400 = max_feasible_fraction(net_h400, ["H2"])
    phi_both, _ = max_feasible_fraction(net_h0, ["H1", "H2"])
    cap_rows = []
    for phi in np.round(np.linspace(0.05, 1.0, 39), 4):
        for label, net_h, srcs in [("both", net_h0, ["H1", "H2"]), ("H2_only", net_h0, ["H2"]),
                                   ("H2_only_DN400", net_h400, ["H2"])]:
            r = dh_solve(net_h, float(phi), srcs)
            cap_rows.append({"phi": phi, "case": label, "converged": r["converged"],
                             "v_max_m_s": r.get("v_max_m_s", np.nan), "dp_src_bar": r.get("dp_src_bar", np.nan),
                             "v_max_pipe": r.get("v_max_pipe", "")})
    pd.DataFrame(cap_rows).to_csv(out / "dh_hydraulic_capability.csv", index=False)

    q_pk = dh.peak_heat_mw.to_numpy()
    buses = dh.lv_bus.to_numpy()
    rows, trafo_design = [], {}

    def record(scenario, t_a, added_mw, dh_mw, net, extra):
        scr = screen.evaluate(net)
        ac = ac_state(net)
        if t_a == T_DESIGN_C:
            trafo_design[scenario] = scr["_trafo_pct"]
        rows.append({"scenario": scenario, "t_a_c": t_a, "heat_fraction": heat_fraction(t_a),
                     "heat_demand_mw": heat_fraction(t_a) * q_pk.sum(), "added_electric_mw": added_mw,
                     "dh_delivered_mw": dh_mw, **extra,
                     **{k: v for k, v in scr.items() if not k.startswith("_")}, **ac})

    for t_a in TEMPS:
        phi = heat_fraction(t_a)
        cop = cop_ashp(t_a)

        # S0 -- DH reference
        h0 = dh_solve(net_h0, phi, ["H1", "H2"])
        pumps = np.array([h0["pump_kw"]["H1"], h0["pump_kw"]["H2"]]) / 1e3
        net = with_loads(net_e0, [pump_bus["H1"], pump_bus["H2"]], pumps, "DHPUMP")
        record("S0_DH", t_a, float(pumps.sum()), phi * q_pk.sum(), net,
               {"cop": np.nan, "dh_v_max_m_s": h0["v_max_m_s"], "dh_dp_src_bar": h0["dp_src_bar"],
                "backup_mw": 0.0})

        # S1 -- building heat pumps (central COP and uncorrected-COP sensitivity)
        for label, c in [("S1_HP", cop), ("S1_HP_cop_uncorrected", cop_ashp(t_a, 1.0))]:
            p_hp = phi * q_pk / c
            net = with_loads(net_e0, buses, p_hp, "HP")
            record(label, t_a, float(p_hp.sum()), 0.0, net,
                   {"cop": c, "dh_v_max_m_s": np.nan, "dh_dp_src_bar": np.nan, "backup_mw": 0.0})

        # S2 / S3 -- GKS unavailable, H2 only (+ direct-electric backup for any deficit)
        for label, net_h, phi_cap in [("S2_GKS_outage", net_h0, phi_h2),
                                      ("S3_GKS_outage_DN400", net_h400, phi_h2_400)]:
            phi_dh = min(phi, phi_cap)
            h = dh_solve(net_h, phi_dh, ["H2"])
            pump = h["pump_kw"]["H2"] / 1e3
            deficit = (phi - phi_dh) * q_pk
            net = with_loads(net_e0, np.r_[pump_bus["H2"], buses], np.r_[pump, deficit], "S2")
            record(label, t_a, float(pump + deficit.sum()), phi_dh * q_pk.sum(), net,
                   {"cop": 1.0, "dh_v_max_m_s": h["v_max_m_s"], "dh_dp_src_bar": h["dp_src_bar"],
                    "backup_mw": float(deficit.sum())})
        last = {r["scenario"]: r for r in rows[-5:]}
        print(f"T={t_a:5.1f} phi={phi:.3f} COP={cop:.2f} | S1 +{last['S1_HP']['added_electric_mw']:5.2f} MW "
              f"tr>100 {last['S1_HP']['scr_overloaded_trafos']:3d} AC={last['S1_HP']['ac_converged']} | "
              f"S2 backup {last['S2_GKS_outage']['backup_mw']:5.2f} MW tr>100 {last['S2_GKS_outage']['scr_overloaded_trafos']:3d} | "
              f"S3 backup {last['S3_GKS_outage_DN400']['backup_mw']:5.2f}", flush=True)

    res = pd.DataFrame(rows)
    from scenario_names import SCENARIO_NAMES
    res["scenario_name"] = res["scenario"].map(SCENARIO_NAMES)
    res.to_csv(out / "heat_power_states.csv", index=False)

    # ---- transformer-area table at the design state (for the map)
    geo = net_e0.bus.geo.map(lambda g: json.loads(g)["coordinates"] if isinstance(g, str) else g["coordinates"])
    tr = pd.DataFrame({"transformer_id": net_e0.trafo.name.to_numpy(),
                       "lon": [geo.at[b][0] for b in net_e0.trafo.lv_bus],
                       "lat": [geo.at[b][1] for b in net_e0.trafo.lv_bus],
                       "sn_mva": net_e0.trafo.sn_mva.to_numpy()})
    for k, v in trafo_design.items():
        tr[f"design_loading_pct_{k}"] = v
    per = dh.groupby("transformer_id").agg(dh_buildings=("building_id", "size"), dh_peak_mw=("peak_heat_mw", "sum"))
    tr = tr.merge(per, left_on="transformer_id", right_index=True, how="left")
    tr[["dh_buildings", "dh_peak_mw"]] = tr[["dh_buildings", "dh_peak_mw"]].fillna(0.0)
    tr.to_csv(out / "transformer_design_state.csv", index=False)

    def first_violation(sc, col="scr_overloaded_trafos"):
        d = res[res.scenario.eq(sc)].sort_values("t_a_c", ascending=False)
        bad = d[(d[col] > 0) | (d.scr_overloaded_lv_km > 0) | (d.scr_overloaded_mv_km > 0)]
        return float(bad.t_a_c.iloc[0]) if len(bad) else None

    def last_ac(sc):
        d = res[res.scenario.eq(sc) & res.ac_converged.astype(bool)]
        return float(d.t_a_c.min()) if len(d) else None

    design = res[res.t_a_c.eq(T_DESIGN_C)].set_index("scenario")
    summary = {
        "assumptions": {
            "design_outdoor_temperature_c": T_DESIGN_C, "heating_limit_c": T_HEAT_LIMIT_C,
            "cop": "COP = 0.85*(6.08 - 0.0941 dT + 0.000464 dT^2), dT = (40 - T_a) - T_a, dT >= 15 K",
            "load_power_factor": LOAD_PF, "pump_efficiency": PUMP_ETA, "handover_dp_bar": HANDOVER_DP_BAR,
            "velocity_hard_screen_m_s": V_HARD_M_S, "source_lift_screen_bar": LIFT_MAX_BAR,
            "planning_utilisation": PLANNING_UTILISATION,
            "dh_mass_flow_policy": "class design temperature difference in all states",
            "electric_baseline": "accepted coincident LV allocation held constant",
        },
        "dh_customers": int(len(dh)), "dh_design_peak_mw": float(q_pk.sum()),
        "dh_transformer_areas": int(dh.transformer_id.nunique()), "transformer_areas": int(len(net_e0.trafo)),
        "transformer_portfolio_mva": float(net_e0.trafo.sn_mva.sum()),
        "pump_attachment_bus": {k: str(net_e0.bus.name.at[v]) for k, v in pump_bus.items()},
        "base_ac": base_ac, "base_screen": base_scr,
        "cop_design": cop_ashp(T_DESIGN_C), "cop_design_uncorrected": cop_ashp(T_DESIGN_C, 1.0),
        "cop_at_0c": cop_ashp(0.0),
        "both_sources_max_fraction": phi_both,
        "h2_only_max_fraction": phi_h2, "h2_only_limit": h2_lim,
        "h2_only_threshold_c": T_HEAT_LIMIT_C - phi_h2 * (T_HEAT_LIMIT_C - T_DESIGN_C),
        "h2_only_dn400_max_fraction": phi_h2_400, "h2_only_dn400_limit": h2_lim_400,
        "intertie_upsized_km": intertie_upsized_km,
        "intertie_total_km": float(corridors.loc[corridors.edge_role.eq("source_intertie"), "length_km"].sum()),
        "first_violation_c": {sc: first_violation(sc) for sc in
                              ["S1_HP", "S1_HP_cop_uncorrected", "S2_GKS_outage", "S3_GKS_outage_DN400"]},
        "lowest_ac_converged_c": {sc: last_ac(sc) for sc in res.scenario.unique()},
        "design_state": design.drop(columns=[c for c in design.columns if c.startswith("_")]).to_dict(orient="index"),
    }
    (out / "heat_power_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ["base_ac", "base_screen", "h2_only_max_fraction", "h2_only_threshold_c",
                                              "h2_only_dn400_max_fraction", "intertie_upsized_km",
                                              "first_violation_c", "lowest_ac_converged_c"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
