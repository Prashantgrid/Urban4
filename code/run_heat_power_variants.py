#!/usr/bin/env python3
"""Heat--power case study, part 2: measured heat-pump ratio, renovation, and costs.

Re-evaluates the design state (and the temperature sweep of the load-summation
screen) of run_heat_power_case.py for

* S1 building heat pumps with two electrical-to-thermal ratios at the design
  temperature: the when2heat COP model (1/COP = 0.64 at -14 degC) and the
  measured ratio of large air-source heat-pump collectives (0.40, Fraunhofer
  IEE/ISE 2026), each for renovation factors f = 1.0, 0.8, 0.6 applied to the
  building heat demand;
* S2 GKS outage (H2 alone, direct-electric backup for the deficit) and S3
  (DN400 intertie) for f = 1.0 and 0.8;

and prices the reinforcement that removes the thermal overloads found by the
screen (lower bound: voltage-driven reinforcement and new station buildings
are not priced) against the intertie upsizing.

Unit costs: LV cable 100 kEUR/km and MV cable 140 kEUR/km incl. urban
earthwork, MV/LV transformer 10 kEUR (dena Verteilnetzstudie 2012, as used in
eDisGo); pre-insulated DH pipe pair DN300 2183 EUR/m, DN400 2902 EUR/m
(Topotherm default cost table).

Usage: python run_heat_power_variants.py --repo /path/to/Urban4 --out results/heat_power_case
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pandapipes as ppi

_spec = importlib.util.spec_from_file_location("hp", Path(__file__).with_name("run_heat_power_case.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

RATIO_MEASURED = 0.40            # MW_el per MW_th, air-source collectives (Fraunhofer IEE/ISE 2026)
RENOVATION = [1.0, 0.8, 0.6]
C_LV_KEUR_KM, C_MV_KEUR_KM, C_TR_KEUR = 100.0, 140.0, 10.0
TR_UNIT_MVA = 0.63
C_DN300_EUR_M, C_DN400_EUR_M = 2183.0, 2902.0


def reinforcement(screen: H.RadialScreen, net, base_req):
    r = screen.evaluate(net)
    s = r["_trafo_pct"] * screen.trafo_sn / 100.0
    req = s / H.PLANNING_UTILISATION
    extra = np.maximum(req - np.maximum(screen.trafo_sn, base_req), 0.0)
    units = np.ceil(extra / TR_UNIT_MVA - 1e-9)
    cost = (C_LV_KEUR_KM * r["scr_overloaded_lv_km"] + C_MV_KEUR_KM * r["scr_overloaded_mv_km"]
            + C_TR_KEUR * units.sum()) / 1e3
    return {**{k: v for k, v in r.items() if not k.startswith("_")},
            "overloaded_areas": int((r["_trafo_pct"] > H.LOADING_MAX + 1e-9).sum()),
            "added_transformer_units": int(units.sum()), "areas_reinforced": int((units > 0).sum()),
            "grid_cost_meur_lower_bound": float(cost)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ms = a.repo / "outputs" / "municipal_scale_v2.5.0"
    integ = a.repo / "outputs" / "integrated_service_resolved"
    net_e0 = pp.from_json(str(ms / "electricity_municipal_pandapower.json"))
    net_h0 = ppi.from_json(str(ms / "district_heating_municipal_pandapipes.json"))
    corridors = pd.read_csv(ms / "district_heating_corridors_municipal.csv")
    dh = pd.read_csv(ms / "district_heating_services_municipal.csv")
    sources = pd.read_csv(ms / "district_heating_sources_municipal.csv").set_index("map_id")
    svc = pd.read_csv(integ / "electricity_service_connections.csv").set_index("building_id")
    name_to_idx = pd.Series(net_e0.bus.index, index=net_e0.bus.name)
    dh["lv_bus"] = dh.building_id.map(svc.building_bus).map(name_to_idx).astype(int)
    q_pk, buses = dh.peak_heat_mw.to_numpy(), dh.lv_bus.to_numpy()
    h2_bus = H.nearest_mv_bus(net_e0, float(sources.at["H2", "lon"]), float(sources.at["H2", "lat"]))

    screen = H.RadialScreen(net_e0)
    b = screen.evaluate(net_e0)
    base_req = b["_trafo_pct"] * screen.trafo_sn / 100.0 / H.PLANNING_UTILISATION
    phi_cap, _ = H.max_feasible_fraction(net_h0, ["H2"])
    net_h400, up_km = H.upsize_intertie(net_h0, corridors, 300.0, 400.0)
    cop_d = H.cop_ashp(H.T_DESIGN_C)

    def p_el(t_a, f, mode):
        phi = H.heat_fraction(t_a)
        if mode == "cop":
            return phi * f * q_pk / H.cop_ashp(t_a)
        return phi * f * q_pk * RATIO_MEASURED * cop_d / H.cop_ashp(t_a)

    rows, states = [], []
    for mode in ["cop", "measured"]:
        for f in RENOVATION:
            first = None
            for t_a in H.TEMPS:
                net = H.with_loads(net_e0, buses, p_el(t_a, f, mode), "HP")
                r = screen.evaluate(net)
                states.append({"ratio_mode": mode, "renovation_f": f, "t_a_c": float(t_a),
                               "added_electric_mw": float(p_el(t_a, f, mode).sum()),
                               "overloaded_areas": int(r["scr_overloaded_trafos"]),
                               "overloaded_lv_km": float(r["scr_overloaded_lv_km"])})
                if first is None and (r["scr_overloaded_trafos"] > 0 or r["scr_overloaded_lv_km"] > 0
                                      or r["scr_overloaded_mv_km"] > 0):
                    first = float(t_a)
            net = H.with_loads(net_e0, buses, p_el(H.T_DESIGN_C, f, mode), "HP")
            rec = {"scenario": "S1_HP", "ratio_mode": mode, "renovation_f": f,
                   "added_electric_mw": float(p_el(H.T_DESIGN_C, f, mode).sum()),
                   "el_to_th_ratio_design": float(p_el(H.T_DESIGN_C, f, mode).sum() / (f * q_pk.sum())),
                   "first_violation_c": first, **reinforcement(screen, net, base_req), **H.ac_state(copy.deepcopy(net)),
                   "measure": "", "measure_cost_meur": np.nan}
            rows.append(rec)
            print(mode, f, round(rec["added_electric_mw"], 2), rec["overloaded_areas"], round(rec["scr_overloaded_lv_km"], 2),
                  rec["added_transformer_units"], round(rec["grid_cost_meur_lower_bound"], 2), first, rec["ac_converged"], flush=True)

    for f in [1.0, 0.8]:
        for label, net_h, cap in [("S2_GKS_outage", net_h0, phi_cap), ("S3_GKS_outage_DN400", net_h400, 1.0)]:
            first = None
            for t_a in list(H.TEMPS):
                phi = H.heat_fraction(t_a) * f
                phi_dh = min(phi, cap)
                deficit = (phi - phi_dh) * q_pk
                if deficit.sum() <= 0:
                    continue
                net = H.with_loads(net_e0, buses, deficit, "BK")
                r = screen.evaluate(net)
                if first is None and (r["scr_overloaded_trafos"] > 0 or r["scr_overloaded_lv_km"] > 0):
                    first = float(t_a)
            phi = H.heat_fraction(H.T_DESIGN_C) * f
            phi_dh = min(phi, cap)
            h = H.dh_solve(net_h, phi_dh, ["H2"])
            deficit = (phi - phi_dh) * q_pk
            pump = h["pump_kw"]["H2"] / 1e3
            net = H.with_loads(net_e0, np.r_[h2_bus, buses], np.r_[pump, deficit], "S2")
            meas = "DN300->DN400 intertie" if label.startswith("S3") else ""
            rec = {"scenario": label, "ratio_mode": "", "renovation_f": f, "added_electric_mw": float(pump + deficit.sum()),
                   "backup_mw": float(deficit.sum()), "dh_delivered_fraction": float(phi_dh / phi),
                   "first_violation_c": first, **reinforcement(screen, net, base_req), **H.ac_state(copy.deepcopy(net)),
                   "measure": meas,
                   "measure_cost_meur": (up_km * C_DN400_EUR_M / 1e3 if meas else np.nan),
                   "measure_increment_meur": (up_km * (C_DN400_EUR_M - C_DN300_EUR_M) / 1e3 if meas else np.nan)}
            rows.append(rec)
            print(label, f, round(rec["backup_mw"], 2), rec["overloaded_areas"], round(rec["grid_cost_meur_lower_bound"], 2), first,
                  flush=True)

    res = pd.DataFrame(rows)
    from scenario_names import SCENARIO_NAMES
    res["scenario_name"] = res["scenario"].map(SCENARIO_NAMES)
    res.to_csv(a.out / "heat_power_variants.csv", index=False)
    pd.DataFrame(states).to_csv(a.out / "heat_power_variant_states.csv", index=False)
    summary = {
        "ratio_measured": RATIO_MEASURED, "ratio_cop_design": 1.0 / cop_d, "renovation": RENOVATION,
        "unit_costs": {"lv_cable_keur_km": C_LV_KEUR_KM, "mv_cable_keur_km": C_MV_KEUR_KM, "transformer_keur": C_TR_KEUR,
                       "transformer_unit_mva": TR_UNIT_MVA, "dn300_eur_m": C_DN300_EUR_M, "dn400_eur_m": C_DN400_EUR_M},
        "h2_only_max_fraction": phi_cap, "intertie_upsized_km": up_km,
        "intertie_new_dn400_meur": up_km * C_DN400_EUR_M / 1e3,
        "intertie_increment_meur": up_km * (C_DN400_EUR_M - C_DN300_EUR_M) / 1e3,
        "variants": res.to_dict(orient="records"),
    }
    (a.out / "heat_power_variants_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
