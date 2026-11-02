#!/usr/bin/env python3
"""Water--power case study, part 4: extreme events with event-based durations.

Everyday fault statistics describe short, independent outages.  Extreme events
last far longer: documented German cases range from 31 h (Berlin-Koepenick
2019, drilled cables) through about five days (Berlin, January 2026, arson on a
cable bridge) to five weeks (Ahr valley 2021, flood-destroyed grid).  This
script therefore reports, per sector, WHEN service is lost and HOW MUCH, for

    E1  loss of the grid supply point (city-wide blackout),
    E2  HQ100 flood,  E3  HQextrem flood (outputs of run_flood_case.py),

and for the measures

    E1: none | backup at the four water stations with 72 h of fuel (BBK
        recommendation), not refuelled | backup at all pumped facilities
        (water stations, 13 lift stations, DH circulation pumps), refuelled;
    E2/E3: as mapped | waterworks site and its supply station protected.

Water service is lost when a zone's supply stops and its storage (24 h of
average-day demand) is spent; the extended-period EPANET runs of
run_water_reliability.py confirm that all customers keep full supply until the
tanks are empty.  PE without water afterwards come from the average-day
pressure-dependent states.  Cumulative losses are evaluated for the three
documented durations.

Usage: python run_extreme_events.py --repo /path/to/Urban4 --water results/water_power_case
                                    --heat results/heat_power_case --out results/water_power_case
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STORAGE_H = 24.0
FUEL_H = 72.0                     # BBK: emergency power for 72 h without refuelling
T_SW_H = 1.0                      # switching after a flood (stations behind flooded ones)
DURATIONS = {"Koepenick 2019 (31 h)": 31.0, "Berlin 2026 (about 5 d)": 120.0, "Ahr 2021 (5 weeks)": 840.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--water", type=Path, required=True)
    ap.add_argument("--heat", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ms = a.repo / "outputs" / "municipal_scale_v2.5.0"
    ws = json.loads((a.water / "water_power_summary.json").read_text())
    rel = json.loads((a.water / "water_reliability_summary.json").read_text())
    fl = json.loads((a.water / "flood_summary.json").read_text())["events"]
    heat = json.loads((a.heat / "heat_power_summary.json").read_text())
    ls = pd.read_csv(ms / "wastewater_pump_stations_municipal.csv")

    pop = float(ws["population"])
    customers = int(ws["total_loads"])
    sanitary_lps = float(ws["total_sanitary_inflow_lps"])
    dh_customers = int(heat["dh_customers"])
    dh_pump_mw = float(heat["design_state"]["S0_DH"]["added_electric_mw"])
    water_backup_mw = (ws["backup_power_kw"]["waterworks_duty"] + sum(ws["backup_power_kw"][z] for z in ["WZ2", "WZ3", "WZ4"])) / 1e3
    lift_backup_mw = float(ls.selected_motor_kw.sum()) / 1e3        # one duty motor per station
    all_backup_mw = water_backup_mw + lift_backup_mw + dh_pump_mw

    rows = []

    def add(event, measure, sector, onset_h, value, unit, label, conditional=False):
        rows.append({"event": event, "measure": measure, "sector": sector, "onset_h": onset_h, "value": value,
                     "unit": unit, "label": label, "conditional": conditional})

    # ---------------- E1 city-wide blackout
    E = "Blackout (grid supply point lost)"
    for m, water_onset, pumps in [("no backup", STORAGE_H, False),
                                  (f"water stations on backup, {FUEL_H:.0f} h fuel", FUEL_H + STORAGE_H, False),
                                  ("all pumped facilities on backup, refuelled", None, True)]:
        add(E, m, "power", 0.0, customers, "customers", f"{customers:,} customers")
        if water_onset is not None:
            add(E, m, "water", water_onset, pop, "PE", f"{pop:,.0f} PE")
        if not pumps:
            add(E, m, "wastewater", 0.0, sanitary_lps, "L/s", f"{sanitary_lps:.0f} L/s unpumped")
            add(E, m, "heat", 0.0, dh_customers, "DH customers", f"{dh_customers} DH customers")
        else:
            add(E, m, "wastewater", 0.0, sanitary_lps, "L/s", "treated only if the plant's CHP runs islanded",
                conditional=True)
            add(E, m, "heat", 0.0, dh_customers, "DH customers", "supplied only if DH plants run islanded",
                conditional=True)

    # ---------------- E2/E3 floods
    for key, name in [("HQ100", "HQ100 flood"), ("HQextrem", "HQextrem flood")]:
        r = fl[key]
        for m, prot in [("as mapped", False), ("waterworks site and supply station protected", True)]:
            c = r["customers_without_power_waterworks_station_protected"] if prot else r["customers_without_power_after_switching"]
            add(name, m, "power", 0.0, c, "customers", f"{c:,} customers")
            w = r["water"]["site and supply station protected" if prot else "as mapped"]
            if w["pe_without_water_after_storage"] > 0.5:
                add(name, m, "water", STORAGE_H, w["pe_without_water_after_storage"], "PE",
                    f"{w['pe_without_water_after_storage']:,.0f} PE")
            ww = r["wastewater"]
            if ww["treatment_plant_flooded"]:
                add(name, m, "wastewater", 0.0, ww["untreated_sanitary_lps"], "L/s",
                    f"{ww['untreated_sanitary_lps']:.0f} L/s untreated")
            elif ww["unpumped_sanitary_lps"] > 0:
                add(name, m, "wastewater", 0.0, ww["unpumped_sanitary_lps"], "L/s",
                    f"{ww['unpumped_sanitary_lps']:.0f} L/s unpumped")
            h = r["heat"]
            if h["sources_out"]:
                add(name, m, "heat", 0.0, 1.0 - h["dh_share_at_design"], "share of design heat",
                    f"shortfall below {h['full_supply_down_to_c']:.1f} °C", conditional=True)
    ev = pd.DataFrame(rows)
    ev.to_csv(a.out / "extreme_events.csv", index=False)

    # cumulative consequences for the documented durations
    cum = []
    for (e, m), g in ev.groupby(["event", "measure"], sort=False):
        for dname, d in DURATIONS.items():
            rec = {"event": e, "measure": m, "duration": dname, "duration_h": d}
            for s, col, fac in [("water", "pe_hours_without_water", 1.0), ("wastewater", "sewage_m3", 3.6),
                                ("power", "customer_hours", 1.0)]:
                x = g[g.sector.eq(s) & ~g.conditional]
                rec[col] = float(sum(v * fac * max(0.0, d - o) for v, o in zip(x.value, x.onset_h)))
            cum.append(rec)
    cum = pd.DataFrame(cum)
    cum.to_csv(a.out / "extreme_events_cumulative.csv", index=False)
    summary = {
        "storage_h": STORAGE_H, "fuel_autonomy_h": FUEL_H, "durations_h": DURATIONS,
        "backup_mw": {"water_stations": water_backup_mw, "lift_stations": lift_backup_mw, "dh_pumps": dh_pump_mw,
                      "all_pumped_facilities": all_backup_mw},
        "population": pop, "customers": customers, "sanitary_lps": sanitary_lps, "dh_customers": dh_customers,
        "events": ev.to_dict(orient="records"), "cumulative": cum.to_dict(orient="records"),
    }
    (a.out / "extreme_events_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    pd.set_option("display.width", 200)
    print(ev.to_string())
    print(cum.pivot_table(index=["event", "measure"], columns="duration_h",
                          values=["pe_hours_without_water", "sewage_m3"], sort=False).round(0).to_string())
    print(json.dumps(summary["backup_mw"], indent=1))


if __name__ == "__main__":
    main()
