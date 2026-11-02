#!/usr/bin/env python3
"""Fig. 9 -- Extreme events (outputs of run_flood_case.py and run_extreme_events.py).

(a) Bavarian flood-hazard extents HQ100 and HQextrem (LfU / WWA Bad Kissingen)
    over the MV network, the MV/LV stations switched off by the flood, and the
    facilities of all four sectors along the Main.
(b) When each sector loses service in a city-wide blackout and in the two
    floods, per measure, against documented German event durations.
Usage: python make_fig09_extreme_events.py --repo /path/to/Urban4 --results results/water_power_case
                                           --flood data/LfU_Schweinfurt_HQ100_HQextrem.gpkg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pandapower as pp
import matplotlib.pyplot as plt
from shapely.geometry import Point, box

import u4style as S
import u4map as M

S.apply()
ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--results", type=Path, required=True)
ap.add_argument("--flood", type=Path, required=True)
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
args = ap.parse_args()
R = args.repo
ms = R / "outputs" / "municipal_scale_v2.5.0"
ext_ev = pd.read_csv(args.results / "extreme_events.csv")
ROWS = [("Blackout (grid supply point lost)", "no backup", "No backup"),
        ("Blackout (grid supply point lost)", "water stations on backup, 72 h fuel", "Backup at water\nstations, 72 h fuel"),
        ("Blackout (grid supply point lost)", "all pumped facilities on backup, refuelled", "Refuelled backup\nat all pumps"),
        ("HQ100 flood", "as mapped", "No protection"),
        ("HQ100 flood", "waterworks site and supply station protected", "Waterworks and its\nstation protected"),
        ("HQextrem flood", "as mapped", "No protection")]
GROUPS = {"Blackout (grid supply point lost)": "City-wide blackout", "HQ100 flood": "100-year flood (HQ100)",
          "HQextrem flood": "Extreme flood (HQextrem)"}


def bar_text(r):
    """Short plain-language label for one service bar."""
    v, onset, cond = float(r.value), float(r.onset_h), bool(r.conditional)
    if r.sector == "power":
        return "all customers" if v >= 20000 else f"{v:,.0f} customers"
    if r.sector == "water":
        days = onset / 24.0
        return f"whole city, after {days:.0f} day" + ("s" if days > 1.5 else "")
    if r.sector == "wastewater":
        if cond:
            return "treated only if its CHP runs islanded"
        what = "not treated" if "untreated" in str(r.label) else "not pumped"
        return ("all sewage " if v > 100 else "half of sewage ") + what
    if cond and "Blackout" in str(r.event):
        return "only if the plants run islanded"
    if cond:
        return "short only below about \u22129 \u00b0C"
    return "all DH customers"
SECT = [("power", S.POWER, "white"), ("water", S.WATER, "white"), ("wastewater", S.SEWER, "white"),
        ("heat", S.HEAT, S.INK)]
LANE, LGAP, RGAP, HEAD = 7.8, 1.3, 4.5, 11.5          # points
# vertical layout of panel (b), top-down in points
items, y = [], 0.0
last_group = None
for evn, meas, lab in ROWS:
    if evn != last_group:
        items.append(("head", GROUPS[evn], y, None))
        y += HEAD
        last_group = evn
    d = ext_ev[ext_ev.event.eq(evn) & ext_ev.measure.eq(meas)]
    lanes = [(sec, col, tc, d[d.sector.eq(sec)].iloc[0]) for sec, col, tc in SECT if d.sector.eq(sec).any()]
    top = y
    for k, ln in enumerate(lanes):
        items.append(("lane", ln, y, lab if k == 0 else None))
        y += LANE + (LGAP if k < len(lanes) - 1 else 0.0)
    items.append(("label", lab, (top + y) / 2.0, None))
    y += RGAP
PLOT_PT = y - RGAP
TOP, REF_PT, LEG_H = 0.17, 19.0, 0.70
MAP_W = 3.40
MAP_H = (PLOT_PT + REF_PT) / 72.0
FIG_H = TOP + MAP_H + LEG_H
LON0, LON1, LATC = 10.168, 10.302, 50.0325
lat_span = MAP_H / MAP_W * (LON1 - LON0) * np.cos(np.radians(50.05))
BBOX = (LON0, LATC - lat_span / 2, LON1, LATC + lat_span / 2)
clip = box(*BBOX)
ext = {k: gpd.read_file(args.flood, layer=k).to_crs(4326).clip(clip) for k in ["HQ100", "HQextrem"]}
assets = pd.read_csv(args.results / "flood_assets.csv").set_index("asset")
res = pd.read_csv(args.results / "mv_line_contingencies_geo.csv")
fac = pd.read_csv(args.results / "facility_attachments.csv").set_index("facility")
summ = json.loads((args.results / "flood_summary.json").read_text())
bld = pd.read_csv(R / "outputs" / "building_sector_demands.csv")
bld = bld[bld.service_eligible.astype(bool)]
net = pp.from_json(str(ms / "electricity_municipal_pandapower.json"))
geo = net.bus.geo.map(lambda g: json.loads(g)["coordinates"] if isinstance(g, str) else g["coordinates"])
st = pd.DataFrame({"bus": net.trafo.hv_bus.to_numpy()})
st["lon"] = [geo.at[b][0] for b in st.bus]
st["lat"] = [geo.at[b][1] for b in st.bus]
pts = gpd.GeoSeries([Point(x, y) for x, y in zip(st.lon, st.lat)], crs=4326)
st["HQ100"] = pts.within(ext["HQ100"].union_all()).to_numpy()
st["HQextrem"] = pts.within(ext["HQextrem"].union_all()).to_numpy()

HQX, HQ1 = "#CFE3F3", "#7FB2DE"
FLOOD_RED = "#C0392B"
fig = plt.figure(figsize=(S.TEXT_W, FIG_H))
ax = fig.add_axes([0.02 / S.TEXT_W, LEG_H / FIG_H, MAP_W / S.TEXT_W, MAP_H / FIG_H])
M.basemap(ax, str(R), bld, building_size=0.10, road_lw=0.35)
ext["HQextrem"].plot(ax=ax, color=HQX, edgecolor="#9CC3E4", linewidth=0.3, zorder=0.6)
ext["HQ100"].plot(ax=ax, color=HQ1, edgecolor="#5B97CC", linewidth=0.3, alpha=0.9, zorder=0.7)
M.draw(ax, M.segments(res), S.MV, 0.55, z=2, alpha=0.75)
sx = st[st.HQextrem & ~st.HQ100]; s1 = st[st.HQ100]
ax.scatter(sx.lon, sx.lat, s=9, marker="s", facecolor="white", edgecolor=FLOOD_RED, linewidths=0.8, zorder=6)
ax.scatter(s1.lon, s1.lat, s=11, marker="s", facecolor=FLOOD_RED, edgecolor="white", linewidths=0.4, zorder=7)

ls = fac[fac.kind.eq("wastewater")]
ls_fl = ls.index.isin(assets.index[assets.HQextrem])
ax.scatter(ls.lon[~ls_fl], ls.lat[~ls_fl], s=18, marker="^", facecolor=S.SHARED, edgecolor="white", linewidths=0.5, zorder=8)
ax.scatter(ls.lon[ls_fl], ls.lat[ls_fl], s=22, marker="^", facecolor=S.SHARED, edgecolor=FLOOD_RED, linewidths=0.9, zorder=8)
ww = assets.loc["WZ1"]
ax.scatter([ww.lon], [ww.lat], s=58, marker="o", facecolor=S.WATER, edgecolor=S.INK, linewidths=0.8, zorder=10)
tp = assets.loc["WWTP"]
ax.scatter([tp.lon], [tp.lat], s=46, marker="D", facecolor=S.SEWER, edgecolor="white", linewidths=0.6, zorder=10)
g1, g2 = assets.loc["H1"], assets.loc["H2"]
ax.scatter([g1.lon], [g1.lat], s=80, marker="*", facecolor=S.HEAT, edgecolor=S.INK, linewidths=0.5, zorder=10)
ax.scatter([g2.lon], [g2.lat], s=34, marker="p", facecolor="white", edgecolor=S.HEAT, linewidths=1.1, zorder=10)
sp = assets.loc["SOURCE"]
ax.scatter([sp.lon], [sp.lat], s=36, marker="d", facecolor=S.INK, edgecolor="white", linewidths=0.5, zorder=10)
tr = net.bus.index[net.bus.name == fac.at["WZ1", "bus_name"]][0]


def label(x, y, text, dx, dy, ha="left"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.4, ha=ha, va="center",
                color=S.INK, zorder=20, linespacing=1.05,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=S.INK, shrinkA=0, shrinkB=3),
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


label(ww.lon, ww.lat, "Waterworks and\nits supply station", 14, -16)
ax.scatter([geo.at[tr][0]], [geo.at[tr][1]], s=24, marker="s", facecolor=FLOOD_RED, edgecolor=S.INK, linewidths=0.6, zorder=9)
label(tp.lon, tp.lat, "Treatment\nplant", -12, -14, ha="right")
label(g1.lon, g1.lat, "GKS", 16, -12)
label(g2.lon, g2.lat, "Boiler plant", -12, 13, ha="right")
label(ls.at["S4", "lon"], ls.at["S4", "lat"], "Main lift\nstation", -2, 34, ha="left")
label(sp.lon, sp.lat, "Grid supply\npoint", -10, 12, ha="right")
ax.set_xlabel(""); ax.set_ylabel("")
M.frame(ax, BBOX)
M.scalebar(ax, 1.0, loc=(0.05, 0.06), fs=6.0)
M.north(ax, loc=(0.06, 0.34), fs=6.4)
ev = summ["events"]

from matplotlib.patches import Patch
handles = [Patch(facecolor=HQ1, edgecolor="#5B97CC", lw=0.4, label="100-year flood (HQ100)"),
           Patch(facecolor=HQX, edgecolor="#9CC3E4", lw=0.4, label="Extreme flood (HQextrem)"),
           M.line_handle(S.MV, 1.0, label="MV line"),
           M.marker_handle("s", FLOOD_RED, "white", 4.6, "Station flooded at HQ100"),
           M.marker_handle("s", "white", FLOOD_RED, 4.2, "Station flooded at HQextrem only", mew=0.9),
           M.marker_handle("o", S.WATER, S.INK, 5.2, "Waterworks"),
           M.marker_handle("^", S.SHARED, "white", 5.0, "Lift station (red edge: flooded)"),
           M.marker_handle("D", S.SEWER, "white", 4.4, "Treatment plant"),
           M.marker_handle("*", S.HEAT, S.INK, 7.0, "GKS (CHP plant)", mew=0.5),
           M.marker_handle("p", "white", S.HEAT, 5.0, "Boiler plant", mew=1.1)]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.005, -0.01), ncol=2, fontsize=6.4, frameon=False,
          handlelength=1.5, columnspacing=0.7, labelspacing=0.22, borderaxespad=0.0)
ax.text(0.995, 0.005, "Flood extents: LfU Bayern; base data © OpenStreetMap contributors, ODbL",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=4.6, color=S.MUTED, zorder=31)

# ------------------------------------------------------------------ (b) service timelines
T0, T1 = 0.5, 1300.0
LABEL_W = 1.02                                  # inches for the row labels
gx0 = 0.02 + MAP_W + 0.22 + LABEL_W
gw = S.TEXT_W - gx0 - 0.05
axg = fig.add_axes([gx0 / S.TEXT_W, LEG_H / FIG_H, gw / S.TEXT_W, (PLOT_PT / 72.0) / FIG_H])
axg.set_xscale("log")
axg.set_xlim(T0, T1)
axg.set_ylim(PLOT_PT, 0)                         # points, top-down
for kind, obj, yy, _ in items:
    if kind == "head":
        axg.text(-LABEL_W / gw, yy + HEAD * 0.52, obj, transform=axg.get_yaxis_transform(), fontsize=7.2,
                 fontweight="bold", ha="left", va="center", color=S.INK, clip_on=False)
        if yy > 0:
            axg.plot([-LABEL_W / gw, 1.0], [yy - RGAP / 2 + 0.3] * 2, transform=axg.get_yaxis_transform(),
                     color="#B7BDC4", lw=0.5, clip_on=False, zorder=1)
    elif kind == "label":
        axg.text(-0.03, yy, obj, transform=axg.get_yaxis_transform(), fontsize=6.9, ha="right", va="center",
                 color=S.INK, linespacing=1.05, clip_on=False)
    else:
        sec, col, tc, r = obj
        x0 = max(float(r.onset_h), T0)
        cond = bool(r.conditional)
        axg.add_patch(plt.Rectangle((x0, yy), T1 - x0, LANE, facecolor=col, alpha=0.28 if cond else 0.90,
                                    edgecolor=col if cond else "none", lw=0.4, hatch="////" if cond else None,
                                    zorder=3))
        txt = bar_text(r)
        if float(r.onset_h) > 0:   # bar starts late: label sits in the empty space before it
            axg.text(x0 / 1.12, yy + LANE / 2 + 0.2, txt, fontsize=6.2, va="center", ha="right",
                     color=col, fontweight="bold", zorder=5, clip_on=True)
        else:
            axg.text(x0 * 1.16, yy + LANE / 2 + 0.2, txt, fontsize=6.2, va="center", ha="left",
                     color=S.INK if cond else tc, zorder=5, clip_on=True)
for tt, lab in [(31, "K\u00f6penick\n2019"), (120, "Berlin\n2026"), (840, "Ahr\n2021")]:
    axg.axvline(tt, color=S.INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=4)
    axg.text(tt, -2.0, lab, fontsize=6.3, ha="center", va="bottom", color=S.INK, linespacing=0.95)
axg.set_yticks([])
axg.spines["left"].set_visible(False)
axg.set_xticks([1, 6, 24, 72, 168, 840])
axg.set_xticklabels(["1 h", "6 h", "1 d", "3 d", "1 wk", "5 wk"])
axg.minorticks_off()
axg.grid(False)
axg.set_xlabel("Time after the event begins", labelpad=2)
fig.text((gx0 - LABEL_W) / S.TEXT_W, 1.0 - 0.02 / FIG_H, "(b) Time until service is lost", fontsize=8.0,
         fontweight="bold", ha="left", va="top", color=S.INK)
fig.text(0.02 / S.TEXT_W, 1.0 - 0.02 / FIG_H, "(a) Sites in the official flood areas", fontsize=8.0,
         fontweight="bold", ha="left", va="top", color=S.INK)
from matplotlib.patches import Patch
hg = [Patch(facecolor=c, edgecolor="none", label=l) for (sname, c, _), l in
      zip(SECT, ["Electricity", "Drinking water", "Wastewater", "District heating"])]
hg.append(Patch(facecolor="white", edgecolor=S.MUTED, hatch="////", lw=0.4, label="Uncertain (see caption)"))
fig.legend(handles=hg, loc="lower left", bbox_to_anchor=((gx0 - LABEL_W) / S.TEXT_W, 0.0), ncol=3, fontsize=6.6,
           frameon=False, handlelength=1.3, columnspacing=0.9, labelspacing=0.25, borderaxespad=0.1)
S.save(fig, args.out, "Fig09_Extreme_Events")
print("wrote", args.out / "Fig09_Extreme_Events.pdf")
