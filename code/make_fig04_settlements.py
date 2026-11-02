#!/usr/bin/env python3
"""Fig. 4 -- Networks generated for three settlement forms (small multiples).

Rows: settlement forms; columns: sectors.  Every panel of a row uses the same
extent, base map, and building layer, so differences between columns are
differences between sector construction rules, and differences between rows
are responses to urban form.  Building laterals/service drops are omitted.
Usage: python make_fig04_settlements.py --repo /path/to/Urban4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import u4style as S
import u4map as M

S.apply()
ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
args = ap.parse_args()
R = args.repo
BC = R / "outputs" / "benchmark_cases"
metrics = pd.read_csv(R / "reference_results" / "benchmark_case_metrics.csv")

cases = [("dense_urban", "(a) Dense urban", "Schweinfurt centre"),
         ("semi_urban", "(b) Semi-urban", "Bergl/Gartenstadt"),
         ("rural_peripheral", "(c) Rural", "Dittelbrunn")]
sectors = ["Electricity", "Drinking water", "Wastewater", "District heating"]

man = {c: json.loads((BC / c / "manifest.json").read_text()) for c, _, _ in cases}
aspects = []
for c, _, _ in cases:
    x0, y0, x1, y1 = man[c]["bbox"]
    aspects.append((y1 - y0) / ((x1 - x0) * np.cos(np.radians(M.LAT0))))

col_w = (S.TEXT_W - 0.30) / 4
heights = [col_w * a for a in aspects]
fig_h = sum(heights) + 0.26 + 3 * 0.10 + 0.30
fig = plt.figure(figsize=(S.TEXT_W, fig_h))
left_in, top_in, row_gap, legend_in = 0.30, 0.26, 0.10, 0.30

def metric(case, sector, col):
    d = metrics[(metrics.case_id == case) & (metrics.sector == sector)]
    return float(d[col].iloc[0]) if len(d) else np.nan

y_cursor = fig_h - top_in
for r, (case, label, form) in enumerate(cases):
    d = BC / case
    bld = pd.read_csv(d / "common_building_ledger.csv")
    e = pd.read_csv(d / "electricity_links.csv"); tr = pd.read_csv(d / "electricity_transformers.csv")
    wl = pd.read_csv(d / "drinking_water_links.csv"); wn = pd.read_csv(d / "drinking_water_nodes.csv")
    sl = pd.read_csv(d / "wastewater_links.csv"); sn = pd.read_csv(d / "wastewater_nodes.csv")
    hc = pd.read_csv(d / "district_heating_corridors.csv"); hs = pd.read_csv(d / "district_heating_sources.csv")
    h_in = heights[r]
    y_cursor -= h_in
    density = metric(case, "electricity", "building_density_per_km2")
    for c in range(4):
        ax = fig.add_axes([(left_in + c * col_w + 0.02) / S.TEXT_W, y_cursor / fig_h,
                           (col_w - 0.04) / S.TEXT_W, h_in / fig_h])
        M.basemap(ax, str(R), bld, local=True, building_size=0.25, road_lw=0.5)
        if c == 0:
            M.draw(ax, M.segments(e[e.asset_type.eq("lv_feeder")]), S.POWER, 0.45, z=3)
            M.draw(ax, M.segments(e[e.asset_type.eq("mv_backbone")]), S.MV, 1.0, z=4)
            ax.scatter(tr.lon, tr.lat, s=9, marker="s", facecolor="white", edgecolor=S.MV, linewidths=0.7, zorder=6)
            txt = f"{metric(case, 'electricity', 'route_length_km'):.0f} km, {len(tr)} transformers"
        elif c == 1:
            M.draw(ax, M.segments(wl[wl.link_type.eq("distribution_main")]), S.WATER, 0.5, z=3)
            src = wn[wn.node_type.eq("reservoir")]
            ax.scatter(src.lon, src.lat, s=34, marker="o", facecolor=S.WATER, edgecolor=S.INK, linewidths=0.7, zorder=8)
            txt = f"{metric(case, 'drinking_water', 'route_length_km'):.0f} km, {int(metric(case, 'drinking_water', 'cycle_rank'))} loops"
        elif c == 2:
            M.draw(ax, M.segments(sl[sl.link_type.eq("gravity_main")]), S.SEWER, 0.5, z=3)
            fm = sl[sl.link_type.eq("force_main")]
            M.draw(ax, M.segments(fm), S.SHARED, 0.8, ls=(0, (2.2, 1.2)), z=4, capstyle="butt")
            out = sn[sn.node_type.eq("outfall")]
            ax.scatter(out.lon, out.lat, s=40, marker="v", facecolor=S.SEWER, edgecolor="white", linewidths=0.6, zorder=8,
                       clip_on=False)
            txt = f"{metric(case, 'wastewater', 'route_length_km'):.0f} km, average path {metric(case, 'wastewater', 'average_service_route_km'):.1f} km"
        else:
            if len(hc):
                M.draw(ax, M.segments(hc[hc.link_type.eq("route")]), S.HEAT, 0.9, z=3)
                M.draw(ax, M.segments(hc[hc.link_type.eq("building_service")]), S.HEAT, 0.3, alpha=0.7, z=3)
                ax.scatter(hs.lon, hs.lat, s=60, marker="*", facecolor=S.HEAT, edgecolor=S.INK, linewidths=0.6, zorder=8)
                n = int(metric(case, "district_heating", "district_heat_contract_equivalents"))
                txt = f"{metric(case, 'district_heating', 'route_length_km'):.0f} km, {n:,} connections"
            else:
                txt = "No DH: heat demand too low"
        M.frame(ax, man[case]["bbox"])
        ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=6.2, ha="left", va="bottom", color=S.INK, zorder=40,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#AEB4BB", lw=0.4, alpha=0.93))
        if r == 0:
            ax.set_title(sectors[c], fontsize=8.0, fontweight="bold", pad=3, color=S.INK)
        if c == 0:
            fig.text(0.075 / S.TEXT_W, (y_cursor + h_in / 2) / fig_h, label,
                     rotation=90, ha="center", va="center", fontsize=7.6, fontweight="bold", color=S.INK)
            fig.text(0.19 / S.TEXT_W, (y_cursor + h_in / 2) / fig_h, form,
                     rotation=90, ha="center", va="center", fontsize=7.0, color=S.INK)
        if c == 3 and r == 0:
            M.north(ax, loc=(0.92, 0.86), fs=6.4)
        if c == 3:
            M.scalebar(ax, 0.5, loc=(0.62, 0.86), fs=6.0)
    y_cursor -= row_gap

# Shared legend
axl = fig.add_axes([0.0, 0.0, 1.0, (legend_in - 0.04) / fig_h]); axl.axis("off")
handles = [M.line_handle(S.POWER, 1.2, label="LV cable"), M.line_handle(S.MV, 1.6, label="MV cable"),
           M.marker_handle("s", "white", S.MV, 4.5, "MV/LV transformer", mew=0.8),
           M.line_handle(S.WATER, 1.2, label="Water main"), M.marker_handle("o", S.WATER, S.INK, 5.0, "Water source"),
           M.line_handle(S.SEWER, 1.2, label="Gravity sewer"), M.line_handle(S.SHARED, 1.2, (0, (2.2, 1.2)), "Pressure pipe"),
           M.marker_handle("v", S.SEWER, "white", 5.5, "Treatment plant"),
           M.line_handle(S.HEAT, 1.4, label="Heat pipe"), M.marker_handle("*", S.HEAT, S.INK, 7.5, "Heat plant")]
axl.legend(handles=handles, loc="center", ncol=5, fontsize=6.6, frameon=False, handlelength=1.8,
           columnspacing=1.2, labelspacing=0.3)
fig.text(0.995, 0.004, "Base data © OpenStreetMap contributors, ODbL", ha="right", va="bottom",
         fontsize=5.0, color=S.MUTED)
S.save(fig, args.out, "Fig04_Settlement_Forms")
print("wrote", args.out / "Fig04_Settlement_Forms.pdf", "height", round(fig_h, 2))
