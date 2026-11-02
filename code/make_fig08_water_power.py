#!/usr/bin/env python3
"""Fig. 8 -- Water--power case study (outputs of run_water_power_case.py).

(a) MV network coloured by the water consequence of each single-line outage,
    with the waterworks, booster stations, lift stations, and grid supply.
(b) Electricity customers disconnected versus population equivalent without
    water, one point per MV line (407 contingencies).
Usage: python make_fig08_water_power.py --repo /path/to/Urban4 --results results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import u4style as S
import u4map as M

S.apply()
ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--results", type=Path, required=True)
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
args = ap.parse_args()
R = args.repo
ms = R / "outputs" / "municipal_scale_v2.5.0"
res = pd.read_csv(args.results / "mv_line_contingencies_geo.csv")
summ = json.loads((args.results / "water_power_summary.json").read_text())
fac = pd.read_csv(args.results / "facility_attachments.csv")
ties = pd.read_csv(args.results / "open_ring_ties.csv")
bld = pd.read_csv(R / "outputs" / "building_sector_demands.csv")
bld = bld[bld.service_eligible.astype(bool)]
pop = summ["population"]

CLASSES = [  # (label, lower bound, upper bound, colour, line width)
    ("No effect on water", -1, 0.5, "#C99AA8", 0.6),
    ("Fewer than 1,000 people", 0.5, 1000, "#6FB1E3", 1.4),
    ("1,000–3,000 people (one zone)", 1000, 3000, "#3F8FD2", 1.8),
    (f"Whole city (about {round(pop, -3):,.0f} people)", 3000, 1e9, "#08306B", 2.6),
]

def cls(v):
    for i, (_, lo, hi, _, _) in enumerate(CLASSES):
        if lo < v <= hi:
            return i
    return 0

res["cls"] = res.M0_w_pop_equiv.map(cls)
spur = res.loc[res.e_loads_lost.eq(res.loc[res.M0_w_pop_equiv >= pop - 1, "e_loads_lost"].min())
               & (res.M0_w_pop_equiv >= pop - 1)].iloc[0]

fig = plt.figure(figsize=(S.TEXT_W, 3.05))
gs = GridSpec(1, 2, figure=fig, width_ratios=[0.86, 1.0], left=0.005, right=0.985, top=0.90, bottom=0.30,
              wspace=0.22)

# ------------------------------------------------------------------ (a)
ax = fig.add_subplot(gs[0, 0])
xs = pd.concat([bld.lon, fac.lon]); ys = pd.concat([bld.lat, fac.lat])
dx, dy = xs.max() - xs.min(), ys.max() - ys.min()
bbox = (xs.min() - 0.01 * dx, ys.min() - 0.02 * dy, xs.max() + 0.01 * dx, ys.max() + 0.02 * dy)
M.basemap(ax, str(R), bld, building_size=0.12, road_lw=0.35)
tie_segs = [[tuple(p) for p in json.loads(g)["coordinates"]] for g in ties.geometry_json]
M.draw(ax, tie_segs, "#8E7FA6", 0.45, ls=(0, (2.2, 1.4)), z=2.5)
for i, (_, _, _, c, lw) in enumerate(CLASSES):
    M.draw(ax, M.segments(res[res.cls.eq(i)]), c, lw, z=3 + i)
w = fac[fac.kind.eq("water")]
ww = w[w.role.eq("waterworks")]; bo = w[w.role.eq("booster")]
ls = fac[fac.kind.eq("wastewater")]
ax.scatter(ls.lon, ls.lat, s=16, marker="^", facecolor=S.SHARED, edgecolor="white", linewidths=0.5, zorder=9)
ax.scatter(bo.lon, bo.lat, s=34, marker="h", facecolor="white", edgecolor=S.WATER, linewidths=1.0, zorder=10)
ax.scatter(ww.lon, ww.lat, s=60, marker="o", facecolor=S.WATER, edgecolor=S.INK, linewidths=0.8, zorder=11)
seg = M.segments(pd.DataFrame([spur]))
if seg:
    mx, my = np.mean([p[0] for p in seg[0]]), np.mean([p[1] for p in seg[0]])
    ax.annotate(f"{spur.length_km:.1f}-km branch line:\nonly {int(spur.e_loads_lost)} customers,\nbut whole city\nwithout water",
                (mx, my), xytext=(bbox[0] + 0.62 * (bbox[2] - bbox[0]), bbox[1] + 0.12 * (bbox[3] - bbox[1])),
                fontsize=6.0, ha="left", va="center", color=S.INK, zorder=20,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=S.INK, shrinkA=1, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#AEB4BB", lw=0.4, alpha=0.95))
M.frame(ax, bbox)
M.scalebar(ax, 2.0, loc=(0.05, 0.04), fs=6.0)
M.north(ax, loc=(0.94, 0.9), fs=6.4)
S.panel_title(ax, "(a) People without water if a section fails")
handles = [M.line_handle(c, lw + 0.4, label=lab) for lab, _, _, c, lw in CLASSES]
handles += [M.line_handle("#8E7FA6", 0.9, ls=(0, (2.2, 1.4)), label="Tie cable (normally open)")]
handles += [M.marker_handle("o", S.WATER, S.INK, 5.5, "Waterworks"),
            M.marker_handle("h", "white", S.WATER, 5.5, "Booster station", mew=1.0),
            M.marker_handle("^", S.SHARED, "white", 5.0, "Sewage lift station")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.01, -0.02), ncol=2, fontsize=6.2, frameon=False,
          handlelength=1.8, columnspacing=0.8, labelspacing=0.3, borderaxespad=0.0)

# ------------------------------------------------------------------ (b)
axb = fig.add_subplot(gs[0, 1])
for i, (lab, _, _, c, _) in enumerate(CLASSES):
    d = res[res.cls.eq(i)]
    axb.scatter(d.e_loads_lost.clip(lower=1), d.M0_w_pop_equiv.clip(lower=0), s=10 if i else 6,
                c=c, edgecolor="white" if i else "none", linewidths=0.4, zorder=3 + i)
axb.set_xscale("log")
axb.set_yscale("symlog", linthresh=100)
axb.set_xlim(8, 4e4)
axb.set_xticks([10, 100, 1000, 10000])
axb.set_xticklabels(["10", "100", "1,000", "10,000"])
axb.set_ylim(-20, 1.6e5)
axb.set_yticks([0, 100, 1000, 10000, 100000])
axb.set_yticklabels(["0", "100", "1,000", "10,000", "100,000"])
axb.set_xlabel("Electricity customers cut off")
axb.set_ylabel("People without water")
axb.annotate(f"{spur.length_km:.1f}-km branch line", (spur.e_loads_lost, spur.M0_w_pop_equiv), xytext=(8, -12),
             textcoords="offset points", fontsize=6.0, ha="left", va="top",
             arrowprops=dict(arrowstyle="-", lw=0.5, color=S.MUTED, shrinkA=0, shrinkB=2))
axb.text(0.97, 0.20, f"407 line sections\nRank correlation {summ['spearman_electric_vs_water']:.2f}",
         transform=axb.transAxes, ha="right", va="center", fontsize=6.3, color=S.INK)
S.panel_title(axb, "(b) Grid impact versus water impact")

S.save(fig, args.out, "Fig08_Water_Power_Case")
print("wrote", args.out / "Fig08_Water_Power_Case.pdf")
