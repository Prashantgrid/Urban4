#!/usr/bin/env python3
"""Fig. 7 -- Heat--power case study (outputs of run_heat_power_case.py).

(a) Map of the district-heating (DH) network, its two sources, the intertie
    segments that bind when GKS is unavailable, and the transformer areas that
    host DH customers, coloured by loading in the heat-pump scenario S1 at the
    -14 degC design state.
(b) Added electrical demand versus outdoor temperature.
(c) Transformer areas above 100 % loading versus outdoor temperature.
(d) Largest DH pipe velocity versus outdoor temperature for three source and
    intertie configurations.
Usage: python make_fig07_heat_power.py --repo /path/to/Urban4 --results results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import u4style as S
import u4map as M

S.apply()
ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--results", type=Path, required=True)
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
args = ap.parse_args()

ms = args.repo / "outputs" / "municipal_scale_v2.5.0"
res = pd.read_csv(args.results / "heat_power_states.csv")
tr = pd.read_csv(args.results / "transformer_design_state.csv")
cap = pd.read_csv(args.results / "dh_hydraulic_capability.csv")
summ = json.loads((args.results / "heat_power_summary.json").read_text())
corr = pd.read_csv(ms / "district_heating_corridors_municipal.csv")
src = pd.read_csv(ms / "district_heating_sources_municipal.csv").set_index("map_id")
dh = pd.read_csv(ms / "district_heating_services_municipal.csv")
bld = pd.read_csv(args.repo / "outputs" / "building_sector_demands.csv")
bld = bld[bld.service_eligible.astype(bool)]

C_S0, C_S1, C_S2, C_S3 = S.HEAT, S.POWER, S.SHARED, "#2A9D8F"
LOAD_CLASSES = [(100, 200, "#F4B6A6"), (200, 400, "#E0664F"), (400, 1e9, "#8E1F12")]

fig = plt.figure(figsize=(S.TEXT_W, 4.05))
gs = GridSpec(3, 2, figure=fig, width_ratios=[1.0, 1.08], height_ratios=[1, 1, 1],
              left=0.005, right=0.985, top=0.955, bottom=0.10, wspace=0.16, hspace=0.30)

# ------------------------------------------------------------------ (a) map
from matplotlib.gridspec import GridSpecFromSubplotSpec
gl = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 0], height_ratios=[1.0, 0.19], hspace=0.02)
ax = fig.add_subplot(gl[0])
axl = fig.add_subplot(gl[1]); axl.axis("off")
dist = corr[corr.edge_role.eq("distribution")]
tie = corr[corr.edge_role.eq("source_intertie")]
tie300 = tie[tie.diameter_mm.eq(300)]
xs = pd.concat([corr.from_lon, corr.to_lon, tr.loc[tr.dh_buildings > 0, "lon"]])
ys = pd.concat([corr.from_lat, corr.to_lat, tr.loc[tr.dh_buildings > 0, "lat"]])
pad_x, pad_y = 0.010, 0.006
bbox = (xs.min() - pad_x, ys.min() - pad_y, xs.max() + pad_x, ys.max() + pad_y)
M.basemap(ax, str(args.repo), bld, building_size=0.35)
M.draw(ax, M.segments(dist), S.HEAT, 0.9, z=3)
M.draw(ax, M.segments(tie), "#9A5B00", 1.6, z=4)
M.draw(ax, M.segments(tie300), S.INK, 2.0, z=5)

other = tr[tr.dh_buildings.eq(0)]
ax.scatter(other.lon, other.lat, s=7, marker="s", facecolor="white", edgecolor="#8B939C", linewidths=0.5, zorder=7)
host = tr[tr.dh_buildings > 0]
col = host["design_loading_pct_S1_HP"]
ok = host[col <= 100]
ax.scatter(ok.lon, ok.lat, s=15, marker="s", facecolor="#E9ECEF", edgecolor=S.POWER, linewidths=0.8, zorder=8)
for lo, hi, c in LOAD_CLASSES:
    sel = host[(col > lo) & (col <= hi)]
    ax.scatter(sel.lon, sel.lat, s=22, marker="s", facecolor=c, edgecolor="white", linewidths=0.6, zorder=9)
for sid, label, off in [("H1", "GKS", (-6, -6)), ("H2", "Boiler\nplant", (-7, 4))]:
    ax.scatter(src.at[sid, "lon"], src.at[sid, "lat"], s=120, marker="*", facecolor=S.HEAT,
               edgecolor=S.INK, linewidths=0.7, zorder=12)
    ax.annotate(label, (src.at[sid, "lon"], src.at[sid, "lat"]), xytext=off, textcoords="offset points",
                fontsize=6.6, fontweight="bold", ha="right" if off[0] < 0 else "left",
                va="top" if off[1] < 0 else "bottom", color=S.INK, zorder=13,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
M.frame(ax, bbox)
ax.set_aspect(ax.get_aspect(), adjustable="datalim")
M.scalebar(ax, 1.0, loc=(0.05, 0.04))
M.north(ax, loc=(0.94, 0.955))
M.attribution(ax)
S.panel_title(ax, "(a) Transformer loading with heat pumps ( \u221214 °C)")
handles = [
    M.line_handle(S.HEAT, 1.4, label="DH pipe"),
    M.line_handle("#9A5B00", 1.8, label="Link pipe between plants"),
    Line2D([0], [0], color=S.INK, lw=2.2, solid_capstyle="butt", label="DN300 heat-link section"),
    M.marker_handle("*", S.HEAT, S.INK, 9, "Heat plant"),
    M.marker_handle("s", "white", "#8B939C", 4.5, "Area without DH customers"),
    M.marker_handle("s", "#E9ECEF", S.POWER, 5.0, "Area with DH customers: ≤100%", mew=0.9),
    M.marker_handle("s", LOAD_CLASSES[0][2], "white", 5.5, "100–200%"),
    M.marker_handle("s", LOAD_CLASSES[1][2], "white", 5.5, "200–400%"),
    M.marker_handle("s", LOAD_CLASSES[2][2], "white", 5.5, ">400%"),
]
# Overlay the DN300 highlight in the legend handle
axl.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.4,
           frameon=False, handlelength=1.8, columnspacing=0.9, handletextpad=0.5, borderaxespad=0.1,
           labelspacing=0.3)

# ------------------------------------------------------------ (b)-(d) states
t = np.sort(res.t_a_c.unique())[::-1]
def series(sc, col):
    d = res[res.scenario.eq(sc)].set_index("t_a_c").reindex(t)
    return d[col].to_numpy(dtype=float)

axb = fig.add_subplot(gs[0, 1])
axc = fig.add_subplot(gs[1, 1], sharex=axb)
axd = fig.add_subplot(gs[2, 1], sharex=axb)

# (b) added electrical demand
axb.plot(t, series("S1_HP", "added_electric_mw"), color=C_S1, label="Heat-pump conversion, COP model")
var = pd.read_csv(args.results / "heat_power_variant_states.csv")
def vseries(mode, f, col):
    d = var[var.ratio_mode.eq(mode) & var.renovation_f.eq(f)].set_index("t_a_c").reindex(t)
    return d[col].to_numpy(dtype=float)
axb.plot(t, vseries("measured", 1.0, "added_electric_mw"), color=C_S1, lw=0.9, ls=(0, (3, 1.6)),
         label="Heat-pump conversion, ratio 0.40")
axb.plot(t, vseries("measured", 0.6, "added_electric_mw"), color=C_S1, lw=0.8, ls=(0, (1, 1.3)),
         label="Ratio 0.40, 40% less heat")
axb.plot(t, series("S2_GKS_outage", "added_electric_mw"), color=C_S2, label="CHP outage + electric backup")
axb.plot(t, series("S0_DH", "added_electric_mw"), color=C_S0, label="Existing DH / reinforced heat link")
axb.set_ylabel("Added load (MW)")
axb.set_ylim(-1, 38)
axb.set_yticks([0, 10, 20, 30])
S.panel_title(axb, "(b) Added electrical load")
axb.legend(loc="upper left", fontsize=6.4, handlelength=1.8, labelspacing=0.25, borderaxespad=0.1,
           title="Same lines in (b) and (c)", title_fontsize=6.0, alignment="left")

# (c) overloaded transformer areas
axc.step(t, series("S0_DH", "scr_overloaded_trafos"), where="mid", color=C_S0, lw=2.6, label="S0 and S3")
axc.step(t, series("S1_HP", "scr_overloaded_trafos"), where="mid", color=C_S1, label="S1")
axc.step(t, vseries("measured", 1.0, "overloaded_areas"), where="mid", color=C_S1, lw=0.9, ls=(0, (3, 1.6)),
         label="S1, measured")
axc.step(t, vseries("measured", 0.6, "overloaded_areas"), where="mid", color=C_S1, lw=0.8, ls=(0, (1, 1.3)),
         label="S1, measured, 40 % renov.")
axc.step(t, series("S2_GKS_outage", "scr_overloaded_trafos"), where="mid", color=C_S2, lw=1.1, label="S2")
axc.axvspan(-2.5, -14.5, color="#EEF0F2", zorder=0, lw=0)
axc.text(-8.25, 22.0, "Heat pumps (COP): AC does not converge", ha="center", va="top", fontsize=6.2, color=S.MUTED)
axc.set_ylabel("Overloaded areas")
axc.set_ylim(-1, 23)
axc.set_yticks([0, 5, 10, 15, 20])
S.panel_title(axc, "(c) Overloaded transformer areas (of 220)")


# (d) DH velocity
axd.plot(t, series("S0_DH", "dh_v_max_m_s"), color=C_S0, label="Existing district heating")
d = cap[cap["case"].eq("H2_only")].sort_values("phi")
d = d[d.phi >= summ["h2_only_max_fraction"] - 0.03]
axd.plot(15 - d.phi * 29.0, d.v_max_m_s, color=C_S2, lw=0.9, ls=(0, (1, 1.2)), label="Boiler plant only, no limit")
axd.plot(t, series("S2_GKS_outage", "dh_v_max_m_s"), color=C_S2, label="CHP outage, existing heat link")
axd.plot(t, series("S3_GKS_outage_DN400", "dh_v_max_m_s"), color=C_S3, ls=(0, (3, 1.6)),
         label="CHP outage, reinforced heat link")
axd.axhline(2.0, color=S.INK, lw=0.7, ls=(0, (1, 1.5)))
axd.text(14.6, 2.06, "2.0 m/s limit", fontsize=6.2, color=S.INK, va="bottom", ha="left")
t_star = summ["h2_only_threshold_c"]
axd.plot([t_star], [2.0], "o", ms=4.5, mfc=C_S2, mec="white", mew=0.6, zorder=5)
axd.annotate(f"about {t_star:.0f} °C".replace("-", "\u2212"), (t_star, 2.0), xytext=(4, -4), textcoords="offset points",
             ha="left", va="top", fontsize=6.4, color=S.INK)
axd.set_ylabel("Flow speed (m/s)")
axd.set_ylim(0, 3.7)
axd.set_yticks([0, 1, 2])
axd.set_xlim(15.5, -14.5)
axd.set_xticks([15, 10, 5, 0, -5, -10, -14])
axd.set_xlabel("Outdoor temperature (°C)")
S.panel_title(axd, "(d) Highest flow speed in the DH network")
axd.legend(loc="upper left", fontsize=6.3, handlelength=1.8, labelspacing=0.2, borderaxespad=0.1, ncol=2,
           columnspacing=1.0)
for a in (axb, axc):
    plt.setp(a.get_xticklabels(), visible=False)

S.save(fig, args.out, "Fig07_Heat_Power_Case")
print("wrote", args.out / "Fig07_Heat_Power_Case.pdf")
