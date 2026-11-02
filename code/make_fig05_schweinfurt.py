#!/usr/bin/env python3
"""Fig. 5 -- Accepted municipality-scale Schweinfurt networks (four sectors).

Every panel shows the same extent and base map; the network is coloured by the
state computed by its native solver in the accepted design state:
  (a) electricity: transformer-area loading from the pandapower AC solution,
  (b) drinking water: design-hour service pressure from EPANET (WNTR),
  (c) wastewater: gravity mains, force mains, lift stations, treatment works,
  (d) district heating: pipes by supply-temperature class, documented sources.
Usage: python make_fig05_schweinfurt.py --repo /path/to/Urban4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp

import u4style as S
import u4map as M

S.apply()
ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
args = ap.parse_args()
R = args.repo
ms = R / "outputs" / "municipal_scale_v2.5.0"
integ = R / "outputs" / "integrated_service_resolved"

bld = pd.read_csv(R / "outputs" / "building_sector_demands.csv")
bld = bld[bld.service_eligible.astype(bool)]
e_links = pd.read_csv(ms / "electricity_links_municipal.csv")
e_nodes = pd.read_csv(integ / "electricity_nodes.csv")
e_man = json.loads((ms / "electricity_native_manifest.json").read_text())
w_links = pd.read_csv(ms / "drinking_water_links.csv")
w_res = pd.read_csv(ms / "drinking_water_native_peak_service_results.csv")
w_station = pd.read_csv(ms / "drinking_water_station_municipal.csv").iloc[0]
w_zone = pd.read_csv(ms / "drinking_water_zone_boundaries.csv")
w_man = json.loads((ms / "drinking_water_native_manifest.json").read_text())
s_links = pd.read_csv(ms / "wastewater_links.csv")
s_nodes = pd.read_csv(ms / "wastewater_nodes.csv")
s_st = pd.read_csv(ms / "wastewater_pump_stations_municipal.csv")
s_man = json.loads((ms / "wastewater_station_native_manifest.json").read_text())
h_corr = pd.read_csv(ms / "district_heating_corridors_municipal.csv")
h_src = pd.read_csv(ms / "district_heating_sources_municipal.csv")
h_man = json.loads((ms / "district_heating_native_manifest.json").read_text())
h_aud = pd.read_csv(ms / "district_heating_connectivity_audit.csv").iloc[0]

# AC transformer loading of the accepted municipal electricity model
net = pp.from_json(str(ms / "electricity_municipal_pandapower.json"))
pp.runpp(net, numba=False)
geo = net.bus.geo.map(lambda g: json.loads(g)["coordinates"] if isinstance(g, str) else g["coordinates"])
trafo = pd.DataFrame({"lon": [geo.at[b][0] for b in net.trafo.lv_bus], "lat": [geo.at[b][1] for b in net.trafo.lv_bus],
                      "loading": net.res_trafo.loading_percent.to_numpy(), "sn": net.trafo.sn_mva.to_numpy()})
src_bus = int(net.ext_grid.bus.iloc[0])
grid_xy = geo.at[src_bus]

# Common extent: all service buildings plus documented facilities
xs = pd.concat([bld.lon, pd.Series([w_station.source_coordinate_lon]), h_src.lon, s_st.lon])
ys = pd.concat([bld.lat, pd.Series([w_station.source_coordinate_lat]), h_src.lat, s_st.lat])
dx, dy = xs.max() - xs.min(), ys.max() - ys.min()
bbox = (xs.min() - 0.015 * dx, ys.min() - 0.02 * dy, xs.max() + 0.015 * dx, ys.max() + 0.02 * dy)

from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(S.TEXT_W, 7.05))
gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 0.16, 1.0, 0.16], left=0.004, right=0.996,
              top=0.972, bottom=0.004, wspace=0.03, hspace=0.07)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
leg_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
for la in leg_axes:
    la.axis("off")
for ax in axes:
    M.basemap(ax, str(R), bld, building_size=0.18)
    M.frame(ax, bbox)

LEG = dict(loc="upper left", bbox_to_anchor=(0.0, -0.012), fontsize=6.6, frameon=False, ncol=3,
           borderaxespad=0.1, handlelength=1.7, labelspacing=0.3, handletextpad=0.5, columnspacing=1.1)

# (a) Electricity -----------------------------------------------------------
ax = axes[0]
lv = e_links[e_links.asset_type.eq("lv_feeder")]
mv = e_links[e_links.asset_type.eq("mv_backbone")]
M.draw(ax, M.segments(lv), S.POWER, 0.22, alpha=0.55, z=2)
M.draw(ax, M.segments(mv), S.MV, 0.75, z=3)
bins = [(0, 40, "#FBE3DD"), (40, 60, "#F2A48F"), (60, 75, "#DE5A43"), (75, 100, "#8E1F12")]
for lo, hi, c in bins:
    sel = trafo[(trafo.loading > lo) & (trafo.loading <= hi)]
    ax.scatter(sel.lon, sel.lat, s=8.5, marker="s", facecolor=c, edgecolor=S.INK, linewidths=0.35, zorder=6)
ax.scatter(*grid_xy, s=60, marker="D", facecolor=S.INK, edgecolor="white", linewidths=0.7, zorder=8)
h = [M.line_handle(S.POWER, 1.0, label="LV cable"), M.line_handle(S.MV, 1.4, label="MV cable"),
     M.marker_handle("D", S.INK, "white", 5.5, "Grid supply point")]
h += [M.marker_handle("s", c, S.INK, 5.0, (f"Transformer loading {lo}–{hi}%" if lo == 0 else f"{lo}–{hi}%"), mew=0.4)
      for lo, hi, c in bins]
ax.legend(handles=h, **LEG)
S.panel_title(ax, "(a) Electricity: transformer loading")

# (b) Drinking water --------------------------------------------------------
ax = axes[1]
mains = w_links[w_links.link_type.eq("distribution_main")]
M.draw(ax, M.segments(mains), S.WATER, 0.45, alpha=0.85, z=3)
wp = w_res.merge(bld[["building_id", "lon", "lat"]], on="building_id", how="inner")
pbins = [(27.5, 40, "#08306B"), (40, 50, "#2171B5"), (50, 60, "#6BAED6"), (60, 70.01, "#C6DBEF")]
for lo, hi, c in pbins:
    sel = wp[(wp.pressure_m >= lo) & (wp.pressure_m < hi)]
    ax.scatter(sel.lon, sel.lat, s=0.9, c=c, linewidths=0, zorder=2)
ax.scatter(w_station.source_coordinate_lon, w_station.source_coordinate_lat, s=55, marker="o",
           facecolor=S.WATER, edgecolor=S.INK, linewidths=0.8, zorder=8)
zb = w_zone[(w_zone.lon - w_station.source_coordinate_lon).abs() + (w_zone.lat - w_station.source_coordinate_lat).abs() > 1e-6]
ax.scatter(zb.lon, zb.lat, s=26, marker="h", facecolor="white", edgecolor=S.WATER, linewidths=0.9, zorder=8)
h = [M.line_handle(S.WATER, 1.2, label="Water main"),
     M.marker_handle("o", S.WATER, S.INK, 5.5, "Waterworks"),
     M.marker_handle("h", "white", S.WATER, 5.5, "Booster station", mew=0.9)]
h += [M.marker_handle("o", c, c, 4.0, (f"Pressure {lo:g}–{min(hi, 70):g} m" if lo < 30 else f"{lo:g}–{min(hi, 70):g} m"))
      for lo, hi, c in pbins]
ax.legend(handles=h, **LEG)
S.panel_title(ax, "(b) Drinking water: peak-hour pressure")

# (c) Wastewater ------------------------------------------------------------
ax = axes[2]
grav = s_links[s_links.link_type.eq("gravity_main")]
force = s_links[s_links.link_type.eq("force_main")]
M.draw(ax, M.segments(grav), S.SEWER, 0.4, alpha=0.85, z=3)
M.draw(ax, M.segments(force), S.SHARED, 0.9, ls=(0, (2.2, 1.2)), z=4, capstyle="butt")
ax.scatter(s_st.lon, s_st.lat, s=26, marker="^", facecolor=S.SHARED, edgecolor="white", linewidths=0.6, zorder=8)
out = s_nodes[s_nodes.node_type.eq("outfall")].iloc[0]
ax.scatter(out.lon, out.lat, s=55, marker="v", facecolor=S.SEWER, edgecolor="white", linewidths=0.7, zorder=9)
h = [M.line_handle(S.SEWER, 1.0, label="Gravity sewer"), M.line_handle(S.SHARED, 1.2, (0, (2.2, 1.2)), "Pressure pipe"),
     M.marker_handle("^", S.SHARED, "white", 5.5, "Lift station"),
     M.marker_handle("v", S.SEWER, "white", 6.0, "Treatment plant")]
ax.legend(handles=h, **LEG)
manholes = int(s_nodes.node_type.isin(["collection_manhole", "subcatchment_terminal_manhole",
                                        "force_transition_manhole", "lift_station", "outfall"]).sum())
S.panel_title(ax, "(c) Wastewater: dry weather")

# (d) District heating ------------------------------------------------------
ax = axes[3]
tc = [(120, "#8C4A00", "Supply 120 °C"), (110, "#D98200", "110 °C"),
      (85, "#F4C46A", "85 °C")]
for temp, c, _ in sorted(tc, key=lambda x: x[0]):
    sel = h_corr[np.isclose(h_corr.required_supply_temperature_c, temp)]
    M.draw(ax, M.segments(sel), c, 1.0 if temp < 120 else 1.2, z=3 + temp / 100)
for r in h_src.itertuples():
    ax.scatter(r.lon, r.lat, s=110, marker="*", facecolor=S.HEAT, edgecolor=S.INK, linewidths=0.7, zorder=9)
    ax.annotate("GKS" if r.map_id == "H1" else "Boiler plant", (r.lon, r.lat),
                xytext=(5, -3) if r.map_id == "H1" else (-5, 5), textcoords="offset points",
                ha="left" if r.map_id == "H1" else "right", va="top" if r.map_id == "H1" else "bottom",
                fontsize=6.3, fontweight="bold", color=S.INK, zorder=10,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
h = [M.line_handle(c, 1.5, label=lab) for _, c, lab in tc] + [M.marker_handle("*", S.HEAT, S.INK, 8.5, "Heat plant")]
ax.legend(handles=h, **LEG)
S.panel_title(ax, "(d) District heating: supply temperature")

for ax in axes:
    M.scalebar(ax, 2.0, loc=(0.79, 0.035))
    M.north(ax, loc=(0.955, 0.90))
leg_axes[3].text(0.995, 0.0, "Base data © OpenStreetMap contributors, ODbL", transform=leg_axes[3].transAxes,
             ha="right", va="bottom", fontsize=5.0, color=S.MUTED)

S.save(fig, args.out, "Fig05_Schweinfurt_Networks")
print("wrote", args.out / "Fig05_Schweinfurt_Networks.pdf")
