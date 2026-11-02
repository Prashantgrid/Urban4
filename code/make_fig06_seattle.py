#!/usr/bin/env python3
"""Fig. 6 -- Seattle route comparison, re-composed for single-column layout.

The route raster is the classified overlay produced by the external comparison
(installed only / Urban4 only / shared); its pixels are used unchanged.  This
script re-crops the two map panels at full resolution, replaces the raster
scale bar by a vector scale bar of identical length (541 px = 0.5 km), and adds
vector titles and a vector legend.
Usage: python make_fig06_seattle.py [--png data/seattle_route_overlay_600dpi.png]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

import u4style as S
import u4map as M

S.apply()
ap = argparse.ArgumentParser()
ap.add_argument("--png", type=Path, default=Path(__file__).resolve().parent / "data" / "seattle_route_overlay_600dpi.png")
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
args = ap.parse_args()

img = np.asarray(Image.open(args.png).convert("RGB"))
crops = {"(a) Drinking water": img[485:3130, 115:2300], "(b) Wastewater": img[485:3130, 2625:4808]}
# The overlay was up-sampled to 600 dpi from a 1488-px source; halving it keeps all
# source detail (>600 ppi at print size) and reduces file size.
crops = {k: np.asarray(Image.fromarray(v).resize((v.shape[1] // 2, v.shape[0] // 2), Image.LANCZOS))
         for k, v in crops.items()}
SCALE = 0.5
PX_PER_KM = 541 / 0.5 * SCALE
INSTALLED, URBAN4, SHARED_ROUTE = "#2F78D1", "#E89000", "#2B8A57"   # colours used in the raster

h_px, w_px = next(iter(crops.values())).shape[:2]
panel_w = (S.COL_W - 0.06) / 2
panel_h = panel_w * h_px / w_px
fig_h = panel_h + 0.40
fig = plt.figure(figsize=(S.COL_W, fig_h))
for i, (title, arr) in enumerate(crops.items()):
    ax = fig.add_axes([(0.02 + i * (panel_w + 0.02)) / S.COL_W, 0.02 / fig_h, panel_w / S.COL_W, panel_h / fig_h])
    ax.imshow(arr, interpolation="none", origin="upper")
    ax.set_xlim(0, arr.shape[1]); ax.set_ylim(arr.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_edgecolor("#9AA1A9")
    # vector scale bar on a white label box that covers the raster scale bar
    ax.add_patch(FancyBboxPatch((70 * SCALE, 2330 * SCALE), 640 * SCALE, 240 * SCALE,
                                boxstyle=f"round,pad={8 * SCALE},rounding_size={30 * SCALE}",
                                fc="white", ec="#AEB4BB", lw=0.4, zorder=5))
    x0 = 113 * SCALE
    ax.plot([x0, x0 + PX_PER_KM * 0.5], [2505 * SCALE] * 2, color=S.INK, lw=1.3, solid_capstyle="butt", zorder=6)
    ax.text(x0 + PX_PER_KM * 0.25, 2470 * SCALE, "0.5 km", ha="center", va="bottom", fontsize=6.4, color=S.INK, zorder=6)
    ax.set_title(title, loc="left", fontsize=7.6, fontweight="bold", pad=2.5, color=S.INK)
handles = [M.line_handle(INSTALLED, 2.2, label="Installed only"), M.line_handle(URBAN4, 2.2, label="Urban4 only"),
           M.line_handle(SHARED_ROUTE, 2.2, label="Both")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.0, frameon=False,
           handlelength=2.2, columnspacing=1.6)
S.save(fig, args.out, "Fig06_Seattle_Routes")
print("wrote", args.out / "Fig06_Seattle_Routes.pdf")
