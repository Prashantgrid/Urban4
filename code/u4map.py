"""Vector base-map helpers shared by the Urban4 map figures.

Base layers come from the retained OpenStreetMap snapshot in the Urban4
repository (data/raw/*.json.xz): major roads and the river Main.  Buildings are
drawn from the common building ledger.  All layers are vector graphics.
"""
from __future__ import annotations

import json
import lzma
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import u4style as S

LAT0 = 50.05
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(LAT0))
KM_PER_DEG_LAT = 110.57


@lru_cache(maxsize=4)
def osm_context(repo: str):
    data = json.load(lzma.open(Path(repo) / "data" / "raw" / "context.json.xz"))
    roads, rivers = [], []
    for el in data["elements"]:
        tags = el.get("tags", {})
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        seg = [(p["lon"], p["lat"]) for p in geom]
        if tags.get("waterway") in {"river", "canal"}:
            rivers.append(seg)
        elif tags.get("highway") in {"motorway", "trunk", "primary", "secondary", "tertiary"}:
            roads.append(seg)
    return roads, rivers


@lru_cache(maxsize=4)
def local_roads(repo: str):
    data = json.load(lzma.open(Path(repo) / "data" / "raw" / "local_roads.json.xz"))
    out = []
    for el in data["elements"]:
        geom = el.get("geometry")
        if geom and len(geom) >= 2:
            out.append([(p["lon"], p["lat"]) for p in geom])
    return out


def segments(df: pd.DataFrame, col: str = "geometry_json") -> list:
    seg = []
    for g in df[col]:
        try:
            pts = json.loads(g)
        except (TypeError, ValueError):
            continue
        if pts and len(pts) >= 2:
            seg.append([(float(p[0]), float(p[1])) for p in pts])
    return seg


def draw(ax, segs, color, lw, ls="-", alpha=1.0, z=2, capstyle="round", rasterized=False):
    if segs:
        lc = LineCollection(segs, colors=color, linewidths=lw, linestyles=ls, alpha=alpha, zorder=z,
                            capstyle=capstyle, joinstyle="round", rasterized=rasterized)
        ax.add_collection(lc)


def basemap(ax, repo: str, buildings: pd.DataFrame | None = None, local: bool = False,
            building_size: float = 0.25, road_lw: float = 0.45):
    roads, rivers = osm_context(repo)
    draw(ax, rivers, S.RIVER, 2.6, z=0.1)
    if local:
        draw(ax, local_roads(repo), "#E7E9EC", 0.35, z=0.2, rasterized=True)
    draw(ax, roads, "#D5D9DE", road_lw, z=0.3)
    if buildings is not None:
        ax.scatter(buildings.lon, buildings.lat, s=building_size, c="#BCC2C9", linewidths=0,
                   alpha=0.55, zorder=0.5, rasterized=True)


def frame(ax, bbox):
    x0, y0, x1, y1 = bbox
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1.0 / math.cos(math.radians(LAT0)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_edgecolor("#9AA1A9")


def scalebar(ax, km: float, loc=(0.04, 0.05), label=None, fs=6.6):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    lon_len = km / KM_PER_DEG_LON
    xa = x0 + loc[0] * (x1 - x0); ya = y0 + loc[1] * (y1 - y0)
    ax.plot([xa, xa + lon_len], [ya, ya], color=S.INK, lw=1.4, solid_capstyle="butt", zorder=30)
    for xx in (xa, xa + lon_len):
        ax.plot([xx, xx], [ya - 0.004 * (y1 - y0), ya + 0.012 * (y1 - y0)], color=S.INK, lw=0.7, zorder=30)
    ax.text(xa + lon_len / 2, ya + 0.018 * (y1 - y0), label or f"{km:g} km", ha="center", va="bottom",
            fontsize=fs, color=S.INK, zorder=30)


def north(ax, loc=(0.95, 0.90), fs=6.8):
    ax.annotate("", xy=(loc[0], loc[1]), xytext=(loc[0], loc[1] - 0.075), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45", lw=0.8, color=S.INK), zorder=30)
    ax.text(loc[0], loc[1] + 0.008, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fs, fontweight="bold", color=S.INK, zorder=30)


def attribution(ax, fs=5.0):
    ax.text(0.995, 0.005, "Base data © OpenStreetMap contributors, ODbL", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=fs, color="#6B7480", zorder=31)


def infobox(ax, text, loc=(0.02, 0.02), fs=6.6, ha="left", va="bottom"):
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, ha=ha, va=va, fontsize=fs, color=S.INK,
            zorder=32, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#AEB4BB", lw=0.5, alpha=0.93))


def line_handle(color, lw=1.6, ls="-", label=""):
    return Line2D([0], [0], color=color, lw=lw, ls=ls, label=label, solid_capstyle="butt")


def marker_handle(marker, face, edge="white", size=6.0, label="", mew=0.6):
    return Line2D([0], [0], marker=marker, color="none", markerfacecolor=face, markeredgecolor=edge,
                  markersize=size, markeredgewidth=mew, label=label)
