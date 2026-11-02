#!/usr/bin/env python3
"""Fig. 3 -- Design-demand relations used to size the Schweinfurt networks.

Each panel plots one sizing relation from the manuscript and marks the value
applied in the Schweinfurt case:
  (a) Eq. (1)  residential service/branch peak with Kerber coincidence,
  (b) Eq. (21) DVGW W 410 hourly peak factor,
  (c) Eq. (22) Harmon dry-weather peak factor,
  (d) Eq. (9)  district-heating design peak from equivalent full-load hours.
The script needs no model outputs; all numbers follow from the equations.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import u4style as S

S.apply()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "figures")
OUT = parser.parse_args().out


def kerber_per_household(n):
    return 14.5 * (0.07 + 0.93 * n ** (-0.75))


def w410(E):
    return 18.1 * E ** (-0.1682)


def harmon(P):
    return np.maximum(2.0, 1.0 + 14.0 / (4.0 + np.sqrt(P)))


def thousands(x, _pos):
    if x >= 1e6:
        return f"{x/1e6:g}M"
    if x >= 1e3:
        return f"{x/1e3:g}k"
    return f"{x:g}"


fig, axes = plt.subplots(1, 4, figsize=(S.TEXT_W, 1.95))
fig.subplots_adjust(left=0.055, right=0.995, bottom=0.25, top=0.86, wspace=0.42)

# (a) Electricity: design peak per household versus downstream households.
ax = axes[0]
n = np.logspace(0, 3, 400)
ax.plot(n, kerber_per_household(n), color=S.POWER)
ax.axhline(14.5 * 0.07, color=S.MUTED, lw=0.7, ls=(0, (3, 2)))
ax.text(1.2, 14.5 * 0.07 * 0.93, "limit 1.0 kW", fontsize=6.6, color=S.MUTED, va="top")
for x, dx, dy in [(1, 5, -2), (30, 7, 9), (300, -4, 16)]:
    y = kerber_per_household(x)
    ax.plot(x, y, "o", ms=4.2, mfc=S.POWER, mec="white", mew=0.6, zorder=4)
    ax.annotate(f"{y:.1f} kW", (x, y), xytext=(dx, dy),
                textcoords="offset points", fontsize=6.6,
                ha="left" if dx > 0 else "center", va="bottom" if dy > 0 else "top", color=S.INK,
                arrowprops=None if x == 1 else dict(arrowstyle="-", lw=0.5, color=S.MUTED,
                                                     shrinkA=0, shrinkB=2.5))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 1000)
ax.set_ylim(0.7, 25)
ax.set_yticks([1, 2, 5, 10, 20])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Households on the cable")
ax.set_ylabel("Peak per household (kW)")
S.panel_title(ax, "(a) Electricity")

# (b) Drinking water: DVGW W 410 hourly peak factor.
ax = axes[1]
E = np.logspace(3, 6, 400)
ax.plot(E, w410(E), color=S.WATER)
E0 = 1.0e5
ax.plot(E0, w410(E0), "o", ms=4.2, mfc=S.WATER, mec="white", mew=0.6, zorder=4)
ax.annotate(f"100,000 people:\nabout {w410(E0):.1f}", (E0, w410(E0)),
            xytext=(-6, -6), textcoords="offset points", ha="right", va="top", fontsize=6.6)
ax.set_xscale("log")
ax.set_xlim(1e3, 1e6)
ax.set_ylim(1.5, 6.0)
ax.xaxis.set_major_formatter(FuncFormatter(thousands))
ax.set_xlabel("People supplied")
ax.set_ylabel("Peak-hour factor")
S.panel_title(ax, "(b) Drinking water")

# (c) Wastewater: Harmon dry-weather peak factor with lower bound 2.
ax = axes[2]
P = np.logspace(-1, 3, 400)  # thousand inhabitants
ax.plot(P, harmon(P), color=S.SEWER)
ax.axhline(2.0, color=S.MUTED, lw=0.7, ls=(0, (3, 2)))
ax.text(0.12, 2.06, "lower limit 2", fontsize=6.6, color=S.MUTED, va="bottom")
for p0, dx, dy, ha in [(1.0, 4, 3, "left"), (10.0, 4, 3, "left")]:
    m0 = float(harmon(p0))
    ax.plot(p0, m0, "o", ms=4.2, mfc=S.SEWER, mec="white", mew=0.6, zorder=4)
    ax.annotate(f"{p0*1000:,.0f} people: {m0:.1f}", (p0, m0), xytext=(dx, dy), textcoords="offset points",
                fontsize=6.6, ha=ha, va="bottom")
ax.set_xscale("log")
ax.set_xlim(0.1, 1000)
ax.set_ylim(1.8, 4.6)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: "1M" if x >= 1000 else (f"{x:g}k" if x >= 1 else f"{x*1000:g}")))
ax.set_xlabel("People in drainage area")
ax.set_ylabel("Peak factor")
S.panel_title(ax, "(c) Wastewater")

# (d) District heating: portfolio design peak from full-load hours.
ax = axes[3]
h = np.linspace(1000, 2500, 300)
H = 87.5e3  # MWh per year, published delivered heat
ax.plot(h, H / h, color=S.HEAT)
ax.plot(1600, H / 1600, "o", ms=4.2, mfc=S.HEAT, mec="white", mew=0.6, zorder=4)
ax.annotate(f"1,600 h: {H/1600:.0f} MW", (1600, H / 1600), xytext=(5, 4),
            textcoords="offset points", ha="left", va="bottom", fontsize=6.6)
ax.set_xlim(1000, 2500)
ax.set_ylim(30, 90)
ax.set_xticks([1000, 1500, 2000, 2500])
ax.set_xlabel("Full-load hours (h)")
ax.set_ylabel("Peak heat demand (MW)")
S.panel_title(ax, "(d) District heating")

S.save(fig, OUT, "Fig03_Design_Relations")
print("wrote", OUT / "Fig03_Design_Relations.pdf")
