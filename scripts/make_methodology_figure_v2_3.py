#!/usr/bin/env python3
"""Create the final Urban4 three-phase methodology figure.

The figure intentionally communicates only the publication-level workflow.
Detailed bounded repair ordering remains implementation documentation rather
than a manuscript contribution.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

NAVY = "#173A63"
RED = "#CB4935"
BLUE = "#2B78B9"
SLATE = "#56657A"
GOLD = "#D99400"
PURPLE = "#8544B0"
GREEN = "#2E7D32"
INK = "#202B38"
LIGHT = "#F7F9FB"
BORDER = "#B5C0CC"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 10.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def rr(ax, x, y, w, h, ec=BORDER, fc="white", lw=1.0, radius=0.006):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.004,rounding_size={radius}",
                       ec=ec, fc=fc, lw=lw)
    ax.add_patch(p)
    return p


def header(ax, y, text, h=0.042):
    ax.add_patch(FancyBboxPatch((0.012, y), 0.976, h,
                                boxstyle="round,pad=0.003,rounding_size=0.004",
                                ec=NAVY, fc=NAVY, lw=0.8))
    ax.text(0.022, y+h/2, text, color="white", fontsize=13.4,
            fontweight="bold", va="center", ha="left")


def arrow(ax, a, b, color=INK, lw=1.6, style="-|>", ms=17, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(16.0, 8.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------------- Phase I ----------------
    header(ax, 0.938, "PHASE I - COMMON EVIDENCE LEDGER")
    rr(ax, 0.012, 0.772, 0.976, 0.158, ec=BORDER, fc=LIGHT, lw=0.8)
    rr(ax, 0.027, 0.792, 0.405, 0.116, ec="#8EA2B8", fc="white", lw=0.9)
    ax.text(0.042, 0.882, "A. Audited inputs", color=NAVY, fontsize=13.0,
            fontweight="bold", va="top")
    ax.text(0.042, 0.854,
            "Boundary, buildings, roads, terrain, facilities, public totals,\n"
            "component catalogues, dates, units, and evidence roles",
            color=INK, fontsize=10.1, va="top", linespacing=1.12)

    rr(ax, 0.568, 0.792, 0.405, 0.116, ec="#8EA2B8", fc="white", lw=0.9)
    ax.text(0.583, 0.882, "B. Frozen common model", color=NAVY, fontsize=13.0,
            fontweight="bold", va="top")
    ax.text(0.583, 0.854,
            "Immutable IDs, reconciled building demands, corridor graph,\n"
            "planning zones, facilities, and provenance records",
            color=INK, fontsize=10.1, va="top", linespacing=1.12)
    arrow(ax, (0.446,0.851), (0.548,0.851), color=NAVY, lw=1.8, ms=20)
    arrow(ax, (0.500,0.772), (0.500,0.742), color=SLATE, lw=1.6, ms=17)

    # ---------------- Phase II ----------------
    header(ax, 0.704, "PHASE II - SECTOR-NATIVE CANDIDATE GENERATION")
    rr(ax, 0.012, 0.445, 0.976, 0.251, ec=BORDER, fc=LIGHT, lw=0.8)

    boxes = [
        (0.025, RED, "A. ELECTRICITY",
         ["Supply points and service zones", "Radial MV/LV feeders", "Discrete cables and transformers"]),
        (0.268, BLUE, "B. DRINKING WATER",
         ["Pressure zones and access nodes", "Backbone mains and services", "Selected second-feed paths"]),
        (0.511, SLATE, "C. WASTEWATER",
         ["Subcatchments and manholes", "Gravity collectors to the outfall", "Lift / force mains where required"]),
        (0.754, GOLD, "D. DISTRICT HEATING",
         ["Heat-density territories", "Partial customers; source boundaries", "Shared-route tree; paired pipes", "No suitable context territory: N/A"]),
    ]
    bw = 0.221
    for x, c, ttl, lines in boxes:
        rr(ax, x, 0.512, bw, 0.158, ec=c, fc="white", lw=1.0)
        ax.add_patch(Rectangle((x,0.628), bw, 0.042, ec=c, fc=c, lw=0))
        ax.text(x+0.010, 0.649, ttl, color="white", fontsize=11.1,
                fontweight="bold", va="center")
        fs = 9.45 if len(lines)==4 else 9.8
        y0 = 0.613
        dy = 0.0285 if len(lines)==4 else 0.034
        for i, line in enumerate(lines):
            ax.text(x+0.010, y0-i*dy, line, color=INK, fontsize=fs, va="top")

    rr(ax, 0.220, 0.462, 0.560, 0.032, ec=GREEN, fc="#F4FAF3", lw=0.9)
    ax.text(0.500, 0.478, "NATIVE ACCEPTANCE  →  BOUNDED REPAIR OR REJECT",
            color=GREEN, fontsize=10.2, fontweight="bold", ha="center", va="center")
    arrow(ax, (0.500,0.445), (0.500,0.414), color=GREEN, lw=1.7, ms=18)

    # ---------------- Phase III ----------------
    header(ax, 0.376, "PHASE III - SCHEMA-DEFINED INTERFACES, FIXED POINT, AND RELEASE")
    rr(ax, 0.012, 0.040, 0.976, 0.328, ec=BORDER, fc=LIGHT, lw=0.8)

    # A validate
    rr(ax, 0.025, 0.185, 0.165, 0.115, ec=PURPLE, fc="white", lw=1.0)
    ax.text(0.034,0.274,"A. Validate interfaces", color=PURPLE, fontsize=10.9, fontweight="bold", va="top")
    ax.text(0.034,0.246,"IDs, buses, equations, and units\nMass mappings; corridor records",
            color=INK, fontsize=8.85, va="top", linespacing=1.15)

    # B solve
    rr(ax, 0.217, 0.185, 0.184, 0.115, ec=BLUE, fc="white", lw=1.0)
    ax.text(0.226,0.274,"B. Solve states and duties", color=BLUE, fontsize=10.9, fontweight="bold", va="top")
    ax.text(0.226,0.246,"Solve applicable native states\nCompute and attach facility duties",
            color=INK, fontsize=8.85, va="top", linespacing=1.15)

    # C fixed point
    rr(ax, 0.427, 0.185, 0.171, 0.115, ec=PURPLE, fc="white", lw=1.0)
    ax.text(0.436,0.274,"C. Coupled fixed point", color=PURPLE, fontsize=10.9, fontweight="bold", va="top")
    ax.text(0.436,0.246,"Electrical solve; reverse mapping\nRe-solve affected native sectors",
            color=INK, fontsize=8.65, va="top", linespacing=1.15)

    arrow(ax,(0.190,0.242),(0.212,0.242), color=SLATE, ms=15)
    arrow(ax,(0.401,0.242),(0.422,0.242), color=SLATE, ms=15)

    # decision diamond
    cx, cy, dx, dy = 0.665, 0.242, 0.055, 0.070
    diamond = Polygon([(cx,cy+dy),(cx+dx,cy),(cx,cy-dy),(cx-dx,cy)],
                      closed=True, ec=GREEN, fc="#F7FBF5", lw=1.0)
    ax.add_patch(diamond)
    ax.text(cx,cy+0.012,"All criteria", ha="center", va="center", color=GREEN,
            fontsize=9.7, fontweight="bold")
    ax.text(cx,cy-0.018,"satisfied?", ha="center", va="center", color=GREEN,
            fontsize=9.7, fontweight="bold")
    arrow(ax,(0.598,0.242),(cx-dx-0.004,0.242), color=SLATE, ms=15)

    # repair/reject box
    rr(ax, 0.760, 0.176, 0.212, 0.132, ec="#CF2F27", fc="white", lw=1.0)
    ax.text(0.770,0.284,"Apply declared bounded repair", color="#B92620",
            fontsize=10.5, fontweight="bold", va="top")
    ax.text(0.770,0.258,
            "Interface / state: one bounded action; repeat Phase III\n"
            "Topology / catalogue: return to responsible Phase-II module\n"
            "No admissible action: reject; no accepted export",
            color=INK, fontsize=7.95, va="top", linespacing=1.22)
    # NO path
    arrow(ax,(cx+dx,cy),(0.755,cy), color="#C72B23", ms=16)
    ax.text(0.735,cy+0.017,"NO",color="#C72B23",fontsize=9.0,fontweight="bold",ha="center")

    # complete feedback arrow for repeat Phase III
    ax.plot([0.866,0.866,0.090,0.090], [0.308,0.338,0.338,0.307],
            color="#C72B23", lw=1.2, ls=(0,(4,3)), zorder=3)
    arrow(ax,(0.090,0.307),(0.090,0.301), color="#C72B23", lw=1.2, ms=13)
    ax.text(0.480,0.343,"interface/state repair: repeat Phase III", color="#B92620",
            fontsize=7.9, ha="center", va="bottom")

    # YES path
    rr(ax, 0.572, 0.088, 0.186, 0.047, ec=GREEN, fc="#F3FAF3", lw=1.0)
    ax.text(0.665,0.1115,"ACCEPT AND EXPORT", color=GREEN, fontsize=11.1,
            fontweight="bold", ha="center", va="center")
    arrow(ax,(cx,cy-dy),(cx,0.138), color=GREEN, ms=16)
    ax.text(cx+0.018,0.151,"YES",color=GREEN,fontsize=9.0,fontweight="bold")

    ax.text(0.500,0.058,
            "Topology/catalogue failure returns to Phase II; no admissible repair terminates without export. Thresholds are never relaxed.",
            color="#C96A00", fontsize=8.35, ha="center", va="center", style="italic")

    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf","svg","png"):
        fig.savefig(FIG/f"Fig01_Urban4_Methodology_Journal.{ext}", dpi=320,
                    bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

if __name__ == "__main__":
    main()
