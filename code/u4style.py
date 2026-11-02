"""Shared figure style for the Urban4 manuscript (IEEE two-column layout).

All figures are drawn at their final printed size so that 8-pt text in the
figure matches the journal body text after inclusion (no rescaling).
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

# IEEE Transactions: column width 3.5 in, text width 7.16 in.
COL_W = 3.5
TEXT_W = 7.16

# Sector colours (identical to the TikZ colours in the manuscript preamble).
POWER = "#D84A3A"   # electricity (LV)
MV = "#7E2948"      # medium-voltage backbone
WATER = "#2474D2"   # drinking water
SEWER = "#354052"   # wastewater (always paired with a dashed line style)
HEAT = "#E69F00"    # district heating
SHARED = "#7A3E9D"  # cross-sector / shared asset
INK = "#1B232C"
MUTED = "#6B7480"
GRID = "#E3E6EA"
BUILDING = "#C9CED4"
RIVER = "#CFE1F2"

SERIF = ["Liberation Serif", "Tinos", "Nimbus Roman", "Times New Roman", "DejaVu Serif"]


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF,
        "font.size": 8.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "legend.frameon": False,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Serif",
        "mathtext.it": "Liberation Serif:italic",
        "mathtext.bf": "Liberation Serif:bold",
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 150,
    })


def panel_title(ax, text: str, **kw) -> None:
    """Left-aligned bold panel label such as '(a) Electricity'."""
    ax.set_title(text, loc="left", fontweight="bold", pad=3, color=INK, **kw)


def save(fig, out_dir, stem: str, png: bool = True) -> None:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02, dpi=400)  # dpi only affects rasterized base-map layers
    if png:
        fig.savefig(out / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
