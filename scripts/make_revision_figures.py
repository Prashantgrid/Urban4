#!/usr/bin/env python3
"""Create figures added in the route-aware Urban4 revision."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT = Path(__file__).resolve().parents[1]
FIG = PROJECT / "figures"

COLORS = {
    "heat": "#E69F00",
    "heat_dark": "#A96900",
    "selected": "#E69F00",
    "excluded": "#AAB1B7",
    "road": "#C9CED2",
    "shared": "#7A3E9D",
    "water": "#2474D2",
    "sewer": "#354052",
    "ink": "#18212B",
    "muted": "#66717E",
    "green": "#2F7D59",
    "red": "#B64B42",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "font.size": 9.2,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": COLORS["ink"],
})


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 260})]:
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.035, **kwargs)
    plt.close(fig)


def _draw_graph(ax, nodes, edges, *, active_edges=(), selected=(), excluded=(), plant=(0, 0), prizes=None):
    active = {tuple(sorted(edge)) for edge in active_edges}
    for u, v in edges:
        key = tuple(sorted((u, v)))
        colour = COLORS["heat"] if key in active else COLORS["road"]
        width = 2.6 if key in active else 1.0
        z = 3 if key in active else 1
        ax.plot([nodes[u][0], nodes[v][0]], [nodes[u][1], nodes[v][1]], color=colour, lw=width, zorder=z)
    for name in excluded:
        x, y = nodes[name]
        ax.scatter(x, y, s=48, c=COLORS["excluded"], marker="o", edgecolors="white", linewidths=0.7, zorder=5)
        ax.text(x, y - 0.22, name, ha="center", va="top", fontsize=7.2, color=COLORS["muted"])
    for name in selected:
        x, y = nodes[name]
        size = 58 + 0.22 * float((prizes or {}).get(name, 0))
        ax.scatter(x, y, s=size, c=COLORS["selected"], marker="o", edgecolors="white", linewidths=0.8, zorder=6)
        ax.text(x, y - 0.22, name, ha="center", va="top", fontsize=7.2, fontweight="bold", color=COLORS["heat_dark"])
    px, py = nodes[plant]
    ax.scatter(px, py, s=150, c=COLORS["shared"], marker="*", edgecolors="white", linewidths=0.9, zorder=7)
    ax.text(px, py - 0.28, "source\nboundary", ha="center", va="top", fontsize=6.8, fontweight="bold", color=COLORS["shared"], linespacing=0.95)
    ax.set_xlim(-0.45, 4.55)
    ax.set_ylim(-0.75, 3.05)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#C3C8CC")


def figure_route_aware_method() -> None:
    nodes = {
        0: (0.25, 1.25), 1: (1.25, 1.25), 2: (2.25, 1.25), 3: (3.25, 1.25),
        4: (4.20, 1.25), 5: (2.25, 2.35), 6: (3.25, 2.35), 7: (2.25, 0.20),
        "B1": (4.20, 1.25), "B2": (3.25, 2.35), "B3": (2.25, 2.35), "B4": (2.25, 0.20),
    }
    edges = [(0,1),(1,2),(2,3),(3,4),(2,5),(5,6),(2,7)]
    prizes = {"B1": 180, "B2": 105, "B3": 90, "B4": 45}
    all_buildings = ["B1", "B2", "B3", "B4"]

    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.45))
    fig.suptitle(
        "Route-aware partial connection: a quota-constrained prize-collecting forest approximation",
        x=0.03, ha="left", fontsize=12.0, fontweight="bold", y=1.035,
    )
    subtitles = [
        "(a) Candidate corridor graph",
        "(b) First selected branch",
        "(c) Shared trunk lowers marginal route",
        "(d) Final quota-constrained forest",
    ]
    for ax, title in zip(axes, subtitles):
        ax.set_title(title, loc="left", fontweight="bold", pad=5)

    _draw_graph(axes[0], nodes, edges, selected=all_buildings, plant=0, prizes=prizes)
    for name, prize in prizes.items():
        x, y = nodes[name]
        axes[0].text(x, y + 0.26, f"$H_b={prize}$", ha="center", fontsize=7.0)
    axes[0].text(0.04, 0.04, "Road edges carry length $L_e$;\nmarker size represents annual heat prize.", transform=axes[0].transAxes, fontsize=7.0, va="bottom")

    active_b1 = [(0,1),(1,2),(2,3),(3,4)]
    _draw_graph(axes[1], nodes, edges, active_edges=active_b1, selected=["B1"], excluded=["B2","B3","B4"], plant=0, prizes=prizes)
    axes[1].text(0.04, 0.05, "$B_1$ is chosen by the largest\n$H_b/[\\Delta L_b(F)+L_b^{svc}]$.", transform=axes[1].transAxes, fontsize=7.2, va="bottom")

    active_b2 = active_b1 + [(2,5),(5,6)]
    _draw_graph(axes[2], nodes, edges, active_edges=active_b2, selected=["B1","B2"], excluded=["B3","B4"], plant=0, prizes=prizes)
    axes[2].annotate(
        "existing trunk\nis free in $\\Delta L$", xy=(1.65,1.25), xytext=(0.55,2.55),
        arrowprops={"arrowstyle":"->","lw":0.9,"color":COLORS["shared"]},
        fontsize=7.1, color=COLORS["shared"], ha="center",
    )
    axes[2].text(0.04, 0.05, "A nearby customer can be selected\nwith only the new branch as penalty.", transform=axes[2].transAxes, fontsize=7.2, va="bottom")

    _draw_graph(axes[3], nodes, edges, active_edges=active_b2, selected=["B1","B2"], excluded=["B3","B4"], plant=0, prizes=prizes)
    axes[3].text(
        0.035, 0.17, "STOP", transform=axes[3].transAxes, fontsize=6.9,
        fontweight="bold", color="white", ha="left", va="center",
        bbox={"boxstyle":"round,pad=0.22","fc":COLORS["green"],"ec":COLORS["green"]},
    )
    axes[3].text(
        0.035, 0.115, "Quota, capacity, or marginal-density rule reached",
        transform=axes[3].transAxes, ha="left", va="center", fontsize=6.5,
        color=COLORS["green"], fontweight="bold",
    )
    axes[3].text(
        0.035, 0.045, "Excluded buildings keep their ledger IDs; only the heat-service flag changes.",
        transform=axes[3].transAxes, ha="left", va="center", fontsize=6.25,
    )

    fig.text(
        0.5, -0.035,
        r"Urban4 uses a deterministic greedy approximation, not a monetary adoption model: "
        r"$b^*=\arg\max H_b^{\alpha}/[\Delta L_b(F)+L_b^{svc}]$ within heat-suitable territories.",
        ha="center", fontsize=8.0,
    )
    fig.subplots_adjust(wspace=0.085, left=0.02, right=0.995, bottom=0.11, top=0.88)
    save(fig, "fig04_route_aware_heat_selection")


def figure_water_sewer_overlap() -> None:
    source = PROJECT / "outputs" / "water_sewer_principle_candidate"
    geo = json.loads((source / "water_wastewater_candidate_segments.geojson").read_text())
    summary = pd.read_csv(source / "water_wastewater_overlap_before_after.csv").set_index("metric")["value"]
    classes = {"shared": [], "water_only": [], "sewer_only": []}
    for feature in geo["features"]:
        classes[feature["properties"]["class"]].append(feature["geometry"]["coordinates"])

    fig = plt.figure(figsize=(10.8, 4.35))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.55, 1.0], wspace=0.12,
        left=0.04, right=0.98, bottom=0.16, top=0.90,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax.add_collection(LineCollection(
        classes["shared"], colors="#C6C9CC", linewidths=0.52,
        alpha=0.78, zorder=1,
    ))
    ax.add_collection(LineCollection(
        classes["water_only"], colors=COLORS["water"], linewidths=1.18,
        alpha=0.96, zorder=3,
    ))
    ax.add_collection(LineCollection(
        classes["sewer_only"], colors=COLORS["sewer"], linewidths=1.18,
        alpha=0.96, zorder=4,
    ))
    all_coords = np.asarray(
        [point for lines in classes.values() for line in lines for point in line],
        dtype=float,
    )
    ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
    ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())
    ax.set_aspect(1.0 / math.cos(math.radians(all_coords[:, 1].mean())))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#C3C8CC")
    ax.set_title(
        "(a) Principle-aligned water and wastewater corridors",
        loc="left", fontweight="bold",
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#C6C9CC", lw=2, label="shared corridor"),
            Line2D([0], [0], color=COLORS["water"], lw=2, label="water only"),
            Line2D([0], [0], color=COLORS["sewer"], lw=2, label="wastewater only"),
        ],
        loc="lower left", fontsize=7.4, frameon=True, framealpha=0.92, ncol=3,
    )

    ax2 = fig.add_subplot(grid[0, 1])
    metrics = ["Water shared", "Wastewater shared", "Jaccard"]
    baseline = np.asarray([
        summary["baseline_water_shared_percent"],
        summary["baseline_sewer_shared_percent"],
        100.0 * summary["baseline_length_weighted_jaccard"],
    ])
    candidate = np.asarray([
        summary["water_shared_percent"],
        summary["sewer_shared_percent"],
        100.0 * summary["length_weighted_jaccard"],
    ])
    y = np.arange(len(metrics))
    height = 0.31
    bars0 = ax2.barh(
        y - height / 2, baseline, height=height,
        color="#C6C9CC", label="building-first shared-road baseline",
    )
    bars1 = ax2.barh(
        y + height / 2, candidate, height=height,
        color=[COLORS["water"], COLORS["sewer"], COLORS["shared"]],
        label="pressure-zone / manhole-first candidate",
    )
    ax2.set_yticks(y, metrics)
    ax2.invert_yaxis()
    ax2.set_xlabel("Shared-corridor measure (%)")
    ax2.set_xlim(0, 108)
    ax2.set_title("(b) Before/after topology audit", loc="left", fontweight="bold")
    ax2.grid(axis="x", alpha=0.28)
    ax2.spines[["top", "right"]].set_visible(False)
    for bars in (bars0, bars1):
        for bar, value in zip(bars, baseline if bars is bars0 else candidate):
            ax2.text(
                min(value + 1.4, 101.0), bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}", va="center", fontsize=7.4,
            )
    ax2.legend(loc="upper right", fontsize=6.9, frameon=False)
    fig.suptitle(
        "Sector-specific topology rules reduce the artefactual near-identity of water and wastewater mains",
        x=0.04, ha="left", fontsize=11.5, fontweight="bold",
    )
    fig.text(
        0.5, 0.045,
        f"Candidate scales: water {summary['water_unique_main_geometry_km']:.1f} km; wastewater {summary['sewer_unique_main_geometry_km']:.1f} km. "
        "Mains only; building laterals are excluded. The reduction emerges from sector-specific access nodes, anchors, and directionality; overlap is not penalized.",
        ha="center", fontsize=7.15, color=COLORS["muted"],
    )
    save(fig, "fig07_water_wastewater_overlap")


if __name__ == "__main__":
    figure_route_aware_method()
    figure_water_sewer_overlap()
    print("wrote revision figures to", FIG)
