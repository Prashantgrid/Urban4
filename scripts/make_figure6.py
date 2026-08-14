#!/usr/bin/env python3
"""Legacy map generator for the archived reproduced coupled baseline.

This script preserves the earlier service-resolved 27-interface numerical
baseline, including its historical synthetic heat-boundary representation,
for reproducibility only. It is not the current municipality-scale full-city
figure. For the v1.9.0 evidence-conditioned engineering candidate (H1 main GKS,
H2 SHW Nord, S0/S1-S13, W1, E1), run
``scripts/make_municipal_scale_figure.py`` after
``scripts/build_municipal_scale_design.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from make_figures import (
    COLORS,
    _draw_base,
    _draw_networks,
    _north_and_scale,
)


PROJECT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT / "figures"
DELIVERABLE = PROJECT / "deliverables" / "fig06_schweinfurt_integration"
STEM = "fig06_schweinfurt_integration"
VERSIONED_STEM = "fig06_schweinfurt_integration_facility_markers_v3"

FACILITY_STYLE = {
    "electricity": {"color": "#C7362F", "marker": "s", "size": 155, "label": "External-grid boundary"},
    "drinking_water": {"color": "#00A6D6", "marker": "o", "size": 140, "label": "Water source / pump"},
    "wastewater": {"color": "#7B4AB5", "marker": "^", "size": 145, "label": "Wastewater facility"},
    "district_heating": {"color": "#F2A900", "marker": "*", "size": 245, "label": "Heat-source boundary"},
}

# Point offsets keep source IDs legible without changing network geometry.
LABEL_OFFSETS = {
    "E1": (8, 8), "W1": (-22, 8), "S1": (-20, -12), "S2": (8, 8), "S3": (-20, -9),
    "H1": (-13, 9), "H2": (8, 8), "H3": (-18, 6), "H4": (8, 6),
    "H5": (-18, 0), "H6": (8, 5), "H7": (-17, -10), "H8": (-16, 7),
    "H9": (-19, -10), "H10": (9, -10), "H11": (8, 7), "H12": (-18, -2),
    "H13": (8, -9), "H14": (-18, -9),
}


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
        "font.size": 11.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _load_full_city_frames() -> dict:
    """Load the accepted four-sector baseline used by the coupled closure."""
    service = PROJECT / "outputs" / "integrated_service_resolved"
    return {
        "electricity_nodes": pd.read_csv(service / "electricity_nodes.csv"),
        "electricity_edges": pd.read_csv(service / "electricity_links.csv"),
        "drinking_water_nodes": pd.read_csv(service / "drinking_water_nodes.csv"),
        "drinking_water_edges": pd.read_csv(service / "drinking_water_links.csv"),
        "wastewater_nodes": pd.read_csv(service / "wastewater_nodes.csv"),
        "wastewater_edges": pd.read_csv(service / "wastewater_links.csv"),
        "district_heating_nodes": pd.read_csv(service / "district_heating_nodes.csv"),
        "district_heating_edges": pd.read_csv(service / "district_heating_corridors.csv"),
        "buildings": pd.read_csv(service / "common_building_service_ledger.csv"),
    }


def _network_extent(frames: dict) -> list[float]:
    """Tightly frame all service-resolved nodes with modest geographic padding."""
    node_frames = [
        frames["electricity_nodes"],
        frames["drinking_water_nodes"],
        frames["wastewater_nodes"],
        frames["district_heating_nodes"],
    ]
    west = min(frame["lon"].min() for frame in node_frames)
    east = max(frame["lon"].max() for frame in node_frames)
    south = min(frame["lat"].min() for frame in node_frames)
    north = max(frame["lat"].max() for frame in node_frames)
    dx = east - west
    dy = north - south
    return [west - 0.018 * dx, south - 0.025 * dy, east + 0.018 * dx, north + 0.025 * dy]


def _save_all(fig: plt.Figure) -> None:
    for folder in (FIGURES, DELIVERABLE):
        folder.mkdir(parents=True, exist_ok=True)
        for stem in (STEM, VERSIONED_STEM):
            fig.savefig(folder / f"{stem}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.025)
            fig.savefig(folder / f"{stem}.svg", format="svg", bbox_inches="tight", pad_inches=0.025)
            fig.savefig(folder / f"{stem}.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.025)


def _draw_high_visibility_facilities(ax: plt.Axes, frames: dict) -> None:
    """Draw and identify the source/facility objects reported in the rating tables."""
    inventory = pd.read_csv(
        PROJECT / "outputs" / "verified_inventory_v1.5.3" / "facility_map_ids.csv"
    )
    for row in inventory.itertuples():
        style = FACILITY_STYLE[row.sector]
        ax.scatter(
            [row.lon], [row.lat], s=style["size"] * 1.34,
            marker=style["marker"], c="white", edgecolors="white",
            linewidths=1.8, zorder=10,
        )
        ax.scatter(
            [row.lon], [row.lat], s=style["size"], marker=style["marker"],
            c=style["color"], edgecolors="#26313B", linewidths=0.85, zorder=11,
        )
        dx, dy = LABEL_OFFSETS.get(row.map_id, (7, 7))
        ax.annotate(
            row.map_id, xy=(row.lon, row.lat), xytext=(dx, dy),
            textcoords="offset points", ha="center", va="center",
            fontsize=7.4, fontweight="bold", color="#17212B", zorder=13,
            bbox=dict(
                boxstyle="round,pad=0.18", facecolor="white",
                edgecolor=style["color"], linewidth=0.75, alpha=0.94,
            ),
            arrowprops=dict(
                arrowstyle="-", color=style["color"], linewidth=0.55,
                shrinkA=2, shrinkB=4,
            ),
        )


def make_figure6() -> None:
    frames = _load_full_city_frames()
    extent = _network_extent(frames)

    # The generated domain is almost square after geographic aspect correction.
    # A compact square canvas prevents the large lateral whitespace created by
    # forcing this map into a landscape multi-panel layout.
    fig, ax = plt.subplots(figsize=(8.25, 8.05))
    _draw_base(ax, frames["buildings"], extent, building_alpha=0.18)
    _draw_networks(ax, frames)
    _draw_high_visibility_facilities(ax, frames)
    _north_and_scale(ax, 5.0)

    network_handles = [
        Line2D([0], [0], color=COLORS["electricity"], lw=2.4),
        Line2D([0], [0], color="#7E2948", lw=2.1, ls="--"),
        Line2D([0], [0], color=COLORS["drinking_water"], lw=2.4),
        Line2D([0], [0], color=COLORS["wastewater"], lw=2.1, ls=(0, (2.3, 1.4))),
        Line2D([0], [0], color=COLORS["district_heating"], lw=2.4),
    ]
    facility_handles = [
        Line2D(
            [0], [0], marker=FACILITY_STYLE[key]["marker"], color="none",
            markerfacecolor=FACILITY_STYLE[key]["color"], markeredgecolor="#26313B",
            markeredgewidth=0.7, markersize=10.2 if key != "heat" else 12.5,
        )
        for key in ("electricity", "drinking_water", "wastewater", "district_heating")
    ]
    labels = [
        "LV/electric",
        "MV backbone",
        "Drinking water",
        "Wastewater",
        "District heating",
        "External-grid boundary (E1)",
        "Water source / pump (W1)",
        "Wastewater facilities (S1--S3)",
        "Heat-source boundaries (H1--H14)",
    ]
    fig.legend(
        network_handles + facility_handles,
        labels,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        columnspacing=1.15,
        handlelength=1.75,
        handletextpad=0.42,
        fontsize=10.0,
        frameon=False,
    )
    fig.subplots_adjust(left=0.012, right=0.995, top=0.994, bottom=0.13)
    _save_all(fig)
    plt.close(fig)


if __name__ == "__main__":
    make_figure6()
