#!/usr/bin/env python3
"""Create benchmark figures from the generated four-sector model."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

from urban4 import schweinfurt_base as base


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
RAW = PROJECT / "data" / "raw"
FIG = PROJECT / "figures"
CASE = json.loads((PROJECT / "cases" / "schweinfurt.json").read_text(encoding="utf-8"))

COLORS = {
    "electricity": "#D84A3A",
    "drinking_water": "#2474D2",
    "wastewater": "#354052",
    "district_heating": "#E69F00",
    "shared": "#7A3E9D",
    "building": "#B8B1A8",
    "road": "#D8D7D4",
    "ink": "#18212B",
    "muted": "#66717E",
    "background": "#FAFAF8",
}

LABELS = {
    "electricity": "Electricity",
    "drinking_water": "Drinking water",
    "wastewater": "Wastewater",
    "district_heating": "District heating",
}

MORPH_COLORS = {
    "dense": "#3B6FB6",
    "suburban": "#57A773",
    "peripheral": "#B98B2F",
    "industrial": "#A14C63",
}

BENCHMARK_CASES = CASE["benchmark_cases"]


mpl.rcParams.update(
    {
        # Nimbus Roman is the installed, metrically compatible Times family.
        # The SVG retains a Times New Roman-first fallback stack.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
        "font.size": 10.2,
        "axes.titlesize": 11.4,
        "axes.labelsize": 10.4,
        "axes.edgecolor": "#87919B",
        "axes.linewidth": 0.7,
        "axes.facecolor": COLORS["background"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "grid.color": "#D9DDE1",
        "grid.linewidth": 0.55,
        "xtick.color": "#37414B",
        "ytick.color": "#37414B",
        "text.color": COLORS["ink"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _load_frames() -> dict[str, pd.DataFrame]:
    service = OUT / "integrated_service_resolved"
    electricity_edges = pd.read_csv(service / "electricity_links.csv")
    water_edges = pd.read_csv(service / "drinking_water_links.csv")
    wastewater_edges = pd.read_csv(service / "wastewater_links.csv")
    heat_edges = pd.read_csv(service / "district_heating_corridors.csv")
    inventory = pd.DataFrame(
        [
            {
                "sector": "electricity", "nodes": len(pd.read_csv(service / "electricity_nodes.csv")),
                "links": len(electricity_edges), "route_length_km": electricity_edges.length_km.sum(),
                "size_median": electricity_edges.cable_size_mm2.median(),
                "size_p90": electricity_edges.cable_size_mm2.quantile(0.9),
            },
            {
                "sector": "drinking_water", "nodes": len(pd.read_csv(service / "drinking_water_nodes.csv")),
                "links": len(water_edges), "route_length_km": water_edges.length_km.sum(),
                "size_median": water_edges.diameter_mm.median(),
                "size_p90": water_edges.diameter_mm.quantile(0.9),
            },
            {
                "sector": "wastewater", "nodes": len(pd.read_csv(service / "wastewater_nodes.csv")),
                "links": len(wastewater_edges), "route_length_km": wastewater_edges.length_km.sum(),
                "size_median": wastewater_edges.diameter_mm.median(),
                "size_p90": wastewater_edges.diameter_mm.quantile(0.9),
            },
            {
                "sector": "district_heating", "nodes": len(pd.read_csv(service / "district_heating_nodes.csv")),
                "links": len(heat_edges), "route_length_km": heat_edges.length_km.sum(),
                "size_median": heat_edges.diameter_mm.median(),
                "size_p90": heat_edges.diameter_mm.quantile(0.9),
            },
        ]
    )
    return {
        "electricity_nodes": pd.read_csv(service / "electricity_nodes.csv"),
        "electricity_edges": electricity_edges,
        "drinking_water_nodes": pd.read_csv(service / "drinking_water_nodes.csv"),
        "drinking_water_edges": water_edges,
        "wastewater_nodes": pd.read_csv(service / "wastewater_nodes.csv"),
        "wastewater_edges": wastewater_edges,
        "district_heating_nodes": pd.read_csv(service / "district_heating_nodes.csv"),
        "district_heating_edges": heat_edges,
        "buildings": pd.read_csv(service / "common_building_service_ledger.csv"),
        "metrics": pd.read_csv(OUT / "morphology_metrics.csv"),
        "inventory": inventory,
        "anchors": pd.read_csv(OUT / "aggregate_anchor_checks.csv"),
        "interfaces": pd.read_csv(service / "coupling_interfaces.csv"),
        "corridors": pd.read_csv(service / "shared_corridor_ledger.csv"),
    }


def _load_benchmark_case(case_id: str) -> dict[str, pd.DataFrame]:
    root = OUT / "benchmark_cases" / case_id
    heat_path = root / "district_heating_corridors.csv"
    if not heat_path.exists():
        heat_path = root / "district_heating_links.csv"
    return {
        "electricity_nodes": pd.read_csv(root / "electricity_nodes.csv"),
        "electricity_edges": pd.read_csv(root / "electricity_links.csv"),
        "drinking_water_nodes": pd.read_csv(root / "drinking_water_nodes.csv"),
        "drinking_water_edges": pd.read_csv(root / "drinking_water_links.csv"),
        "wastewater_nodes": pd.read_csv(root / "wastewater_nodes.csv"),
        "wastewater_edges": pd.read_csv(root / "wastewater_links.csv"),
        "district_heating_nodes": pd.read_csv(root / "district_heating_nodes.csv"),
        "district_heating_edges": pd.read_csv(heat_path),
        "buildings": pd.read_csv(root / "common_building_ledger.csv"),
    }


def _road_segments() -> list[list[tuple[float, float]]]:
    data = base.read_raw_json("local_roads.json")
    segments = []
    for element in data["elements"]:
        geometry = element.get("geometry") or []
        if len(geometry) > 1:
            segments.append([(float(p["lon"]), float(p["lat"])) for p in geometry])
    return segments


ROADS = _road_segments()


def _edge_segments(frame: pd.DataFrame) -> list[list[tuple[float, float]]]:
    return [json.loads(item) for item in frame["geometry_json"]]


def _set_map_extent(ax: plt.Axes, bbox: list[float] | tuple[float, ...]) -> None:
    west, south, east, north = bbox
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1.0 / math.cos(math.radians((south + north) / 2.0)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B7BDC4")


def _draw_base(
    ax: plt.Axes,
    buildings: pd.DataFrame,
    bbox: list[float] | tuple[float, ...],
    building_alpha: float = 0.20,
) -> None:
    ax.add_collection(LineCollection(ROADS, colors=COLORS["road"], linewidths=0.32, alpha=0.82, zorder=0))
    west, south, east, north = bbox
    subset = buildings[
        buildings["lon"].between(west, east) & buildings["lat"].between(south, north)
    ]
    ax.scatter(
        subset["lon"], subset["lat"], s=1.0, c=COLORS["building"],
        alpha=building_alpha, edgecolors="none", rasterized=True, zorder=1,
    )
    _set_map_extent(ax, bbox)


def _draw_networks(
    ax: plt.Axes,
    frames: dict[str, pd.DataFrame],
    sectors: tuple[str, ...] = tuple(LABELS),
    native_width: bool = False,
) -> None:
    # Draw service laterals lightly and engineering backbones strongly.  This
    # is a symbology choice only: every generated edge remains in the figure.
    service_types = {
        "electricity": {"building_service"},
        "drinking_water": {"building_service"},
        "wastewater": {"property_lateral", "property_force_main"},
        "district_heating": {"building_service"},
    }
    styles = {
        "electricity": (0.78, "solid", 0.88),
        "drinking_water": (0.70, "solid", 0.78),
        "wastewater": (0.72, (0, (2.3, 1.4)), 0.80),
        "district_heating": (0.92, "solid", 0.90),
    }
    size_columns = {
        "electricity": "cable_size_mm2",
        "drinking_water": "diameter_mm",
        "wastewater": "diameter_mm",
        "district_heating": "diameter_mm",
    }
    for sector in sectors:
        edge = frames[f"{sector}_edges"]
        if sector == "electricity" and "asset_type" in edge:
            edge = edge[edge["asset_type"] != "transformer_connection"].copy()
        base_width, linestyle, alpha = styles[sector]
        widths: float | np.ndarray = base_width
        if native_width:
            size_series = edge[size_columns[sector]].astype(float)
            values = size_series.fillna(size_series.median()).to_numpy()
            lo, hi = np.percentile(values, [10, 95])
            widths = 0.35 + 2.25 * np.clip((values - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        type_col = "asset_type" if sector == "electricity" else "link_type"
        if type_col in edge and not native_width:
            service = edge[edge[type_col].isin(service_types[sector])]
            backbone = edge[~edge[type_col].isin(service_types[sector])]
            if not service.empty:
                ax.add_collection(LineCollection(
                    _edge_segments(service), colors=COLORS[sector], linewidths=0.10,
                    linestyles=linestyle, alpha=0.045, zorder=2, rasterized=True,
                ))
            if not backbone.empty:
                ax.add_collection(LineCollection(
                    _edge_segments(backbone), colors=COLORS[sector], linewidths=base_width,
                    linestyles=linestyle, alpha=alpha, zorder=3,
                ))
            # The MV tree is the upstream connection between radial LV areas;
            # use a distinct darker colour so it cannot be mistaken for islands.
            if sector == "electricity" and "voltage_level" in edge:
                mv = edge[edge["voltage_level"].eq("MV")]
                if not mv.empty:
                    ax.add_collection(LineCollection(
                        _edge_segments(mv), colors="#7E2948", linewidths=1.28,
                        linestyles=(0, (3.0, 1.6)), alpha=0.98, zorder=5,
                    ))
        else:
            ax.add_collection(LineCollection(
                _edge_segments(edge), colors=COLORS[sector], linewidths=widths,
                linestyles=linestyle, alpha=alpha, zorder=3,
            ))


def _draw_facilities(ax: plt.Axes, frames: dict[str, pd.DataFrame], sector: str | None = None) -> None:
    sectors = [sector] if sector else list(LABELS)
    for item in sectors:
        nodes = frames[f"{item}_nodes"]
        if item == "electricity":
            selected = nodes[nodes["root"].astype(bool)]
            marker = "s"
        elif item == "drinking_water":
            selected = nodes[nodes["node_type"].isin(["source", "storage", "wellfield"])]
            marker = "o"
        elif item == "wastewater":
            selected = nodes[nodes["node_type"].isin(["pump", "lift_station", "cso", "outfall"])]
            marker = "^"
        else:
            selected = nodes[nodes["node_type"] == "plant"]
            marker = "*"
        marker_size = (12 if len(selected) > 30 else 28) if item != "district_heating" else 58
        ax.scatter(
            selected["lon"], selected["lat"], s=marker_size,
            marker=marker, c=COLORS[item], edgecolors="white", linewidths=0.65,
            zorder=6,
        )


def _north_and_scale(ax: plt.Axes, km: float = 5.0) -> None:
    west, east = ax.get_xlim()
    south, north = ax.get_ylim()
    latitude = (south + north) / 2.0
    dx = km / (111.32 * math.cos(math.radians(latitude)))
    x0 = west + 0.045 * (east - west)
    y0 = south + 0.055 * (north - south)
    ax.plot([x0, x0 + dx], [y0, y0], color=COLORS["ink"], lw=2.0, zorder=8)
    ax.plot([x0, x0], [y0 - 0.006 * (north - south), y0 + 0.006 * (north - south)], color=COLORS["ink"], lw=1.0)
    ax.plot([x0 + dx, x0 + dx], [y0 - 0.006 * (north - south), y0 + 0.006 * (north - south)], color=COLORS["ink"], lw=1.0)
    ax.text(x0 + dx / 2.0, y0 + 0.018 * (north - south), f"{km:g} km", ha="center", va="bottom", fontsize=7.6)
    xn = east - 0.055 * (east - west)
    yn = north - 0.075 * (north - south)
    ax.annotate("N", xy=(xn, yn), xytext=(xn, yn - 0.055 * (north - south)), ha="center", va="center", fontsize=8.2, fontweight="bold", arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": COLORS["ink"]})


def _save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG / f"{stem}.pdf"
    png_path = FIG / f"{stem}.png"
    svg_path = FIG / f"{stem}.svg"
    pdf_tmp = FIG / f".{stem}.pdf.tmp"
    png_tmp = FIG / f".{stem}.png.tmp"
    svg_tmp = FIG / f".{stem}.svg.tmp"
    fig.savefig(pdf_tmp, format="pdf", bbox_inches="tight", pad_inches=0.04)
    pdf_tmp.replace(pdf_path)
    fig.savefig(png_tmp, format="png", dpi=240, bbox_inches="tight", pad_inches=0.04)
    png_tmp.replace(png_path)
    fig.savefig(svg_tmp, format="svg", bbox_inches="tight", pad_inches=0.04)
    svg_tmp.replace(svg_path)
    plt.close(fig)


def figure_framework() -> None:
    """Show evidence, native topology generation, and the coupled fixed point."""
    fig, ax = plt.subplots(figsize=(13.4, 7.65))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 7.65)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, text: str, color: str, fill: str = "white", fs: float = 6.65, title_fs: float = 7.6) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08", linewidth=1.05, edgecolor=color, facecolor=fill)
        ax.add_patch(patch)
        title_artist = ax.text(x + 0.12, y + h - 0.12, title, color=color, fontsize=title_fs, fontweight="bold", va="top", linespacing=1.05)
        body_artist = ax.text(x + 0.12, y + h - 0.42, text, color=COLORS["ink"], fontsize=fs, va="top", linespacing=1.20)
        title_artist.set_clip_path(patch)
        body_artist.set_clip_path(patch)

    def arrow(a: tuple[float, float], b: tuple[float, float], color: str = "#697580") -> None:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color=color, connectionstyle="arc3,rad=0.0"))

    ax.text(0.18, 7.39, "Shared-ledger topology synthesis followed by explicit multi-infrastructure coupling", fontsize=13.0, fontweight="bold")
    ax.text(0.18, 7.10, "Native models are accepted first; only then are solver duties and electrical states exchanged to a fixed point.", fontsize=8.15, color=COLORS["muted"])

    ax.add_patch(Rectangle((0.14, 6.48), 13.10, 0.34, color="#E9EDF0", ec="none"))
    ax.text(0.27, 6.65, "I  ·  COMMON EVIDENCE, DEMAND RECONCILIATION, AND CLUSTERING", va="center", fontsize=8.7, fontweight="bold", color="#4A5661")
    box(0.18, 5.26, 2.42, 0.96, "RAW INPUTS", "OSM buildings, roads, land use\nvisible facilities and terrain\nofficial aggregates + libraries", "#52616B", "#F5F7F8", 5.75)
    box(2.90, 5.26, 2.42, 0.96, "REGISTER + AUDIT", "clip the declared boundary\nvalidate IDs and geometry\nclassify evidence A/B/C/D", COLORS["shared"], "#F8F3FA", 5.75)
    box(5.62, 5.26, 2.42, 0.96, "BUILDING LEDGER", "stable building identity and use\nfloor-area / population priors\nfour sector-demand priors", COLORS["shared"], "#F8F3FA", 5.75)
    box(8.34, 5.26, 2.42, 0.96, "RECONCILE + CLUSTER", "scale once to sector totals\nk-means++ membership; weighted centroid\nsnap and merge duplicate road nodes", COLORS["shared"], "#F8F3FA", 5.75)
    box(11.06, 5.26, 2.18, 0.96, "FROZEN COMMON MODEL", "building-to-zone membership\nregistered corridors/facilities\nconserved four-sector demand", "#2F6F59", "#F2F7F4", 5.55)
    for x in [2.61, 5.33, 8.05, 10.77]:
        arrow((x, 5.74), (x + 0.27, 5.74))

    ax.add_patch(Rectangle((0.14, 4.62), 13.10, 0.34, color="#E5F0EA", ec="none"))
    ax.text(0.27, 4.79, "II  ·  SECTOR-NATIVE TOPOLOGY, DISCRETE SIZING, REPAIR, AND ACCEPTANCE", va="center", fontsize=8.7, fontweight="bold", color="#2F6F59")
    sector_boxes = [
        ("ELECTRICITY", "supply clusters → radial feeders\n+ normally-open ties", "pandapower convergence\n0.90–1.10 p.u.; loading ≤100%", COLORS["electricity"]),
        ("DRINKING WATER", "source paths → tree\n+ useful loop chords", "WNTR/EPANET convergence\npressure and velocity limits", COLORS["drinking_water"]),
        ("WASTEWATER", "terrain-cost paths → gravity\n+ lift stations / force mains", "SWMM continuity and convergence\nno dry-weather flooding", COLORS["wastewater"]),
        ("DISTRICT HEATING", "heat-density cells → territories\n+ route-aware partial-connection forest", "N/A accepted when no territory exists\notherwise hydraulics, velocity, route loss", COLORS["district_heating"]),
    ]
    for index, (title, topology, gate, color) in enumerate(sector_boxes):
        y = 3.91 - index * 0.72
        box(0.22, y, 2.18, 0.63, title, topology, color, "white", 4.85, 6.25)
        arrow((2.42, y + 0.315), (2.71, y + 0.315), color)
        box(2.73, y, 5.24, 0.63, "CONSTRUCT + SIZE", "route on common corridors · accumulate sector flow\nround up in declared library · apply only permitted repair", color, "#FAFAF8", 4.85, 6.15)
        arrow((7.99, y + 0.315), (8.28, y + 0.315), color)
        box(8.30, y, 4.88, 0.63, "NATIVE ACCEPTANCE GATE", gate, color, "#FAFAF8", 4.85, 6.15)
    ax.text(12.95, 4.79, "FAIL → revise affected row", ha="right", va="center", fontsize=6.5, color="#7B5A20", fontweight="bold")

    ax.add_patch(Rectangle((0.14, 1.38), 13.10, 0.34, color="#EFE7F4", ec="none"))
    ax.text(0.27, 1.55, "III  ·  BIDIRECTIONAL FIXED-POINT CO-SIMULATION AND FINAL EXPORT GATE", va="center", fontsize=8.7, fontweight="bold", color=COLORS["shared"])
    box(0.20, 0.39, 2.18, 0.81, "EXPLICIT INTERFACES", "facility↔bus IDs\nwater delivery→sanitary inflow\nshared-corridor identities", COLORS["shared"], "#F8F3FA", 5.6)
    box(2.72, 0.39, 2.50, 0.81, "SECTOR SOLVES", "WNTR + SWMM + pandapipes\nQ, H, flow, temperature → motor MW", "#2F6F59", "#F2F7F4", 5.6)
    box(5.58, 0.39, 2.26, 0.81, "ELECTRICAL SOLVE", "pandapower returns bus voltage,\nloading, and energization", COLORS["electricity"], "#FFF5F3", 5.6)
    box(8.20, 0.39, 2.42, 0.81, "FIXED-POINT TEST", "ΔV · ΔP · delivered water · return T\nconverged and every limit satisfied?", COLORS["shared"], "#F8F3FA", 5.55)
    box(10.98, 0.39, 2.20, 0.81, "ACCEPTED EXPORT", "all applicable native files + states\ninterface ledger + gate certificate", "#2F6F59", "#F2F7F4", 5.6)
    for x1, x2 in [(2.40, 2.70), (5.24, 5.56), (7.86, 8.18), (10.64, 10.96)]:
        arrow((x1, 0.795), (x2, 0.795))
    ax.add_patch(FancyArrowPatch((7.05, 0.38), (3.68, 0.38), arrowstyle="-|>", mutation_scale=10, linewidth=1.25, color=COLORS["electricity"], connectionstyle="arc3,rad=-0.08"))
    ax.text(
        5.36, 0.31,
        "bus voltage/energization → availability, speed, head, flow, operation",
        ha="center", fontsize=6.05, color=COLORS["electricity"], zorder=12,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )
    ax.add_patch(FancyArrowPatch((9.40, 1.23), (3.70, 1.23), arrowstyle="-|>", mutation_scale=9, linewidth=0.9, linestyle=(0, (3, 2)), color="#9A6B20", connectionstyle="arc3,rad=0.0"))
    ax.text(6.55, 1.27, "not converged / limit violated → revise interface or operation, then repeat", ha="center", fontsize=6.25, color="#7B5A20")
    _save(fig, "fig01_framework")


def figure_clustering_hierarchy() -> None:
    """Separate numerical demand zoning from sector-specific supply clustering."""
    case_frames = _load_benchmark_case("dense_urban")
    root = OUT / "benchmark_cases" / "dense_urban"
    zones = pd.read_csv(root / "common_demand_zones.csv")
    transformers = pd.read_csv(root / "electricity_transformers.csv")
    heat_nodes = pd.read_csv(root / "district_heating_nodes.csv")
    consumers = heat_nodes[heat_nodes["node_type"] == "consumer_substation"]

    fig = plt.figure(figsize=(13.4, 6.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.18], wspace=0.045)
    ax = fig.add_subplot(grid[0, 0])
    map_ax = fig.add_subplot(grid[0, 1])
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 6.15)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str, fill: str, body_fs: float = 6.8) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08", facecolor=fill, edgecolor=color, linewidth=1.05))
        patch = ax.patches[-1]
        title_artist = ax.text(x + 0.13, y + h - 0.13, title, va="top", fontsize=8.0, fontweight="bold", color=color)
        body_artist = ax.text(x + 0.13, y + h - 0.46, body, va="top", fontsize=body_fs, linespacing=1.20)
        title_artist.set_clip_path(patch)
        body_artist.set_clip_path(patch)

    def arrow(a: tuple[float, float], b: tuple[float, float], color: str = COLORS["shared"]) -> None:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, linewidth=1.0))

    ax.text(0.08, 5.88, "One building identity, three distinct clustering roles", fontsize=12.2, fontweight="bold")
    ax.text(0.08, 5.57, "Common zones preserve demand identity; supply clusters answer sector-specific planning questions.", fontsize=7.8, color=COLORS["muted"])
    box(0.12, 4.55, 5.55, 0.77, "LEVEL 0  ·  REGISTERED BUILDINGS", "OSM ID + use + floor area → population, electricity, water, sanitary wastewater, and heat priors", "#52616B", "#F5F7F8")
    arrow((2.90, 4.52), (2.90, 4.20))
    box(0.12, 3.27, 5.55, 0.90, "LEVEL 1  ·  COMMON DEMAND-ZONE PARTITION", "K-means++ assigns spatial membership; each non-empty cluster then receives a floor-area-weighted centroid.\nCentroids snap to the road graph, duplicate snaps merge, and all reconciled totals remain conserved.", COLORS["shared"], "#F8F3FA")
    arrow((2.90, 3.24), (2.90, 2.90))
    box(0.12, 1.78, 1.70, 1.08, "ELECTRICITY", "Assign zones by road distance.\nAdd the farthest zone as a site\nuntil reach / capacity rules pass.", COLORS["electricity"], "#FFF5F3", 6.15)
    box(2.05, 1.78, 1.70, 1.08, "WATER + SEWER", "Use every common zone as a\nhydraulic demand or inflow.\nNo second customer partition.", COLORS["drinking_water"], "#F3F7FD", 6.15)
    box(3.98, 1.78, 1.70, 1.08, "DISTRICT HEAT", "Rank eligible connections.\nCluster connected demand to source boundaries\nand hydraulic consumer nodes.", COLORS["district_heating"], "#FFF9EC", 6.15)
    for x, color in [(0.97, COLORS["electricity"]), (2.90, COLORS["drinking_water"]), (4.83, COLORS["district_heating"])]:
        arrow((2.90, 3.24), (x, 2.90), color)
    box(0.12, 0.43, 5.55, 0.88, "ACCEPTANCE OF THE HIERARCHY", "One assignment per building · non-empty zones · valid road snap\nDemand conservation: relative error $<10^{-6}$\nSupply reach, capacity, and native solvers are checked later; demand totals remain frozen.", "#2F6F59", "#F2F7F4", 6.45)

    dense_bbox = next(item["bbox"] for item in BENCHMARK_CASES if item["id"] == "dense_urban")
    _draw_base(map_ax, case_frames["buildings"], dense_bbox, building_alpha=0.34)
    map_ax.scatter(zones["lon"], zones["lat"], s=13, c=COLORS["shared"], edgecolors="white", linewidths=0.35, zorder=5)
    map_ax.scatter(transformers["lon"], transformers["lat"], s=28, marker="s", c=COLORS["electricity"], edgecolors="white", linewidths=0.55, zorder=6)
    map_ax.scatter(consumers["lon"], consumers["lat"], s=25, marker="D", c=COLORS["district_heating"], edgecolors="white", linewidths=0.5, zorder=6)
    _north_and_scale(map_ax, 0.5)
    map_ax.set_title("Schweinfurt Innenstadt: independently initialized dense-urban run", loc="left", fontsize=10.5, fontweight="bold", pad=5)
    map_ax.text(0.018, 0.027, f"{len(case_frames['buildings']):,} buildings  →  {len(zones)} common zones  →  {len(transformers)} transformer clusters\nHeat uses {len(consumers)} hydraulic consumer nodes; water and sewer retain all common zones.", transform=map_ax.transAxes, fontsize=7.35, linespacing=1.34, bbox={"boxstyle": "round,pad=0.28", "fc": "white", "ec": "#C8CDD2", "alpha": 0.93}, zorder=10)
    handles = [
        Line2D([0], [0], marker="o", lw=0, markerfacecolor=COLORS["building"], markeredgecolor="none", markersize=5, label="Registered building"),
        Line2D([0], [0], marker="o", lw=0, markerfacecolor=COLORS["shared"], markeredgecolor="white", markersize=6, label="Common demand zone"),
        Line2D([0], [0], marker="s", lw=0, markerfacecolor=COLORS["electricity"], markeredgecolor="white", markersize=6, label="Transformer supply cluster"),
        Line2D([0], [0], marker="D", lw=0, markerfacecolor=COLORS["district_heating"], markeredgecolor="white", markersize=6, label="Heat consumer aggregation"),
    ]
    map_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.91), fontsize=7.15, frameon=True, facecolor="white", edgecolor="#D3D7DA", framealpha=0.92)
    fig.subplots_adjust(left=0.015, right=0.995, top=0.985, bottom=0.02)
    _save(fig, "fig02_clustering_hierarchy")


def figure_topology_logic() -> None:
    """Expose the four topology algorithms as explicit planner decisions."""
    fig, ax = plt.subplots(figsize=(13.4, 7.25))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 7.25)
    ax.axis("off")

    columns = [(1.55, 2.65), (4.42, 2.73), (7.37, 2.82), (10.41, 2.77)]
    headers = ["1  ANCHORS + DEMAND", "2  CONSTRUCT GRAPH", "3  SIZE / REPAIR", "4  ACCEPT + EXPORT"]
    for (x, width), header in zip(columns, headers):
        ax.text(x + width / 2, 6.45, header, ha="center", va="center", fontsize=8.7, fontweight="bold", color=COLORS["muted"])

    rows = [
        (
            "electricity", "ELECTRICITY",
            "Initial sites = max(ceil(P/0.50),\nceil(|Z|/4)); K-means centres\nare snapped to road nodes.",
            "Add the farthest zone until\nmax d$_G$(zone, site) ≤ R$_m$.\nShortest-path union → MST;\nopen MV ties only in asset graph.",
            "Downstream coincident power.\n80% transformer/ampacity rule;\n4.5–5.5% feeder ΔV budget;\nchoose a discrete cable + circuits.",
            "Operating cycle rank = 0;\n0.90–1.10 p.u.; line and\ntransformer loading ≤100%;\npandapower converges.",
        ),
        (
            "drinking_water", "DRINKING WATER",
            "Mapped source/storage; 20 m\nelevation bands; 0.12 km demand-access\ncells; 341 km full-city main scale\nregistered before construction.",
            "Connect source to aggregated access\nnodes through a backbone; attach buildings\nby short laterals; add second feeds to\ncritical and high-demand access nodes.",
            "Accumulate peak flow; target\nv = 1.2 m s$^{-1}$; round up to DN80–600.\nUpsize local friction bottlenecks before\na bounded source-head adjustment.",
            "Connected; cycle rank > 0; proxy\n20–110 m pressure envelope; v ≤2 m s$^{-1}$.\nRevised EPANET candidate exported;\nWNTR native rerun pending.",
        ),
        (
            "wastewater", "WASTEWATER",
            "Mapped treatment outfall; 0.21 km\nterrain-aware subcatchments; 236 km gravity,\n6,200 manholes, 13 km pressure mains, and\n13 station compounds registered.",
            "Build an outfall-directed network over\nmanholes, not buildings; place interval,\ndirection, and junction manholes; attach\nproperties through short laterals.",
            "Apply hard cover and grade tests;\nclassify gravity and pressure-transition\nsegments; size with Manning n=0.013 and\nDN200–2000 catalogues.",
            "Directed acyclic graph; every node\nreaches the outfall; inventory and cover\nscreens pass structurally. Measured DEM,\nstation consolidation, and SWMM rerun pending.",
        ),
        (
            "district_heating", "DISTRICT HEATING",
            "Screen 250 m cells by gross-area\nheat density. Merge queen-adjacent\nsuitable cells; permit local islands or\na valid zero-network outcome.",
            "Apply the public connection quota.\nDemand-ranked or route-aware selection\ngrows source-rooted corridor forests;\nexport paired supply/return records.",
            "$\\dot{m}=\\dot{Q}/(c_p\\Delta T)$, $\\Delta T=35$ K.\nRound up to DN25–DN800; report\nroute loss, differential pressure,\nand circulation-pump duty.",
            "Plant reachable; $\\Delta p≥0.5$ bar;\nvelocity ≤2.0 m s$^{-1}$; route loss ≤15%;\npandapipes converges for accepted baseline.",
        ),
    ]
    row_y = [5.23, 3.82, 2.41, 1.00]
    for (key, label, *cells), y in zip(rows, row_y):
        color = COLORS[key]
        ax.add_patch(FancyBboxPatch((0.18, y), 1.13, 1.05, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=color, edgecolor=color, linewidth=1.0))
        ax.text(0.745, y + 0.525, label.replace(" ", "\n", 1), ha="center", va="center", fontsize=7.6, fontweight="bold", color="white", linespacing=1.12)
        for index, ((x, width), text_value) in enumerate(zip(columns, cells)):
            ax.add_patch(FancyBboxPatch((x, y), width, 1.05, boxstyle="round,pad=0.04,rounding_size=0.07", facecolor="white", edgecolor=color, linewidth=1.0))
            ax.text(x + 0.12, y + 0.91, text_value, va="top", fontsize=7.15, linespacing=1.20)
            if index < 3:
                ax.add_patch(FancyArrowPatch((x + width + 0.015, y + 0.525), (columns[index + 1][0] - 0.02, y + 0.525), arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=color))

    ax.text(0.18, 6.98, "Sector-native topology construction, sizing, and stopping rules", fontsize=13.0, fontweight="bold")
    ax.text(0.18, 6.68, "Every row consumes the same registered zones and corridor graph; only the engineering decision sequence changes.", fontsize=8.2, color=COLORS["muted"])
    ax.text(0.18, 0.37, "Hard screen = topology + native numerical envelope.  Calibration target = case-specific published count/total.  Comparison metric = reported without forcing a pass.", fontsize=8.1, fontweight="bold")
    _save(fig, "fig02_topology_logic")


def figure_integrated_map(frames: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(12.2, 6.35))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.48, 1.48, 1.0, 1.0], wspace=0.055, hspace=0.13)
    ax = fig.add_subplot(grid[:, :2])
    _draw_base(ax, frames["buildings"], CASE["bbox"], building_alpha=0.16)
    _draw_networks(ax, frames)
    _draw_facilities(ax, frames)
    letters = "BCDE"
    for letter, item in zip(letters, CASE["morphologies"]):
        west, south, east, north = item["map_bbox"]
        ax.add_patch(Rectangle((west, south), east - west, north - south, fill=False, edgecolor=MORPH_COLORS[item["id"]], linewidth=1.1, linestyle=(0, (3, 2)), zorder=7))
        ax.text(west + 0.004 * (east - west), north - 0.012 * (north - south), letter, ha="left", va="top", fontsize=8.2, fontweight="bold", color="white", bbox={"boxstyle": "square,pad=0.13", "fc": MORPH_COLORS[item["id"]], "ec": "none"}, zorder=9)
    _north_and_scale(ax, 5.0)
    ax.set_title("(a) Full Schweinfurt integration", loc="left", fontweight="bold", pad=5)

    zoom_axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2, 4)]
    for letter, item, zoom in zip(letters, CASE["morphologies"], zoom_axes):
        _draw_base(zoom, frames["buildings"], item["map_bbox"], building_alpha=0.25)
        _draw_networks(zoom, frames)
        _draw_facilities(zoom, frames)
        name = item["label"].split(" - ")[-1]
        zoom.set_title(f"({letter.lower()}) {name}", loc="left", fontsize=10.1, color=MORPH_COLORS[item["id"]], fontweight="bold", pad=3)
        _north_and_scale(zoom, 0.5 if item["id"] == "dense" else 1.0)

    handles = [Line2D([0], [0], color=COLORS[k], lw=2.1, linestyle=(0, (2.3, 1.4)) if k == "wastewater" else "-") for k in LABELS]
    handles.insert(1, Line2D([0], [0], color="#7E2948", lw=1.8, ls="--"))
    handles += [Line2D([0], [0], marker=m, color="none", markerfacecolor="#59636E", markeredgecolor="white", markersize=7, label=l) for m, l in [("s", "Power source"), ("o", "Water facility"), ("^", "Wastewater facility"), ("*", "Heat-source boundary")]]
    labels = ["LV/electric", "MV backbone", "Drinking water", "Wastewater", "District heating"] + ["Power source", "Water facility", "Wastewater facility", "Heat-source boundary"]
    fig.legend(handles, labels, ncol=9, loc="lower center", bbox_to_anchor=(0.5, 0.004), columnspacing=0.75, handlelength=1.55, fontsize=8.1)
    fig.subplots_adjust(bottom=0.082, top=0.965, left=0.02, right=0.99)
    _save(fig, "fig03_integrated_map")


def figure_morphology_zooms() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.75))
    metrics = pd.read_csv(OUT / "benchmark_case_metrics.csv")
    for ax, item in zip(axes.flat, BENCHMARK_CASES):
        case_id = item["id"]
        morphology = item["morphology"]
        case_frames = _load_benchmark_case(case_id)
        _draw_base(ax, case_frames["buildings"], item["bbox"], building_alpha=0.28)
        _draw_networks(ax, case_frames)
        _draw_facilities(ax, case_frames)
        subset = metrics[metrics["case_id"] == case_id]
        density = subset["building_density_per_km2"].iloc[0]
        route = subset["route_length_km"].sum()
        heat = subset.loc[subset["sector"] == "district_heating", "district_heat_contract_equivalents"].iloc[0]
        panel = "ABC"[list(BENCHMARK_CASES).index(item)]
        ax.set_title(f"({panel.lower()}) {item['place_name']} — {item['morphology_label']}", loc="left", color=MORPH_COLORS[morphology], fontweight="bold", pad=4, fontsize=10.2)
        ax.text(0.015, 0.025, f"{density:,.0f} buildings km$^{{-2}}$\n{route:.0f} km total generated route · {heat:.0f} heat connections", transform=ax.transAxes, fontsize=8.0, linespacing=1.30, bbox={"boxstyle": "round,pad=0.23", "fc": "white", "ec": "#CBD0D5", "alpha": 0.90}, zorder=10)
        _north_and_scale(ax, 0.5 if morphology == "dense" else 1.0)
    handles = [Line2D([0], [0], color=COLORS[k], lw=2.0, linestyle=(0, (2.3, 1.4)) if k == "wastewater" else "-") for k in LABELS]
    handles.insert(1, Line2D([0], [0], color="#7E2948", lw=1.7, ls="--"))
    fig.legend(handles, ["LV/electric", "MV backbone", "Drinking water", "Wastewater", "District heating"], ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.005), columnspacing=1.05, fontsize=8.2)
    fig.subplots_adjust(wspace=0.055, bottom=0.11, top=0.93, left=0.015, right=0.995)
    _save(fig, "fig02_independent_context_cases")


def figure_sector_maps(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 9.25))
    inventory = frames["inventory"].set_index("sector")
    for ax, sector in zip(axes.flat, LABELS):
        _draw_base(ax, frames["buildings"], CASE["bbox"], building_alpha=0.12)
        _draw_networks(ax, frames, sectors=(sector,), native_width=True)
        _draw_facilities(ax, frames, sector)
        row = inventory.loc[sector]
        suffix = "mm²" if sector == "electricity" else "mm"
        ax.set_title(f"{LABELS[sector]} — {int(row.nodes):,} nodes, {row.route_length_km:.1f} km", loc="left", fontweight="bold", color=COLORS[sector], pad=5)
        ax.text(0.015, 0.025, f"Line width ∝ size  ·  median {row.size_median:g} {suffix}  ·  P90 {row.size_p90:g} {suffix}", transform=ax.transAxes, fontsize=7.3, bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.82})
        _north_and_scale(ax, 5.0)
    fig.suptitle("Sector-native topology and engineering size allocation", x=0.05, y=0.995, ha="left", fontsize=13.0, fontweight="bold")
    fig.subplots_adjust(hspace=0.13, wspace=0.075, top=0.955, bottom=0.02, left=0.02, right=0.99)
    _save(fig, "fig04_sector_maps")


def figure_morphology_metrics(frames: dict[str, pd.DataFrame]) -> None:
    data = pd.read_csv(OUT / "benchmark_case_metrics.csv")
    seeds = pd.read_csv(OUT / "clustering_robustness_summary.csv")
    heat = pd.read_csv(OUT / "heat_selection_analysis" / "heat_selection_threshold_sweep.csv")
    morphs = [item["morphology"] for item in BENCHMARK_CASES]
    x = np.arange(len(morphs))
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 5.9))
    width = 0.18
    for offset, sector in enumerate(LABELS):
        subset = data[data["sector"] == sector].set_index("morphology").loc[morphs]
        axes[0, 0].bar(
            x + (offset - 1.5) * width,
            subset["network_length_per_1000_buildings_km"], width,
            color=COLORS[sector], label=LABELS[sector],
        )
        axes[0, 1].bar(
            x + (offset - 1.5) * width,
            subset["average_service_route_km"], width, color=COLORS[sector],
        )

    seed_order = [item["id"] for item in BENCHMARK_CASES]
    seed_metric = seeds[seeds.metric.eq("nearest_zone_spacing_p95_km")].set_index("case_id").loc[seed_order]
    median = seed_metric["median"].to_numpy(float)
    axes[1, 0].errorbar(
        x, median,
        yerr=np.vstack((median - seed_metric["p05"].to_numpy(float), seed_metric["p95"].to_numpy(float) - median)),
        fmt="o", color=COLORS["shared"], ecolor=COLORS["shared"], capsize=4,
        lw=1.5, ms=6,
    )
    axes[1, 0].plot(x, median, color=COLORS["shared"], lw=1.0, alpha=0.7)

    heat_styles = {
        "demand_ranked": (COLORS["shared"], "s", "Demand-ranked"),
        "route_aware": (COLORS["district_heating"], "o", "Route-aware"),
    }
    for policy, (colour, marker, label) in heat_styles.items():
        subset = heat[heat.selection_policy.eq(policy)].sort_values("minimum_heat_density_mwh_ha")
        axes[1, 1].plot(
            subset.minimum_heat_density_mwh_ha, subset.route_trench_km,
            marker=marker, lw=1.9, color=colour, label=label,
        )
    canonical = heat[heat.minimum_heat_density_mwh_ha.eq(800.0)].set_index("selection_policy")
    if {"demand_ranked", "route_aware"}.issubset(canonical.index):
        reduction = 100.0 * (1.0 - canonical.loc["route_aware", "route_trench_km"] / canonical.loc["demand_ranked", "route_trench_km"])
        axes[1, 1].annotate(
            f"{reduction:.0f}% less trench\nat 800 MWh ha$^{{-1}}$ a$^{{-1}}$",
            xy=(800.0, canonical.loc["route_aware", "route_trench_km"]),
            xytext=(520.0, 15.5),
            arrowprops={"arrowstyle":"->","lw":0.9,"color":COLORS["district_heating"]},
            fontsize=7.3, color=COLORS["district_heating"],
        )

    titles = [
        "(a) Route per 1,000 registered buildings",
        "(b) Mean source-to-service route",
        "(c) Common-zone initialization spread",
        "(d) Heat screen and customer-selection sensitivity",
    ]
    ylabels = [
        "Route (km / 1,000 buildings)", "Mean route (km)",
        "P95 nearest-zone spacing (km)", "Heat trench route (km)",
    ]
    case_labels = [
        "Innenstadt\n(dense)", "Bergl/Gartenstadt\n(semi-urban)",
        "Dittelbrunn\n(peripheral)",
    ]
    for index, (ax, title, ylabel) in enumerate(zip(axes.flat, titles, ylabels)):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        if index < 3:
            ax.set_xticks(x, case_labels, fontsize=8.0)
        else:
            ax.set_xlabel("Minimum heat density (MWh ha$^{-1}$ a$^{-1}$)")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylim(0, max(axes[0, 0].get_ylim()[1], 115))
    axes[0, 0].legend(ncol=2, fontsize=8.0, loc="upper right")
    axes[1, 1].legend(loc="upper right", fontsize=8.0)
    fig.subplots_adjust(hspace=0.40, wspace=0.30, top=0.965, bottom=0.13, left=0.085, right=0.91)
    _save(fig, "fig05_topology_response")


def figure_equipment_distributions(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.9))
    size_columns = {
        "electricity": ("cable_size_mm2", "Cable cross-section (mm²)"),
        "drinking_water": ("diameter_mm", "Nominal pipe diameter (mm)"),
        "wastewater": ("diameter_mm", "Nominal sewer diameter (mm)"),
        "district_heating": ("diameter_mm", "Nominal pipe diameter (mm)"),
    }
    for ax, sector in zip(axes.flat, LABELS):
        column, xlabel = size_columns[sector]
        counts = frames[f"{sector}_edges"][column].dropna().value_counts().sort_index()
        shares = counts / counts.sum() * 100.0
        y = np.arange(len(counts))
        ax.barh(y, shares, color=COLORS[sector], alpha=0.88)
        ax.set_yticks(y, [f"{v:g}" for v in counts.index])
        ax.set_xlabel("Share of generated links (%)")
        ax.set_ylabel(xlabel)
        ax.set_title(LABELS[sector], loc="left", fontweight="bold", color=COLORS[sector])
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for yi, value in zip(y, shares):
            if value >= 4:
                ax.text(value + 0.5, yi, f"{value:.0f}%", va="center", fontsize=7.3)
        ax.set_xlim(0, max(shares.max() * 1.18, 20))
    fig.suptitle("Discrete engineering libraries produce interpretable equipment portfolios", x=0.055, y=0.995, ha="left", fontsize=13.0, fontweight="bold")
    fig.subplots_adjust(hspace=0.31, wspace=0.27, top=0.94, bottom=0.08, left=0.09, right=0.98)
    _save(fig, "fig06_equipment_distributions")


def figure_coupling(frames: dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(11.1, 5.35))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.05, 1.0], wspace=0.17, hspace=0.36)
    ax_map = fig.add_subplot(grid[:, 0])
    ax_convergence = fig.add_subplot(grid[0, 1])
    ax_interface = fig.add_subplot(grid[1, 1])
    _draw_base(ax_map, frames["buildings"], CASE["bbox"], building_alpha=0.10)
    corridors = frames["corridors"]
    corridor_colors = {2: "#65809A", 3: "#8757A5", 4: "#C03888"}
    for count in [2, 3, 4]:
        subset = corridors[corridors["sector_count"] == count]
        segments = [[(r.from_lon, r.from_lat), (r.to_lon, r.to_lat)] for r in subset.itertuples()]
        ax_map.add_collection(LineCollection(segments, colors=corridor_colors[count], linewidths=0.65 + 0.34 * count, alpha=0.83, zorder=4))
    _north_and_scale(ax_map, 5.0)
    ax_map.set_title("Road segments reused by two or more generated sectors", loc="left", fontweight="bold")
    ax_map.legend([Line2D([0], [0], color=corridor_colors[c], lw=1.2 + c * 0.3) for c in [2, 3, 4]], ["2 sectors", "3 sectors", "4 sectors"], ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.07))

    service = OUT / "integrated_service_resolved"
    history = pd.read_csv(service / "coupled_nominal_convergence_history.csv")
    coupling_manifest = json.loads((service / "coupled_nominal_interface_manifest.json").read_text(encoding="utf-8"))
    iterations = history["iteration"].to_numpy()
    voltage_residual = history["maximum_voltage_residual_pu"].clip(lower=1e-10)
    power_residual = history["maximum_power_residual_mw"].clip(lower=1e-10)
    ax_convergence.semilogy(iterations, voltage_residual, marker="o", ms=4.0, color=COLORS["electricity"], label="max |ΔV| (p.u.)")
    ax_convergence.semilogy(iterations, power_residual, marker="s", ms=3.8, color=COLORS["shared"], label="max |ΔP| (MW)")
    ax_convergence.axhline(1e-6, color=COLORS["electricity"], ls="--", lw=0.8, alpha=0.7)
    ax_convergence.axhline(1e-5, color=COLORS["shared"], ls="--", lw=0.8, alpha=0.7)
    ax_convergence.set_xticks(iterations)
    ax_convergence.set_xlabel("Coupling iteration")
    ax_convergence.set_ylabel("Fixed-point residual")
    ax_convergence.set_title("Bidirectional state exchange converges", loc="left", fontweight="bold")
    ax_convergence.grid(True, which="both", axis="y")
    ax_convergence.spines[["top", "right"]].set_visible(False)
    ax_convergence.legend(fontsize=7.0, ncol=2, loc="upper right")
    ax_convergence.text(0.02, 0.05, f"accepted after {len(history)} iterations · min terminal {coupling_manifest['minimum_terminal_voltage_pu']:.3f} p.u.", transform=ax_convergence.transAxes, fontsize=7.0, fontweight="bold")

    state = pd.read_csv(service / "coupled_facility_interface_states.csv")
    capacity = state.groupby("to_sector")["closed_interface_power_mw"].sum().reindex(["drinking_water", "wastewater", "district_heating"]).fillna(0.0)
    interface_labels = ["Drinking water", "Wastewater", "District heating"]
    y = np.arange(len(capacity))
    ax_interface.barh(y, capacity, color=[COLORS["drinking_water"], COLORS["wastewater"], COLORS["district_heating"]])
    ax_interface.set_yticks(y, interface_labels)
    ax_interface.invert_yaxis()
    ax_interface.set_xlabel("Converged electrical duty (MW)")
    ax_interface.set_title(f"Final interface duties — {capacity.sum():.3f} MW", loc="left", fontweight="bold")
    ax_interface.grid(axis="x")
    ax_interface.set_axisbelow(True)
    ax_interface.spines[["top", "right"]].set_visible(False)
    for yi, value in zip(y, capacity):
        ax_interface.text(value + max(capacity) * 0.025, yi, f"{value:.2f} MW", va="center", fontsize=8.0)
    fig.subplots_adjust(top=0.965, bottom=0.12, left=0.04, right=0.98)
    _save(fig, "fig07_coupling_corridors")


def figure_hydraulic_leak_response() -> None:
    """Show elapsed-time propagation of the controlled hydraulic leak."""
    data = pd.read_csv(OUT / "hydraulic_leak_coupling" / "hydraulic_leak_timeseries.csv")
    manifest = json.loads(
        (OUT / "hydraulic_leak_coupling" / "hydraulic_leak_manifest.json").read_text(encoding="utf-8")
    )
    event_minute = int(manifest["scenario"]["leak_start_minute"])
    time = data["elapsed_minute"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.25), sharex=True)

    def event_background(ax: plt.Axes) -> None:
        ax.axvspan(event_minute, float(time.max()), color="#DCEAF7", alpha=0.45, zorder=0)
        ax.axvline(event_minute, color="#41566C", ls="--", lw=0.9, zorder=5)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 0]
    event_background(ax)
    pump_line = ax.plot(
        time, data["pump_flow_l_s"], color=COLORS["drinking_water"], lw=2.1,
        marker="o", ms=2.8, markevery=2, label="Pump flow",
    )[0]
    ax.set_ylabel("Pump flow (L s$^{-1}$)")
    ax.set_ylim(125, 159)
    ax2 = ax.twinx()
    leak_line = ax2.plot(
        time, data["leak_flow_l_s"], color="#00A6A6", lw=1.8,
        ls="--", label="Leak flow",
    )[0]
    ax2.fill_between(time, 0, data["leak_flow_l_s"], color="#00A6A6", alpha=0.10)
    ax2.set_ylabel("Leak flow (L s$^{-1}$)", color="#007D7D")
    ax2.tick_params(axis="y", colors="#007D7D")
    ax2.set_ylim(0, max(27.0, float(data["leak_flow_l_s"].max()) * 1.10))
    ax.legend([pump_line, leak_line], ["Pump flow", "Leak flow"], loc="lower right", ncol=2, fontsize=8.6)
    ax.set_title("(a) Leak and source flow", loc="left", fontweight="bold")

    ax = axes[0, 1]
    event_background(ax)
    ax.plot(
        time, data["pump_discharge_pressure_m"], color="#285F9E", lw=2.0,
        label="Pump-discharge node",
    )
    ax.plot(
        time, data["critical_node_pressure_m"], color="#D55E00", lw=2.0,
        label="Hydraulically weakest node",
    )
    ax.axhline(20.0, color="#A33A3A", ls=":", lw=1.2, label="20 m acceptance limit")
    ax.axhline(
        float(manifest["target_discharge_pressure_m"]), color="#285F9E",
        ls="--", lw=0.9, alpha=0.7,
    )
    ax.set_ylim(15, 114)
    ax.set_ylabel("Pressure (m)")
    ax.legend(loc="center right", fontsize=8.4)
    ax.set_title("(b) Pressure response", loc="left", fontweight="bold")

    ax = axes[1, 0]
    event_background(ax)
    command_line = ax.plot(
        time, data["pump_speed_command_pu"], color="#704A8A", lw=1.5,
        ls="--", label="Speed command",
    )[0]
    actual_line = ax.plot(
        time, data["pump_actual_speed_pu"], color="#9A6CB4", lw=1.8,
        label="Speed after returned-voltage adapter",
    )[0]
    ax.set_ylabel("Pump speed (p.u.)")
    ax.set_ylim(0.985, 1.055)
    ax.legend(
        [command_line, actual_line],
        ["Controller command", "After voltage adapter"],
        loc="lower right", fontsize=8.4,
    )
    ax.set_title("(c) Bounded pump-speed control", loc="left", fontweight="bold")

    ax = axes[1, 1]
    event_background(ax)
    power_line = ax.plot(
        time, 1000.0 * data["pump_electrical_power_mw"], color=COLORS["electricity"],
        lw=2.1, marker="s", ms=2.8, markevery=2, label="Pump active power",
    )[0]
    ax.set_ylabel("Pump active power (kW)")
    ax.set_ylim(190, 250)
    ax2 = ax.twinx()
    baseline_voltage = float(data.loc[data["elapsed_minute"] < event_minute, "pump_bus_voltage_pu"].iloc[-1])
    voltage_change = 1e4 * (data["pump_bus_voltage_pu"] - baseline_voltage)
    voltage_line = ax2.plot(
        time, voltage_change, color="#2B3440", lw=1.8,
        ls="--", label=r"Pump-bus $\Delta V$",
    )[0]
    ax2.set_ylabel(r"Pump-bus $\Delta V$ ($10^{-4}$ p.u.)", color="#2B3440")
    ax2.tick_params(axis="y", colors="#2B3440")
    margin = max(0.25, 0.15 * max(abs(float(voltage_change.min())), abs(float(voltage_change.max())), 1.0))
    ax2.set_ylim(float(voltage_change.min()) - margin, float(voltage_change.max()) + margin)
    ax.legend(
        [power_line, voltage_line], ["Pump active power", r"Pump-bus $\Delta V$"],
        loc="lower right", fontsize=8.4,
    )
    ax.text(
        0.02, 0.08, r"System max-line-loading change $<10^{-8}$ percentage points",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.1, color=COLORS["muted"],
    )
    ax.set_title("(d) Electrical duty and local voltage", loc="left", fontweight="bold")

    for ax in axes[1, :]:
        ax.set_xlabel("Elapsed operating time (min)")
    for ax in axes.flat:
        ax.set_xlim(float(time.min()), float(time.max()))
        ax.set_xticks(np.arange(float(time.min()), float(time.max()) + 1, 15))
    fig.text(
        0.98, 0.012, "Blue shading begins at leak opening; points are sequential quasi-steady fixed points.",
        ha="right", fontsize=8.0, color=COLORS["muted"],
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.34, top=0.965, bottom=0.13, left=0.085, right=0.92)
    _save(fig, "fig10_hydraulic_leak_response")


def figure_electrical_supply_disturbance() -> None:
    """Show electrical-to-water propagation and the returned load response."""
    data = pd.read_csv(
        OUT / "electrical_supply_disturbance" / "electrical_supply_disturbance_timeseries.csv"
    )
    manifest = json.loads(
        (
            OUT
            / "electrical_supply_disturbance"
            / "electrical_supply_disturbance_manifest.json"
        ).read_text(encoding="utf-8")
    )
    scenario = manifest["scenario"]
    start = int(scenario["fault_start_minute"])
    clear = int(scenario["fault_clear_start_minute"])
    recovered = int(scenario["voltage_recovery_complete_minute"])
    time = data["elapsed_minute"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.25), sharex=True)

    def event_background(ax: plt.Axes) -> None:
        ax.axvspan(start, clear, color="#F3D8D4", alpha=0.62, zorder=0)
        ax.axvspan(clear, recovered, color="#F3E5BD", alpha=0.62, zorder=0)
        ax.axvline(start, color="#8F3A32", ls="--", lw=0.9, zorder=5)
        ax.axvline(clear, color="#8B6B1D", ls="--", lw=0.9, zorder=5)
        ax.axvline(recovered, color="#397255", ls=":", lw=0.9, zorder=5)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 0]
    event_background(ax)
    terminal = ax.plot(
        time,
        data["drive_terminal_voltage_pu"],
        color=COLORS["electricity"],
        lw=2.2,
        marker="o",
        ms=2.8,
        markevery=2,
        label="Pump-drive terminal",
    )[0]
    bus = ax.plot(
        time,
        data["connection_bus_voltage_pu"],
        color="#354052",
        lw=1.5,
        ls="--",
        label=f"Grid bus {manifest['connection_bus_id']}",
    )[0]
    factor = ax.plot(
        time,
        data["service_voltage_factor"],
        color="#7B5EA7",
        lw=1.4,
        ls=":",
        label="Service-voltage factor",
    )[0]
    ax.axhline(0.90, color="#B58219", lw=1.0, ls="--", label="Drive recovery threshold")
    ax.axhline(0.85, color="#9E2A2B", lw=1.0, ls=":", label="Drive trip threshold")
    ax.set_ylim(0.82, 1.06)
    ax.set_ylabel("Voltage / factor (p.u.)")
    ax.legend([terminal, bus, factor], [terminal.get_label(), bus.get_label(), factor.get_label()], loc="lower right", fontsize=8.2)
    ax.set_title("(a) Local electrical depression", loc="left", fontweight="bold")

    ax = axes[0, 1]
    event_background(ax)
    speed = ax.plot(
        time,
        data["pump_actual_speed_pu"],
        color="#7B5EA7",
        lw=2.1,
        marker="o",
        ms=2.8,
        markevery=2,
        label="Pump speed",
    )[0]
    ax.set_ylim(0.60, 1.05)
    ax.set_ylabel("Pump speed (p.u.)")
    ax2 = ax.twinx()
    power = ax2.plot(
        time,
        1000.0 * data["pump_electrical_power_mw"],
        color=COLORS["electricity"],
        lw=1.9,
        ls="--",
        label="Pump active power",
    )[0]
    ax2.set_ylim(50, 220)
    ax2.set_ylabel("Pump active power (kW)", color=COLORS["electricity"])
    ax2.tick_params(axis="y", colors=COLORS["electricity"])
    ax.legend([speed, power], ["Pump speed", "Pump active power"], loc="lower right", fontsize=8.3)
    ax.set_title("(b) Voltage-sensitive pump drive", loc="left", fontweight="bold")

    ax = axes[1, 0]
    event_background(ax)
    pressure = data["critical_node_pressure_m"].to_numpy()
    ax.plot(
        time,
        pressure,
        color=COLORS["drinking_water"],
        lw=2.2,
        marker="s",
        ms=2.8,
        markevery=2,
        label="Hydraulically weakest node",
    )
    ax.axhline(20.0, color="#9E2A2B", ls="--", lw=1.1, label="20 m normal-state limit")
    ax.fill_between(time, pressure, 20.0, where=pressure < 20.0, color="#D95F59", alpha=0.18)
    ax.set_ylim(min(-25.0, float(pressure.min()) - 4.0), 50.0)
    ax.set_ylabel("Critical pressure (m)")
    ax.legend(loc="lower right", fontsize=8.3)
    ax.set_title("(c) Hydraulic pressure violation", loc="left", fontweight="bold")

    ax = axes[1, 1]
    event_background(ax)
    delivered = ax.plot(
        time,
        100.0 * data["delivered_water_fraction"],
        color=COLORS["drinking_water"],
        lw=2.2,
        marker="o",
        ms=2.8,
        markevery=2,
        label="Delivered water",
    )[0]
    ax.set_ylim(82, 101)
    ax.set_ylabel("Delivered water (%)")
    ax2 = ax.twinx()
    loading = ax2.plot(
        time,
        data["maximum_line_loading_percent"],
        color="#354052",
        lw=1.8,
        ls="--",
        label="Maximum line loading",
    )[0]
    ax2.set_ylim(79.5, 82.5)
    ax2.set_ylabel("Maximum line loading (%)", color="#354052")
    ax2.tick_params(axis="y", colors="#354052")
    ax.legend([delivered, loading], ["Delivered water", "Maximum line loading"], loc="lower right", fontsize=8.3)
    ax.text(
        0.02, 0.08, r"System max-line-loading change $<10^{-8}$ percentage points",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.1, color=COLORS["muted"],
    )
    ax.set_title("(d) Water delivery and grid loading", loc="left", fontweight="bold")

    for ax in axes[1, :]:
        ax.set_xlabel("Elapsed operating time (min)")
    for ax in axes.flat:
        ax.set_xlim(float(time.min()), float(time.max()))
        ax.set_xticks(np.arange(float(time.min()), float(time.max()) + 1, 15))
    fig.text(
        0.98,
        0.012,
        "Red: depressed local service voltage; amber: prescribed recovery; points are sequential quasi-steady fixed points.",
        ha="right",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.34, top=0.965, bottom=0.13, left=0.085, right=0.92)
    _save(fig, "fig11_electrical_supply_disturbance")


def figure_experiment_decomposition() -> None:
    controlled = pd.read_csv(OUT / "benchmark_case_metrics.csv")
    policy = pd.read_csv(OUT / "benchmark_policy_case_metrics.csv")
    seeds = pd.read_csv(OUT / "clustering_seed_sensitivity.csv")
    ablation = pd.read_csv(OUT / "common_ledger_ablation.csv")
    order = [item["id"] for item in BENCHMARK_CASES]
    names = ["Innenstadt\n(dense)", "Bergl/Gartenstadt\n(semi-urban)", "Dittelbrunn\n(peripheral)"]
    x = np.arange(3)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.5))

    c_power = controlled[controlled["sector"] == "electricity"].set_index("case_id").loc[order]
    p_power = policy[policy["sector"] == "electricity"].set_index("case_id").loc[order]
    axes[0, 0].bar(x - 0.18, c_power["maximum_service_route_km"], 0.36, color="#72849A", label="Controlled")
    axes[0, 0].bar(x + 0.18, p_power["maximum_service_route_km"], 0.36, color="#D78731", label="Context-aware policy")
    axes[0, 0].set_ylabel("Maximum LV service route (km)")
    axes[0, 0].legend(fontsize=7.6)

    c_heat = controlled[controlled["sector"] == "district_heating"].set_index("case_id").loc[order]
    p_heat = policy[policy["sector"] == "district_heating"].set_index("case_id").loc[order]
    axes[0, 1].bar(x - 0.18, c_heat["district_heat_contract_equivalents"], 0.36, color="#72849A")
    axes[0, 1].bar(x + 0.18, p_heat["district_heat_contract_equivalents"], 0.36, color="#D78731")
    axes[0, 1].set_ylabel("Represented heat connections")

    seed_data = [seeds.loc[seeds["case_id"] == case_id, "water_zone_cv"].to_numpy() for case_id in order]
    boxes = axes[1, 0].boxplot(seed_data, positions=x, widths=0.55, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], [MORPH_COLORS[item["morphology"]] for item in BENCHMARK_CASES]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1, 0].set_ylabel("Zone water-demand coefficient of variation")

    a = ablation.set_index("case_id").loc[order]
    axes[1, 1].bar(x, a["independent_sector_clustering_mean_interface_offset_km"], color="#A56A9A", alpha=0.86)
    axes[1, 1].plot(x, a["independent_sector_clustering_p95_interface_offset_km"], color=COLORS["ink"], marker="o", lw=1.5, label="P95")
    axes[1, 1].axhline(0, color="#3A8F67", lw=2.0, label="Common ledger: exact identity")
    axes[1, 1].set_ylabel("Water–sewer interface offset (km)")
    axes[1, 1].legend(fontsize=7.2)

    titles = [
        "A  LV route: evidence effect versus policy effect",
        "B  Heat-zone selection rule: controlled versus policy-aware",
        "C  Twenty-seed clustering robustness",
        "D  Common-ledger ablation",
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9.4)
        ax.set_xticks(x, names, fontsize=7.6)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Controlled settlement-context response is separated from planning policy and stochastic clustering", x=0.055, y=0.995, ha="left", fontsize=12.8, fontweight="bold")
    fig.subplots_adjust(hspace=0.32, wspace=0.24, top=0.93, bottom=0.09, left=0.08, right=0.98)
    _save(fig, "fig09_experiment_decomposition")


def figure_verification(frames: dict[str, pd.DataFrame]) -> None:
    manifest = json.loads((OUT / "model_manifest_four_sector.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(13.4, 7.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], hspace=0.42, wspace=0.25)
    ax_constraint = fig.add_subplot(grid[0, 0])
    ax_comparison = fig.add_subplot(grid[1, 0])
    ax_solver = fig.add_subplot(grid[0, 1])
    ax_diagnostic = fig.add_subplot(grid[1, 1])

    labels = {
        "Electricity LV annual energy": "LV annual energy",
        "Electricity LV coincident peak": "LV coincident peak",
        "MS/LV withdrawal locations": "MS/LV withdrawal locations",
        "MS/LV installed capacity": "MS/LV installed capacity",
        "Direct-retail drinking-water allocation": "Direct-retail water",
        "District-heat represented sales contracts": "Heat sales contracts",
        "District-heat annual sales": "Heat annual sales",
        "Drinking-water route": "Water route",
        "Wastewater route inventory": "Wastewater route",
        "Wastewater force mains": "Force-main route",
        "Wastewater pump stations": "Derived / published lift stations",
        "Wastewater overflow basins": "Overflow basins",
        "District-heat generated route": "Heat route",
    }
    anchors = frames["anchors"].copy()
    anchors["ratio"] = anchors["generated"] / anchors["reference"]
    constraints = anchors[anchors["evidence_role"] == "input_constraint"].copy()
    comparisons = anchors[anchors["evidence_role"] == "held_out_comparison"].copy()

    for ax, data, title, color in [
        (ax_constraint, constraints, "1  Input constraints satisfied — not validation", "#72849A"),
        (ax_comparison, comparisons, "2  Held-out external comparisons", "#3A8F67"),
    ]:
        y = np.arange(len(data))
        bar_colors = [color] * len(data) if ax is ax_constraint else ["#3A8F67" if 0.9 <= value <= 1.1 else "#D78731" for value in data["ratio"]]
        ax.barh(y, data["ratio"], color=bar_colors, height=0.68)
        ax.axvline(1.0, color=COLORS["ink"], lw=1.0)
        if ax is ax_comparison:
            ax.axvspan(0.9, 1.1, color="#DDEDE5", alpha=0.65, zorder=0)
        ax.set_yticks(y, [labels.get(item, item) for item in data["metric"]], fontsize=7.4)
        ax.invert_yaxis()
        ax.set_xlim(0, max(2.2, float(data["ratio"].max()) * 1.18))
        ax.set_xlabel("Generated / published quantity")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for yi, ratio in zip(y, data["ratio"]):
            ax.text(ratio + 0.025, yi, f"{(ratio - 1) * 100:+.1f}%", va="center", fontsize=7.0)

    solver = manifest["solver_readiness"]
    coupled = solver["bidirectional_cosimulation"]
    final = coupled["final_sector_states"]
    power = final["electricity"]
    water = final["drinking_water"]
    swmm = final["wastewater"]
    heat = final["district_heating"]
    cards = [
        ("pandapower fixed point", f"{power['minimum_voltage_pu']:.3f}–{power['maximum_voltage_pu']:.3f} p.u.; line {power['maximum_line_loading_percent']:.1f}%", COLORS["electricity"]),
        ("WNTR explicit head pump", f"{water['minimum_pressure_m']:.1f}–{water['maximum_pressure_m']:.1f} m; delivered {100*water['delivered_water_fraction']:.1f}%", COLORS["drinking_water"]),
        ("SWMM", f"|continuity| {abs(swmm['flow_routing_continuity_error_percent']):.3f}%; nonconvergence {swmm['steps_not_converging_percent']:.2f}%; flooding {swmm['flooding_loss_million_liter']:.3f} ML", COLORS["wastewater"]),
        ("pandapipes paired circuit", f"min {heat['minimum_pressure_bar']:.1f} bar; return {heat['return_temperature_c']:.1f} °C; max {heat['maximum_velocity_m_s']:.2f} m/s", COLORS["district_heating"]),
    ]
    ax_solver.axis("off")
    ax_solver.set_title("3  Accepted coupled fixed-point state", loc="left", fontweight="bold", pad=5)
    for index, (title, text, color) in enumerate(cards):
        y0 = 0.78 - index * 0.23
        ax_solver.add_patch(FancyBboxPatch((0.02, y0), 0.94, 0.16, transform=ax_solver.transAxes, boxstyle="round,pad=0.015", facecolor="white", edgecolor=color, linewidth=1.15))
        ax_solver.text(0.05, y0 + 0.115, title, transform=ax_solver.transAxes, fontsize=8.3, fontweight="bold", color=color)
        ax_solver.text(0.05, y0 + 0.045, text, transform=ax_solver.transAxes, fontsize=7.0)

    ax_diagnostic.axis("off")
    ax_diagnostic.set_title("4  Diagnostics and interpretation limits", loc="left", fontweight="bold", pad=5)
    native_water = solver["epanet_wntr"]
    diagnostic_lines = [
        f"• Fixed point: {coupled['iterations']} iterations; {len(coupled.get('repairs_applied', []))} recorded operational repair; final interface duty {power['coupling_active_power_mw']:.3f} MW.",
        f"• Water pressure initially breached the 110 m ceiling after voltage feedback; the accepted pump command is documented in the repair ledger.",
        f"• {100*native_water['fraction_links_below_0_05_m_s']:.1f}% of water links are below 0.05 m/s; 48-h age P95 is {native_water['water_age_p95_hours_at_48h']:.1f} h.",
        f"• Paired heat supply/return is solved thermohydraulically: {heat['delivered_heat_mw']:.2f} MW delivered, {heat['distribution_heat_loss_mw']:.2f} MW distribution loss.",
        "• Building footprints and measured DEM are absent from the retained snapshot; archetype areas and class-C terrain remain limitations.",
    ]
    ax_diagnostic.text(0.02, 0.92, "\n\n".join(diagnostic_lines), transform=ax_diagnostic.transAxes, va="top", fontsize=7.5, linespacing=1.30, wrap=True)
    fig.suptitle("Constraint satisfaction, external comparison, numerical checks, and diagnostics are not conflated", x=0.04, y=0.995, ha="left", fontsize=12.8, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.18, right=0.98)
    _save(fig, "fig08_verification")


def main() -> None:
    frames = _load_frames()
    figure_framework()
    figure_clustering_hierarchy()
    figure_topology_logic()
    figure_morphology_zooms()
    figure_integrated_map(frames)
    figure_sector_maps(frames)
    figure_morphology_metrics(frames)
    figure_equipment_distributions(frames)
    figure_coupling(frames)
    figure_hydraulic_leak_response()
    figure_electrical_supply_disturbance()
    figure_experiment_decomposition()
    figure_verification(frames)
    print(f"Wrote 13 benchmark figures to {FIG}")


if __name__ == "__main__":
    main()
