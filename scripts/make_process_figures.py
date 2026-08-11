#!/usr/bin/env python3
"""Build the two spatial process figures used by the Urban4 manuscript.

The script visualizes existing accepted CSV outputs; it does not regenerate or
re-solve any infrastructure model.  ``--data-root`` is the root of an Urban4
reproducibility checkout containing ``outputs/benchmark_cases/dense_urban`` and
``data/raw/local_roads.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

from urban4 import schweinfurt_base as base


COLORS = {
    "electricity": "#D84A3A",
    "electricity_mv": "#7E2948",
    "drinking_water": "#2474D2",
    "wastewater": "#354052",
    "district_heating": "#E69F00",
    "shared": "#7A3E9D",
    "building": "#9AA3AA",
    "road": "#D5D8D9",
    "ink": "#18212B",
    "muted": "#66717E",
    "paper": "#FBFBF9",
    "green": "#2F7D59",
}

USE_COLORS = {
    "Residential": "#CF6B72",
    "Commercial/public": "#4A86C5",
    "Industrial": "#8B5AA8",
    "Auxiliary": "#A9A397",
    "Other": "#6D8C76",
}

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
        "font.size": 9.8,
        "axes.titlesize": 9.2,
        "axes.edgecolor": "#AEB5BA",
        "axes.linewidth": 0.65,
        "axes.facecolor": COLORS["paper"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": COLORS["ink"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def road_segments(path: Path) -> list[list[tuple[float, float]]]:
    data = base.read_json_path(path)
    segments: list[list[tuple[float, float]]] = []
    for element in data["elements"]:
        geometry = element.get("geometry") or []
        if len(geometry) > 1:
            segments.append([(float(p["lon"]), float(p["lat"])) for p in geometry])
    return segments


def geometry_segments(frame: pd.DataFrame) -> list[list[tuple[float, float]]]:
    return [json.loads(value) for value in frame["geometry_json"]]


def geometry_intersects_bbox(value: str, bbox: list[float]) -> bool:
    """Return True when a polyline's bounding box intersects ``bbox``."""
    points = json.loads(value)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return max(xs) >= bbox[0] and min(xs) <= bbox[2] and max(ys) >= bbox[1] and min(ys) <= bbox[3]


def set_extent(ax: plt.Axes, bbox: list[float]) -> None:
    west, south, east, north = bbox
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1.0 / math.cos(math.radians((south + north) / 2.0)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B7BDC4")


def draw_base(
    ax: plt.Axes,
    roads: list[list[tuple[float, float]]],
    buildings: pd.DataFrame,
    bbox: list[float],
    *,
    points: bool = True,
    building_alpha: float = 0.34,
) -> pd.DataFrame:
    ax.add_collection(
        LineCollection(roads, colors=COLORS["road"], linewidths=0.34, alpha=0.83, zorder=0)
    )
    subset = buildings[
        buildings["lon"].between(bbox[0], bbox[2])
        & buildings["lat"].between(bbox[1], bbox[3])
    ]
    if points:
        ax.scatter(
            subset["lon"], subset["lat"], s=2.0, c=COLORS["building"],
            alpha=building_alpha, edgecolors="none", rasterized=True, zorder=1,
        )
    set_extent(ax, bbox)
    return subset


def panel_title(ax: plt.Axes, letter: str, title: str, subtitle: str | None = None) -> None:
    ax.set_title(f"({letter})  {title}", loc="left", fontweight="bold", pad=5.5)
    if subtitle:
        ax.text(
            0.012, 0.018, subtitle, transform=ax.transAxes, fontsize=6.6,
            color=COLORS["ink"], linespacing=1.18,
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#CCD1D4", "alpha": 0.94},
            zorder=12,
        )


def statistics_strip(ax: plt.Axes, text: str) -> None:
    """Place map statistics in a dedicated strip instead of over the map."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.012, 0.66, text, ha="left", va="center", fontsize=6.55,
        color=COLORS["ink"], linespacing=1.16,
    )


def use_group(value: str) -> str:
    text = str(value).lower()
    if any(token in text for token in ("house", "residential", "apart", "terrace", "detached", "dorm")):
        return "Residential"
    if any(token in text for token in ("commercial", "retail", "office", "school", "hospital", "public")):
        return "Commercial/public"
    if any(token in text for token in ("industrial", "warehouse")):
        return "Industrial"
    if any(token in text for token in ("roof", "garage", "shed", "carport")):
        return "Auxiliary"
    return "Other"


def service_group(value: str) -> str:
    """Map the accepted ledger class to the compact figure legend.

    ``building_type`` is the raw OSM/InfDB attribute and may legitimately be
    generic.  Demand synthesis, however, uses ``service_class`` after the
    evidence-aware registration step.  Plotting the former as if it were the
    latter makes generic ``building=yes`` records look unclassified even when
    their accepted ledger class is residential.
    """
    text = str(value).lower()
    if text == "residential":
        return "Residential"
    if text in {"commercial", "public"}:
        return "Commercial/public"
    if text == "industrial":
        return "Industrial"
    if text == "auxiliary":
        return "Auxiliary"
    return "Other"


def reconstruct_clustering(buildings: pd.DataFrame, count: int = 79, seed: int = 20260803):
    xy = np.column_stack((71.55 * buildings["lon"].to_numpy(), 111.20 * buildings["lat"].to_numpy()))
    _, labels = kmeans2(xy, count, minit="++", iter=60, seed=seed)
    centroids = []
    for cluster in range(count):
        members = buildings.iloc[np.where(labels == cluster)[0]]
        if members.empty:
            centroids.append((np.nan, np.nan))
            continue
        weights = np.maximum(members["floor_area_m2"].to_numpy(dtype=float), 20.0)
        centroids.append(
            (
                float(np.average(members["lon"], weights=weights)),
                float(np.average(members["lat"], weights=weights)),
            )
        )
    return labels, np.asarray(centroids)


def add_flow_arrows(fig: plt.Figure, axes: np.ndarray) -> None:
    # Snake order mirrors pylovo's compact process layout: top row left-to-right,
    # then the bottom row right-to-left.  It avoids a diagonal arrow across maps.
    ordered = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2], axes[1, 1], axes[1, 0]]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for left_idx, right_idx in pairs:
        left = ordered[left_idx].get_position()
        right = ordered[right_idx].get_position()
        if left_idx == 2:
            # Route the row-turn arrow through the right margin so it does not
            # cross the lower-right panel title.
            start = (0.994, left.y0 - 0.004)
            end = (0.994, right.y1 + 0.004)
        else:
            moving_right = right.x0 > left.x0
            start = ((left.x1 + 0.002) if moving_right else (left.x0 - 0.002), (left.y0 + left.y1) / 2)
            end = ((right.x0 - 0.002) if moving_right else (right.x1 + 0.002), (right.y0 + right.y1) / 2)
        fig.add_artist(
            FancyArrowPatch(
                start, end, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10,
                lw=0.9, color="#77818A", zorder=30,
            )
        )


def add_evidence_process_arrows(fig: plt.Figure, axes: np.ndarray) -> None:
    """Draw only the unobstructed horizontal parts of the Fig. 2 sequence."""
    colour = "#77818A"

    def horizontal(source: plt.Axes, target: plt.Axes) -> None:
        left = source.get_position()
        right = target.get_position()
        moving_right = right.x0 > left.x0
        start = (
            left.x1 + 0.004 if moving_right else left.x0 - 0.004,
            (left.y0 + left.y1) / 2,
        )
        end = (
            right.x0 - 0.004 if moving_right else right.x1 + 0.004,
            (right.y0 + right.y1) / 2,
        )
        fig.add_artist(
            FancyArrowPatch(
                start, end, transform=fig.transFigure, arrowstyle="-|>",
                mutation_scale=10, lw=0.9, color=colour, zorder=30,
            )
        )

    horizontal(axes[0, 0], axes[0, 1])
    horizontal(axes[0, 1], axes[0, 2])
    horizontal(axes[1, 2], axes[1, 1])
    horizontal(axes[1, 1], axes[1, 0])


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def figure_evidence_to_clusters(data_root: Path, output: Path) -> None:
    case = data_root / "outputs" / "benchmark_cases" / "dense_urban"
    buildings = pd.read_csv(case / "common_building_ledger.csv")
    zones = pd.read_csv(case / "common_demand_zones.csv")
    roads = road_segments(data_root / "data" / "raw" / "local_roads.json")
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    seed = int(manifest["controlled_parameters"]["seed"])
    requested = max(12, int(round(len(buildings) / manifest["controlled_parameters"]["buildings_per_requested_zone"])))
    labels, centroids = reconstruct_clustering(buildings, count=requested, seed=seed)
    buildings = buildings.copy()
    buildings["cluster"] = labels
    buildings["use_group"] = buildings["service_class"].map(service_group)

    # Every panel uses the same accepted dense-case boundary.  Earlier versions
    # mixed a 2,852-building zoom with the 6,326-building ledger, which made the
    # apparent counts and class proportions impossible to compare.
    focus = list(manifest["bbox"])
    # Dedicated statistics rows keep explanatory text and the colour scale off
    # the maps.  Unlike Fig. 3, this is a true six-step process, so a compact
    # snake connector is retained.
    fig = plt.figure(figsize=(7.16, 5.10))
    grid = fig.add_gridspec(
        4, 3,
        height_ratios=(1.0, 0.180, 1.0, 0.145),
        left=0.027, right=0.972, bottom=0.092, top=0.962,
        wspace=0.095, hspace=0.185,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, col]) for col in range(3)],
            [fig.add_subplot(grid[2, col]) for col in range(3)],
        ],
        dtype=object,
    )
    strips = np.asarray(
        [
            [fig.add_subplot(grid[1, col]) for col in range(3)],
            [fig.add_subplot(grid[3, col]) for col in range(3)],
        ],
        dtype=object,
    )

    ax = axes[0, 0]
    subset = draw_base(ax, roads, buildings, focus)
    panel_title(ax, "a", "Spatial evidence")
    statistics_strip(
        strips[0, 0],
        f"{len(subset):,} buildings | road corridors retained\n"
        "Stable source IDs and provenance attached",
    )

    ax = axes[0, 1]
    draw_base(ax, roads, buildings, focus, points=False)
    for group, color in USE_COLORS.items():
        part = buildings[(buildings["use_group"] == group) & buildings["lon"].between(focus[0], focus[2]) & buildings["lat"].between(focus[1], focus[3])]
        ax.scatter(part["lon"], part["lat"], s=3.0, c=color, alpha=0.78, edgecolors="none", rasterized=True)
    window_counts = (
        buildings[
            buildings["lon"].between(focus[0], focus[2])
            & buildings["lat"].between(focus[1], focus[3])
        ]["use_group"].value_counts()
    )
    residential_count = int(window_counts.get("Residential", 0))
    residential_share = 100.0 * residential_count / max(int(window_counts.sum()), 1)
    panel_title(ax, "b", "Building service classes")
    statistics_strip(
        strips[0, 1],
        f"Residential: {residential_count:,}/{int(window_counts.sum()):,} "
        f"({residential_share:.1f}%)\n"
        "Evidence: source tag, address, or inference",
    )

    ax = axes[0, 2]
    draw_base(ax, roads, buildings, focus, points=False)
    part = buildings[buildings["lon"].between(focus[0], focus[2]) & buildings["lat"].between(focus[1], focus[3])].copy()
    score = (
        part["electricity_peak_kw"] / max(part["electricity_peak_kw"].median(), 1e-9)
        + part["water_m3_year"] / max(part["water_m3_year"].median(), 1e-9)
        + part["heat_candidate_mwh_year"] / max(part["heat_candidate_mwh_year"].median(), 1e-9)
    )
    sizes = 2.0 + 12.0 * np.clip(np.log1p(score) / np.log1p(np.percentile(score, 98)), 0, 1)
    scatter = ax.scatter(
        part["lon"], part["lat"], s=sizes, c=part["electricity_peak_kw"], cmap="magma_r",
        alpha=0.72, edgecolors="none", rasterized=True, zorder=3,
    )
    panel_title(ax, "c", "Four-demand ledger")
    strips[0, 2].set_axis_off()
    strips[0, 2].set_xlim(0, 1)
    strips[0, 2].set_ylim(0, 1)
    cbar_ax = strips[0, 2].inset_axes([0.055, 0.420, 0.890, 0.180])
    cbar = fig.colorbar(scatter, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=5.55, length=1.8, pad=1.2)
    cbar.ax.set_title("Electricity peak $P_b$ (kW)", fontsize=5.9, pad=1.0)

    ax = axes[1, 2]
    draw_base(ax, roads, buildings, focus, points=False)
    palette = plt.get_cmap("turbo")
    part = buildings[buildings["lon"].between(focus[0], focus[2]) & buildings["lat"].between(focus[1], focus[3])]
    ax.scatter(part["lon"], part["lat"], s=3.1, c=part["cluster"], cmap=palette, alpha=0.72, edgecolors="none", rasterized=True)
    visible_centres = centroids[(centroids[:, 0] >= focus[0]) & (centroids[:, 0] <= focus[2]) & (centroids[:, 1] >= focus[1]) & (centroids[:, 1] <= focus[3])]
    ax.scatter(visible_centres[:, 0], visible_centres[:, 1], s=24, marker="x", c="black", linewidths=0.85, zorder=6)
    panel_title(ax, "d", "K-means++ clusters")
    statistics_strip(
        strips[1, 2],
        f"{requested} requests | seed {seed} | max. 60 iterations\n"
        "Spatial memberships are unweighted",
    )

    ax = axes[1, 1]
    draw_base(ax, roads, buildings, focus, points=True, building_alpha=0.18)
    visible_zones = zones[zones["lon"].between(focus[0], focus[2]) & zones["lat"].between(focus[1], focus[3])]
    tree = cKDTree(zones[["lon", "lat"]].to_numpy())
    for centre in visible_centres:
        _, idx = tree.query(centre)
        target = zones.iloc[int(idx)]
        ax.annotate(
            "", xy=(target.lon, target.lat), xytext=(centre[0], centre[1]),
            arrowprops={"arrowstyle": "-|>", "lw": 0.6, "color": COLORS["shared"], "alpha": 0.72},
            zorder=5,
        )
    ax.scatter(visible_centres[:, 0], visible_centres[:, 1], s=18, marker="x", c="#4C535A", linewidths=0.75, zorder=6)
    ax.scatter(visible_zones["lon"], visible_zones["lat"], s=26, c=COLORS["shared"], edgecolors="white", linewidths=0.45, zorder=7)
    panel_title(ax, "e", "Weighted centroid snapping")
    statistics_strip(
        strips[1, 1],
        "Floor-area centroid to nearest road node\n"
        "Duplicate snaps merged; memberships retained",
    )

    ax = axes[1, 0]
    draw_base(ax, roads, buildings, focus, points=True, building_alpha=0.13)
    marker_sizes = 14 + 75 * visible_zones["electricity_peak_mw"] / max(zones["electricity_peak_mw"].quantile(0.95), 1e-9)
    ax.scatter(
        visible_zones["lon"], visible_zones["lat"], s=np.clip(marker_sizes, 14, 95),
        c=COLORS["shared"], edgecolors="white", linewidths=0.55, zorder=7,
    )
    panel_title(ax, "f", "Accepted zone ledger")
    statistics_strip(
        strips[1, 0],
        f"{len(buildings):,} buildings to {len(zones)} zones | size: zone peak\n"
        r"Demand totals conserved ($<10^{-6}$ relative error)",
    )

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="none", markersize=5, label=k) for k, c in USE_COLORS.items()]
    handles += [
        Line2D([0], [0], marker="x", color="#4C535A", lw=0, markersize=5, label="weighted centroid"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["shared"], markeredgecolor="white", markersize=5, label="road-snapped zone"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=6.3, frameon=False, bbox_to_anchor=(0.5, 0.010), handletextpad=0.32, columnspacing=0.78)
    add_evidence_process_arrows(fig, axes)
    save(fig, output, "fig02_evidence_to_clusters")


def draw_network(ax: plt.Axes, frame: pd.DataFrame, color: str, *, dashed: bool = False, width: float = 1.1) -> None:
    ax.add_collection(
        LineCollection(
            geometry_segments(frame), colors=color, linewidths=width,
            linestyles=(0, (2.4, 1.35)) if dashed else "solid", alpha=0.88, zorder=4,
        )
    )


def node_subset(frame: pd.DataFrame, values: tuple[str, ...]) -> pd.DataFrame:
    return frame[frame["node_type"].isin(values)]


def rooted_component_map(
    nodes: pd.DataFrame,
    links: pd.DataFrame,
    *,
    node_id: str,
    root_types: tuple[str, ...],
) -> dict[str, int]:
    """Assign each node to a root-centred undirected component.

    Roots are ordered west-to-east for stable colours across repeated renders.
    The accepted heat benchmark has exactly one heat-source boundary per component.
    """
    adjacency = {str(value): set() for value in nodes[node_id]}
    for left, right in links[["from_node", "to_node"]].itertuples(index=False, name=None):
        left, right = str(left), str(right)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    roots = nodes[nodes["node_type"].isin(root_types)].sort_values(["lon", "lat"])
    assignment: dict[str, int] = {}
    for component, root in enumerate(roots[node_id].astype(str)):
        if root in assignment:
            continue
        assignment[root] = component
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency.get(current, set()):
                if neighbour not in assignment:
                    assignment[neighbour] = component
                    queue.append(neighbour)
    return assignment


def figure_clusters_to_topologies_legacy(data_root: Path, output: Path) -> None:
    case = data_root / "outputs" / "benchmark_cases" / "dense_urban"
    buildings = pd.read_csv(case / "common_building_ledger.csv")
    zones = pd.read_csv(case / "common_demand_zones.csv")
    transformers = pd.read_csv(case / "electricity_transformers.csv")
    roads = road_segments(data_root / "data" / "raw" / "local_roads.json")
    sectors = {
        "electricity": (pd.read_csv(case / "electricity_nodes.csv"), pd.read_csv(case / "electricity_links.csv")),
        "drinking_water": (pd.read_csv(case / "drinking_water_nodes.csv"), pd.read_csv(case / "drinking_water_links.csv")),
        "wastewater": (pd.read_csv(case / "wastewater_nodes.csv"), pd.read_csv(case / "wastewater_links.csv")),
        "district_heating": (pd.read_csv(case / "district_heating_nodes.csv"), pd.read_csv(case / "district_heating_links.csv")),
    }
    bbox = [10.215, 50.037, 10.249, 50.060]
    # The independent tile terminates at 26 upstream 20-kV boundary buses.
    # Plot the already accepted integrated-city MV backbone as a contextual
    # overlay only; it is not included in the tile's graph or solver metrics.
    mv_context = pd.read_csv(data_root / "outputs" / "power_lines.csv")
    mv_context = mv_context[
        mv_context["voltage_level"].eq("MV")
        & mv_context["asset_type"].eq("mv_backbone")
        & mv_context["geometry_json"].map(lambda value: geometry_intersects_bbox(value, bbox))
    ].copy()
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.50))
    # Figure caption carries the global explanation; equal panel rectangles use
    # the available journal-column width without a duplicate in-figure heading.
    plt.subplots_adjust(left=0.027, right=0.992, bottom=0.105, top=0.950, wspace=0.095, hspace=0.235)

    ax = axes[0, 0]
    draw_base(ax, roads, buildings, bbox, points=False)
    typed_buildings = buildings.copy()
    typed_buildings["use_group"] = typed_buildings["building_type"].map(use_group)
    # Draw the dominant unclassified group first so tagged uses remain visible.
    draw_order = ["Other", "Auxiliary", "Residential", "Commercial/public", "Industrial"]
    for group in draw_order:
        part = typed_buildings[typed_buildings["use_group"].eq(group)]
        sizes = np.clip(1.1 + 0.095 * np.sqrt(part["floor_area_m2"].to_numpy()), 1.2, 8.5)
        ax.scatter(
            part["lon"], part["lat"], s=sizes, c=USE_COLORS[group],
            alpha=0.58 if group == "Other" else 0.78, edgecolors="none",
            rasterized=True, zorder=1 if group == "Other" else 2,
        )
    ax.scatter(
        zones["lon"], zones["lat"], s=17, facecolors="white",
        edgecolors=COLORS["shared"], linewidths=0.85, zorder=7,
    )
    panel_title(
        ax, "a", "Common building ledger",
        "6,326 buildings · 39 OSM tags · 79 common zones\nmarker area follows the floor-area archetype",
    )

    ax = axes[0, 1]
    draw_base(ax, roads, buildings, bbox, points=True, building_alpha=0.075)
    draw_network(ax, mv_context, COLORS["electricity_mv"], dashed=True, width=0.95)
    draw_network(ax, sectors["electricity"][1], COLORS["electricity"], width=1.20)
    electric_loads = sectors["electricity"][0][
        sectors["electricity"][0]["node_type"].isin(("load", "transformer_load"))
    ]
    ax.scatter(
        electric_loads["lon"], electric_loads["lat"], s=11,
        facecolors="white", edgecolors=COLORS["electricity"], linewidths=0.65, zorder=6,
    )
    ax.scatter(transformers["lon"], transformers["lat"], s=18, marker="s", c=COLORS["electricity"], edgecolors="white", linewidths=0.45, zorder=7)
    colocated = sectors["electricity"][0][sectors["electricity"][0]["node_type"].eq("transformer_load")]
    ax.scatter(colocated["lon"], colocated["lat"], s=5, marker="o", c="white", edgecolors=COLORS["ink"], linewidths=0.35, zorder=8)
    panel_title(
        ax, "b", "Electricity - MV + LV",
        "dark dashed = accepted city-case MV context\ncoral = 26 zone-level LV equivalents (79 loads)",
    )

    ax = axes[0, 2]
    draw_base(ax, roads, buildings, bbox, points=True, building_alpha=0.075)
    draw_network(ax, sectors["drinking_water"][1], COLORS["drinking_water"], width=0.95)
    water_sources = node_subset(sectors["drinking_water"][0], ("source", "reservoir", "storage"))
    water_demands = node_subset(sectors["drinking_water"][0], ("demand",))
    ax.scatter(water_demands["lon"], water_demands["lat"], s=8, facecolors="white", edgecolors=COLORS["drinking_water"], linewidths=0.55, zorder=6)
    ax.scatter(water_sources["lon"], water_sources["lat"], s=30, marker="*", c=COLORS["drinking_water"], edgecolors="white", linewidths=0.5, zorder=7)
    panel_title(ax, "c", "Water - looped", "79/79 demands are source-connected · 14 cycles\nEPANET/WNTR pass")

    ax = axes[1, 2]
    draw_base(ax, roads, buildings, bbox, points=True, building_alpha=0.075)
    sewer_links = sectors["wastewater"][1]
    draw_network(ax, sewer_links[sewer_links["link_type"].eq("gravity")], COLORS["wastewater"], dashed=True, width=1.05)
    lifts = sewer_links[~sewer_links["link_type"].eq("gravity")]
    if not lifts.empty:
        draw_network(ax, lifts, "#7A3E9D", width=1.35)
    sewer_inflows = node_subset(sectors["wastewater"][0], ("inflow",))
    outfalls = node_subset(sectors["wastewater"][0], ("outfall", "treatment", "lift_station", "pump"))
    ax.scatter(sewer_inflows["lon"], sewer_inflows["lat"], s=8, facecolors="white", edgecolors=COLORS["wastewater"], linewidths=0.55, zorder=6)
    ax.scatter(outfalls["lon"], outfalls["lat"], s=28, marker="^", c=COLORS["wastewater"], edgecolors="white", linewidths=0.45, zorder=7)
    panel_title(ax, "d", "Wastewater - directed", "79/79 inflows reach one outfall · acyclic\ngravity/lift links · SWMM pass")

    ax = axes[1, 1]
    draw_base(ax, roads, buildings, bbox, points=True, building_alpha=0.075)
    ax.scatter(zones["lon"], zones["lat"], s=7, c="#C4C8CC", edgecolors="white", linewidths=0.25, zorder=2)
    plants = node_subset(sectors["district_heating"][0], ("plant", "source"))
    heat_consumers = node_subset(sectors["district_heating"][0], ("consumer_substation",))
    heat_components = rooted_component_map(
        sectors["district_heating"][0], sectors["district_heating"][1],
        node_id="node_id", root_types=("plant", "source"),
    )
    heat_palette = ["#B96B00", "#E69F00", "#F2C14E"]
    heat_nodes = sectors["district_heating"][0].copy()
    heat_nodes["component"] = heat_nodes["node_id"].astype(str).map(heat_components)
    heat_links = sectors["district_heating"][1].copy()
    heat_links["component"] = heat_links["from_node"].astype(str).map(heat_components)
    for component, color in enumerate(heat_palette):
        draw_network(ax, heat_links[heat_links["component"].eq(component)], color, width=1.38)
        consumers = heat_nodes[
            heat_nodes["component"].eq(component)
            & heat_nodes["node_type"].eq("consumer_substation")
        ]
        component_plants = heat_nodes[
            heat_nodes["component"].eq(component)
            & heat_nodes["node_type"].isin(("plant", "source"))
        ]
        ax.scatter(
            consumers["lon"], consumers["lat"], s=13, facecolors="white",
            edgecolors=color, linewidths=0.78, zorder=6,
        )
        ax.scatter(
            component_plants["lon"], component_plants["lat"], s=48, marker="*",
            c=color, edgecolors="white", linewidths=0.55, zorder=7,
        )
        for row in component_plants.itertuples(index=False):
            ax.annotate(
                f"H{component + 1}", (row.lon, row.lat), xytext=(3, 3),
                textcoords="offset points", fontsize=5.4, fontweight="bold",
                color=color, zorder=8,
            )
    panel_title(
        ax, "e", "Heat - three local systems",
        "H1-H3 are separate source-centred systems\n20/79 zones · 32 route links · pandapipes pass",
    )

    ax = axes[1, 0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B7BDC4")
    release = FancyBboxPatch(
        (0.035, 0.205), 0.250, 0.555,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        fc="white", ec=COLORS["green"], lw=1.1,
    )
    ax.add_patch(release)
    ax.text(0.160, 0.665, "RELEASE", ha="center", va="center", fontsize=6.2, fontweight="bold", color=COLORS["green"])
    ax.text(0.160, 0.520, "83", ha="center", va="center", fontsize=10.0, fontweight="bold", color=COLORS["green"])
    ax.text(0.160, 0.430, "typed\ninterfaces", ha="center", va="center", fontsize=5.05, color=COLORS["green"], linespacing=1.05)
    ax.text(0.160, 0.315, "CSV + native\nsolver models", ha="center", va="center", fontsize=5.35, linespacing=1.16)

    ax.annotate(
        "", xy=(0.298, 0.485), xytext=(0.370, 0.485),
        arrowprops={"arrowstyle": "-|>", "lw": 1.05, "color": COLORS["green"]},
    )
    gate = FancyBboxPatch(
        (0.380, 0.205), 0.240, 0.555,
        boxstyle="round,pad=0.014,rounding_size=0.026",
        fc="#EFF7F2", ec=COLORS["green"], lw=1.2,
    )
    ax.add_patch(gate)
    ax.text(0.500, 0.665, "ACCEPT", ha="center", va="center", fontsize=6.25, fontweight="bold", color=COLORS["green"])
    ax.text(0.500, 0.520, "PASS", ha="center", va="center", fontsize=10.0, fontweight="bold", color=COLORS["green"])
    ax.text(0.500, 0.350, "topology\ncoverage\nsolver limits", ha="center", va="center", fontsize=5.25, linespacing=1.10)

    ax.annotate(
        "", xy=(0.635, 0.485), xytext=(0.685, 0.485),
        arrowprops={"arrowstyle": "-|>", "lw": 1.05, "color": COLORS["green"]},
    )
    native_frame = FancyBboxPatch(
        (0.695, 0.205), 0.270, 0.555,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        fc="#F7F8F8", ec="#AEB5BA", lw=0.9,
    )
    ax.add_patch(native_frame)
    ax.text(0.830, 0.690, "4 NATIVE MODELS", ha="center", va="center", fontsize=5.85, fontweight="bold")
    native_tiles = [
        ("E", COLORS["electricity"], 0.718, 0.485),
        ("W", COLORS["drinking_water"], 0.842, 0.485),
        ("S", COLORS["wastewater"], 0.718, 0.285),
        ("H", COLORS["district_heating"], 0.842, 0.285),
    ]
    for symbol, color, x, y in native_tiles:
        tile = FancyBboxPatch(
            (x, y), 0.105, 0.145,
            boxstyle="round,pad=0.006,rounding_size=0.016",
            fc="white", ec=color, lw=1.0,
        )
        ax.add_patch(tile)
        ax.text(x + 0.0525, y + 0.0725, symbol, ha="center", va="center", fontsize=8.4, fontweight="bold", color=color)
    panel_title(ax, "f", "Acceptance and export")

    use_legend = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=USE_COLORS[group],
            markeredgecolor="none", markersize=4.2,
            label="Other/unclassified" if group == "Other" else group,
        )
        for group in ("Residential", "Commercial/public", "Industrial", "Auxiliary", "Other")
    ]
    use_legend.append(Line2D([0], [0], marker="o", color=COLORS["shared"], markerfacecolor="white", lw=0, markersize=4.5, label="common zone"))
    fig.legend(handles=use_legend, loc="lower center", ncol=6, fontsize=6.15, frameon=False, bbox_to_anchor=(0.5, 0.006), handletextpad=0.32, columnspacing=0.85)
    add_flow_arrows(fig, axes)
    save(fig, output, "fig03_clusters_to_topologies")


def figure_clusters_to_topologies(data_root: Path, output: Path) -> None:
    """Visualize the accepted building/service-resolved dense case."""
    case = data_root / "outputs" / "benchmark_cases" / "dense_urban"
    buildings = pd.read_csv(case / "common_building_ledger.csv")
    zones = pd.read_csv(case / "common_demand_zones.csv")
    transformers = pd.read_csv(case / "electricity_transformers.csv")
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    roads = road_segments(data_root / "data" / "raw" / "local_roads.json")
    electric_nodes = pd.read_csv(case / "electricity_nodes.csv")
    electric_links = pd.read_csv(case / "electricity_links.csv")
    water_nodes = pd.read_csv(case / "drinking_water_nodes.csv")
    water_links = pd.read_csv(case / "drinking_water_links.csv")
    sewer_nodes = pd.read_csv(case / "wastewater_nodes.csv")
    sewer_links = pd.read_csv(case / "wastewater_links.csv")
    heat_nodes = pd.read_csv(case / "district_heating_nodes.csv")
    heat_links = pd.read_csv(case / "district_heating_corridors.csv")
    bbox = [10.215, 50.037, 10.249, 50.060]
    # Map rows and statistics rows are separate.  This keeps every physical
    # topology visible and prevents text boxes from hiding the southern part of
    # each network.  The four sector panels are parallel branches; no arrows
    # are drawn between them because one sector is not generated from another.
    fig = plt.figure(figsize=(7.16, 5.08))
    grid = fig.add_gridspec(
        4, 3,
        height_ratios=(1.0, 0.180, 1.0, 0.145),
        left=0.026, right=0.992, bottom=0.085, top=0.962,
        wspace=0.085, hspace=0.135,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, col]) for col in range(3)],
            [fig.add_subplot(grid[2, col]) for col in range(3)],
        ],
        dtype=object,
    )
    strips = np.asarray(
        [
            [fig.add_subplot(grid[1, col]) for col in range(3)],
            [fig.add_subplot(grid[3, col]) for col in range(3)],
        ],
        dtype=object,
    )

    ax = axes[0, 0]
    draw_base(ax, roads, buildings, bbox, points=False)
    typed = buildings.copy()
    typed["use_group"] = typed["service_class"].map(service_group)
    for group in ["Other", "Auxiliary", "Residential", "Commercial/public", "Industrial"]:
        part = typed[typed.use_group.eq(group)]
        size = np.clip(1.2 + 0.095 * np.sqrt(part.floor_area_m2.to_numpy()), 1.2, 8.5)
        ax.scatter(
            part.lon, part.lat, s=size, c=USE_COLORS[group],
            alpha=0.58 if group == "Other" else 0.78,
            edgecolors="none", rasterized=True, zorder=2,
        )
    ax.scatter(
        zones.lon, zones.lat, s=16, facecolors="white",
        edgecolors=COLORS["shared"], linewidths=0.8, zorder=7,
    )
    panel_title(ax, "a", "Common buildings and zones")
    statistics_strip(
        strips[0, 0],
        f"{len(buildings):,} buildings | {len(zones)} shared zones\n"
        "Colour: service class | size: floor-area prior",
    )

    ax = axes[0, 1]
    draw_base(ax, roads, buildings, bbox, points=True, building_alpha=0.06)
    draw_network(
        ax, electric_links[electric_links.asset_type.eq("building_service")],
        "#F2B2AA", width=0.20,
    )
    draw_network(
        ax, electric_links[electric_links.asset_type.eq("lv_feeder")],
        COLORS["electricity"], width=0.58,
    )
    draw_network(
        ax, electric_links[electric_links.voltage_level.eq("MV")],
        COLORS["electricity_mv"], dashed=True, width=1.25,
    )
    electric_services = electric_nodes[
        electric_nodes.node_type.isin(["building_service", "mv_customer"])
    ]
    ax.scatter(
        electric_services.lon, electric_services.lat, s=1.5,
        c=COLORS["electricity"], alpha=0.45, edgecolors="none",
        rasterized=True, zorder=5,
    )
    ax.scatter(
        transformers.lon, transformers.lat, s=17, marker="s",
        c=COLORS["electricity"], edgecolors="white", linewidths=0.4, zorder=7,
    )
    source = electric_nodes[electric_nodes.node_type.eq("mv_source")]
    ax.scatter(source.lon, source.lat, s=42, marker="*", c=COLORS["electricity_mv"], edgecolors="white", linewidths=0.5, zorder=8)
    power_state = manifest["solver_readiness"]["pandapower"]
    panel_title(ax, "b", "Electricity: MV + radial LV")
    statistics_strip(
        strips[0, 1],
        f"{len(transformers)} transformers | {len(electric_services):,} service buses\n"
        rf"Connected MV/LV graph | $V_{{\min}}$ {power_state['minimum_voltage_pu']:.3f} p.u.",
    )

    ax = axes[0, 2]
    draw_base(ax, roads, buildings, bbox, points=True, building_alpha=0.06)
    draw_network(ax, water_links[water_links.link_type.eq("building_service")], "#A9C9EF", width=0.10)
    water_mains = water_links[water_links.link_type.eq("distribution_main")].copy()
    water_graph = nx.Graph()
    for row in water_mains.itertuples():
        water_graph.add_edge(str(row.from_node), str(row.to_node), weight=float(row.length_km))
    water_tree = nx.minimum_spanning_tree(water_graph, weight="weight")
    tree_edges = {frozenset((u, v)) for u, v in water_tree.edges()}
    water_mains["is_loop_chord"] = [
        frozenset((str(row.from_node), str(row.to_node))) not in tree_edges
        for row in water_mains.itertuples()
    ]
    draw_network(ax, water_mains[~water_mains.is_loop_chord], COLORS["drinking_water"], width=0.64)
    draw_network(ax, water_mains[water_mains.is_loop_chord], COLORS["shared"], width=1.45)
    water_services = water_nodes[water_nodes.node_type.eq("building_service")]
    water_sources = water_nodes[water_nodes.node_type.isin(["reservoir", "pump_outlet"])]
    ax.scatter(water_services.lon, water_services.lat, s=1.4, c=COLORS["drinking_water"], alpha=0.42, edgecolors="none", rasterized=True, zorder=5)
    ax.scatter(water_sources.lon, water_sources.lat, s=30, marker="*", c=COLORS["drinking_water"], edgecolors="white", linewidths=0.5, zorder=7)
    water_state = manifest["solver_readiness"]["epanet_wntr"]
    panel_title(ax, "c", "Water: looped mains")
    statistics_strip(
        strips[0, 2],
        f"{len(water_services):,} laterals | {int(water_mains.is_loop_chord.sum())} loop chords highlighted\n"
        rf"Peak $p_{{\min}}$ {water_state['peak_minimum_pressure_m']:.1f} m | EPANET pass",
    )

    ax = axes[1, 2]
    draw_base(ax, roads, buildings, bbox, points=True, building_alpha=0.06)
    draw_network(ax, sewer_links[sewer_links.link_type.eq("property_lateral")], "#AAB0B8", width=0.10)
    gravity_mains = sewer_links[sewer_links.link_type.eq("gravity_main")].copy()
    draw_network(ax, gravity_mains, COLORS["wastewater"], dashed=False, width=0.66)
    # Sparse flow arrows make the directed collection logic visible without
    # obscuring the road-constrained geometry.
    arrow_sample = gravity_mains.sort_values("length_km", ascending=False).iloc[::max(1, len(gravity_mains)//65)].head(65)
    qx, qy, qu, qv = [], [], [], []
    for value in arrow_sample.geometry_json:
        points = json.loads(value)
        if len(points) < 2:
            continue
        a = points[max(0, len(points)//2 - 1)]
        b = points[min(len(points)-1, len(points)//2)]
        dx, dy = b[0]-a[0], b[1]-a[1]
        norm = math.hypot(dx, dy)
        if norm <= 0:
            continue
        scale = min(norm, 0.00032) / norm
        qx.append(a[0]); qy.append(a[1]); qu.append(dx*scale); qv.append(dy*scale)
    if qx:
        ax.quiver(qx, qy, qu, qv, angles="xy", scale_units="xy", scale=1,
                  color=COLORS["wastewater"], width=0.0045, headwidth=3.5,
                  headlength=4.2, headaxislength=3.5, zorder=8)
    force = sewer_links[sewer_links.link_type.str.contains("force", na=False)]
    if not force.empty:
        draw_network(ax, force, COLORS["shared"], width=1.15)
    sewer_services = sewer_nodes[
        sewer_nodes.node_type.isin(["building_property_connection", "property_lift_station"])
    ]
    sewer_facilities = sewer_nodes[sewer_nodes.node_type.isin(["outfall", "lift_station"])]
    ax.scatter(sewer_services.lon, sewer_services.lat, s=1.4, c=COLORS["wastewater"], alpha=0.40, edgecolors="none", rasterized=True, zorder=5)
    ax.scatter(sewer_facilities.lon, sewer_facilities.lat, s=29, marker="^", c=COLORS["wastewater"], edgecolors="white", linewidths=0.45, zorder=7)
    sewer_state = manifest["solver_readiness"]["swmm"]
    panel_title(ax, "d", "Wastewater: directed tree")
    statistics_strip(
        strips[1, 2],
        f"{len(sewer_services):,} inflows | arrows show outfall direction\n"
        f"Continuity error {sewer_state['flow_routing_continuity_error_percent']:.3f}% | no flooding",
    )

    ax = axes[1, 1]
    draw_base(ax, roads, buildings, bbox, points=True, building_alpha=0.06)
    draw_network(ax, heat_links[heat_links.link_type.eq("building_service")], "#F4D292", width=0.19)
    draw_network(ax, heat_links[heat_links.link_type.eq("route")], COLORS["district_heating"], width=0.80)
    plants = heat_nodes[heat_nodes.node_type.eq("plant")]
    heat_services = heat_nodes[heat_nodes.node_type.eq("consumer_substation")]
    ax.scatter(heat_services.lon, heat_services.lat, s=1.7, c=COLORS["district_heating"], alpha=0.52, edgecolors="none", rasterized=True, zorder=5)
    ax.scatter(plants.lon, plants.lat, s=45, marker="*", c=COLORS["district_heating"], edgecolors="white", linewidths=0.55, zorder=7)
    heat_state = manifest["solver_readiness"]["pandapipes_heat"]
    panel_title(ax, "e", "Heat: source-rooted routes")
    statistics_strip(
        strips[1, 1],
        f"{len(plants)} heat-source boundaries | {len(heat_services):,} substations\n"
        "Supply/return topology exported | pandapipes hydraulic screen",
    )

    ax = axes[1, 0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B7BDC4")
    native = [
        ("pandapower", COLORS["electricity"]),
        ("EPANET/WNTR", COLORS["drinking_water"]),
        ("SWMM", COLORS["wastewater"]),
        ("pandapipes", COLORS["district_heating"]),
    ]
    # A vertical join is more legible than three adjacent prose-heavy boxes at
    # this panel size: native solvers -> common gate -> reproducible exports.
    positions = [(0.055, 0.725), (0.525, 0.725), (0.055, 0.575), (0.525, 0.575)]
    for (label, color), (x, y) in zip(native, positions):
        tile = FancyBboxPatch(
            (x, y), 0.420, 0.105,
            boxstyle="round,pad=0.008,rounding_size=0.016",
            fc="white", ec=color, lw=1.0,
        )
        ax.add_patch(tile)
        ax.text(x + 0.210, y + 0.052, label, ha="center", va="center", fontsize=6.35, fontweight="bold", color=color)

    # Join the four native-solver branches at one acceptance gate.
    for x in (0.265, 0.735):
        ax.plot([x, x], [0.565, 0.515], color="#77818A", lw=0.8, clip_on=False)
    ax.plot([0.265, 0.735], [0.515, 0.515], color="#77818A", lw=0.8)
    ax.annotate("", xy=(0.500, 0.455), xytext=(0.500, 0.515), arrowprops={"arrowstyle": "-|>", "lw": 0.95, "color": "#77818A"})
    gate = FancyBboxPatch(
        (0.185, 0.285), 0.630, 0.165,
        boxstyle="round,pad=0.012,rounding_size=0.024",
        fc="#EFF7F2", ec=COLORS["green"], lw=1.2,
    )
    ax.add_patch(gate)
    ax.text(0.500, 0.378, "4/4 NATIVE MODELS PASS", ha="center", va="center", fontsize=7.55, fontweight="bold", color=COLORS["green"])
    ax.text(0.500, 0.322, "topology and native-solver screens", ha="center", va="center", fontsize=5.95)
    ax.annotate("", xy=(0.500, 0.235), xytext=(0.500, 0.285), arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": COLORS["green"]})
    release = FancyBboxPatch(
        (0.120, 0.075), 0.760, 0.155,
        boxstyle="round,pad=0.012,rounding_size=0.024",
        fc="white", ec=COLORS["green"], lw=1.1,
    )
    ax.add_patch(release)
    ax.text(0.500, 0.176, "EXPORT", ha="center", va="center", fontsize=6.15, fontweight="bold", color=COLORS["green"])
    ax.text(0.500, 0.124, "CSV + native models + stable IDs", ha="center", va="center", fontsize=6.25)
    panel_title(ax, "f", "Acceptance and export")
    statistics_strip(strips[1, 0], "Only accepted topologies enter Phase III coupling")

    legend = [
        Line2D([0], [0], color=COLORS["electricity_mv"], lw=1.3, ls="--", label="MV"),
        Line2D([0], [0], color=COLORS["electricity"], lw=1.2, label="LV/electric"),
        Line2D([0], [0], color=COLORS["drinking_water"], lw=1.2, label="water"),
        Line2D([0], [0], color=COLORS["wastewater"], lw=1.2, ls="--", label="wastewater"),
        Line2D([0], [0], color=COLORS["district_heating"], lw=1.2, label="district heat"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["shared"], markeredgecolor="white", markersize=4.5, label="common zone"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=6.6, frameon=False, bbox_to_anchor=(0.5, 0.006), handletextpad=0.35, columnspacing=0.85)
    save(fig, output, "fig03_clusters_to_topologies")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    figure_evidence_to_clusters(args.data_root.resolve(), args.output.resolve())
    figure_clusters_to_topologies(args.data_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
