#!/usr/bin/env python3
"""Generate the non-redundant Urban4 transferability map.

The dense Innenstadt case is already shown sector by sector in Figure 3.
Figure 4 therefore contains only the independently regenerated semi-urban and
peripheral cases.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from make_figures import (
    BENCHMARK_CASES,
    COLORS,
    MORPH_COLORS,
    _draw_base,
    _draw_facilities,
    _draw_networks,
    _load_benchmark_case,
    _north_and_scale,
)


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "figures"
STEM = "fig04_transfer_contexts"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
        "font.size": 11.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def main() -> None:
    cases = [item for item in BENCHMARK_CASES if item["id"] != "dense_urban"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.3))

    for letter, ax, item in zip(("a", "b"), axes, cases):
        frames = _load_benchmark_case(item["id"])
        _draw_base(ax, frames["buildings"], item["bbox"], building_alpha=0.27)
        _draw_networks(ax, frames)
        _draw_facilities(ax, frames)
        ax.set_title(
            f"({letter}) {item['place_name']} -- {item['morphology_label']}",
            loc="left",
            color=MORPH_COLORS[item["morphology"]],
            fontweight="bold",
            fontsize=12.0,
            pad=5,
        )
        _north_and_scale(ax, 1.0)

    handles = [
        Line2D([0], [0], color=COLORS["electricity"], lw=2.2),
        Line2D([0], [0], color="#7E2948", lw=2.0, ls="--"),
        Line2D([0], [0], color=COLORS["drinking_water"], lw=2.2),
        Line2D([0], [0], color=COLORS["wastewater"], lw=2.0, ls=(0, (2.3, 1.4))),
        Line2D([0], [0], color=COLORS["district_heating"], lw=2.2),
    ]
    labels = ["LV/electric", "MV backbone", "Drinking water", "Wastewater", "District heating"]
    fig.legend(
        handles,
        labels,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        columnspacing=1.1,
        handlelength=1.7,
        fontsize=9.8,
        frameon=False,
    )
    fig.subplots_adjust(left=0.015, right=0.995, top=0.94, bottom=0.105, wspace=0.055)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{STEM}.pdf", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT / f"{STEM}.svg", bbox_inches="tight", pad_inches=0.035)
    fig.savefig(OUT / f"{STEM}.png", dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


if __name__ == "__main__":
    main()
