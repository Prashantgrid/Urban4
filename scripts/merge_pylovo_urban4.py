#!/usr/bin/env python3
"""Command-line bridge from pylovo LV JSON files to a connected Urban4 grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban4 import schweinfurt_base as base
from urban4.pylovo_bridge import merge_pylovo_grids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_dir", type=Path, help="directory containing pylovo pandapower JSON grids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peak-mw", type=float, required=True, help="official coincident LV peak")
    parser.add_argument("--roads", default="local_roads.json", help="retained road snapshot name")
    args = parser.parse_args()
    road = base.build_road_graph(base.load_elements(args.roads))
    paths = sorted(args.grid_dir.glob("*.json"))
    _, _, summary = merge_pylovo_grids(paths, road, args.output, coincident_peak_mw=args.peak_mw)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
