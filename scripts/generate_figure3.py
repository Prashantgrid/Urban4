#!/usr/bin/env python3
"""Generate the Urban4 clustering-to-topology audit figure only.

This entry point reads existing accepted CSV/native-model exports.  It neither
regenerates topologies nor reruns any infrastructure solver.  The plotting
implementation is kept in ``make_process_figures.py`` so manuscript and
standalone renders use exactly the same code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from make_process_figures import figure_clusters_to_topologies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Urban4 checkout containing outputs/ and data/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory receiving fig03_clusters_to_topologies.pdf and .png",
    )
    args = parser.parse_args()
    figure_clusters_to_topologies(args.data_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
