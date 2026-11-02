#!/usr/bin/env python3
"""Transformer-capacity sensitivity for heat-pump conversion; no AC power flow.

Generated transformers are sized to the current coincident peak (98 MVA in
total), whereas the operator reports 217 MVA of installed MV/LV capacity for
its whole service area. This script scales every transformer rating by a
common factor and repeats the apparent-power sum check of
run_base_load_sensitivity.py at -14 degC:

  k = 1.30  published capacity scaled by the retained building share
            (217 MVA x 22,641 / 38,500 = 128 MVA, divided by 98 MVA)
  k = 2.22  full published capacity (217 MVA / 98 MVA)

Spatial allocation, existing demand (100 %), and added heat-pump demand are
unchanged. Both demands use power factor 0.96, as in the main sum check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PUBLISHED_MVA = 217.0
RETAINED_BUILDINGS = 22641
OSM_BUILDINGS = 38500


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/heat_power_case")
    args = parser.parse_args()
    root = args.results
    tr = pd.read_csv(root / "transformer_design_state.csv")
    summary = json.loads((root / "heat_power_summary.json").read_text())
    td = float(summary["assumptions"]["design_outdoor_temperature_c"])
    lift = max(15.0, 40.0 - 2.0 * td)
    cop = 0.85 * (6.08 - 0.0941 * lift + 0.000464 * lift**2)
    pf = 0.96

    base = tr.design_loading_pct_S0_DH.to_numpy()
    heat = tr.dh_peak_mw.fillna(0).to_numpy()
    sn = tr.sn_mva.to_numpy()
    generated_mva = float(sn.sum())
    factors = {
        "generated": 1.0,
        "published_per_retained_building": PUBLISHED_MVA * RETAINED_BUILDINGS / OSM_BUILDINGS / generated_mva,
        "published_total": PUBLISHED_MVA / generated_mva,
    }

    # Reproduce the published reference count before scaling.
    ref = base + heat / cop / pf / sn * 100.0
    if int((ref > 100.0 + 1e-9).sum()) != 19:
        raise ValueError("Reference overload count (19 areas) was not reproduced")

    rows = []
    for mode, ratio, name in [("cop", 1.0 / cop, "COP model"), ("measured", 0.40, "Study-based ratio 0.40")]:
        for renovation in [1.0, 0.8, 0.6]:
            added = heat * ratio * renovation / pf / sn * 100.0
            for label, k in factors.items():
                loading = (base + added) / k
                rows.append({"ratio_mode": mode, "ratio_name": name, "renovation_f": renovation,
                             "capacity_case": label, "rating_factor": round(k, 4),
                             "total_rating_mva": round(generated_mva * k, 1),
                             "overloaded_areas": int((loading > 100.0 + 1e-9).sum()),
                             "max_transformer_loading_pct": float(loading.max())})
    out = pd.DataFrame(rows)
    out.to_csv(root / "transformer_capacity_sensitivity.csv", index=False)
    print(out.pivot_table(index=["ratio_name", "renovation_f"], columns="capacity_case",
                          values="overloaded_areas").to_string())


if __name__ == "__main__":
    main()
