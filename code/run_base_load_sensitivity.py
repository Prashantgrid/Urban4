#!/usr/bin/env python3
"""Transformer screening from the retained design-state table; no AC power flow.

Reconstruct the published sum check before scaling existing aggregate demand.
Both the reference demand and added conversion demand have power factor 0.96.
Circulation pumps in the existing-DH case attach at MV, outside LV transformers.
Internal scenario/ratio IDs are retained to match the original case-study files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/heat_power_case")
    args = parser.parse_args()
    root = args.results
    tr = pd.read_csv(root / "transformer_design_state.csv")
    old = pd.read_csv(root / "heat_power_variants.csv")
    summary = json.loads((root / "heat_power_summary.json").read_text())
    td = float(summary["assumptions"]["design_outdoor_temperature_c"])
    lift = max(15.0, 40.0 - 2.0 * td)
    cop = 0.85 * (6.08 - 0.0941 * lift + 0.000464 * lift**2)
    pf = 0.96
    base = tr.design_loading_pct_S0_DH.to_numpy()
    heat = tr.dh_peak_mw.fillna(0).to_numpy()
    sn = tr.sn_mva.to_numpy()
    added_reference = heat / cop / pf / sn * 100.0
    error = float(np.max(np.abs(base + added_reference - tr.design_loading_pct_S1_HP)))
    if error > 1e-9:
        raise ValueError(f"Reference reconstruction differs by {error} percentage points")

    rows, details = [], []
    for mode, ratio in [("cop", 1.0 / cop), ("measured", 0.40)]:
        for renovation in [1.0, 0.8, 0.6]:
            added = heat * ratio * renovation / pf / sn * 100.0
            reference = old[(old.scenario == "S1_HP") & (old.ratio_mode == mode)
                            & np.isclose(old.renovation_f, renovation)]
            if len(reference) != 1:
                raise ValueError("Expected one retained heat-pump variant")
            for gamma in [1.0, 0.8, 0.6]:
                loading = gamma * base + added
                count = int((loading > 100.0 + 1e-9).sum())
                if gamma == 1.0 and count != int(reference.iloc[0].overloaded_areas):
                    raise ValueError("Published reference overload count was not reproduced")
                rows.append({"scenario": "S1_HP", "scenario_name": "Heat-pump conversion",
                             "ratio_mode": mode,
                             "ratio_name": "COP model" if mode == "cop" else "Study-based ratio 0.40",
                             "renovation_f": renovation, "base_load_factor": gamma,
                             "reference_lv_peak_mw": 41.0, "existing_lv_demand_mw": 41.0 * gamma,
                             "added_heat_pump_mw": float((heat * ratio * renovation).sum()),
                             "overloaded_areas": count,
                             "max_transformer_loading_pct": float(loading.max()),
                             "evaluation": "transformer sum check; existing spatial allocation and ratings fixed"})
                detail = tr[["transformer_id", "sn_mva", "dh_buildings", "dh_peak_mw"]].copy()
                detail["ratio_mode"] = mode
                detail["renovation_f"] = renovation
                detail["base_load_factor"] = gamma
                detail["loading_pct"] = loading
                detail["overloaded"] = loading > 100.0 + 1e-9
                details.append(detail)
    pd.DataFrame(rows).to_csv(root / "base_load_sensitivity.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(root / "base_load_sensitivity_by_transformer.csv", index=False)
    (root / "base_load_sensitivity_summary.json").write_text(json.dumps({
        "design_temperature_c": td, "power_factor": pf, "cop_design": cop,
        "reference_max_error_percentage_points": error,
        "published_variants_reproduced": 6,
        "scope": "Deterministic transformer apparent-power check; no cable or AC-voltage recalculation",
        "formula": "loading_pct = gamma * reference_loading_pct + 100 * f * r_design * heat_peak_mw / (0.96 * rating_mva)",
        "variants": rows}, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index=["ratio_name", "renovation_f"],
                                   columns="base_load_factor", values="overloaded_areas").to_string())
    print(f"Reference reconstruction error: {error:.3g} percentage points")


if __name__ == "__main__":
    main()
