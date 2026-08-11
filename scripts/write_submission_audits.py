#!/usr/bin/env python3
"""Write the authoritative route inventory and compact robustness summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
INTEGRATED = OUT / "integrated_service_resolved"


def _route_rows() -> list[dict[str, object]]:
    definitions = [
        ("electricity", "electricity_links.csv", "asset_type", {
            "mv_backbone": "MV backbone",
            "lv_feeder": "LV feeders",
            "building_service": "building service drops",
        }),
        ("drinking_water", "drinking_water_links.csv", "link_type", {
            "distribution_main": "distribution mains",
            "building_service": "building laterals",
        }),
        ("wastewater", "wastewater_links.csv", "link_type", {
            "gravity_main": "gravity mains",
            "force_main": "network force mains",
            "property_lateral": "property laterals",
            "property_force_main": "property force mains",
        }),
        ("district_heating", "district_heating_corridors.csv", "link_type", {
            "route": "trench routes",
            "building_service": "building service connections",
        }),
    ]
    rows: list[dict[str, object]] = []
    for sector, filename, type_column, labels in definitions:
        frame = pd.read_csv(INTEGRATED / filename)
        for link_type, group in frame.groupby(type_column):
            rows.append({
                "run_id": "urban4-submission-20260806",
                "evidence_resolution": "building/service-resolved",
                "sector": sector,
                "component_code": str(link_type),
                "component_label": labels.get(str(link_type), str(link_type)),
                "element_count": int(len(group)),
                "one_way_route_length_km": float(group.length_km.sum()),
                "comparison_rule": (
                    "compare with published route/trench inventory"
                    if link_type in {"mv_backbone", "lv_feeder", "distribution_main", "gravity_main", "force_main", "route"}
                    else "report separately; exclude unless reference explicitly includes services"
                ),
            })
    return rows


def _robustness_summary() -> pd.DataFrame:
    frame = pd.read_csv(OUT / "clustering_seed_sensitivity.csv")
    rows = []
    metrics = [
        "realized_zones", "buildings_per_zone_median",
        "electricity_peak_zone_cv", "water_zone_cv",
        "nearest_zone_spacing_median_km", "nearest_zone_spacing_p95_km",
    ]
    for case_id, group in frame.groupby("case_id"):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            rows.append({
                "case_id": case_id,
                "stochastic_stage": "common-zone initialization",
                "metric": metric,
                "seeds": len(values),
                "median": float(np.median(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
                "coefficient_of_variation": float(np.std(values) / max(abs(np.mean(values)), 1e-12)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    inventory = pd.DataFrame(_route_rows())
    inventory.to_csv(OUT / "authoritative_route_inventory.csv", index=False)
    robustness = _robustness_summary()
    robustness.to_csv(OUT / "clustering_robustness_summary.csv", index=False)
    ablation = pd.read_csv(OUT / "common_ledger_ablation.csv")
    audit = {
        "run_id": "urban4-submission-20260806",
        "authoritative_topology_directory": str(INTEGRATED.relative_to(PROJECT)),
        "route_inventory": "outputs/authoritative_route_inventory.csv",
        "route_definition": "one-way physical route by component; services are separate",
        "paired_heat_pipe_rule": "supply plus return physical pipe length equals twice the heat trench route plus twice any paired service connection",
        "clustering_robustness": "outputs/clustering_robustness_summary.csv",
        "clustering_scope": "20 initializations of the stochastic common-zone stage; not presented as 20 full native-solver reruns",
        "ablation_scope": "spatial zone-interface alignment only; immutable building identity is preserved by the shared ledger and is not claimed to be statistically inferred",
        "ablation_cases": len(ablation),
    }
    (OUT / "submission_evidence_manifest.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
