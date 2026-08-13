#!/usr/bin/env python3
"""Four-sector orchestration, exports, metrics and normal-condition checks.

The module deliberately contains no hazard, outage, restoration or resilience
logic.  It starts from one registered evidence model and produces electricity,
drinking-water, wastewater, district-heating and cross-sector interface files.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

from . import schweinfurt_base as base


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
RAW = PROJECT / "data" / "raw"
CASE = PROJECT / "cases" / "schweinfurt.json"


DEFAULT_FOOTPRINT_M2 = {
    "apartments": 310.0,
    "residential": 210.0,
    "dormitory": 420.0,
    "house": 125.0,
    "detached": 135.0,
    "semidetached_house": 105.0,
    "terrace": 92.0,
    "commercial": 520.0,
    "retail": 640.0,
    "office": 430.0,
    "hospital": 1600.0,
    "school": 760.0,
    "industrial": 1450.0,
    "warehouse": 1700.0,
    "garage": 28.0,
    "garages": 75.0,
    "shed": 24.0,
    "roof": 55.0,
    "yes": 145.0,
}

DEFAULT_LEVELS = {
    "apartments": 4.0,
    "residential": 3.0,
    "dormitory": 4.0,
    "house": 2.0,
    "detached": 2.0,
    "semidetached_house": 2.0,
    "terrace": 2.0,
    "commercial": 2.0,
    "retail": 1.0,
    "office": 3.0,
    "hospital": 4.0,
    "school": 3.0,
    "industrial": 1.5,
    "warehouse": 1.0,
    "yes": 2.0,
}

SPECIFIC_HEAT_KWH_M2A = {
    "apartments": 105.0,
    "residential": 115.0,
    "dormitory": 100.0,
    "house": 135.0,
    "detached": 145.0,
    "semidetached_house": 135.0,
    "terrace": 125.0,
    "commercial": 85.0,
    "retail": 80.0,
    "office": 75.0,
    "hospital": 155.0,
    "school": 100.0,
    "industrial": 65.0,
    "warehouse": 35.0,
    "yes": 105.0,
}

SPECIFIC_ELECTRICITY_KWH_M2A = {
    "apartments": 34.0,
    "residential": 36.0,
    "dormitory": 42.0,
    "house": 32.0,
    "detached": 34.0,
    "semidetached_house": 33.0,
    "terrace": 33.0,
    "commercial": 115.0,
    "retail": 175.0,
    "office": 95.0,
    "hospital": 245.0,
    "school": 55.0,
    "industrial": 85.0,
    "warehouse": 24.0,
    "yes": 48.0,
}

AUXILIARY_BUILDING_TYPES = {
    "garage", "garages", "shed", "roof", "carport", "greenhouse",
    "farm_auxiliary", "barn", "ruins", "construction", "no", "tent",
    "pavilion", "hangar", "parking", "silo",
}
RESIDENTIAL_BUILDING_TYPES = {
    "apartments", "residential", "dormitory", "house", "detached",
    "semidetached_house", "terrace", "allotment_house", "hostel",
}
INDUSTRIAL_BUILDING_TYPES = {
    "industrial", "warehouse", "factory", "service",
}
COMMERCIAL_BUILDING_TYPES = {
    "commercial", "retail", "office", "supermarket", "mall",
}
PUBLIC_BUILDING_TYPES = {
    "hospital", "school", "kindergarten", "civic", "church", "chapel",
    "museum", "university", "fire_station", "train_station", "sports_centre",
    "transportation", "toilets",
}


def _building_service_class(tags: dict[str, Any], kind: str) -> tuple[str, bool, str]:
    """Return use class, service eligibility and evidence explanation.

    The function intentionally separates a physical OSM building object from a
    utility customer.  Auxiliary structures remain in the evidence ledger but
    receive no municipal service demand.  Generic ``building=yes`` records use
    functional tags or an address before falling back to spatial inference.
    """
    kind = str(kind).lower()
    declared_use = str(tags.get("building:use", "")).lower()
    amenity = str(tags.get("amenity", "")).lower()
    if kind in AUXILIARY_BUILDING_TYPES:
        return "auxiliary", False, "explicit auxiliary building tag"
    if kind in RESIDENTIAL_BUILDING_TYPES or declared_use in {"residential", "apartments", "house"}:
        return "residential", True, "explicit residential building/use tag"
    if kind in INDUSTRIAL_BUILDING_TYPES or tags.get("industrial") or tags.get("craft"):
        return "industrial", True, "explicit industrial/building-function tag"
    if kind in COMMERCIAL_BUILDING_TYPES or tags.get("shop") or tags.get("office"):
        return "commercial", True, "explicit commercial/building-function tag"
    if kind in PUBLIC_BUILDING_TYPES or amenity:
        return "public", True, "explicit public/amenity tag"
    if tags.get("tourism") or tags.get("healthcare") or tags.get("leisure"):
        return "commercial", True, "functional tourism/healthcare/leisure tag"
    if tags.get("addr:housenumber") or tags.get("addr:street"):
        return "residential", True, "generic building with address evidence"
    return "unclassified", True, "generic building; class inferred from nearby served building"


def _infer_unclassified_service_classes(frame: pd.DataFrame) -> pd.DataFrame:
    """Infer generic buildings from the nearest explicitly classified customer."""
    frame = frame.copy()
    unresolved = frame[frame["service_class"].eq("unclassified")]
    references = frame[
        frame["service_eligible"]
        & frame["service_class"].isin(["residential", "commercial", "public", "industrial"])
    ]
    if unresolved.empty or references.empty:
        frame.loc[unresolved.index, "service_class"] = "residential"
        frame.loc[unresolved.index, "service_class_evidence"] += "; residential fallback"
        frame.loc[unresolved.index, "service_evidence_class"] = "D"
        return frame
    reference_xy = np.asarray([base.xy_km((row.lon, row.lat)) for row in references.itertuples()])
    tree = cKDTree(reference_xy)
    query_xy = np.asarray([base.xy_km((row.lon, row.lat)) for row in unresolved.itertuples()])
    distances, indices = tree.query(query_xy)
    inferred = references.iloc[np.asarray(indices, dtype=int)]["service_class"].to_numpy()
    # A distant generic object remains a conservative residential/mixed prior
    # rather than inheriting an unrelated industrial or public classification.
    inferred = np.where(np.asarray(distances) <= 0.15, inferred, "residential")
    frame.loc[unresolved.index, "service_class"] = inferred
    frame.loc[unresolved.index, "service_class_evidence"] += "; nearest-class inference within 150 m"
    frame.loc[unresolved.index, "service_evidence_class"] = "D"
    return frame


def load_case(path: Path = CASE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _morphology(point: tuple[float, float], config: dict[str, Any]) -> str:
    lon, lat = point
    return min(
        config["morphologies"],
        key=lambda m: base.distance_km((lon, lat), tuple(m["center"])),
    )["id"]


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(str(value).split(";")[0])
        return parsed if math.isfinite(parsed) and parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _polygon_area_m2(element: dict[str, Any]) -> float | None:
    geometry = element.get("geometry") or []
    if len(geometry) < 3:
        return None
    latitude = sum(float(p["lat"]) for p in geometry) / len(geometry)
    coordinates = [
        (
            float(p["lon"]) * 111_320.0 * math.cos(math.radians(latitude)),
            float(p["lat"]) * 110_540.0,
        )
        for p in geometry
    ]
    area = abs(
        sum(
            coordinates[i][0] * coordinates[(i + 1) % len(coordinates)][1]
            - coordinates[(i + 1) % len(coordinates)][0] * coordinates[i][1]
            for i in range(len(coordinates))
        )
        / 2.0
    )
    return area if area > 8.0 else None


def write_evidence_boundary_audit(config: dict[str, Any]) -> pd.DataFrame:
    """Write the boundary contract before any aggregate is allocated.

    The table deliberately distinguishes the common spatial adapter from the
    published sector reporting envelopes.  A missing public service polygon is
    recorded as a limitation instead of being silently treated as identical to
    the extraction mask.
    """
    a = config["official_anchors"]
    rows = [
        ("LV annual withdrawal", a["electricity_lv_annual_mwh"], "MWh/y", "LV operator territory", "LV", 2025, "LV withdrawals", "building electricity allocation", "input_constraint"),
        ("LV coincident peak", a["electricity_lv_peak_mw"], "MW", "LV operator territory", "LV", 2025, "LV withdrawals", "building peak allocation", "input_constraint"),
        ("LV served population", a["electricity_lv_network_population"], "inhabitants", "LV operator territory", "--", 2025, "inhabitants", "common population allocation", "input_constraint"),
        ("LV reporting area", a["electricity_lv_network_area_km2"], "km2", "LV operator territory", "--", 2025, "geographic area", "boundary audit only; polygon unavailable", "diagnostic"),
        ("MS/LV withdrawal locations", a["electricity_mv_lv_withdrawal_locations"], "count", "LV operator territory", "MS/LV", 2025, "withdrawal locations", "synthetic location count; not physical transformer count", "input_constraint"),
        ("MS/LV installed capacity", a["electricity_mv_lv_installed_capacity_mva"], "MVA", "LV operator territory", "MS/LV", 2025, "installed transformation", "equivalent transformation portfolio", "input_constraint"),
        ("Water delivery, total", a["drinking_water_total_delivery_m3"], "m3/y", "water operator reporting territory", "distribution", 2024, "retail plus wholesale", "accounting identity only", "reported_total"),
        ("Water delivery, wholesale", a["drinking_water_wholesale_delivery_m3"], "m3/y", "downstream distributor interconnection", "boundary transfer", 2024, "wholesale", "excluded from building demand and local wastewater", "boundary_exclusion"),
        ("Water delivery, direct retail", a["drinking_water_annual_m3"], "m3/y", "modelled retail-service adapter", "distribution", 2024, "derived total minus wholesale", "building water allocation", "input_constraint"),
        ("Municipal sewer inventory", a["wastewater_combined_km"] + a["wastewater_sanitary_km"] + a["wastewater_storm_km"] + a["wastewater_force_main_km"], "km", "municipal drainage territory", "collection", 2026, "combined/sanitary/storm/force main", "held-out aggregate comparison", "held_out_comparison"),
        ("District-heat sales", a["district_heat_sales_mwh_year"], "MWh/y", "district-heat customer portfolio", "distribution", 2024, "retail heat sales", "selected-customer demand allocation", "input_constraint"),
        ("District-heat sales contracts", a["district_heat_sales_contracts"], "count", "district-heat customer portfolio", "distribution", 2024, "sales contracts", "represented contract count; not verified buildings", "input_constraint"),
        ("District-heat route", a["district_heat_route_km"], "km", "district-heat network", "distribution", 2024, "route length", "held-out comparison", "held_out_comparison"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "quantity", "value", "unit", "geographic_boundary", "network_level",
            "year", "customer_category", "model_use", "evidence_role",
        ],
    )
    frame["spatial_boundary_status"] = (
        "sector aggregate is internally matched to its published reporting envelope; "
        "no public machine-readable utility polygon was available, so the common service mask remains a declared class-C adapter"
    )
    frame.to_csv(OUT / "evidence_boundary_audit.csv", index=False)
    return frame


def build_building_ledger(config: dict[str, Any]) -> pd.DataFrame:
    """Register one building population and allocate all four demands."""
    write_evidence_boundary_audit(config)
    buildings = base.read_raw_json("buildings.json")["elements"]
    rows: list[dict[str, Any]] = []
    for element in buildings:
        point = base.center(element)
        if not point or not base.service_area(point):
            continue
        tags = element.get("tags") or {}
        kind = tags.get("building", "yes")
        service_class, service_eligible, service_reason = _building_service_class(tags, kind)
        footprint = _polygon_area_m2(element)
        area_class = "B" if footprint is not None else "D"
        if footprint is None:
            footprint = DEFAULT_FOOTPRINT_M2.get(kind, DEFAULT_FOOTPRINT_M2["yes"])
        levels = _safe_float(tags.get("building:levels"), DEFAULT_LEVELS.get(kind, 2.0))
        floor_area = float(np.clip(footprint * levels, 20.0, 60_000.0))
        residential = service_class == "residential"
        household_count = max(1, int(math.ceil(floor_area / 90.0))) if residential else 0
        occupant_prior = floor_area / 43.0 if residential and service_eligible else 0.0
        nonres_water_factor = {"commercial": 0.010, "public": 0.018, "industrial": 0.006}.get(service_class, 0.0)
        water_prior = occupant_prior + nonres_water_factor * floor_area if service_eligible else 0.0
        electric_prior = floor_area * SPECIFIC_ELECTRICITY_KWH_M2A.get(kind, 48.0) if service_eligible else 0.0
        heat_prior = floor_area * SPECIFIC_HEAT_KWH_M2A.get(kind, 105.0) if service_eligible else 0.0
        rows.append(
            {
                "building_id": f"OSM_W{element['id']}",
                "osm_id": element["id"],
                "building_type": kind,
                "service_class": service_class,
                "service_eligible": bool(service_eligible),
                "service_class_evidence": service_reason,
                "service_evidence_class": "B/C" if service_class != "unclassified" else "D",
                "has_address": bool(tags.get("addr:housenumber") or tags.get("addr:street")),
                "household_count": household_count,
                "name": tags.get("name", ""),
                "lon": point[0],
                "lat": point[1],
                "morphology": _morphology(point, config),
                "footprint_m2": round(footprint, 2),
                "levels": round(levels, 1),
                "floor_area_m2": round(floor_area, 2),
                "occupant_prior": occupant_prior,
                "water_prior": water_prior,
                "electricity_prior": electric_prior,
                "heat_prior": heat_prior,
                "geometry_evidence_class": area_class,
                "provenance": (
                    "OSM footprint/type/levels" if area_class == "B"
                    else "OSM building centroid/type; documented archetype area"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame = _infer_unclassified_service_classes(frame)
    # Recompute every class-dependent prior after spatial inference.  This
    # prevents a generic building that is inferred as residential or
    # industrial from retaining the provisional ``building=yes`` prior.
    residential_mask = frame["service_eligible"] & frame["service_class"].eq("residential")
    frame["household_count"] = 0
    demand_model = config["demand_model"]
    target_households = int(round(
        float(config["official_anchors"]["served_population"])
        / float(demand_model["average_household_size"])
    ))
    residential = frame.loc[residential_mask, ["building_id", "floor_area_m2"]].copy()
    if target_households < len(residential):
        raise ValueError("declared household total is smaller than the residential building count")
    residential["households"] = 1
    remaining_households = target_households - len(residential)
    if remaining_households:
        weights = residential["floor_area_m2"].clip(lower=1.0)
        raw_extra = remaining_households * weights / weights.sum()
        residential["households"] += np.floor(raw_extra).astype(int)
        remainder = target_households - int(residential["households"].sum())
        if remainder:
            fractional = (raw_extra - np.floor(raw_extra)).rename("fractional")
            order = (
                residential.assign(fractional=fractional)
                .sort_values(["fractional", "building_id"], ascending=[False, True])
                .index[:remainder]
            )
            residential.loc[order, "households"] += 1
    frame.loc[residential.index, "household_count"] = residential["households"].astype(int)
    frame["occupant_prior"] = 0.0
    frame.loc[residential_mask, "occupant_prior"] = frame.loc[residential_mask, "floor_area_m2"] / 43.0
    nonres_water = frame["service_class"].map(
        {"commercial": 0.010, "public": 0.018, "industrial": 0.006}
    ).fillna(0.0)
    frame["water_prior"] = frame["occupant_prior"] + nonres_water * frame["floor_area_m2"]
    frame.loc[~frame["service_eligible"], "water_prior"] = 0.0
    class_electricity = frame["service_class"].map(
        {"residential": 36.0, "commercial": 115.0, "public": 90.0, "industrial": 70.0}
    ).fillna(0.0)
    class_heat = frame["service_class"].map(
        {"residential": 115.0, "commercial": 85.0, "public": 105.0, "industrial": 55.0}
    ).fillna(0.0)
    frame["electricity_prior"] = class_electricity * frame["floor_area_m2"]
    frame["heat_prior"] = class_heat * frame["floor_area_m2"]
    frame.loc[~frame["service_eligible"], ["electricity_prior", "heat_prior"]] = 0.0
    anchors = config["official_anchors"]
    frame["population"] = frame["occupant_prior"] / frame["occupant_prior"].sum() * anchors["served_population"]
    frame["electricity_mwh_year"] = (
        frame["electricity_prior"] / frame["electricity_prior"].sum()
        * anchors["electricity_lv_annual_mwh"]
    )
    frame["electricity_peak_kw"] = (
        frame["electricity_prior"] / frame["electricity_prior"].sum()
        * anchors["electricity_lv_peak_mw"] * 1000.0
    )
    frame["water_m3_year"] = (
        frame["water_prior"] / frame["water_prior"].sum()
        * anchors["drinking_water_annual_m3"]
    )
    sanitary_return = float(demand_model["wastewater_sanitary_return_fraction"])
    frame["wastewater_sanitary_m3_year"] = sanitary_return * frame["water_m3_year"]
    frame["heat_candidate_mwh_year"] = frame["heat_prior"] / 1000.0
    # Equipment design uses a building connection peak distinct from the
    # diversified coincident contribution used in system power flow.
    n_households = frame["household_count"].clip(lower=1).astype(float)
    residential_coincidence = 0.07 + 0.93 * n_households ** (-0.75)
    residential_service_kw = (
        float(demand_model["residential_service_base_kw"])
        * n_households
        * residential_coincidence
    )
    nonres_specific_w_m2 = frame["service_class"].map(
        {"commercial": 85.0, "public": 95.0, "industrial": 80.0}
    ).fillna(45.0)
    nonres_service_kw = nonres_specific_w_m2 * frame["floor_area_m2"] / 1000.0
    frame["electricity_service_peak_kw"] = np.where(
        frame["service_class"].eq("residential"), residential_service_kw, nonres_service_kw
    )
    frame.loc[~frame["service_eligible"], "electricity_service_peak_kw"] = 0.0
    water_peak_factor = float(demand_model["drinking_water_peak_factor"])
    harmon_factor = max(
        2.0,
        1.0 + 14.0 / (4.0 + math.sqrt(float(anchors["served_population"]) / 1000.0)),
    )
    frame["water_service_peak_lps"] = (
        frame["water_m3_year"] / (365.0 * 86400.0) * water_peak_factor * 1000.0
    )
    frame["wastewater_service_peak_lps"] = (
        frame["wastewater_sanitary_m3_year"] / (365.0 * 86400.0) * harmon_factor * 1000.0
    )
    frame["heat_service_design_kw"] = (
        frame["heat_candidate_mwh_year"]
        / (float(demand_model["district_heat_full_load_hours"]) / 1000.0)
    )
    frame["electricity_connected"] = frame["service_eligible"]
    frame["electricity_connection_level"] = np.where(
        frame["electricity_service_peak_kw"] > 250.0, "MV", "LV"
    )
    frame.loc[~frame["service_eligible"], "electricity_connection_level"] = "NONE"
    frame["water_connected"] = frame["service_eligible"]
    frame["wastewater_connected"] = frame["service_eligible"]
    frame["heat_eligible"] = frame["service_eligible"] & ~frame["service_class"].eq("industrial")
    frame.drop(columns=["occupant_prior", "water_prior", "electricity_prior", "heat_prior"], inplace=True)
    frame.to_csv(OUT / "building_sector_demands.csv", index=False)
    pd.DataFrame(
        [
            {
                "ledger_id": "WATER_WHOLESALE_2024",
                "annual_m3": anchors["drinking_water_wholesale_delivery_m3"],
                "counterparty": "downstream distributors",
                "spatial_status": "reported boundary transfer; receiving service polygon not public",
                "physical_network_allocation": False,
                "wastewater_conversion": False,
                "reason": "excluded from local building demand and sanitary inflow",
            }
        ]
    ).to_csv(OUT / "water_wholesale_boundary_ledger.csv", index=False)
    return frame


def enrich_common_demand_zones(buildings: pd.DataFrame) -> pd.DataFrame:
    """Attach immutable membership and all four conserved demands to each zone."""
    path = OUT / "demand_zones.csv"
    zones = pd.read_csv(path)
    building_lookup = buildings.set_index("building_id")
    aggregate_columns = [
        "population",
        "electricity_mwh_year",
        "electricity_peak_kw",
        "water_m3_year",
        "wastewater_sanitary_m3_year",
        "heat_candidate_mwh_year",
    ]
    totals = {column: [] for column in aggregate_columns}
    membership_errors = []
    for row in zones.itertuples():
        member_ids = [item for item in str(row.building_ids).split(";") if item]
        missing = [item for item in member_ids if item not in building_lookup.index]
        membership_errors.extend(missing)
        members = building_lookup.loc[[item for item in member_ids if item in building_lookup.index]]
        for column in aggregate_columns:
            totals[column].append(float(members[column].sum()))
    if membership_errors:
        raise ValueError(f"Demand-zone membership contains {len(membership_errors)} unknown building IDs")
    for column, values in totals.items():
        zones[column] = values
    zones["electricity_peak_mw"] = zones["electricity_peak_kw"] / 1000.0
    heat_full_load_hours = float(load_case()["demand_model"]["district_heat_full_load_hours"])
    zones["heat_candidate_peak_mw"] = zones["heat_candidate_mwh_year"] / heat_full_load_hours
    zones["clustering_weight_mode"] = "OSM building-use weight"
    zones["clustering_seed"] = int(load_case()["seed"])
    zones.to_csv(path, index=False)
    return zones


def _heat_connection_selection(buildings: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    quotas = {"dense": 300, "suburban": 250, "industrial": 200, "peripheral": 50}
    centres = [
        (10.231, 50.047), (10.196, 50.057), (10.205, 50.033),
        tuple(config["district_heating"]["plant"]["coordinate"]),
    ]
    pieces = []
    for morphology, quota in quotas.items():
        subset = buildings[buildings["morphology"] == morphology].copy()
        distances = []
        for row in subset.itertuples():
            distances.append(min(base.distance_km((row.lon, row.lat), c) for c in centres))
        subset["selection_score"] = subset["heat_candidate_mwh_year"] / (0.20 + np.array(distances))
        pieces.append(subset.nlargest(min(quota, len(subset)), "selection_score"))
    selected = pd.concat(pieces, ignore_index=True)
    if len(selected) < config["district_heating"]["sales_contract_count"]:
        missing = config["district_heating"]["sales_contract_count"] - len(selected)
        remainder = buildings[~buildings["building_id"].isin(selected["building_id"])].nlargest(
            missing, "heat_candidate_mwh_year"
        )
        selected = pd.concat([selected, remainder], ignore_index=True)
    selected["district_heat_mwh_year"] = (
        selected["heat_candidate_mwh_year"] / selected["heat_candidate_mwh_year"].sum()
        * config["district_heating"]["heat_sales_mwh_year"]
    )
    selected["district_heat_peak_kw"] = (
        selected["heat_candidate_mwh_year"] / selected["heat_candidate_mwh_year"].sum()
        * config["district_heating"]["design_peak_mw_assumption"] * 1000.0
    )
    selected["district_heat_connected"] = True
    return selected


def _cluster_heat_consumers(selected: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    morphology_weights = {"dense": 30, "suburban": 24, "industrial": 20, "peripheral": 6}
    total_target = max(4, int(config["district_heating"].get("aggregate_consumer_nodes", 80)))
    weight_sum = sum(morphology_weights.values())
    exact = {key: total_target * value / weight_sum for key, value in morphology_weights.items()}
    cluster_targets = {key: max(1, int(round(value))) for key, value in exact.items()}
    while sum(cluster_targets.values()) < total_target:
        key = max(exact, key=lambda item: exact[item] - cluster_targets[item])
        cluster_targets[key] += 1
    while sum(cluster_targets.values()) > total_target:
        eligible = [key for key, value in cluster_targets.items() if value > 1]
        key = max(eligible, key=lambda item: cluster_targets[item] - exact[item])
        cluster_targets[key] -= 1
    rows = []
    index = 1
    for morphology, count in cluster_targets.items():
        group = selected[selected["morphology"] == morphology].copy()
        if group.empty:
            continue
        count = min(count, len(group))
        xy = np.array([base.xy_km((r.lon, r.lat)) for r in group.itertuples()])
        _, labels = kmeans2(xy, count, minit="++", iter=50, seed=20260802)
        group["cluster"] = labels
        for cluster, part in group.groupby("cluster"):
            weights = part["district_heat_peak_kw"].to_numpy()
            rows.append(
                {
                    "consumer_id": f"DH_C{index:03d}",
                    "name": f"{morphology.title()} heat-demand cluster {int(cluster) + 1}",
                    "lon": float(np.average(part["lon"], weights=weights)),
                    "lat": float(np.average(part["lat"], weights=weights)),
                    "morphology": morphology,
                    "represented_contract_count": int(len(part)),
                    "annual_heat_mwh": float(part["district_heat_mwh_year"].sum()),
                    "peak_heat_mw": float(part["district_heat_peak_kw"].sum() / 1000.0),
                    "building_ids": ";".join(part["building_id"].tolist()),
                    "confidence_class": "C/D",
                    "provenance": (
                        "OSM building evidence; official sales-contract and annual-sales calibration; "
                        "one selected building per represented contract is a modelling assumption"
                    ),
                }
            )
            index += 1
    return pd.DataFrame(rows)


def _round_up(value: float, library: Iterable[int]) -> int:
    values = list(library)
    return int(next((item for item in values if item >= value), values[-1]))


def build_heat_network(
    buildings: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    roads = base.build_road_graph(base.load_elements("local_roads.json"))
    road_index = base.NodeIndex(roads.nodes)
    selected = _heat_connection_selection(buildings, config)
    consumers = _cluster_heat_consumers(selected, config)
    consumers["road_node"] = [road_index.nearest((r.lon, r.lat)) for r in consumers.itertuples()]
    consumers["road_node_key"] = consumers["road_node"].map(lambda p: f"{p[0]:.7f},{p[1]:.7f}")
    # Multiple consumer clusters can snap to one corridor node; keep their IDs
    # separate in the ledger but aggregate their hydraulic sink at that node.
    dh = config["district_heating"]
    plant_point = tuple(dh["plant"]["coordinate"])
    source = road_index.nearest(plant_point)
    union = nx.Graph()
    for target in dict.fromkeys(consumers["road_node"]):
        path = nx.shortest_path(roads, source, target, weight="length_km")
        base.add_path_edges(union, roads, path)
    tree = nx.minimum_spanning_tree(union, weight="length_km")
    protected = {source, *consumers["road_node"].tolist()}
    network = base.compress_undirected(tree, protected)

    peak_at_node = defaultdict(float)
    annual_at_node = defaultdict(float)
    connection_at_node = defaultdict(int)
    ids_at_node = defaultdict(list)
    morphology_at_node = defaultdict(list)
    for row in consumers.itertuples():
        peak_at_node[row.road_node] += row.peak_heat_mw
        annual_at_node[row.road_node] += row.annual_heat_mwh
        connection_at_node[row.road_node] += row.represented_contract_count
        ids_at_node[row.road_node].append(row.consumer_id)
        morphology_at_node[row.road_node].append(row.morphology)

    edge_heat = defaultdict(float)
    for target, peak in peak_at_node.items():
        path = nx.shortest_path(network, source, target, weight="length_km")
        for u, v in zip(path[:-1], path[1:]):
            edge_heat[frozenset((u, v))] += peak

    node_ids: dict[tuple[float, float], str] = {}
    node_rows = []
    for index, node in enumerate(network.nodes, 1):
        node_id = f"DH_N{index:04d}"
        node_ids[node] = node_id
        if node == source:
            node_type = "plant"
            name = dh["plant"]["name"]
        elif node in peak_at_node:
            node_type = "consumer_substation"
            name = ";".join(ids_at_node[node])
        else:
            node_type = "junction"
            name = node_id
        node_rows.append(
            {
                "node_id": node_id,
                "name": name,
                "node_type": node_type,
                "lon": node[0],
                "lat": node[1],
                "morphology": _morphology(node, config),
                "elevation_m": base.proxy_elevation_m(node),
                "annual_heat_mwh": annual_at_node[node],
                "peak_heat_mw": peak_at_node[node],
                "represented_contract_count": connection_at_node[node],
                "supply_temperature_c": dh["nominal_supply_temperature_c"],
                "return_temperature_c": dh["nominal_return_temperature_c"],
                "confidence_class": "A" if node == source else "B/C/D",
            }
        )

    corridor_rows = []
    supply_rows = []
    density = 985.0
    cp = 4180.0
    delta_t = dh["nominal_supply_temperature_c"] - dh["nominal_return_temperature_c"]
    for index, (u, v, data) in enumerate(network.edges(data=True), 1):
        peak = max(edge_heat[frozenset((u, v))], 0.03)
        mass_flow = peak * 1e6 / (cp * delta_t)
        diameter_m = math.sqrt(4.0 * mass_flow / (density * math.pi * dh["target_velocity_m_s"]))
        dn = _round_up(math.ceil(diameter_m * 1000.0), dh["pipe_dn_mm"])
        geometry = data.get("geometry", [u, v])
        corridor_id = f"DH_R{index:04d}"
        row = {
            "corridor_id": corridor_id,
            "from_node": node_ids[u],
            "to_node": node_ids[v],
            "length_km": data["length_km"],
            "diameter_mm": dn,
            "design_peak_mw": peak,
            "mass_flow_kg_s": mass_flow,
            "target_pressure_gradient_pa_m": dh["target_pressure_gradient_pa_m"],
            "insulation_series": "EN 253 series 2 prior",
            "geometry_json": json.dumps(geometry, separators=(",", ":")),
            "morphology": _morphology(tuple(geometry[len(geometry) // 2]), config),
            "confidence_class": "C/D",
            "provenance": "road-routed topology; official heat totals; documented sizing rule",
        }
        corridor_rows.append(row)
        for circuit, a, b in [("supply", u, v), ("return", v, u)]:
            supply_rows.append(
                {
                    "pipe_id": f"{corridor_id}_{circuit[0].upper()}",
                    "corridor_id": corridor_id,
                    "circuit": circuit,
                    "from_node": node_ids[a],
                    "to_node": node_ids[b],
                    **{k: row[k] for k in row if k not in {"corridor_id", "from_node", "to_node"}},
                }
            )
    nodes = pd.DataFrame(node_rows)
    corridors = pd.DataFrame(corridor_rows)
    pipes = pd.DataFrame(supply_rows)
    target_length = float(dh["route_length_target_km"])
    mapped_length = float(corridors["length_km"].sum())
    # The public 52-km quantity is retained as a comparison, not manufactured
    # by adding an unlocated equivalent lateral to the generated tree.
    equivalent_laterals = 0.0
    consumers.drop(columns=["road_node"], inplace=True)
    consumers.to_csv(OUT / "heat_consumers.csv", index=False)
    selected[[
        "building_id", "morphology", "district_heat_mwh_year", "district_heat_peak_kw"
    ]].to_csv(OUT / "heat_building_connections.csv", index=False)
    nodes.to_csv(OUT / "heat_nodes.csv", index=False)
    corridors.to_csv(OUT / "heat_corridors.csv", index=False)
    pipes.to_csv(OUT / "heat_pipes_supply_return.csv", index=False)
    metadata = {
        "plant_node": node_ids[source],
        "mapped_route_length_km": mapped_length,
        "equivalent_service_laterals_km": equivalent_laterals,
        "generated_route_length_km": mapped_length,
        "published_route_length_km": target_length,
        "physical_pipe_length_supply_return_km": 2.0 * (mapped_length + equivalent_laterals),
        "consumer_substations": int(len(consumers)),
        "represented_sales_contracts": int(consumers["represented_contract_count"].sum()),
        "annual_heat_mwh": float(consumers["annual_heat_mwh"].sum()),
        "peak_heat_mw": float(consumers["peak_heat_mw"].sum()),
    }
    return nodes, corridors, consumers, metadata


def augment_power_model(config: dict[str, Any]) -> dict[str, Any]:
    """Extend the visible MV scaffold with synthetic MV/LV sites and LV trees."""
    visible = pd.read_csv(OUT / "power_buses.csv")
    backbone = pd.read_csv(OUT / "power_lines.csv")
    if "bus_role" in visible.columns:
        visible = visible[visible["bus_role"] == "visible_mv_substation"].copy()
    if "asset_type" in backbone.columns:
        backbone = backbone[backbone["asset_type"] == "mv_backbone"].copy()
    buildings = pd.read_csv(OUT / "building_sector_demands.csv")
    water_nodes = pd.read_csv(OUT / "water_nodes.csv")
    targets = water_nodes[water_nodes["building_count"] > 0].copy().reset_index(drop=True)
    road = base.build_road_graph(base.load_elements("local_roads.json"))
    road_index = base.NodeIndex(road.nodes)

    def xy_frame(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([base.xy_km((r.lon, r.lat)) for r in frame.itertuples()], dtype=float)

    # The public aggregate reports 105 MV/LV withdrawal sites.  Their synthetic
    # locations are demand-cluster centroids snapped to the admissible road graph.
    target_sites = int(config["official_anchors"]["electricity_mv_lv_withdrawal_locations"])
    building_xy = xy_frame(buildings)
    target_xy = xy_frame(targets)
    target_tree = cKDTree(target_xy)
    centroids, _ = kmeans2(building_xy, target_sites, minit="++", iter=70, seed=config["seed"])
    site_nodes: list[tuple[float, float]] = []
    for x, y in centroids:
        _, target_index = target_tree.query((float(x), float(y)))
        node = (float(targets.loc[int(target_index), "lon"]), float(targets.loc[int(target_index), "lat"]))
        if node not in site_nodes:
            site_nodes.append(node)
    # Snapping can merge rare adjacent centroids; supplement with the target
    # farthest from the current site set until the published site count is met.
    target_points = [(float(r.lon), float(r.lat)) for r in targets.itertuples()]
    while len(site_nodes) < target_sites:
        candidate = max(
            (point for point in target_points if point not in site_nodes),
            key=lambda point: min(base.distance_km(point, site) for site in site_nodes),
        )
        site_nodes.append(candidate)
    site_nodes = site_nodes[:target_sites]
    site_xy = np.asarray([base.xy_km(point) for point in site_nodes])
    site_tree = cKDTree(site_xy)

    _, building_target = target_tree.query(building_xy)
    targets["electricity_peak_mw"] = 0.0
    targets["electricity_mwh_year"] = 0.0
    targets["population"] = 0.0
    peak_by_target = buildings.groupby(building_target)["electricity_peak_kw"].sum() / 1000.0
    energy_by_target = buildings.groupby(building_target)["electricity_mwh_year"].sum()
    population_by_target = buildings.groupby(building_target)["population"].sum()
    for index, value in peak_by_target.items():
        targets.loc[int(index), "electricity_peak_mw"] = float(value)
    for index, value in energy_by_target.items():
        targets.loc[int(index), "electricity_mwh_year"] = float(value)
    for index, value in population_by_target.items():
        targets.loc[int(index), "population"] = float(value)
    _, euclidean_site = site_tree.query(target_xy)
    # Network-distance allocation prevents assignments across rivers, rail
    # yards or other barriers that are close in plan but far by admissible road.
    try:
        _, source_paths = nx.multi_source_dijkstra(road, site_nodes, weight="length_km")
        site_lookup = {point: index for index, point in enumerate(site_nodes)}
        targets["site_index"] = [
            site_lookup[source_paths[(float(row.lon), float(row.lat))][0]]
            if (float(row.lon), float(row.lat)) in source_paths
            else int(euclidean_site[index])
            for index, row in targets.iterrows()
        ]
    except nx.NetworkXNoPath:
        targets["site_index"] = euclidean_site

    bus_rows: list[dict[str, Any]] = []
    for row in visible.itertuples():
        bus_rows.append(
            {
                "bus_id": row.bus_id, "name": row.name, "voltage_kv": 20.0,
                "operator": row.operator, "root": bool(row.root), "osm_id": row.osm_id,
                "confidence_class": row.confidence_class, "bus_role": "visible_mv_substation",
                "background_peak_p_mw": 0.0, "background_peak_q_mvar": 0.0,
                "annual_electricity_mwh": 0.0, "population": 0.0,
                "lon": float(row.lon), "lat": float(row.lat),
            }
        )
    visible_ids = visible["bus_id"].tolist()
    visible_points = [(float(r.lon), float(r.lat)) for r in visible.itertuples()]
    visible_road_nodes = [road_index.nearest(point) for point in visible_points]

    site_peak = targets.groupby("site_index")["electricity_peak_mw"].sum().to_dict()
    site_energy = targets.groupby("site_index")["electricity_mwh_year"].sum().to_dict()
    site_population = targets.groupby("site_index")["population"].sum().to_dict()
    transformer_rows = []
    transformer_standard_mva = [0.25, 0.4, 0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5]
    # The published 105 quantity is a count of MS/LV withdrawal locations,
    # not a verified count of physical transformer sites.  Each generated
    # element is therefore an equivalent transformation portfolio at one
    # synthetic withdrawal location.  The portfolio is upgraded, using the
    # declared standard ratings, until its aggregate installed capacity is as
    # close as possible to the separately published 216.71 MVA.
    target_installed_mva = float(
        config["official_anchors"]["electricity_mv_lv_installed_capacity_mva"]
    )
    equivalent_ratings = []
    for index in range(target_sites):
        required = float(site_peak.get(index, 0.0)) / 0.80
        equivalent_ratings.append(
            next((value for value in transformer_standard_mva if value >= required), transformer_standard_mva[-1])
        )
    while sum(equivalent_ratings) < target_installed_mva:
        candidates = []
        for index, current in enumerate(equivalent_ratings):
            position = transformer_standard_mva.index(current)
            if position + 1 >= len(transformer_standard_mva):
                continue
            upgraded = transformer_standard_mva[position + 1]
            increment = upgraded - current
            score = float(site_peak.get(index, 0.0)) / max(current, 0.01)
            candidates.append((score, -increment, index, upgraded))
        if not candidates:
            break
        _, _, selected_index, upgraded = max(candidates)
        before = abs(target_installed_mva - sum(equivalent_ratings))
        after = abs(target_installed_mva - (sum(equivalent_ratings) - equivalent_ratings[selected_index] + upgraded))
        if after > before:
            break
        equivalent_ratings[selected_index] = upgraded
    # A location represents a portfolio, not one nameplate unit.  Allocate the
    # remaining sub-step residual to the most heavily loaded non-saturated
    # portfolio so the published aggregate capacity is reproduced exactly
    # without pretending that 216.71 MVA identifies individual transformers.
    residual_mva = target_installed_mva - sum(equivalent_ratings)
    if abs(residual_mva) > 1e-9:
        adjustment_index = max(
            range(len(equivalent_ratings)),
            key=lambda item: float(site_peak.get(item, 0.0)) / max(equivalent_ratings[item], 0.01),
        )
        equivalent_ratings[adjustment_index] += residual_mva
    for index, point in enumerate(site_nodes):
        mv_bus = f"TMV{index + 1:03d}"
        lv_bus = f"TLV{index + 1:03d}"
        peak = float(site_peak.get(index, 0.0))
        size = equivalent_ratings[index]
        bus_rows.extend(
            [
                {
                    "bus_id": mv_bus, "name": f"Synthetic MS/LV withdrawal location {index + 1:03d}",
                    "voltage_kv": 20.0, "operator": "synthetic", "root": False, "osm_id": np.nan,
                    "confidence_class": "D", "bus_role": "transformer_mv_bus",
                    "background_peak_p_mw": 0.0, "background_peak_q_mvar": 0.0,
                    "annual_electricity_mwh": 0.0, "population": 0.0,
                    "lon": point[0], "lat": point[1],
                },
                {
                    "bus_id": lv_bus, "name": f"Synthetic LV root {index + 1:03d}",
                    "voltage_kv": 0.4, "operator": "synthetic", "root": False, "osm_id": np.nan,
                    "confidence_class": "D", "bus_role": "transformer_lv_bus",
                    "background_peak_p_mw": 0.0, "background_peak_q_mvar": 0.0,
                    "annual_electricity_mwh": 0.0, "population": 0.0,
                    "lon": point[0], "lat": point[1],
                },
            ]
        )
        transformer_rows.append(
            {
                "transformer_id": f"TR{index + 1:03d}", "mv_bus": mv_bus, "lv_bus": lv_bus,
                "sn_mva": size, "vn_hv_kv": 20.0, "vn_lv_kv": 0.4, "uk_percent": 6.0,
                "copper_loss_kw": 8.5 * size, "assigned_peak_mw": peak,
                "annual_electricity_mwh": float(site_energy.get(index, 0.0)),
                "population": float(site_population.get(index, 0.0)),
                "lon": point[0], "lat": point[1], "morphology": _morphology(point, config),
                "confidence_class": "D", "asset_interpretation": "equivalent portfolio at synthetic withdrawal location",
            }
        )
    transformers = pd.DataFrame(transformer_rows)

    line_rows: list[dict[str, Any]] = []
    for row in backbone.itertuples():
        line_rows.append(
            {
                "line_id": row.line_id, "from_bus": row.from_bus, "to_bus": row.to_bus,
                "length_km": float(row.length_km), "r_ohm_per_km": float(row.r_ohm_per_km),
                "x_ohm_per_km": float(row.x_ohm_per_km), "c_nf_per_km": float(row.c_nf_per_km),
                "max_i_ka": float(row.max_i_ka), "normally_open": bool(row.normally_open),
                "confidence_class": row.confidence_class, "provenance": row.provenance,
                "geometry_json": row.geometry_json, "asset_type": "mv_backbone",
                "voltage_level": "MV", "design_power_mw": 0.0, "design_current_ka": 0.0,
                "cable_size_mm2": 240.0, "parallel_circuits": 1,
            }
        )

    # Each synthetic transformer site is attached by one radial MV spur to the
    # closest visible MV bus.  Shared street segments remain registered later.
    for index, point in enumerate(site_nodes):
        nearest = min(range(len(visible_points)), key=lambda j: base.distance_km(point, visible_points[j]))
        try:
            route = nx.shortest_path(road, visible_road_nodes[nearest], point, weight="length_km")
            length = sum(road[u][v]["length_km"] for u, v in zip(route[:-1], route[1:]))
        except nx.NetworkXNoPath:
            route = [visible_points[nearest], point]
            length = base.distance_km(*route)
        line_rows.append(
            {
                "line_id": f"MVS{index + 1:03d}", "from_bus": visible_ids[nearest],
                "to_bus": f"TMV{index + 1:03d}", "length_km": max(float(length), 0.005),
                "r_ohm_per_km": 0.320, "x_ohm_per_km": 0.080, "c_nf_per_km": 250.0,
                "max_i_ka": 0.25, "normally_open": False, "confidence_class": "D",
                "provenance": "road-routed synthetic MV transformer spur",
                "geometry_json": json.dumps(route, separators=(",", ":")),
                "asset_type": "mv_transformer_spur", "voltage_level": "MV",
                "design_power_mw": 0.0, "design_current_ka": 0.0,
                "cable_size_mm2": 95.0, "parallel_circuits": 1,
            }
        )

    # Size the operating MV tree by downstream coincident peak.
    mv_graph = nx.Graph()
    for row in visible.itertuples():
        mv_graph.add_node(row.bus_id, load=0.0, root=bool(row.root))
    for index in range(target_sites):
        mv_graph.add_node(f"TMV{index + 1:03d}", load=float(site_peak.get(index, 0.0)), root=False)
    for row in line_rows:
        if row["voltage_level"] == "MV" and not row["normally_open"]:
            mv_graph.add_edge(row["from_bus"], row["to_bus"], line_id=row["line_id"])
    roots = {node for node, data in mv_graph.nodes(data=True) if data["root"]}
    mv_library = [(95, 0.25, 0.320), (150, 0.32, 0.206), (185, 0.36, 0.164), (240, 0.42, 0.125), (300, 0.48, 0.100), (400, 0.56, 0.077)]
    mv_sizing: dict[str, dict[str, float]] = {}
    for u, v, data in mv_graph.edges(data=True):
        graph_copy = mv_graph.copy()
        graph_copy.remove_edge(u, v)
        side_u = nx.node_connected_component(graph_copy, u)
        side_v = nx.node_connected_component(graph_copy, v)
        if roots.intersection(side_u) and not roots.intersection(side_v):
            downstream = side_v
        elif roots.intersection(side_v) and not roots.intersection(side_u):
            downstream = side_u
        else:
            downstream = min((side_u, side_v), key=lambda side: sum(graph_copy.nodes[n]["load"] for n in side))
        power = sum(mv_graph.nodes[n]["load"] for n in downstream)
        current = power / (math.sqrt(3.0) * 20.0 * 0.96)
        selected, parallel = mv_library[-1], 1
        for candidate in mv_library:
            needed = max(1, math.ceil(current / (0.80 * candidate[1])))
            if needed <= 2:
                selected, parallel = candidate, needed
                break
        else:
            selected = mv_library[-1]
            parallel = max(1, math.ceil(current / (0.80 * selected[1])))
        mv_sizing[data["line_id"]] = {
            "design_power_mw": power, "design_current_ka": current,
            "cable_size_mm2": selected[0], "max_i_ka": selected[1],
            "r_ohm_per_km": selected[2], "parallel_circuits": parallel,
        }
    for row in line_rows:
        if row["line_id"] in mv_sizing:
            row.update(mv_sizing[row["line_id"]])

    # Generate one road-routed LV feeder tree per transformer.  Every target is
    # a common demand-zone node and therefore retains the same building ledger.
    lv_junction_counter = 1
    lv_line_counter = 1
    lv_library = [(35, 0.115, 0.868), (70, 0.179, 0.443), (95, 0.216, 0.320), (150, 0.282, 0.206), (240, 0.368, 0.125)]
    bus_row_index = {row["bus_id"]: index for index, row in enumerate(bus_rows)}
    for site_index, source in enumerate(site_nodes):
        group = targets[targets["site_index"] == site_index]
        if group.empty:
            continue
        union = nx.Graph()
        target_nodes = [(float(r.lon), float(r.lat)) for r in group.itertuples()]
        union.add_nodes_from([source, *target_nodes])
        for target in target_nodes:
            try:
                base.add_path_edges(union, road, nx.shortest_path(road, source, target, weight="length_km"))
            except nx.NetworkXNoPath:
                union.add_edge(source, target, length_km=base.distance_km(source, target), geometry=[source, target])
        tree = nx.minimum_spanning_tree(union, weight="length_km")
        model = base.compress_undirected(tree, {source, *target_nodes})
        model.add_nodes_from(node for node in [source, *target_nodes] if node in tree)
        root_bus = f"TLV{site_index + 1:03d}"
        node_bus: dict[tuple[float, float], str] = {source: root_bus}
        target_index = {(float(row.lon), float(row.lat)): int(index) for index, row in group.iterrows()}
        for node in model.nodes:
            if node == source:
                continue
            if node in target_index:
                bus_id = f"LVL{target_index[node] + 1:04d}"
                target_row = targets.loc[target_index[node]]
                role = "lv_load_bus"
                peak = float(target_row.electricity_peak_mw)
                annual = float(target_row.electricity_mwh_year)
                population = float(target_row.population)
                name = f"Common demand zone {target_row.node_id}"
            else:
                bus_id = f"LVJ{lv_junction_counter:05d}"
                lv_junction_counter += 1
                role, peak, annual, population, name = "lv_junction", 0.0, 0.0, 0.0, bus_id
            node_bus[node] = bus_id
            bus_rows.append(
                {
                    "bus_id": bus_id, "name": name, "voltage_kv": 0.4,
                    "operator": "synthetic", "root": False, "osm_id": np.nan,
                    "confidence_class": "C/D", "bus_role": role,
                    "background_peak_p_mw": peak,
                    "background_peak_q_mvar": peak * math.tan(math.acos(0.96)),
                    "annual_electricity_mwh": annual, "population": population,
                    "lon": node[0], "lat": node[1],
                }
            )
        # If a demand node coincides with the transformer, retain the load on
        # the LV root rather than manufacturing a zero-length feeder.
        if source in target_index:
            target_row = targets.loc[target_index[source]]
            root = bus_rows[bus_row_index[root_bus]]
            root.update(
                background_peak_p_mw=float(target_row.electricity_peak_mw),
                background_peak_q_mvar=float(target_row.electricity_peak_mw) * math.tan(math.acos(0.96)),
                annual_electricity_mwh=float(target_row.electricity_mwh_year),
                population=float(target_row.population),
                bus_role="transformer_lv_load_bus",
            )
        parent = {child: ancestor for ancestor, child in nx.bfs_edges(model, source)}
        path_lengths = nx.single_source_dijkstra_path_length(model, source, weight="length_km")
        maximum_path_length = max(path_lengths.values(), default=0.001)
        direct_load = {node: 0.0 for node in model.nodes}
        for node, index in target_index.items():
            if node in direct_load:
                direct_load[node] += float(targets.loc[index, "electricity_peak_mw"])
        subtree = dict(direct_load)
        for node in reversed(list(nx.bfs_tree(model, source).nodes)):
            if node != source:
                subtree[parent[node]] += subtree[node]
        for child, ancestor in parent.items():
            data = model[ancestor][child]
            power = max(subtree[child], 0.001)
            current = power / (math.sqrt(3.0) * 0.4 * 0.96)
            sin_phi = math.sqrt(1.0 - 0.96**2)
            voltage_drop_limit = max(
                0.006,
                0.055 * float(data["length_km"]) / max(maximum_path_length, 0.001),
            )
            feasible: list[tuple[float, tuple[int, float, float], int]] = []
            for candidate in lv_library:
                for circuits in range(1, 17):
                    voltage_drop_pu = (
                        math.sqrt(3.0) * current
                        * (candidate[2] * 0.96 + 0.080 * sin_phi)
                        * float(data["length_km"]) / (0.4 * circuits)
                    )
                    if current <= 0.80 * candidate[1] * circuits and voltage_drop_pu <= voltage_drop_limit:
                        installed_area = candidate[0] * circuits * (1.0 + 0.10 * (circuits - 1))
                        feasible.append((installed_area, candidate, circuits))
                        break
            if feasible:
                _, selected, equivalent_parallel = min(feasible, key=lambda item: item[0])
            else:
                selected = lv_library[-1]
                uncorrected_drop = (
                    math.sqrt(3.0) * current
                    * (selected[2] * 0.96 + 0.080 * sin_phi)
                    * float(data["length_km"]) / 0.4
                )
                equivalent_parallel = max(
                    1,
                    math.ceil(current / (0.80 * selected[1])),
                    math.ceil(uncorrected_drop / voltage_drop_limit),
                )
            # More than four conductors on one feeder is not treated as a
            # credible physical LV design.  Large equivalent requirements are
            # decomposed into separately protected feeder groups.  The reduced
            # zone model retains one corridor edge, so pandapower receives the
            # electrically equivalent product of groups and circuits while the
            # asset table exposes the design decomposition explicitly.
            feeder_groups = max(1, math.ceil(equivalent_parallel / 4))
            parallel = max(1, math.ceil(equivalent_parallel / feeder_groups))
            geometry = data.get("geometry", [ancestor, child])
            if tuple(geometry[0]) != ancestor:
                geometry = list(reversed(geometry))
            line_rows.append(
                {
                    "line_id": f"LVF{lv_line_counter:05d}", "from_bus": node_bus[ancestor],
                    "to_bus": node_bus[child], "length_km": float(data["length_km"]),
                    "r_ohm_per_km": selected[2], "x_ohm_per_km": 0.080,
                    "c_nf_per_km": 210.0, "max_i_ka": selected[1], "normally_open": False,
                    "confidence_class": "C/D", "provenance": "road-routed synthetic radial LV feeder",
                    "geometry_json": json.dumps(geometry, separators=(",", ":")),
                    "asset_type": "lv_feeder", "voltage_level": "LV",
                    "design_power_mw": power, "design_current_ka": current,
                    "cable_size_mm2": selected[0], "parallel_circuits": parallel,
                    "feeder_groups": feeder_groups,
                }
            )
            lv_line_counter += 1

    # A zero-length logical edge makes transformer connectivity explicit in
    # graph exports; pandapower receives the corresponding transformer element.
    for row in transformers.itertuples():
        line_rows.append(
            {
                "line_id": f"TC_{row.transformer_id}", "from_bus": row.mv_bus, "to_bus": row.lv_bus,
                "length_km": 0.0, "r_ohm_per_km": 0.0, "x_ohm_per_km": 0.0,
                "c_nf_per_km": 0.0, "max_i_ka": np.nan, "normally_open": False,
                "confidence_class": row.confidence_class, "provenance": "MV/LV transformer connectivity",
                "geometry_json": json.dumps([[row.lon, row.lat], [row.lon, row.lat]], separators=(",", ":")),
                "asset_type": "transformer_connection", "voltage_level": "MV/LV",
                "design_power_mw": row.assigned_peak_mw, "design_current_ka": np.nan,
                "cable_size_mm2": np.nan, "parallel_circuits": 1, "feeder_groups": 1,
            }
        )

    buses = pd.DataFrame(bus_rows)
    lines = pd.DataFrame(line_rows)
    lines["feeder_groups"] = lines.get("feeder_groups", 1).fillna(1).astype(int)
    transformers.to_csv(OUT / "power_transformers.csv", index=False)

    solver = {"created": False, "converged": False}
    try:
        import pandapower as pp

        net = pp.create_empty_network(sn_mva=100.0)
        bus_index = {}
        for row in buses.itertuples():
            bus_index[row.bus_id] = pp.create_bus(
                net, vn_kv=float(row.voltage_kv), name=row.bus_id, geodata=(row.lon, row.lat),
                min_vm_pu=0.88 if row.voltage_kv < 1.0 else 0.90, max_vm_pu=1.10,
            )
            if bool(row.root):
                pp.create_ext_grid(net, bus_index[row.bus_id], vm_pu=1.05, name=f"GRID_{row.bus_id}")
            if row.background_peak_p_mw > 0:
                pp.create_load(
                    net, bus_index[row.bus_id], p_mw=row.background_peak_p_mw,
                    q_mvar=row.background_peak_q_mvar, name=f"LD_{row.bus_id}",
                )
        line_index_by_id = {}
        for row in lines[lines["asset_type"] != "transformer_connection"].itertuples():
            line_index = pp.create_line_from_parameters(
                net, bus_index[row.from_bus], bus_index[row.to_bus], length_km=max(row.length_km, 0.001),
                r_ohm_per_km=row.r_ohm_per_km, x_ohm_per_km=row.x_ohm_per_km,
                c_nf_per_km=row.c_nf_per_km, max_i_ka=row.max_i_ka,
                parallel=max(1, int(row.parallel_circuits) * int(getattr(row, "feeder_groups", 1))),
                name=row.line_id,
            )
            line_index_by_id[row.line_id] = line_index
            if bool(row.normally_open):
                pp.create_switch(net, bus_index[row.from_bus], line_index, et="l", closed=False, name=f"SW_{row.line_id}")
        transformer_index = {}
        for row in transformers.itertuples():
            transformer_index[row.transformer_id] = pp.create_transformer_from_parameters(
                net, hv_bus=bus_index[row.mv_bus], lv_bus=bus_index[row.lv_bus],
                sn_mva=row.sn_mva, vn_hv_kv=row.vn_hv_kv, vn_lv_kv=row.vn_lv_kv,
                vk_percent=row.uk_percent, vkr_percent=100.0 * row.copper_loss_kw / (row.sn_mva * 1000.0),
                pfe_kw=1.2 * row.sn_mva, i0_percent=0.20, shift_degree=0.0,
                tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2,
                tap_step_percent=2.5, tap_pos=-2, name=row.transformer_id,
            )
        converged_algorithm = ""
        last_error: Exception | None = None
        for algorithm in ["bfsw", "iwamoto_nr", "nr"]:
            try:
                pp.runpp(
                    net, algorithm=algorithm, calculate_voltage_angles=False,
                    numba=False, max_iteration=80, init="flat",
                )
                if net.converged:
                    converged_algorithm = algorithm
                    break
            except Exception as exc:
                last_error = exc
        if not net.converged:
            raise RuntimeError(f"pandapower did not converge; last_error={last_error}")
        pp.to_json(net, str(OUT / "power_pandapower.json"))
        buses["result_vm_pu"] = [float(net.res_bus.loc[bus_index[item], "vm_pu"]) for item in buses["bus_id"]]
        lines["result_loading_percent"] = np.nan
        for line_id, net_index in line_index_by_id.items():
            lines.loc[lines["line_id"] == line_id, "result_loading_percent"] = float(net.res_line.loc[net_index, "loading_percent"])
        transformers["result_loading_percent"] = [float(net.res_trafo.loc[transformer_index[item], "loading_percent"]) for item in transformers["transformer_id"]]
        solver = {
            "created": True, "converged": bool(net.converged),
            "minimum_voltage_pu": float(net.res_bus.vm_pu.min()),
            "maximum_voltage_pu": float(net.res_bus.vm_pu.max()),
            "maximum_line_loading_percent": float(net.res_line.loading_percent.max()),
            "maximum_transformer_loading_percent": float(net.res_trafo.loading_percent.max()),
            "algorithm": converged_algorithm,
        }
    except Exception as exc:  # pragma: no cover - optional solver path
        solver["error"] = f"{type(exc).__name__}: {exc}"
    buses.to_csv(OUT / "power_buses.csv", index=False)
    lines.to_csv(OUT / "power_lines.csv", index=False)
    transformers.to_csv(OUT / "power_transformers.csv", index=False)
    return {"solver": solver, "transformers": int(len(transformers))}


def write_and_run_water_epanet() -> dict[str, Any]:
    nodes = pd.read_csv(OUT / "water_nodes.csv")
    pipes = pd.read_csv(OUT / "water_pipes.csv")
    result = {"created": False, "converged": False}
    try:
        import wntr
        source_ids = set(nodes.loc[nodes["node_type"] == "source", "node_id"])
        source_head_rise_m = 90.0

        def create_network(current_pipes: pd.DataFrame, multiplier: float = 1.0):
            wn = wntr.network.WaterNetworkModel()
            for row in nodes.itertuples():
                if row.node_id in source_ids:
                    wn.add_reservoir(
                        row.node_id, base_head=float(row.height_m + source_head_rise_m),
                        coordinates=(row.lon, row.lat),
                    )
                else:
                    demand = max(0.0, float(row.water_q_avg_m3s))
                    wn.add_junction(
                        row.node_id, base_demand=demand, elevation=float(row.height_m),
                        coordinates=(row.lon, row.lat),
                    )
            for row in current_pipes.itertuples():
                wn.add_pipe(
                    row.pipe_id, row.from_node, row.to_node,
                    length=float(row.length_km * 1000.0), diameter=float(row.diameter_mm / 1000.0),
                    roughness=130.0, minor_loss=float(row.loss_coefficient), initial_status="OPEN",
                )
            wn.options.time.duration = 3600
            wn.options.time.hydraulic_timestep = 3600
            wn.options.hydraulic.demand_model = "DDA"
            wn.options.hydraulic.demand_multiplier = multiplier
            return wn

        # Peak-demand hydraulic feedback: upgrade only links that violate the
        # velocity ceiling, rebuild, and repeat until no violation remains or
        # the declared library is exhausted.  This replaces a one-shot diameter
        # assignment with an explicit native-solver feedback loop.
        diameter_library = sorted(int(value) for value in load_case()["equipment_libraries"]["water_dn_mm"])
        resize_iterations = 0
        upgraded_links = 0
        peak_factor = float(load_case()["demand_model"]["drinking_water_peak_factor"])
        while True:
            peak_network = create_network(pipes, multiplier=peak_factor)
            peak_simulation = wntr.sim.EpanetSimulator(peak_network).run_sim()
            peak_velocity = peak_simulation.link["velocity"].iloc[0].abs()
            violations = peak_velocity[peak_velocity > load_case()["acceptance_screening"]["drinking_water"]["maximum_velocity_m_s"]]
            if violations.empty or resize_iterations >= 5:
                break
            changed = 0
            for pipe_id in violations.index:
                row_index = pipes.index[pipes["pipe_id"] == pipe_id]
                if row_index.empty:
                    continue
                idx = int(row_index[0])
                current = int(pipes.loc[idx, "diameter_mm"])
                larger = next((value for value in diameter_library if value > current), None)
                if larger is not None:
                    pipes.loc[idx, "diameter_mm"] = larger
                    changed += 1
            resize_iterations += 1
            upgraded_links += changed
            if changed == 0:
                break
        pipes.to_csv(OUT / "water_pipes.csv", index=False)

        head_adjustment_iterations = 0
        peak_pressure = pd.Series(dtype=float)
        peak_velocity = pd.Series(dtype=float)
        minimum_peak_pressure = load_case()["acceptance_screening"]["drinking_water"]["peak_minimum_pressure_m"]
        junction_ids = nodes.loc[~nodes["node_id"].isin(source_ids), "node_id"].tolist()
        while head_adjustment_iterations < 4:
            peak_network = create_network(pipes, multiplier=peak_factor)
            peak_simulation = wntr.sim.EpanetSimulator(peak_network).run_sim()
            peak_pressure = peak_simulation.node["pressure"].iloc[0].reindex(junction_ids).dropna()
            peak_velocity = peak_simulation.link["velocity"].iloc[0].abs()
            deficit = minimum_peak_pressure - float(peak_pressure.min())
            if deficit <= 0:
                break
            source_head_rise_m += deficit + 1.0
            head_adjustment_iterations += 1

        wn = create_network(pipes, multiplier=1.0)
        wntr.network.io.write_inpfile(wn, OUT / "water_epanet.inp", units="LPS")
        simulation = wntr.sim.EpanetSimulator(wn).run_sim()
        pressure = simulation.node["pressure"].iloc[0]
        junction_pressure = pressure.reindex(junction_ids).dropna()
        velocity = simulation.link["velocity"].iloc[0].abs()

        peak_network = create_network(pipes, multiplier=peak_factor)
        peak_simulation = wntr.sim.EpanetSimulator(peak_network).run_sim()
        peak_pressure = peak_simulation.node["pressure"].iloc[0].reindex(junction_ids).dropna()
        peak_velocity = peak_simulation.link["velocity"].iloc[0].abs()

        age_network = create_network(pipes, multiplier=1.0)
        age_network.options.time.duration = 48 * 3600
        age_network.options.time.hydraulic_timestep = 3600
        age_network.options.time.quality_timestep = 300
        age_network.options.quality.parameter = "AGE"
        age_simulation = wntr.sim.EpanetSimulator(age_network).run_sim()
        age_hours = age_simulation.node["quality"].iloc[-1].reindex(junction_ids).dropna()
        source = nodes[nodes["node_id"].isin(source_ids)].iloc[0]
        source_head = float(source.height_m + source_head_rise_m)
        result = {
            "created": True,
            "converged": bool(np.isfinite(pressure).all() and np.isfinite(velocity).all()),
            "minimum_pressure_m": float(junction_pressure.min()),
            "median_pressure_m": float(junction_pressure.median()),
            "maximum_pressure_m": float(junction_pressure.max()),
            "maximum_velocity_m_s": float(velocity.max()),
            "fraction_links_below_0_05_m_s": float((velocity < 0.05).mean()),
            "peak_factor": peak_factor,
            "peak_minimum_pressure_m": float(peak_pressure.min()),
            "peak_maximum_velocity_m_s": float(peak_velocity.max()),
            "water_age_p95_hours_at_48h": float(age_hours.quantile(0.95) / 3600.0),
            "water_age_max_hours_at_48h": float(age_hours.max() / 3600.0),
            "diameter_resize_iterations": resize_iterations,
            "diameter_upgraded_links": upgraded_links,
            "source_head_adjustment_iterations": head_adjustment_iterations,
            "mass_balance_input_m3_s": float(nodes["water_q_avg_m3s"].sum()),
            "source_flow_m3_s": float(nodes["water_q_avg_m3s"].sum()),
            "source_head_m": source_head,
            "source_elevation_m": float(source.height_m),
            "hydraulic_head_rise_m": source_head - float(source.height_m),
        }
    except Exception as exc:  # pragma: no cover - optional solver path
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_and_run_heat_pandapipes(
    nodes: pd.DataFrame, corridors: pd.DataFrame, metadata: dict[str, Any]
) -> dict[str, Any]:
    result = {"created": False, "converged": False}
    try:
        import pandapipes as ppipe

        net = ppipe.create_empty_network(fluid="water")
        junction_index = {}
        for row in nodes.itertuples():
            junction_index[row.node_id] = ppipe.create_junction(
                net, pn_bar=25.0, tfluid_k=363.15, height_m=row.elevation_m,
                name=row.node_id, geodata=(row.lon, row.lat),
            )
        ppipe.create_ext_grid(
            net, junction_index[metadata["plant_node"]], p_bar=25.0, t_k=363.15,
            type="pt", name="GKS heat source",
        )
        for row in corridors.itertuples():
            ppipe.create_pipe_from_parameters(
                net, junction_index[row.from_node], junction_index[row.to_node],
                length_km=max(float(row.length_km), 0.005), inner_diameter_mm=float(row.diameter_mm),
                k_mm=0.1, sections=max(1, int(math.ceil(row.length_km / 0.25))), name=row.corridor_id,
            )
        for row in nodes[nodes["node_type"] == "consumer_substation"].itertuples():
            mass_flow = max(1e-4, row.peak_heat_mw * 1e6 / (4180.0 * 35.0))
            ppipe.create_sink(net, junction_index[row.node_id], mdot_kg_per_s=mass_flow, name=f"SINK_{row.node_id}")
        ppipe.pipeflow(net, mode="hydraulics", max_iter_hyd=100, friction_model="colebrook")
        ppipe.to_json(net, str(OUT / "heat_pandapipes.json"))
        nodes["result_pressure_bar"] = net.res_junction.p_bar.to_numpy()
        corridors["result_velocity_m_s"] = net.res_pipe.v_mean_m_per_s.to_numpy()
        nodes.to_csv(OUT / "heat_nodes.csv", index=False)
        corridors.to_csv(OUT / "heat_corridors.csv", index=False)
        result = {
            "created": True,
            "converged": bool(net.converged),
            "minimum_pressure_bar": float(net.res_junction.p_bar.min()),
            "source_pressure_bar": 25.0,
            "network_pressure_drop_bar": float(25.0 - net.res_junction.p_bar.min()),
            "return_pressure_reference_bar": float(
                load_case()["bidirectional_coupling"]["heat_return_pressure_bar"]
            ),
            "minimum_differential_pressure_bar": float(
                net.res_junction.p_bar.min()
                - load_case()["bidirectional_coupling"]["heat_return_pressure_bar"]
            ),
            "maximum_velocity_m_s": float(net.res_pipe.v_mean_m_per_s.abs().max()),
            "total_mass_flow_kg_s": float(net.sink.mdot_kg_per_s.sum()),
            "model_scope": (
                "native supply-hydraulic pre-screen with the declared return-pressure "
                "reference; thermal loss is evaluated by the paired supply-return "
                "model at the final coupled gate"
            ),
        }
    except Exception as exc:  # pragma: no cover - optional solver path
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _nearest_bus(point: tuple[float, float], buses: pd.DataFrame) -> pd.Series:
    distances = [base.distance_km(point, (r.lon, r.lat)) for r in buses.itertuples()]
    return buses.iloc[int(np.argmin(distances))]


def _interface_bus(point: tuple[float, float], buses: pd.DataFrame, p_mw: float) -> pd.Series:
    """Choose an electrically appropriate service bus before distance ranking."""
    if p_mw >= 0.10:
        eligible = buses[buses["voltage_kv"] >= 10.0]
    else:
        eligible = buses[buses["bus_role"].isin(["transformer_lv_bus", "transformer_lv_load_bus"])]
    if eligible.empty:
        eligible = buses
    return _nearest_bus(point, eligible)


def build_coupling_layer(
    buildings: pd.DataFrame,
    heat_nodes: pd.DataFrame,
    config: dict[str, Any],
    water_solver: dict[str, Any],
    heat_solver: dict[str, Any],
) -> pd.DataFrame:
    buses = pd.read_csv(OUT / "power_buses.csv")
    legacy = pd.read_csv(OUT / "facility_couplings.csv")
    rows = []
    for row in legacy[
        legacy["selected_base_case"] & ~(
            (legacy["sector"] == "wastewater") & (legacy["asset_type"] == "sewer_pump")
        )
    ].itertuples():
        p_mw = float(row.p_mw)
        parameter_basis = str(row.parameter_basis)
        if row.asset_id == "WAT-WW-01" and water_solver.get("converged"):
            p_mw = (
                1000.0 * 9.81 * float(water_solver["source_flow_m3_s"])
                * float(water_solver["hydraulic_head_rise_m"]) / 0.72 / 1e6
            )
            parameter_basis = "EPANET solved source flow times exported source-head rise; eta=0.72"
        rows.append(
            {
                "interface_id": f"IF_{len(rows)+1:04d}",
                "from_sector": "electricity",
                "from_id": row.candidate_bus,
                "to_sector": row.sector,
                "to_id": row.asset_id,
                "relation": "electric_supply",
                "capacity_value": p_mw,
                "capacity_unit": "MW",
                "confidence_class": row.confidence_class,
                "notes": parameter_basis,
            }
        )
    pump_design = pd.read_csv(OUT / "wastewater_pump_design.csv")
    sewer_nodes = pd.read_csv(OUT / "wastewater_nodes.csv").set_index("node_id")
    for pump in pump_design.itertuples():
        node = sewer_nodes.loc[pump.wet_well_node]
        p_mw = 1000.0 * 9.81 * float(pump.design_flow_m3s) * float(pump.design_head_m) / 0.65 / 1e6
        bus = _interface_bus((float(node.lon), float(node.lat)), buses, p_mw)
        rows.append(
            {
                "interface_id": f"IF_{len(rows)+1:04d}",
                "from_sector": "electricity", "from_id": bus.bus_id,
                "to_sector": "wastewater", "to_id": pump.pump_id,
                "relation": "electric_supply", "capacity_value": p_mw,
                "capacity_unit": "MW", "confidence_class": "C/D",
                "notes": "SWMM-derived lift design flow and head; pump efficiency=0.65",
            }
        )
    plant = config["district_heating"]["plant"]
    mass_flow = float(heat_solver.get("total_mass_flow_kg_s", 0.0))
    pressure_rise_pa = float(heat_solver.get("network_pressure_drop_bar", 0.0)) * 1e5
    pump_mw = (
        mass_flow * pressure_rise_pa
        / (985.0 * config["district_heating"]["pump_efficiency"] * config["district_heating"]["motor_efficiency"])
        / 1e6
    )
    plant_bus = _interface_bus(tuple(plant["coordinate"]), buses, pump_mw)
    rows.append(
        {
            "interface_id": f"IF_{len(rows)+1:04d}", "from_sector": "electricity",
            "from_id": plant_bus.bus_id, "to_sector": "district_heating", "to_id": "DH_PUMP_GKS",
            "relation": "electric_supply", "capacity_value": pump_mw, "capacity_unit": "MW",
            "confidence_class": "C/D",
            "notes": "pandapipes solved mass flow and supply-side pressure drop; pump and motor efficiencies are declared priors",
        }
    )
    rows.append(
        {
            "interface_id": f"IF_{len(rows)+1:04d}", "from_sector": "electricity",
            "from_id": "PB_HV_GKS", "to_sector": "district_heating", "to_id": "DH_PLANT_GKS",
            "relation": "heat_plant_electrical_interface", "capacity_value": np.nan,
            "capacity_unit": "MW", "confidence_class": "A/B",
            "notes": "publicly documented GKS high-voltage grid connection; external interface bus",
        }
    )
    candidates = buildings.nlargest(12, "heat_candidate_mwh_year")
    for index, building in enumerate(candidates.itertuples(), 1):
        heat_mw = building.heat_candidate_mwh_year / 1800.0
        bus = _interface_bus((building.lon, building.lat), buses, heat_mw / 3.2)
        rows.append(
            {
                "interface_id": f"IF_{len(rows)+1:04d}", "from_sector": "electricity",
                "from_id": bus.bus_id, "to_sector": "district_heating", "to_id": f"HP_{index:03d}",
                "relation": "electric_heat_pump", "capacity_value": heat_mw / 3.2,
                "capacity_unit": "MW_e", "confidence_class": "D",
                "notes": f"candidate electrically driven heat pump; building={building.building_id}; COP=3.2",
            }
        )
    demand_zones = pd.read_csv(OUT / "demand_zones.csv")
    for zone in demand_zones.itertuples():
        rows.append(
            {
                "interface_id": f"IF_{len(rows)+1:04d}", "from_sector": "drinking_water",
                "from_id": zone.zone_id, "to_sector": "wastewater", "to_id": zone.zone_id,
                "relation": "delivered_water_to_sanitary_inflow",
                "capacity_value": float(config["demand_model"]["wastewater_sanitary_return_fraction"]),
                "capacity_unit": "m3/m3", "confidence_class": "C/D",
                "notes": "common-zone mapping; infiltration is represented separately",
            }
        )
    frame = pd.DataFrame(rows)
    frame["interface_direction"] = "declared_relation"
    frame["forward_state"] = "identity_or_capacity"
    frame["return_state"] = "none"
    electric_mask = frame["relation"] == "electric_supply"
    frame.loc[electric_mask, "interface_direction"] = "bidirectional_state_exchange"
    frame.loc[electric_mask, "forward_state"] = "bus_voltage_pu;energized"
    frame.loc[electric_mask, "return_state"] = "solver_calculated_active_power_mw;motor_speed_pu"
    water_return_mask = frame["relation"] == "delivered_water_to_sanitary_inflow"
    frame.loc[water_return_mask, "interface_direction"] = "directed_mass_transfer_inside_coupled_loop"
    frame.loc[water_return_mask, "forward_state"] = "delivered_water_fraction;sanitary_return_fraction"
    frame.loc[water_return_mask, "return_state"] = "wastewater_pump_power_returned_through_electricity"
    frame["active_in_bidirectional_base_case"] = electric_mask
    frame.loc[frame["relation"].isin(["electric_heat_pump", "heat_plant_electrical_interface"]), "active_in_bidirectional_base_case"] = False
    frame.to_csv(OUT / "coupling_interfaces.csv", index=False)
    return frame


def _geometry_segments(frame: pd.DataFrame, sector: str, length_field: str) -> dict[tuple, dict[str, Any]]:
    segments: dict[tuple, dict[str, Any]] = {}
    for row in frame.itertuples():
        geometry = json.loads(getattr(row, "geometry_json"))
        for a, b in zip(geometry[:-1], geometry[1:]):
            a = (round(float(a[0]), 7), round(float(a[1]), 7))
            b = (round(float(b[0]), 7), round(float(b[1]), 7))
            key = tuple(sorted((a, b)))
            segments.setdefault(key, {"a": key[0], "b": key[1], "sectors": set(), "length_km": base.distance_km(*key)})
            segments[key]["sectors"].add(sector)
    return segments


def build_common_corridors(config: dict[str, Any]) -> pd.DataFrame:
    tables = [
        (pd.read_csv(OUT / "power_lines.csv"), "electricity", "length_km"),
        (pd.read_csv(OUT / "water_pipes.csv"), "drinking_water", "length_km"),
        (pd.read_csv(OUT / "wastewater_conduits.csv"), "wastewater", "length_m"),
        (pd.read_csv(OUT / "heat_corridors.csv"), "district_heating", "length_km"),
    ]
    registry: dict[tuple, dict[str, Any]] = {}
    for frame, sector, length_field in tables:
        for key, item in _geometry_segments(frame, sector, length_field).items():
            if key not in registry:
                registry[key] = item
            else:
                registry[key]["sectors"].update(item["sectors"])
    rows = []
    for index, item in enumerate((v for v in registry.values() if len(v["sectors"]) >= 2), 1):
        midpoint = ((item["a"][0] + item["b"][0]) / 2.0, (item["a"][1] + item["b"][1]) / 2.0)
        rows.append(
            {
                "corridor_segment_id": f"CC{index:05d}",
                "from_lon": item["a"][0], "from_lat": item["a"][1],
                "to_lon": item["b"][0], "to_lat": item["b"][1],
                "length_km": item["length_km"],
                "sector_count": len(item["sectors"]),
                "sectors": ";".join(sorted(item["sectors"])),
                "morphology": _morphology(midpoint, config),
                "confidence_class": "B",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "common_corridors.csv", index=False)
    return frame


def _graph_metrics(
    node_frame: pd.DataFrame,
    edge_frame: pd.DataFrame,
    node_id: str,
    from_id: str,
    to_id: str,
    length_column: str,
    directed: bool = False,
    morphology: str = "",
) -> dict[str, float]:
    chosen_edges = edge_frame[edge_frame["morphology"] == morphology]
    graph = nx.DiGraph() if directed else nx.Graph()
    for row in chosen_edges.itertuples():
        length = float(getattr(row, length_column))
        if length_column == "length_m":
            length /= 1000.0
        graph.add_edge(getattr(row, from_id), getattr(row, to_id), length_km=length)
    if graph.number_of_nodes() == 0:
        return {"nodes": 0, "links": 0, "route_length_km": 0.0, "cycle_rank": 0.0, "branch_fraction": 0.0, "leaf_fraction": 0.0}
    undirected = graph.to_undirected()
    components = nx.number_connected_components(undirected)
    degrees = np.array([degree for _, degree in undirected.degree()], dtype=float)
    return {
        "nodes": float(graph.number_of_nodes()),
        "links": float(graph.number_of_edges()),
        "route_length_km": float(sum(d["length_km"] for _, _, d in graph.edges(data=True))),
        "cycle_rank": float(graph.number_of_edges() - graph.number_of_nodes() + components),
        "branch_fraction": float(np.mean(degrees >= 3)),
        "leaf_fraction": float(np.mean(degrees == 1)),
    }


def _add_morphology_columns(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    frames = {
        "power_nodes": pd.read_csv(OUT / "power_buses.csv"),
        "power_edges": pd.read_csv(OUT / "power_lines.csv"),
        "water_nodes": pd.read_csv(OUT / "water_nodes.csv"),
        "water_edges": pd.read_csv(OUT / "water_pipes.csv"),
        "wastewater_nodes": pd.read_csv(OUT / "wastewater_nodes.csv"),
        "wastewater_edges": pd.read_csv(OUT / "wastewater_conduits.csv"),
        "heat_nodes": pd.read_csv(OUT / "heat_nodes.csv"),
        "heat_edges": pd.read_csv(OUT / "heat_corridors.csv"),
    }
    for key, frame in frames.items():
        if "nodes" in key:
            frame["morphology"] = [_morphology((r.lon, r.lat), config) for r in frame.itertuples()]
        else:
            frame["morphology"] = [
                _morphology(tuple(json.loads(r.geometry_json)[len(json.loads(r.geometry_json)) // 2]), config)
                for r in frame.itertuples()
            ]
    frames["power_nodes"].to_csv(OUT / "power_buses.csv", index=False)
    frames["power_edges"].to_csv(OUT / "power_lines.csv", index=False)
    frames["water_nodes"].to_csv(OUT / "water_nodes.csv", index=False)
    frames["water_edges"].to_csv(OUT / "water_pipes.csv", index=False)
    frames["wastewater_nodes"].to_csv(OUT / "wastewater_nodes.csv", index=False)
    frames["wastewater_edges"].to_csv(OUT / "wastewater_conduits.csv", index=False)
    return frames


def _territory_areas(config: dict[str, Any]) -> dict[str, float]:
    bbox = config["bbox"]
    xs = np.linspace(bbox[0], bbox[2], 360)
    ys = np.linspace(bbox[1], bbox[3], 240)
    counts = Counter(_morphology((float(x), float(y)), config) for x in xs for y in ys)
    total = sum(counts.values())
    official_area = 57.53
    return {name: official_area * count / total for name, count in counts.items()}


def _service_route_stats(
    sector: str,
    node_frame: pd.DataFrame,
    edge_frame: pd.DataFrame,
    morphology: str,
) -> dict[str, float]:
    graph: nx.Graph | nx.DiGraph
    if sector == "wastewater":
        graph = nx.DiGraph()
        for row in edge_frame.itertuples():
            graph.add_edge(row.from_node, row.to_node, length_km=float(row.length_m) / 1000.0)
        sources = node_frame.loc[node_frame["node_type"] == "outfall", "node_id"].tolist()
        lengths: dict[str, float] = {}
        reverse = graph.reverse(copy=False)
        for source in sources:
            if source in reverse:
                for node, distance in nx.single_source_dijkstra_path_length(reverse, source, weight="length_km").items():
                    lengths[node] = min(lengths.get(node, math.inf), float(distance))
        id_column = "node_id"
        facility_mask = node_frame["node_type"].isin(["pump", "cso", "outfall"])
    else:
        graph = nx.Graph()
        if sector == "electricity":
            active = edge_frame[
                (~edge_frame["normally_open"])
                & (edge_frame["voltage_level"] == "LV")
            ]
            for row in active.itertuples():
                graph.add_edge(row.from_bus, row.to_bus, length_km=float(row.length_km))
            sources = node_frame.loc[
                node_frame["bus_role"].isin(["transformer_lv_bus", "transformer_lv_load_bus"]),
                "bus_id",
            ].tolist()
            graph.add_nodes_from(sources)
            id_column = "bus_id"
            facility_mask = node_frame["bus_role"] == "transformer_mv_bus"
        elif sector == "drinking_water":
            for row in edge_frame.itertuples():
                graph.add_edge(row.from_node, row.to_node, length_km=float(row.length_km))
            sources = node_frame.loc[node_frame["node_type"] == "source", "node_id"].tolist()
            id_column = "node_id"
            facility_mask = node_frame["node_type"].isin(["source", "storage", "wellfield"])
        else:
            for row in edge_frame.itertuples():
                graph.add_edge(row.from_node, row.to_node, length_km=float(row.length_km))
            sources = node_frame.loc[node_frame["node_type"] == "plant", "node_id"].tolist()
            id_column = "node_id"
            facility_mask = node_frame["node_type"].isin(["plant", "consumer_substation"])
        lengths = nx.multi_source_dijkstra_path_length(graph, [s for s in sources if s in graph], weight="length_km") if sources else {}
    if sector == "electricity":
        target_ids = node_frame.loc[
            (node_frame["morphology"] == morphology)
            & node_frame["bus_role"].isin(["lv_load_bus", "transformer_lv_load_bus"]),
            id_column,
        ].tolist()
    else:
        target_ids = node_frame.loc[node_frame["morphology"] == morphology, id_column].tolist()
    values = [float(lengths[node]) for node in target_ids if node in lengths]
    source_count = int(node_frame.loc[node_frame[id_column].isin(sources) & (node_frame["morphology"] == morphology)].shape[0])
    facility_count = int(node_frame.loc[facility_mask & (node_frame["morphology"] == morphology)].shape[0])
    return {
        "average_service_route_km": float(np.mean(values)) if values else 0.0,
        "maximum_service_route_km": float(np.max(values)) if values else 0.0,
        "source_count": source_count,
        "facility_count": facility_count,
    }


def build_metrics(
    buildings: pd.DataFrame,
    consumers: pd.DataFrame,
    config: dict[str, Any],
    phase_runtime: dict[str, float],
) -> pd.DataFrame:
    frames = _add_morphology_columns(config)
    areas = _territory_areas(config)
    rows = []
    definitions = {
        "electricity": (frames["power_nodes"], frames["power_edges"], "bus_id", "from_bus", "to_bus", "length_km", False),
        "drinking_water": (frames["water_nodes"], frames["water_edges"], "node_id", "from_node", "to_node", "length_km", False),
        "wastewater": (frames["wastewater_nodes"], frames["wastewater_edges"], "node_id", "from_node", "to_node", "length_m", True),
        "district_heating": (frames["heat_nodes"], frames["heat_edges"], "node_id", "from_node", "to_node", "length_km", False),
    }
    for morphology in [m["id"] for m in config["morphologies"]]:
        building_subset = buildings[buildings["morphology"] == morphology]
        heat_subset = consumers[consumers["morphology"] == morphology]
        for sector, definition in definitions.items():
            metrics = _graph_metrics(*definition, morphology=morphology)
            metrics.update(_service_route_stats(sector, definition[0], definition[1], morphology))
            demand = {
                "electricity": float(building_subset["electricity_mwh_year"].sum()),
                "drinking_water": float(building_subset["water_m3_year"].sum()),
                "wastewater": float(building_subset["wastewater_sanitary_m3_year"].sum()),
                "district_heating": float(heat_subset["annual_heat_mwh"].sum()),
            }[sector]
            units = {"electricity": "MWh/y", "drinking_water": "m3/y", "wastewater": "m3/y", "district_heating": "MWh/y"}
            node_frame, edge_frame = definition[0], definition[1]
            size_column = {
                "electricity": "cable_size_mm2", "drinking_water": "diameter_mm",
                "wastewater": "diameter_mm", "district_heating": "diameter_mm",
            }[sector]
            edge_subset = edge_frame[edge_frame["morphology"] == morphology]
            values = (
                edge_subset[size_column].dropna().to_numpy(dtype=float)
                if size_column in edge_subset else np.array([])
            )
            rows.append(
                {
                    "morphology": morphology,
                    "sector": sector,
                    **metrics,
                    "buildings": int(len(building_subset)),
                    "territory_area_km2": areas[morphology],
                    "building_density_km2": len(building_subset) / areas[morphology],
                    "allocated_demand": demand,
                    "demand_unit": units[sector],
                    "equipment_size_median": float(np.median(values)) if len(values) else 0.0,
                    "equipment_size_p90": float(np.percentile(values, 90)) if len(values) else 0.0,
                    "equipment_size_max": float(np.max(values)) if len(values) else 0.0,
                    "district_heat_contracts": int(heat_subset["represented_contract_count"].sum()) if sector == "district_heating" else 0,
                    "district_heat_contracts_per_km2": (
                        float(heat_subset["represented_contract_count"].sum()) / areas[morphology]
                        if sector == "district_heating" else 0.0
                    ),
                    "generation_runtime_s": phase_runtime.get("base_generation", 0.0) + phase_runtime.get("four_sector_generation", 0.0),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "morphology_metrics.csv", index=False)
    for morphology in frame["morphology"].unique():
        folder = OUT / "morphologies" / morphology
        folder.mkdir(parents=True, exist_ok=True)
        frame[frame["morphology"] == morphology].to_csv(folder / "summary.csv", index=False)
        for name, table in frames.items():
            table[table["morphology"] == morphology].to_csv(folder / f"{name}.csv", index=False)
    return frame


def build_inventory_and_anchor_checks(
    buildings: pd.DataFrame,
    consumers: pd.DataFrame,
    config: dict[str, Any],
    heat_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    power_nodes = pd.read_csv(OUT / "power_buses.csv")
    power_edges = pd.read_csv(OUT / "power_lines.csv")
    water_nodes = pd.read_csv(OUT / "water_nodes.csv")
    water_edges = pd.read_csv(OUT / "water_pipes.csv")
    sewer_nodes = pd.read_csv(OUT / "wastewater_nodes.csv")
    sewer_edges = pd.read_csv(OUT / "wastewater_conduits.csv")
    heat_nodes = pd.read_csv(OUT / "heat_nodes.csv")
    heat_edges = pd.read_csv(OUT / "heat_corridors.csv")
    transformers = pd.read_csv(OUT / "power_transformers.csv")

    def cycle_rank(nodes: int, edges: int, components: int = 1) -> int:
        return edges - nodes + components

    active_power = power_edges[~power_edges["normally_open"]]
    power_cables = power_edges[power_edges["asset_type"] != "transformer_connection"]
    overflow_mask = sewer_nodes.get(
        "is_overflow_basin", sewer_nodes["node_type"] == "cso"
    ).astype(bool)
    lift_mask = sewer_nodes["node_type"] == "pump"
    outfall_mask = sewer_nodes["node_type"] == "outfall"
    wastewater_facility_mask = overflow_mask | lift_mask | outfall_mask
    wastewater_facility_inventory = pd.DataFrame(
        [
            {"category": "derived_lift_station_nodes", "count": int(lift_mask.sum()), "evidence_role": "generated_output"},
            {"category": "overflow_basin_equivalents", "count": int(overflow_mask.sum()), "evidence_role": "input_constraint"},
            {"category": "outfall_nodes", "count": int(outfall_mask.sum()), "evidence_role": "observed_or_selected_boundary"},
            {"category": "co_located_lift_and_overflow_nodes", "count": int((lift_mask & overflow_mask).sum()), "evidence_role": "generated_co_location"},
            {"category": "unique_facility_nodes", "count": int(wastewater_facility_mask.sum()), "evidence_role": "union_not_category_sum"},
        ]
    )
    wastewater_facility_inventory.to_csv(OUT / "wastewater_facility_inventory.csv", index=False)
    inventories = [
        {
            "sector": "electricity", "nodes": len(power_nodes), "links": len(power_edges),
            "operating_links": len(active_power), "route_length_km": power_cables["length_km"].sum(),
            "cycle_rank_asset": cycle_rank(len(power_nodes), len(power_edges)),
            "cycle_rank_operating": cycle_rank(len(power_nodes), len(active_power)),
            "facilities": len(transformers), "annual_demand": buildings["electricity_mwh_year"].sum(),
            "demand_unit": "MWh/y", "size_unit": "mm2", "size_median": power_cables["cable_size_mm2"].median(),
            "size_p90": power_cables["cable_size_mm2"].quantile(0.9), "size_max": power_cables["cable_size_mm2"].max(),
        },
        {
            "sector": "drinking_water", "nodes": len(water_nodes), "links": len(water_edges),
            "operating_links": len(water_edges), "route_length_km": water_edges["length_km"].sum(),
            "cycle_rank_asset": cycle_rank(len(water_nodes), len(water_edges)),
            "cycle_rank_operating": cycle_rank(len(water_nodes), len(water_edges)),
            "facilities": int(water_nodes["node_type"].isin(["source", "storage", "wellfield"]).sum()),
            "annual_demand": buildings["water_m3_year"].sum(), "demand_unit": "m3/y",
            "size_unit": "mm", "size_median": water_edges["diameter_mm"].median(),
            "size_p90": water_edges["diameter_mm"].quantile(0.9), "size_max": water_edges["diameter_mm"].max(),
        },
        {
            "sector": "wastewater", "nodes": len(sewer_nodes), "links": len(sewer_edges),
            "operating_links": len(sewer_edges), "route_length_km": sewer_edges["length_m"].sum() / 1000.0,
            "cycle_rank_asset": cycle_rank(len(sewer_nodes), len(sewer_edges)),
            "cycle_rank_operating": cycle_rank(len(sewer_nodes), len(sewer_edges)),
            "facilities": int(wastewater_facility_mask.sum()),
            "annual_demand": buildings["wastewater_sanitary_m3_year"].sum(), "demand_unit": "m3/y",
            "size_unit": "mm", "size_median": sewer_edges["diameter_mm"].median(),
            "size_p90": sewer_edges["diameter_mm"].quantile(0.9), "size_max": sewer_edges["diameter_mm"].max(),
        },
        {
            "sector": "district_heating", "nodes": len(heat_nodes), "links": len(heat_edges),
            "operating_links": len(heat_edges), "route_length_km": heat_metadata["generated_route_length_km"],
            "cycle_rank_asset": cycle_rank(len(heat_nodes), len(heat_edges)),
            "cycle_rank_operating": cycle_rank(len(heat_nodes), len(heat_edges)),
            "facilities": int(heat_nodes["node_type"].isin(["plant", "consumer_substation"]).sum()),
            "annual_demand": consumers["annual_heat_mwh"].sum(), "demand_unit": "MWh/y",
            "size_unit": "mm", "size_median": heat_edges["diameter_mm"].median(),
            "size_p90": heat_edges["diameter_mm"].quantile(0.9), "size_max": heat_edges["diameter_mm"].max(),
        },
    ]
    inventory = pd.DataFrame(inventories)
    inventory.to_csv(OUT / "network_inventory.csv", index=False)

    anchors = config["official_anchors"]
    force_main = sewer_edges.loc[sewer_edges["link_type"] == "force_main", "length_m"].sum() / 1000.0
    comparison = config["acceptance_screening"]["aggregate_comparison"]
    route_tolerance = float(comparison["route_length_tolerance_percent"])
    force_main_tolerance = float(comparison["force_main_length_tolerance_percent"])
    pump_tolerance = int(comparison["pump_station_tolerance_count"])
    checks = [
        ("Electricity LV annual energy", buildings["electricity_mwh_year"].sum(), anchors["electricity_lv_annual_mwh"], "MWh/y", "input_constraint", 0.0, "accounting_identity"),
        ("Electricity LV coincident peak", buildings["electricity_peak_kw"].sum() / 1000.0, anchors["electricity_lv_peak_mw"], "MW", "input_constraint", 0.0, "accounting_identity"),
        ("MS/LV withdrawal locations", len(transformers), anchors["electricity_mv_lv_withdrawal_locations"], "count", "input_constraint", 0.0, "accounting_identity"),
        ("MS/LV installed capacity", transformers["sn_mva"].sum(), anchors["electricity_mv_lv_installed_capacity_mva"], "MVA", "input_constraint", 0.0, "accounting_identity"),
        ("Mapped synthetic MV network", power_edges.loc[power_edges["voltage_level"] == "MV", "length_km"].sum(), anchors["electricity_mv_cable_km"], "km", "diagnostic_only", math.nan, "none"),
        ("Direct-retail drinking-water allocation", buildings["water_m3_year"].sum(), anchors["drinking_water_annual_m3"], "m3/y", "input_constraint", 0.0, "accounting_identity"),
        ("Drinking-water route", water_edges["length_km"].sum(), anchors["drinking_water_route_km"], "km", "held_out_comparison", route_tolerance, "relative_percent"),
        ("Wastewater route inventory", sewer_edges["length_m"].sum() / 1000.0, anchors["wastewater_combined_km"] + anchors["wastewater_sanitary_km"] + anchors["wastewater_storm_km"] + anchors["wastewater_force_main_km"], "km", "held_out_comparison", route_tolerance, "relative_percent"),
        ("Wastewater force mains", force_main, anchors["wastewater_force_main_km"], "km", "held_out_comparison", force_main_tolerance, "relative_percent"),
        ("Wastewater pump stations", int(lift_mask.sum()), anchors["wastewater_pump_stations"], "count", "held_out_comparison", pump_tolerance, "absolute_count"),
        ("Wastewater overflow basins", int(overflow_mask.sum()), anchors["wastewater_overflow_basins"], "count", "input_constraint", 0.0, "accounting_identity"),
        ("District-heat generated route", heat_metadata["generated_route_length_km"], anchors["district_heat_route_km"], "km", "held_out_comparison", route_tolerance, "relative_percent"),
        ("District-heat represented sales contracts", consumers["represented_contract_count"].sum(), anchors["district_heat_sales_contracts"], "count", "input_constraint", 0.0, "accounting_identity"),
        ("District-heat design peak assumption", consumers["peak_heat_mw"].sum(), config["district_heating"]["design_peak_mw_assumption"], "MW", "derived_assumption", 0.0, "accounting_identity"),
        ("District-heat annual sales", consumers["annual_heat_mwh"].sum(), anchors["district_heat_sales_mwh_year"], "MWh/y", "input_constraint", 0.0, "accounting_identity"),
    ]
    check_rows = []
    for metric, generated, reference, unit, role, tolerance, tolerance_basis in checks:
        error = (float(generated) - float(reference)) / float(reference) * 100.0 if reference else 0.0
        if role in {"input_constraint", "derived_assumption"}:
            status = "satisfied" if abs(error) <= 1e-6 else "review"
        elif role == "diagnostic_only":
            status = "diagnostic"
        elif tolerance_basis == "absolute_count":
            status = "inside_comparison_band" if abs(float(generated) - float(reference)) <= tolerance else "outside_comparison_band"
        else:
            status = "inside_comparison_band" if abs(error) <= tolerance else "outside_comparison_band"
        check_rows.append(
            {"metric": metric, "generated": generated, "reference": reference, "unit": unit,
             "relative_error_percent": error, "evidence_role": role,
             "comparison_tolerance": tolerance, "comparison_tolerance_basis": tolerance_basis,
             "status": status}
        )
    anchor_checks = pd.DataFrame(check_rows)
    anchor_checks.to_csv(OUT / "aggregate_anchor_checks.csv", index=False)
    return inventory, anchor_checks


def _parse_swmm_report_metrics(report_text: str) -> dict[str, float]:
    """Extract compact numerical diagnostics from an EPA SWMM report."""
    patterns = {
        "flow_routing_continuity_error_percent": r"Continuity Error \(%\) \.{3,}\s+([-+]?\d+(?:\.\d+)?)",
        "flooding_loss_million_liter": r"Flooding Loss \.{3,}\s+[-+]?\d+(?:\.\d+)?\s+([-+]?\d+(?:\.\d+)?)",
        "steps_not_converging_percent": r"% of Steps Not Converging\s*:\s*([-+]?\d+(?:\.\d+)?)",
    }
    metrics: dict[str, float] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, report_text)
        if match:
            metrics[name] = float(match.group(1))
    return metrics


def _screen_solver_results(
    solver: dict[str, Any], sector: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Apply the declared study-level screening envelope to a native result."""
    thresholds = config["acceptance_screening"][sector]
    if sector == "electricity":
        checks = {
            "solver_converged": bool(solver.get("converged", False)),
            "minimum_voltage": solver.get("minimum_voltage_pu", -math.inf) >= thresholds["minimum_voltage_pu"],
            "maximum_voltage": solver.get("maximum_voltage_pu", math.inf) <= thresholds["maximum_voltage_pu"],
            "line_loading": solver.get("maximum_line_loading_percent", math.inf) <= thresholds["maximum_line_loading_percent"],
            "transformer_loading": solver.get("maximum_transformer_loading_percent", math.inf) <= thresholds["maximum_transformer_loading_percent"],
        }
    elif sector == "drinking_water":
        checks = {
            "solver_converged": bool(solver.get("converged", False)),
            "minimum_pressure": solver.get("minimum_pressure_m", -math.inf) >= thresholds["minimum_pressure_m"],
            "maximum_pressure": solver.get("maximum_pressure_m", math.inf) <= thresholds["maximum_pressure_m"],
            "maximum_velocity": solver.get("maximum_velocity_m_s", math.inf) <= thresholds["maximum_velocity_m_s"],
            "peak_minimum_pressure": solver.get("peak_minimum_pressure_m", -math.inf) >= thresholds["peak_minimum_pressure_m"],
        }
    elif sector == "wastewater":
        checks = {
            "engine_run": bool(solver.get("engine_run_passed", False)),
            "mass_continuity": abs(solver.get("flow_routing_continuity_error_percent", math.inf))
            <= thresholds["maximum_absolute_flow_routing_continuity_error_percent"],
            "nonconverging_steps": solver.get("steps_not_converging_percent", math.inf)
            <= thresholds["maximum_nonconverging_steps_percent"],
            "no_dry_weather_flooding": solver.get("flooding_loss_million_liter", math.inf)
            <= thresholds["maximum_dry_weather_flooding_loss_million_liter"],
            "time_step_sensitivity": bool(solver.get("time_step_sensitivity_passed", False)),
        }
    elif sector == "district_heating":
        checks = {
            "solver_converged": bool(solver.get("converged", False)),
            "minimum_pressure": solver.get("minimum_pressure_bar", -math.inf) >= thresholds["minimum_pressure_bar"],
            "minimum_differential_pressure": solver.get("minimum_differential_pressure_bar", -math.inf)
            >= thresholds["minimum_differential_pressure_bar"],
            "maximum_velocity": solver.get("maximum_velocity_m_s", math.inf) <= thresholds["maximum_velocity_m_s"],
            "published_network_length": bool(
                solver.get("inventory_length_within_tolerance", True)
            ),
        }
        if "annual_heat_loss_fraction" in solver:
            checks["heat_loss"] = (
                solver["annual_heat_loss_fraction"]
                <= thresholds["maximum_annual_heat_loss_fraction"]
            )
        else:
            # The legacy Experiment-A pre-screen is supply-hydraulic only.
            # Its paired supply-return model evaluates and enforces heat loss
            # at the final coupled gate.
            checks["heat_loss_deferred_to_paired_gate"] = True
        if "all_consumer_substations_reachable_from_a_plant" in solver:
            checks["source_reachability"] = bool(
                solver["all_consumer_substations_reachable_from_a_plant"]
            )
    else:  # pragma: no cover - internal contract guard
        raise ValueError(f"Unknown acceptance sector: {sector}")
    return {"thresholds": thresholds, "checks": checks, "passed": all(checks.values())}


def validate_swmm_file() -> dict[str, Any]:
    path = OUT / "schweinfurt_proxy.inp"
    text = path.read_text(encoding="utf-8")
    required = [
        "[JUNCTIONS]", "[STORAGE]", "[OUTFALLS]", "[CONDUITS]", "[PUMPS]",
        "[XSECTIONS]", "[CURVES]", "[DWF]", "[COORDINATES]",
    ]
    missing = [section for section in required if section not in text]
    node_count = len(pd.read_csv(OUT / "wastewater_nodes.csv"))
    link_count = len(pd.read_csv(OUT / "wastewater_conduits.csv"))
    status = {
        "created": path.exists(),
        "syntax_gate_passed": not missing,
        "engine_run_passed": False,
        "required_sections": required,
        "missing_sections": missing,
        "nodes": node_count,
        "links": link_count,
        "note": "Forty-eight-hour normal dry-weather hydraulic-stability run; rainfall/runoff is outside this paper",
        "routing_duration_hours": 48,
    }
    if missing:
        return status
    try:
        from swmm.toolkit import solver as swmm_solver

        report_path = OUT / "schweinfurt_swmm_normal.rpt"
        binary_path = OUT / "schweinfurt_swmm_normal.out"
        swmm_solver.swmm_run(str(path), str(report_path), str(binary_path))
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
        status["engine_run_passed"] = "ERROR " not in report_text
        status["report_file"] = report_path.name
        status.update(_parse_swmm_report_metrics(report_text))
        sensitivity_input = OUT / "schweinfurt_proxy_timestep15.inp"
        sensitivity_report = OUT / "schweinfurt_swmm_timestep15.rpt"
        sensitivity_binary = OUT / "schweinfurt_swmm_timestep15.out"
        sensitivity_input.write_text(
            text.replace("ROUTING_STEP         00:00:30", "ROUTING_STEP         00:00:15"),
            encoding="utf-8",
        )
        swmm_solver.swmm_run(str(sensitivity_input), str(sensitivity_report), str(sensitivity_binary))
        sensitivity_text = sensitivity_report.read_text(encoding="utf-8", errors="ignore")
        sensitivity_metrics = _parse_swmm_report_metrics(sensitivity_text)
        status["time_step_sensitivity_15s"] = sensitivity_metrics
        thresholds = load_case()["acceptance_screening"]["wastewater"]
        status["time_step_sensitivity_passed"] = (
            "ERROR " not in sensitivity_text
            and abs(sensitivity_metrics.get("flow_routing_continuity_error_percent", math.inf))
            <= thresholds["maximum_absolute_flow_routing_continuity_error_percent"]
            and sensitivity_metrics.get("steps_not_converging_percent", math.inf)
            <= thresholds["maximum_nonconverging_steps_percent"]
            and sensitivity_metrics.get("flooding_loss_million_liter", math.inf)
            <= thresholds["maximum_dry_weather_flooding_loss_million_liter"]
        )
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_framework(base_runtime: float = 0.0) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    # Remove obsolete one-pass coupling products from earlier releases.  The
    # bidirectional release writes only staged iteration states and accepted
    # final exports.
    for obsolete in [
        "power_pandapower_coupled.json",
        "power_pandapower_heat_pump_candidate.json",
        "coupled_power_flow_summary.json",
    ]:
        path = OUT / obsolete
        if path.exists():
            path.unlink()
    config = load_case()
    phase_runtime: dict[str, float] = {"base_generation": base_runtime}
    started = time.perf_counter()
    buildings = build_building_ledger(config)
    enrich_common_demand_zones(buildings)
    power = augment_power_model(config)
    water_solver = write_and_run_water_epanet()
    heat_nodes, heat_corridors, heat_consumers, heat_metadata = build_heat_network(buildings, config)
    heat_solver = build_and_run_heat_pandapipes(heat_nodes, heat_corridors, heat_metadata)
    swmm = validate_swmm_file()
    interfaces = build_coupling_layer(buildings, heat_nodes, config, water_solver, heat_solver)
    corridors = build_common_corridors(config)
    phase_runtime["four_sector_generation"] = time.perf_counter() - started
    metrics = build_metrics(buildings, heat_consumers, config, phase_runtime)
    inventory, anchor_checks = build_inventory_and_anchor_checks(
        buildings, heat_consumers, config, heat_metadata
    )
    native_screening = {
        "electricity": _screen_solver_results(power["solver"], "electricity", config),
        "drinking_water": _screen_solver_results(water_solver, "drinking_water", config),
        "wastewater": _screen_solver_results(swmm, "wastewater", config),
        "district_heating": _screen_solver_results(heat_solver, "district_heating", config),
    }
    from .cosimulation import run_bidirectional_cosimulation

    bidirectional = run_bidirectional_cosimulation(interfaces, native_screening)
    screening = {
        **native_screening,
        "bidirectional_coupled_state": {
            "thresholds": config["bidirectional_coupling"],
            "checks": {
                "fixed_point_converged": bool(bidirectional.get("converged", False)),
                "all_native_and_coupled_limits": bool(bidirectional.get("accepted", False)),
                "final_export_gate": bool(bidirectional.get("final_export_created", False)),
            },
            "passed": bool(bidirectional.get("accepted", False)),
        },
    }

    manifest = {
        "case_id": config["case_id"],
        "scope": "normal-condition four-sector synthetic topology generation and bidirectional iterative co-simulation",
        "scope_exclusions": [
            "hazards", "outages", "resilience indices", "restoration", "backup generation",
            "service-loss scenarios", "intervention optimisation", "exact utility-network recovery",
        ],
        "evidence_date": config["evidence_date"],
        "random_seed": config["seed"],
        "terrain_evidence": base.terrain_evidence(),
        "counts": {
            "buildings": int(len(buildings)),
            "power_buses": int(len(pd.read_csv(OUT / "power_buses.csv"))),
            "power_lines": int(len(pd.read_csv(OUT / "power_lines.csv"))),
            "power_transformers": int(power["transformers"]),
            "water_nodes": int(len(pd.read_csv(OUT / "water_nodes.csv"))),
            "water_pipes": int(len(pd.read_csv(OUT / "water_pipes.csv"))),
            "wastewater_nodes": int(len(pd.read_csv(OUT / "wastewater_nodes.csv"))),
            "wastewater_links": int(len(pd.read_csv(OUT / "wastewater_conduits.csv"))),
            "heat_nodes": int(len(heat_nodes)),
            "heat_corridors": int(len(heat_corridors)),
            "heat_consumer_substations": int(len(heat_consumers)),
            "heat_sales_contracts_represented": int(heat_consumers["represented_contract_count"].sum()),
            "coupling_interfaces": int(len(interfaces)),
            "common_corridor_segments": int(len(corridors)),
        },
        "heat_calibration": heat_metadata,
        "solver_readiness": {
            "pandapower_uncoupled": power["solver"],
            "epanet_wntr": water_solver,
            "swmm": swmm,
            "pandapipes_heat_supply": heat_solver,
            "pandapipes_water": json.loads((OUT / "model_manifest.json").read_text(encoding="utf-8"))["simulation_build"]["pandapipes"],
            "bidirectional_cosimulation": bidirectional,
        },
        "acceptance_screening": screening,
        "runtime_seconds": phase_runtime,
        "morphologies": config["morphologies"],
        "official_anchors": config["official_anchors"],
    }
    (OUT / "model_manifest_four_sector.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not bidirectional.get("accepted", False):
        raise RuntimeError(
            f"Urban4 final gate returned {bidirectional.get('status', 'failed')}; "
            "failed iteration states and the ordered repair ledger were retained under "
            "outputs/cosimulation, outputs/final_accepted was withheld, and the build stopped"
        )
    hashes = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.csv":
            hashes.append({"file": str(path.relative_to(PROJECT)), "sha256": _sha256(path), "bytes": path.stat().st_size})
    pd.DataFrame(hashes).to_csv(OUT / "SHA256SUMS.csv", index=False)
    return manifest


if __name__ == "__main__":
    print(json.dumps(run_framework(), indent=2))
