#!/usr/bin/env python3
"""Validated InfDB/pylovo evidence adapter for Urban4.

The adapter deliberately consumes a small, versioned CSV/GeoJSON contract
instead of importing the InfDB PostgreSQL database into Urban4.  InfDB remains
the source of buildings, census attributes, admissible road corridors and
building-to-road connection lines; pylovo remains the source of building-
resolved LV networks.  Urban4 owns the other sector generators and the common
interface ledger.

All geometries crossing the contract are EPSG:4326 GeoJSON.  A retained OSM
snapshot may be used when the InfDB export is absent, but the selected backend
is written to a status file and is never silently relabelled as InfDB.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, shape

from . import schweinfurt_base as base


INFDB_COMMIT = "5f6784257cbc0c8a0ecf0454c75abfa3db5d9d4d"
PYLOVO_COMMIT = "fb9938dada80fcdb4172ecd0577c7ce812b39c37"
REQUIRED_BUILDING_COLUMNS = {
    "objectid", "building_use_id", "floor_area", "floor_number",
    "occupants", "households", "geometry_json",
}
REQUIRED_WAY_COLUMNS = {"way_id", "geometry_json"}
REQUIRED_CONNECTION_COLUMNS = {"objectid", "way_id", "geometry_json"}


@dataclass(frozen=True)
class EvidenceBackendStatus:
    requested: str
    selected: str
    complete: bool
    reason: str
    infdb_export_dir: str
    infdb_commit: str = INFDB_COMMIT
    pylovo_commit: str = PYLOVO_COMMIT


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"objectid": "string", "way_id": "string"})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} misses required columns: {', '.join(missing)}")
    return frame


def _geojson(value: Any, expected: str | None = None):
    if isinstance(value, str):
        value = json.loads(value)
    geometry = shape(value)
    if expected and geometry.geom_type not in expected.split("|"):
        raise ValueError(f"Expected {expected}, received {geometry.geom_type}")
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Empty or invalid geometry in InfDB export")
    return geometry


def validate_infdb_export(export_dir: Path) -> dict[str, Any]:
    """Validate the minimal InfDB evidence contract and its key relations."""
    buildings = _read_csv(export_dir / "infdb_buildings.csv", REQUIRED_BUILDING_COLUMNS)
    ways = _read_csv(export_dir / "infdb_ways.csv", REQUIRED_WAY_COLUMNS)
    connections = _read_csv(
        export_dir / "infdb_connection_lines.csv", REQUIRED_CONNECTION_COLUMNS
    )
    if buildings["objectid"].duplicated().any():
        raise ValueError("infdb_buildings.csv contains duplicate objectid values")
    if ways["way_id"].duplicated().any():
        raise ValueError("infdb_ways.csv contains duplicate way_id values")
    unknown_buildings = set(connections["objectid"].dropna()) - set(buildings["objectid"])
    unknown_ways = set(connections["way_id"].dropna()) - set(ways["way_id"])
    if unknown_buildings or unknown_ways:
        raise ValueError(
            f"Broken connection foreign keys: {len(unknown_buildings)} buildings, "
            f"{len(unknown_ways)} ways"
        )
    building_geometries = [_geojson(value, "Polygon|MultiPolygon") for value in buildings.geometry_json]
    way_geometries = [_geojson(value, "LineString|MultiLineString") for value in ways.geometry_json]
    connection_geometries = [_geojson(value, "LineString") for value in connections.geometry_json]
    return {
        "building_count": len(buildings),
        "way_count": len(ways),
        "connection_count": len(connections),
        "buildings_with_connection": int(connections.objectid.nunique()),
        "building_geometry_types": sorted({item.geom_type for item in building_geometries}),
        "way_geometry_types": sorted({item.geom_type for item in way_geometries}),
        "maximum_connection_length_km": float(max(
            (_linestring_length_km(item) for item in connection_geometries), default=0.0
        )),
        "crs": "EPSG:4326",
        "infdb_commit": INFDB_COMMIT,
        "pylovo_commit": PYLOVO_COMMIT,
    }


def resolve_evidence_backend(config: dict[str, Any], project: Path) -> EvidenceBackendStatus:
    setting = config.get("evidence_backend", {})
    requested = str(setting.get("preferred", "infdb")).lower()
    export_dir = project / setting.get("infdb_export_dir", "data/infdb/schweinfurt_09662000")
    try:
        audit = validate_infdb_export(export_dir)
        return EvidenceBackendStatus(
            requested=requested,
            selected="infdb",
            complete=True,
            reason=f"validated {audit['building_count']} buildings and {audit['connection_count']} service lines",
            infdb_export_dir=str(export_dir),
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        if bool(setting.get("strict", False)):
            raise RuntimeError(f"Strict InfDB backend failed validation: {exc}") from exc
        return EvidenceBackendStatus(
            requested=requested,
            selected=str(setting.get("fallback", "retained_osm_snapshot")),
            complete=False,
            reason=f"InfDB contract unavailable or invalid: {type(exc).__name__}: {exc}",
            infdb_export_dir=str(export_dir),
        )


def write_backend_status(status: EvidenceBackendStatus, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "evidence_backend_status.json"
    path.write_text(json.dumps(asdict(status), indent=2) + "\n", encoding="utf-8")
    return path


def _linestring_length_km(line: LineString) -> float:
    return float(sum(base.distance_km(a, b) for a, b in zip(line.coords[:-1], line.coords[1:])))


def _iter_lines(geometry):
    if geometry.geom_type == "LineString":
        yield geometry
    elif geometry.geom_type == "MultiLineString":
        yield from geometry.geoms
    else:
        raise ValueError(f"Unsupported road geometry {geometry.geom_type}")


def load_infdb_road_graph(export_dir: Path) -> nx.Graph:
    """Build the common admissible-corridor graph from InfDB ways."""
    ways = _read_csv(export_dir / "infdb_ways.csv", REQUIRED_WAY_COLUMNS)
    graph = nx.Graph()
    for row in ways.itertuples():
        for line in _iter_lines(_geojson(row.geometry_json, "LineString|MultiLineString")):
            coordinates = [(float(x), float(y)) for x, y, *_ in line.coords]
            for u, v in zip(coordinates[:-1], coordinates[1:]):
                if u == v:
                    continue
                length = base.distance_km(u, v)
                if length <= 0:
                    continue
                graph.add_edge(
                    u, v, length_km=length, geometry=[u, v], way_id=str(row.way_id),
                    corridor_source="InfDB basemap road", evidence_class="B",
                )
    if not graph:
        raise ValueError("InfDB road export produced an empty graph")
    component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(component).copy()


def _service_class(building_use_id: Any, building_use: Any) -> tuple[str, bool]:
    code = str(building_use_id or "").strip()
    label = str(building_use or "").lower()
    if any(token in label for token in ("garage", "parking", "shed", "agricultural auxiliary")):
        return "auxiliary", False
    if code.startswith(("10", "11")) or any(token in label for token in ("residential", "dwelling", "house")):
        return "residential", True
    if code.startswith("21") or any(token in label for token in ("industrial", "factory", "warehouse")):
        return "industrial", True
    if any(token in label for token in ("school", "hospital", "public", "church", "administration")):
        return "public", True
    return "commercial", True


def _safe_positive(value: Any, default: float) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _reconcile(frame: pd.DataFrame, prior: str, output: str, total: float) -> None:
    weights = frame[prior].clip(lower=0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        raise ValueError(f"Cannot reconcile {output}: zero eligible prior")
    frame[output] = weights / weights.sum() * float(total)


def build_infdb_building_ledger(export_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Create the common four-sector building ledger from an InfDB export."""
    source = _read_csv(export_dir / "infdb_buildings.csv", REQUIRED_BUILDING_COLUMNS)
    connection_path = export_dir / "infdb_connection_lines.csv"
    connections = _read_csv(connection_path, REQUIRED_CONNECTION_COLUMNS)
    connection_lookup = connections.drop_duplicates("objectid").set_index("objectid")
    pylovo_path = export_dir / "pylovo_buildings.csv"
    pylovo = pd.read_csv(pylovo_path, dtype={"objectid": "string"}) if pylovo_path.exists() else pd.DataFrame()
    pylovo_peak = (
        pylovo.dropna(subset=["objectid"]).drop_duplicates("objectid").set_index("objectid")["peak_load_in_kw"]
        if {"objectid", "peak_load_in_kw"}.issubset(pylovo.columns) else pd.Series(dtype=float)
    )
    ro_heat_path = export_dir / "ro_heat_buildings.csv"
    ro_heat = pd.read_csv(ro_heat_path, dtype={"objectid": "string"}) if ro_heat_path.exists() else pd.DataFrame()
    heat_column = next((item for item in ("annual_heat_kwh", "space_heat_kwh", "heat_demand_kwh") if item in ro_heat), None)
    ro_heat_lookup = (
        ro_heat.drop_duplicates("objectid").set_index("objectid")[heat_column]
        if heat_column and "objectid" in ro_heat else pd.Series(dtype=float)
    )

    rows: list[dict[str, Any]] = []
    for row in source.itertuples():
        objectid = str(row.objectid)
        polygon = _geojson(row.geometry_json, "Polygon|MultiPolygon")
        centroid: Point = polygon.centroid
        service_class, eligible = _service_class(
            getattr(row, "building_use_id", ""), getattr(row, "building_use", "")
        )
        footprint = float(polygon.area * 111_200.0 * 71_550.0)
        floor_area = _safe_positive(getattr(row, "floor_area", None), footprint)
        levels = _safe_positive(getattr(row, "floor_number", None), max(1.0, floor_area / max(footprint, 1.0)))
        occupants = _safe_positive(getattr(row, "occupants", None), 0.0) if eligible else 0.0
        households = int(round(_safe_positive(getattr(row, "households", None), 0.0))) if service_class == "residential" else 0
        if eligible and service_class == "residential" and occupants <= 0:
            occupants = floor_area / 43.0
        if eligible and service_class == "residential" and households <= 0:
            households = max(1, int(math.ceil(floor_area / 90.0)))
        peak_prior = _safe_positive(pylovo_peak.get(objectid), 0.0)
        if peak_prior <= 0 and eligible:
            n_households = max(households, 1)
            peak_prior = (14.5 * n_households * (0.07 + 0.93 * n_households ** (-0.75))) if service_class == "residential" else floor_area * {"commercial": 0.085, "public": 0.095, "industrial": 0.080}.get(service_class, 0.06)
        heat_prior = _safe_positive(ro_heat_lookup.get(objectid), 0.0)
        if heat_prior <= 0 and eligible:
            heat_prior = floor_area * {"residential": 115.0, "commercial": 85.0, "public": 105.0, "industrial": 55.0}.get(service_class, 80.0)
        connection = connection_lookup.loc[objectid] if objectid in connection_lookup.index else None
        connection_geometry = _geojson(connection.geometry_json, "LineString") if connection is not None else None
        rows.append({
            "building_id": f"INFDB_{objectid}", "infdb_objectid": objectid,
            "osm_id": getattr(row, "osm_id", ""), "building_type": getattr(row, "building_type", ""),
            "building_use_id": getattr(row, "building_use_id", ""), "building_use": getattr(row, "building_use", ""),
            "service_class": service_class, "service_eligible": eligible,
            "service_class_evidence": "InfDB building_use/building_use_id",
            "service_evidence_class": "B", "geometry_evidence_class": "A/B",
            "lon": float(centroid.x), "lat": float(centroid.y),
            "footprint_m2": footprint, "levels": levels, "floor_area_m2": floor_area,
            "household_count": households, "occupant_prior": occupants,
            "electricity_prior": peak_prior, "water_prior": occupants + floor_area * {"commercial": 0.010, "public": 0.018, "industrial": 0.006}.get(service_class, 0.0),
            "heat_prior": heat_prior, "infdb_way_id": str(connection.way_id) if connection is not None else "",
            "connection_geometry_json": connection.geometry_json if connection is not None else "",
            "connection_length_km": _linestring_length_km(connection_geometry) if connection_geometry is not None else np.nan,
            "provenance": "InfDB LoD2/Zensus/basemap-road contract",
        })
    frame = pd.DataFrame(rows)
    frame.loc[~frame.service_eligible, ["occupant_prior", "electricity_prior", "water_prior", "heat_prior"]] = 0.0
    anchors = config["official_anchors"]
    _reconcile(frame, "occupant_prior", "population", anchors["served_population"])
    _reconcile(frame, "electricity_prior", "electricity_mwh_year", anchors["electricity_lv_annual_mwh"])
    _reconcile(frame, "electricity_prior", "electricity_peak_kw", anchors["electricity_lv_peak_mw"] * 1000.0)
    _reconcile(frame, "water_prior", "water_m3_year", anchors["drinking_water_annual_m3"])
    demand_model = config["demand_model"]
    frame["wastewater_sanitary_m3_year"] = float(demand_model["wastewater_sanitary_return_fraction"]) * frame.water_m3_year
    frame["heat_candidate_mwh_year"] = frame.heat_prior / 1000.0
    frame["electricity_service_peak_kw"] = frame.electricity_prior
    frame["water_service_peak_lps"] = frame.water_m3_year / (365.0 * 86400.0) * float(demand_model["drinking_water_peak_factor"]) * 1000.0
    harmon_factor = max(2.0, 1.0 + 14.0 / (4.0 + math.sqrt(float(anchors["served_population"]) / 1000.0)))
    frame["wastewater_service_peak_lps"] = frame.wastewater_sanitary_m3_year / (365.0 * 86400.0) * harmon_factor * 1000.0
    frame["heat_service_design_kw"] = frame.heat_candidate_mwh_year / (float(demand_model["district_heat_full_load_hours"]) / 1000.0)
    frame["electricity_connected"] = frame.service_eligible
    frame["electricity_connection_level"] = np.where(frame.electricity_service_peak_kw > 250.0, "MV", "LV")
    frame.loc[~frame.service_eligible, "electricity_connection_level"] = "NONE"
    frame["water_connected"] = frame.service_eligible
    frame["wastewater_connected"] = frame.service_eligible
    frame["heat_eligible"] = frame.service_eligible & ~frame.service_class.eq("industrial")
    frame.drop(columns=["occupant_prior", "water_prior", "electricity_prior", "heat_prior"], inplace=True)
    return frame
