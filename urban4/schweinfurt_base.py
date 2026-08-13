#!/usr/bin/env python3
"""Generate the electricity, drinking-water and wastewater base layers.

This case adapter retains the reproducible July 2026 Schweinfurt evidence
snapshot.  The city-independent four-sector orchestration, district-heating
module, shared-building ledger and normal-condition acceptance tests live in
``urban4.framework``.  No hazard, outage or resilience scenario is executed by
the publication workflow.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import random
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
CASE_CONFIG = json.loads((ROOT / "cases" / "schweinfurt.json").read_text(encoding="utf-8"))
RNG = random.Random(int(CASE_CONFIG["seed"]))

MODEL_BBOX = (49.985, 10.14, 50.095, 10.31)  # south, west, north, east
CONTEXT_BBOX = (49.97, 10.10, 50.13, 10.40)
WATER_ANNUAL_M3 = float(CASE_CONFIG["official_anchors"]["drinking_water_annual_m3"])
WATER_Q_AVG = WATER_ANNUAL_M3 / (365.0 * 86400.0)
DEMAND_MODEL = CASE_CONFIG["demand_model"]
WATER_PEAK_FACTOR = float(DEMAND_MODEL["drinking_water_peak_factor"])
# Local sanitary inflow follows the common retail-water ledger.  The public
# 19,000 m3/d treatment-plant quantity has a different system boundary and is
# retained only as external comparison evidence.
SEWER_DRY_M3_DAY = float(DEMAND_MODEL["wastewater_sanitary_return_fraction"]) * WATER_ANNUAL_M3 / 365.0
SEWER_PEAK_FACTOR = max(
    2.0,
    1.0 + 14.0 / (4.0 + math.sqrt(float(CASE_CONFIG["official_anchors"]["served_population"]) / 1000.0)),
)
SEWER_WET_M3_DAY = SEWER_PEAK_FACTOR * SEWER_DRY_M3_DAY
SEWER_Q_DRY = SEWER_DRY_M3_DAY / 86400.0
SEWER_Q_WET = SEWER_WET_M3_DAY / 86400.0
WWTP_LOAD_MW = 10_500.0 / 24.0 / 1000.0
# Official 2025 simultaneous LV peak (Stadtwerke Schweinfurt network data).
# Direct HV/MV industrial withdrawals are retained as aggregate evidence but
# are not silently redistributed to building-level LV loads.
BACKGROUND_GRID_PEAK_MW = float(CASE_CONFIG["official_anchors"]["electricity_lv_peak_mw"])

WATER_DN = np.array([80, 100, 125, 150, 200, 250, 300, 350, 400, 500, 600])
SEWER_DN = np.array([200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh OSM extracts through Overpass.")
    parser.add_argument("--water-zones", type=int, default=750)
    parser.add_argument("--power-ties", type=int, default=5)
    return parser.parse_args()


def in_bbox(lon: float, lat: float, bbox=MODEL_BBOX) -> bool:
    south, west, north, east = bbox
    return west <= lon <= east and south <= lat <= north


def xy_km(point: tuple[float, float]) -> tuple[float, float]:
    """Project WGS84 coordinates to the declared local metric case frame.

    The fixed factors are evaluated for Schweinfurt (approximately 50 degrees
    north).  They keep the dependency footprint small and are explicitly
    labelled as a local equirectangular frame rather than an EPSG projection.
    """
    lon, lat = point
    return lon * 71.55, lat * 111.20


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = xy_km(a)
    bx, by = xy_km(b)
    return math.hypot(ax - bx, ay - by)


def line_length_km(points: list[tuple[float, float]]) -> float:
    return sum(distance_km(points[i - 1], points[i]) for i in range(1, len(points)))


def center(element: dict[str, Any]) -> tuple[float, float] | None:
    if element.get("type") == "node":
        if element.get("lon") is None:
            return None
        return float(element["lon"]), float(element["lat"])
    item = element.get("center")
    if item:
        return float(item["lon"]), float(item["lat"])
    geometry = element.get("geometry") or []
    if geometry:
        return (
            sum(float(p["lon"]) for p in geometry) / len(geometry),
            sum(float(p["lat"]) for p in geometry) / len(geometry),
        )
    bounds = element.get("bounds")
    if bounds:
        return (
            (float(bounds["minlon"]) + float(bounds["maxlon"])) / 2,
            (float(bounds["minlat"]) + float(bounds["maxlat"])) / 2,
        )
    return None


def geometry(element: dict[str, Any]) -> list[tuple[float, float]]:
    return [(round(float(p["lon"]), 7), round(float(p["lat"]), 7)) for p in element.get("geometry") or []]


def tags(element: dict[str, Any]) -> dict[str, str]:
    return element.get("tags") or {}


def overpass(query: str) -> dict[str, Any]:
    endpoints = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass-api.de/api/interpreter",
    ]
    encoded = urllib.parse.urlencode({"data": query}).encode()
    for endpoint in endpoints:
        try:
            request = urllib.request.Request(
                endpoint,
                data=encoded,
                headers={"User-Agent": "TUM-Schweinfurt-proxy/1.0"},
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.load(response)
            if "elements" in data:
                return data
        except Exception as exc:
            print(f"Overpass endpoint failed: {exc}", file=sys.stderr)
    raise RuntimeError("No Overpass endpoint returned data")


def ensure_raw(refresh: bool) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sources = {
        "infrastructure.json": Path("/tmp/schweinfurt-infra.json"),
        "context.json": Path("/tmp/schweinfurt-context.json"),
        "local_roads.json": Path("/tmp/schweinfurt-local-roads.json"),
        "buildings.json": Path("/tmp/schweinfurt-buildings.json"),
        "sewer_detail.json": Path("/tmp/schweinfurt-sewer-detail.json"),
    }
    for name, source in sources.items():
        target = RAW / name
        compressed = target.with_suffix(target.suffix + ".xz")
        if not target.exists() and not compressed.exists() and source.exists():
            shutil.copy2(source, target)

    queries = {
        "infrastructure.json": """[out:json][timeout:240];(
nwr["power"~"^(substation|transformer|line|minor_line|cable|switch|terminal)$"](49.97,10.10,50.13,10.40);
nwr["man_made"~"^(water_works|water_well|water_tower|reservoir_covered|storage_tank|pumping_station|wastewater_plant|wastewater_basin)$"](49.97,10.10,50.13,10.40);
nwr["utility"="sewerage"](49.97,10.10,50.13,10.40);
way["man_made"="pipeline"]["substance"~"^(water|fresh_water|sewage|wastewater)$"](49.97,10.10,50.13,10.40);
);out tags center geom;""",
        "context.json": """[out:json][timeout:240];(
way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"](49.97,10.10,50.13,10.40);
way["waterway"~"^(river|canal)$"](49.97,10.10,50.13,10.40);
nwr["place"~"^(city|town|village)$"](49.97,10.10,50.13,10.40);
);out tags center geom;""",
        "local_roads.json": """[out:json][timeout:240];
way["highway"~"^(primary|secondary|tertiary|unclassified|residential|living_street|service)$"](49.985,10.14,50.095,10.31);
out tags geom;""",
        "buildings.json": """[out:json][timeout:240];
way["building"](49.985,10.14,50.095,10.31);out tags center geom;""",
        "sewer_detail.json": """[out:json][timeout:240];(
nwr["manhole"~"^(sewer|drain)$"](49.985,10.14,50.095,10.31);
nwr["sewer"](49.985,10.14,50.095,10.31);
way["waterway"="drain"](49.985,10.14,50.095,10.31);
);out tags center geom;""",
    }
    for name, query in queries.items():
        target = RAW / name
        compressed = target.with_suffix(target.suffix + ".xz")
        if refresh or (not target.exists() and not compressed.exists()):
            target.write_text(json.dumps(overpass(query), ensure_ascii=False), encoding="utf-8")
    missing = [
        name
        for name in queries
        if not (RAW / name).exists()
        and not (RAW / name).with_suffix(Path(name).suffix + ".xz").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing raw data: {missing}")


def read_json_path(path: Path) -> dict[str, Any]:
    """Read JSON from ``path`` or its lossless ``.xz`` archive."""
    if path.exists():
        with path.open("rt", encoding="utf-8") as stream:
            return json.load(stream)
    compressed = path.with_suffix(path.suffix + ".xz")
    if compressed.exists():
        with lzma.open(compressed, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    raise FileNotFoundError(f"Missing JSON evidence: {path} or {compressed}")


def read_raw_json(name: str) -> dict[str, Any]:
    return read_json_path(RAW / name)


def load_elements(name: str) -> list[dict[str, Any]]:
    return read_raw_json(name)["elements"]


def build_road_graph(elements: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    for element in elements:
        points = geometry(element)
        if len(points) < 2:
            continue
        road_class = tags(element).get("highway", "road")
        for a, b in zip(points[:-1], points[1:]):
            if a == b:
                continue
            length = distance_km(a, b)
            if graph.has_edge(a, b):
                if length < graph[a][b]["length_km"]:
                    graph[a][b].update(length_km=length, road_class=road_class)
            else:
                graph.add_edge(a, b, length_km=length, road_class=road_class)
    component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(component).copy()


class NodeIndex:
    def __init__(self, nodes: Iterable[tuple[float, float]]):
        self.nodes = list(nodes)
        self.tree = cKDTree(np.array([xy_km(p) for p in self.nodes]))

    def nearest(self, point: tuple[float, float]) -> tuple[float, float]:
        _, index = self.tree.query(np.array(xy_km(point)))
        return self.nodes[int(index)]


def service_area(point: tuple[float, float]) -> bool:
    mask = CASE_CONFIG["service_mask"]
    if mask["type"] != "union_of_circles":
        raise ValueError(f"Unsupported service mask: {mask['type']}")
    centres = [
        (tuple(component["center"]), float(component["radius_km"]))
        for component in mask["components"]
    ]
    return any(distance_km(point, centre) <= radius for centre, radius in centres)


def building_weight(kind: str) -> float:
    if kind in {"industrial", "warehouse"}:
        return 3.0
    if kind in {"commercial", "retail", "office", "hospital"}:
        return 2.5
    if kind in {"apartments", "residential", "dormitory"}:
        return 2.0
    if kind in {"house", "detached", "semidetached_house", "terrace"}:
        return 1.0
    if kind in {"garage", "garages", "shed", "roof"}:
        return 0.1
    return 0.7


def demand_zones(
    buildings: list[dict[str, Any]],
    road_index: NodeIndex,
    count: int,
) -> list[dict[str, Any]]:
    records = []
    for element in buildings:
        point = center(element)
        if point and service_area(point):
            records.append(
                (
                    point,
                    building_weight(tags(element).get("building", "yes")),
                    f"OSM_W{element['id']}",
                )
            )
    points = np.array([xy_km(item[0]) for item in records])
    if len(points) < count:
        raise RuntimeError("Too few buildings for requested demand zones")
    _, labels = kmeans2(
        points,
        count,
        minit="++",
        iter=40,
        seed=int(CASE_CONFIG["seed"]),
    )
    zones: list[dict[str, Any]] = []
    seen: dict[tuple[float, float], dict[str, Any]] = {}
    for cluster in range(count):
        members = np.where(labels == cluster)[0]
        if not len(members):
            continue
        weight = sum(records[int(i)][1] for i in members)
        weighted_lon = sum(records[int(i)][0][0] * records[int(i)][1] for i in members) / weight
        weighted_lat = sum(records[int(i)][0][1] * records[int(i)][1] for i in members) / weight
        member_ids = [records[int(i)][2] for i in members]
        snapped = road_index.nearest((weighted_lon, weighted_lat))
        if snapped in seen:
            seen[snapped]["weight"] += weight
            seen[snapped]["building_count"] += len(members)
            seen[snapped]["building_ids"].extend(member_ids)
        else:
            item = {
                "road_node": snapped,
                "centroid": (weighted_lon, weighted_lat),
                "weight": weight,
                "building_count": len(members),
                "building_ids": member_ids,
            }
            zones.append(item)
            seen[snapped] = item
    total_weight = sum(z["weight"] for z in zones)
    for i, zone in enumerate(zones, 1):
        zone["zone_id"] = f"DZ{i:03d}"
        zone["centroid_lon"] = zone["centroid"][0]
        zone["centroid_lat"] = zone["centroid"][1]
        zone["road_lon"] = zone["road_node"][0]
        zone["road_lat"] = zone["road_node"][1]
        fraction = zone["weight"] / total_weight
        zone["water_q_avg_m3s"] = WATER_Q_AVG * fraction
        zone["water_q_peak_m3s"] = WATER_Q_AVG * WATER_PEAK_FACTOR * fraction
        zone["sewer_q_dry_m3s"] = SEWER_Q_DRY * fraction
        zone["sewer_q_wet_m3s"] = SEWER_Q_WET * fraction
    return zones


_DEM_CACHE: tuple[cKDTree, np.ndarray] | None | bool = False


def _dem_cache() -> tuple[cKDTree, np.ndarray] | None:
    """Load optional open-DEM samples from data/raw/dem_points.csv once."""
    global _DEM_CACHE
    if _DEM_CACHE is False:
        path = RAW / "dem_points.csv"
        if path.exists():
            samples = pd.read_csv(path)
            required = {"lon", "lat", "elevation_m"}
            if not required.issubset(samples.columns):
                raise ValueError(f"{path} must contain {sorted(required)}")
            coordinates = np.asarray([xy_km((row.lon, row.lat)) for row in samples.itertuples()])
            _DEM_CACHE = (cKDTree(coordinates), samples["elevation_m"].to_numpy(dtype=float))
        else:
            _DEM_CACHE = None
    return _DEM_CACHE if _DEM_CACHE is not False else None


def terrain_evidence() -> dict[str, str]:
    if _dem_cache() is not None:
        return {"source": "data/raw/dem_points.csv", "class": "B", "method": "nearest open-DEM sample"}
    return {"source": "deterministic Main-valley terrain prior", "class": "C", "method": "analytic fallback"}


def proxy_elevation_m(point: tuple[float, float]) -> float:
    """Return open-DEM elevation when supplied, otherwise a Class-C prior."""
    cached = _dem_cache()
    if cached is not None:
        tree, values = cached
        _, index = tree.query(np.asarray(xy_km(point)))
        return round(float(values[int(index)]), 2)
    lon, lat = point
    main_lat = 50.035 + 0.035 * (lon - 10.14)
    north_km = (lat - main_lat) * 111.2
    valley_distance = abs(north_km)
    broad_north_rise = max(0.0, lat - 50.04) * 520.0
    west_rise = max(0.0, 10.205 - lon) * 120.0
    smooth = 4.0 * math.sin((lon - 10.14) * 38.0) * math.cos((lat - 49.985) * 31.0)
    return round(207.0 + 5.5 * valley_distance + broad_north_rise + west_rise + smooth, 2)


def cluster_wells(elements: list[dict[str, Any]], radius_km: float = 1.0) -> list[dict[str, Any]]:
    wells = []
    for element in elements:
        t = tags(element)
        if t.get("man_made") != "water_well":
            continue
        if not (t.get("drinking_water") == "yes" or "Stadtwerke Schweinfurt" in t.get("operator", "") or t.get("name")):
            continue
        point = center(element)
        if point:
            wells.append((point, t.get("name", "")))
    parents = list(range(len(wells)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i in range(len(wells)):
        for j in range(i):
            if distance_km(wells[i][0], wells[j][0]) <= radius_km:
                a, b = find(i), find(j)
                if a != b:
                    parents[b] = a
    grouped: dict[int, list[tuple[tuple[float, float], str]]] = defaultdict(list)
    for i, item in enumerate(wells):
        grouped[find(i)].append(item)
    result = []
    for index, cluster in enumerate(grouped.values(), 1):
        point = (
            sum(item[0][0] for item in cluster) / len(cluster),
            sum(item[0][1] for item in cluster) / len(cluster),
        )
        names = " ".join(item[1] for item in cluster)
        if "Seelenvater" in names:
            name = "Seelenvater wellfield"
        elif "Zeller Grund" in names:
            name = "Zeller Grund wellfield"
        elif "Bibrabrunnen" in names or "Weihersbrunnen" in names:
            name = "Bibrabrunnen / Weihersbrunnen"
        elif len(cluster) >= 10:
            name = "Main-valley wellfield"
        else:
            name = f"Mapped well cluster {index}"
        result.append({"name": name, "point": point, "well_count": len(cluster)})
    return result


def named_facilities(elements: list[dict[str, Any]], kinds: set[str]) -> list[dict[str, Any]]:
    result = []
    for element in elements:
        t = tags(element)
        if t.get("man_made") not in kinds:
            continue
        point = center(element)
        if not point:
            continue
        result.append(
            {
                "name": t.get("name") or t.get("man_made", "facility").replace("_", " ").title(),
                "kind": t.get("man_made"),
                "operator": t.get("operator", "not tagged"),
                "point": point,
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
            }
        )
    return result


def add_path_edges(target: nx.Graph, source: nx.Graph, path: list[tuple[float, float]]) -> None:
    for u, v in zip(path[:-1], path[1:]):
        target.add_edge(u, v, **source[u][v])


def compress_undirected(graph: nx.Graph, protected: set[tuple[float, float]]) -> nx.Graph:
    junctions = {node for node in graph if graph.degree(node) != 2 or node in protected}
    compressed = nx.Graph()
    visited: set[frozenset] = set()
    for start in junctions:
        for neighbour in graph.neighbors(start):
            key = frozenset((start, neighbour))
            if key in visited:
                continue
            visited.add(key)
            route = [start, neighbour]
            length = graph[start][neighbour]["length_km"]
            previous, current = start, neighbour
            while current not in junctions:
                choices = [n for n in graph.neighbors(current) if n != previous]
                if not choices:
                    break
                nxt = choices[0]
                visited.add(frozenset((current, nxt)))
                route.append(nxt)
                length += graph[current][nxt]["length_km"]
                previous, current = current, nxt
            compressed.add_edge(start, current, length_km=length, geometry=route)
    return compressed


def water_topology(
    road: nx.Graph,
    road_index: NodeIndex,
    zones: list[dict[str, Any]],
    infrastructure: list[dict[str, Any]],
) -> tuple[nx.Graph, tuple[float, float], list[dict[str, Any]], list[dict[str, Any]]]:
    facilities = named_facilities(infrastructure, {"water_works", "reservoir_covered", "water_tower"})
    waterworks = next(item for item in facilities if item["name"] == "Wasserwerk Schweinfurt")
    source = road_index.nearest(waterworks["point"])
    wells = cluster_wells(infrastructure)
    anchors = [road_index.nearest(item["point"]) for item in facilities if service_area(item["point"])]
    anchors += [road_index.nearest(item["point"]) for item in wells if service_area(item["point"])]
    targets = [zone["road_node"] for zone in zones] + anchors

    subgraph = nx.Graph()
    for target in targets:
        try:
            add_path_edges(subgraph, road, nx.shortest_path(road, source, target, weight="length_km"))
        except nx.NetworkXNoPath:
            pass

    base_length = sum(data["length_km"] for _, _, data in subgraph.edges(data=True))
    loop_target = 0.12 * base_length
    added = 0.0
    candidates = []
    for u, v, data in road.edges(data=True):
        if u in subgraph and v in subgraph and not subgraph.has_edge(u, v):
            if data["length_km"] <= 0.35:
                candidates.append((data["length_km"], u, v, data))
    candidates.sort(key=lambda item: item[0])
    for length, u, v, data in candidates:
        if added >= loop_target:
            break
        subgraph.add_edge(u, v, **data, tie_candidate=True)
        added += length

    protected = {source, *[zone["road_node"] for zone in zones], *anchors}
    model = compress_undirected(subgraph, protected)

    q_design = defaultdict(float)
    for zone in zones:
        target = zone["road_node"]
        if target not in model:
            continue
        first = nx.shortest_path(model, source, target, weight="length_km")
        first_edges = {frozenset((u, v)) for u, v in zip(first[:-1], first[1:])}
        for u, v in zip(first[:-1], first[1:]):
            q_design[frozenset((u, v))] += 0.70 * zone["water_q_peak_m3s"]
        try:
            second = nx.shortest_path(
                model,
                source,
                target,
                weight=lambda u, v, d: d["length_km"] * (5.0 if frozenset((u, v)) in first_edges else 1.0),
            )
        except nx.NetworkXNoPath:
            second = first
        for u, v in zip(second[:-1], second[1:]):
            q_design[frozenset((u, v))] += 0.30 * zone["water_q_peak_m3s"]

    for u, v, data in model.edges(data=True):
        q = max(q_design[frozenset((u, v))], 0.0008)
        diameter = math.sqrt(4.0 * q / math.pi / 1.0)
        dn = int(WATER_DN[np.searchsorted(WATER_DN, min(600, math.ceil(diameter * 1000)), side="left")])
        data.update(
            q_design_m3s=q,
            diameter_mm=dn,
            roughness_k_mm=0.20,
            loss_coefficient=0.0,
            confidence_class="B/C",
            provenance="road-constrained synthetic route; hydraulic sizing prior",
        )
    return model, source, facilities, wells


def compress_directed(graph: nx.DiGraph, protected: set[tuple[float, float]]) -> nx.DiGraph:
    pass_through = {
        node
        for node in graph
        if graph.in_degree(node) == 1 and graph.out_degree(node) == 1 and node not in protected
    }
    starts = set(graph) - pass_through
    compressed = nx.DiGraph()
    visited: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for start in starts:
        for successor in graph.successors(start):
            if (start, successor) in visited:
                continue
            route = [start, successor]
            visited.add((start, successor))
            length = graph[start][successor]["length_km"]
            current = successor
            while current in pass_through:
                nxt = next(graph.successors(current))
                visited.add((current, nxt))
                route.append(nxt)
                length += graph[current][nxt]["length_km"]
                current = nxt
            compressed.add_edge(start, current, length_km=length, geometry=route)
    return compressed


def wastewater_topology(
    road: nx.Graph,
    road_index: NodeIndex,
    zones: list[dict[str, Any]],
    infrastructure: list[dict[str, Any]],
) -> tuple[nx.DiGraph, tuple[float, float], list[tuple[float, float]], list[tuple[float, float]], dict]:
    plants = named_facilities(infrastructure, {"wastewater_plant"})
    plant = next(item for item in plants if item["name"] == "Kläranlage Schweinfurt")
    outlet = road_index.nearest(plant["point"])

    directed = nx.DiGraph()
    node_elevation = {node: proxy_elevation_m(node) for node in road.nodes}
    elevation_span = max(max(node_elevation.values()) - min(node_elevation.values()), 1.0)
    rise_penalty_alpha = float(CASE_CONFIG["wastewater"]["terrain_rise_penalty_alpha_integrated"])
    for u, v, data in road.edges(data=True):
        zu, zv = node_elevation[u], node_elevation[v]
        length = data["length_km"]
        directed.add_edge(u, v, length_km=length, cost=length * (1.0 + rise_penalty_alpha * max(0.0, zv - zu) / elevation_span))
        directed.add_edge(v, u, length_km=length, cost=length * (1.0 + rise_penalty_alpha * max(0.0, zu - zv) / elevation_span))

    union = nx.DiGraph()
    for zone in zones:
        try:
            path = nx.shortest_path(directed, zone["road_node"], outlet, weight="cost")
        except nx.NetworkXNoPath:
            continue
        for u, v in zip(path[:-1], path[1:]):
            union.add_edge(u, v, length_km=road[u][v]["length_km"])

    protected = {outlet, *[zone["road_node"] for zone in zones]}
    model = compress_directed(union, protected)

    # Keep the solver boundary as a true single-inlet outfall. Independent
    # catchment branches may otherwise meet only at the treatment-plant node,
    # which is valid graphically but rejected by SWMM's outfall definition.
    predecessors = list(model.predecessors(outlet))
    if len(predecessors) > 1:
        terminal = (outlet[0], outlet[1] + 0.000045)
        while terminal in model:
            terminal = (terminal[0], terminal[1] + 0.00001)
        for predecessor in predecessors:
            data = dict(model[predecessor][outlet])
            geometry = list(data.get("geometry", [predecessor, outlet]))
            if tuple(geometry[0]) != predecessor:
                geometry.reverse()
            geometry[-1] = terminal
            model.remove_edge(predecessor, outlet)
            model.add_edge(
                predecessor, terminal,
                length_km=max(distance_km(predecessor, terminal), 0.005),
                geometry=geometry,
            )
        model.add_edge(
            terminal, outlet,
            length_km=max(distance_km(terminal, outlet), 0.005),
            geometry=[terminal, outlet],
        )

    edge_flow = defaultdict(float)
    for zone in zones:
        node = zone["road_node"]
        if node not in model or not nx.has_path(model, node, outlet):
            continue
        path = nx.shortest_path(model, node, outlet)
        for u, v in zip(path[:-1], path[1:]):
            edge_flow[(u, v)] += zone["sewer_q_wet_m3s"]

    for u, v, data in model.edges(data=True):
        q = max(edge_flow[(u, v)], 0.001)
        ground_drop = proxy_elevation_m(u) - proxy_elevation_m(v)
        slope = max(0.0025, ground_drop / max(data["length_km"] * 1000.0, 1.0))
        manning = 0.013
        diameter = (q * manning * 4.0 ** (5.0 / 3.0) / (math.pi * math.sqrt(slope))) ** (3.0 / 8.0)
        needed = min(2000, max(200, math.ceil(diameter * 1000)))
        dn = int(SEWER_DN[np.searchsorted(SEWER_DN, needed, side="left")])
        data.update(
            q_wet_m3s=q,
            q_dry_m3s=q * SEWER_Q_DRY / SEWER_Q_WET,
            diameter_mm=dn,
            slope=slope,
            manning_n=manning,
            link_type="gravity",
            confidence_class="B/C",
            provenance="DEM/road-constrained synthetic sewer; Manning sizing prior",
        )

    scored = []
    for u, v, data in model.edges(data=True):
        if v == outlet:
            continue
        lift = max(0.0, proxy_elevation_m(v) - proxy_elevation_m(u))
        scored.append((lift * max(data["q_wet_m3s"], 0.001) + 0.02 * data["q_wet_m3s"], u, v))
    scored.sort(reverse=True)
    pumps: list[tuple[float, float]] = []
    for _, u, _ in scored:
        if all(distance_km(u, existing) >= 0.45 for existing in pumps):
            pumps.append(u)
        if len(pumps) == 13:
            break
    if len(pumps) < 13:
        for _, u, _ in scored:
            if u not in pumps:
                pumps.append(u)
            if len(pumps) == 13:
                break

    force_length = 0.0
    for pump in pumps:
        current = pump
        local = 0.0
        while current != outlet and model.out_degree(current):
            successor = max(model.successors(current), key=lambda n: model[current][n]["q_wet_m3s"])
            if successor == outlet:
                break
            edge = model[current][successor]
            edge["link_type"] = "force_main"
            local += edge["length_km"]
            force_length += edge["length_km"]
            current = successor
            if local >= 1.0 or force_length >= 13.0:
                break
        if force_length >= 13.0:
            break

    node_flow = {}
    for node in model:
        node_flow[node] = sum(model[u][node]["q_wet_m3s"] for u in model.predecessors(node))
    cso_candidates = sorted((value, node) for node, value in node_flow.items() if node not in pumps and node != outlet)
    csos: list[tuple[float, float]] = []
    for _, node in reversed(cso_candidates):
        if all(distance_km(node, existing) >= 0.25 for existing in csos):
            csos.append(node)
        if len(csos) == 23:
            break
    for _, node in reversed(cso_candidates):
        if len(csos) == 23:
            break
        if node not in csos:
            csos.append(node)

    total_weight = sum(1.0 + model.in_degree(node) + model.out_degree(node) for node in model)
    manholes = {}
    remaining = 6200
    nodes = list(model)
    for node in nodes[:-1]:
        count = max(1, round(6200 * (1.0 + model.in_degree(node) + model.out_degree(node)) / total_weight))
        count = min(count, remaining - (len(nodes) - len(manholes) - 1))
        manholes[node] = count
        remaining -= count
    manholes[nodes[-1]] = remaining
    metadata = {
        "force_main_length_km_proxy": round(force_length, 3),
        "equivalent_manhole_count": sum(manholes.values()),
        "manholes": manholes,
        "plant": plant,
    }
    return model, outlet, pumps, csos, metadata


def power_topology(
    road: nx.Graph,
    road_index: NodeIndex,
    buildings: list[dict[str, Any]],
    infrastructure: list[dict[str, Any]],
    tie_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buses = []
    seen = set()
    for element in infrastructure:
        t = tags(element)
        if t.get("power") != "substation":
            continue
        point = center(element)
        if not point or not in_bbox(*point):
            continue
        voltage = t.get("voltage", "")
        if "20000" not in voltage:
            continue
        name = t.get("name") or f"OSM substation {element['id']}"
        key = (name, round(point[0], 5), round(point[1], 5))
        if key in seen:
            continue
        seen.add(key)
        buses.append(
            {
                "name": name,
                "point": point,
                "voltage_kv": 20.0,
                "operator": t.get("operator", "not tagged"),
                "root": name in {"U10", "U07"} or "110000" in voltage,
                "osm_id": element.get("id"),
                "confidence_class": "A",
            }
        )
    if not buses:
        raise RuntimeError("No mapped 20 kV substations")
    if not any(bus["root"] for bus in buses):
        buses[0]["root"] = True
    for i, bus in enumerate(buses, 1):
        bus["bus_id"] = f"PB{i:03d}"
        bus["road_node"] = road_index.nearest(bus["point"])

    complete = nx.Graph()
    for i in range(len(buses)):
        complete.add_node(i)
        for j in range(i):
            complete.add_edge(i, j, weight=distance_km(buses[i]["point"], buses[j]["point"]))
    tree = nx.minimum_spanning_tree(complete, weight="weight")
    selected = {(min(u, v), max(u, v)): False for u, v in tree.edges()}
    non_tree = sorted(
        (data["weight"], min(u, v), max(u, v))
        for u, v, data in complete.edges(data=True)
        if not tree.has_edge(u, v)
    )
    for _, u, v in non_tree[:tie_count]:
        selected[(u, v)] = True

    lines = []
    for index, ((u, v), normally_open) in enumerate(selected.items(), 1):
        try:
            route = nx.shortest_path(road, buses[u]["road_node"], buses[v]["road_node"], weight="length_km")
            length = sum(road[a][b]["length_km"] for a, b in zip(route[:-1], route[1:]))
        except nx.NetworkXNoPath:
            route = [buses[u]["point"], buses[v]["point"]]
            length = distance_km(*route)
        lines.append(
            {
                "line_id": f"PL{index:03d}",
                "from_bus": buses[u]["bus_id"],
                "to_bus": buses[v]["bus_id"],
                "length_km": max(0.05, length),
                "geometry": route,
                "r_ohm_per_km": 0.125,
                "x_ohm_per_km": 0.080,
                "c_nf_per_km": 250.0,
                "max_i_ka": 0.63,
                "normally_open": normally_open,
                "confidence_class": "B/C",
                "provenance": "synthetic road-routed 20 kV feeder; generic cable prior",
            }
        )

    weighted_buildings = []
    for element in buildings:
        point = center(element)
        if point and service_area(point):
            weighted_buildings.append((point, building_weight(tags(element).get("building", "yes"))))
    bus_weights = defaultdict(float)
    for point, weight in weighted_buildings:
        nearest = min(buses, key=lambda item: distance_km(point, item["point"]))
        bus_weights[nearest["bus_id"]] += weight
    total = sum(bus_weights.values())
    for bus in buses:
        bus["background_peak_p_mw"] = BACKGROUND_GRID_PEAK_MW * bus_weights[bus["bus_id"]] / total
        bus["background_peak_q_mvar"] = bus["background_peak_p_mw"] * math.tan(math.acos(0.96))
    return buses, lines


def asset_couplings(
    buses: list[dict[str, Any]],
    water_facilities: list[dict[str, Any]],
    wells: list[dict[str, Any]],
    sewer: nx.DiGraph,
    pumps: list[tuple[float, float]],
    sewer_meta: dict,
) -> pd.DataFrame:
    assets: list[dict[str, Any]] = []
    waterworks = next(item for item in water_facilities if item["name"] == "Wasserwerk Schweinfurt")
    p_waterworks = 1000 * 9.81 * WATER_Q_AVG * 45.0 / 0.72 / 1e6
    assets.append(
        {
            "asset_id": "WAT-WW-01",
            "asset_name": waterworks["name"],
            "sector": "drinking_water",
            "asset_type": "waterworks_booster",
            "point": waterworks["point"],
            "p_mw": p_waterworks,
            "q_mvar": p_waterworks * math.tan(math.acos(0.90)),
            "parameter_basis": "rho*g*Q*H/eta; H=45 m and eta=0.72 are engineering priors",
        }
    )
    active_wells = [item for item in wells if service_area(item["point"])]
    well_total = sum(item["well_count"] for item in active_wells) or 1
    for i, well in enumerate(active_wells, 1):
        q = WATER_Q_AVG * well["well_count"] / well_total
        p = 1000 * 9.81 * q * 30.0 / 0.72 / 1e6
        assets.append(
            {
                "asset_id": f"WAT-WF-{i:02d}",
                "asset_name": well["name"],
                "sector": "drinking_water",
                "asset_type": "wellfield_pumps",
                "point": well["point"],
                "p_mw": p,
                "q_mvar": p * math.tan(math.acos(0.90)),
                "parameter_basis": "well-count allocation; H=30 m and eta=0.72 are engineering priors",
            }
        )
    reservoirs = [item for item in water_facilities if item["kind"] in {"reservoir_covered", "water_tower"} and service_area(item["point"])]
    for i, reservoir in enumerate(reservoirs, 1):
        assets.append(
            {
                "asset_id": f"WAT-ST-{i:02d}",
                "asset_name": reservoir["name"],
                "sector": "drinking_water",
                "asset_type": "storage_controls",
                "point": reservoir["point"],
                "p_mw": 0.005,
                "q_mvar": 0.0016,
                "parameter_basis": "controls and telemetry engineering prior",
            }
        )
    for i, node in enumerate(pumps, 1):
        outgoing = list(sewer.out_edges(node, data=True))
        q = max((data["q_wet_m3s"] for _, _, data in outgoing), default=SEWER_Q_DRY / 13)
        head = max(8.0, max((proxy_elevation_m(v) - proxy_elevation_m(node) + 6.0 for _, v, _ in outgoing), default=8.0))
        p = 1000 * 9.81 * min(q, SEWER_Q_DRY) * head / 0.68 / 1e6
        assets.append(
            {
                "asset_id": f"SEW-PS-{i:02d}",
                "asset_name": f"Synthetic pumping station {i:02d}",
                "sector": "wastewater",
                "asset_type": "sewer_pump",
                "point": node,
                "p_mw": p,
                "q_mvar": p * math.tan(math.acos(0.88)),
                "parameter_basis": "terrain bottleneck; hydraulic power with eta=0.68 prior",
            }
        )
    plant = sewer_meta["plant"]
    assets.append(
        {
            "asset_id": "SEW-WWTP-01",
            "asset_name": plant["name"],
            "sector": "wastewater",
            "asset_type": "wastewater_treatment_plant",
            "point": plant["point"],
            "p_mw": WWTP_LOAD_MW,
            "q_mvar": WWTP_LOAD_MW * math.tan(math.acos(0.95)),
            "parameter_basis": "official 10,500 kWh/day treatment-process electricity demand",
        }
    )

    rows = []
    for asset in assets:
        candidates = sorted(buses, key=lambda bus: distance_km(asset["point"], bus["point"]))[:3]
        raw = [math.exp(-distance_km(asset["point"], bus["point"]) / 1.5) for bus in candidates]
        normalizer = sum(raw)
        for rank, (bus, likelihood) in enumerate(zip(candidates, raw), 1):
            rows.append(
                {
                    "asset_id": asset["asset_id"],
                    "asset_name": asset["asset_name"],
                    "sector": asset["sector"],
                    "asset_type": asset["asset_type"],
                    "lon": asset["point"][0],
                    "lat": asset["point"][1],
                    "candidate_rank": rank,
                    "candidate_bus": bus["bus_id"],
                    "candidate_bus_name": bus["name"],
                    "road_distance_proxy_km": round(distance_km(asset["point"], bus["point"]), 3),
                    "assignment_probability": likelihood / normalizer,
                    "selected_base_case": rank == 1,
                    "p_mw": asset["p_mw"],
                    "q_mvar": asset["q_mvar"],
                    "confidence_class": "B/C",
                    "parameter_basis": asset["parameter_basis"],
                }
            )
    return pd.DataFrame(rows)


def dataframe_exports(
    water: nx.Graph,
    water_source: tuple[float, float],
    zones: list[dict[str, Any]],
    water_facilities: list[dict[str, Any]],
    wells: list[dict[str, Any]],
    sewer: nx.DiGraph,
    sewer_outlet: tuple[float, float],
    pumps: list[tuple[float, float]],
    csos: list[tuple[float, float]],
    sewer_meta: dict,
    buses: list[dict[str, Any]],
    lines: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    water_node_ids = {node: f"WJ{i:04d}" for i, node in enumerate(water.nodes, 1)}
    demand_by_node = {zone["road_node"]: zone for zone in zones}
    storage_nodes = {
        min(water.nodes, key=lambda n: distance_km(n, item["point"])): item["name"]
        for item in water_facilities
        if item["kind"] in {"reservoir_covered", "water_tower"} and service_area(item["point"])
    }
    well_nodes = {
        min(water.nodes, key=lambda n: distance_km(n, item["point"])): item["name"]
        for item in wells
        if service_area(item["point"])
    }
    water_nodes = []
    for node, node_id in water_node_ids.items():
        kind = "junction"
        name = node_id
        if node == water_source:
            kind, name = "source", "Wasserwerk Schweinfurt"
        elif node in storage_nodes:
            kind, name = "storage", storage_nodes[node]
        elif node in well_nodes:
            kind, name = "wellfield", well_nodes[node]
        zone = demand_by_node.get(node)
        water_nodes.append(
            {
                "node_id": node_id,
                "name": name,
                "node_type": kind,
                "lon": node[0],
                "lat": node[1],
                "height_m": proxy_elevation_m(node),
                "pn_bar": 6.0,
                "water_q_avg_m3s": zone["water_q_avg_m3s"] if zone else 0.0,
                "water_q_peak_m3s": zone["water_q_peak_m3s"] if zone else 0.0,
                "building_count": zone["building_count"] if zone else 0,
                "confidence_class": "B/C" if kind == "junction" else "A/B",
            }
        )
    water_pipes = []
    for i, (u, v, data) in enumerate(water.edges(data=True), 1):
        water_pipes.append(
            {
                "pipe_id": f"WP{i:04d}",
                "from_node": water_node_ids[u],
                "to_node": water_node_ids[v],
                "length_km": data["length_km"],
                "diameter_mm": data["diameter_mm"],
                "roughness_k_mm": data["roughness_k_mm"],
                "loss_coefficient": data["loss_coefficient"],
                "q_design_m3s": data["q_design_m3s"],
                "geometry_json": json.dumps(data["geometry"], separators=(",", ":")),
                "confidence_class": data["confidence_class"],
                "provenance": data["provenance"],
            }
        )

    sewer_node_ids = {node: f"SJ{i:04d}" for i, node in enumerate(sewer.nodes, 1)}
    sewer_demand = {zone["road_node"]: zone for zone in zones}
    sewer_nodes = []
    for node, node_id in sewer_node_ids.items():
        zone = sewer_demand.get(node)
        node_type = "outfall" if node == sewer_outlet else "pump" if node in pumps else "cso" if node in csos else "junction"
        ground = proxy_elevation_m(node)
        distance_to_outlet = (
            nx.shortest_path_length(sewer, node, sewer_outlet, weight="length_km")
            if nx.has_path(sewer, node, sewer_outlet)
            else 0.0
        )
        invert = min(ground - 1.5, proxy_elevation_m(sewer_outlet) - 3.0 + distance_to_outlet * 1000 * 0.003)
        sewer_nodes.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "lon": node[0],
                "lat": node[1],
                "ground_elevation_m": ground,
                "invert_elevation_m": round(invert, 3),
                "max_depth_m": max(2.0, round(ground - invert, 3)),
                "dry_inflow_m3s": zone["sewer_q_dry_m3s"] if zone else 0.0,
                "wet_inflow_m3s": zone["sewer_q_wet_m3s"] if zone else 0.0,
                "equivalent_manhole_count": sewer_meta["manholes"].get(node, 0),
                "confidence_class": "B/C",
            }
        )
    sewer_conduits = []
    for i, (u, v, data) in enumerate(sewer.edges(data=True), 1):
        sewer_conduits.append(
            {
                "link_id": f"SC{i:04d}",
                "from_node": sewer_node_ids[u],
                "to_node": sewer_node_ids[v],
                "link_type": data["link_type"],
                "length_m": data["length_km"] * 1000,
                "diameter_mm": data["diameter_mm"],
                "manning_n": data["manning_n"],
                "slope": data["slope"],
                "q_dry_m3s": data["q_dry_m3s"],
                "q_wet_m3s": data["q_wet_m3s"],
                "geometry_json": json.dumps(data["geometry"], separators=(",", ":")),
                "confidence_class": data["confidence_class"],
                "provenance": data["provenance"],
            }
        )

    power_buses = pd.DataFrame(
        [
            {
                k: v
                for k, v in bus.items()
                if k not in {"point", "road_node"}
            }
            | {"lon": bus["point"][0], "lat": bus["point"][1]}
            for bus in buses
        ]
    )
    power_lines = pd.DataFrame(
        [
            {k: v for k, v in line.items() if k != "geometry"}
            | {"geometry_json": json.dumps(line["geometry"], separators=(",", ":"))}
            for line in lines
        ]
    )
    return {
        "water_nodes": pd.DataFrame(water_nodes),
        "water_pipes": pd.DataFrame(water_pipes),
        "wastewater_nodes": pd.DataFrame(sewer_nodes),
        "wastewater_conduits": pd.DataFrame(sewer_conduits),
        "power_buses": power_buses,
        "power_lines": power_lines,
    }


def observed_features(infrastructure: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    features = {"power": [], "water": [], "wastewater": []}
    for element in infrastructure:
        t = tags(element)
        geom = geometry(element)
        point = center(element)
        if t.get("power") in {"line", "minor_line", "cable"} and len(geom) >= 2:
            features["power"].append(
                {
                    "type": "Feature",
                    "properties": {
                        "name": t.get("name") or "Mapped electrical line",
                        "voltage": t.get("voltage", "not tagged"),
                        "status": "observed",
                    },
                    "geometry": {"type": "LineString", "coordinates": geom},
                }
            )
        if t.get("man_made") == "pipeline" and t.get("substance") in {"water", "fresh_water"} and len(geom) >= 2:
            features["water"].append(
                {
                    "type": "Feature",
                    "properties": {
                        "name": "Mapped water pipeline",
                        "diameter_mm": t.get("diameter", "not tagged"),
                        "status": "observed",
                    },
                    "geometry": {"type": "LineString", "coordinates": geom},
                }
            )
        if t.get("man_made") == "wastewater_plant" and point:
            features["wastewater"].append(
                {
                    "type": "Feature",
                    "properties": {"name": t.get("name") or "Mapped WWTP", "status": "observed"},
                    "geometry": {"type": "Point", "coordinates": point},
                }
            )
    return features


def to_geojson(
    frames: dict[str, pd.DataFrame],
    couplings: pd.DataFrame,
    observed: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    features = [feature for group in observed.values() for feature in group]
    node_lookup = {}
    for layer, name in [("water", "water_nodes"), ("wastewater", "wastewater_nodes"), ("power", "power_buses")]:
        frame = frames[name]
        id_col = "bus_id" if layer == "power" else "node_id"
        for row in frame.to_dict("records"):
            node_lookup[row[id_col]] = (row["lon"], row["lat"])
            props = {k: v for k, v in row.items() if k not in {"lon", "lat"} and pd.notna(v)}
            props["layer"] = layer
            props["status"] = "observed" if layer == "power" else "synthetic"
            features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
                }
            )
    for layer, name, fcol, tcol in [
        ("water", "water_pipes", "from_node", "to_node"),
        ("wastewater", "wastewater_conduits", "from_node", "to_node"),
        ("power", "power_lines", "from_bus", "to_bus"),
    ]:
        for row in frames[name].to_dict("records"):
            geometry_points = json.loads(row["geometry_json"])
            props = {k: v for k, v in row.items() if k != "geometry_json" and pd.notna(v)}
            props.update(layer=layer, status="synthetic")
            features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "LineString", "coordinates": geometry_points},
                }
            )
    for row in couplings[couplings["selected_base_case"]].to_dict("records"):
        start = (row["lon"], row["lat"])
        end = node_lookup[row["candidate_bus"]]
        props = {k: v for k, v in row.items() if k not in {"lon", "lat"} and pd.notna(v)}
        props.update(layer="coupling", status="synthetic")
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "LineString", "coordinates": [start, end]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def write_swmm(frames: dict[str, pd.DataFrame]) -> None:
    nodes = frames["wastewater_nodes"]
    links = frames["wastewater_conduits"]
    pump_links = links[links["swmm_link_type"] == "pump"]
    pump_nodes = set(pump_links["from_node"])
    lines = [
        "[TITLE]",
        ";; Schweinfurt reduced synthetic sewer; see METHODOLOGY.md",
        "",
        "[OPTIONS]",
        "FLOW_UNITS           CMS",
        "INFILTRATION         HORTON",
        "FLOW_ROUTING         DYNWAVE",
        "FORCE_MAIN_EQUATION  H-W",
        "START_DATE           07/01/2026",
        "START_TIME           00:00:00",
        "REPORT_START_DATE    07/01/2026",
        "REPORT_START_TIME    00:00:00",
        "END_DATE             07/03/2026",
        "END_TIME             00:00:00",
        "ROUTING_STEP         00:00:30",
        "VARIABLE_STEP        0.75",
        "MINIMUM_STEP         0.1",
        "MAX_TRIALS           100",
        "HEAD_TOLERANCE       0.005",
        "SURCHARGE_METHOD     SLOT",
        "",
        "[JUNCTIONS]",
        ";;Name Elevation MaxDepth InitDepth SurDepth Aponded",
    ]
    for row in nodes[(nodes.node_type != "outfall") & (~nodes.node_id.isin(pump_nodes))].itertuples():
        lines.append(f"{row.node_id} {row.invert_elevation_m:.3f} {row.max_depth_m:.3f} 0 0 0")
    lines += ["", "[STORAGE]", ";;Name Elev MaxDepth InitDepth Shape A1 A2 A0 SurDepth Fevap Psi Ksat IMD"]
    for row in nodes[nodes.node_id.isin(pump_nodes)].itertuples():
        lines.append(
            f"{row.node_id} {row.invert_elevation_m:.3f} {row.max_depth_m:.3f} "
            "0 FUNCTIONAL 0 0 25 0 0"
        )
    lines += ["", "[OUTFALLS]", ";;Name Elevation Type StageData Gated RouteTo"]
    for row in nodes[nodes.node_type == "outfall"].itertuples():
        lines.append(f"{row.node_id} {row.invert_elevation_m:.3f} FREE NO")
    lines += ["", "[CONDUITS]", ";;Name FromNode ToNode Length Roughness InOffset OutOffset InitFlow MaxFlow"]
    for row in links[links["swmm_link_type"] == "conduit"].itertuples():
        roughness = row.manning_n
        lines.append(
            f"{row.link_id} {row.from_node} {row.to_node} {row.length_m:.2f} "
            f"{roughness:.4f} 0 0 0 0"
        )
    lines += ["", "[PUMPS]", ";;Name FromNode ToNode PumpCurve Status Startup Shutoff"]
    for index, row in enumerate(pump_links.itertuples(), 1):
        lines.append(f"PU{index:03d} {row.from_node} {row.to_node} PC{index:03d} ON 0.15 0.05")
    lines += ["", "[XSECTIONS]", ";;Link Shape Geom1 Geom2 Geom3 Geom4 Barrels"]
    for row in links[links["swmm_link_type"] == "conduit"].itertuples():
        lines.append(f"{row.link_id} CIRCULAR {row.diameter_mm / 1000:.3f} 0 0 0 1")
    lines += ["", "[CURVES]", ";;Name Type X-Value Y-Value"]
    for index, row in enumerate(pump_links.itertuples(), 1):
        design_flow = max(float(row.pump_design_flow_m3s), 0.0002)
        lines.extend(
            [
                f"PC{index:03d} PUMP4 0 0",
                f"PC{index:03d} 0.15 {design_flow:.8f}",
                f"PC{index:03d} 2.0 {design_flow:.8f}",
            ]
        )
    lines += ["", "[DWF]", ";;Node Constituent Baseline Patterns"]
    for row in nodes[nodes.dry_inflow_m3s > 0].itertuples():
        lines.append(f"{row.node_id} FLOW {row.dry_inflow_m3s:.8f}")
    lines += ["", "[COORDINATES]", ";;Node X-Coord Y-Coord"]
    for row in nodes.itertuples():
        x, y = xy_km((row.lon, row.lat))
        lines.append(f"{row.node_id} {x * 1000:.2f} {y * 1000:.2f}")
    lines += ["", "[TAGS]", ";;Object Type Name Tag"]
    for row in nodes.itertuples():
        if row.node_type in {"pump", "cso"}:
            lines.append(f"NODE {row.node_id} {row.node_type}")
    lines.append("")
    (OUT / "schweinfurt_proxy.inp").write_text("\n".join(lines), encoding="utf-8")


def prepare_wastewater_hydraulics(frames: dict[str, pd.DataFrame]) -> None:
    """Derive cover-bounded gravity inverts and explicit lift-station edges.

    The retained terrain is a class-C prior, so this is a transparent design
    construction rather than a reconstruction of surveyed inverts.  Gravity
    segments use a 0.0002 minimum design grade with 1.5--8 m cover.  Whenever
    that envelope cannot reach the downstream invert, the edge is converted to
    a wet-well/pump connection for the SWMM normal-condition model.
    """
    nodes = frames["wastewater_nodes"].copy().set_index("node_id", drop=False)
    links = frames["wastewater_conduits"].copy()
    graph = nx.DiGraph()
    for row in links.itertuples():
        graph.add_edge(row.from_node, row.to_node, length_m=float(row.length_m), link_id=row.link_id)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Wastewater collection graph must be acyclic before hydraulic preparation")
    outfalls = nodes.index[nodes["node_type"] == "outfall"].tolist()
    if len(outfalls) != 1:
        raise ValueError("Exactly one wastewater outfall is required")
    outfall = outfalls[0]
    inverts: dict[str, float] = {outfall: float(nodes.loc[outfall, "ground_elevation_m"] - 3.0)}
    pump_edges: set[tuple[str, str]] = set()
    wastewater_config = CASE_CONFIG["wastewater"]
    minimum_grade = float(wastewater_config["minimum_gravity_grade"])
    minimum_cover = float(wastewater_config["minimum_cover_m"])
    maximum_cover = float(wastewater_config["maximum_cover_m"])
    for downstream in reversed(list(nx.topological_sort(graph))):
        if downstream not in inverts:
            continue
        for upstream in graph.predecessors(downstream):
            length_m = float(graph[upstream][downstream]["length_m"])
            desired = inverts[downstream] + minimum_grade * length_m
            shallow_limit = float(nodes.loc[upstream, "ground_elevation_m"] - minimum_cover)
            deep_limit = float(nodes.loc[upstream, "ground_elevation_m"] - maximum_cover)
            if desired > shallow_limit:
                inverts[upstream] = float(nodes.loc[upstream, "ground_elevation_m"] - 2.0)
                pump_edges.add((upstream, downstream))
            else:
                inverts[upstream] = max(desired, deep_limit)
    if len(inverts) != len(nodes):
        raise ValueError("Every wastewater node must reach the outfall before invert design")

    original_cso = set(nodes.index[nodes["node_type"] == "cso"])
    nodes["is_overflow_basin"] = nodes.index.isin(original_cso)
    nodes["node_type"] = "junction"
    nodes.loc[list(original_cso), "node_type"] = "cso"
    nodes.loc[outfall, "node_type"] = "outfall"
    nodes["invert_elevation_m"] = [round(inverts[item], 3) for item in nodes.index]
    nodes["cover_depth_m"] = nodes["ground_elevation_m"] - nodes["invert_elevation_m"]
    nodes["max_depth_m"] = nodes["cover_depth_m"].clip(lower=2.0)
    nodes["hydraulic_evidence_class"] = "C"

    links["swmm_link_type"] = "conduit"
    links["pump_design_flow_m3s"] = 0.0
    links["pump_head_m"] = 0.0
    links["design_velocity_m_s"] = 0.0
    links["design_depth_ratio"] = 0.0
    pump_rows = []
    for index, row in links.iterrows():
        edge = (row["from_node"], row["to_node"])
        area = math.pi * (float(row["diameter_mm"]) / 1000.0) ** 2 / 4.0
        dry_flow = max(float(row["q_dry_m3s"]), 0.0)
        links.loc[index, "design_velocity_m_s"] = dry_flow / max(area, 1e-9)
        links.loc[index, "design_depth_ratio"] = min(1.0, math.sqrt(dry_flow / max(float(row["q_wet_m3s"]), 1e-9)))
        if edge not in pump_edges:
            links.loc[index, "link_type"] = "gravity"
            continue
        upstream, downstream = edge
        nodes.loc[upstream, "node_type"] = "pump"
        design_flow = max(1.3 * dry_flow, 0.0002)
        head = max(2.0, inverts[downstream] - inverts[upstream] + 3.0)
        links.loc[index, "link_type"] = "force_main"
        links.loc[index, "swmm_link_type"] = "pump"
        links.loc[index, "pump_design_flow_m3s"] = design_flow
        links.loc[index, "pump_head_m"] = head
        pump_rows.append(
            {
                "pump_id": f"PU{len(pump_rows)+1:03d}",
                "wet_well_node": upstream,
                "discharge_node": downstream,
                "design_flow_m3s": design_flow,
                "design_head_m": head,
                "wet_well_area_m2": 25.0,
                "control_curve": "PUMP4 depth-flow; on 0.15 m; off 0.05 m",
                "evidence_class": "C/D",
            }
        )
    frames["wastewater_nodes"] = nodes.reset_index(drop=True)
    frames["wastewater_conduits"] = links
    pd.DataFrame(pump_rows).to_csv(OUT / "wastewater_pump_design.csv", index=False)


def write_pandapipes(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    try:
        import pandapipes as ppipe

        nodes = frames["water_nodes"]
        pipes = frames["water_pipes"]
        net = ppipe.create_empty_network(name="Schweinfurt drinking-water proxy", fluid="water")
        junction = {}
        for row in nodes.itertuples():
            junction[row.node_id] = ppipe.create_junction(
                net,
                pn_bar=row.pn_bar,
                tfluid_k=283.15,
                height_m=row.height_m,
                name=row.node_id,
                geodata=(row.lon, row.lat),
            )
        source = nodes[nodes.node_type == "source"].iloc[0]
        ppipe.create_ext_grid(net, junction[source.node_id], p_bar=9.0, t_k=283.15, name="Wasserwerk source proxy")
        for row in nodes[nodes.water_q_avg_m3s > 0].itertuples():
            ppipe.create_sink(
                net,
                junction[row.node_id],
                mdot_kg_per_s=row.water_q_avg_m3s * 998.2,
                name=f"demand_{row.node_id}",
            )
        for row in pipes.itertuples():
            ppipe.create_pipe_from_parameters(
                net,
                from_junction=junction[row.from_node],
                to_junction=junction[row.to_node],
                length_km=max(row.length_km, 0.001),
                inner_diameter_mm=row.diameter_mm,
                k_mm=row.roughness_k_mm,
                loss_coefficient=row.loss_coefficient,
                name=row.pipe_id,
                geodata=json.loads(row.geometry_json),
            )
        status = {"created": True, "converged": False}
        try:
            ppipe.pipeflow(
                net,
                friction_model="colebrook",
                max_iter_hyd=100,
                tol_p=1e-4,
                tol_v=1e-4,
            )
            status["converged"] = bool(net.converged)
            if net.converged:
                nodes["result_p_bar"] = [
                    float(net.res_junction.loc[junction[row.node_id], "p_bar"]) for row in nodes.itertuples()
                ]
                pipes["result_v_mean_mps"] = [
                    float(net.res_pipe.loc[index, "v_mean_m_per_s"]) for index in net.pipe.index
                ]
        except Exception as exc:
            status["error"] = str(exc)
        ppipe.to_json(net, str(OUT / "water_pandapipes.json"))
        return status
    except Exception as exc:
        return {"created": False, "converged": False, "error": str(exc)}


def write_pandapower(
    frames: dict[str, pd.DataFrame],
    couplings: pd.DataFrame,
) -> dict[str, Any]:
    try:
        import pandapower as pp

        buses = frames["power_buses"]
        lines = frames["power_lines"]
        net = pp.create_empty_network(name="Schweinfurt 20 kV proxy", f_hz=50.0, sn_mva=100.0)
        bus_index = {}
        for row in buses.itertuples():
            bus_index[row.bus_id] = pp.create_bus(
                net, vn_kv=20.0, name=row.bus_id, geodata=(row.lon, row.lat), min_vm_pu=0.90, max_vm_pu=1.10
            )
        for row in buses[buses.root].itertuples():
            pp.create_ext_grid(net, bus_index[row.bus_id], vm_pu=1.0, name=f"110/20 kV source {row.name}")
        for row in lines.itertuples():
            pp.create_line_from_parameters(
                net,
                bus_index[row.from_bus],
                bus_index[row.to_bus],
                row.length_km,
                row.r_ohm_per_km,
                row.x_ohm_per_km,
                row.c_nf_per_km,
                row.max_i_ka,
                name=row.line_id,
                geodata=json.loads(row.geometry_json),
                in_service=not bool(row.normally_open),
                max_loading_percent=100.0,
            )
        for row in buses.itertuples():
            pp.create_load(
                net,
                bus_index[row.bus_id],
                p_mw=row.background_peak_p_mw * 0.65,
                q_mvar=row.background_peak_q_mvar * 0.65,
                name=f"background_{row.bus_id}",
            )
        selected = couplings[couplings.selected_base_case]
        for row in selected.itertuples():
            pp.create_load(
                net,
                bus_index[row.candidate_bus],
                p_mw=row.p_mw,
                q_mvar=row.q_mvar,
                name=row.asset_id,
            )
        status = {"created": True, "converged": False}
        try:
            pp.runpp(net, calculate_voltage_angles=False)
            status["converged"] = bool(net.converged)
            if net.converged:
                buses["result_vm_pu"] = [
                    float(net.res_bus.loc[bus_index[row.bus_id], "vm_pu"]) for row in buses.itertuples()
                ]
        except Exception as exc:
            status["error"] = str(exc)
        pp.to_json(net, str(OUT / "power_pandapower.json"))
        return status
    except Exception as exc:
        return {"created": False, "converged": False, "error": str(exc)}


def write_interactive_map(geojson: dict[str, Any]) -> None:
    try:
        import folium

        map_object = folium.Map(
            location=[50.045, 10.225],
            zoom_start=12,
            tiles=None,
            control_scale=True,
            prefer_canvas=True,
        )
        folium.TileLayer(
            tiles="https://tiles.openinframap.org/standard/{z}/{x}/{y}.png",
            name="OpenInfraMap",
            attr=(
                '<a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>, '
                '<a href="https://openinframap.org/copyright">OpenInfraMap</a>'
            ),
            max_zoom=18,
        ).add_to(map_object)
        folium.TileLayer(
            tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            name="OpenStreetMap fallback",
            attr="OpenStreetMap contributors",
            show=False,
        ).add_to(map_object)

        groups = {
            "Observed infrastructure": folium.FeatureGroup(name="Observed infrastructure", show=True),
            "Synthetic electricity": folium.FeatureGroup(name="Synthetic 20 kV proxy", show=True),
            "Synthetic drinking water": folium.FeatureGroup(name="Synthetic drinking water", show=True),
            "Synthetic wastewater": folium.FeatureGroup(name="Synthetic wastewater", show=True),
            "Electrical couplings": folium.FeatureGroup(name="Facility-to-grid coupling", show=True),
        }
        for group in groups.values():
            group.add_to(map_object)

        colours = {"power": "#8e44ad", "water": "#0678be", "wastewater": "#c75419", "coupling": "#2f3640"}
        for feature in geojson["features"]:
            props = feature["properties"]
            geom = feature["geometry"]
            layer = props.get("layer")
            status = props.get("status", "observed")
            if status == "observed":
                group = groups["Observed infrastructure"]
            elif layer == "power":
                group = groups["Synthetic electricity"]
            elif layer == "water":
                group = groups["Synthetic drinking water"]
            elif layer == "wastewater":
                group = groups["Synthetic wastewater"]
            else:
                group = groups["Electrical couplings"]
            tooltip = props.get("name") or props.get("asset_name") or props.get("pipe_id") or props.get("link_id") or props.get("line_id")
            popup = "<br>".join(f"<b>{k}</b>: {v}" for k, v in props.items() if k not in {"geometry_json"} and v not in {"", None})
            if geom["type"] == "Point":
                lon, lat = geom["coordinates"]
                folium.CircleMarker(
                    [lat, lon],
                    radius=4 if status == "observed" else 2.5,
                    color=colours.get(layer, "#333333"),
                    fill=True,
                    fill_opacity=0.85,
                    tooltip=tooltip,
                    popup=folium.Popup(popup, max_width=420),
                ).add_to(group)
            elif geom["type"] == "LineString":
                coords = [[lat, lon] for lon, lat in geom["coordinates"]]
                diameter = props.get("diameter_mm", 0)
                weight = 1.2 + min(4.0, float(diameter) / 180.0) if isinstance(diameter, (int, float)) else 2.0
                folium.PolyLine(
                    coords,
                    color=colours.get(layer, "#333333"),
                    weight=weight if layer in {"water", "wastewater"} else 2.0,
                    opacity=0.95 if status == "observed" else 0.72,
                    dash_array=None if status == "observed" else ("6 5" if layer == "coupling" else "3 3"),
                    tooltip=tooltip,
                    popup=folium.Popup(popup, max_width=420),
                ).add_to(group)
        folium.LayerControl(collapsed=False).add_to(map_object)
        title = """
        <div style="position:fixed;top:10px;left:50px;z-index:9999;background:rgba(255,255,255,.94);
        padding:8px 10px;border:1px solid #999;font:13px sans-serif">
        <b>Schweinfurt three-layer proxy</b><br>
        Solid = public OSM/OpenInfraMap observation; dotted = data-constrained synthetic route.<br>
        Pipe line width represents initial DN. Exact buried topology and electrical feeders are not verified.
        </div>"""
        map_object.get_root().html.add_child(folium.Element(title))
        map_object.save(OUT / "schweinfurt_three_layer_openinframap.html")
    except Exception as exc:
        (OUT / "map_error.txt").write_text(str(exc), encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-schweinfurt")
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_raw(args.refresh)
    infrastructure = load_elements("infrastructure.json")
    roads = load_elements("local_roads.json")
    buildings = load_elements("buildings.json")
    road = build_road_graph(roads)
    road_index = NodeIndex(road.nodes)
    zones = demand_zones(buildings, road_index, args.water_zones)

    water, water_source, water_facilities, wells = water_topology(road, road_index, zones, infrastructure)
    sewer, sewer_outlet, pumps, csos, sewer_meta = wastewater_topology(
        road, road_index, zones, infrastructure
    )
    buses, lines = power_topology(road, road_index, buildings, infrastructure, args.power_ties)
    couplings = asset_couplings(buses, water_facilities, wells, sewer, pumps, sewer_meta)
    frames = dataframe_exports(
        water,
        water_source,
        zones,
        water_facilities,
        wells,
        sewer,
        sewer_outlet,
        pumps,
        csos,
        sewer_meta,
        buses,
        lines,
    )
    prepare_wastewater_hydraulics(frames)
    for name, frame in frames.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
    couplings.to_csv(OUT / "facility_couplings.csv", index=False)
    zone_frame = pd.DataFrame(zones).drop(columns=["road_node", "centroid"])
    zone_frame["building_ids"] = zone_frame["building_ids"].map(lambda ids: ";".join(ids))
    zone_frame.to_csv(OUT / "demand_zones.csv", index=False)

    observed = observed_features(infrastructure)
    geojson = to_geojson(frames, couplings, observed)
    (OUT / "proxy_network.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    write_swmm(frames)
    pipe_status = write_pandapipes(frames)
    power_status = write_pandapower(frames, couplings)
    write_interactive_map(geojson)

    manifest = {
        "case_name": "Schweinfurt registered electricity-water-wastewater base layers",
        "generated_on": "2026-07-27",
        "model_bbox": MODEL_BBOX,
        "confidence_statement": "Not an actual utility network; observed and inferred elements are labelled.",
        "published_reference_accounts": {
            "water_delivery_m3_per_year": WATER_ANNUAL_M3,
            "served_population": int(CASE_CONFIG["official_anchors"]["served_population"]),
            "sewer_length_km_official": 249,
            "pressure_main_km_official_range": [13, 15],
            "pumping_stations": 13,
            "rain_overflow_basins": 23,
            "manholes": 6200,
            "wwtp_dry_weather_m3_per_day": SEWER_DRY_M3_DAY,
            "wwtp_rain_weather_m3_per_day": SEWER_WET_M3_DAY,
            "wwtp_energy_kwh_per_day": 10500,
        },
        "proxy_counts": {
            "demand_zones": len(zones),
            "water_nodes": len(frames["water_nodes"]),
            "water_pipes": len(frames["water_pipes"]),
            "water_route_length_km": round(frames["water_pipes"].length_km.sum(), 2),
            "wastewater_nodes": len(frames["wastewater_nodes"]),
            "wastewater_conduits": len(frames["wastewater_conduits"]),
            "wastewater_route_length_km": round(frames["wastewater_conduits"].length_m.sum() / 1000, 2),
            "wastewater_force_main_km": round(
                frames["wastewater_conduits"].query("link_type == 'force_main'").length_m.sum() / 1000, 2
            ),
            "synthetic_pump_stations": int((frames["wastewater_nodes"]["node_type"] == "pump").sum()),
            "synthetic_cso_nodes": int(frames["wastewater_nodes"].get("is_overflow_basin", frames["wastewater_nodes"]["node_type"] == "cso").sum()),
            "power_buses": len(frames["power_buses"]),
            "power_lines": len(frames["power_lines"]),
            "facility_assets": int(couplings.asset_id.nunique()),
            "coupling_candidates": len(couplings),
            "observed_power_features": len(observed["power"]),
            "observed_water_pipelines": len(observed["water"]),
        },
        "simulation_build": {"pandapipes": pipe_status, "pandapower": power_status},
        "case_adapter_limitations": [
            "terrain proxy with DGM/DEM",
            "pipe roughness and material priors",
            "tank/storage capacities",
            "pump curves and efficiency",
            "facility feeder assignments",
            "20 kV cable parameters and switching state",
        ],
    }
    (OUT / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name, frame in frames.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
    print(json.dumps(manifest["proxy_counts"], indent=2))
    print(json.dumps(manifest["simulation_build"], indent=2))


if __name__ == "__main__":
    main()
