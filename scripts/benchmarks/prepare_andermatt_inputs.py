#!/usr/bin/env python3
"""Reconstruct the controlled Andermatt case from the frozen public snapshots."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import geopandas as gpd
from shapely import union_all
from shapely.geometry import LineString, box

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.benchmarks.run_andermatt_topotherm import json_write


def main() -> None:
    data = ROOT / "data/andermatt_benchmark"
    owner = "Netzgesellschaft Andermatt AG"
    pts = gpd.read_file(data / "fwa_leitungspunkt_einspeisung.geojson").explode(index_parts=False)
    pts = pts.loc[pts.eigentuemer.eq(owner) & pts.art.eq("Hausuebergabestation")].copy()
    # The window is set by supplied customer locations, not by reference route length.
    minx, miny, maxx, maxy = pts.total_bounds
    domain = box(minx - 250., miny - 250., maxx + 250., maxy + 250.)
    lines = gpd.read_file(data / "fwa_trasse_linie.geojson")
    lines = lines.loc[lines.eigentuemer.eq(owner) & lines.status.eq("in_Betrieb")]
    reference = union_all([g.intersection(domain) for g in lines.geometry])
    northern_boundary = LineString([(domain.bounds[0], domain.bounds[3]),
                                    (domain.bounds[2], domain.bounds[3])])
    inlet = reference.intersection(northern_boundary)
    if inlet.geom_type != "Point":
        raise ValueError("The frozen case must have one unambiguous northern feed")
    pts["customer_id"] = pts.obj_id
    pts["peak_kw"], pts["flh"] = 50., 1600.
    target = data / "common_inputs.gpkg"
    pts[["customer_id", "peak_kw", "flh", "geometry"]].to_file(target, layer="customers")
    gpd.GeoDataFrame({"source_id":["northern_feed_interface"]},geometry=[inlet],crs=2056).to_file(target,layer="source")
    gpd.GeoDataFrame({"area_km2":[domain.area/1e6]},geometry=[domain],crs=2056).to_file(target,layer="boundary")
    gpd.GeoDataFrame({"reference":["active_cadastral_local"]},geometry=[reference],crs=2056).explode(index_parts=False).to_file(target,layer="installed_reference")
    previous = json.loads((data / "case.json").read_text())
    previous.update(customers=len(pts), domain_bounds_2056=list(domain.bounds),
                    source_xy_2056=[inlet.x,inlet.y], reference_status_filter="in_Betrieb")
    json_write(data / "case.json",previous)
    print(f"Prepared {len(pts)} observed transfer points; active mapped routes: {reference.length/1000:.6f} km")


if __name__ == "__main__":
    main()
