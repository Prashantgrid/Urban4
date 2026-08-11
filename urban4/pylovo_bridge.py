#!/usr/bin/env python3
"""Merge building-resolved pylovo LV grids into one connected Urban4 MV/LV net."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from . import schweinfurt_base as base
from .service_topologies import MV_CABLES, _choose_cable, _union_tree, geometry


def _point(value: Any) -> tuple[float, float]:
    if isinstance(value, str):
        value = json.loads(value)
    coordinates = value.get("coordinates") if isinstance(value, dict) else None
    if not coordinates or len(coordinates) < 2:
        raise ValueError(f"Missing pylovo bus point geometry: {value!r}")
    return float(coordinates[0]), float(coordinates[1])


def _prefix_net(net, grid_id: str):
    """Add stable origin columns before pandapower reindexes during merge."""
    for table_name in ("bus", "line", "trafo", "load", "switch", "ext_grid"):
        table = getattr(net, table_name)
        if table.empty:
            continue
        table["origin_grid_id"] = grid_id
        table["source_element_index"] = table.index.astype(str)
        if "name" in table.columns:
            table["name"] = [f"{grid_id}::{value}" for value in table["name"].fillna(table_name)]
    return net


def _load_design_peak(loads: pd.DataFrame) -> pd.Series:
    if "max_p_mw" in loads:
        maximum = pd.to_numeric(loads.max_p_mw, errors="coerce").fillna(0.0)
    else:
        maximum = pd.Series(0.0, index=loads.index)
    active = pd.to_numeric(loads.get("p_mw", 0.0), errors="coerce").fillna(0.0)
    return pd.concat([maximum, active], axis=1).max(axis=1).clip(lower=0.0)


def _building_token(load_name: str) -> str:
    match = re.search(r"Load\s+([^\s:]+)", str(load_name).split("::", 1)[-1])
    return match.group(1) if match else ""


def merge_pylovo_grids(
    grid_json_paths: Iterable[Path],
    road: nx.Graph,
    output_dir: Path,
    *,
    coincident_peak_mw: float,
    source_point: tuple[float, float] | None = None,
    power_factor: float = 0.96,
) -> tuple[Any, dict[str, pd.DataFrame], dict[str, Any]]:
    """Create one citywide pandapower model from independent pylovo LV nets.

    Each input pylovo model normally has its own external grid at the HV side of
    one distribution transformer.  Those slacks are removed.  Their HV buses
    are then connected through one road-routed Urban4 MV topology and a single
    upstream source.  LV topology and equipment are otherwise retained.
    """
    import pandapower as pp
    from pandapower.toolbox import merge_nets

    paths = [Path(item) for item in grid_json_paths]
    if not paths:
        raise ValueError("No pylovo pandapower JSON grids were supplied")
    nets = []
    repaired_zero_length_lines = 0
    for index, path in enumerate(paths, 1):
        net = pp.from_json(str(path))
        if net.trafo.empty or net.load.empty:
            raise ValueError(f"{path.name} is not a building-resolved pylovo LV grid")
        degenerate = pd.to_numeric(net.line.length_km, errors="coerce").fillna(0.0) <= 0.0
        repaired_zero_length_lines += int(degenerate.sum())
        net.line.loc[degenerate, "length_km"] = 0.001
        grid_id = str(getattr(net, "grid_result_id", "") or path.stem or f"PYL{index:04d}")
        nets.append(_prefix_net(net, grid_id))
    merged = nets[0]
    for net in nets[1:]:
        merged = merge_nets(merged, net, validate=False, merge_results=False)

    # Remove the artificial one-slack-per-transformer representation before
    # adding the connected MV network.
    original_slack_count = len(merged.ext_grid)
    merged.ext_grid.drop(merged.ext_grid.index, inplace=True)
    design = _load_design_peak(merged.load)
    if design.sum() <= 0:
        raise ValueError("pylovo loads contain neither p_mw nor max_p_mw design values")
    scale = float(coincident_peak_mw) / float(design.sum())
    merged.load["service_design_p_mw"] = design
    merged.load["coincident_scale"] = scale
    merged.load["p_mw"] = design * scale
    merged.load["q_mvar"] = merged.load.p_mw * math.tan(math.acos(power_factor))
    merged.load["building_token"] = merged.load.name.map(_building_token)

    hv_buses = sorted(set(int(value) for value in merged.trafo.hv_bus))
    hv_coordinates = {bus: _point(merged.bus.at[bus, "geo"]) for bus in hv_buses}
    road_index = base.NodeIndex(road.nodes)
    snapped = {bus: road_index.nearest(point) for bus, point in hv_coordinates.items()}
    sites = list(dict.fromkeys(snapped.values()))
    if source_point is None:
        weights = []
        coordinates = []
        for bus in hv_buses:
            trafos = merged.trafo[merged.trafo.hv_bus.eq(bus)]
            lv_buses = set(int(value) for value in trafos.lv_bus)
            assigned = merged.load[merged.load.bus.isin(lv_buses)].p_mw.sum()
            weights.append(max(float(assigned), 0.001))
            coordinates.append(snapped[bus])
        centre = (
            float(np.average([point[0] for point in coordinates], weights=weights)),
            float(np.average([point[1] for point in coordinates], weights=weights)),
        )
        source = road_index.nearest(centre)
    else:
        source = road_index.nearest(source_point)
    tree = _union_tree(road, source, sites)

    # Site demand includes all downstream LV loads, not only loads directly on
    # the transformer LV bus.
    site_peak = {site: 0.0 for site in sites}
    for bus in hv_buses:
        grid_ids = set(merged.trafo.loc[merged.trafo.hv_bus.eq(bus), "origin_grid_id"].astype(str))
        site_peak[snapped[bus]] += float(merged.load[merged.load.origin_grid_id.astype(str).isin(grid_ids)].p_mw.sum())
    parent = {child: ancestor for ancestor, child in nx.bfs_edges(tree, source)}
    subtree = {node: float(site_peak.get(node, 0.0)) for node in tree}
    for node in reversed(list(nx.bfs_tree(tree, source).nodes)):
        if node != source:
            subtree[parent[node]] += subtree[node]

    road_bus: dict[tuple[float, float], int] = {}
    for node in tree:
        road_bus[node] = pp.create_bus(
            merged, vn_kv=20.0,
            name="URBAN4_MV_SOURCE" if node == source else f"URBAN4_MV_J{len(road_bus) + 1:05d}",
            geodata=node, type="b",
        )
        merged.bus.at[road_bus[node], "origin_grid_id"] = "URBAN4_MV"
        merged.bus.at[road_bus[node], "source_element_index"] = "URBAN4"
    pp.create_ext_grid(merged, road_bus[source], vm_pu=1.03, name="URBAN4_SINGLE_UPSTREAM_SOURCE")

    mv_rows: list[dict[str, Any]] = []
    for line_number, (child, ancestor) in enumerate(parent.items(), 1):
        data = tree[ancestor][child]
        power = max(float(subtree[child]), 0.001)
        current = power / (math.sqrt(3.0) * 20.0 * power_factor)
        size, ampacity, resistance, parallel, drop = _choose_cable(
            current, float(data["length_km"]), 20.0, MV_CABLES, 0.03, max_parallel=4
        )
        name = f"URBAN4_MV_L{line_number:05d}"
        pp.create_line_from_parameters(
            merged, road_bus[ancestor], road_bus[child], max(float(data["length_km"]), 0.001),
            resistance, 0.08, 250.0, ampacity, parallel=parallel, name=name,
            geodata=geometry(data, ancestor, child), origin_grid_id="URBAN4_MV",
            source_element_index="URBAN4",
        )
        mv_rows.append({
            "line_id": name, "from_bus": int(road_bus[ancestor]), "to_bus": int(road_bus[child]),
            "length_km": float(data["length_km"]), "cable_size_mm2": size,
            "parallel_circuits": parallel, "design_power_mw": power,
            "design_current_ka": current, "design_voltage_drop_pu": drop,
            "geometry_json": json.dumps(geometry(data, ancestor, child), separators=(",", ":")),
        })

    spur_rows: list[dict[str, Any]] = []
    for spur_number, hv_bus in enumerate(hv_buses, 1):
        site = snapped[hv_bus]
        point = hv_coordinates[hv_bus]
        length = max(base.distance_km(site, point), 0.001)
        grid_ids = set(merged.trafo.loc[merged.trafo.hv_bus.eq(hv_bus), "origin_grid_id"].astype(str))
        power = max(float(merged.load[merged.load.origin_grid_id.astype(str).isin(grid_ids)].p_mw.sum()), 0.001)
        current = power / (math.sqrt(3.0) * 20.0 * power_factor)
        size, ampacity, resistance, parallel, drop = _choose_cable(
            current, length, 20.0, MV_CABLES, 0.01, max_parallel=2
        )
        name = f"URBAN4_MV_TR_SPUR{spur_number:04d}"
        pp.create_line_from_parameters(
            merged, road_bus[site], hv_bus, length, resistance, 0.08, 250.0, ampacity,
            parallel=parallel, name=name,
            geodata=None if base.distance_km(site, point) < 1e-6 else [site, point],
            origin_grid_id="URBAN4_MV", source_element_index="URBAN4",
        )
        spur_rows.append({
            "line_id": name, "from_bus": int(road_bus[site]), "to_bus": int(hv_bus),
            "grid_ids": ";".join(sorted(grid_ids)), "length_km": length,
            "cable_size_mm2": size, "parallel_circuits": parallel,
            "design_power_mw": power, "geometry_json": json.dumps([site, point], separators=(",", ":")),
        })

    # Normalize pandas extension dtypes introduced by mixing upstream pylovo
    # tables with newly created Urban4 elements.  A file round-trip is used
    # because it follows exactly the loading path of the released model and is
    # more stable across pandapower versions than an in-memory JSON-string
    # round-trip for extension dtypes.
    output_dir.mkdir(parents=True, exist_ok=True)
    precheck_path = output_dir / ".pylovo_urban4_precheck.json"
    pp.to_json(merged, str(precheck_path))
    merged = pp.from_json(str(precheck_path))

    solver_errors: list[str] = []
    # Newton--Raphson is attempted first.  Some upstream pylovo examples carry
    # non-contiguous auxiliary indices that the BFSW implementation rejects;
    # a failed BFSW attempt can also leave cached matrices unsuitable for the
    # next algorithm in the same process.
    for algorithm in ("nr", "iwamoto_nr", "bfsw"):
        try:
            # pylovo transformer records can carry non-zero vector-group phase
            # shifts, so pandapower's automatic/DC-aware initialization is
            # required; a flat-angle start can fail on an otherwise valid net.
            pp.runpp(merged, algorithm=algorithm, numba=False, init="auto", max_iteration=100)
            if merged.converged:
                break
        except Exception as exc:
            solver_errors.append(f"{algorithm}: {type(exc).__name__}: {exc}")
    if not merged.converged:
        raise RuntimeError(
            "Connected pylovo/Urban4 MV-LV pandapower model did not converge; "
            + " | ".join(solver_errors)
        )
    precheck_path.unlink(missing_ok=True)

    graph = nx.Graph()
    graph.add_edges_from((int(row.from_bus), int(row.to_bus)) for row in merged.line.itertuples() if row.in_service)
    graph.add_edges_from((int(row.hv_bus), int(row.lv_bus)) for row in merged.trafo.itertuples() if row.in_service)
    graph.add_nodes_from(int(index) for index in merged.bus.index[merged.bus.in_service])
    component_count = nx.number_connected_components(graph)
    if component_count != 1 or len(merged.ext_grid) != 1:
        raise AssertionError(f"Expected one energized topology, found {component_count} components and {len(merged.ext_grid)} slacks")

    pp.to_json(merged, str(output_dir / "electricity_pandapower_pylovo_urban4.json"))
    bus_export = merged.bus.copy().reset_index(names="bus_index")
    line_export = merged.line.copy().reset_index(names="line_index")
    trafo_export = merged.trafo.copy().reset_index(names="transformer_index")
    load_export = merged.load.copy().reset_index(names="load_index")
    for name, frame in (
        ("electricity_nodes.csv", bus_export), ("electricity_links.csv", line_export),
        ("electricity_transformers.csv", trafo_export), ("electricity_building_loads.csv", load_export),
    ):
        frame.to_csv(output_dir / name, index=False)
    pd.DataFrame(mv_rows).to_csv(output_dir / "electricity_mv_backbone.csv", index=False)
    pd.DataFrame(spur_rows).to_csv(output_dir / "electricity_mv_transformer_spurs.csv", index=False)
    summary = {
        "created": True, "converged": True,
        "input_pylovo_grids": len(paths), "removed_pylovo_ext_grids": original_slack_count,
        "repaired_zero_length_lv_lines": repaired_zero_length_lines,
        "final_ext_grids": len(merged.ext_grid), "graph_components": component_count,
        "buses": len(merged.bus), "lines": len(merged.line), "transformers": len(merged.trafo),
        "loads": len(merged.load), "building_tokens": int(merged.load.building_token.replace("", np.nan).nunique()),
        "coincident_peak_mw": float(merged.load.p_mw.sum()), "coincidence_scale": scale,
        "minimum_voltage_pu": float(merged.res_bus.vm_pu.min()),
        "maximum_voltage_pu": float(merged.res_bus.vm_pu.max()),
        "maximum_line_loading_percent": float(merged.res_line.loading_percent.max()),
        "maximum_transformer_loading_percent": float(merged.res_trafo.loading_percent.max()),
        "solver": "pandapower runpp AC power flow; not OPF",
    }
    (output_dir / "pylovo_urban4_merge_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return merged, {
        "buses": bus_export, "lines": line_export, "transformers": trafo_export,
        "loads": load_export, "mv_backbone": pd.DataFrame(mv_rows),
        "mv_spurs": pd.DataFrame(spur_rows),
    }, summary
