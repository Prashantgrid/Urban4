#!/usr/bin/env python3
"""Route-aware partial-connection selection for synthetic district heating.

This module deliberately does not model tariffs, household behaviour, or
investment timing.  It provides a deterministic, auditable approximation to a
quota-constrained prize-collecting Steiner forest: annual building heat demand
is the prize and newly added road length is the routing penalty.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from . import schweinfurt_base as base


@dataclass(frozen=True)
class RouteAwareHeatConfig:
    demand_exponent: float = 1.0
    minimum_increment_km: float = 0.005
    spatial_quota_level: str = "cell"  # cell or territory
    selection_rounding: str = "largest_remainder"

    def validate(self) -> None:
        if self.demand_exponent <= 0:
            raise ValueError("demand_exponent must be positive")
        if self.minimum_increment_km <= 0:
            raise ValueError("minimum_increment_km must be positive")
        if self.spatial_quota_level not in {"cell", "territory"}:
            raise ValueError("spatial_quota_level must be 'cell' or 'territory'")


def allocate_integer_quotas(
    weights: dict[Any, float],
    total: int,
    ceilings: dict[Any, int],
) -> dict[Any, int]:
    """Allocate an integer total with largest remainders and hard ceilings."""
    if total < 0:
        raise ValueError("total cannot be negative")
    if not weights:
        return {}
    keys = list(weights)
    clean = {key: max(float(weights[key]), 0.0) for key in keys}
    if sum(clean.values()) <= 0:
        clean = {key: 1.0 for key in keys}
    target = min(total, sum(max(int(ceilings.get(key, 0)), 0) for key in keys))
    denominator = sum(clean.values())
    raw = {key: target * clean[key] / denominator for key in keys}
    quota = {
        key: min(int(math.floor(raw[key])), max(int(ceilings.get(key, 0)), 0))
        for key in keys
    }
    remaining = target - sum(quota.values())
    order = sorted(
        keys,
        key=lambda key: (
            -(raw[key] - math.floor(raw[key])),
            -clean[key],
            str(key),
        ),
    )
    while remaining > 0:
        progressed = False
        for key in order:
            if quota[key] < max(int(ceilings.get(key, 0)), 0):
                quota[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return quota


def _single_source_predecessor(
    road: nx.Graph,
    source: tuple[float, float],
    targets: Iterable[tuple[float, float]],
) -> tuple[dict[tuple[float, float], float], dict[tuple[float, float], tuple[float, float]]]:
    """Dijkstra search that stops once all requested target nodes are settled."""
    remaining = set(targets)
    distances: dict[tuple[float, float], float] = {source: 0.0}
    predecessor: dict[tuple[float, float], tuple[float, float]] = {}
    queue: list[tuple[float, tuple[float, float]]] = [(0.0, source)]
    settled: set[tuple[float, float]] = set()
    while queue and remaining:
        distance, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        remaining.discard(node)
        for neighbour, data in road[node].items():
            candidate = distance + float(data.get("length_km", 1.0))
            if candidate < distances.get(neighbour, math.inf):
                distances[neighbour] = candidate
                predecessor[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    return distances, predecessor


def _path_to_root(
    target: tuple[float, float],
    root: tuple[float, float],
    predecessor: dict[tuple[float, float], tuple[float, float]],
) -> list[tuple[float, float]]:
    if target == root:
        return [root]
    if target not in predecessor:
        return []
    path = [target]
    while path[-1] != root:
        parent = predecessor.get(path[-1])
        if parent is None:
            return []
        path.append(parent)
    path.reverse()
    return path


def _canonical_edge(
    u: tuple[float, float], v: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (u, v) if u <= v else (v, u)


def select_route_aware_heat_connections(
    road: nx.Graph,
    candidates: pd.DataFrame,
    *,
    target_count: int,
    config: RouteAwareHeatConfig = RouteAwareHeatConfig(),
    root_by_territory: dict[Any, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Select a spatially distributed, route-aware synthetic customer set.

    Parameters
    ----------
    road:
        Undirected admissible corridor graph with ``length_km`` edge weights.
    candidates:
        Heat-suitable buildings containing immutable IDs, road attachments,
        annual heat demand, service length, territory IDs, and cell IDs.
    target_count:
        Public contract-equivalent quota or context-case connection ceiling.

    Notes
    -----
    The algorithm allocates the quota across suitable cells (or territories)
    in proportion to annual heat, then grows one rooted shortest-path forest per
    territory.  Inside each spatial quota group, a building is added according
    to

        score_b = H_b ** alpha / (Delta L_b(F) + L_b,service),

    where Delta L is only the road length not already present in the forest.
    The fixed quota keeps observed contract totals separate from behavioural or
    tariff assumptions.
    """
    config.validate()
    if candidates.empty or target_count <= 0:
        return candidates.iloc[0:0].copy(), {
            "selection_policy": "route_aware_prize_collecting_forest",
            "selected_connection_count": 0,
            "reason": "no candidate or zero quota",
            "configuration": asdict(config),
        }, pd.DataFrame()
    required = {
        "building_id", "road_node", "road_lon", "road_lat",
        "service_length_km", "heat_candidate_mwh_year",
        "heat_territory_id", "heat_cell_x", "heat_cell_y",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise KeyError(f"route-aware heat selection missing columns: {missing}")
    frame = candidates.copy()
    target_count = min(int(target_count), len(frame))
    if target_count <= 0:
        return frame.iloc[0:0].copy(), {
            "selection_policy": "route_aware_prize_collecting_forest",
            "selected_connection_count": 0,
            "reason": "quota reduced to zero",
            "configuration": asdict(config),
        }, pd.DataFrame()

    frame["heat_cell_key"] = list(zip(frame.heat_cell_x.astype(int), frame.heat_cell_y.astype(int)))
    territory_weights = frame.groupby("heat_territory_id").heat_candidate_mwh_year.sum().to_dict()
    territory_ceilings = frame.groupby("heat_territory_id").size().astype(int).to_dict()
    territory_quotas = allocate_integer_quotas(territory_weights, target_count, territory_ceilings)

    selected_indices: list[int] = []
    audit_rows: list[dict[str, Any]] = []
    total_forest_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    root_by_territory_map: dict[int, tuple[float, float]] = {}

    for territory_id, territory in frame.groupby("heat_territory_id", sort=True):
        territory = territory.copy()
        territory_quota = int(territory_quotas.get(territory_id, 0))
        if territory_quota <= 0:
            continue
        weights = np.maximum(territory.heat_candidate_mwh_year.to_numpy(float), 1e-9)
        centre = (
            float(np.average(territory.road_lon, weights=weights)),
            float(np.average(territory.road_lat, weights=weights)),
        )
        supplied_root = (root_by_territory or {}).get(territory_id)
        root = (
            supplied_root if supplied_root in road
            else base.NodeIndex(road.nodes).nearest(centre)
        )
        root_by_territory_map[int(territory_id)] = root
        unique_targets = list(dict.fromkeys(territory.road_node.tolist()))
        distance, predecessor = _single_source_predecessor(road, root, unique_targets)

        path_nodes_by_index: dict[int, list[tuple[float, float]]] = {}
        path_edges_by_index: dict[int, tuple[tuple[tuple[float, float], tuple[float, float]], ...]] = {}
        path_length_by_index: dict[int, float] = {}
        for index, row in territory.iterrows():
            path = _path_to_root(row.road_node, root, predecessor)
            if not path:
                continue
            edges = tuple(_canonical_edge(u, v) for u, v in zip(path[:-1], path[1:]))
            path_nodes_by_index[index] = path
            path_edges_by_index[index] = edges
            path_length_by_index[index] = float(distance.get(row.road_node, 0.0))

        if config.spatial_quota_level == "cell":
            group_col = "heat_cell_key"
        else:
            group_col = "heat_territory_id"
        group_weights = territory.groupby(group_col).heat_candidate_mwh_year.sum().to_dict()
        group_ceilings = territory.groupby(group_col).size().astype(int).to_dict()
        group_quotas = allocate_integer_quotas(group_weights, territory_quota, group_ceilings)
        remaining_by_group: dict[Any, set[int]] = {
            key: set(group.index).intersection(path_nodes_by_index)
            for key, group in territory.groupby(group_col)
        }
        selected_per_group = defaultdict(int)
        forest_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
        forest_nodes: set[tuple[float, float]] = {root}

        active_groups = [key for key, quota in group_quotas.items() if quota > 0]
        while active_groups and sum(selected_per_group.values()) < territory_quota:
            progressed = False
            # Round-robin across cells prevents one compact block from consuming
            # the full public quota while retaining route sharing within the
            # territory-wide forest.
            active_groups = sorted(
                active_groups,
                key=lambda key: (
                    -(group_quotas[key] - selected_per_group[key]),
                    -float(group_weights.get(key, 0.0)),
                    str(key),
                ),
            )
            for group_key in list(active_groups):
                if selected_per_group[group_key] >= group_quotas[group_key]:
                    continue
                available = remaining_by_group.get(group_key, set())
                if not available:
                    continue
                best: tuple[float, float, str, int, float, float] | None = None
                for index in available:
                    row = frame.loc[index]
                    edges = path_edges_by_index[index]
                    new_route_km = sum(
                        float(road[u][v].get("length_km", 1.0))
                        for u, v in edges if (u, v) not in forest_edges
                    )
                    service_km = max(float(row.service_length_km), 0.0)
                    denominator = max(config.minimum_increment_km, new_route_km + service_km)
                    score = float(row.heat_candidate_mwh_year) ** config.demand_exponent / denominator
                    candidate_tuple = (
                        -score,
                        denominator,
                        str(row.building_id),
                        int(index),
                        new_route_km,
                        score,
                    )
                    if best is None or candidate_tuple < best:
                        best = candidate_tuple
                if best is None:
                    continue
                _, denominator, _, index, new_route_km, score = best
                row = frame.loc[index]
                edges = path_edges_by_index[index]
                forest_edges.update(edges)
                forest_nodes.update(path_nodes_by_index[index])
                available.remove(index)
                selected_indices.append(index)
                selected_per_group[group_key] += 1
                audit_rows.append({
                    "selection_order": len(selected_indices),
                    "building_id": row.building_id,
                    "heat_territory_id": int(territory_id),
                    "heat_cell_x": int(row.heat_cell_x),
                    "heat_cell_y": int(row.heat_cell_y),
                    "annual_heat_mwh_unscaled": float(row.heat_candidate_mwh_year),
                    "service_length_km": float(row.service_length_km),
                    "root_path_length_km": path_length_by_index[index],
                    "incremental_trench_km": new_route_km,
                    "marginal_route_plus_service_km": denominator,
                    "route_aware_score_mwh_per_km": score,
                    "provisional_root_lon": root[0],
                    "provisional_root_lat": root[1],
                })
                progressed = True
            active_groups = [
                key for key in active_groups
                if selected_per_group[key] < group_quotas[key]
                and bool(remaining_by_group.get(key))
            ]
            if not progressed:
                break
        total_forest_edges.update(forest_edges)

    selected = frame.loc[selected_indices].copy()
    selected["heat_selection_policy"] = "route_aware_prize_collecting_forest"
    audit = pd.DataFrame(audit_rows)
    if not audit.empty:
        audit_index = audit.set_index("building_id")
        selected["incremental_trench_km"] = selected.building_id.map(audit_index.incremental_trench_km)
        selected["route_aware_score_mwh_per_km"] = selected.building_id.map(
            audit_index.route_aware_score_mwh_per_km
        )
        selected["provisional_root_lon"] = selected.heat_territory_id.astype(int).map(
            {key: value[0] for key, value in root_by_territory_map.items()}
        )
        selected["provisional_root_lat"] = selected.heat_territory_id.astype(int).map(
            {key: value[1] for key, value in root_by_territory_map.items()}
        )
    total_route_km = sum(
        float(road[u][v].get("length_km", 1.0)) for u, v in total_forest_edges
    )
    metadata = {
        "selection_policy": "route_aware_prize_collecting_forest",
        "objective_interpretation": "annual heat prize divided by newly added trench plus service length; non-monetary",
        "selected_connection_count": len(selected),
        "requested_connection_count": target_count,
        "provisional_forest_trench_km": total_route_km,
        "selected_unscaled_annual_heat_mwh": float(selected.heat_candidate_mwh_year.sum()),
        "territory_quotas": {str(key): int(value) for key, value in territory_quotas.items()},
        "configuration": asdict(config),
    }
    return selected, metadata, audit
