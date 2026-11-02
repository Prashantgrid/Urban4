"""Spatial external-validation helpers for Urban4.

These routines separate directional corridor coverage from a symmetric
diagnostic and from a length-matched road-network null model. They do not
download third-party utility data; callers provide projected Shapely line
geometries with a documented common CRS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union


@dataclass(frozen=True)
class CorridorCoverage:
    installed_to_generated: float
    generated_to_installed: float

    @property
    def harmonic_mean(self) -> float:
        a = self.installed_to_generated
        b = self.generated_to_installed
        return 0.0 if a <= 0.0 or b <= 0.0 else 2.0 * a * b / (a + b)


def _merged(lines: Iterable) -> object:
    geoms = [geom for geom in lines if geom is not None and not geom.is_empty]
    return unary_union(geoms) if geoms else GeometryCollection()


def directed_corridor_coverage(reference_lines: Iterable, candidate_lines: Iterable, buffer_m: float) -> float:
    """Fraction of reference-route length lying within buffer_m of candidate."""
    if buffer_m < 0:
        raise ValueError("buffer_m must be non-negative")
    reference = _merged(reference_lines)
    candidate = _merged(candidate_lines)
    denominator = float(reference.length)
    if denominator <= 0.0 or candidate.is_empty:
        return 0.0
    covered = reference.intersection(candidate.buffer(float(buffer_m)))
    return float(np.clip(covered.length / denominator, 0.0, 1.0))


def symmetric_corridor_coverage(installed_lines: Iterable, generated_lines: Iterable, buffer_m: float) -> CorridorCoverage:
    """Return both directional corridor coverages for the same buffer."""
    installed = list(installed_lines)
    generated = list(generated_lines)
    return CorridorCoverage(
        installed_to_generated=directed_corridor_coverage(installed, generated, buffer_m),
        generated_to_installed=directed_corridor_coverage(generated, installed, buffer_m),
    )


def length_matched_random_subset(
    road_edges: list,
    target_length: float,
    *,
    rng: np.random.Generator,
) -> list:
    """Draw a road-edge subset whose accumulated length first reaches target."""
    if target_length <= 0.0 or not road_edges:
        return []
    order = rng.permutation(len(road_edges))
    chosen = []
    length = 0.0
    for index in order:
        geom = road_edges[int(index)]
        if geom is None or geom.is_empty:
            continue
        chosen.append(geom)
        length += float(geom.length)
        if length >= target_length:
            break
    return chosen


def corridor_null_distribution(
    installed_lines: Iterable,
    generated_lines: Iterable,
    road_edges: list,
    buffer_m: float,
    *,
    repetitions: int = 100,
    seed: int = 20260803,
) -> dict:
    """Compare observed symmetric coverage with equal-length random road subsets."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    installed = list(installed_lines)
    generated = list(generated_lines)
    observed = symmetric_corridor_coverage(installed, generated, buffer_m)
    generated_length = float(_merged(generated).length)
    rng = np.random.default_rng(seed)
    null_values = []
    for _ in range(repetitions):
        null = length_matched_random_subset(road_edges, generated_length, rng=rng)
        null_values.append(
            symmetric_corridor_coverage(installed, null, buffer_m).harmonic_mean
        )
    null_array = np.asarray(null_values, dtype=float)
    observed_value = observed.harmonic_mean
    std = float(null_array.std(ddof=1)) if len(null_array) > 1 else 0.0
    z_score = (
        (observed_value - float(null_array.mean())) / std
        if std > 0.0 else math.nan
    )
    percentile = float(100.0 * np.mean(null_array <= observed_value))
    return {
        "installed_to_generated": observed.installed_to_generated,
        "generated_to_installed": observed.generated_to_installed,
        "symmetric_harmonic_mean": observed_value,
        "null_mean": float(null_array.mean()),
        "null_std": std,
        "observed_percentile": percentile,
        "z_score": z_score,
        "repetitions": int(repetitions),
        "seed": int(seed),
        "null_definition": "equal-total-length subset of the same road-edge universe",
    }
