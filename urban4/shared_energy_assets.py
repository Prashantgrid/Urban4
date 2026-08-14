"""Shared conversion-asset interfaces for coupled Urban4 studies.

The module keeps a documented physical asset common to two sector layers without
inventing a conversion curve that is not supported by data.  A shared asset can
therefore expose electrical and thermal ports, a common availability state, and
optional prescribed operating points.  Detailed CHP, power-to-heat, electrolyzer
or other conversion models can replace the prescribed operating-point adapter in
later studies without changing the interface contract.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import schweinfurt_base as base


RELATION = "shared_conversion_asset"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _nearest_power_node(
    point: tuple[float, float],
    power_nodes: pd.DataFrame,
    *,
    minimum_voltage_kv: float,
) -> dict[str, Any]:
    """Return a deterministic synthetic electrical attachment point.

    The connection is a generated-model attachment, not a claim about the real
    utility interconnection.  Preference is given to MV/source-side nodes and
    then to any node at or above the configured minimum voltage.
    """
    if power_nodes.empty:
        raise ValueError("Cannot attach a shared asset to an empty electricity model")

    frame = power_nodes.copy()
    if "voltage_kv" not in frame.columns or "bus_id" not in frame.columns:
        raise ValueError("Power-node table must contain bus_id and voltage_kv")

    preferred_types = {
        "mv_source",
        "mv_junction",
        "transformer_mv_bus",
        "mv_customer",
    }
    preferred = frame[
        frame.get("node_type", pd.Series(index=frame.index, dtype=str))
        .astype(str)
        .isin(preferred_types)
        & (frame.voltage_kv.astype(float) >= float(minimum_voltage_kv))
    ]
    eligible = preferred if not preferred.empty else frame[
        frame.voltage_kv.astype(float) >= float(minimum_voltage_kv)
    ]
    if eligible.empty:
        raise ValueError(
            f"No generated electrical node at or above {minimum_voltage_kv:g} kV"
        )

    return min(
        eligible.itertuples(index=False),
        key=lambda row: (
            base.distance_km(point, (float(row.lon), float(row.lat))),
            str(row.bus_id),
        ),
    )._asdict()


def _thermal_source_candidates(heat_nodes: pd.DataFrame) -> pd.DataFrame:
    if heat_nodes.empty:
        return heat_nodes
    if "node_type" in heat_nodes.columns:
        plants = heat_nodes[heat_nodes.node_type.astype(str).eq("plant")]
        if not plants.empty:
            return plants
    return heat_nodes


def _thermal_id_column(frame: pd.DataFrame) -> str:
    for candidate in ("node_id", "map_id", "source_id", "asset_id"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Heat-source table has no node/source identifier column")


def _nearest_heat_source(
    point: tuple[float, float],
    heat_nodes: pd.DataFrame,
    *,
    preferred_source_id: str | None = None,
) -> tuple[str, float, str]:
    """Return thermal source ID, distance, and the connection basis."""
    candidates = _thermal_source_candidates(heat_nodes)
    if candidates.empty:
        raise ValueError("Cannot attach a shared asset to an empty heat-source model")
    if not {"lon", "lat"}.issubset(candidates.columns):
        raise ValueError("Heat-source table must contain lon and lat")

    id_col = _thermal_id_column(candidates)
    if preferred_source_id:
        exact = candidates[candidates[id_col].astype(str).eq(str(preferred_source_id))]
        if not exact.empty:
            row = exact.iloc[0]
            distance = base.distance_km(
                point, (float(row.lon), float(row.lat))
            )
            return str(row[id_col]), float(distance), "configured heat-source identifier"

    row = min(
        candidates.itertuples(index=False),
        key=lambda item: (
            base.distance_km(point, (float(item.lon), float(item.lat))),
            str(getattr(item, id_col)),
        ),
    )
    source_id = str(getattr(row, id_col))
    distance = base.distance_km(point, (float(row.lon), float(row.lat)))
    return source_id, float(distance), "nearest generated heat-source node to documented facility location"


def configured_shared_assets(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return enabled shared-energy assets from the case configuration."""
    raw = config.get("shared_energy_assets", [])
    if isinstance(raw, Mapping):
        raw = list(raw.values())
    if not isinstance(raw, list):
        raise ValueError("shared_energy_assets must be a list or mapping")
    assets: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each shared-energy asset must be a mapping")
        asset = dict(item)
        if bool(asset.get("enabled", True)):
            assets.append(asset)
    return assets


def build_shared_asset_interfaces(
    config: Mapping[str, Any],
    power_nodes: pd.DataFrame,
    heat_nodes: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build interface rows for documented shared conversion assets.

    No electrical or thermal setpoint is inferred.  If a setpoint is absent, the
    corresponding port remains availability-only until a later study supplies a
    dispatch model or explicit operating point.
    """
    rows: list[dict[str, Any]] = []
    for asset in configured_shared_assets(config):
        asset_id = str(asset.get("asset_id", "")).strip()
        name = str(asset.get("name", "")).strip()
        asset_type = str(asset.get("asset_type", "shared_conversion_asset")).strip()
        point_raw = asset.get("documented_coordinate")
        if not asset_id or not name:
            raise ValueError("Shared-energy asset requires asset_id and name")
        if not isinstance(point_raw, (list, tuple)) or len(point_raw) != 2:
            raise ValueError(f"Shared asset {asset_id} requires documented_coordinate=[lon, lat]")
        point = (float(point_raw[0]), float(point_raw[1]))

        electrical = dict(asset.get("electrical_port", {}))
        thermal = dict(asset.get("thermal_port", {}))
        if not electrical or not thermal:
            raise ValueError(f"Shared asset {asset_id} requires electrical_port and thermal_port")

        minimum_kv = float(electrical.get("minimum_connection_voltage_kv", 10.0))
        power_row = _nearest_power_node(point, power_nodes, minimum_voltage_kv=minimum_kv)
        power_distance = base.distance_km(
            point, (float(power_row["lon"]), float(power_row["lat"]))
        )

        thermal_source_id, thermal_distance, thermal_basis = _nearest_heat_source(
            point,
            heat_nodes,
            preferred_source_id=(
                None if _is_missing(thermal.get("source_id"))
                else str(thermal.get("source_id"))
            ),
        )

        p_setpoint = electrical.get("prescribed_active_power_mw")
        q_setpoint = thermal.get("prescribed_heat_output_mw")
        p_value = np.nan if _is_missing(p_setpoint) else float(p_setpoint)
        q_value = np.nan if _is_missing(q_setpoint) else float(q_setpoint)

        dispatch_mode = str(asset.get("dispatch_mode", "availability_only"))
        if dispatch_mode == "availability_only" and (
            not np.isnan(p_value) or not np.isnan(q_value)
        ):
            raise ValueError(
                f"Shared asset {asset_id}: availability_only cannot contain prescribed setpoints"
            )
        if dispatch_mode not in {"availability_only", "prescribed_operating_point"}:
            raise ValueError(
                f"Shared asset {asset_id}: unsupported dispatch_mode={dispatch_mode!r}"
            )

        evidence_role = str(
            asset.get(
                "evidence_role",
                "documented shared asset; generated sector connection points",
            )
        )
        rows.append(
            {
                "building_id": "",
                "from_sector": "electricity",
                "from_id": str(power_row["bus_id"]),
                "to_sector": "district_heating",
                "to_id": thermal_source_id,
                "relation": RELATION,
                "facility_type": asset_type,
                "exchange_variable": "common_availability_and_optional_prescribed_operating_point",
                "nominal_power_mw": p_value,
                "conversion": np.nan,
                "unit": "MW_e, MW_th, boolean",
                "directionality": "shared_asset_state",
                "connected_bus_type": str(power_row.get("node_type", "")),
                "connected_bus_voltage_kv": float(power_row["voltage_kv"]),
                "connection_distance_km": float(power_distance),
                "forward_duty_model": (
                    "prescribed electrical injection only when supplied; no inferred CHP heat-to-power curve"
                ),
                "reverse_response_model": (
                    "one common scenario availability state gates both electrical and thermal ports"
                ),
                "evidence_role": evidence_role,
                "shared_asset_id": asset_id,
                "shared_asset_name": name,
                "shared_asset_type": asset_type,
                "shared_asset_available": bool(asset.get("available", True)),
                "shared_dispatch_mode": dispatch_mode,
                "electrical_setpoint_mw": p_value,
                "thermal_setpoint_mw": q_value,
                "electrical_nameplate_mw": (
                    np.nan
                    if _is_missing(electrical.get("nameplate_active_power_mw"))
                    else float(electrical["nameplate_active_power_mw"])
                ),
                "thermal_nameplate_mw": (
                    np.nan
                    if _is_missing(thermal.get("nameplate_heat_output_mw"))
                    else float(thermal["nameplate_heat_output_mw"])
                ),
                "thermal_source_id": thermal_source_id,
                "thermal_connection_distance_km": float(thermal_distance),
                "electrical_connection_basis": (
                    "nearest compatible generated MV node to documented facility location; not a claimed real utility connection"
                ),
                "thermal_connection_basis": thermal_basis,
            }
        )
    return rows


def resolve_shared_asset_states(
    shared_interfaces: pd.DataFrame,
    availability_overrides: Mapping[str, bool] | None = None,
) -> pd.DataFrame:
    """Resolve a shared asset's common state for a coupled operating point.

    Positive electrical setpoints are generation injections.  Missing setpoints
    remain missing when the asset is available; Urban4 does not infer dispatch.
    If the common asset is unavailable, any configured electrical and thermal
    output is set to zero and both ports are marked unavailable.
    """
    if shared_interfaces.empty:
        return pd.DataFrame(
            columns=[
                "shared_asset_id",
                "shared_asset_name",
                "available",
                "electrical_port_available",
                "thermal_port_available",
                "electrical_injection_mw",
                "thermal_injection_mw",
            ]
        )

    overrides = dict(availability_overrides or {})
    rows: list[dict[str, Any]] = []
    for item in shared_interfaces.itertuples(index=False):
        asset_id = str(item.shared_asset_id)
        available = bool(overrides.get(asset_id, bool(item.shared_asset_available)))
        p_setpoint = getattr(item, "electrical_setpoint_mw", np.nan)
        q_setpoint = getattr(item, "thermal_setpoint_mw", np.nan)
        p_value = np.nan if _is_missing(p_setpoint) else float(p_setpoint)
        q_value = np.nan if _is_missing(q_setpoint) else float(q_setpoint)
        if not available:
            if not np.isnan(p_value):
                p_value = 0.0
            if not np.isnan(q_value):
                q_value = 0.0
        rows.append(
            {
                "shared_asset_id": asset_id,
                "shared_asset_name": str(item.shared_asset_name),
                "shared_asset_type": str(item.shared_asset_type),
                "dispatch_mode": str(item.shared_dispatch_mode),
                "available": available,
                "electrical_port_available": available,
                "thermal_port_available": available,
                "electrical_connection_id": str(item.from_id),
                "thermal_connection_id": str(item.thermal_source_id),
                "electrical_injection_mw": p_value,
                "thermal_injection_mw": q_value,
            }
        )
    return pd.DataFrame(rows)


def heat_source_availability(states: pd.DataFrame) -> dict[str, bool]:
    """Return source-ID -> availability for a heat-sector adapter."""
    if states.empty:
        return {}
    return {
        str(row.thermal_connection_id): bool(row.thermal_port_available)
        for row in states.itertuples(index=False)
    }


def apply_heat_source_availability_to_network(
    net: Any,
    states: pd.DataFrame,
    *,
    strict: bool = True,
) -> list[str]:
    """Apply shared-asset availability to named pandapipes heat sources.

    The heat generator itself remains the sector boundary condition; this
    adapter only gates the named source.  It does not invent a CHP dispatch or
    alter customer heat demand.
    """
    if states.empty:
        return []
    if not hasattr(net, "ext_grid"):
        raise ValueError("Heat network has no ext_grid table")
    table = net.ext_grid
    if "name" not in table.columns:
        if strict:
            raise ValueError(
                "Heat-source table has no names; regenerate the network with named source boundaries"
            )
        return []
    if "in_service" not in table.columns:
        table["in_service"] = True

    applied: list[str] = []
    for row in states.itertuples(index=False):
        source_id = str(row.thermal_connection_id)
        mask = table.name.astype(str).eq(source_id)
        if not mask.any():
            if strict:
                raise ValueError(
                    f"Shared asset {row.shared_asset_id} refers to unknown heat source {source_id}"
                )
            continue
        table.loc[mask, "in_service"] = bool(row.thermal_port_available)
        applied.append(source_id)
    return applied


def solve_heat_network_for_shared_asset_state(
    output: Any,
    states: pd.DataFrame,
) -> dict[str, Any]:
    """Re-solve the exported heat network after shared-source availability changes.

    The built-in adapter supports availability changes only.  A prescribed CHP
    thermal dispatch requires a dedicated conversion/dispatch model because the
    current pandapipes network represents heat sources as hydraulic/temperature
    boundaries rather than fixed thermal injections.
    """
    from pathlib import Path

    output = Path(output)
    if states.empty:
        return {
            "required": False,
            "executed": False,
            "converged": True,
            "sources_changed": [],
        }
    if states.thermal_injection_mw.notna().any():
        raise ValueError(
            "The built-in shared-asset heat adapter does not enforce prescribed thermal dispatch; "
            "supply a detailed CHP/heat-source model for that study."
        )

    model_candidates = [
        output / "district_heating_pandapipes.json",
        output / "district_heating_municipal_pandapipes.json",
    ]
    model_path = next((path for path in model_candidates if path.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            "No exported pandapipes heat model found for shared-asset availability re-solve"
        )

    import pandapipes as pp

    net = pp.from_json(str(model_path))
    applied = apply_heat_source_availability_to_network(net, states, strict=True)
    pp.pipeflow(net, mode="hydraulics", max_iter_hyd=100)
    converged = bool(getattr(net, "converged", False))
    export_path = output / "district_heating_pandapipes_shared_asset_state.json"
    pp.to_json(net, str(export_path))

    result: dict[str, Any] = {
        "required": True,
        "executed": True,
        "converged": converged,
        "sources_changed": applied,
        "export_file": export_path.name,
    }
    if hasattr(net, "res_junction") and len(net.res_junction):
        if "p_bar" in net.res_junction.columns:
            result["minimum_pressure_bar"] = float(net.res_junction.p_bar.min())
    if hasattr(net, "res_pipe") and len(net.res_pipe):
        if "v_mean_m_per_s" in net.res_pipe.columns:
            result["maximum_velocity_m_s"] = float(
                net.res_pipe.v_mean_m_per_s.abs().max()
            )
    return result
