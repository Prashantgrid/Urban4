#!/usr/bin/env python3
"""Electrical-network-originating contingency scenarios for Urban4.

This module addresses a limitation of the archived interface-only voltage test:
the disturbance should originate in the electrical network rather than from a
multiplicative voltage factor at the pump terminal.

Two levels are provided deliberately:

1. ``prescreen_single_line_outages`` needs only the archived pandapower JSON.
   It identifies in-service line outages that remove source reachability from a
   selected interface bus.  This is a topological scenario-selection audit,
   not a power-flow or hydraulic result.
2. ``run_network_originating_water_event`` requires pandapower and WNTR.  It
   executes either a line outage or an external-grid voltage-setpoint change,
   obtains the resulting pandapower bus voltage/energization state, propagates
   that state to the existing Urban4 pump-drive/WNTR model, and iterates the
   returned pump duty through pandapower.  No local service-voltage multiplier
   is used.

The v2.7 verification set includes both the pre-screen and an executed PL044
outage.  PL044 is selected as the largest of the eight source-cut candidates
identified for PB046.  It is therefore a deliberately severe but bounded
upstream-branch event, not a probability-weighted reliability estimate.
"""
from __future__ import annotations

import json
import math
from io import StringIO
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .cosimulation import PROJECT, OUT, _config, _motor_speed, _run_power
from .coupling_scenario import _run_water_with_leak


def _json_dataframe(obj: dict[str, Any], key: str) -> pd.DataFrame:
    item = obj["_object"][key]
    if not isinstance(item, dict) or item.get("_class") != "DataFrame":
        raise KeyError(f"{key!r} is not a serialized pandas DataFrame")
    return pd.read_json(StringIO(item["_object"]), orient="split")


def _archived_power_graph(power_json: Path) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    raw = json.loads(Path(power_json).read_text(encoding="utf-8"))
    bus = _json_dataframe(raw, "bus")
    line = _json_dataframe(raw, "line")
    trafo = _json_dataframe(raw, "trafo")
    ext_grid = _json_dataframe(raw, "ext_grid")
    switch = _json_dataframe(raw, "switch")

    graph = nx.Graph()
    for index, row in bus.iterrows():
        if bool(row.get("in_service", True)):
            graph.add_node(int(index))

    open_lines: set[int] = set()
    if not switch.empty and {"et", "closed", "element"}.issubset(switch.columns):
        mask = (switch["et"].astype(str) == "l") & (~switch["closed"].astype(bool))
        open_lines = set(switch.loc[mask, "element"].astype(int).tolist())

    for index, row in line.iterrows():
        if bool(row.get("in_service", True)) and int(index) not in open_lines:
            graph.add_edge(
                int(row["from_bus"]), int(row["to_bus"]),
                element="line", element_index=int(index), name=str(row["name"]),
            )
    for index, row in trafo.iterrows():
        if bool(row.get("in_service", True)):
            graph.add_edge(
                int(row["hv_bus"]), int(row["lv_bus"]),
                element="trafo", element_index=int(index), name=str(row["name"]),
            )

    if "in_service" in ext_grid.columns:
        src_mask = ext_grid["in_service"].fillna(True).astype(bool)
    else:
        src_mask = pd.Series(True, index=ext_grid.index)
    sources = ext_grid.loc[src_mask, "bus"].astype(int)
    return graph, bus, pd.DataFrame({"bus": sources.tolist()})


def prescreen_single_line_outages(
    power_json: Path,
    *,
    target_bus_name: str = "PB046",
) -> pd.DataFrame:
    """Return in-service single-line outages that remove target source reachability.

    The result is a *topological pre-screen*.  It says which line removals
    disconnect the interface bus from every archived external-grid node.  It
    does not claim post-contingency voltage, protection action, or water loss.
    """
    graph, bus, sources_df = _archived_power_graph(Path(power_json))
    hits = bus.index[bus["name"].astype(str) == target_bus_name].tolist()
    if len(hits) != 1:
        raise ValueError(f"Expected one bus named {target_bus_name}, found {len(hits)}")
    target = int(hits[0])
    sources = [int(v) for v in sources_df["bus"].tolist()]
    if not any(nx.has_path(graph, s, target) for s in sources if s in graph):
        raise RuntimeError(f"Target {target_bus_name} is not source-reachable in the base graph")

    rows: list[dict[str, Any]] = []
    for u, v, data in list(graph.edges(data=True)):
        if data.get("element") != "line":
            continue
        attrs = dict(data)
        graph.remove_edge(u, v)
        reachable_sources = [
            s for s in sources if s in graph and target in graph and nx.has_path(graph, s, target)
        ]
        disconnected = len(reachable_sources) == 0
        energized_nodes: set[int] = set()
        for source in sources:
            if source in graph:
                energized_nodes.update(nx.node_connected_component(graph, source))
        deenergized_nodes = set(graph.nodes).difference(energized_nodes)
        graph.add_edge(u, v, **attrs)
        if disconnected:
            rows.append(
                {
                    "line_id": str(attrs["name"]),
                    "line_index": int(attrs["element_index"]),
                    "from_bus": str(bus.loc[int(u), "name"]),
                    "to_bus": str(bus.loc[int(v), "name"]),
                    "target_bus": target_bus_name,
                    "deenergized_bus_count": int(len(deenergized_nodes)),
                    "event_class": "single_line_outage_isolates_target",
                    "hydraulic_consequence_status": "not_executed_by_prescreen",
                }
            )
    return pd.DataFrame(rows).sort_values(["line_index", "line_id"]).reset_index(drop=True)


def _line_outage_connectivity(
    power_json: Path,
    *,
    outaged_line: str,
    target_bus_name: str,
) -> dict[str, Any]:
    """Return the exact source-reachability consequence of one named line cut."""
    graph, bus, sources_df = _archived_power_graph(Path(power_json))
    matches = [
        (u, v, data) for u, v, data in graph.edges(data=True)
        if data.get("element") == "line" and str(data.get("name")) == outaged_line
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one in-service line named {outaged_line}, found {len(matches)}")
    target_hits = bus.index[bus["name"].astype(str) == target_bus_name].tolist()
    if len(target_hits) != 1:
        raise ValueError(f"Expected one bus named {target_bus_name}, found {len(target_hits)}")
    target = int(target_hits[0])
    u, v, data = matches[0]
    graph.remove_edge(u, v)
    sources = [int(value) for value in sources_df["bus"].tolist()]
    energized_nodes: set[int] = set()
    for source in sources:
        if source in graph:
            energized_nodes.update(nx.node_connected_component(graph, source))
    deenergized_nodes = set(graph.nodes).difference(energized_nodes)
    return {
        "line_id": str(data["name"]),
        "line_index": int(data["element_index"]),
        "from_bus": str(bus.loc[int(u), "name"]),
        "to_bus": str(bus.loc[int(v), "name"]),
        "target_bus": target_bus_name,
        "target_source_reachable": bool(target in energized_nodes),
        "deenergized_bus_count": int(len(deenergized_nodes)),
        "deenergized_buses": sorted(str(bus.loc[node, "name"]) for node in deenergized_nodes),
    }


def export_default_prescreen() -> Path:
    """Write the archived PB046 contingency pre-screen used for scenario selection."""
    source = OUT / "final_accepted" / "power_pandapower_bidirectional.json"
    if not source.exists():
        source = OUT / "power_pandapower.json"
    out_dir = OUT / "network_contingency_prescreen_v2.7.0"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = prescreen_single_line_outages(source, target_bus_name="PB046")
    path = out_dir / "single_line_outages_isolating_PB046.csv"
    result.to_csv(path, index=False)
    manifest = {
        "status": "executed_topology_prescreen_only",
        "purpose": "rank electrical-network-originating source-cut candidates and select a declared stress case",
        "source_power_json": str(source.relative_to(PROJECT)),
        "target_bus": "PB046",
        "candidate_count": int(len(result)),
        "important_boundary": (
            "This table is a topology-only selection audit. Hydraulic consequences are reported separately "
            "only for the executed PL044 native pandapower-WNTR case."
        ),
        "csv": str(path.relative_to(PROJECT)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def run_network_originating_water_event(
    *,
    outaged_line: str | None = None,
    ext_grid_vm_pu: float | None = None,
    affected_interface: str = "WAT-WW-01",
    speed_command_pu: float = 0.99,
    maximum_iterations: int = 12,
) -> dict[str, Any]:
    """Execute a true electrical-network-originating waterworks stress test.

    Exactly one of ``outaged_line`` or ``ext_grid_vm_pu`` must be supplied.
    The pandapower network event is solved first.  The resulting PB046 voltage
    (or loss of source reachability handled by pandapower connectivity) is the
    pump-drive input.  The changed WNTR pump duty is returned to pandapower
    until the interface closes.  Unlike the archived service-factor test,
    there is no imposed multiplier between the solved bus and drive terminal.

    This function requires the optional native dependencies and therefore is
    not called by the dependency-light unit test suite.
    """
    if (outaged_line is None) == (ext_grid_vm_pu is None):
        raise ValueError("Specify exactly one of outaged_line or ext_grid_vm_pu")

    # Imports are deliberately lazy so the release remains inspectable
    # in environments without the native simulation packages.
    import pandapower  # noqa: F401  # pragma: no cover
    import wntr  # noqa: F401  # pragma: no cover

    config = _config()
    coupling = config["bidirectional_coupling"]
    drive = coupling["drive_model"]
    relaxation = float(coupling["relaxation_factor"])
    state_path = OUT / "final_accepted" / "bidirectional_interface_states.csv"
    manifest_path = OUT / "final_accepted" / "bidirectional_acceptance_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("accepted", False):
        raise RuntimeError("Network-contingency scenario requires an accepted base state")
    state = pd.read_csv(state_path)
    mask = state["asset_id"].astype(str) == affected_interface
    if int(mask.sum()) != 1:
        raise ValueError(f"Expected one interface {affected_interface}")
    idx = int(state.index[mask][0])
    bus_id = str(state.loc[idx, "bus_id"])
    base_bus_voltage = float(state.loc[idx, "voltage_pu"])
    bus_voltage = base_bus_voltage

    scratch = OUT / "network_originating_contingency_v2.7.0"
    scratch.mkdir(parents=True, exist_ok=True)
    power_json = OUT / "final_accepted" / "power_pandapower_bidirectional.json"
    if not power_json.exists():
        power_json = OUT / "power_pandapower.json"
    connectivity = (
        _line_outage_connectivity(
            power_json, outaged_line=outaged_line, target_bus_name=bus_id,
        )
        if outaged_line else {
            "target_bus": bus_id,
            "target_source_reachable": True,
            "deenergized_bus_count": 0,
            "deenergized_buses": [],
        }
    )
    if outaged_line:
        import pandapower as pp

        base_net = pp.from_json(str(power_json))
        deenergized_names = set(connectivity["deenergized_buses"])
        deenergized_indices = set(
            base_net.bus.index[
                base_net.bus["name"].astype(str).isin(deenergized_names)
            ].astype(int).tolist()
        )
        deenergized_load_mask = base_net.load["bus"].astype(int).isin(deenergized_indices)
        interface_load_mask = base_net.load["name"].astype(str).str.startswith("BIDIR_IF_")
        static_load_mask = ~interface_load_mask
        deenergized_static_mask = deenergized_load_mask & static_load_mask
        deenergized_trafo_mask = base_net.trafo["lv_bus"].astype(int).isin(deenergized_indices)
        affected_interfaces = state[state["bus_id"].astype(str).isin(deenergized_names)]
        total_static_load_mw = float(base_net.load.loc[static_load_mask, "p_mw"].sum())
        unserved_static_load_mw = float(base_net.load.loc[deenergized_static_mask, "p_mw"].sum())
        connectivity.update(
            {
                "base_bus_count": int(len(base_net.bus)),
                "deenergized_bus_fraction": float(len(deenergized_indices) / len(base_net.bus)),
                "base_static_load_mw": total_static_load_mw,
                "unserved_static_load_mw": unserved_static_load_mw,
                "unserved_static_load_fraction": float(unserved_static_load_mw / total_static_load_mw),
                "deenergized_static_load_object_count": int(deenergized_static_mask.sum()),
                "base_transformer_count": int(len(base_net.trafo)),
                "deenergized_transformer_count": int(deenergized_trafo_mask.sum()),
                "base_powered_interface_count": int(len(state)),
                "deenergized_powered_interface_count": int(len(affected_interfaces)),
                "deenergized_interface_power_mw": float(affected_interfaces["power_mw"].sum()),
                "deenergized_interface_assets": affected_interfaces["asset_id"].astype(str).tolist(),
            }
        )
    base_power_mw = float(state.loc[idx, "power_mw"])
    base_water = manifest["final_sector_states"]["drinking_water"]
    base_power = manifest["final_sector_states"]["electricity"]
    rows: list[dict[str, Any]] = []
    converged = False
    for iteration in range(1, maximum_iterations + 1):  # pragma: no cover
        previous_power = float(state.loc[idx, "power_mw"])
        power, voltages = _run_power(
            state,
            scratch,
            config,
            write_export=False,
            outaged_lines={outaged_line} if outaged_line else None,
            ext_grid_vm_pu=ext_grid_vm_pu,
        )
        bus_voltage = float(voltages.get(bus_id, math.nan))
        pump_speed = _motor_speed(bus_voltage, speed_command_pu, drive)
        water = _run_water_with_leak(
            pump_speed,
            0.0,
            float(config["hydraulic_leak_demonstration"]["discharge_coefficient"]),
            config,
        )
        calculated_power = float(water["pump_electrical_power_mw"])
        state.loc[idx, "calculated_power_mw"] = calculated_power
        state.loc[idx, "power_mw"] = (1.0 - relaxation) * previous_power + relaxation * calculated_power
        power_error = abs(float(state.loc[idx, "power_mw"]) - previous_power)
        rows.append(
            {
                "iteration": iteration,
                "outaged_line": outaged_line or "",
                "ext_grid_vm_pu": ext_grid_vm_pu if ext_grid_vm_pu is not None else math.nan,
                "pump_bus_id": bus_id,
                "pump_bus_voltage_pu": bus_voltage,
                "pump_energized": bool(math.isfinite(bus_voltage) and bus_voltage >= float(drive["trip_voltage_pu"])),
                "target_source_reachable": bool(connectivity["target_source_reachable"]),
                "deenergized_bus_count": int(connectivity["deenergized_bus_count"]),
                "pump_speed_pu": pump_speed,
                "pump_power_mw": float(state.loc[idx, "power_mw"]),
                "calculated_pump_power_mw": calculated_power,
                "critical_pressure_m": float(water["minimum_pressure_m"]),
                "delivered_water_fraction": float(water["delivered_water_fraction"]),
                "minimum_energized_bus_voltage_pu": float(power["minimum_voltage_pu"]),
                "maximum_line_loading_percent": float(power["maximum_line_loading_percent"]),
                "power_residual_mw": power_error,
            }
        )
        if power_error <= float(coupling["interface_power_absolute_tolerance_mw"]):
            converged = True
            break

    df = pd.DataFrame(rows)
    result_path = scratch / "network_originating_water_event_iterations.csv"
    df.to_csv(result_path, index=False)
    final = df.iloc[-1]
    summary = pd.DataFrame([
        {
            "state": "accepted_base",
            "outaged_line": "",
            "line_from_bus": "",
            "line_to_bus": "",
            "target_bus": bus_id,
            "target_source_reachable": True,
            "deenergized_bus_count": 0,
            "pump_bus_voltage_pu": base_bus_voltage,
            "pump_energized": True,
            "interface_pump_power_kw": 1000.0 * base_power_mw,
            "native_calculated_pump_power_kw": 1000.0 * float(base_water["pump_electrical_power_mw"]),
            "minimum_pressure_m": float(base_water["minimum_pressure_m"]),
            "delivered_water_percent": 100.0 * float(base_water["delivered_water_fraction"]),
            "minimum_energized_bus_voltage_pu": float(base_power["minimum_voltage_pu"]),
            "maximum_line_loading_percent": float(base_power["maximum_line_loading_percent"]),
        },
        {
            "state": f"{outaged_line}_outage" if outaged_line else "network_event",
            "outaged_line": outaged_line or "",
            "line_from_bus": connectivity.get("from_bus", ""),
            "line_to_bus": connectivity.get("to_bus", ""),
            "target_bus": bus_id,
            "target_source_reachable": bool(final["target_source_reachable"]),
            "deenergized_bus_count": int(final["deenergized_bus_count"]),
            "pump_bus_voltage_pu": float(final["pump_bus_voltage_pu"]),
            "pump_energized": bool(final["pump_energized"]),
            "interface_pump_power_kw": 1000.0 * float(final["pump_power_mw"]),
            "native_calculated_pump_power_kw": 1000.0 * float(final["calculated_pump_power_mw"]),
            "minimum_pressure_m": float(final["critical_pressure_m"]),
            "delivered_water_percent": 100.0 * float(final["delivered_water_fraction"]),
            "minimum_energized_bus_voltage_pu": float(final["minimum_energized_bus_voltage_pu"]),
            "maximum_line_loading_percent": float(final["maximum_line_loading_percent"]),
        },
    ])
    summary_path = scratch / "network_originating_outage_summary.csv"
    summary.to_csv(summary_path, index=False)
    result = {
        "status": "executed_and_closed" if converged else "failed_to_close",
        "native_models_executed": ["pandapower", "EPANET/WNTR"],
        "event": {"outaged_line": outaged_line, "ext_grid_vm_pu": ext_grid_vm_pu},
        "selection": connectivity,
        "important_difference_from_archived_test": (
            "drive terminal voltage is the solved pandapower bus voltage; no service-voltage factor is applied"
        ),
        "iterations": int(len(df)),
        "thresholds_relaxed": False,
        "physical_native_pump_power_kw": 1000.0 * float(final["calculated_pump_power_mw"]),
        "closed_interface_pump_power_kw": 1000.0 * float(final["pump_power_mw"]),
        "minimum_pressure_m": float(final["critical_pressure_m"]),
        "delivered_water_percent": 100.0 * float(final["delivered_water_fraction"]),
        "minimum_energized_bus_voltage_pu": float(final["minimum_energized_bus_voltage_pu"]),
        "maximum_line_loading_percent": float(final["maximum_line_loading_percent"]),
        "important_interpretation": (
            "pandapower marks the source-disconnected PB046 result as not energized (NaN voltage) and solves "
            "the remaining energized component; no zero voltage is fabricated for the isolated bus"
        ),
        "coupling_scope": (
            "The native EPANET/WNTR consequence is propagated for WAT-WW-01. Other powered interfaces "
            "on the disconnected branch are reported as electrically de-energized contracts; their sector-native "
            "consequences are not inferred by this scenario."
        ),
        "iteration_csv": str(result_path.relative_to(PROJECT)),
        "summary_csv": str(summary_path.relative_to(PROJECT)),
    }
    (scratch / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Urban4 network-originating electrical contingency runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--line", help="Pandapower line name to outage, e.g. PL044")
    group.add_argument("--ext-grid-vm-pu", type=float, help="External-grid voltage set point in p.u.")
    parser.add_argument("--prescreen", action="store_true", help="Only export the PB046 topological outage prescreen")
    args = parser.parse_args()
    if args.prescreen or (args.line is None and args.ext_grid_vm_pu is None):
        print(export_default_prescreen())
    else:
        result = run_network_originating_water_event(outaged_line=args.line, ext_grid_vm_pu=args.ext_grid_vm_pu)
        print(json.dumps(result, indent=2))
