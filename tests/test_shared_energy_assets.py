from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from urban4.shared_energy_assets import (
    RELATION,
    build_shared_asset_interfaces,
    heat_source_availability,
    resolve_shared_asset_states,
)

ROOT = Path(__file__).resolve().parents[1]
CASE = json.loads((ROOT / "cases" / "schweinfurt.json").read_text(encoding="utf-8"))
SERVICE = ROOT / "outputs" / "integrated_service_resolved"
MUNICIPAL = ROOT / "outputs" / "municipal_scale_v2.5.0"


def _service_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(SERVICE / "electricity_nodes.csv"),
        pd.read_csv(SERVICE / "district_heating_nodes.csv"),
    )


def test_gks_shared_asset_is_recorded_without_inventing_dispatch() -> None:
    power, heat = _service_inputs()
    rows = build_shared_asset_interfaces(CASE, power, heat)
    assert len(rows) == 1
    row = rows[0]
    assert row["relation"] == RELATION
    assert row["shared_asset_id"] == "GKS_CHP"
    assert row["shared_asset_name"].startswith("GKS main CHP")
    assert row["shared_dispatch_mode"] == "availability_only"
    assert np.isnan(float(row["electrical_setpoint_mw"]))
    assert np.isnan(float(row["thermal_setpoint_mw"]))
    assert float(row["electrical_nameplate_mw"]) == 29.0
    assert "not a claimed real utility connection" in row["electrical_connection_basis"]
    assert "no inferred CHP heat-to-power curve" in row["forward_duty_model"]


def test_gks_electrical_port_uses_generated_mv_attachment() -> None:
    power, heat = _service_inputs()
    row = build_shared_asset_interfaces(CASE, power, heat)[0]
    match = power.loc[power.bus_id.astype(str).eq(str(row["from_id"]))].iloc[0]
    assert float(match.voltage_kv) >= 10.0
    assert str(match.node_type) in {
        "mv_source", "mv_junction", "transformer_mv_bus", "mv_customer"
    }
    assert float(row["connection_distance_km"]) >= 0.0


def test_municipality_gks_thermal_port_resolves_documented_h1() -> None:
    power = pd.read_csv(SERVICE / "electricity_nodes.csv")
    heat_sources = pd.read_csv(MUNICIPAL / "district_heating_sources_municipal.csv")
    row = build_shared_asset_interfaces(CASE, power, heat_sources)[0]
    assert row["thermal_source_id"] == "H1"
    assert row["to_id"] == "H1"
    assert row["thermal_connection_basis"] == "configured heat-source identifier"
    assert float(row["thermal_connection_distance_km"]) < 1e-9


def test_common_availability_gates_both_ports_for_prescribed_future_case() -> None:
    cfg = copy.deepcopy(CASE)
    asset = cfg["shared_energy_assets"][0]
    asset["dispatch_mode"] = "prescribed_operating_point"
    asset["electrical_port"]["prescribed_active_power_mw"] = 5.0
    asset["thermal_port"]["prescribed_heat_output_mw"] = 12.0
    power, heat = _service_inputs()
    frame = pd.DataFrame(build_shared_asset_interfaces(cfg, power, heat))

    available = resolve_shared_asset_states(frame)
    assert bool(available.loc[0, "available"])
    assert float(available.loc[0, "electrical_injection_mw"]) == 5.0
    assert float(available.loc[0, "thermal_injection_mw"]) == 12.0

    unavailable = resolve_shared_asset_states(frame, {"GKS_CHP": False})
    assert not bool(unavailable.loc[0, "available"])
    assert not bool(unavailable.loc[0, "electrical_port_available"])
    assert not bool(unavailable.loc[0, "thermal_port_available"])
    assert float(unavailable.loc[0, "electrical_injection_mw"]) == 0.0
    assert float(unavailable.loc[0, "thermal_injection_mw"]) == 0.0
    assert heat_source_availability(unavailable) == {
        str(unavailable.loc[0, "thermal_connection_id"]): False
    }


def test_availability_only_shared_asset_does_not_change_benchmark_dispatch() -> None:
    power, heat = _service_inputs()
    frame = pd.DataFrame(build_shared_asset_interfaces(CASE, power, heat))
    states = resolve_shared_asset_states(frame)
    assert states.electrical_injection_mw.isna().all()
    assert states.thermal_injection_mw.isna().all()


def test_nominal_coupling_reads_shared_asset_contract_without_inventing_generation(
    tmp_path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace
    import urban4.framework as framework
    import urban4.service_coupling as coupling

    class FakeNet:
        def __init__(self) -> None:
            self.bus = pd.DataFrame(
                {"name": ["BUS_LOAD", "BUS_GKS"]}, index=[0, 1]
            )
            self.load = pd.DataFrame({"name": pd.Series(dtype=str)})
            self.res_bus = pd.DataFrame({"vm_pu": [1.0, 1.0]}, index=[0, 1])
            self.res_line = pd.DataFrame({"loading_percent": [10.0]})
            self.res_trafo = pd.DataFrame({"loading_percent": [10.0]})

    net = FakeNet()
    recorded_sgens: list[float] = []

    def create_sgens(_net, *, p_mw, **kwargs):
        recorded_sgens.extend(float(x) for x in np.asarray(p_mw))
        return np.arange(len(np.asarray(p_mw)))

    fake_pp = SimpleNamespace(
        from_json=lambda _path: net,
        create_loads=lambda *args, **kwargs: np.array([0]),
        create_sgens=create_sgens,
        runpp=lambda *args, **kwargs: None,
        to_json=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "pandapower", fake_pp)
    monkeypatch.setattr(framework, "load_case", lambda: CASE)
    monkeypatch.setattr(coupling, "_drive_speed", lambda voltage, drive=None: np.ones(len(voltage)))

    interfaces = pd.DataFrame([
        {
            "interface_id": "I_LOAD",
            "from_sector": "electricity",
            "from_id": "BUS_LOAD",
            "to_sector": "drinking_water",
            "to_id": "PUMP",
            "relation": "electrically_driven_facility",
            "nominal_power_mw": 0.1,
        },
        {
            "interface_id": "I_GKS",
            "from_sector": "electricity",
            "from_id": "BUS_GKS",
            "to_sector": "district_heating",
            "to_id": "H1",
            "relation": RELATION,
            "shared_asset_id": "GKS_CHP",
            "shared_asset_name": "GKS",
            "shared_asset_type": "combined_heat_and_power_source",
            "shared_asset_available": True,
            "shared_dispatch_mode": "availability_only",
            "electrical_setpoint_mw": np.nan,
            "thermal_setpoint_mw": np.nan,
            "thermal_source_id": "H1",
        },
    ])
    interfaces.to_csv(tmp_path / "coupling_interfaces.csv", index=False)
    (tmp_path / "electricity_pandapower.json").write_text("{}")

    result = coupling.run_nominal_interface_closure(tmp_path)
    assert result["accepted"]
    assert result["shared_conversion_assets"] == 1
    assert result["shared_asset_electrical_injection_mw"] == 0.0
    assert recorded_sgens == []
    states = pd.read_csv(tmp_path / "shared_energy_asset_states.csv")
    assert bool(states.loc[0, "available"])
    assert bool(states.loc[0, "thermal_port_available"])
    assert np.isnan(states.loc[0, "electrical_injection_mw"])


def test_nominal_coupling_can_apply_a_prescribed_shared_asset_operating_point(
    tmp_path, monkeypatch
) -> None:
    import sys
    from types import SimpleNamespace
    import urban4.framework as framework
    import urban4.service_coupling as coupling

    class FakeNet:
        def __init__(self) -> None:
            self.bus = pd.DataFrame(
                {"name": ["BUS_LOAD", "BUS_GKS"]}, index=[0, 1]
            )
            self.load = pd.DataFrame({"name": pd.Series(dtype=str)})
            self.res_bus = pd.DataFrame({"vm_pu": [1.0, 1.0]}, index=[0, 1])
            self.res_line = pd.DataFrame({"loading_percent": [10.0]})
            self.res_trafo = pd.DataFrame({"loading_percent": [10.0]})

    net = FakeNet()
    recorded_sgens: list[float] = []

    def create_sgens(_net, *, p_mw, **kwargs):
        recorded_sgens.extend(float(x) for x in np.asarray(p_mw))
        return np.arange(len(np.asarray(p_mw)))

    fake_pp = SimpleNamespace(
        from_json=lambda _path: net,
        create_loads=lambda *args, **kwargs: np.array([0]),
        create_sgens=create_sgens,
        runpp=lambda *args, **kwargs: None,
        to_json=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "pandapower", fake_pp)
    monkeypatch.setattr(framework, "load_case", lambda: CASE)
    monkeypatch.setattr(coupling, "_drive_speed", lambda voltage, drive=None: np.ones(len(voltage)))

    interfaces = pd.DataFrame([
        {
            "interface_id": "I_LOAD",
            "from_sector": "electricity",
            "from_id": "BUS_LOAD",
            "to_sector": "drinking_water",
            "to_id": "PUMP",
            "relation": "electrically_driven_facility",
            "nominal_power_mw": 0.1,
        },
        {
            "interface_id": "I_GKS",
            "from_sector": "electricity",
            "from_id": "BUS_GKS",
            "to_sector": "district_heating",
            "to_id": "H1",
            "relation": RELATION,
            "shared_asset_id": "GKS_CHP",
            "shared_asset_name": "GKS",
            "shared_asset_type": "combined_heat_and_power_source",
            "shared_asset_available": True,
            "shared_dispatch_mode": "prescribed_operating_point",
            "electrical_setpoint_mw": 5.0,
            "thermal_setpoint_mw": 12.0,
            "thermal_source_id": "H1",
        },
    ])
    interfaces.to_csv(tmp_path / "coupling_interfaces.csv", index=False)
    (tmp_path / "electricity_pandapower.json").write_text("{}")

    result = coupling.run_nominal_interface_closure(tmp_path)
    assert result["accepted"]
    assert result["shared_asset_electrical_injection_mw"] == 5.0
    assert recorded_sgens == [5.0]

    recorded_sgens.clear()
    result = coupling.run_nominal_interface_closure(
        tmp_path, shared_asset_availability={"GKS_CHP": False}
    )
    assert result["accepted"]
    assert result["shared_asset_electrical_injection_mw"] == 0.0
    assert recorded_sgens == []
    states = pd.read_csv(tmp_path / "shared_energy_asset_states.csv")
    assert not bool(states.loc[0, "available"])
    assert not bool(states.loc[0, "thermal_port_available"])


def test_heat_side_adapter_gates_named_source_with_same_common_availability() -> None:
    from types import SimpleNamespace
    from urban4.shared_energy_assets import apply_heat_source_availability_to_network

    net = SimpleNamespace(
        ext_grid=pd.DataFrame(
            {"name": ["H1", "H2"], "in_service": [True, True]}
        )
    )
    states = pd.DataFrame([
        {
            "shared_asset_id": "GKS_CHP",
            "thermal_connection_id": "H1",
            "thermal_port_available": False,
        }
    ])
    applied = apply_heat_source_availability_to_network(net, states)
    assert applied == ["H1"]
    assert not bool(net.ext_grid.loc[net.ext_grid.name.eq("H1"), "in_service"].iloc[0])
    assert bool(net.ext_grid.loc[net.ext_grid.name.eq("H2"), "in_service"].iloc[0])


def test_municipal_builder_exports_gks_shared_asset_contract() -> None:
    path = MUNICIPAL / "shared_energy_assets_municipal.csv"
    assert path.exists()
    frame = pd.read_csv(path)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.shared_asset_id == "GKS_CHP"
    assert row.thermal_source_id == "H1"
    assert float(row.connected_bus_voltage_kv) >= 10.0
    assert row.shared_dispatch_mode == "availability_only"
    assert pd.isna(row.electrical_setpoint_mw)
    assert pd.isna(row.thermal_setpoint_mw)
