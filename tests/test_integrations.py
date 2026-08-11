from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from urban4.framework import load_case
from urban4.infdb_adapter import build_infdb_building_ledger, validate_infdb_export
from urban4.pylovo_bridge import merge_pylovo_grids
from urban4.integration_contract import (
    _nearest_compatible_bus,
    write_shared_corridor_ledger,
)
from urban4.native_acceptance import _swmm_metric
from urban4.service_coupling import _drive_speed
from urban4.coupling_repair import (
    declared_policy,
    next_catalogue_value,
    phase_return_for_failure,
    terminal_failure,
    validate_interface_state,
)
from urban4.cosimulation import _select_and_apply_limit_repair


class InfDBContractTest(unittest.TestCase):
    def test_contract_keys_and_aggregate_reconciliation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            buildings = pd.DataFrame([
                {
                    "objectid": "B1", "building_use_id": "1000", "building_use": "residential",
                    "building_type": "detached", "floor_area": 180.0, "floor_number": 2,
                    "occupants": 4, "households": 2,
                    "geometry_json": json.dumps({"type": "Polygon", "coordinates": [[[10.0,50.0],[10.001,50.0],[10.001,50.001],[10.0,50.001],[10.0,50.0]]]}),
                },
                {
                    "objectid": "B2", "building_use_id": "2110", "building_use": "industrial",
                    "building_type": "factory", "floor_area": 800.0, "floor_number": 1,
                    "occupants": 0, "households": 0,
                    "geometry_json": json.dumps({"type": "Polygon", "coordinates": [[[10.002,50.0],[10.003,50.0],[10.003,50.001],[10.002,50.001],[10.002,50.0]]]}),
                },
            ])
            ways = pd.DataFrame([{"way_id": "W1", "geometry_json": json.dumps({"type": "LineString", "coordinates": [[10.0,49.999],[10.004,49.999]]})}])
            connections = pd.DataFrame([
                {"objectid": "B1", "way_id": "W1", "geometry_json": json.dumps({"type": "LineString", "coordinates": [[10.0005,50.0],[10.0005,49.999]]})},
                {"objectid": "B2", "way_id": "W1", "geometry_json": json.dumps({"type": "LineString", "coordinates": [[10.0025,50.0],[10.0025,49.999]]})},
            ])
            buildings.to_csv(root / "infdb_buildings.csv", index=False)
            ways.to_csv(root / "infdb_ways.csv", index=False)
            connections.to_csv(root / "infdb_connection_lines.csv", index=False)
            audit = validate_infdb_export(root)
            self.assertEqual(audit["building_count"], 2)
            ledger = build_infdb_building_ledger(root, load_case())
            anchors = load_case()["official_anchors"]
            self.assertAlmostEqual(ledger.population.sum(), anchors["served_population"])
            self.assertAlmostEqual(ledger.electricity_peak_kw.sum() / 1000.0, anchors["electricity_lv_peak_mw"])
            self.assertAlmostEqual(ledger.water_m3_year.sum(), anchors["drinking_water_annual_m3"])
            self.assertEqual(set(ledger.building_id), {"INFDB_B1", "INFDB_B2"})


class PylovoBridgeTest(unittest.TestCase):
    def test_independent_slack_is_replaced_by_connected_mv_source(self):
        try:
            import pandapower as pp
        except ModuleNotFoundError:
            self.skipTest("pandapower is an optional native-solver dependency")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            net = pp.create_empty_network()
            hv = pp.create_bus(net, vn_kv=20.0, name="MVbus 1", geodata=(10.0, 50.0))
            lv = pp.create_bus(net, vn_kv=0.4, name="LVbus 1", geodata=(10.0, 49.9999))
            customer = pp.create_bus(net, vn_kv=0.4, name="Consumer Nodebus B1", geodata=(10.001, 49.9999))
            pp.create_ext_grid(net, hv, vm_pu=1.0)
            pp.create_transformer_from_parameters(
                net, hv, lv, sn_mva=0.4, vn_hv_kv=20.0, vn_lv_kv=0.4,
                vk_percent=6.0, vkr_percent=1.0, pfe_kw=0.8, i0_percent=0.2,
                shift_degree=0.0,
            )
            pp.create_line_from_parameters(net, lv, customer, 0.05, 0.443, 0.08, 0.0, 0.179)
            pp.create_load(net, customer, p_mw=0.0, q_mvar=0.0, max_p_mw=0.03, name="Load B1 household 1")
            path = root / "pylovo_grid_1.json"
            pp.to_json(net, str(path))
            road = nx.Graph()
            road.add_node((10.0, 50.0))
            output = root / "merged"
            merged, _, summary = merge_pylovo_grids([path], road, output, coincident_peak_mw=0.02)
            self.assertTrue(summary["converged"])
            self.assertEqual(summary["removed_pylovo_ext_grids"], 1)
            self.assertEqual(summary["final_ext_grids"], 1)
            self.assertEqual(summary["graph_components"], 1)
            self.assertEqual(len(merged.trafo), 1)
            self.assertEqual(len(merged.load), 1)
            self.assertAlmostEqual(merged.load.p_mw.sum(), 0.02)


class ExecutableIntegrationContractTest(unittest.TestCase):
    def test_medium_drive_connects_at_transformer_lv_switchboard(self):
        buses = pd.DataFrame([
            {"bus_id": "deep_lv", "node_type": "lv_junction", "voltage_kv": 0.4, "lon": 10.0, "lat": 50.0},
            {"bus_id": "trafo_lv", "node_type": "transformer_lv_bus", "voltage_kv": 0.4, "lon": 10.01, "lat": 50.0},
            {"bus_id": "mv", "node_type": "mv_junction", "voltage_kv": 20.0, "lon": 10.02, "lat": 50.0},
        ])
        selected = _nearest_compatible_bus((10.0, 50.0), buses, 0.075)
        self.assertEqual(selected.bus_id, "trafo_lv")
        self.assertEqual(selected.node_type, "transformer_lv_bus")

    def test_drive_envelope_returns_voltage_to_sector_speed(self):
        speed = _drive_speed(np.asarray([1.0, 0.90, 0.825, 0.75, 0.70]))
        np.testing.assert_allclose(speed, [1.0, 0.95, 0.0, 0.0, 0.0])

    def test_shared_corridor_identity_uses_physical_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            geometry = json.dumps([[10.0, 50.0], [10.001, 50.0]])
            pd.DataFrame([{"line_id": "E1", "geometry_json": geometry}]).to_csv(root / "electricity_links.csv", index=False)
            pd.DataFrame([{"link_id": "W1", "geometry_json": geometry}]).to_csv(root / "drinking_water_links.csv", index=False)
            pd.DataFrame([{"link_id": "S1", "geometry_json": json.dumps([[10.1, 50.0], [10.101, 50.0]])}]).to_csv(root / "wastewater_links.csv", index=False)
            pd.DataFrame([{"corridor_id": "H1", "geometry_json": geometry}]).to_csv(root / "district_heating_corridors.csv", index=False)
            ledger = write_shared_corridor_ledger(root)
            shared = ledger[ledger.is_shared_corridor]
            self.assertEqual(len(shared), 1)
            self.assertEqual(shared.iloc[0].sector_count, 3)
            self.assertEqual(shared.iloc[0].sectors, "district_heating|drinking_water|electricity")

    def test_swmm_dot_leader_metrics_are_not_silently_defaulted(self):
        text = "Continuity Error (%) .....        -2.460\n"
        self.assertEqual(_swmm_metric(text, "continuity"), -2.460)


class CoupledRepairPolicyTest(unittest.TestCase):
    def test_declared_priority_and_terminal_outcome_are_fixed(self):
        config = load_case()
        policy = declared_policy(config)
        self.assertEqual(
            policy["priority_order"],
            [
                "interface_contract",
                "numerical_damping",
                "bounded_operating_setpoint",
                "next_catalogue_component",
                "phase_ii_topology_regeneration",
                "failed_no_feasible_repair",
            ],
        )
        self.assertEqual(terminal_failure("exhausted").action, "failed_no_feasible_repair")
        self.assertEqual(phase_return_for_failure("drinking_water").phase, "Phase II-B")

    def test_catalogue_upgrade_is_discrete_and_stops_at_largest_size(self):
        self.assertEqual(next_catalogue_value(100, [80, 100, 125, 150]), 125)
        self.assertIsNone(next_catalogue_value(150, [80, 100, 125, 150]))

    def test_invalid_interface_contract_fails_before_solver(self):
        state = pd.DataFrame([
            {"interface_id": "I1", "bus_id": "", "sector": "water", "asset_id": "P1", "power_mw": -0.1}
        ])
        self.assertEqual(
            validate_interface_state(state),
            ["missing_bus_id", "invalid_interface_power"],
        )

    def test_operating_adjustment_precedes_component_change(self):
        config = load_case()
        policy = declared_policy(config)
        last = {
            "electricity": {"passed": True, "checks": {}},
            "drinking_water": {
                "passed": False,
                "checks": {
                    "maximum_pressure": False, "minimum_pressure": True,
                    "delivered_demand": True, "maximum_velocity": True,
                },
            },
            "wastewater": {
                "passed": True,
                "checks": {"mass_continuity": True, "engine_run": True},
            },
            "district_heating": {
                "passed": True,
                "checks": {"minimum_pressure": True, "maximum_velocity": True},
            },
        }
        commands = {"WAT-WW-01": 1.0, "DH_PUMP_GKS": 1.0}
        action, upstream = _select_and_apply_limit_repair(
            iteration=2, last=last, commands=commands,
            water_pipe_overrides={}, heat_pipe_overrides={},
            line_overrides={}, transformer_overrides={},
            runtime={"wastewater_duration_hours": 6, "heat_lift_factor": 1.0},
            repair_counts={sector: 0 for sector in policy["sector_tie_break_order"]},
            policy=policy, config=config,
        )
        self.assertIsNone(upstream)
        self.assertEqual(action["priority"], 3)
        self.assertEqual(action["action"], "reduce_water_pump_speed_command")
        self.assertAlmostEqual(commands["WAT-WW-01"], 0.99)

    def test_unrepairable_sector_requests_upstream_phase_then_terminal_failure(self):
        config = load_case()
        policy = declared_policy(config)
        last = {
            "electricity": {"passed": True, "checks": {}},
            "drinking_water": {"passed": True, "checks": {}},
            "wastewater": {
                "passed": False,
                "checks": {
                    "mass_continuity": True, "engine_run": True,
                    "no_flooding": False, "nonconverging_steps": True,
                },
            },
            "district_heating": {"passed": True, "checks": {}},
        }
        action, upstream = _select_and_apply_limit_repair(
            iteration=4, last=last,
            commands={"WAT-WW-01": 1.0, "DH_PUMP_GKS": 1.0},
            water_pipe_overrides={}, heat_pipe_overrides={},
            line_overrides={}, transformer_overrides={},
            runtime={"wastewater_duration_hours": 24, "heat_lift_factor": 1.0},
            repair_counts={sector: 0 for sector in policy["sector_tie_break_order"]},
            policy=policy, config=config,
        )
        self.assertIsNone(action)
        self.assertEqual(upstream, "wastewater")
        self.assertEqual(phase_return_for_failure(upstream).action, "return_to_phase_ii_c")
        self.assertTrue(terminal_failure("all actions failed").terminal)


if __name__ == "__main__":
    unittest.main()
