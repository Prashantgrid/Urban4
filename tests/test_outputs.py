from __future__ import annotations

import json
import unittest
from pathlib import Path

import networkx as nx
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs"
CASE = json.loads((PROJECT / "cases" / "schweinfurt.json").read_text(encoding="utf-8"))


class GeneratedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = [
            OUT / "model_manifest_four_sector.json",
            OUT / "benchmark_case_metrics.csv",
            OUT / "building_sector_demands.csv",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise unittest.SkipTest(f"Run python run_all.py first; missing {missing}")
        cls.manifest = json.loads((OUT / "model_manifest_four_sector.json").read_text(encoding="utf-8"))
        cls.benchmarks = json.loads((OUT / "benchmark_cases" / "manifest.json").read_text(encoding="utf-8"))

    def test_common_demand_totals_match_anchors(self) -> None:
        buildings = pd.read_csv(OUT / "building_sector_demands.csv")
        anchors = CASE["official_anchors"]
        self.assertAlmostEqual(buildings["electricity_mwh_year"].sum(), anchors["electricity_lv_annual_mwh"], places=1)
        self.assertAlmostEqual(buildings["water_m3_year"].sum(), anchors["drinking_water_annual_m3"], places=0)
        self.assertAlmostEqual(buildings["wastewater_sanitary_m3_year"].sum(), anchors["drinking_water_annual_m3"], places=0)
        self.assertEqual(int(buildings["household_count"].sum()), 34400)
        wholesale = pd.read_csv(OUT / "water_wholesale_boundary_ledger.csv")
        self.assertAlmostEqual(wholesale["annual_m3"].sum(), anchors["drinking_water_wholesale_delivery_m3"], places=0)
        self.assertFalse(wholesale["physical_network_allocation"].any())

    def test_common_zone_membership_and_four_demands_are_traceable(self) -> None:
        buildings = pd.read_csv(OUT / "building_sector_demands.csv")
        zones = pd.read_csv(OUT / "demand_zones.csv")
        member_ids = [
            building_id
            for value in zones["building_ids"]
            for building_id in str(value).split(";")
            if building_id
        ]
        self.assertEqual(len(member_ids), len(buildings))
        self.assertEqual(len(member_ids), len(set(member_ids)))
        self.assertEqual(set(member_ids), set(buildings["building_id"]))
        for column in [
            "population",
            "electricity_mwh_year",
            "electricity_peak_kw",
            "water_m3_year",
            "wastewater_sanitary_m3_year",
            "heat_candidate_mwh_year",
        ]:
            self.assertAlmostEqual(zones[column].sum(), buildings[column].sum(), places=5)
        self.assertTrue((zones["clustering_seed"] == CASE["seed"]).all())

    def test_case_adapter_boundary_and_metric_frame_are_declared(self) -> None:
        self.assertEqual(CASE["projected_crs"], "LOCAL_EQUIRECTANGULAR_AT_50N")
        self.assertEqual(CASE["service_mask"]["evidence_class"], "C")
        self.assertEqual(CASE["service_mask"]["type"], "union_of_circles")
        self.assertGreaterEqual(len(CASE["service_mask"]["components"]), 1)

    def test_integrated_heat_aggregates_match_declared_calibration(self) -> None:
        anchors = CASE["official_anchors"]
        consumers = pd.read_csv(OUT / "heat_consumers.csv")
        inventory = pd.read_csv(OUT / "network_inventory.csv").set_index("sector")
        self.assertEqual(int(consumers["represented_contract_count"].sum()), anchors["district_heat_sales_contracts"])
        self.assertAlmostEqual(consumers["annual_heat_mwh"].sum(), anchors["district_heat_sales_mwh_year"], places=6)
        relative_error = abs(
            inventory.loc["district_heating", "route_length_km"]
            - anchors["district_heat_route_km"]
        ) / anchors["district_heat_route_km"]
        self.assertLessEqual(
            relative_error,
            CASE["district_heating"]["route_length_tolerance_fraction"],
        )

    def test_three_cases_are_independent_and_solver_ready(self) -> None:
        self.assertEqual(set(self.benchmarks), {"dense_urban", "semi_urban", "rural_peripheral"})
        expected_places = {
            "dense_urban": "Schweinfurt Innenstadt",
            "semi_urban": "Bergl/Gartenstadt",
            "rural_peripheral": "Dittelbrunn",
        }
        self.assertEqual(
            {case_id: case["place_name"] for case_id, case in self.benchmarks.items()},
            expected_places,
        )
        for case in self.benchmarks.values():
            self.assertTrue(case["independent_generation"])
            self.assertIn("topology-generation", case["study_role"])
            solvers = case["solver_readiness"]
            self.assertTrue(solvers["pandapower"]["converged"])
            self.assertTrue(solvers["epanet_wntr"]["converged"])
            self.assertTrue(solvers["swmm"]["engine_run_passed"])
            self.assertTrue(solvers["pandapipes_heat"]["converged"])

    def test_controlled_and_policy_experiments_are_separate(self) -> None:
        metrics = pd.read_csv(OUT / "benchmark_case_metrics.csv")
        policy = pd.read_csv(OUT / "benchmark_policy_case_metrics.csv")
        self.assertEqual(set(metrics["experiment_mode"]), {"controlled_morphology"})
        self.assertEqual(set(policy["experiment_mode"]), {"morphology_aware_policy"})
        for case in self.benchmarks.values():
            self.assertEqual(case["controlled_parameters"]["maximum_lv_service_radius_km"], 0.70)
            self.assertEqual(case["controlled_parameters"]["district_heat_connection_fraction"], 0.25)
            self.assertIn(
                "contiguous cells",
                case["controlled_parameters"]["district_heat_selection"],
            )

    def test_distributional_topology_metrics_are_exported(self) -> None:
        metrics = pd.read_csv(OUT / "benchmark_case_metrics.csv")
        required = {
            "mean_degree",
            "branch_fraction",
            "leaf_fraction",
            "edge_length_p50_km",
            "edge_length_p90_km",
            "route_stretch_p50",
            "route_stretch_p90",
            "network_length_per_1000_buildings_km",
            "buildings_per_source_mean",
        }
        self.assertTrue(required.issubset(metrics.columns))
        self.assertFalse(metrics[list(required)].isna().any().any())
        self.assertTrue((metrics["edge_length_p90_km"] >= metrics["edge_length_p50_km"]).all())
        self.assertTrue((metrics["route_stretch_p90"] >= metrics["route_stretch_p50"]).all())

    def test_water_is_looped_and_wastewater_is_directed_tree(self) -> None:
        metrics = pd.read_csv(OUT / "benchmark_case_metrics.csv")
        self.assertTrue((metrics[metrics["sector"] == "drinking_water"]["cycle_rank"] > 0).all())
        self.assertTrue((metrics[metrics["sector"] == "wastewater"]["cycle_rank"] == 0).all())

    def test_integrated_electricity_asset_and_operating_graphs(self) -> None:
        inventory = pd.read_csv(OUT / "network_inventory.csv").set_index("sector")
        self.assertGreater(inventory.loc["electricity", "cycle_rank_asset"], 0)
        self.assertEqual(inventory.loc["electricity", "cycle_rank_operating"], 0)

    def test_required_coupling_relations_exist(self) -> None:
        interfaces = pd.read_csv(OUT / "coupling_interfaces.csv")
        relations = set(interfaces["relation"])
        self.assertTrue({
            "delivered_water_to_sanitary_inflow",
            "electric_supply",
            "electric_heat_pump",
            "heat_plant_electrical_interface",
        }.issubset(relations))
        zones = pd.read_csv(OUT / "demand_zones.csv")
        mappings = interfaces[interfaces["relation"] == "delivered_water_to_sanitary_inflow"]
        self.assertEqual(len(mappings), len(zones))
        active = interfaces[interfaces["active_in_bidirectional_base_case"]]
        self.assertTrue((active["interface_direction"] == "bidirectional_state_exchange").all())
        self.assertTrue(active["return_state"].str.contains("solver_calculated_active_power_mw").all())

    def test_integrated_native_exports_pass(self) -> None:
        solvers = self.manifest["solver_readiness"]
        self.assertTrue(solvers["pandapower_uncoupled"]["converged"])
        self.assertTrue(solvers["epanet_wntr"]["converged"])
        self.assertTrue(solvers["swmm"]["engine_run_passed"])
        self.assertTrue(solvers["pandapipes_heat_supply"]["converged"])
        expected = [
            OUT / "power_pandapower.json",
            OUT / "water_epanet.inp",
            OUT / "schweinfurt_proxy.inp",
            OUT / "heat_pandapipes.json",
        ]
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in expected))

    def test_declared_engineering_screening_envelope_passes(self) -> None:
        self.assertTrue(all(item["passed"] for item in self.manifest["acceptance_screening"].values()))
        for case in self.benchmarks.values():
            self.assertTrue(all(item["passed"] for item in case["acceptance_screening"].values()))

    def test_evidence_boundaries_and_transformer_interpretation(self) -> None:
        audit = pd.read_csv(OUT / "evidence_boundary_audit.csv")
        self.assertIn("Water delivery, wholesale", set(audit["quantity"]))
        self.assertIn("MS/LV withdrawal locations", set(audit["quantity"]))
        transformers = pd.read_csv(OUT / "power_transformers.csv")
        self.assertAlmostEqual(
            transformers["sn_mva"].sum(),
            CASE["official_anchors"]["electricity_mv_lv_installed_capacity_mva"],
            delta=0.25,
        )
        self.assertTrue(transformers["asset_interpretation"].str.contains("equivalent portfolio").all())

    def test_bidirectional_fixed_point_is_converged_and_export_gated(self) -> None:
        coupled = self.manifest["solver_readiness"]["bidirectional_cosimulation"]
        self.assertTrue(coupled["accepted"])
        self.assertTrue(coupled["converged"])
        self.assertTrue(coupled["final_export_created"])
        self.assertGreaterEqual(coupled["iterations"], CASE["bidirectional_coupling"]["minimum_iterations"])
        final = coupled["final_sector_states"]
        self.assertTrue(all(final[sector]["passed"] for sector in final))
        self.assertGreater(final["electricity"]["attached_interface_loads"], 0)
        self.assertGreater(final["electricity"]["coupling_active_power_mw"], 0)
        self.assertLessEqual(final["drinking_water"]["maximum_pressure_m"], 110.0)
        self.assertGreaterEqual(final["drinking_water"]["minimum_pressure_m"], 20.0)
        self.assertEqual(
            final["wastewater"]["sanitary_inflow_factor"],
            final["drinking_water"]["delivered_water_fraction"],
        )
        self.assertGreater(final["district_heating"]["source_heat_mw"], final["district_heating"]["delivered_heat_mw"])
        expected = [
            OUT / "final_accepted" / "power_pandapower_bidirectional.json",
            OUT / "final_accepted" / "water_epanet_bidirectional.inp",
            OUT / "final_accepted" / "wastewater_swmm_bidirectional.inp",
            OUT / "final_accepted" / "heat_pandapipes_bidirectional.json",
            OUT / "final_accepted" / "bidirectional_acceptance_manifest.json",
        ]
        self.assertTrue(all(path.exists() and path.stat().st_size > 0 for path in expected))

    def test_bidirectional_residuals_and_repair_are_auditable(self) -> None:
        history = pd.read_csv(OUT / "cosimulation" / "bidirectional_convergence_history.csv")
        final = history.iloc[-1]
        settings = CASE["bidirectional_coupling"]
        self.assertTrue(bool(final["converged"]))
        self.assertLessEqual(final["voltage_residual_pu"], settings["voltage_tolerance_pu"])
        self.assertLessEqual(
            final["interface_power_l1_relative_residual"],
            settings["interface_power_l1_relative_tolerance"],
        )
        repairs = pd.read_csv(OUT / "cosimulation" / "bidirectional_repairs.csv")
        allowed_actions = {
            "extend_unchanged_swmm_initialization_run",
            "reduce_water_pump_speed_command",
            "upsize_limiting_heat_pipe_next_dn",
        }
        self.assertTrue(set(repairs["action"]).issubset(allowed_actions))
        self.assertEqual(repairs["attempt"].tolist(), list(range(1, len(repairs) + 1)))
        self.assertTrue(set(repairs["priority"]).issubset({2, 3, 4}))
        self.assertEqual(set(repairs["outcome"]), {"applied"})
        water_repairs = repairs[
            repairs["action"] == "reduce_water_pump_speed_command"
        ]
        if not water_repairs.empty:
            self.assertTrue((water_repairs["asset_id"] == "WAT-WW-01").all())
            self.assertTrue((water_repairs["new_value"] < water_repairs["old_value"]).all())
        heat_repairs = repairs[
            repairs["action"] == "upsize_limiting_heat_pipe_next_dn"
        ]
        self.assertGreaterEqual(len(heat_repairs), 1)
        self.assertTrue(heat_repairs["asset_id"].str.startswith("DH_R").all())
        self.assertTrue((heat_repairs["new_value"] > heat_repairs["old_value"]).all())
        final_heat = self.manifest["solver_readiness"]["bidirectional_cosimulation"][
            "final_sector_states"
        ]["district_heating"]
        self.assertTrue(final_heat["checks"]["minimum_differential_pressure"])
        self.assertTrue(final_heat["checks"]["heat_loss"])
        coupled = json.loads(
            (OUT / "cosimulation" / "bidirectional_coupling_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(coupled["repair_policy"]["policy_id"], "Urban4-PhaseIII-RP1")
        self.assertEqual(coupled["status"], "accepted")
        self.assertIsNone(coupled["terminal_failure"])
        iteration_dirs = sorted(
            path.name
            for path in (OUT / "cosimulation" / "iterations").iterdir()
            if path.is_dir()
        )
        self.assertEqual(len(iteration_dirs), len(history))
        self.assertEqual(iteration_dirs[-1], f"iter_{len(history):02d}")
        for obsolete in [
            "power_pandapower_coupled.json",
            "power_pandapower_heat_pump_candidate.json",
            "coupled_power_flow_summary.json",
        ]:
            self.assertFalse((OUT / obsolete).exists())

    def test_hydraulic_leak_response_is_time_resolved_and_cross_sector(self) -> None:
        scenario_dir = OUT / "hydraulic_leak_coupling"
        timeline = pd.read_csv(scenario_dir / "hydraulic_leak_timeseries.csv")
        manifest = json.loads((scenario_dir / "hydraulic_leak_manifest.json").read_text(encoding="utf-8"))
        settings = CASE["hydraulic_leak_demonstration"]
        self.assertTrue(manifest["accepted"])
        self.assertEqual(int(timeline["elapsed_minute"].iloc[0]), 0)
        self.assertEqual(int(timeline["elapsed_minute"].iloc[-1]), settings["simulation_end_minute"])
        self.assertTrue(timeline["elapsed_minute"].is_monotonic_increasing)
        pre = timeline[timeline["elapsed_minute"] < settings["leak_start_minute"]].iloc[-1]
        onset = timeline[timeline["elapsed_minute"] == settings["leak_start_minute"]].iloc[0]
        final = timeline.iloc[-1]
        self.assertEqual(float(pre["leak_flow_l_s"]), 0.0)
        self.assertGreater(float(onset["leak_flow_l_s"]), 0.0)
        self.assertGreater(float(final["pump_electrical_power_mw"]), float(pre["pump_electrical_power_mw"]))
        self.assertLess(float(final["pump_bus_voltage_pu"]), float(pre["pump_bus_voltage_pu"]))
        # The scenario is only 0.083% of city peak, so the reported system-wide
        # maximum loading is unchanged to numerical precision; the local bus
        # voltage and pump duty above are the informative cross-sector signals.
        self.assertLessEqual(
            abs(
                float(final["maximum_line_loading_percent"])
                - float(pre["maximum_line_loading_percent"])
            ),
            1e-6,
        )
        self.assertGreater(
            float(final["pump_electrical_power_mw"]) / float(pre["pump_electrical_power_mw"]) - 1.0,
            0.10,
        )
        self.assertLess(
            abs(float(final["pump_bus_voltage_pu"]) - float(pre["pump_bus_voltage_pu"])),
            0.001,
        )
        self.assertLess(float(onset["critical_node_pressure_m"]), float(pre["critical_node_pressure_m"]))
        self.assertGreater(float(final["critical_node_pressure_m"]), float(onset["critical_node_pressure_m"]))
        self.assertTrue(timeline["inner_coupling_converged"].all())
        self.assertTrue(timeline["water_limits_passed"].all())
        self.assertTrue(timeline["electricity_limits_passed"].all())
        self.assertTrue(timeline["step_accepted"].all())
        self.assertTrue((timeline["critical_node_pressure_m"] >= 20.0).all())
        self.assertTrue((timeline["maximum_pressure_m"] <= 110.0).all())
        self.assertTrue((timeline["maximum_line_loading_percent"] <= 100.0).all())
        self.assertTrue((timeline["maximum_transformer_loading_percent"] <= 100.0).all())
        self.assertTrue((scenario_dir / "water_epanet_hydraulic_leak_final.inp").stat().st_size > 0)

    def test_electrical_disturbance_propagates_to_water_and_recovers(self) -> None:
        scenario_dir = OUT / "electrical_supply_disturbance"
        timeline = pd.read_csv(scenario_dir / "electrical_supply_disturbance_timeseries.csv")
        manifest = json.loads(
            (scenario_dir / "electrical_supply_disturbance_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        settings = CASE["electrical_supply_disturbance_demonstration"]
        self.assertTrue(manifest["scenario_complete"])
        self.assertEqual(int(timeline["elapsed_minute"].iloc[0]), 0)
        self.assertEqual(int(timeline["elapsed_minute"].iloc[-1]), settings["simulation_end_minute"])
        self.assertTrue(timeline["elapsed_minute"].is_monotonic_increasing)
        pre = timeline[timeline["phase"] == "normal_pre"].iloc[-1]
        fault = timeline[timeline["phase"] == "fault_depressed"]
        post = timeline[timeline["phase"] == "normal_post"].iloc[-1]
        worst = fault.loc[fault["critical_node_pressure_m"].idxmin()]
        self.assertTrue(bool(pre["normal_state_limits_passed"]))
        self.assertTrue(bool(post["normal_state_limits_passed"]))
        self.assertLess(float(worst["drive_terminal_voltage_pu"]), 0.90)
        self.assertGreater(float(worst["drive_terminal_voltage_pu"]), 0.85)
        self.assertLess(float(worst["pump_actual_speed_pu"]), 0.75)
        self.assertLess(float(worst["critical_node_pressure_m"]), 20.0)
        self.assertLess(float(worst["delivered_water_fraction"]), 0.95)
        self.assertLess(float(worst["pump_electrical_power_mw"]), float(pre["pump_electrical_power_mw"]))
        self.assertLessEqual(
            abs(
                float(worst["maximum_line_loading_percent"])
                - float(pre["maximum_line_loading_percent"])
            ),
            1e-6,
        )
        self.assertAlmostEqual(
            float(post["delivered_water_fraction"]),
            float(pre["delivered_water_fraction"]),
            places=6,
        )
        self.assertTrue(timeline["inner_coupling_converged"].all())
        self.assertTrue(timeline["electricity_limits_passed"].all())
        self.assertTrue(timeline["disturbed_state_retained"].all())
        self.assertTrue(
            (scenario_dir / "water_epanet_worst_electrical_disturbance.inp").stat().st_size > 0
        )
        self.assertTrue(
            (scenario_dir / "power_pandapower_bidirectional.json").stat().st_size > 0
        )

    def test_wastewater_terrain_grade_sensitivity_reproduces_baseline(self) -> None:
        sensitivity_dir = OUT / "sensitivity"
        matrix = pd.read_csv(sensitivity_dir / "wastewater_terrain_grade_sensitivity.csv")
        manifest = json.loads(
            (sensitivity_dir / "wastewater_terrain_grade_sensitivity_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        settings = CASE["wastewater"]["terrain_grade_sensitivity"]
        self.assertTrue(manifest["topology_held_fixed"])
        self.assertEqual(
            len(matrix),
            len(settings["relief_scale_factors"]) * len(settings["minimum_gravity_grades"]),
        )
        baseline = matrix[matrix["is_baseline"]]
        self.assertEqual(len(baseline), 1)
        accepted_nodes = pd.read_csv(OUT / "wastewater_nodes.csv")
        accepted_links = pd.read_csv(OUT / "wastewater_conduits.csv")
        self.assertEqual(
            int(baseline.iloc[0]["lift_edge_count"]),
            int((accepted_nodes["node_type"] == "pump").sum()),
        )
        self.assertAlmostEqual(
            float(baseline.iloc[0]["force_main_length_km"]),
            float(
                accepted_links.loc[
                    accepted_links["link_type"] == "force_main", "length_m"
                ].sum()
                / 1000.0
            ),
            places=5,
        )
        self.assertEqual(
            (int(matrix["lift_edge_count"].min()), int(matrix["lift_edge_count"].max())),
            tuple(manifest["ranges"]["lift_edge_count"]),
        )
        self.assertLess(float(matrix["force_main_length_km"].min()), 5.0)
        self.assertGreater(float(matrix["force_main_length_km"].max()), 14.0)

    def test_swmm_stability_and_multi_seed_outputs(self) -> None:
        swmm = self.manifest["solver_readiness"]["swmm"]
        self.assertEqual(swmm["routing_duration_hours"], 48)
        self.assertEqual(swmm["steps_not_converging_percent"], 0.0)
        self.assertEqual(swmm["flooding_loss_million_liter"], 0.0)
        self.assertTrue(swmm["time_step_sensitivity_passed"])
        seeds = pd.read_csv(OUT / "clustering_seed_sensitivity.csv")
        self.assertEqual(len(seeds), 60)
        self.assertEqual(seeds.groupby("case_id")["seed"].nunique().min(), 20)
        reference_zones = {
            case_id: case["counts"]["demand_zones"]
            for case_id, case in self.benchmarks.items()
        }
        self.assertEqual(reference_zones["semi_urban"], 82)
        semi_seed_counts = seeds.loc[seeds["case_id"] == "semi_urban", "realized_zones"]
        self.assertEqual((int(semi_seed_counts.min()), int(semi_seed_counts.max())), (83, 88))
        ablation = pd.read_csv(OUT / "common_ledger_ablation.csv")
        self.assertTrue((ablation["independent_sector_clustering_p95_interface_offset_km"] > 0).all())

    def test_wastewater_facility_ledger_and_evidence_roles(self) -> None:
        facilities = pd.read_csv(OUT / "wastewater_facility_inventory.csv").set_index("category")
        lifts = int(facilities.loc["derived_lift_station_nodes", "count"])
        basins = int(facilities.loc["overflow_basin_equivalents", "count"])
        outfalls = int(facilities.loc["outfall_nodes", "count"])
        overlap = int(facilities.loc["co_located_lift_and_overflow_nodes", "count"])
        unique = int(facilities.loc["unique_facility_nodes", "count"])
        self.assertEqual(unique, lifts + basins + outfalls - overlap)

        checks = pd.read_csv(OUT / "aggregate_anchor_checks.csv").set_index("metric")
        self.assertEqual(checks.loc["Wastewater overflow basins", "evidence_role"], "input_constraint")
        self.assertEqual(checks.loc["Drinking-water route", "comparison_tolerance_basis"], "relative_percent")
        self.assertEqual(float(checks.loc["Drinking-water route", "comparison_tolerance"]), 10.0)
        self.assertEqual(checks.loc["Wastewater pump stations", "comparison_tolerance_basis"], "absolute_count")
        self.assertEqual(float(checks.loc["Wastewater pump stations", "comparison_tolerance"]), 2.0)

    def test_common_corridor_registry_has_multi_sector_records(self) -> None:
        corridors = pd.read_csv(OUT / "common_corridors.csv")
        self.assertTrue((corridors["sector_count"] >= 2).all())
        self.assertGreaterEqual(corridors["sector_count"].max(), 4)


if __name__ == "__main__":
    unittest.main()
