# Urban4 CSV topology and solver exports

The accepted service-resolved integration is under `outputs/integrated_service_resolved/`. Every physical terminal retains `building_id`; every link retains stable end IDs, type, size, length, and `geometry_json`.

| Sector/layer | Physical topology CSVs | Service/state CSVs | Native model |
|---|---|---|---|
| Common | `common_building_service_ledger.csv` | `evidence_backend_status.json` | — |
| Electricity | `electricity_nodes.csv`, `electricity_links.csv`, `electricity_transformers.csv` | `electricity_service_connections.csv`, `electricity_solver_results.json` | `electricity_pandapower.json`, `electricity_pandapower_with_interfaces.json` |
| Drinking water | `drinking_water_nodes.csv`, `drinking_water_links.csv` | `drinking_water_service_connections.csv`, `drinking_water_solver_results.json` | `drinking_water_epanet.inp` |
| Wastewater | `wastewater_nodes.csv`, `wastewater_links.csv` | `wastewater_service_connections.csv`, `wastewater_solver_results.json` | `wastewater_swmm.inp`, 30/15-s reports and binary outputs |
| District heat | `district_heating_nodes.csv`, `district_heating_corridors.csv`, `district_heating_supply_return_pipes.csv` | `district_heating_service_connections.csv`, `district_heating_sources.csv`, `district_heating_solver_results.json` | `district_heating_pandapipes.json` |
| Coupling | — | `coupling_interfaces.csv`, `coupled_facility_interface_states.csv`, `coupled_nominal_convergence_history.csv` | `coupled_nominal_interface_manifest.json` |
| Shared corridors | — | `shared_corridor_ledger.csv`, `shared_corridor_summary.json` | — |
| Release | — | `coupling_interface_audit.json`, `integrated_generation_manifest.json` | — |

The full four-solver coupling controller additionally writes `outputs/cosimulation/bidirectional_repairs.csv`. Its rows preserve the declared priority, sector, target asset, old/new values, reason, phase, and outcome. `outputs/cosimulation/bidirectional_coupling_manifest.json` records either `accepted` or the terminal `failed_no_feasible_repair`; a terminal failure never creates `outputs/final_accepted/`.


## Current municipality-scale accepted engineering inventory (v2.5.0)

`outputs/municipal_scale_v2.5.0/` is the current accepted evidence-conditioned full-city engineering design. It contains:

- `municipal_scale_summary.csv`: electricity, drinking-water, wastewater, and district-heating system scale and practical equipment selections;
- `electricity_transformers_municipal.csv`: the 105-site transformer allocation used to close the public transformation-capacity scale;
- `electricity_native_manifest.json`, `electricity_municipal_pandapower.json`, and `electricity_repair_log.csv`: executed AC state, fixed limits, and logged repairs;
- `drinking_water_station_municipal.csv`: W1 source flow, head, operating duty, and duty/assist/standby motor selection;
- `drinking_water_native_manifest.json`, zone/node/link/service tables, EPANET inputs, result tables, and repair log;
- `wastewater_pump_stations_municipal.csv`: S1-S13 municipal pumpwork screening duties, force-main sizes, TDH, and duty/standby motor selections;
- `wastewater_treatment_facility_municipal.csv`: the separate S0 treatment-process facility evidence and screening load;
- `wastewater_station_native_manifest.json`, SWMM input/report/results, 815-terminal mapping, and repair log;
- `district_heating_sources_municipal.csv`: H1 main GKS and H2 SHW Nord source-boundary evidence, synthetic hydraulic allocation, source-level circulation screen, and motor selections;
- `district_heating_temperature_profiles.csv`: the HT120/MT110/LT85 synthetic service classes derived from directly verified operator envelopes;
- `district_heating_published_area_profiles.csv`: the directly verified B1/B2, F1-Sued, Maintal, and V6 public operating envelopes retained as evidence;
- `district_heating_connectivity_audit.csv`: source/component reachability audit showing that every represented heat substation belongs to the active H1-H2-intertied component;
- `district_heating_corridors_municipal.csv` and `district_heating_services_municipal.csv`: current single-component heat-routing candidate;
- `district_heating_native_manifest.json`, pandapipes model/results, and repair log;
- `fig07_schweinfurt_municipal_scale_v2_5_0.{pdf,svg,png}`: aligned four-panel accepted municipal designs.

All four v2.5.0 municipality-scale sector outputs pass their declared native
acceptance criteria with unchanged thresholds. They remain distinct from the
Experiment-A coupled benchmark; no full municipality-scale coupled fixed point
is claimed.

## Archived reproduced coupling baseline

`outputs/publication_ready_v1.5.3/` and `outputs/integrated_service_resolved/` preserve the earlier reproduced 27-interface coupled baseline. Its H1-H14 heat boundaries are retained only so that the published numerical coupling proof remains reproducible. They are not the current municipality-scale interpretation of Schweinfurt's district-heating supply architecture.

## Terminal contract

- electricity: one building bus/load and service cable for every eligible LV service; direct MV customers remain explicit;
- drinking water: one service junction and lateral per eligible building;
- wastewater: one property node and gravity/force lateral per eligible building;
- district heat: one substation and service route per selected building;
- water-to-wastewater: one exact mass-mapping interface per common service identity;
- powered facilities: selected bus, voltage class, distance, nominal duty, forward-duty equation, and reverse-response equation.

Demand zones are clustering and reporting objects. They are not physical service terminals.

## Independent context cases

`outputs/benchmark_cases/dense_urban/`, `semi_urban/`, and `rural_peripheral/` each contain an independently clipped common ledger, planning zones, all four service-resolved topology CSV sets, native models, solver results, interfaces, and a manifest.

## InfDB/pylovo alternative exports

If upstream pylovo pandapower JSON files are supplied with a validated InfDB contract, the electricity output additionally includes:

- `electricity_pandapower_pylovo_urban4.json`;
- `electricity_building_loads.csv`;
- `electricity_mv_backbone.csv`;
- `electricity_mv_transformer_spurs.csv`.

The merge preserves upstream LV identities and replaces independent pylovo slack buses with one Urban4 MV source. The current packaged fallback run does not claim to be an InfDB or pylovo result.

## Figures and audits

Every publication figure is available in `figures/` as PDF and SVG, with PNG previews. Figure 3 is reproducible with `scripts/make_process_figures.py`; its graph/coverage audit is `outputs/figure3_topology_audit.json` and `FIGURE3_TOPOLOGY_AUDIT.md`.

Exact shared road-segment reuse denotes synthetic co-location only; it is not evidence of an installed common trench.
