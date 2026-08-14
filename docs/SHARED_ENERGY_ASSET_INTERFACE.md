# Urban4 shared conversion-asset interface

## Purpose

Urban4 generates synthetic sector networks for coupled studies. Some physical assets
belong to more than one sector. The shared-energy-asset interface records those assets
once and exposes explicit sector ports without inventing a conversion law that is not
supported by public data.

GKS is the first configured instance. The same interface can later represent CHP,
power-to-heat, large heat pumps, electrolyzers or other shared conversion assets.

## GKS default contract

- common asset ID: `GKS_CHP`
- documented coordinate: GKS Schweinfurt
- electrical port: deterministically attached to the nearest compatible generated MV
  node; this is a synthetic-model attachment, not a claim about the real utility bus
- thermal port: H1 where the municipality heat-source table is available; otherwise the
  nearest generated heat-source node
- common availability state: one state gates both ports
- electrical nameplate metadata: 29 MW (12 MW + 17 MW published turbine ratings)
- default dispatch mode: `availability_only`
- default electrical setpoint: unset
- default thermal setpoint: unset

Therefore adding the interface does not change the current benchmark dispatch or
numerical results.

## Later coupled studies

`resolve_shared_asset_states()` creates the common operating state. Availability can be
overridden per scenario, for example `{"GKS_CHP": false}`.

The electricity adapter can apply an explicitly prescribed positive electrical setpoint
as a pandapower generation injection. It never substitutes the 29 MW nameplate as a
dispatch value.

`apply_heat_source_availability_to_network()` gates the matching named pandapipes heat
source. `solve_heat_network_for_shared_asset_state()` can then re-run the exported heat
network after an availability change.

The built-in heat adapter deliberately does **not** enforce a prescribed CHP thermal
output. A study that needs a heat-to-power feasible region, unit commitment, turbine
extraction curve or plant auxiliary model must provide a dedicated conversion/dispatch
model. This prevents Urban4 from inventing an unsupported CHP characteristic.

## Output files

- `coupling_interfaces.csv`: includes relation `shared_conversion_asset`
- `shared_energy_assets.csv`: service-resolved shared-asset contract
- `shared_energy_asset_states.csv`: resolved state produced by coupled closure
- `shared_energy_assets_municipal.csv`: municipality-scale contract

## Release behavior

Availability-only GKS adds no electrical generation and no thermal setpoint to the
normal benchmark. Existing 27 electrically driven facility interfaces remain unchanged.
The new shared asset is counted separately.
