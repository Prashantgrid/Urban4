# Municipality-scale build, repair, and release logic (v2.5.0)

Urban4 separates observed evidence, synthetic topology decisions, native-solver
acceptance, and coupled acceptance. Public totals and documented facilities
condition generation; they are not reused as independent validation and do not
imply reconstruction of confidential installed edges.

## Common state machine

For every sector the generator freezes its inputs and limits, builds a complete
candidate, executes the relevant native solver, diagnoses the limiting asset,
applies the next allowed catalogue/control repair, and reruns the entire model.
The threshold set is immutable. All repairs are logged. Only an accepted state
is written to the municipal release manifest.

## Electricity

- Build 105 synthetic MV/LV service sites and radial feeders from the shared
  road/building evidence; allocate 0.80–4.00 MVA catalogue transformers until
  the 216.71 MVA public calibration scale is closed (216.95 MVA generated).
- Execute AC power flow in pandapower.
- If voltage, loading, or LV path-drop limits fail, advance the limiting cable
  by one catalogue class; if the class is exhausted, add one separately
  protected parallel feeder circuit. Rerun after each round.
- Bounds: at most 20 construction splits, five repair rounds, and 500 conductor
  changes.
- Accepted output: 0.94954–1.03000 p.u.; maximum line/transformer loading
  82.18/33.53%; maximum LV path drop 0.05354 p.u.; zero repair actions.

## Drinking water

- Form four 20-m elevation pressure zones before routing. Aggregate demand on a
  0.12-km access grid, route each zone independently on roads, add explicit
  controlled zone boundaries, and add six second feeds to critical access
  nodes. The 341-km public main inventory is a calibration target with a fixed
  10% tolerance; the generated network is 369.095 km (+8.24%).
- Size from accumulated peak flow using an ordered DN catalogue and execute
  average and two-times-demand states in EPANET through WNTR.
- First adjust the failed zone's head-controlled boundary. If no feasible head
  shift can close the pressure window, advance the highest-loss main on the
  active critical path by one DN class. Rerun both states after every action.
- Bounds: four head adjustments per zone and 60 pipe upgrades.
- Accepted output after 34 logged actions: average pressure 35.00–70.00 m;
  two-times-demand pressure 27.63–69.88 m; peak velocity 1.220 m/s; minimum
  delivered-demand fraction 0.9999998.

## Wastewater

- Retain the terrain-aware full-city spatial layer (6,179 manholes, 235.191 km
  gravity collection, 13.102 km force main, 815 evidence-conditioned
  subcatchments and 13 station compounds).
- Construct a station-scale SWMM hydraulic equivalent with exactly 13 wet wells,
  13 active PUMP3 objects and 13 force mains. The equipment inventory contains
  26 installed physical pump units (one duty and one standby per station). The
  815 terminal inflows preserve the
  3.362-million-m³/a registered sanitary total.
- Execute 24 h DYNWAVE at 60 s. If continuity, nonconvergence, or flooding fails,
  extend initialization 24→48 h, halve the routing step 60→30→15 s, then enlarge
  wet-well area by 25% per action. The fixed budget is eight actions.
- Accepted output: −0.044% continuity error; 0% nonconverging steps; 0% flooding;
  no repair action required.

## District heating

- Screen customers using the declared heat-density/route grid, retain 841
  contract equivalents and 87.5 GWh/a, connect both documented H1/H2 boundaries,
  and add an intertie so all substations belong to one source-reachable graph.
  The selected comparable route is 51.863 km against the 52.0-km calibration
  input.
- Assign HT120/MT110/LT85 temperature classes from published operating
  envelopes, select paired supply/return DN and PN classes, and execute the
  network in pandapipes.
- For a velocity or pressure-gradient violation, advance every violating
  corridor by one DN class and rerun. Bounds: six rounds and 500 upgrades.
- Accepted output after one round/31 upgrades: maximum native velocity
  1.49993 m/s; maximum catalogue gradient 99.75 Pa/m; critical source screen
  7.60 bar; route heat loss 7.066 GWh/a (8.08%). The 10-bar solver slack is a
  numerical reference only and is not interpreted as municipal absolute
  operating pressure.

## Four-sector release

`outputs/municipal_scale_v2.5.0/manifest.json` sets
The legacy manifest field `release_gate_passed=true` is written only when all four native manifests pass and every
represented heat substation remains source reachable. The municipality-scale
states are accepted sector models; a separate municipality-scale coupled
fixed-point result is not claimed.
