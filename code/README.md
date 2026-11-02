# Final Urban4 study code

Run commands below from the repository root with Python 3.12. The study uses
the corrected v2.9 core source from commit
`4943579c03566f543267e0889de53e314f75d7ea`, now integrated with the final scripts.

## Reproduce the retained sensitivity results

```bash
python -m pip install -e '.[test,study]'
python code/reproduce_study.py
```

This copies the frozen tables to `outputs/manuscript_study/results/`, recomputes
both transformer sensitivities, and regenerates Figures 3, 6, and 7 there.
The retained tables in `results/` are never overwritten by this command.
Figure 7 uses the retained vector map and regenerates its three numerical panels.
Figure 6 uses the retained classified Seattle overlay.

## Recompute the case studies from native models

```bash
python run_all.py
python code/reproduce_study.py --from-models
```

The first command builds the four sector models; the second performs the
following studies in dependency order:

| Script | Calculation |
|---|---|
| `run_heat_power_case.py` | Winter sweep of existing DH, heat-pump conversion, CHP outage with electric backup, and a reinforced heat link |
| `run_heat_power_variants.py` | Study-based heat-pump ratio, renovation variants, and overload-removal cost screens |
| `run_water_power_case.py` | Single MV-line contingencies and their water/wastewater impacts |
| `run_water_reliability.py` | Open-ring ties, AC-checked restoration, storage, annual losses, and blackout EPANET time series |
| `run_flood_case.py` | Flood exposure, damaged power supply, and cross-sector consequences |
| `run_extreme_events.py` | Time to service loss and cumulative losses for the documented durations |
| `run_base_load_sensitivity.py` | Transformer sum checks at 100%, 80%, and 60% reference electrical demand |
| `run_capacity_sensitivity.py` | Transformer sum checks with generated ratings, published capacity per retained building, and full published capacity |

The full-case mode also generates Figures 3 through 9. Native models are read
from `outputs/municipal_scale_v2.5.0/`; this historical directory name is retained
for compatibility. Outputs go to `outputs/manuscript_study/` by default.
Use `--repo PATH` to select rebuilt models and `--output PATH` for a separate
output directory. The two transformer sensitivities recalculate apparent-power
loading; they do not recalculate AC voltages or cable loading.

## Individual script examples

```bash
python code/run_base_load_sensitivity.py --results results/heat_power_case
python code/run_capacity_sensitivity.py --results results/heat_power_case
python code/make_fig07_heat_power_revised.py --out outputs/figure_check
```

These individual commands write their specified results; use the reproduction
wrapper to keep the reference tables separate automatically.

`u4style.py` supplies shared typography and sector colours. `u4map.py` draws
maps from the retained OSM snapshot. `scenario_names.py` maps stable machine
keys to descriptive names. The legacy ratio key `measured` denotes the
study-based 0.40 sensitivity and is retained for compatibility.

## Data and scope

`code/data/` contains the retained flood-hazard extents, Seattle route overlay,
and vector reference map. The LfU flood extents (HQ100 and HQextrem; Main, Wern,
Marienbach; WWA Bad Kissingen) were retrieved on 30 September 2026. The data
retain their source attribution and original terms; see `code/data/README.md`.

The studies use the declared generated networks and scenario assumptions.
The 0.40 heat-pump factor is a study-based aggregate sensitivity, and the COP
and temperature relations are explicit assumptions. Retained CSV/JSON files
and their SHA-256 checksums are documented in `results/manifest.json`.
