# Andermatt: installed-route accuracy comparison

Executed on 3 November 2026 using actual Urban4 and Topotherm code. This is a **conditional local DH routing benchmark** with common observed connection locations, not an accuracy test of demand estimation or customer selection.

The result is a tradeoff. Urban4 reproduces more of the mapped extent; Topotherm's smaller network follows installed corridors more closely. The data support a quantified, case-specific routing claim, not a claim that either method is uniformly more accurate.

## Shared experiment

- 85 cadastral house-transfer points; all 85 are required connections for both methods.
- One local inlet at the mapped northern incoming trunk, excluding the upstream Göschenen plant and transmission line.
- Uniform scenario demands of 50 kW per point and 1,600 full-load hours; 4,250 kW total. No measured customer demands are available.
- One frozen OSM snapshot, common allowed road classes, nearest-road-node service attachments and candidate graph. Highway classes excluded: motorway, motorway_link, trunk, trunk_link, steps, construction, proposed and raceway. Road candidates extend 200 m beyond the fixed scoring window. Unused non-terminal leaf spurs are pruned; degree-2 chains are contracted with alternative parallel corridors retained. The reduced graph has 411 nodes and 551 edges.
- The common 1.803 km² scoring window is the observed point bounding box expanded by 250 m. All networks are clipped to that window for scoring. The same 511 OSM building centroids are used for the building-distance measure (valid building polygons of at least 20 m²).
- Cadastral active route geometry is held out from route construction, apart from the declared inlet location. No overlap or morphology score is used to tune either model.

Urban4 uses its route-aware heat selector with the published exponent 0.5 and all observed points required. With all candidates selected, routing is its rooted shortest-path forest plus the common service connectors; this setup does not test its customer-selection policy. Topotherm 0.6.0 uses its official single-timestep **forced** connection model at commit `ec108e94b2e3f823a154afc47531156b1c85539f`, with default pipe cost/loss regressions. Source investment is zero, capacity is 8,500 kW, and constant forced-connection revenue is omitted. Exact economics are saved in `topotherm_run.json`. Contraction preserves corridor lengths and shortest paths; the thermal regression is applied to each contracted corridor, not to every original OSM segment.

## Same six Table XII measures

Ratios are generated / installed; **1 is agreement**, not a universal higher-is-better score. Units below apply to the installed absolute column. Both methods use the same explicit definitions, saved in `scoring_protocol.json`.

| Measure | Installed absolute | Urban4 ratio | Topotherm ratio |
| --- | ---: | ---: | ---: |
| Route length / area (km/km²) | 6.024 | 0.880 | 0.622 |
| Route length / observed connection (m/point) | 127.80 | 0.880 | 0.622 |
| Mean nearest-neighbour connection spacing (m) | 28.68 | 1.000 | 1.000 |
| Mean building-to-route distance (m) | 28.95 | 1.390 | 1.750 |
| Mean branch length (m) | 21.09 | 2.412 | 1.760 |
| Terminal share | 0.5010 | 0.975 | 1.004 |

Connection spacing is a **fixed input control**, not independent accuracy evidence. The first two ratios measure the same route extent with different denominators. Branches run between degree-not-equal-to-2 nodes after common 0.5-m geometry quantisation and noding; terminal share uses the degree-2-suppressed graph. These are inferred geometry descriptors, not verified physical connectivity. Precision sensitivity at 0.25, 0.5 and 1 m is included in `topology_precision_sensitivity.csv`.

Installed, Urban4 and Topotherm unique local route lengths are **10.863, 9.559 and 6.754 km**, respectively. Mean building-to-route distances are **28.95, 40.23 and 50.68 m**.

## Direct spatial agreement

Installed-route coverage is the fraction of mapped reference length within the tolerance of the generated network. Generated-route agreement is the reverse fraction. Both are length weighted using geometric corridor intersection, not counts of vertices. The harmonic mean describes the balance of those two fractions; it is not an independent measurement.

| Tolerance | Urban4 reference covered | Topotherm reference covered | Urban4 generated matching | Topotherm generated matching |
| --- | ---: | ---: | ---: | ---: |
| 10 m | 50.0% | 43.5% | 55.0% | 63.2% |
| 25 m | 68.1% | 62.7% | 74.4% | 86.7% |
| 50 m | 83.3% | 78.7% | 81.7% | 91.7% |

The 25-m harmonic means are 71.1% and 72.8% for Urban4 and Topotherm. Mean generated-to-installed route error is **28.04 m versus 15.04 m**, and its 95th percentile is **134.60 m versus 61.45 m**. Reverse installed-to-generated mean error is **27.12 m versus 31.23 m**. Distance errors use arclength midpoint samples with maximum 5-m spacing and length weights. The entire road candidate universe can cover 90.6% of installed length within 25 m, which exposes the road-corridor representation limit.

## Solver and validation evidence

Topotherm ran with Pyomo 6.10.1 and HiGHS 1.15.1, two threads, seed 20261103, a 600-s limit and requested 1% MIP relative gap. It finished after **421.96 s**, with incumbent €889,678.05/year and lower bound €880,782.32/year, giving **0.999881% relative gap**. The solver reports `optimal` at this tolerance; the result is not an exact zero-gap proof. There are 239 selected reduced edges and all 85 customers are connected. Source supply is 4,402.12 kW, including modelled heat losses.

The independent replay audit checks node balances, directional capacities, binary integrality, connectivity, the common graph and contraction distances. Maximum nodal residual is 1.89e-11 kW, maximum capacity violation 4.01e-7 kW and contraction shortest-path difference 4.45e-16 km, all within numerical tolerances. Six focused geometry/contraction tests pass. Urban4's routing kernel took 0.239 s; that timing excludes input preparation and should not be presented as a full application runtime or an equivalent optimisation objective.

## Reproduce

Run from the repository root with Python 3.11 or newer. A complete rerun requires git access for the pinned Topotherm dependency; scoring the frozen results requires no live GIS queries and no solver execution.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements/andermatt-benchmark.txt
.venv/bin/python -m pytest -q tests/test_andermatt_benchmark.py

# Re-score and audit the published solution without re-optimising:
.venv/bin/python scripts/benchmarks/score_andermatt_topotherm.py --results reference_results/andermatt_topotherm
.venv/bin/python scripts/benchmarks/audit_andermatt_solution.py --results reference_results/andermatt_topotherm

# Fresh solve into the ignored outputs directory, then score and plot:
.venv/bin/python scripts/benchmarks/run_andermatt_topotherm.py
.venv/bin/python scripts/benchmarks/score_andermatt_topotherm.py
.venv/bin/python scripts/benchmarks/audit_andermatt_solution.py
.venv/bin/python scripts/benchmarks/plot_andermatt_comparison.py
.venv/bin/python scripts/benchmarks/report_andermatt_benchmark.py
```

Frozen sources are in `data/andermatt_benchmark`; frozen solution geometry, decisions, solver log and scores are in `reference_results/andermatt_topotherm`. Input hashes are in `source_manifest.json`, output hashes in `results_manifest.json`. Versions used are in `software_versions.json`. Rebuilding `common_inputs.gpkg` from the frozen WFS snapshots is optional (`prepare_andermatt_inputs.py`); a rebuilt GeoPackage can have different byte-level metadata while retaining the same geometries.

## Manuscript text

> We evaluated conditional district-heating route reconstruction in Andermatt against active cadastral routes, using 85 observed transfer points, one local inlet and identical OSM road candidates for Urban4 and Topotherm. Uniform 50-kW scenario loads were imposed at all points. Urban4 and Topotherm generated 9.56 and 6.75 km of routes against 10.86 km mapped in the common window. At a 25-m tolerance, Urban4 covered 68.1% of installed route length and Topotherm 62.7%; the reverse fractions were 74.4% and 86.7%. Mean generated-to-installed route distances were 28.0 and 15.0 m. Urban4 better reproduced mapped extent and building proximity, whereas Topotherm better localised its more compact network. Topotherm reached a 1% MIP gap in 422 s. These results quantify a routing tradeoff conditional on supplied connections and scenario loads; they do not validate customer selection, measured heat demand or hydraulic performance.

**Editorial integration:** this is a new, frozen **85-point** case. The former Table XII Andermatt row used a **50-point** sample whose exact inputs and extraction script are unavailable. Replace that row and its related 6.6 building-distance discussion with this case, or label the cases separately. Do not present the new ratios as a numerical reproduction or improvement of the old row. Update any abstract/conclusion claim that relies on the old Andermatt ratios. A case comparison supplies no statistical population confidence interval or multi-city generalisation.

## Primary sources

- [Lisag public geoservices](https://www.lisag.ch/geodienste) and [Uri route dataset](https://opendata.swiss/de/dataset/fernwarme-leitungen). These supply detailed route geometries; the [SFOE thermal-network inventory](https://opendata.swiss/en/dataset/thermische-netze-nahwarme-fernwarme-fernkalte) alone is not a detailed installed-route reference.
- [Official Topotherm repository, pinned commit](https://github.com/jylambert/topotherm/tree/ec108e94b2e3f823a154afc47531156b1c85539f).
- [Heizwerk Gotthard operator description](https://www.heizwerk-gotthard.ch/index.php/das-heizwerk-gotthard), establishing the upstream Göschenen context.
- [OpenStreetMap attribution and ODbL](https://www.openstreetmap.org/copyright). Exact query URLs and file hashes are retained with the data.
