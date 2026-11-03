# Győr, Hungary — Urban4 v0.2 working case

3 November 2026 · synthetic, solver-checked network models · local partner review pending.

Urban4 has been used to build a **common-building** representation of four networks in Győr, Hungary. This case is a *synthetic engineering baseline*. Generated routes, transformer and pump locations, district-heating injection boundaries, customer consumption and operating settings should not be read as maps or measurements of installed utility infrastructure.

| Network | Nodes | Links | Building connections | Native solver |
| --- | ---: | ---: | ---: | --- |
| Electricity | 33,047 | 32,647 | 24,963 | pandapower — passed |
| Drinking water | 32,461 | 32,547 | 24,963 | EPANET — passed |
| Wastewater | 32,384 | 32,383 | 24,963 | SWMM — passed |
| District heating | 2,612 | 2,604 | 1,682 | pandapipes — passed |

The common register contains 27,519 retained building footprints and 24,963 candidate service buildings. The service basis and building identifiers are shared across electricity, water and wastewater. The district-heating selection includes a subset of 1,682 buildings.

**Numerical checks.** In the drinking-water model, customer pressure was 31.5–66.0 m at average demand and 29.0–68.0 m at the 2.5× design hour. Sewer continuity errors were −0.588% (10 s), −0.591% (5 s) and +0.420% (2× sanitary inflow), with no reported flooding. District-heating steady supply hydraulics, differential pressure, velocity and route heat-loss screens passed.

**Coupling.** The converged operating snapshot uses 213 utility facility/electricity interfaces and 1.793 MW of utility facility electrical demand. Minimum voltage was 0.932 pu, maximum line loading 83.8%, and maximum transformer loading 90.8%. All connected drives satisfied the chosen availability criterion. The snapshot combines coincident building electric demand, average water demand, a six-hour dry-weather sewer initialization and design heat load.

**Principal local checks requested:**

- Confirm service polygons, mapped building use, storey counts and major industrial loads.
- Verify actual MV/LV voltage levels, feeders, substations, transformer sizes and electrical load profiles.
- Verify drinking-water inlets (Révfalu and Szőgye), zones, pumps, pipe sizes and demand.
- Verify sewer catchments, external inflows, combined systems, inverts, pumping stations and receiving treatment facilities.
- Verify connected district-heat customers, installed routes, heat sources, temperatures, annual sales and peak loads.
- Verify electric connections and backup supply for water, sanitary and heat facilities, including controls and relevant disturbance scenarios.

Aggregate annual and peak demand, approximate facility locations, public asset layouts and marked-up maps are welcome if detailed utility GIS cannot be shared. Please supply units, reference year and geographical coverage for every correction.

**Archive status:** The full delivered model archive (`Gyor_Urban4_partner_baseline_v0_2.zip`, 63,724,406 bytes, SHA-256 `748354f37b65841745a682aa669ccd62ed805e431341485cbb66a1caa43f9b04`) is supplied separately for partner review. **It is not yet stored in this GitHub repository.** It contains the GeoPackage, native solver models and inputs, building ledger, coupling records, test results, figure files, raw evidence snapshots and runnable case-specific scripts. This README and the accompanying metadata are an index, not a substitute for the archive.

The archive was produced against Urban4 commit [`4d74b8c`](https://github.com/Prashantgrid/Urban4/commit/4d74b8c37f7270bae862875de736d6ec7a43ee0e). The archive includes a small city-portability patch and `run_working_models.py` to repeat delivered solver checks. For raw regeneration, refer to the archive README.

Source map/evidence attribution: © OpenStreetMap contributors, ODbL 1.0. Terrain prior: SRTM90m through Open Topo Data. Preserve source provenance and applicable reuse obligations.
