# InfDB and pylovo integration contract

Urban4 integrates InfDB and pylovo through stable exports; it does not copy or
fork either project. This keeps ownership and licenses clear while preserving
their useful native identities.

## Responsibility boundary

| Component | Evidence or model retained by Urban4 |
|---|---|
| InfDB | LoD2 building geometry and use, Zensus occupants/households, basemap road ways, building-to-road connection lines, optional ro-heat building demands |
| pylovo | building-to-load mapping, transformer and feeder identities, LV buses/lines/loads/transformers, pandapower/OpenDSS-compatible LV models |
| Urban4 | connected MV topology, water, wastewater, district heat, shared corridor and interface ledgers, native acceptance criteria, bidirectional co-simulation |

Pinned revisions used to define the adapter are InfDB
`5f6784257cbc0c8a0ecf0454c75abfa3db5d9d4d` and pylovo
`fb9938dada80fcdb4172ecd0577c7ce812b39c37`.

## Why this is meaningful rather than cosmetic

1. `objectid` becomes the immutable Urban4 building identity.
2. InfDB service lines determine the physical building-to-road attachment where
   available; a nearest-road reconstruction is no longer substituted.
3. pylovo LV buses, feeders, transformers and building loads remain explicit.
4. Urban4 removes pylovo's per-grid external slacks, connects all transformer
   HV buses through one road-routed MV graph, and creates one upstream source.
5. Water laterals, sewer property connections and selected heat substations use
   the same building IDs and admissible corridors.
6. Zones are retained for clustering, calibration and reporting only. They are
   not the physical terminal nodes of the released networks.
7. Facility connections are selected by duty class: large duties connect to an
   MV node, medium drives to a transformer LV switchboard, and small lifts to a
   local LV service. The resulting pandapower model must pass again after all
   facility duties are attached.
8. Exact reused road segments receive stable cross-sector corridor identities;
   these are generated colocation records, not installed common-trench claims.

## Export and validation

Run `scripts/export_infdb_urban4.sql` with the InfDB PostgreSQL database, place
the resulting files under `data/infdb/schweinfurt_09662000`, and include the
pylovo pandapower JSON models in a `pylovo_grids` subdirectory. Urban4 validates
required columns, key uniqueness, foreign keys and GeoJSON before selecting the
backend. If this contract is missing, the retained OSM snapshot is selected and
`outputs/evidence_backend_status.json` records the reason. A fallback run is
never described as an InfDB run.

The core release does not require PostgreSQL. Reproducing the InfDB extraction
requires the upstream InfDB Docker/PostGIS environment and its own data-source
licenses; reproducing Urban4 from an already exported contract does not.

The currently packaged Schweinfurt run selected `retained_osm_snapshot` because
no InfDB PostgreSQL export was supplied. The adapter and pylovo merge are tested
and executable, but that fallback run must not be identified as an InfDB case. `evidence_backend_status.json` records this distinction.
