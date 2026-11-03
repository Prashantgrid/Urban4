# Andermatt frozen public inputs

Snapshot retrieved on 3 November 2026. Coordinate reference system for all metric calculations is EPSG:2056 (metres).

| Input | Source / use |
| --- | --- |
| `fwa_trasse_linie.geojson` | Lisag AG / Geoinformation Kanton Uri, public WFS `leitungen:fwa_trasse_linie`; reference routes, held out except the declared local inlet |
| `fwa_leitungspunkt_einspeisung.geojson` | Same WFS, `leitungen:fwa_leitungspunkt_einspeisung`; 85 operator-owned `Hausuebergabestation` points supplied to both methods |
| `fwa_bauwerk_flaeche.geojson` | Same WFS, plant/structure context retained for traceability; not used as a model source |
| `osm_snapshot.osm` | Raw OpenStreetMap map API XML, retained unchanged |
| `osm_snapshot.json` | Parsed OSM ways, including highway and building geometry, used by the benchmark |
| `common_inputs.gpkg` | Frozen derived layers: customers, source, boundary, installed_reference |
| `case.json` | Fixed bounds, local inlet, scenario loads, filters and scope |
| `*_request.json` | Exact request URLs and retrieval provenance |
| `sfoe_metadata.json` | Context for the former inventory citation; not a route reference or a model input |
| `source_manifest.json` | Byte sizes and SHA-256 checksums of the frozen source files |

The common window is the bounding box of the 85 observed transfer points, expanded by 250 m, covering 1.803 km². The reference uses the operator `Netzgesellschaft Andermatt AG` and status `in_Betrieb`, clips to this window and counts unique one-way route geometry. Parallel supply/return pipes and duplicate overlaps are not double counted. There are 600 active operator-owned route features in the wider retrieved snapshot; clipping and union yield 10.863 km in the evaluation window.

Both methods receive the northern incoming trunk crossing as the same local inlet. The upstream Göschenen plant and mountain transmission route are outside this local distribution experiment. This avoids assuming that the small cadastral structure inside Andermatt is the main heat plant.

Each observed transfer point receives a scenario peak load of 50 kW and 1,600 equivalent full-load hours. These are not measured demands. The point layer is not assumed to enumerate all actual customers. The current 85-point case is distinct from the manuscript's earlier 50-point sample; the earlier exact raw sample and extraction protocol were not retained.

## Attribution and reuse

- Cadastral data: **Lisag AG / Geoinformation Kanton Uri**, [Fernwärme Leitungen catalogue](https://opendata.swiss/de/dataset/fernwarme-leitungen), [public WFS](https://geo.ur.ch/wfs?request=getCapabilities). Resource terms: [Open use](https://opendata.swiss/terms-of-use#terms_open). Feature import date in the snapshot: 30 October 2026. Preserve provider attribution and consult the linked resource terms when reusing.
- Road/building data: **© OpenStreetMap contributors**, [ODbL and attribution](https://www.openstreetmap.org/copyright). The OSM snapshot and OSM-derived geometry retain those terms; the repository's MIT software licence does not replace source data licences.
- Reference topology is inferred from 2D mapped geometry. Crossings, tiny gaps, boundary endpoints and service mapping may not represent verified hydraulic connectivity.

Full execution protocol, limitations and the manuscript replacement text are in [`docs/andermatt_topotherm_benchmark.md`](../../docs/andermatt_topotherm_benchmark.md).
