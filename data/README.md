# Data provenance and licensing

## Retained geospatial evidence

The files in `data/raw/` are frozen Overpass API responses for the Schweinfurt
study extent. They were retrieved on 27 July 2026 and retained as lossless
`.json.xz` archives. Urban4 reads the archives directly; decompression restores
the exact public map state used by Urban4 v2.7.0.

| File | Role |
| --- | --- |
| `buildings.json.xz` | Building geometries, identifiers, and public map tags |
| `local_roads.json.xz` | Roads and paths used as admissible routing evidence |
| `infrastructure.json.xz` | Publicly visible infrastructure objects |
| `sewer_detail.json.xz` | Publicly mapped wastewater and drainage details |
| `context.json.xz` | Land use, administrative, and cartographic context |

The retained bounding box is 10.14 to 10.31 degrees east and 49.985 to 50.095
degrees north. The declared service mask and all aggregate evidence are stored
in `cases/schweinfurt.json`.

## OpenStreetMap licence

The retained JSON archives contain OpenStreetMap data:

> © OpenStreetMap contributors, available under the Open Database License
> (ODbL) 1.0.

Attribution and licence information are available at
<https://www.openstreetmap.org/copyright>.

The MIT licence at the repository root applies to Urban4 source code. It does
not replace the ODbL or the original terms of other cited public sources.

## Public utility evidence

Published totals, operating envelopes, and source locations are recorded as
constraints or fixed public boundaries in `cases/schweinfurt.json`. Source
URLs are retained beside the relevant values. These quantities do not provide
confidential pipe or cable geometry.

## Claim boundary

Urban4 does not reconstruct installed utility networks. It does not contain
private customer data, tariffs, confidential asset nameplates, plant dispatch,
or private control policies. Generated edges and customer allocations are
synthetic unless explicitly labelled as observed public evidence.
