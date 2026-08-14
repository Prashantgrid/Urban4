# Urban4 v2.8.0 verification

- Python: **3.12**
- Automated tests: **70 passed, 0 failed**
- Represented households: **34,400**
- Building-level water-to-wastewater mappings: **22,641**
- Powered facility interfaces in the retained coupled benchmark: **27**

## Updated demand settings

| Sector | Setting used in the verified run |
|---|---|
| Electricity | Kerber residential coincidence equation and published annual totals |
| Drinking water | 2.62 design peak factor; 5.6 million m³/a system delivery |
| Wastewater | 4.10 million m³/a sanitary upper bound; Harmon peak factor |
| District heating | 87.5 GWh/a and 1,600 full-load hours, giving 54.688 MW |

## Municipality-scale native outputs

| Sector | Native solver result |
|---|---|
| Electricity | pandapower converged; 0.950 p.u. minimum voltage; 82.2% maximum line loading |
| Drinking water | EPANET/WNTR converged; 27.7–70.0 m peak pressure; 1.20 m/s maximum velocity |
| Wastewater | SWMM converged; −0.047% continuity error; no modelled flooding |
| District heating | pandapipes converged; 1.43 m/s maximum velocity; 99.8 Pa/m maximum catalogue pressure gradient |

The complete records are regenerated under `outputs/`. The retained aggregate
heating-route comparison with Topotherm 0.6.0 is stored in
`reference_results/topotherm_comparison/`. That comparison checks route
construction only; it is not an installed-network validation. Granular
municipality node, edge, and demand files are intentionally not added to this
source release.

## Stress tests

- Downstream water leak: pump flow **+18.1%**, pump power **+16.0%**, first
  pressure change **−11.7 m**, final pressure change **−1.8 m**.
- Main-waterworks feeder outage: delivered demand falls to **0.009%** and the
  minimum modelled pressure reaches about **−60 m** before reconnection.

No engineering acceptance limit was relaxed.
