# Urban4 v2.9.0 verification

- Python: **3.12**
- Automated tests: **83 passed, 0 failed**
- Represented households: **34,400**
- Building-level water-to-wastewater mappings: **22,641**
- Powered facility interfaces in the retained coupled benchmark: **27**
- Shared conversion assets: **1 availability-only GKS CHP interface**

## Release lineage

Version 2.9.0 is built directly on the corrected v2.8.0 rerun. The audited
electricity, drinking-water, wastewater, and district-heating demand equations
from v2.8.0 remain the numerical baseline. Version 2.9.0 adds reference
grounding, regression protection, and the shared-asset interface; it does not
revert to the v2.7.0 equations.

## Reference-grounded settings

| Item | Setting used in the verified run |
|---|---|
| Electricity | Kerber residential coincidence equation and published annual totals |
| Drinking water | DVGW W 410 evaluated at approximately 100,000 operator-served inhabitants: exact 2.610228786, upward-rounded to 2.62 for EPANET |
| Wastewater | 4.10 million m³/a sanitary upper bound; Harmon peak factor |
| District heating | 87.5 GWh/a and 1,600 full-load hours, giving 54.688 MW |
| Drive response | Binary electrical availability for the feeder-outage claim; no invented normal-operation voltage-speed droop |

The municipal population of 68,799 and the water operator's approximately
100,000 served inhabitants are separate evidence anchors and are tested as
such. The 4.1-million-m³/a direct-retail water volume is allocated to the
building layers. The 5.6-million-m³/a total delivery, including the wholesale
boundary volume, is used only for conservative source-station screening.

## Municipality-scale native outputs

| Sector | Native solver result |
|---|---|
| Electricity | pandapower converged; 0.950 p.u. minimum voltage; 82.2% maximum line loading; 33.5% maximum transformer loading |
| Drinking water | EPANET/WNTR accepted after 28 bounded repairs: 24 pipe upsizes and 4 head adjustments; 27.673–69.834 m peak pressure; 1.199 m/s maximum velocity |
| Wastewater | SWMM accepted; −0.047% continuity error; no modelled flooding |
| District heating | pandapipes accepted; 1.43 m/s maximum velocity; 99.8 Pa/m maximum catalogue pressure gradient; 7.60 bar source-lift screen; 8.08% heat loss |

The water source-station screen gives 177.6 L/s average and 465.2 L/s design
flow, with four 315-kW units: three duty and one standby. Pump and motor
efficiencies remain declared screening inputs rather than measured or
manufacturer-certified values.

## Coupled benchmark and shared asset

The building-level coupled benchmark converges in two iterations. Closed
interface demand is 0.492 MW, minimum voltage is 0.940 p.u., and maximum line
and transformer loadings are 90.1% and 82.3%.

The GKS interface records one physical CHP asset with an electrical port, a
district-heating port, and one common availability state. The documented
29-MW electrical nameplate is metadata only. With both operating setpoints
unset, the default availability-only interface adds no generation or heat
dispatch and leaves the benchmark numbers unchanged.

## Stress tests

- Downstream water leak: pump flow **+17.649%**, pump power **+15.242%**,
  first monitored pressure change **−11.497 m**, final pressure change
  **−1.996 m**, and waterworks connection-voltage change
  **−7.877×10⁻⁵ p.u.**
- Main-waterworks feeder outage: 304 of 1,255 buses are disconnected; the
  remaining energized network reaches 0.9235 p.u. minimum voltage and 80.52%
  maximum line loading. Delivered water demand falls to **0.0091%** and the
  minimum modelled pressure reaches **−60.45 m**.

No engineering acceptance threshold was relaxed. Generated files, solver
manifests, audits, figures, and checksums are recreated by:

```bash
python run_all.py
```
