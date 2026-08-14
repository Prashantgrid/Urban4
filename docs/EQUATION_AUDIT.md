# Urban4 equation audit

Date: 12 August 2026

Scope: every displayed equation in the methodology, the equations written in the interface table, and each numerical coefficient that changes demand or component sizing. The audit compares these relations with the corrected Python implementation and regenerated results.

## Decision rule

An equation is acceptable only when it is one of the following:

1. a conservation law or unit conversion;
2. a published empirical relation used inside its stated range;
3. an accounting identity defined by Urban4;
4. an author-defined numerical or selection rule whose effect is tested.

Calling an unsupported number an “Urban4 assumption” is not enough when that number changes asset sizes.

## Main finding

The physical and accounting equations are sound. The unsupported demand and response constants were replaced in the code before the final rerun:

- households now come from published population and regional household size, not gross floor area divided by 90 m²;
- residential electricity uses the local 3500-kWh household case, and non-residential peaks use BDEW profiles;
- water uses the published population-based factor 2.62;
- the registered wastewater flow is treated as a 100% sanitary upper bound and uses the population-based Harmon factor 2.139;
- heat uses the pinned F|Heat function values and 1600 full-load hours;
- heat loss uses DN-specific carrier and casing dimensions and published insulation conductivity;
- treatment power uses the reported local electricity use;
- the unsupported voltage-to-speed curve was removed.

The full workflow was regenerated. All native sector checks, both coupled checks, and all 70 automated tests passed without changing the acceptance limits.

## Equation-by-equation check

| ID | Equation or choice | Where used | Basis found | Verdict | Required correction |
|---|---|---|---|---|---|
| E1 | \(D_{b,s}=p_{b,s}D_s^{off}/\sum_jp_{j,s}\) | Annual electricity and water allocation | Urban4 normalization identity | Pass | State that it preserves the known total but does not validate the building weights. |
| E1a | Building-level electricity and water allocation weights | Spatial allocation before E1 | Residential electricity uses household counts; non-residential electricity uses pinned F|Heat function values; residential water follows population and non-residential water uses equivalent-population weights before reconciliation | Pass as a documented fallback | Replace with metered building data when available. Reconciliation preserves the published total but does not validate individual buildings. |
| E2 | \(P_{b,E}^{coinc}=p_{b,E}P_E^{off}/\sum_jp_{j,E}\) | Shared transformers and feeders | Urban4 normalization identity | Pass | Keep. The official coincident peak must be cited. |
| E3 | \(14.5n[0.07+0.93n^{-3/4}]\) kW | Residential service cables | DIN 18015-1 service basis and Kerber coincidence relation used by pyLOVO | Pass | Keep the exact Kerber relation and citations. |
| E4 | Municipality households = published population / Bavarian mean household size | Household count before E3 | Official population and Bavarian 2024 mean of 2.0 persons per household | Pass as a regional fallback | State clearly that this is a represented household ledger, not a count of real dwellings. Integer households are allocated by residential floor area. |
| E5 | \(P_b^{pk}=\max_t\mathcal L_{g(b)}(E_b,t)\) | Non-residential service cables | BDEW G0/G1/G3 standard load profiles implemented through demandlib 0.2.2 | Pass | Keep the broad Urban4-to-BDEW class mapping visible and replace it with measured profiles when available. |
| E6 | \(q^{avg}=D/(365\,86400)\) | Water and wastewater average flow | Unit conversion | Pass | Keep. |
| E7 | \(q_W^{pk}=f_{ph}(P)q_W^{avg}\), with \(f_{ph}=2.62\) at 68,799 people | Water pipes and pumps | Ontario's published population-based peak-hour table | Pass as a published fallback | Local hourly utility data remain preferable. The selected population band and factor are recorded in the result manifest. |
| E8 | \(D_S=D_W^{retail}\) as a registered sanitary upper bound | Building water-to-sewer mapping | Explicit conservative boundary condition, not a measured return fraction | Pass as an authored upper-bound case | Do not describe it as measured sewage volume. Add dry-weather calibration and extraneous flow for detailed design. |
| E9 | \(q_S^{pk}=M(P)D_S/(365\,86400)\), \(M(P)=\max[2,1+14/(4+\sqrt P)]\) | Sewer pipes and pumps | Published Harmon relation, with \(P\) in thousands | Pass within the stated sanitary-only scope | The rerun uses 2.139 at 68,799 people and adds no rainfall infiltration. |
| E10 | \(\widetilde H=h_uA/1000\), with use-specific generic F|Heat function rows | Heat screening and customer ranking | Values copied from `building_info.xlsx` at pinned F|Heat commit `7783446...` | Pass as a documented fallback | Construction-age data are unavailable, so generic function rows are used and the limitation is stated. |
| E11 | \(H_b=x_b\widetilde H_bE_H^{off}/\sum_jx_j\widetilde H_j\) | Final heat allocation | Urban4 normalization identity | Pass | Keep; it preserves the official total but does not validate customer selection. |
| E12 | \(\dot Q=H/h^{FL}\), with 1600 h/a | Heat pipes and circulation pumps | Definition of equivalent full-load hours; 1600 h/a is the documented F|Heat fallback | Pass as a documented fallback | The rerun gives 54.688 MW. Use an operator load-duration curve when available. |
| E13 | Initial transformer count from maximum of capacity and customer-count ceilings | Electrical K-means initialization | Urban4 integer feasibility rule | Pass | Keep. Cite or list the limits used in the two denominators. |
| E14 | \(v=4Q/(\pi d^2)\) | Water and heat pipe sizing | Continuity, \(Q=vA\) | Pass | Keep. |
| E15 | \(P=\rho gQH/(\eta_p\eta_m)\) and \(P=\Delta p\dot V/\eta\) | Water, sewer, and heat pumps | Hydraulic power balance | Pass with condition | Keep the equations. Replace generic efficiencies with catalogue or measured values, or report an efficiency range. |
| E16 | \(Q=AR^{2/3}S^{1/2}/n\) | Gravity sewer capacity | Manning equation; implemented by SWMM | Pass | Keep. The code value \(n=0.013\) lies inside EPA SWMM ranges for common concrete, plastic, and vitrified-clay pipe, but material-specific values are better. |
| E17 | \(\delta_c=\sum_b\widetilde H_b/A_c\) | Heat-density screen | Definition of areal heat density | Pass | Keep. The cell width and threshold are separate choices and need sensitivity results. |
| E18 | \(L^{net}=L^{route}+L^{service}\) | Heat inventory comparison | Length-accounting definition | Pass | Keep and clearly state that this is one-way trench length. |
| E19 | \(\arg\max \widetilde H_b^\alpha/(\Delta L_b+L_b^{svc})\) | Heat customer order | Urban4 greedy heuristic motivated by shared-route/Steiner work | Pass as an authored algorithm, not as a standard | Keep only with the tested \(\alpha\) values and sensitivity results. Do not call it a Steiner-tree equation. |
| E20 | Relative route-length error \(\epsilon_L\le0.10\) | Heat candidate acceptance | Urban4 matching criterion | Conditional | A 10% band is not a physical law. Report 5/10/15% sensitivity or explain that it is a pre-set data-matching tolerance, not independent validation. |
| E21 | \(\dot m=\dot Q/[c_p(T_s-T_r)]\) | Heat pipe flow | Steady energy balance | Pass | Keep. State the water property values and temperature source. |
| E22 | \(E_{loss}=\sum_e\psi_eL_et\), with \(\psi_e\) calculated by DN and temperature | Heat-loss result | Published CPV carrier/casing dimensions and Powerpipe PUR conductivity 0.026 W/(m K) | Pass as a catalogue screening calculation | It is not a full EN 13941 buried-pipe model. A 5/10/15 °C ground-temperature sensitivity is recorded. |
| E23 | \(Q_S=Q_W^{retail}\) | Cross-sector mass interface | Same stated sanitary upper-bound condition as E8 | Pass within the stated scenario | Building IDs provide exact row matching; the value is a conservative boundary condition, not a measured ratio. |
| E24 | \(P_{WWTP}=10{,}500/24=437.5\) kW | Treatment electrical interface | Reported local site electricity use of 10,500 kWh/day | Pass for average normal-condition power | Do not interpret it as a peak rating. A measured load profile is required for peak power. |
| E25 | Fixed-point chain between sector state, power, voltage, and speed | Coupled solution | Definition of the partitioned co-simulation | Pass | Keep. |
| E26 | Energized state enables the documented speed command; disconnected state sets speed to zero | Reverse electrical-to-pump interface | Explicit binary availability contract; no invented voltage-speed curve | Pass for the stated quasi-steady interface | Manufacturer undervoltage, protection, and controller curves are still needed for equipment-specific dynamic claims. |
| E27 | \(P^{k+1}=(1-\omega)P^k+\omega\widehat P^k\) | Coupling convergence | Standard under-relaxation form; co-simulation literature supports relaxation for stability | Pass | The corrected service-resolved closure converged in two iterations and the reduced stress-test model in eight, both with \(\omega=0.60\). The unused fallback values are numerical controls, not physical parameters. |

## Consequences for the corrected model

The code, topologies, component choices, native-solver outputs, interfaces, event tests, tables, and figures were regenerated. The main changes are:

- the municipality water design factor increased from 2.0 to 2.62 and required larger distribution mains and an additional 315-kW peak-duty pump;
- the sanitary-flow upper bound increased from 3.362 to 4.100 million m³/a;
- the heat peak changed from 58.333 to 54.688 MW;
- the heat customer split, route length, pipe mix, pump ratings, and calculated loss changed;
- electrical loading changed because building demand and non-residential peaks were recalculated;
- the unsupported voltage-speed response was removed.

K-means has four distinct roles in the code. It proposes electricity groups; it conditionally divides a heat territory when several synthetic source areas are needed; it creates reporting zones that do not affect assets; and it aggregates demand nodes in the reduced stress-test model. Water and wastewater do not use K-means to build their physical networks. K-means affects sizing only indirectly when its electricity or conditional heat grouping changes the routes over which loads are accumulated.

## Primary or authoritative sources checked

- Kerber thesis: https://mediatum.ub.tum.de/998003
- pyLOVO method reference: https://doi.org/10.1016/j.segan.2024.101617
- DIN 18015-1: https://doi.org/10.31030/3143895
- DVGW W 410: https://www.dvgw-regelwerk.de/technische-regel/w-410/51a6b5
- Ontario drinking-water design guideline: https://www.ontario.ca/document/design-guidelines-drinking-water-systems/general-design-consideration-and-source-development
- Ontario sewage-flow guideline and Harmon equation: https://www.ontario.ca/book/export/html/59954
- EPA SWMM hydraulic manual: https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P10145M6.TXT
- F\|Heat documentation: https://f-heat-qgis.readthedocs.io/en/latest/api_documentation.html
- IWU/BBSR non-residential building reference method: https://www.iwu.de/forschung/energie/vergleichswerte-energieverbrauch-nichtwohngebaeude/
- Schweinfurt treatment-plant figures: https://www.schweinfurt.de/rathaus-politik/stadtentwaesserung/anlage1/zahlen--fakten/index.html
- Sicklinger et al. co-simulation method: https://doi.org/10.1002/nme.4637

## Audit status

The corrected methodology and rerun are internally consistent within the declared limits. The networks are not utility asset records; the sanitary case is an upper bound without rainfall infiltration; and the heat-loss equation is a catalogue screening calculation. Detailed design still requires measured demands, pump curves, service boundaries, protection data, and operating records.
