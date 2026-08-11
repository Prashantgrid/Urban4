# Urban4 IEEE package v2.7.0 verification

- Full native-solver test suite: **70 passed, 0 skipped, 0 failed**.
- Frozen Phase-I evidence table: **26,434 rows**, including **22,641
  service-eligible buildings**.
- Coupling contract: **22,668 rows** = 22,641 exact building mass mappings +
  27 powered-facility interfaces.
- Experiment A: 27 powered interfaces accepted in eight iterations with all
  generic sector acceptance criteria retained.
- Experiment B: all four municipality-scale native manifests report
  `status=accepted`, the legacy field `release_gate_passed=true`, and
  `thresholds_relaxed=false`.

## Municipality-scale native outputs

| Sector | Solver | Non-imposed executed result | Repairs |
|---|---|---|---:|
| Electricity | pandapower | 0.94954 p.u. minimum voltage; 82.18% maximum line loading; 0.05354 p.u. maximum LV path drop | 0 |
| Drinking water | EPANET/WNTR | 27.63 m peak minimum pressure; 69.88 m peak maximum; 1.220 m/s peak velocity | 34 |
| Wastewater | EPA SWMM | −0.044% continuity error; 0% nonconverging steps; 0% flooding | 0 |
| District heating | pandapipes | 1.49993 m/s; 99.75 Pa/m; 7.60 bar source screen; 8.08% heat loss | 31 |

Every repair uses the declared catalogue/control order and is followed by a
complete native rerun. No acceptance threshold is relaxed. JSON manifests use
relative archive paths and contain no machine-specific absolute paths.

## Executed electrical-network contingency

- Event: PL044 (PB030--PB033) opened in pandapower; selected as the largest
  source-disconnected footprint among eight PB046 source-cut candidates.
- Electrical footprint: 304/1,255 buses, 10.571/40.962 MW ordinary static
  load, 24/105 transformers, and 7/42 powered interfaces de-energized.
- Remaining component: pandapower converged at 0.92350 p.u. minimum voltage
  and 80.52% maximum line loading.
- Main waterworks: PB046 source-disconnected; native pump duty 0 kW;
  EPANET/WNTR minimum pressure -60.45 m and delivered water 0.009%.
- Coupling closure: seven under-relaxed exchanges; final exchanged duty
  0.332 kW, below the unchanged 1-kW tolerance; thresholds not relaxed.
- Scope: the other six affected interface contracts are recorded as
  electrically de-energized, but no unexecuted sector-native consequences are
  assigned to them.

The wastewater hydraulic layer contains 13 active SWMM pump objects. The
equipment inventory independently records one duty and one standby unit per
station, for 26 installed physical pump units.

## Manuscript preflight

- IEEE `IEEEtran` journal manuscript compiles without unresolved citations,
  unresolved cross-references, overfull boxes, or LaTeX errors.
- Output: 18 US-Letter pages in genuine IEEE two-column format, PDF 1.5, all
  fonts embedded.
- Figures and all 18 rendered pages were visually inspected.
- Poppler reports embedded-font type mismatch warnings originating from mixed
  embedded figure fonts; run the target journal's PDF checker before upload.
