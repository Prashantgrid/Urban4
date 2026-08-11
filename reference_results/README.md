# Reference results

This directory contains compact records from the Urban4 v2.7.0 publication
run. They allow reported claims to be checked without storing the complete
553 MB generated-output tree in routine Git history.

Included records cover:

- benchmark and morphology metrics;
- common-ledger and clustering checks;
- accepted bidirectional native models and interface states;
- municipality-scale native manifests and repair logs;
- the hydraulic-originating response;
- the executed PL044 electrical-network contingency.

The municipality-scale model directory retains the internal name
`municipal_scale_v2.5.0` because that accepted numerical state was generated in
v2.5.0 and carried unchanged into the overall v2.7.0 publication release.
Version 2.7.0 adds the executed PL044 contingency and its explicit scope.

The full generated tree can be recreated with:

```bash
python run_all.py --skip-paper
```

The complete frozen archive will also accompany the tagged publication release
and archival DOI. Reference files preserve the original schema keys, including
older fields named `release_gate_passed`; in the manuscript and current
documentation these are described as engineering acceptance criteria.

