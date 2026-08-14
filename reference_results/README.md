# Reference results

This directory contains compact records from the Urban4 v2.8.0 verification
run. They allow reported claims to be checked without storing the complete
553 MB generated-output tree in routine Git history.

Included records cover:

- benchmark and morphology metrics;
- common-ledger and clustering checks;
- accepted bidirectional native models and interface states;
- municipality-scale native manifests and repair logs;
- the hydraulic-originating response;
- the executed main-waterworks feeder contingency;
- the fixed-input Topotherm routing comparison.

The municipality-scale model directory retains the internal name
`municipal_scale_v2.5.0` because that accepted numerical state was generated in
v2.5.0 because that path is part of the output schema. Version 2.8.0 updates
the equations, demand settings, catalogue choices, and rerun records while
leaving that internal directory name unchanged.

The full generated tree can be recreated with:

```bash
python run_all.py
```

The complete frozen archive can be attached to a tagged archival release. Reference files preserve the original schema keys, including
older fields named `release_gate_passed`; current documentation describes these
as engineering acceptance criteria.
