# Final study reproduction

The final source update integrates the corrected v2.9 core on the documented
manuscript branch with the 2 November 2026 case-study scripts and retained
result tables. The repository overview uses the revised layered-city graphic
in `docs/figures/urban4_methodology.png`. It is an AI-assisted illustration of
the method, rather than a generated case-network map.

## Reproduction levels

1. `python code/reproduce_study.py`: reconstructs both transformer sensitivity
   tables and Figures 3, 6, and 7 from retained numerical and map evidence.
2. `python run_all.py`: rebuilds native sector models and runs the core
   verification suite. The audit writer called by this workflow is restored.
3. `python code/reproduce_study.py --from-models`: recomputes all heat-power
   and water-power case studies and Figures 3 through 9 from those native
   model exports.

The first level writes separate output copies, preserving all retained
reference records. It does not rerun native-model simulation. Levels 2 and 3
require the full solver environment and can take considerably longer.

## Validation scope for this upload

The upload checks target the new retained-results workflow and the core
reference-grounding, review-correction, and integration tests that run without
generated outputs. The new study regression reconstructs all entries of both
transformer sensitivity tables, checks the below-50-person-hours annual-loss
claim for both repair-time assumptions, and checks the reported critical MV
branch and all-station backup results against the retained event records.

The historical full native-model verification is documented separately in
`VERIFICATION_v2.9.0.md`. Its 83-test result is a retained verification record;
it is not presented as a new full model run for this upload. Tests that read
generated native outputs require level 2 first.
