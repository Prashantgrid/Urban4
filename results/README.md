# Retained final manuscript results

These CSV/JSON files are the frozen heat-power and water-power result records
from `Urban4_IEEE_submission.zip` (2 November 2026). `manifest.json` records
all original numerical files, their sizes, and SHA-256 hashes. The core source
used by the supplied study scripts is commit
`4943579c03566f543267e0889de53e314f75d7ea`.

- `heat_power_case/`: reference transformer states, temperature sweeps,
  heat-pump/renovation variants, hydraulic capability, and both transformer
  sensitivities.
- `water_power_case/`: line contingencies, facility attachments, open-ring
  restoration, storage-dependent annual losses, blackout EPANET time series,
  flood exposure, and extreme-event losses.

The internal key `PE` or `pop_equiv` denotes demand-weighted people without
water; it is not the wastewater population-equivalent design unit.

Recalculate the sensitivities and selected figures in a separate directory:

```bash
python code/reproduce_study.py
```

For a complete new case-study run, rebuild the native models and then run
`python code/reproduce_study.py --from-models`. Generated outputs stay in
`outputs/manuscript_study/`; these retained reference files remain unchanged.
The older compact core-verification records stay in `reference_results/`.
