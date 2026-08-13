# Changelog

## v2.8.0, 13 August 2026

- Replaced unsupported demand fallbacks with declared, cited demand models.
- Allocated the reported population to exactly 34,400 represented households.
- Applied the 2.62 drinking-water peak factor and provided four source-pump
  units, including standby.
- Applied the Harmon wastewater peak factor and kept gravity-sewer and
  pressure-main lengths separate.
- Used 1,600 district-heating full-load hours and added bounded catalogue-size
  repair for heat pipes that exceed the velocity limit.
- Corrected the leak-controller setting and regenerated its result and figure:
  pump flow rises 18.1%, pump power rises 16.0%, and the first pressure drop is
  11.7 m.
- Added the equation audit, aggregate Topotherm routing comparison, and
  verification summary used for the submission.
- Kept the retained compressed evidence snapshots usable in a clean checkout
  and made direct script entry points import the local package consistently.
- Made the repair-audit test validate the allowed applied repair sequence
  without requiring an unnecessary water-pump adjustment.
- Verified the updated source with 70 passing tests.

## v2.7.0, 11 August 2026

- Prepared the publication repository from the complete v2.7.0 study archive.
- Included the common-evidence generator, four sector modules, interface
  registry, coupled benchmark, and PL044 pandapower-WNTR contingency.
- Included five retained OpenStreetMap snapshots and the Schweinfurt case
  configuration.
- Included the 70-test verification suite and compact accepted-model records.
- Separated regenerable heavy outputs from the normal Git history.
- Fixed the clean-build workflow so the compact source and asset inventory is
  regenerated from current interface states instead of assumed prebuilt files.
- Added the heat-selection threshold and policy sweep to the ordered workflow,
  removing a second implicit dependency on prebuilt figure inputs.
- Ordered the original water/wastewater corridor audit before the alternative
  construction so its comparison baseline is always present in a clean build.
