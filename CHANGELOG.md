# Changelog

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
