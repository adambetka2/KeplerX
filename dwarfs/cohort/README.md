# Cohort builders

Programs that froze the programme's injection and control cohorts and build its
target lists. They run on the Mac against the mounted corpus (`discSSD`,
`discPCIe`); finished cohorts were transferred to the server with hash receipts.

- `calibrate.py freeze|prepare|run|report|year-scramble|remodel` freezes a
  physical-injection experiment from a host population under
  `calibration-plan.json`, prepares detector-only inputs in native M/K-dwarf
  curves, and replays a detector module on them. `calibration-config.json` is
  the development detector configuration for `run --config`.
- `evolve_controls.py --stage final-synthetic-plan|final-synthetic6000|final-controls3000|remodel-synthetic`
  builds the parent-disjoint 10,000-case gate cohort: 6,000 synthetic planets,
  1,000 real planets and 3,000 controls (catalogue false positives, native
  eclipsing binaries, instrumental artifacts, originals, inverted and
  year-scrambled curves).
- `mk_targets.py` writes the M/K survey target list
  `../../results/mk-survey-targets-20261014.csv`: every M and K host with its
  shard, G, quarter count, the white-noise SNR of a 1,000 ppm transit and the
  hopeless flag, from the census and SAP noise tables in
  `dump/mk-cohort-census-20261013/` (uv with pyarrow and pandas).
- `SPACELAB_CALIBRATION_ROOT` names the repository root; the frozen study
  folder with `freeze.json`, `plan.json`, `trials.csv` and `checks/` is
  `dump/shallow-calibration-20261010`.

Every output binds the generator, preparation, transit machinery and reader
hashes. A changed program refuses to extend a frozen experiment.
