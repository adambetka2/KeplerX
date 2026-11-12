# Survey system

The frozen discovery pipeline of the 406,547-source KBonus survey, completed on
10 October 2026. Protocol: `../docs/survey-protocol.md`. Results:
`../docs/survey-results.md`. Detector method: `../docs/survey-method.md`.

- `deterministic/` prepares LC2 inputs, runs the Rust detector and the shallow
  detector on one photometric view, builds the proposal queue and launches vetting.
- `shallow/` is the shallow BLS detector; its `run.py` is also the common
  library of the research detector in `../dwarfs/shallow/`.
- `vetting/` applies the shape, orbital, instrument, localization and
  contamination gates and writes HELD, DISMISSED or BOI dispositions.
- `photometry/` reads LC2 shards, rebuilds the corpus from MAST and builds the
  M- and K-dwarf selections.
- `review/` is the offline BOI review application.
- `catalogue/` adds derived annotations after a new vetting run completes.
- `run.py` launches a stage with the shared modules on the path. `runtime.py`
  binds every run to one implementation hash and the CPU affinity.
  `evaluation_scope.py` binds scoped evaluations to frozen inputs.
