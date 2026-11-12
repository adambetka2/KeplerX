# Research detector lineage

M/K programme engines extend `../../pipeline/shallow/run.py` and its BLS kernel.
`current-designs.json` keeps the adopted SAP design: knot spacing 0.75 d,
the 0.02 d duration dropped. Source digests and numerical runtimes bind designs.

- `evolution.py`: blind rounds, snapshots, scoring, export and component resolution;
  `scan_records` supports full arrays or compact proposal evidence.
- `evolve_detector.py`, `evolve_noise.py`: duration grids and blind local noise.
- `evolve_template.py`: physical profiles, fixed spline nuisance and consistency gates.
- `evolve_native_coherence.py`, `evolve_global_fundamental.py`: event ranking
  and final likelihood conditional on one coherent harmonic background.
- `evolve_psf_flat.py`: delivered Flat pilot with constant offsets; the adopted
  SAP design stays separate. Runbook: `../../docs/psf-flat-retrieval.md`.
- `masks.py`, `null_calibration.py`: cadence masks and time-slide noise checks.
- `batch_bls.c`, `coherence.c`: kernels compiled under `dump/shallow-bls/`.
- `test_*.py`: masks, event statistics, likelihoods and evidence contracts.

Python 3.11 (`batman` requires distutils). Run with `uv`, `../requirements.txt`
and pytest: `python -m pytest dwarfs/shallow`.
