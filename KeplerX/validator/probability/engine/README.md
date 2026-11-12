# TRICERATOPS engine

Adapted from TRICERATOPS `d20875c` with the reviewed Kepler V3 patch.
Installed priors, flux tables and physical constants remain pinned dependencies.
`triceratops.py` keeps the upstream scenario order and output normalization.
`marginal_likelihoods.py` integrates the exact likelihood and evidence helpers.
`splines.py` reproduces the qualified Mac FITPACK arithmetic in float64.
Radius and temperature share a basis; deterministic flux ratios are reused.
Sampling order, attempted-draw denominators and retained outputs are preserved.
`require_unmodified()` rejects changed dependencies and unqualified FITPACK builds.
Use `fast_triceratops.engine()` inside a separate performance experiment.
Queued campaigns use this copy through the frozen workflow source.
Controls and timings are documented in `docs/fast-triceratops.md`.
The full benchmark receipts live in `results/fast-triceratops/`.
