# Fast TRICERATOPS

CPU acceleration of pinned TRICERATOPS `d20875c` and of `TRICERATOPS_KEPLER_V3`
with the same `calc_probs` outputs (lnZ, probabilities, FPP, NFPP). Float64 only:
Apple GPUs have no float64 and the tested float32 likelihood biases lnZ.
- `accelerated()`: stock route. `kernels.py` takes every model flux from
  pytransit's own compiled kernel, vetoes binaries before their primary, and
  fuses dilution and cost; `scenarios.py` replaces the per-draw LDC loops.
- `accelerated_v3()`: V3 route (patched runtime). `joint.py` fuses the quarter
  likelihood and skips draws whose weight is provably exactly zero, resolved in
  rebuilt scenario functions; bitwise equal to upstream. `certified=True` also
  skips weights below e^-50 of the best (lnZ change below one ulp), bounded
  through pytransit's measured accuracy instead of its own values.
- `engine()`: static copied V3 scenarios with exact kernels and shared splines.
- `workflow`: frozen-source experiment runner with preserved retries and receipts.
- `native`: separately qualified batman orbit experiment; no accepted fit rerun.
Method, controls and measurements live in `docs/fast-triceratops*.md`.
Queued campaigns use the exact engine. Run checks through `uv run --python PYTHON`:
stock tests use `test_equivalence.py`; V3 controls use `test_joint.py`,
`test_engine.py`, `test_splines.py`, `test_native.py` and `test_workflow.py`.
