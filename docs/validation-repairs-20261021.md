# Validation repairs, October 21

Install repairs in a new runtime and use separate candidate generations. Preserve
historical software, evidence and outcomes. TRICERATOPS remains unchanged at
`d20875c27d6761349cf032728bb4a59ec732a84d`.

## Dependencies

The juliet patch fixes beta support, rejects a zero-prior MCMC proposal before
evaluating its likelihood, and forwards the declared dynesty Generator to
equal-weight resampling. The batman patch solves the same elliptic Kepler
equation with safeguarded Newton/bisection and the analytic bracket `[M-e,M+e]`.
It does not clip eccentricity or change its distribution. Invalid eccentricity
fails before parallel evaluation. Local versions are `2.2.10+spacelab.1` for
juliet and `2.5.3+spacelab.2` for batman-package. The first diagnostic build kept
the bracket too conservatively; its runtime and interrupted short control are
preserved. The reviewed solver accepts contracting Newton steps and terminates
when a step rounds to the current root.

Acquire these exact inputs into `SOURCES`, retaining transfer receipts:

- [batman v2.5.3 source](https://codeload.github.com/lkreidberg/batman/tar.gz/6fc570e209fa2e3dadd2ffa4c535f528a7bef495), named `batman-source.tar.gz`.
- [juliet 2.2.10 wheel](https://files.pythonhosted.org/packages/3c/e6/9936e81ef5fa1ac27c8bbb26baab605dc945dd1e220b6ff0d19facf84044/juliet-2.2.10-py3-none-any.whl).

The preparation command checks release SHA256 values before applying tracked
patches. Build batman on each execution host; a Mac wheel cannot run on Linux.
Use new `RUNTIME` and `BUILD` directories:

```bash
uv venv --python 3.12.13 "$RUNTIME"
uv pip sync --python "$RUNTIME/bin/python" post_boi/requirements.txt
uv pip install --python "$RUNTIME/bin/python" wheel==0.48.0
uv run --python "$RUNTIME/bin/python" python -m post_boi.dependencies "$SOURCES" "$BUILD"
uv build --wheel --no-build-isolation --python "$RUNTIME/bin/python" --out-dir "$BUILD/wheels" "$BUILD/batman"
uv run --python "$RUNTIME/bin/python" python -m wheel pack "$BUILD/juliet" -d "$BUILD/wheels"
uv pip install --no-deps --python "$RUNTIME/bin/python" "$BUILD"/wheels/*.whl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=10 uv run --python "$RUNTIME/bin/python" python -m pytest post_boi -q
uv pip check --python "$RUNTIME/bin/python"
```

Preserve build logs, `BUILD/sources.json`, wheel hashes and environment. Controls
include independent SciPy densities/orbits, rejected-prior behavior, real juliet
fits and frozen official Kepler-10 probability parity. Candidate-specific model
replay and posterior review still precede FPP.

## Preparation

Do not require every posterior contact to fit the original detector-clock mask.
Count actual observed cadences and measure signal transfer through detrending.
`post_boi.protection` removes specified valid joint posterior transit models from
the raw observations, refits detrending and reports the signal change and its
scale relative to formal uncertainty. This is a model-dependent diagnostic;
correlated noise and posterior support still need review. No universal ppm veto
or automatic admission is installed.

For a material effect, preserve the first fit and create a separate preparation
with an explicitly justified `protection_mask` (`path` and `basis`). Its NPZ
contains `source_id`, full original `time_bkjd` and boolean `protected`. It adds
protection without removing observations. Check baseline support, refit and
compare before accepting the clock. Do not choose a mask or fit by FPP.

## Apertures and escalation

The old custom wings/pointing prescription is not established calibration merely
because it ran. The stock Gaussian and calibrated Kepler core disagree for
crowded own-SAP apertures. Published KBonus FFI models are an existing calibration
source to investigate, with finite-support and archive-version limits retained.
Core-only comparisons are not complete measured fractions. Delivered CROWDSAP
does not supply each contaminant's contribution.

`post_boi.aperture.physical_aperture` resolves array-to-detector mapping by
replaying the original accepted SAP cadences from authenticated parent pixels.
It checks the full saved mask count and the extraction's background conventions
within a float32 arithmetic bound. KIC 4948140's Q4/8/12/16 source headers name
an origin different from the parent data; pixel replay binds the saved array to
the parent frame. Preserve the conflicting headers. Gaia positions use the WCS
frame's own origin. This repair changes geometry, not photons or dilution.
All 34 quarters of the two reopened cases pass the full-data replay; the small
committed fixture checks Q1/4/12 and rejects corrupted source flux.

Published KBonus FFI models and unchanged lkprf 1.1.1 have now been tested.
The FFI archive lacks Q1/Q17. Engineering Kepler calibration arrays contain
signed negative values before interpolation; unchanged lkprf can return negative
contaminant fractions. Neither a negative fraction nor finite support is a
physical measurement of absent contamination. Do not silently clip fractions,
invent wings or adopt a response by its FPP.

The authorized nonnegative calibration and linear interpolation comparison is
complete. The local lkprf version is `1.1.1+spacelab.1`; the tracked patch only
projects negative Kepler calibration samples to zero, renormalizes each plane,
and changes spline degree to one. Original calibration files, spatial weights,
finite footprint, pixel threshold and Tmag weights remain fixed. No stellar
fraction is clipped after evaluation. Kepler-10's probability/evidence output
matches a direct unchanged TRICERATOPS call exactly. This establishes numerical
behavior, not a new empirical calibration of crowded fields.

The largest relative target-dilution changes are 0.0188% (KIC 7595917), 0.0872%
(KIC 4948140) and 0.0000103% (Kepler-10). Full-quarter pooling nevertheless
distorts capable-neighbor shapes because aperture response varies. The largest
identical channel/aperture group for both BOIs is Q3/7/11/15, selected by photon
count before FPP. Its observed-shape response diagnostics are small for target
and neighbor. Refit this declared selection, preserve all other quarters for
independent checks, and use `calibrated_kepler` for modeled response fractions.
Do not call these direct stellar-flux measurements. See the
[repair status](../results/boi-validation/validation-repairs-20261021.json).

If an unclear software/instrument issue prevents valid inference, stop affected
work and ask Adam. Do not park more candidates for the same unresolved method
problem. The server owner must hold for the patch and run Linux controls before
resuming. This work runs numerically on the Mac only.
