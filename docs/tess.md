# TESS replication of BOIs

`tess/` measures supplied signals after vetting and before `post_boi/`.
It reads the completed survey or M/K handoff without changing a disposition.
There is no blind period search, detection threshold, stellar prior, depth cut,
FPP calculation or calibration claim.

## Run

Create a separate Python 3.12 runtime on external storage with
`uv venv --python 3.12 RUNTIME` and
`uv pip install --python RUNTIME/bin/python -r tess/requirements.txt`.
Run from the repository root:

```bash
uv run --python RUNTIME/bin/python -m tess export-mk VETTING_RUN INPUT.json
uv run --python RUNTIME/bin/python -m tess coverage INPUT.json COVERAGE
uv run --python RUNTIME/bin/python -m tess run INPUT.json OUTPUT --sectors 14
```

Omit `--sectors` to use every returned sector; `--source-ids` selects Gaia IDs.
`--kernel-cache PATH` shares the spacecraft kernels between runs; otherwise
they live in `OUTPUT/../tess-kernels`. `python -m tess kernels PATH` prepares
the cache separately. The standard library downloads about 1 GiB once.
Its manifest is frozen and verified on reuse; changed kernels require a new cache.
Use a new output directory for changed inputs, settings, code or runtime.
Unchanged completed sectors resume after file verification. A single writer
holds the output lock. Query/download/processing failures retain tracebacks,
produce `PARTIAL` and a nonzero exit; they are not zero-coverage observations.
The default free-space reserve is 20 GiB. Run serially; downloads are shared
by all signals on a host. Raw FITS stay in the output directory.

## Inputs and clock

`export-mk` verifies completed batch bindings through the existing catalogue
reader, selects every BOI, and preserves run-scoped signal IDs, event IDs,
discovery views and source-evidence hashes. It uses each ledger's period,
epoch and duration, never a rank-one signal or a parent TPF KIC as the source.
BKJD is converted to BJD_TDB by adding 2454833.0. Gaia DR3 positions and proper
motions, when present, determine the aperture position at each sector's midpoint.

The JSON contains `hosts`, each with `source_id` (string), `ra_deg`, `dec_deg`,
optional `astrometry` and `signals`. Each signal has `signal_id`, `clock` and
optional `clock_trials`. Every clock supplies `period_days`, `epoch_bjd_tdb`,
`duration_days` and optional `covariance_t0_period_days2` in (epoch, period) order.
All explicit trials are reported; no best trial is adopted automatically.
The covariance propagates as `[1,n] C [1,n]^T` to each predicted event.
Missing covariance remains null. Fixed-clock non-recovery cannot exclude a
signal with an uncertain extrapolation, alias, period evolution or TTVs.

## Pixels and measurements

[MAST TESScut](https://mast.stsci.edu/tesscut/docs/getting_started.html) supplies
SPOC FFI cutouts at the supplied coordinates. Coverage means archive footprint
coverage, not usable transit coverage. FITS times must be barycentric TDB
mid-exposure times. TESScut's `TIMECORR` is for the FFI reference position.
The module removes it and applies the target-specific correction from
[lkspacecraft](https://lightkurve.github.io/lkspacecraft/tutorials/calculating-barycentric-time-correction.html)
with the native spacecraft TDB time and archived SPICE kernels. Both time arrays
and both corrections remain in the output. `TIMEDEL` supplies the coadd integration span; the separate
live `EXPOSURE` and background-subtraction header are retained.

`tess/settings.json` records the starting extraction: an 11-pixel square,
1.5-pixel source radius and 3–5-pixel background annulus. Pixel centres determine
binary masks. The annular mean is subtracted with full propagation of the
reported independent pixel variances. These are explicit, adjustable extraction
choices, not optimized apertures or source deblending. In crowded fields, stars
in either mask can dilute, amplify or invert a measured signal. The masks,
native fluxes, pixel errors, quality flags and WCS remain available for review.

The quality mask 17087 is the documented
[Lightkurve TESS default](https://lightkurve.github.io/lightkurve/_modules/lightkurve/utils.html#TessQualityFlags).
No flux sigma clipping is applied. Nonfinite data and nonpositive uncertainties
cannot enter the fit. All retained and rejected rows remain in the native file.

For each expected primary and phase-0.5 secondary, weighted least squares fits
`a + b*(t-tc)/duration - A*h`, where `h` is the exact overlap of a box transit
with each coadd. The baseline is fitted simultaneously with the transit.
The starting half-window is three durations, limited by the midpoint to the
adjacent primary/secondary so windows do not share cadences. Baseline counts
on both sides and observed in-transit exposure are reported. A missing or
singular fit is explicit. Other supplied signals are masked from the baseline;
overlapping transits remain measured and are counted as ambiguous overlaps.

Depth is signed `A/a`; its formal error includes intercept/depth covariance.
A nonpositive baseline retains the electron-rate amplitude but no fractional
depth. Samples use `100*(F/F_out-1)` with the fitted local baseline. Inverse
formal-variance weights combine events, with separate formal, residual-scaled
and event-cluster scatter errors. The latter groups the same cycle across
sectors/cameras and is undefined for one transit. Odd/even and sector values
are retained. Formal S/N is not a red-noise false-alarm probability. Shared
systematics and small event counts limit all these uncertainty estimates.
The secondary measurement tests phase 0.5 only, not all eccentric phases.

The same linear depth estimator produces a signed difference image and formal
pixel errors. Its signed first moment inside the fixed aperture is reported
with a covariance and offset in pixels; it is not a PRF localization or proof
of Gaia host identity. Inspect the full image for contaminants outside the
aperture. No Gaia completeness, dilution correction or common host is inferred.

## Outputs and verification

JSON retains each event, sector, clock and combined measurement. NPZ files
retain native photometry, masks, median/difference images and errors. The
`samples` columns are native row index, time from transit in days, relative
flux percent, formal error percent and exposure overlap fraction.

The frozen HAT-P-7 b sector-14 control uses the published Kepler clock and
native TESS aperture measurements. Paired 40, 100 and 1000 ppm increments
check the shallow-signal response on the same real noise. They establish
numerical response, not population completeness or a TESS sensitivity floor.
Full MAST-to-output checks and the M/K handoff receipt are recorded in
`results/tess-module-20261020.json`.
