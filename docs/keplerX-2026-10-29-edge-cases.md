# V3 edge cases and live runs, 29 October 2026

Four slow or failing behaviours from the 28 October profile were followed
to their causes and repaired. Then random real campaign curves were run
through the live batch runner. Commits: `3c0838fff`, `28adb7c9d` and
`34be1f74b`. Receipts are under `KeplerX/results/performance-engineering/20261029/`.

## Harmonic cleaning (exact, about 13,000 times faster)

Upstream `clean_tces_of_harmonics` compares every detection with every
other in two Python loops. Both loops reduce to maxima over index intervals
of the period-sorted detections. The window ends never move backwards, so
sliding maxima in `kernels.c` (`v3_harmonic_window_max`,
`v3_harmonic_dominated`) answer them in O(n log n). The interval and
monotonicity argument is in `kernels.c`. The kept sets, cuts, frames and
`drop_duplicates` are still upstream's.

- The saved 163,994-detection pass of curve 7ce647 took 941 s upstream and
  0.07 s compiled, and returned the same 7,542 rows.
- Four other saved passes (3,202–4,637 detections) matched.
- `tests/test_harmonics.py` covers:
  - adversarial frames with exact harmonics, neighbouring doubles, tied mes
    values and duplicated rows;
  - period pairs whose rounded ratio lands exactly on a window edge;
  - a dense grid.
- A further 500 fuzzed frames matched.
- Changing any window inequality from `<` to `<=` (or the reverse) makes the
  tests fail.

The compiled stage no longer calls upstream's cleaning. The exceptions are
inputs with NaN mes, other tolerances, and raising NumPy error policies;
these fall back to upstream. The hash-pinned upstream copy is unchanged.

## Why curves exploded and lost the GPU: degenerate segments

Neither the explosion nor the GPU skip was caused by noise or by formal
errors.

- **Where they come from.** Upstream starts a new segment after a gap once
  13 days have passed. On sparse curves this leaves segments of 1, 2, 4 or
  8 cadences. Upstream cannot pad these, so the adapter's repaired padding
  is used. Reflecting so few cadences gives an exactly symmetric series,
  and its wavelet details cancel.
- **What the whitening gives.** The MAD is zero or a rounding residual.
  Whitening then returns NaN, or denominators of 1e23 to 1e63 with SES up
  to 1e32.
- **Effect on the search.** A cadence with SES around 1e15 pushes nearly
  every trial period over threshold, which produced the 270,000-detection
  passes. A NaN cadence silently discards every (period, duration) row
  whose fold bin it shares with three cadences. Either kind keeps the pass
  off the GPU: NaN fails the screen's finiteness check, and a 1e30
  denominator fails its range check or makes the quantisation useless.

Both engines now drop a segment whose whitening is non-finite or has a
denominator above 1e12. The cadences are cleared in the caller's mask, so
`time[mask]` stays aligned. The threshold is safe:

- Across 13,447 pass-1 segments of 600 random calibration curves, every
  other segment stayed below 142. The degenerate ones started at 1.0e23.
- 16 of the 600 curves (2.7%) had such segments. This matches the earlier
  3% explosion rate.
- The six native goldens (18 passes) were unchanged bit for bit.

| Curve | Cadences | Before | After |
| --- | ---: | --- | --- |
| 3c1c2c28 | 11,708 | 269,842 detections in one pass (cleaning unfinished) | 6.4 s, 5 passes |
| debae9c6 | 52,393 | 273,667 detections in one pass | 3.0 s, 0 candidates |
| 7ce64796 | 17,279 | 163,994 detections in pass 3 | 5.6 s |
| 2f8f9ab5 | 17,723 | 146 s, passes 2–7 off the GPU | 12.1 s |
| d0d6ef50 | 6,562 | 51 s, 2.45 M periods resolved on the CPU | 2.7 s |
| f33b16f7 | 12,575 | 42 s, no GPU | 3.6 s |
| faf7183c | 2,094 | 44 s, no GPU | 3.4 s |
| ccd7bfb2 | 5,860 | 22 s, no GPU | 1.6 s |

Curves with genuine variability can still send 20,000–64,000 detections
into a pass (01fc4246, 794e831d). Cleaning them now takes about 10 ms.

## Preparation rejections are SAP-only sources

155,624 of the 406,547 LC2 sources (38.3%) contain no finite PSF Flat flux
on any cadence. KBonus delivered only SAP for them; this is shown by
`finite_psf_flat_kbonus = 0` in every shard's `source.parquet`. Rejecting
them is correct. The message now says so instead of "insufficient or
duplicate timestamps". 250,923 sources have PSF Flat. The eligible dwarf
inventory (Gaia Teff and logg, PSF Flat delivered) has 189,704.

## Transit refinement residuals: sigma instead of sigma squared

`FastVetter` now divides refinement residuals by sigma
(`vetting.refinement_weight = "sigma"`). `"sigma_squared"` restores
upstream, whose two branches are identical. Results on the 12 validation
signals (`sigma/sigma_vs_sigma2.log`, `sigma/chi2.log`):

| Signals | Period | Depth | Duration | Chi-square |
| --- | --- | --- | --- | --- |
| KIC 12644822, control 6815, gold planet | rel. 1e-8 to 1e-6 | 0.03–0.12% | ≤ 0.1% | lower in all three |
| nine weak short-period fixture fits | up to 3e-3 | up to 70% | up to 42% | lower in 6 of 9 |

- The weak fits move to other local optima of a multimodal fit, even though
  sigma is nearly constant on that curve.
- Fixture signal 8 no longer fails its cosine fit. The failure-retention
  test therefore uses `"sigma_squared"`.
- Not changed, but noted: upstream `sine_residual` also divides by
  `err**2`, while `box_residual` and `spsd_residual` divide by `err`. The
  sine-versus-box BIC in local morphology therefore compares
  chi-squares on different scales.

## Live runs on real campaign curves

400 random eligible dwarfs, freshly prepared from LC2, were run with
prepare, detect and vet on eight workers (`live/run1`):

- Elapsed time was 9 min 46 s. All 400 completed with no stage or metric
  errors and no GPU skips.
- 83 curves (21%) had candidates.
- Medians per case: prepare 0.19 s, detect 5.0 s, vet 0.01 s. Means: 0.24 s,
  6.3 s and 2.8 s.

**The detection phase is GPU-bound.**

- The device was busy for 381 s of the 453 s until the last detection (84%).
- Detection used 825 CPU seconds against 2,529 worker-seconds of wall time.
  Workers wait for the GPU.
- Ten workers detected no faster (455 s).
- Leaving the scheduling to Metal, with no claim, was 5% slower.

On a single worker, 40 random live curves spent 49% of detection in
whitening and 42% in the period search, which is mostly GPU screening.

**Screening.** The GPU claim now covers device work only:

- Layout and period records are built before the claim.
- A pass's cadence inputs (19 MB of quantised pairs on a long curve) are
  copied to the device once instead of once per chunk.
- The hold beyond device time fell from 7–8% to 2–3%.
- On the same 200 live curves and eight workers, the last detection
  finished at 189 s instead of 208 s (9% less).
- All 200 detection and vetting results were identical, as were the saved
  native pass-1 inputs.

**Throughput estimate.** At the measured 0.95 s per curve, the 189,704
eligible dwarfs need about 50 hours (2.1 days) of continuous operation on
this machine. All 250,923 PSF Flat sources would need 2.8 days. This is a
projection from 200–400 random curves, not a campaign measurement:

- Vetting mix, pixel localization and I/O will differ.
- The vetting tail (up to 218 s for one 10-candidate curve) only matters at
  the end of a batch.

**Vetting cost.** Multi-candidate curves are mostly eclipsing binaries
found repeatedly at one period. The transit refinement dominates their
vetting, up to 72 s for one signal: up to 9,500 leastsq evaluations,
82% of them in the limb-darkening integral. This is CPU time, and the CPU
still had headroom in the GPU-bound batch.
