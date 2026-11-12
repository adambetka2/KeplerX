# KeplerX Detector: speed on survey curves

2 November 2026. Changes: `KeplerX/detector/compiled_search.py`
(FFT whitening bank, Vetter FFT path), `candidate_ephemerides.py` (alias
resolution), `gap_padding.py` (long-gap segmentation, noise-window option),
`kernel_bindings.py` and `search_kernels.c` (`v3_mad_pairs`),
`flattened_templates.py` (window cache), `upstream_adapter.py`,
`vetter/program.py`, `config.json` (detection: `whitening`, `segmentation`,
`resolve_aliases`), tests. Evidence
uses the out-of-sample survey benchmark (`KeplerX/data/survey-benchmark`,
built separately: real LC2 curves of development families that
the calibration never drew, 31 strata of star class and G magnitude with
weights, and 493 confirmed-planet hosts).

## Why

Adam asked (2 November) for a further 50% off the Detector's time on a
diverse set of KBonus curves, and for evidence that the day's Detector
changes generalize beyond the calibration sample.

## Where the time went

A cProfile of the production Detector on 12 random development curves:
the compiled whitening bank was 40% of Detector wall time and the per-level
FFT path (segments that skip the bank) another 17%; waiting on the GPU
screen 19%. Whitening is 71% of Detector CPU.

- The bank computes each template's convolutions directly. At coarse
  wavelet levels a template's support spans the whole segment, so each
  level costs O(n²); the bank's direct sums took 0.064 s of a 2,048-point
  segment's 0.070 s, the running medians the rest.
- Upstream starts a new segment at a gap only once the running segment is
  longer than `sector_size` (13 days). A short stretch therefore merges
  across any later gap, however long, and the padding fills the gap with
  reflected flux: one development curve whitened 1,741 cadences spanning
  682 days as a 65,536-point segment (64% of the 12 curves' whitening by an
  n² cost). On 241 survey curves (`diverse-quick`) the largest padded
  segment is 65,536 points for 10 curves, 16,384–32,768 for 49.

## Changes

- `detection.whitening = "fft"` (`FastSearch._fft_bank`): the same MAD
  weights, but each duration's level convolutions are multiplied in the
  frequency domain, summed over levels and inverted once. Weights and their
  transforms are computed once per (level, window) pair; template spectra
  are cached per segment length. SES equals the exact path's to 5e-14;
  candidates agreed on all 12 development curves and every spot check.
  `"exact"` keeps upstream's arithmetic bit for bit (the native engine and
  other stages always whiten exactly).
- `detection.segmentation = "gaps"`: upstream's rule, and every gap longer
  than 30 days (`gap_padding.LONG_GAP_DAYS`) also ends a segment, after its
  last cadence. On the 241 survey curves it changes 27% of curves (a
  13-day limit would change 59% for the same saving) and takes the largest
  padded segment to 4,096 points.
- Flattened template windows are fitted once per width per process (the
  least-squares fit was repeated for every segment length; bitwise equal).

## Results

Production against `whitening "fft"` with `segmentation "gaps"`, both
legs in the same process per curve (other batches shared the machine, so
wall times are inflated and only CPU is clean):

| Set | Detector CPU per curve | Change | GPU-s per curve | Candidates |
| --- | ---: | ---: | ---: | ---: |
| 241 survey curves (`diverse-quick`), weighted by stratum | 3.30 → 1.41 s | −57% | 1.05 → 1.04 | 402 → 337 |
| same, unweighted | 4.25 → 1.63 s | −62% | | |
| 493 confirmed-planet hosts (`known-planets`) | 7.52 → 2.78 s | −63% | 1.91 → 1.90 | 1,334 → 1,320 |

- Recall on the 620 searchable confirmed planets: 589 exact, 3 harmonic,
  28 missed with both settings; no planet changed outcome.
- Synthetic truth (`eval_detect.py`, 348 paired development-quick cases
  and the 114 long-transit cases): detectable planets 116/148 exact against
  114; null curves with a candidate 30.0% against 33.3%, candidates per
  null 0.99 against 1.17; long-transit signals 105/126 against 104. Eight
  signals changed: six gained (a known 54-day planet among them), two lost
  (an 84-day planet now reported at a harmonic and a contaminating binary).
- Every candidate difference traces to the segmentation: with FFT
  whitening alone the candidates are identical. On the survey curves the
  split curves lose most of their candidates (146 → 84 on 54 curves), many
  from curves whose reflected-gap padding produced 8–22 detections.

## Batches on the survey benchmark

Production batches ran on the benchmark (8 workers, scored
with `calibration.benchmark_scoring`): this morning's Detector (`f7f201a7e`,
`results/survey-benchmark/2026-11-02-diverse-quick-and-known-planets`, on a
busy machine) and FFT whitening with gap segmentation (`9f125d796`,
`results/survey-benchmark/2026-11-02-diverse-full-current-production`, 896
curves in 11.7 minutes):

| Per survey curve, stratum-weighted | Morning | FFT + gaps | Change |
| --- | ---: | ---: | ---: |
| Detection CPU | 3.20 s | 1.33 s | −58% |
| Detection wall (8 workers) | 7.06 s | 3.42 s | −52% |
| Vetting CPU | 2.00 s | 1.72 s | −14% (Vetter FFT whitening) |
| Curves with a candidate | 33.9% | 30.3% | |
| Candidates per curve | 1.67 | 1.37 | |

Confirmed planets found exactly stayed at 589 of 631 (vetting keeps 98.0%),
and 272 on single-quarter curves. The survey of 250,923 curves projects to
about 2.0 days on 8 workers instead of 3.6.

A paired, quiet-machine timing (12:47–12:55 UTC, nothing else running):
the 241 `diverse-quick` curves on 8 workers with the same code and vetting,
only the detection settings differing (this morning's: exact whitening,
upstream segmentation, no alias resolution; final: FFT whitening, gap
segmentation, alias resolution),
`results/performance-engineering/2026-11-02-detector-speed-on-survey-curves/timing-{morning,final}`:

| Per survey curve | Morning, weighted / plain | Final, weighted / plain | Change (weighted / plain) |
| --- | ---: | ---: | ---: |
| Detection CPU | 2.88 / 3.81 s | 1.38 / 1.63 s | −52% / −57% |
| Detection wall (8 workers) | 4.92 / 6.17 s | 3.51 / 3.84 s | −29% / −38% |
| Vetting CPU | 1.80 / 2.56 s | 1.48 / 2.12 s | −18% / −17% |
| Candidates per curve | 1.67 | 1.34 | |
| Whole batch | 287 s | 200 s | −30% |

The Detector's CPU is halved; its wall time in a batch falls by a third,
because workers now wait on the shared GPU screen (wall 3.5 s against 1.4 s
of CPU per curve). The busy-machine comparison above overstates the wall
gain.

## What bounds a batch now

The GPU screen is unchanged at about 1 GPU-s per survey curve (1.9 for a
full Kepler target) and the GPU is shared by all workers, so an 8-worker
batch is now bounded by the GPU rather than by whitening. A CPU-only screen
costs 25 times the CPU of the GPU path (230 against 8.7 CPU-s on 8 survey
curves), so falling back to the CPU when the GPU is busy does not help.

`detection.noise_window_days` (default null, upstream's windows) sets one
duration whose 7 x duration x 2^level windows estimate every template's
noise, as Kepler's TPS estimates noise per wavelet scale rather than per
pulse; durations then share their weights and transforms. Untested for
recall; not in production.

## A coarser period grid: measured, rejected out of sample

With CPU cheap, the GPU screen is the batch's limit: on the survey curves the
68% (weighted) with no candidate take 27% of GPU time (0.35 GPU-s each),
while a detection costs a further full pass (1.3–2.6 GPU-s per curve with
candidates). Each pass screens every trial period.

`detection.period_step_factor = 3` with `detection.screen_threshold = 6.3`
screens a third fewer trial periods. Alone it flips strong signals to
harmonics: the coarser grid lowers the fundamental's screened MES below an
alias's, and the cleaning, ordered by screened MES, keeps the alias (a
65.9-day planet reported at 131.9 days, a 5.82-day one at 2.91).
`detection.resolve_aliases` (`candidate_ephemerides.strongest_alias`)
refines each accepted detection's P/2, P/3, 2P and 3P and takes the alias
only when its refined MES, and its MES without its strongest event, are at
least 5% higher. The first condition keeps P when the extra events of P/k
fall in gaps; the second was added after a 100.45-day multiple on
Kepler-484's curve landed one SES-420 event and displaced the 10.05-day
planet (MES 89).

On the synthetic sets step 3 with alias resolution looked better than
production: 119 against 116 of 148 paired planets, 108 against 105 of 126
long-transit signals, null curves with a candidate 28.7% against 30.0%,
188 against 185 of 193 on the unbiased set, and 34% less GPU time; 32% less
GPU on the survey curves with 17% fewer candidates. On the 620 searchable
confirmed planets it found 586 against 589 (with the guarded alias
resolution): two harmonics became exact, five planets were missed. Two of
them have transits much shorter than their hosts' density predicts (1.6
hours at 9.8 and 31.8 days), where the coarser grid's phase drift costs most,
and two sit near the threshold (MES 8.4, and a 38-ppm planet). The synthetic
planets did not show this, so step 3 is not adopted. A middle step of 2.5
with screen 6.45 (with alias resolution) saves 22% of GPU time and costs
about one planet everywhere: 590 against 591 confirmed planets, 104
against 106 long-transit signals and a known 54-day planet on the synthetic
set, 187 against 188 of 193 on the unbiased set; also not adopted. Step 4 (screen 6.0) lost planets at
S/N 9–14 even on the synthetic sets, and one shared noise window of 0.25 days
for every template (`detection.noise_window_days`) cut whitening CPU by a
third but found 114 of 148 with more false alarms.

### Alias resolution on the production grid (adopted)

`detection.resolve_aliases` with the production grid (step 2, screen 6.6),
production's legs reused from the same day's runs:

| Set | Production | With alias resolution |
| --- | ---: | ---: |
| 620 searchable confirmed planets: exact / harmonic / missed | 589 / 3 / 28 | 591 / 1 / 28 |
| 424 synthetic cases: detectable planets exact (paired 148) | 116 | 116 |
| long-transit signals exact (126) | 105 | 106 |
| null curves with a candidate / candidates per null | 30.0% / 0.99 | 30.0% / 0.97 |
| 241 survey curves: with a candidate (weighted) / candidates | 31.7% / 337 | 31.7% / 324 |
| survey curves' Detector CPU | 315 s | 346 s |

Kepler-354 d and Kepler-1967 b, reported at a harmonic before, are found at
their own periods; no confirmed planet is lost. On the unbiased 300 cases it
finds 188 of 193 against 185. On the synthetic cases a
46-day planet beside a 27-day one and a planet at S/N 7.4 are gained; a
1.34-day planet on a curve dominated by 0.65-day stellar variability is
lost (switches to the variability's aliases turn later detections into
repeats and the search stops before reaching it), and a contaminating
binary resolves to half its period. On the survey curves 230 of 241 keep
the same candidates. The switches' MES ratios (1.17–2.0 when right, 1.27–1.41
when wrong) do not separate the two, so the margin stays at 5%.

## The final Detector on the development-quick manifest

`cfe2a4e3c` (flattened templates, FFT whitening, gap segmentation, alias
resolution), `KeplerX/results/calibration-previews/2026-11-02-development-quick-run-final-detector`
(batch shared with other runs, so its wall time is not evidence):

| | 1 November | 2 November, flattened templates | Final |
| --- | ---: | ---: | ---: |
| 50% / 90% efficiency at information S/N (targets 8.5 / 12) | 8.76 / 12.37 | 7.66 / 10.41 | 7.57 / 10.35 |
| Detectable planets exact (plain / weighted) | 82.5 / 81.5% | 88.6 / 89.5% | 89.3 / 89.9% |
| Occurrence-integrated completeness | 81.0% | 88.4% | 88.8% |
| Known planets in range | 88.9% | 93.3% | 94.4% |
| Null originals with a candidate / candidates per case | 21.0% / 0.72 | 26.9% / 0.97 | 25.8% / 0.91 |
| Detection CPU per case | 4.33 s | 4.33 s | 2.00 s |

The held-out check of the linear-score vetting policy on
this run (constants fitted before the FFT and gap changes): 96.0% of planets
kept (known 97.7%, validated all), 9.3% of unexplained candidates kept, null
originals with a false alarm 118 → 30, and efficiency after the policy at
information S/N 7.85 / 10.95: both targets met.

## Generalization of the day's Detector changes

Each change rests on a property of transit search, not on the calibration
sample, and the out-of-sample benchmark checks it on curves the calibration
never drew:

| Change | Why it holds beyond KBonus's calibration sample | Mission-specific part |
| --- | --- | --- |
| Whitening about each segment's mean | Median-normalised segments carry a mean offset into the undecimated transform's approximation level, which every template's own mean projects onto; any median normalisation does this | None |
| Epoch-aware cleaning, refined ephemerides, depth-consistency check | Corrects upstream heuristics that ignore epochs (harmonic cleaning) or reject planets under a strong signal's mask (85%-of-MES re-fold) | Tolerances in transit durations and cadences |
| Coarse period grid, lower screen, refinement | A two-stage search: the screen only has to pass the signal, the refinement restores the grid's loss | Step factor and screen threshold set on Kepler's 4-year baseline |
| Exact aliases as repeats | Period algebra (p/q aliases on the alias spacing) | None |
| Templates through the delivered detrending | A matched filter must match the delivered shape; the smoother's response is computed, not fitted | KBonus's 2-day-knot spline (`template_flattening`, null elsewhere) |
| Unit-spread normalisation of flattened templates | Empirical noise calibration per template | None |
| FFT whitening | Same mathematics, different summation | None |
| Long-gap segmentation | A gap longer than any intra-quarter (or intra-sector) gap separates independent stretches; filling it with reflected flux only adds artifacts | 30 days, longer than Kepler's quarterly and TESS's orbital gaps |

Against the 1 November Detector, on the out-of-sample sets (both legs per
curve in one process, `results/performance-engineering/2026-11-02-detector-speed-on-survey-curves`):

| | 1 November | Final |
| --- | ---: | ---: |
| 249 searchable confirmed planets on 200 random hosts: exact / missed | 233 / 15 | 239 / 9 |
| Detector CPU / GPU per host | 7.90 / 4.41 s | 3.47 / 2.14 s |
| 241 survey curves, stratum-weighted: Detector CPU / GPU per curve | 2.27 / 1.40 s | 1.16 / 0.87 s |
| survey curves' candidates | 201 | 324 |

Seven confirmed planets are found that 1 November missed (Kepler-20 e,
Kepler-36 b, Kepler-149 d at its own period, Kepler-1406 b, Kepler-1407 b,
Kepler-1640 b, Kepler-1069 b) and one is lost (Kepler-138 b, 10.3 days,
138 ppm). The survey curves carry 61% more candidates: the price of the
long-transit and near-threshold sensitivity, which the vetting policy
answers (9.3% of unexplained candidates kept).

The out-of-sample sets also stopped two changes that the calibration's
synthetic planets favoured: the coarser period grid, which missed five
confirmed planets (two with transits far shorter than their hosts' density
predicts), and the first version of alias resolution, which let a single
extreme event masquerade as an alias. Both were measured on
real curves the calibration never drew before any production change.

