# KeplerX vetting: data layout, speed and a proposed policy, 2 November 2026

Vetting is 44% of KeplerX's CPU time on the 240-case benchmark slice
(`results/performance-engineering/2026-11-01/batch-benchmark`: 760 of
1,718 CPU-s), concentrated in cases with candidates (median 0.07 s, maximum
83 s per case). The batch is GPU-bound today, so this CPU is spare while the
GPU limits, and it becomes the limit as soon as the GPU work shrinks; it is
all of the time of a vetting-only rerun.

## Combined result

Development-quick set (1,752 cases, revision 4, 8 workers), with the parallel
Detector work (GPU, recovery) and this record's changes together:

| Run | Wall | Detect CPU/case | Vet CPU/case | Vetted signals | Vet CPU/signal |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 November (previous Detector) | 62.1 min | 4.33 s | 3.52 s | 2,425 | 2.54 s |
| 2 November, new Detector (5de0587eb) | 39.0 min | 4.11 s | 3.53 s | 3,269 | 1.89 s |
| 2 November, final state (2b25ce598) | 37.7 min | 3.71 s | 2.97 s | 2,908 | 1.79 s |

(1 November's vetting table lost rows to its signal-only key, so its signal
count is the candidates in its detections.) The last two runs score the same:
efficiency 50% at S/N 8.13, 90% at 11.28, detectable planets 86.7% exact.
The batch is now CPU-bound; vetting is 44% of its CPU. Records:
`KeplerX/results/calibration-previews/2026-11-02-development-quick-run-*`.

Workers on the 240-case slice of 1 November (detect and vet, main at
8e95e4a97, every case complete): 8 workers 320.6 s, 10 workers 312.1 s, 12
workers 315.3 s. Beyond 8 workers the batch gains at most 3%: what remains
is the queue for the GPU claim and the tail, not cores (a repeat of the
8-worker leg overlapped other jobs and is not counted).

## Benchmark

24 cases of that slice with candidates (the 12 with most vetting CPU and 12
drawn at random, 121 signals), detected once with the current code, then
vetted repeatedly in one process from the saved detections
(`scratchpad replay.py`; receipts not kept). Reference and candidate run
from exports of the same commit, so the comparison covers only Vetter files.
Every change below left every vetting file of all 24 cases byte-identical.

Profile after the first change (332 CPU-s under the profiler): odd/even
depth fits 33%, transit refinement 25%, chi-square tests 14%, local
morphology 10%, MES metrics 7%, cosine-versus-transit 6.5%. MINPACK's own
work inside `leastsq` is 77 s of it: the odd/even and refinement residuals
have one row per cadence (about 60,000).

## Changes

1. **Transit model on per-orbit runs** (`vetter/transit_model.py`). The model
   found its near-transit samples with a pass over every supersample point,
   and once the in-transit window exceeded 8% of the orbit it integrated all
   300,000 of them on every evaluation. The cadence times now carry a flat
   bucket table (bucket j: first cadence at or after start + j width, width
   = baseline / cadences). Each orbit's run of cadences that can hold an
   in-transit sample is two table reads, widened by the exposure half-width,
   a 1e-9 phase margin and the table's crowding; only those samples get
   batman's sky separation and the limb-darkening integral, and only those
   cadences are averaged, with batman's NumPy expression. A sample's flux
   depends on no other sample and is exactly 1.0 outside the window, so the
   light curve is the same bit for bit. Interpolation of the integral is
   still chosen by the point count the previous model passed (all points
   above 8%, else the near points), counted exactly only near the 2,048
   threshold. The memo and sky-separation caches went: on the worst case
   they hit 4 times in 10,194 calls. NumPy's `searchsorted` over the 300,000
   supersample times cost about 160 µs a call (cache misses); the bucket
   table costs about 5 µs.
   Vetting CPU 400.8 -> 311.9 s (-22%); worst case 83 -> 33 s.
2. **Cauchy objective without scipy.stats** (`vetter/exact_fit.py`). The
   odd/even fits start from L-BFGS-B on `neglogcauchy`, which lmfit reduces
   with `scipy.stats.cauchy.logpdf` (0.7 ms of argument handling per call,
   20,000 calls). For loc 0 and scale 1 that is `log(1.0/np.pi/(1.0+x*x))`
   on the same doubles; the replacement evaluates exactly that (checked on
   Cauchy residuals up to 1e200, overflow included).
3. **Pre-ingress shortcut in the trapezoid and box kernels**
   (`vetter/residual_kernels.c`). Where both logistic arguments are at most
   -40, each term is below expit(-40) = 4.25e-18; with |depth| <= 1 that is
   under a quarter ulp of any baseline |c0| >= 0.5, so `c0 + box` rounds to
   `c0` and the two `exp` calls are skipped.
4. **No hashed transform cache in the chi-square tests**
   (`vetter/accelerated_vetter.py`). The unmasked flux transform was kept in
   a dict keyed by the SHA-1 of the padded flux. Padding depends on each
   signal's duration, so it rarely hit, and hashing about 1 MB cost about as
   much as the wavelet transform it saved; the transform is now computed.

5. **Gap filling with its median computed once** (`vetter/gap_filling.py`).
   Upstream's `pad_time_series` recomputes `np.median(y)` of the whole array
   for every gap it reflects, inside its gap loop, although y does not
   change there. The chi-square tests pad the whole curve once per signal:
   77 ms for a 49,143-cadence curve with 111 reflected gaps, 60 ms of it
   medians. The replica is upstream's code with that value computed once;
   4,800 calls on 200 curves (whole and random segments, three fill modes)
   returned the same bytes and the same exceptions, 3.2 times faster.

All five together: vetting CPU 400.8 -> 279.2 s on the 24 cases (-30%),
every vetting file identical apart from the added box metrics below.

Kept: the two-entry limb-darkening table cache in `residual_kernels.c`. A
table rebuild costs 0.5 ms against 0.01-0.8 ms for an evaluation, and the
refinement's Jacobian changes the planet radius in one column of five, so
dropping it would add about 30 s per 24 cases. Its replacement is one table
per star over (separation, radius), built once; not done.

Tried and not adopted: the odd/even trapezoid fits pass every cadence of
the folded curve to MINPACK although only rows within the parameter bounds'
reach of the transit depend on anything but the baseline c0. Out-of-window
rows reduce exactly (in real arithmetic) to two rows, sqrt(S1)(c0 - mean)
and a constant, which keep the sum of squares, J^T J, J^T r, the column
norms and lmfit's ndata. On the 24 cases it cut vetting CPU a further 15%
(290 -> 248 s) and left every other metric unchanged, but the odd/even
values themselves moved: median relative change 1e-6 to 1e-4, yet a quarter
of the 121 signals moved by more than 50% and about 20 standard errors
flipped between a value and none. These fits sit on bounds and in flat
valleys, so their result depends on the optimizer's path at rounding level;
the statistic barely separates planets from false positives (AUC 0.42 on
the development run). The transit refinement cannot be reduced this way at
all: its period moves the late-epoch windows by up to 1% of the period per
orbit.

## A stable odd/even comparison

`vetter/odd_even_box.py` (component `odd_even_box`, metrics
`odd_even_box_*`) holds the refined ephemeris fixed, so each transit's depth
is a weighted mean: its baseline cadences one to three durations from the
centre minus its cadences wholly inside the box, with the error of that
difference. Each parity's depth is the inverse-variance mean of its
transits, its error widened by the square root of their reduced chi-square
when above one; the statistic is |odd - even| over the combined error. On
the development run's candidates (current revision-4 arrays with the run's
refined ephemerides) it separates planets from false positives as well as
upstream's fits (AUC 0.59 against 0.59; against binaries 0.54 and 0.53),
measures 94% of candidates instead of 79%, has a tighter planet distribution
(99th percentile 3.0 against 4.9) and takes 2 ms per signal instead of about
0.9 s. It adds little to the linear policy here because most binaries in
the sample are found at their true period, where odd and even eclipses are
alike; the strong secondary is what flags them. Upstream's odd/even fits are
kept until Adam decides; dropping them would remove a third of vetting CPU.

## Run database

- Candidate, vetting and BOI rows are keyed by (case, signal). Signal IDs
  name the source star and calibration cases share stars, so the old
  signal-only key dropped rows (325 candidates in the benchmark's files, 308
  rows). Scoring reads the files and was not affected.
- Every stage's run.json carried the worker's configuration, 75 source
  digests and environment: 37% of the 1 November development run's database
  (21.7 of 58 MB), about 3 GB at survey scale. Batch workers now write each
  such record once as CASE/provenance/SHA256.json; the run keeps it once in
  the `provenance` table and run.json holds {"sha256": ...}
  (`RunOutputs.provenance` reads it back). Single-stage commands keep full
  records.

## Seen, not changed

- Preparing a survey curve from LC2 hashes the whole 58 MB source index
  (`receipt(request["index"])`, 40 ms of a 60-80 ms load) for every curve:
  about 2.8 CPU-hours over 250,000 curves, 0.6% of a survey's CPU. The index
  is a run input; its receipt belongs in the run's provenance, not in every
  packet. Left as is: every quick fix is a per-process cache.
- Packets store the normalised flux and error beside their native copies
  (`native_flux`, `native_error`) and a `mask` that is `exclusion_bits == 0`.
  Normalised flux is native divided by the quarter scale, the operation
  prepare performs, so it can be recomputed bit for bit; dropping the three
  would shrink packets by about a third (compression and two decompressions
  per curve, about 110 ms of CPU, and the disk budget). It changes the packet
  format and the calibration sample's receipts, so it waits for a rebuild.

## Vetting policies (linear score adopted)

Until 2 November vetting applied no policy. On the 1 November development-quick run
(calibration-sample revision 3, 2,425 candidates) 1,845 candidates are
unexplained false alarms and 405 are synthetic, known or validated planets.
`calibration/vetting_policy.py` fits the policies of `vetter/policy.py` and
measures them out of fold, five folds by source star (a star's copies never
sit on both sides): each fold is judged by a policy fitted on the others.

- `consistency-and-uniqueness`: dismiss when the temporal chi-square
  statistic is below 3.63 (the transits do not support the detection
  consistently; Seader et al. 2013) or the strongest other dip in the folded
  curve exceeds 0.70 of the detection's MES (DR25 Robovetter's model-shift
  uniqueness idea). Each cut dismisses 2% of the fitted planets.
- `linear-score`: a penalised logistic regression on six features (signed
  logs of the temporal chi-square statistic and of the global transit-model
  reduced chi-square, the strongest dip and the strongest brightening at
  other phases over the MES, log period, log MES), cut where it dismisses 4%
  of the fitted planets. At fixed temporal consistency a stronger MES lowers
  the score: real transits gain consistency with strength.

Out of fold:

| Kept | Two cuts | Linear score |
| --- | ---: | ---: |
| Planets (synthetic, known, validated) | 96.0% | 96.3% |
| Known Kepler planets | 98.8% | 98.8% |
| Validated planets | 7 of 7 | 7 of 7 |
| Planets at MES 7.1-8.5 | 94% | 96% |
| Planets at MES 8.5-10 | 84% | 89% |
| Unexplained false alarms | 19.6% | 7.8% |
| Eclipsing binaries | 67% | 62% |
| Blended eclipsing binaries | 94% | 94% |
| Leaked binaries | 58% | 29% |
| Leaked KOI false positives | 77% | 48% |
| Null original curves with a false alarm (of 96) | 40 | 17 |

**Refit on the current Detector.** The 2 November development-quick run
(new Detector, revision 4, 1,752 cases, 39 minutes on 8 workers against 62)
is a held-out test of the fits above: applied unchanged, two cuts kept 96.7%
of planets and dismissed 84% of unexplained false alarms (null originals 98
-> 39); the linear score kept 95.1% (all known and validated planets) and
dismissed 95% (98 -> 17). Refitted on that run (constants now in
`vetter/policy.py`, record
`KeplerX/results/calibration-audits/2026-11-02-vetting-policy-new-detector/`),
out of fold:

| Kept | Two cuts (2% each) | Linear score (4%) |
| --- | ---: | ---: |
| Planets (synthetic, known, validated) | 96.2% | 96.2% |
| Known and validated planets | 100% | 100% |
| Planets at MES 7.1-8.5 | 94% | 94% |
| Planets at MES 8.5-10 | 86% | 92% |
| Unexplained false alarms | 13.6% | 7.7% |
| Eclipsing binaries | 62% | 60% |
| Blended eclipsing binaries | 86% | 86% |
| Leaked binaries | 55% | 30% |
| Leaked KOI false positives | 76% | 44% |
| Null original curves with a false alarm (of 98) | 32 | 21 |

Detection then vetting still meets both efficiency targets on that run
(scoring now reports `efficiency_after_policy` for each policy: planets found
exactly whose candidate the policy keeps, against information S/N):

| | S/N at 50% | S/N at 90% |
| --- | ---: | ---: |
| Detector alone | 8.13 | 11.28 |
| Then linear score | 8.17 [7.83, 8.73] | 11.06 [10.19, 12.19] |
| Then two cuts | 8.37 [7.94, 8.91] | 11.85 [10.56, 13.38] |

(gamma fits with 50 family bootstraps; vetting only removes detections, so
the lower 90% point of the linear score is the fit's freedom, within its
interval.)

A gradient-boosted model on all metrics (scikit-learn, same grouping;
`learned-model-headroom.py` in the record folder) kept 96% of planets and
5.4% of unexplained false alarms, so the linear score is close to what these
metrics hold. Blended binaries need the pixels.
Record: `KeplerX/results/calibration-audits/2026-11-02-vetting-policy/`.

**Adopted, then rebuilt on shape alone.** Adam adopted `linear-score`
(config.json `vetting.policy`). Its constants held unchanged on the run with
flattened long templates
(`results/calibration-previews/2026-11-02-development-quick-run-flattened-templates`,
commit 42f0b594b): 95.7% of planets kept, 8.2% of unexplained false alarms,
null originals with a false alarm 123 -> 29. The full check then failed:
it dismissed TrES-2 b (the end-to-end fixture: quarter 0, MES 98, four
transits; score -2.23 against a cut of -2.10, the MES term alone -8.6).

The cause is general. The temporal chi-square statistic is Z / sqrt(chi2 /
(n - 1)); a planet whose transit depths scatter by a fraction f has chi2 /
(n - 1) about 1 + f^2 Z^2 / (n - 1), so the statistic saturates at
sqrt(n - 1) / f while the MES keeps growing. With the MES weighted
negatively (-4.3 on log MES against +3.1 on the log statistic) any planet
is dismissed once strong enough, and sooner with few transits (TESS
sectors, long periods, a single quarter). The negative weight comes from
the calibration sample's design: injected planets are steered to the
detection limit, so there a strong signal is usually a false alarm. Judged
as if every planet it kept were k times stronger with the same shape, the
adopted score dismissed 7% at k = 2, 30% at 4, 65% at 8, 89% at 16.

The policy now reads only scale-free quantities (`vetter/policy.py`): the
transits' depth scatter per unit signal from the temporal chi-square, f =
sqrt((chi2/(n-1) - 1)(n - 1)) / Z, with n the statistic's cadences (a new
cheap metric, `temporal_chi2_cadences`, computed before the policy); the
same for the chi-square across wavelet scales; the strongest other dip and
the strongest brightening over the MES; the global transit fit's excess
reduced chi-square per unit signal; log period; the MAD of the MES with the
detection's transits masked (noise that is not Gaussian); and log MES with
its weight held at or above zero (the fit holds it at zero). The fitter
(`calibration/vetting_policy.py`) now reports, out of fold, the share of
kept planets dismissed when k times stronger; it is zero for this policy at
every k by construction. The two-cut alternative did not survive the
change (in scale-free form, at 2% of planets per cut, it dismissed only 51%
of negatives) and is removed.

Fitted on the flattened-template run (cadence counts backfilled from the
light curves; record
`results/calibration-audits/2026-11-02-vetting-policy-scale-free/`):

| Linear score, 4% planet budget | Out of fold | Applied to the previous Detector's run |
| --- | ---: | ---: |
| Planets kept | 94.8% | 96.0% |
| Known and validated planets kept | | 98.9% |
| Unexplained false alarms kept | 9.9% | 9.4% |
| Null original curves with a false alarm | | 98 -> 26 |
| Kept planets dismissed at 2, 4, 8, 16 times the signal | 0 | 0 |

It gives up two points of false-alarm rejection (7.7% kept before) for a
boundary that does not depend on how strong the calibration's planets
happened to be. TrES-2 b scores +2.2 against a cut of -2.2.

Applied unchanged to the development-quick run of current production (FFT
whitening and gap segmentation in Detector and Vetter, 9f125d796;
`results/calibration-previews/2026-11-02-development-quick-run-current-production`):
96.0% of planets kept (known 97.7%, validated all), 9.2% of unexplained false
alarms kept, null originals with a false alarm 118 -> 29, efficiency after
the policy 50% at S/N 7.84 and 90% at 10.89 (Detector alone 7.58 and 10.34).
A refit there scored 95.8% and 9.4% out of fold, so the constants stay
(`results/calibration-audits/2026-11-02-vetting-policy-current-production/`).
On the final Detector of the day (alias resolution added, cfe2a4e3c;
`results/calibration-previews/2026-11-02-development-quick-run-final-detector`),
unchanged again: 96.0% of planets kept (known 97.7%, validated all), 9.3% of
unexplained false alarms, null originals 118 -> 30, efficiency after the
policy 7.85 and 10.95 (Detector alone 7.57 and 10.35); a refit scored 95.3%
and 9.3% out of fold
(`results/calibration-audits/2026-11-02-vetting-policy-final-detector/`).

**Policy first, then the costly metrics** (`vetting.dismissed_metrics`,
default `"all"`). With `"policy"` and a named policy, the components the
policy reads (MES, chi-square, weak secondary, cosine against transit) run
first; a signal the policy dismisses, with no failure among them, skips
upstream's odd/even trapezoid fits and local morphology, listed in its
disposition as `skipped_components`. No component changes another's input,
so the order changes no value. On the 24 cases with `linear-score` (112 of
121 signals dismissed; the benchmark favours false-alarm-rich curves),
vetting CPU fell from 305 to 204 s (-33%); kept signals' metrics were
identical to the full run, dismissed signals kept every metric they
computed, and the default (`"all"`, no policy) left every vetting file
identical to before. On the new Detector's development run, vetting is 46%
of all CPU and 81% of candidates are unexplained false alarms.

`calibration/scoring.py` reports, for every policy, what it would keep by
label (`vetting_policies` in `summary.json`). `vetting.policy` in the
configuration applies one: a dismissal makes the disposition `DISMISSED`
with what fired and its values; a policy never accepts, an unavailable
feature never dismisses, and metric failures stay `UNRESOLVED`. The default
configuration names none, so vetting outputs are unchanged.
