# KeplerX on curves it never saw, and localization speed, 2 November 2026

Adam asked whether the day's improvements were fitted to the calibration
sample rather than to transit science, to test on the rest of the KBonus
curves on the two discs, and to make localization fast. This records the
out-of-sample benchmarks, what they found, and what changed.

## Which improvements could overfit

Most of the day's speed work is exact: the Vetter's bucketed transit model,
the Cauchy reducer, the pre-ingress shortcut, the fused outlier veto and
epoch cleaning, the Detector's search grid and whitening state, the run
database. Each was checked bit for bit (or to the stated rounding) against
the code it replaced, so it cannot have learned anything from the sample.
Two things can: a vetting policy, whose weights are fitted, and the
Detector's science settings (template bank, flattening, thresholds), which
were chosen on the calibration sample. Both are tested below on curves the
calibration never used.

## Benchmarks

`calibration/survey_benchmark.py` builds `data/survey-benchmark` from the
LC2 inventory (406,547 sources on both discs). Every source is in a
development family (the holdout stays untouched) and is not a calibration
host; each is prepared exactly as production.

- diverse: 896 curves, up to 32 per stratum of star class (Gaia Teff/logg)
  x G magnitude (31 strata), with weights (stratum size / drawn) for
  survey-representative averages; diverse-quick is the first 8 per stratum
  (241). 448 come from each disc, from 533 of the 814 shards.
- known-planets: 493 KBonus curves of confirmed-planet hosts that the
  calibration sample's known-planet lane did not draw (development half of
  the 959 unused), 632 planets.
- known-planets-one-quarter: the same curves cut to their quarter with the
  most usable cadences (about 90 days), so each planet shows a few transits,
  as in a TESS sector or for a long period.

`calibration/localization_benchmark.py` builds `data/localization-benchmark`:
100 eligible sources on the 44 parents whose TPFs (eight or more quarters,
1,351 in all) are on the discs, each with a localization request.
`calibration/benchmark_scoring.py` scores any batch run on them with the
calibration scorer's matching rule.

## Detection holds out of sample

Production (f7f201a7e) on diverse-quick and known-planets, 734 cases in
35 minutes on 8 workers
(`KeplerX/results/survey-benchmark/2026-11-02-diverse-quick-and-known-planets`):

| | Out of sample | Calibration sample |
| --- | ---: | ---: |
| Known planets with three transits found at their period | 589 of 631 (93.3%) | 93.3% of known planets in range |
| Survey curves with a candidate (weighted) | 33.9% | 26.9% of null originals |
| Detection CPU per survey curve (weighted) | 3.2 s | 4.3 s per calibration case |

Known-planet recovery is the same on curves the settings never saw. Survey
curves carry somewhat more candidates than the calibration's null originals,
which are drawn differently; the vetting policy below is what removes them.

## The vetting policy did overfit, and was rebuilt

The adopted linear score dismissed TrES-2 b in the end-to-end test. Its MES
weight was negative: in the calibration sample the injected planets sit near
the detection threshold, so there a strong signal is usually a false alarm,
and the fit learned to distrust strength. Real planets are not distributed
that way. The policy now reads only the signal's shape, and strength can only
add confidence
([vetting speed and policy](keplerX-2026-11-02-vetting-speed-and-policy.md)).
On the confirmed planets it never saw, found at their exact period:

| Kept by vetting | First score | Scale-free score |
| --- | ---: | ---: |
| Full curves, all (589) | 97.5% | 98.0% |
| Full curves, MES 100-300 (24) | 95.8% | 100% |
| Full curves, MES above 300 (26) | 88.5% | 100% |
| One quarter, all (272) | 98.5% | 98.9% |
| One quarter, MES above 300 (6) | 83% | 100% |

Current production (9f125d796: FFT whitening and gap segmentation in both
stages) gives the same on these planets: 589 of 631 found at their period,
98.0% of them kept (100% above MES 20), the same twelve dismissed; on one
quarter 272 found, 98.9% kept
(`KeplerX/results/survey-benchmark/2026-11-02-known-planets-current-production`).
Both scores keep about the same share of survey curves with a candidate
(11.1% and 11.6% of all survey curves, weighted). The scale-free score loses no strong
planet; the twelve it dismisses on full curves are all below MES 17, mostly
weaker members of multi-planet systems. On one quarter both dismiss
Kepler-840 b (MES 290), whose events there do not support it (temporal
statistic -13) beside a folded dip 55% as strong.

## Vetting on survey curves

With the Detector's FFT whitening (0cd9e1c7e: Detector CPU -57% on the
weighted survey mix, the same known planets found), vetting is the larger
CPU stage. On 30 random survey curves with candidates (128 signals, 21
kept) it took 1.40 CPU-s per signal: the chi-square tests 28%, the MES and
weak-secondary statistics 24% (both re-whiten the curve under their own
transit masks, as upstream defines them), the refinement fits 22%, upstream's
odd/even fits 17% and local morphology 12%.
With `vetting.dismissed_metrics: "policy"` the dismissed signals skip the
last two: 1.05 CPU-s per signal (-25%), the same 21 kept signals with the
same metrics. The Vetter's statistics are now whitened by the same FFT
sums as the Detector (`vetting.whitening: "fft"`, 9f125d796): on those 30
curves at current production vetting took 137.5 instead of 152.3 CPU-s
(-10%), every disposition equal and the MES-family metrics within 5e-15
relative, all others bit for bit. The chi-square tests take upstream's own
masked path and are unchanged. The heaviest single signal (a P = 1.33 d false alarm on a
known-planet host) took 107 s: its refinement fit wandered to a/R* below
1 + Rp/R*, where half of every orbit is in transit, and upstream's optimizer
made 6,153 evaluations there.

## Localization

On a 15-quarter case (KOI-4) localization took 46 s, of which 37 s waited on
the network: the ESA Gaia archive (one cone per TPF aperture shape, 10 s to
over a minute this morning, and four 60-second timeouts on another case) and
lkprf's PRF downloads (once per channel). Imports took 3.6 s (paid once per
batch worker) and the arithmetic about 2.5 s. Across the survey there are
113,402 parents with PSF Flat sources.

Changes, all on main:

- Signals the vetting policy dismissed are not localized; their dismissal
  stands. On the localization benchmark 63 of 100 sources have candidates
  but only 28 open their pixels, and 49 of 381 signals are localized.
- An unreachable archive leaves the case's signals UNRESOLVED with the
  reason, instead of failing the stage and stopping the batch.
- A parent quarter in which the source has no cadences (one benchmark source
  is off silicon in quarters 3, 7, 11 and 15) is skipped, not fatal.
- A quarter whose trend columns explain the transit ("not estimable": it
  says nothing about where the light comes from) is left out and listed,
  instead of marking every signal of the case UNRESOLVED (the October rule,
  which mirrored native's singular solves). On the production localization
  run this measures 5 of the 7 unresolved kept signals, from their other
  13-17 quarters; all 183 localized signals' measurements are unchanged.
  Of the other two, lmfit left one centre without uncertainties because
  three quarters' amplitudes rested on their lower bound of zero, where its
  bounded transform has no gradient; such a joint fit is now repeated from
  its solution with those amplitudes held at the bound, and that signal is
  measured (1.6 arcsec, 0.8 sigma), every other result unchanged. One
  remains unresolved ("one or more variable did not affect the fit").
- The per-pixel regression eliminates the B-spline columns by banded
  Cholesky: a short transit on a long quarter has about 1,000 nuisance
  columns, and the dense per-pixel factorization was 65% of a 17-quarter
  case. On the benchmark's 28 localized cases (316 quarters) it takes 8.2 s
  instead of 32.9 s, depths within 5e-13 of their error; the 17-quarter
  case went from 16-18 to 5 CPU-s.
- `localizator/gaia_field.py` serves cones from a local table of a whole
  field (flat columns in declination zones, two binary searches per zone,
  about 1 ms per cone) with the archive's table bit for bit (golden test,
  and all 21 stored archive cones). The per-case cone store, written and
  never read, is gone.
- Without a field table, a parent's cones (one per TPF aperture shape, 2.3
  per parent, all on the same centre) come from one archive query at the
  widest radius, the narrower ones taken from its sources: equal to the
  archive's own answers for all 68 stored cones, and the twelve KOI-host
  localizations identical with 12 archive lookups instead of about 28.

The localization's science on real objects: the twelve KOI hosts whose
TPFs (12-18 quarters) are on the discs, localized with their KOI clocks
(`KeplerX/results/localization-benchmark/2026-11-02-koi-hosts`, at
c18807f07). All 14 signals were measured. Every centroid-offset false
positive (the seven flagged `co`) lies 1.1-14.2 arcsec from the target at
8-427 sigma; the stellar-eclipse false positive K07699.01 (`ss`) lies on
the target (0.10 arcsec), as an eclipse of the target must; KOI-4.01 lies
0.12 arcsec away. Each case took 0.8-2.8 CPU-s once its Gaia cones were
stored.

The localization benchmark through production, the same 100 sources:

| | Morning (second run, 3f11d840f) | Now (third run, 9f125d796) |
| --- | ---: | ---: |
| Batch wall on 8 workers | 9.4 min | 2.9 min |
| Detection CPU, all cases | 711 s | 252 s |
| Vetting CPU, all cases | 759 s | 553 s |
| Localization CPU, all cases | 73 s | 35 s |
| Localization per localized case, wall / CPU | 24.9 s / 2.6 s | 2.2 s / 1.3 s |
| Cases that open their pixels | 28 | 26 |

(`KeplerX/results/localization-benchmark/2026-11-02-second-run` and
`-third-run`.) The wall rows are not a fair comparison: the morning run
fetched many of its parents' Gaia cones from the archive and shared the
machine, while by the third run the cones were stored. The CPU rows,
measured per process, are the comparison; a parent new to the archive now
costs one query instead of one per aperture shape.

## Current production on all 896 survey curves

Production at 9f125d796 (the FFT Detector with gap segmentation, the
Vetter's FFT whitening, the scale-free score), 896 stratified survey curves
in 11.7 minutes on 8 workers, no errors
(`KeplerX/results/survey-benchmark/2026-11-02-diverse-full-current-production`;
a single-process vetting replay, niced, ran beside it for about ten minutes,
so its wall figures may be slightly high; CPU is per process).
Survey-weighted, per curve:

| | Morning production (241 curves, f7f201a7e) | Now (896 curves) |
| --- | ---: | ---: |
| Detection CPU | 3.20 s | 1.33 s (-58%) |
| Detection wall on 8 workers (machines differently busy) | 7.06 s | 3.42 s |
| Vetting CPU | 2.00 s | 1.72 s (-14%) |
| Curves with a candidate | 33.9% | 30.3% |
| Candidates per curve | 1.67 | 1.37 |
| Curves with a kept candidate | 11.1% (first score) | 10.2% |

The two wall columns ran beside different loads, so they are not a fair
comparison. The quiet-machine batches (diverse-quick, 241
curves, 8 workers, nothing else running, the same code and vetting with only
the detection settings differing;
`KeplerX/results/performance-engineering/2026-11-02-detector-speed-on-survey-curves`)
are: Detector CPU 2.88 -> 1.38 s per curve weighted (-52%), Detector wall
4.92 -> 3.51 s (-29%), the whole batch 287 -> 200 s (-30%). CPU is halved;
batch wall is now bound by the shared GPU screen. At the measured batch rate
(896 curves in 11.7 minutes) the whole survey's 250,923 PSF Flat curves
would take about 2.3 days on 8 workers for detection and vetting (about 3.5
at the morning's quiet rate), 213 CPU-hours, before localization. Of the 136 candidates kept on these curves, 8 are a
parent target's KOI and 1 a neighbour's (all leaks of known signals); 127
are unexplained: 80 at MES 10 or below, where the policy keeps what it
cannot tell from a planet by shape, and 43 at MES 10-50 and 4 above 50,
the review queue of new signals on background stars.

A robustness sweep of 5,000 survey curves drawn uniformly (none from the
benchmark sets; `survey_benchmark --sweep`), detection and vetting with
stop_on_error off, ran without a single failure: the 1,375 longest at
2a4dc1c83 (stopped for the quiet-machine timing), the other 3,625 on
the final commit cfe2a4e3c in 36 minutes on 8 workers
(`KeplerX/results/survey-benchmark/2026-11-02-robustness-sweep` and `-rest`).
Uniformly drawn, 24.9% of survey curves carry a candidate and 7.6% keep one
(the stratified estimate above, 10.2%, weights a smaller sample); detection
took 1.11 and vetting 1.19 CPU-s per curve on average, the slowest curve
15.5 and 68 CPU-s.

By star class (survey-weighted within each):

| Class | Share of survey | With a candidate | With a kept one | Detection CPU | Vetting CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| K | 58.7% | 31.2% | 11.6% | 1.37 s | 1.77 s |
| G | 15.0% | 19.2% | 5.0% | 1.05 s | 0.96 s |
| No Gaia Teff/logg | 13.3% | 45.0% | 12.6% | 1.66 s | 2.94 s |
| M | 7.0% | 18.2% | 2.6% | 0.87 s | 0.63 s |
| Subgiant | 2.4% | 38.7% | 18.1% | 1.51 s | 2.27 s |
| F | 2.0% | 27.0% | 9.9% | 1.37 s | 2.13 s |
| Giant | 1.5% | 23.7% | 6.2% | 0.97 s | 0.40 s |
| Hot | 0.1% | 35.7% | 12.4% | 1.93 s | 4.02 s |

## Pixels streamed from MAST, and the Kepler field's Gaia table (evening)

At 18:15 UTC Adam asked for the field's Gaia table for all of KBonus, and
for survey pixels the way the v1 and v2 campaigns took them: download,
localize, delete. Those campaigns fetched parent TPFs from MAST's public
S3 bucket (`pipeline/vetting/vetting_pixels.py`), only for the vetting
cohort's events, into each batch's pixel cache. After vetting,
`pipeline/deterministic/vet_survey.py` (`evict_batch_pixels`) wrote a
receipt (file, size, sha256, URL) and deleted the cache.

KeplerX now keeps no pixel files at all (78ae6055d):

- A request with `"archive": "mast"` lists the parent's folder in the
  bucket. The localizer streams the long-cadence quarters it needs into
  memory over 8 connections and keeps each file only until its pixels are
  read (`localizator/mast_pixels.py`). Each file's URL, size and sha256 go
  into `localization.json` (`pixels`). `pixel_requests attach ...
  --archive mast` gives every manifest row without stored TPFs such a
  request, and only cases with a kept signal fetch anything.
- A quarter with no prepared cadence within one transit duration of a
  localized signal is not read (`quarters_without_transits`).
  Upstream's transit column is zero there, and MultiLocalize drops the
  quarter.
- Each TPF is reduced on reading to its headers and its time, flux and
  flux-error columns. Peak memory on a 67 MB parent fell from 0.70 to
  0.51 GB.
- Replaying the 26 benchmark cases that open pixels gives the previous
  code's results field for field, from disc and from MAST. The S3 copy of
  a stored TPF is the same bytes.

What the survey would stream: of the uniform 5,000-curve sweep, 379
curves keep a candidate (481 signals). Their transit quarters on MAST come
to 59 MB per curve on average (median 19 MB, 99th percentile 0.84 GB,
largest 1.8 GB), or 1.1 TB over 250,923 curves. Reading every quarter the
source has data in would be 1.36 TB, and the parent's whole folder
2.35 TB. Reading only transit quarters saves 17%, because 45% of kept
curves have a signal longer than 90 days.

From the Mac one case streams at about 5 MB/s, and 8 worker processes
together at 15.5 MB/s (16 processes, 16.2 MB/s). MAST's own servers alone
gave 11.2 MB/s, and both hosts together 14.7 MB/s. The line carries
27 MB/s to a nearby CDN, so the ceiling is the route to the US east coast.
The survey's 1.1 TB is therefore about 20 hours of transfer, inside the
~55 hours of detection and vetting on 8 workers, with workers waiting
while their own case's files arrive.

The Gaia table (`KeplerX/data/gaia-dr3-kepler-field`, git-ignored, built
18:25-18:53 UTC):

- 66 strips of 0.25 degrees in declination (36.25-52.75). Each covers
  only the right ascension its KBonus sources' cones can reach, with a
  0.25-degree margin: 183 square degrees instead of the 277 of the
  bounding box. The span of each 1-arcminute zone is stored (`zone_ra.npy`),
  and a cone outside it goes to the archive.
- 9,522,097 sources (G <= 25) in 0.40 GB of columns. The largest strip
  holds 268,643 rows.
- Trap found on the first attempt: the archive caps any result at 50,000
  rows unless asked for more, and says so only in a QUERY_STATUS OVERFLOW
  line of the VOTable. One strip came back with exactly 50,000 rows. Strips
  now ask for up to 3,000,000 rows, and an OVERFLOW answer is refused,
  for strips and single cones alike.
- The 26 benchmark cases localize identically with their cones from the
  table and from the archive. The golden test now takes KIC 12644822's
  cone from it.
- A wider check of 60 random cones against the archive could not run:
  from about 19:05 UTC the archive answered no cone in 30 minutes, and at
  19:35 it returned HTTP 503. Localization no longer depends on it.
- The builder names each strip by its declination and right-ascension
  span (a rebuild with another margin cannot reuse a wrong strip) and
  deletes the strips once the table is written.

## What is KBonus-specific, and where

The search, vetting and localization methods are mission-neutral; the
choices made for KBonus are settings or adapters:

| Assumption | Where | For another mission |
| --- | --- | --- |
| BKJD (TDB) times, quarters, Kepler quality bits kept, 5-sigma retained-flag clip | `photometry/program.py`, config `photometry` | the mission's time system, sectors or fields, its quality bits |
| Field-wide artifact epochs (BKJD 200, 246, 373, ...) | config `photometry.field_artifact_epochs` | the mission's own list, or none |
| Templates passed through KBonus's 2-day-knot flattening | config `detection.template_flattening` | the knot of that light curve's detrending, or null for raw flux |
| Segments split at gaps over the sector size | config `detection.segmentation` | the same rule: it follows the data's gaps |
| Thresholds (MES 7.1, screen 6.6, three transits) | config `detection` | recalibrate with injections on that mission's curves |
| Vetting policy weights | `vetter/policy.py` | refit; the features are shape-only and must stay so (the fitter reports kept planets lost when stronger) |
| Kepler PRF (lkprf.KeplerPRF), 4 arcsec pixels, TPF header keys | `localizator/program.py`, `gaia_cone_search.radius_arcsec` | lkprf's TESS PRF and 21 arcsec pixels are already in upstream's code; Roman needs a PRF model |
| Gaia DR3 field table, strips spanning the KBonus sources | `localizator.gaia_field build` (reads the KBonus inventory) | the mission's source list |
| MAST bucket, Kepler long-cadence file stamps and quarters | `localizator/mast_pixels.py` | TESS TPFs are in the same bucket under their own prefix and sector names; Roman's pixels are elsewhere |

## Open for Adam

- Resolved in the evening: the Gaia table is built and configured
  (`localization.gaia_field`), and survey pixels stream from MAST with
  nothing stored (see above).
