# V3 throughput toward one day, night of 29 October 2026

This note follows [the edge-case note](keplerX-2026-10-29-edge-cases.md). Commits:
`8087f365d`, `15cb3b56e` and `1a2b754d3`. Receipts are under
`KeplerX/results/performance-engineering/20261029/` (`live/`, `repeats/`, `ldfast/`,
`grid/`, `coop/`).

**Answer.** With upstream's search grids the pipeline is GPU-bound at about
0.87 s per curve on this M1 Pro:

- 1.9 days for the 189,704 eligible dwarfs;
- 2.5 days for all 250,923 PSF Flat sources.

The 0.87 s is the 4,000-curve rate (0.82 s), plus the 5.6% more passes that
`repeat_passes` 1 needed on 400 curves.

One day is reached only by searching fewer trial periods.
`detection.period_step_factor = 2` halves them. It measured 0.53 times the
upstream grid's batch time: about 0.46 s per curve, or 1.0 day for the
dwarfs and 1.3 days for all PSF Flat sources. It costs 2.0% of the best MES on average,
with 10% of injected signals losing 4.8% or more. The setting stays off; it
is a calibration decision.

## What changed

1. **One signal, one candidate** (`detection.repeat_policy = "period"`,
   `detector/repeat_rule.py`).
   - **The problem.** Upstream's harmonic removal returns a lone detection
     unexamined, and keeps a harmonic whose epoch differs. So eclipsing
     binaries were found again and again at their own period, until ten
     candidates were found or the masks swallowed the curve.
   - **The fix.** The passes stay upstream's: same harmonic removal, masks
     and ten-candidate limit. A detection at a known candidate's own period,
     at any epoch, is no longer a new candidate. After `repeat_passes + 1`
     (default 2) consecutive passes that add nothing new, the search ends.
   - **Why other ratios keep the epoch test.** In the two-planet control
     f6603e, the search finds 3 × 12.04 d before 12.04 d.
   - **Why one residual pass is tolerated.** An immediate stop lost two
     injected planets: each appeared one residual pass after its sibling.

   Measured on 200 injected multi-planet systems (427 signals) and 100
   injected binaries from `calibration-sample` (detection only):

   | Policy | Planets recovered | Binaries recovered | Binary candidates | Binary GPU | Planet GPU |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | upstream | 103 | 54 | 519 | 576 s | 459 s |
   | `repeat_passes` 0 | 101 | 54 | 181 | 264 s | 438 s |
   | **`repeat_passes` 1** | **103** | **54** | **278** | **432 s** | **445 s** |
   | `repeat_passes` 2 | 103 | 54 | 323 | 521 s | 460 s |

   The policy also removed a crash: a 0.66 d binary whose masks left zero
   cadences.

   Vetting masks each candidate's earlier detections
   (`_get_previous_tce_masks`). `detections.json` therefore records every
   masked detection in order as `search_detections`; repeats have no signal
   ID. Vetting builds its detection list from those, and vets only the
   candidates. So the masks are the ones upstream's vetting would use; only
   the recorded `tce_id` now counts repeats.
2. **Grids that admit nothing.** One curve had 420 cadences over 9.7 d of a
   dense M dwarf, so no template duration was admissible. It is now recorded
   as `unsearchable` with no candidates, instead of an error.
3. **Limb darkening through interpolants** (`vetter/residual_kernels.c`).
   - **Where.** For fixed radius and law the flux is analytic in d on
     [0, 0.99995 − p − 0.02]. There, cached piecewise Chebyshev interpolants
     of the integral replace it whenever a call has at least 2,048 points.
     Ingress, egress, the limb zone and short calls keep the integral.
   - **Accuracy.** Against the integral, the median error is 5e-15 and the
     worst 6e-12. The worst case is at the integral's own x = 0.9 rule switch
     for p near 0.5, inside its error against adaptive quadrature there.
   - **Speed.** A 0.95 d signal vetted in 5.7 s instead of 28 s; an eclipsing
     binary in 8.2 s instead of 18.3 s.
   - **Unchanged outputs.** The 11 surviving validation signals keep their
     periods and depths. The fixture curves fall below the 2,048-point limit
     and are unchanged.
4. **Two GPU chunks in flight** (`V3_GPU_SLOTS`, default 2). The same 200 live
   curves took 159.6 and 159.3 s, against 168.1 and 166.1 s with one slot. The
   detection and vetting results were identical.

## Live measurements

Random eligible dwarfs, freshly prepared from LC2, ran through prepare,
detect and vet on eight workers. CPU and GPU totals are in seconds. Elapsed
time under other concurrent jobs is not comparable, so
elapsed rows are from quiet periods only. GPU seconds overlap with two slots.

| Batch | Curves | Candidates | Detect CPU | Vet CPU | Errors | Vetting tail (max per curve) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| before (`run6`), upstream iteration | 1,596 | 1,176 | 3,521 | 8,225 | 2 | 1,668 s |
| `run7`, immediate stop on repeats | 1,596 | 820 | 2,862 | 1,462 | 0 | 75 s |
| final (`run14`), `repeat_passes` 1 | 1,596 | 900 | 2,941 | 1,383 | 0 | 50 s |

| Same 400 prepared curves | Last detection | Batch end | Candidates | Vet CPU |
| --- | ---: | ---: | ---: | ---: |
| `run1` (28 Oct code) | 454 s | 585 s | 314 | 1,114 s |
| `run12-final-step1` (final code) | 376 s | 383 s | 258 | 414 s |
| `run12-final-step2` (final code, period step 2) | 198 s | 204 s | 198 | 276 s |

After the changes, CPU is about 3 core-seconds per curve, so 6–10
workers saturate the GPU: 6 workers took 160 s, 8 took 159 s and 10 took
164 s. The GPU's cost per pass scales with trial periods × baseline slots
(39–68 ps each), not with cadences. Sparse curves therefore cost up to 2.9
times more per cadence.

A further 4,000 random dwarfs, prepared from LC2, took 3,298 s on eight
workers: 0.82 s per curve. This run used the repeat policy's earlier
immediate stop. 778 curves had candidates, with 2,298 candidates in total.
Four failed:

- Two curves had 19 and 27 usable cadences spread over 1,200 days. Their
  segments whitened degenerately, and upstream's recheck failed. Curves with
  fewer than 100 usable cadences (about two days of data) are now
  `unsearchable`, as are curves whose every segment is degenerate. A later
  pass with fewer than 100 cadences left ends the iteration.
- One source has a quarter whose PSF fit failed: every flux is negative
  (median −70) beside eleven quarters at 695. That quarter is now excluded
  (exclusion bit 16) instead of the whole source.
- One source with 2 valid cadences remains a preparation rejection.

The 102 known-object controls ran through detect and vet with the final code.
All completed and no signal's vetting failed:

| Controls | Cases | Candidates | Catalogue clocks recovered |
| --- | ---: | ---: | ---: |
| known planets | 40 | 77 | 55 of 56 |
| eclipsing-binary false positives | 31 | 113 | 25 of 31 |
| centroid-offset false positives | 31 | 33 | 15 of 31 |

Localization was not exercised at scale: VizieR was unreachable from this
machine overnight. Since 29 October the Gaia query goes to the ESA Gaia
archive instead (`localizator/gaia_cone_search.py`, the identical table; the KIC 12644822
control localizes identically from either source), with responses cached
under `runtime/cache/gaia-dr3`.

## Disk

A case writes 1.2 MB, and 90% of that is the prepared photometry packet
(1.13 MB on average). The 189,704 dwarfs would need 230 GB; the internal disk
has 41 GB free.

`execution.retain_photometry = "candidates"` deletes the packet arrays of
completed cases without candidates. `packet.json` keeps their SHA-256, so
`prepare` reproduces and verifies them. With 82% of curves null, a full study
then needs about 60 GB.

`batch.json` lists every case in memory. A full study should run as batches
of about 10,000 curves.

## Running a full study

- **Manifests.** Write manifests of about 10,000 photometry requests each.
  The live runs used `results/performance-engineering/20261029/live/manifest.py`, which
  takes Gaia Teff/logg and fiducial densities from the eligible inventory.
  Run them one after another with `run.sh batch MANIFEST OUT --workers 8`.
- **Settings.** Keep the default two GPU slots. Set
  `execution.retain_photometry: "candidates"` and `stop_on_error: false`.
  Six to eight workers saturate the GPU.
- **Expected time.** About 0.87 s per curve on the upstream grid (1.9 days
  for the 189,704 dwarfs), or about 0.46 s with `period_step_factor: 2`.
- **Disk.** About 60 GB with released packets.
- **Localization** needs the ESA Gaia archive, or its cached responses.
- **Expected data failures.** Sources with fewer than 3 valid cadences
  remain preparation rejections (1 in 4,000).

## Search-grid trade-off

These are measured on 240 transits injected into 30 real null curves
(`grid/loss.py`). Each value is the loss of the best MES near the true
period, against upstream's full grid.

| Trial periods kept | Mean MES loss | 90th percentile | 99th percentile | GPU time | Days, 189,704 dwarfs |
| --- | ---: | ---: | ---: | ---: | ---: |
| all (upstream) | 0 | 0 | 0 | 1.0 | 1.9 |
| 2 of 3 (about step × 1.5) | 1.3% | 2.8% | 4.8% | ~0.67 | ~1.3 |
| 1 of 2 (step × 2) | 2.0% | 4.8% | 7.5% | 0.53 measured | 1.0 |
| 1 of 3 | 3.8% | 7.5% | 10.7% | ~0.33 | ~0.7 |

Every other template duration lost 2.3% (90th percentile 7.8%) for an
estimated 15% of GPU time, a worse trade.

## Rejected

- **Cooperative epoch windows in the screen kernel.** One lookup per period
  and epoch, coalesced loads and threadgroup slots. It was 1.4–1.7 times
  slower. It was also unsafe: on 145 of 200 curves the minimum cadence
  spacing is below the bin width at P > 20 d.
- **An occupancy bitmap that skips empty epochs.** It was exact. It saved
  11–17% on sparse curves but cost 3–5% on dense ones, for a net of about
  2–4%.
- **No GPU lock**: 5% slower. **Three slots**: no better than two. **Ten
  workers**: no better than eight.

Details are in `coop/REJECTED.md`.

## Validation set

The 6815 control's packet was deleted with `ppm10000-v1`. Of the 12
validation signals, 11 remain: KIC 12644822, the gold planet and the nine
fixture signals.
