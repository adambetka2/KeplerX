# V3: original stock programs versus current main

28 October 2026. Current production commit: `3077ee8cf`.
Superseded the same evening: production is now main at `c688a328b`, measured in
[the screen kernel note](keplerX-2026-10-28-gpu-screening-kernel.md). On the six controls it is
3.26 s with eight whitening threads and 5.31 s with one, and the ten-worker 90/10
probe takes 26.5 s against 35.8 s.
The campaign model is **406,000 detector runs, with actual vetting on roughly
10% of curves**. Applying the full positive-control pipeline to all 406,000
curves was the wrong forecast for that model. The earlier 25-day illustration
used an all-positive cohort and must not be read as this campaign's ETA.

## System comparison

Times are seconds. Stock means original upstream/native calculations, not the
already optimized compiled baseline used for incremental engineering ABBA.

| System and measured workload | Original stock | Current production | Recorded stock/current ratio |
| --- | ---: | ---: | ---: |
| Detector: mean of six identical saved controls, 18 total passes | 726.60 | 7.24 | 100.3× |
| Vetter: one long positive control, one signal, complete metrics | 30.29 | 4.66 | 6.50× |
| Localizator: one signal, one quarter, warm localization | 0.979 | 0.702 | 1.40× |
| Photometry ingestion/preparation | No stock/current timing comparison | Inputs already prepared | Not claimed |
| Calibration/sample construction and BOI bookkeeping | No stock/current timing comparison | Not the per-curve detection benchmark | Not claimed |

The detector's current replay was measured once per control on one worker and
one numerical thread. Every candidate, every search-pass result, and final
mask agreed with the saved native reference. Stock profiles have nested timers
and save their first-pass arrays; current timing excludes reference recomputation
and imports. They are historical versus current observations, not a fresh paired
ABBA. That distinction particularly affects the tiny short-curve examples.
The six-control mean is not a population-weighted mean.

Vetter uses the identical prepared-packet SHA and detector candidate in
both versions; the entire saved result dictionaries match exactly. Stock's
30.29 s is the stage-body record, not the 34.58 s fresh-process launcher time.
Current 4.66 s is the mean of two stage observations in the complete-CLI ABBA.

Localization's native warm mean is from the matched PRF cache experiment.
Its paired optimized mean was 0.770 s, a 1.27× speedup
(21.3% less elapsed, with exact metrics and pixel arrays). The latest two
complete-pipeline stage observations averaged 0.702 s and retain
the same result dictionary. The historical/current 1.40× ratio includes timing
context differences; the isolated paired result is the stronger attribution.
The older 6.69 s cold localization record includes a different clock and cold
setup and is not used to claim a kernel speedup.

## Detector controls, original stock to current

| Control | Detector passes | Final candidates | Stock seconds | Current seconds | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2a69` | 1 | 0 | 0.748 | 0.132 | 5.7× |
| `56c4` | 1 | 0 | 460.063 | 4.714 | 97.6× |
| `6815` | 2 | 1 | 900.328 | 9.026 | 99.8× |
| `737f` | 2 | 1 | 767.452 | 8.070 | 95.1× |
| `fc6d` | 7 | 11 | 5.074 | 0.353 | 14.4× |
| `d8ba` | 5 | 4 | 2225.931 | 21.168 | 105.2× |
| Arithmetic mean | — | — | 726.600 | 7.244 | 100.3× |

The separate real KIC 12644822 stock check recorded 397.54 s detection but was
capped at one pass; current detection on that control averages 9.99 s for two
passes. Those totals are useful context but are not used as a matched-work
speedup. Its one-signal vetting is comparable and appears above.

## A measured ten-worker 90/10 workload

Two complete CLI probes used nine copies of long null control `56c4` and one
copy of long single-candidate control `6815`. All ten run detection; only the
positive case performs numerical vetting. Empty vetting receipts for nulls are
included in runner overhead. No localization requests were supplied. Both
probes use ten workers, one numerical thread per worker, nice 19, and the same
current source, manifest and configuration. No optimization experiment runs
concurrently. Each new CLI creates its worker pool; this is not a warmed
steady-state measurement over thousands of distinct curves.

| Quantity | Repeat 1 | Repeat 2 | Mean |
| --- | ---: | ---: | ---: |
| Ten-case complete elapsed seconds | 41.80 | 39.73 | 40.76 |
| Total process CPU seconds | 70.69 | 69.08 | 69.89 |
| Shared GPU service seconds | 25.04 | 22.37 | 23.71 |

Direct measured-batch extrapolation:

`406000 / 10 * 40.760615 / 86400 = 19.15 days`.

That is **about 19 days for this fixed long-curve 90/10 scenario**, at continuous
operation. The two repeats individually extrapolate to
18.67–19.64 days; this is a repeat range, not a confidence interval.
Continuous long-lived workers could amortize startup and end-of-batch idle time
more effectively. Distinct real inputs, shorter curves, more candidates,
I/O and device contention could change the result in either direction.

For original stock, the saved detector CPU measurements for this mix are:

`0.9 * 452.324522 + 0.1 * 879.708720 = 495.062942 CPU seconds/curve`.

Adding 40,600 vettings using the original positive-control process measurement
of 34.264760 CPU seconds gives **234.2 days under ideal ten-core
CPU scaling**. This is an optimistic capacity estimate, not a measured native
ten-worker campaign duration. It uses the KIC 12644822 native vetting cost as
a proxy for one signal on each selected long curve; actual selected signals
vary. The M1's unequal core speeds, contention and sustained throughput can
make actual stock duration longer. Detector time dominates this estimate.

Thus the illustrative comparison is **roughly 234 days stock versus 19 days
current**, with different evidence strength for the two projections. The
single-worker detector's roughly 100× gain does not imply 100× throughput with
ten workers sharing one GPU.

This forecast excludes photometry preparation and pixel localization; the
fraction receiving localization, number of quarters and number of signals per
selected curve are not specified. It is a controlled software-workload
projection, not a measured ETA for the full 406,000-object population.

## Receipts

All paths below are relative to the repository root.

- Original detector goldens and their preserved-file index:
  `KeplerX/results/performance-engineering/20261028/astra-3seconds/native-goldens/`.
- Current detector measurements and source hashes:
  `KeplerX/results/performance-engineering/20261028/astra-3seconds/stage-report-detector-summary.json`;
  harness `stage_report_replay.py` in the same directory.
- Original Vetter record:
  `KeplerX/results/software-checks/runner-native-v1/output/control-1e25d3f07e72/vetting/run.json`.
- Original and optimized warm localization pairs:
  `KeplerX/results/performance-engineering/20261027/prf-product-probe.json`.
- Latest complete-pipeline stage averages:
  `KeplerX/results/performance-engineering/20261028/astra-3seconds/complete-cli-abba-1/{1,2}-candidate/batch.json`.
- Mixed ten-worker measurements, pinned manifest/configuration and per-case stages:
  `KeplerX/results/performance-engineering/20261028/astra-3seconds/mixed90-10-report.json`;
  `mixed90-10.jsonl` and `measure_mixed_report.py` in the same directory.

These bounded runs only establish the requested report; optimization work
remains stopped. The validated production changes remain on main.
