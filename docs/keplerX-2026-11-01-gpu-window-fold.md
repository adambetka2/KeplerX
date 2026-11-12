# KeplerX Detector: window fold for long periods

1 November 2026. Changes: `KeplerX/detector/gpu_screening.metal`, `gpu.m`,
`gpu.py`, `fast.py`. Receipts are under
`KeplerX/results/performance-engineering/20261101/window-fold/`.

## Why

The 31 October development-quick run (1,750 cases, 68 min, 8 workers) was
GPU-bound. One process alone saturates the M1 Pro GPU, and workers spent 66%
of their Detector time waiting for it. On a long curve (62k cadences, 270k
trial periods, 1.55 s per pass), periods above 12 d were 24% of the trials and
62% of the GPU time. They admit 9–13 durations each and ran through the gather
`screen` kernel, at about 2–3 times the fold kernel's cost per cell-cadence.

## Cost model

Per pass the GPU cost is a sum over periods of N times c(P), where N is the
number of cadences. The fold kernel histograms cadences with threadgroup
atomics:

    c ≈ (1 + 2m) a + l / G

where m is the number of admitted durations, a the cost of one threadgroup
atomic integer, l the cost of loading one cadence (time plus 8m bytes), and
G the number of periods sharing that load. Short periods fit 32 periods per
group and run at this floor. From about 6 d a whole histogram (nbins × (1 + 2m)
integers) no longer fits 7,680 threadgroup integers for more than one or two
periods, and above that the gather screen took over.

## Window fold

A tile holds G consecutive periods and the bin window [b0, b0 + R). For epoch
k, every cadence in the window's bins of any of its periods lies in

    [k Pmin + b0 wmin, k Pmax + (b0 + R) wmax)

The tile reads the padded index ranges [L_k, min(H_k, L_{k+1})) from the time
table. L_k and H_k grow with k, so these ranges are disjoint and cover every
such cadence once: a cadence i in some [L_k, H_k) lies in the range of the
largest k* with L_{k*} ≤ i. Each visited cadence is classified for every
period by `interior_bin` / `fold_bin` (the fold kernel's arithmetic), and its
integers are added to the window's histogram. Bins, flags and integer sums
are therefore those of the fold kernel, and integer sums do not depend on
order.

## Threshold-aware reduction

With K = T/P cadences per bin, a long-period bin holds only 7–18 cadences, so
reducing every (bin, duration) to the four-field Top cost as much as
accumulating it (about 160 ps per item at 75–200 d). The CPU resolve uses a
cell only when its largest upper bound U exceeds the threshold τ. For any
other cell:

- low only enters floor = max(τ, max over cells of low); low ≤ U ≤ τ cannot
  raise it;
- second and arg are read only for candidate cells.

So the kernels now receive τ (the largest float32 at most τ). A cell with
U ≤ τ reports (U, 0, U, its bin): still bounds (0 bounds max(0, MES) from
below, U bounds every other bin from above), found with one rsqrt and one
`simd_max` per 32 bins. Only cells above τ (about 1% of cells) run the full
SIMD Top reduction. The fold kernel's reduction was serial per cell and is now
one SIMD group per cell. `Context.screen` keeps every bound by default
(τ = −∞).

## Measurements

Shapes (paired, chunk-interleaved device time against HEAD d3c77d657, three
long first passes of 47–64k cadences):

| Configuration | Device time vs HEAD |
| --- | ---: |
| 15 KB histogram, 256 threads, R 32 (64 from 3,072 bins), fold groups of ≥ 2 periods | **0.767** |
| same, fold groups of ≥ 1 period | 0.811 |
| same, R 32 everywhere | 0.820 |
| same, R 64 everywhere | 0.831 |
| same, R 128 above 3,072 bins | 0.883 |
| 30 KB histogram, 512 threads | 1.03 |

Per period range (best configuration): 3.9–9 d 0.71, 7–13 d 0.62,
11–21 d 0.63, 18–38 d 0.67, 33–78 d 0.73, 72–200 d 0.83; below 4 d unchanged.
A 15 KB histogram leaves room for two threadgroups per core; the 30 KB one
allows one.

Exactness on real data: 50 development cases (the ten-case spread of
the 31 October development-quick run and 40 uninjected originals) ran the full iterative Detector
search under HEAD and under the new kernels. Candidates, every searched
detection, candidate flags and final masks are bitwise identical in all 50.
Summed GPU device time fell from 100.8 s to 74.2 s (−26%). The test suite
(260 tests) passes.

## Remaining cost

Window fold decomposition by timing-only variants (one long first pass):

| Range | Total | Setup and epoch loop | Membership | Word loads | Atomics | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 75–200 d | 0.275 s | 0.079 | 0.037 | 0.074 | 0.02 | 0.058 |
| 20–36 d | 0.128 s | 0.021 | 0.054 | 0.018 | 0.022 | 0.013 |
| 12–20 d | 0.114 s | 0.016 | 0.054 | 0.014 | 0.020 | 0.010 |
| 8–12 d | 0.094 s | 0.010 | 0.056 | 0.001 | 0.016 | 0.011 |

In the short-period fold (P < 4 d, 0.274 s), loading cadences costs about
0.045 s and the 32-period inner loop the rest. The device time of three
long passes is now 22% below 4 d, 19% at 4–13 d, 35% at 13–75 d and 25% at
75–200 d.

## Hardware counters

Xcode's GPU counters record from the command line (`xcrun xctrace record
--template 'Metal System Trace' --instrument 'Metal GPU Counters'`). Mean
values over busy samples, production kernels, one long first pass (62k
cadences) per period range:

| Range | ALU limiter | Threadgroup load limiter | Buffer read limiter | Last-level cache limiter | MMU limiter | Occupancy | Read bandwidth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| < 4 d (fold) | 74% | 27% | 11% | 28% | 7% | 32% | 22 GB/s |
| 4–13 d | 68% | 17% | 18% | 33% | 21% | 31% | 66 GB/s |
| 13–75 d (window) | 55% | 10% | 30% | 39% | 34% | 28% | 104 GB/s |
| 75–200 d (window) | 42% | 4% | 41% | 44% | 41% | 24% | 103 GB/s |

The short-period fold is ALU-bound, with F32 at only 13%, so integer,
conversion and address work dominate. Long-period tiles have no limiter
above 45%. They are latency-bound with few work items per tile.

Per integer added to a histogram, the ranges cost 3.8 ps (< 4 d), 5.2 ps
(4–13 d), 6.9 ps (13–75 d) and 10.9 ps (75–200 d). Three long passes add
5.85e11 integers. At the short fold's rate they would take 2.22 s instead
of 3.46 s; this is the ceiling for kernel work at mid and long periods.

Between consecutive passes, 20–88% of a curve's cadences change their
numerators and denominators. Later passes re-whiten every segment that
holds a masked transit. In six cases, between none and 80% of cadences are
unchanged, and the median change is 0.02–0.6 σ. A screen of the changed
cadences alone would save little, and keeping per-bin sums between passes
would need about 19 GB per pass.

## Batch

A random slice of 240 development-quick cases ran through the batch runner
on 8 workers. Old code (d3c77d657, scratch export) took 556 s; the window
fold and the CPU changes below took 456 s (−18%). That run also carried the
skip bound below, which its guard disabled on these real clocks.
Summed GPU seconds, including queueing, fell from 927 to 692. All 240
detection files (candidates, searched detections, grids) and all 240
vetting files (every metric and error) are identical. Workers still spent
65% of Detector time waiting for the GPU (68% before), so the run remains
GPU-bound. Receipts are in
`KeplerX/results/performance-engineering/20261101/batch-benchmark/`.

## Rejected

- **Cauchy–Schwarz skip bound for long periods** (built in 75d6d9235,
  removed). For any bin, MES ≤ sqrt(Σ over its cadences of max(0, n)²/d).
  With one cadence per bin and epoch, this is at most sqrt(Σ_k z_k²), where
  z_k is the largest SES+ over a sub-window's cadence span at epoch k. On a
  long pass this proves 87% of 75–200 d tiles below 7.1 (8-bin sub-windows,
  4-cadence block maxima). But on real Kepler clocks the cadence spacing
  varies from 0.020428 to 0.020434 d, while long-period bin widths lie within
  4e-6 d of the exposure, so a bin can hold two cadences in one epoch. The
  provable bound doubles the terms and proves only 36–58%. Its check also
  reads K·R·m/BLK block maxima per tile, about six times the fold's own
  visits at long periods, where few epochs make the fold cheap. Paired, it
  made 72–200 d 2.5 times slower. A one-value envelope over durations is
  cheap but too loose. Sign-free bounds of this kind grow as sqrt(K) and
  cannot help below ~70 d.
- **Removing fallback divergence.** interior_bin's margin grows as
  2^-20 P, so up to 36% of SIMD steps at 150 d would run fold_bin. A
  timing-only variant with a near-zero margin is 0.998 of production
  overall and 0.970 at 75–200 d. An exact single-rounding phase
  (t − (kN + b)w in one fused multiply-add) would therefore gain at most 3%
  at long periods.
- **Blocked word layout** (32-cadence blocks holding all their duration
  words, against one 1 MB plane per duration pair). Bitwise identical,
  1.001 of production. The MMU limiter does not come from page scatter.
- **Window threadgroups of 512 or 1,024 threads.** 1.27 and 1.17 times
  slower: a long-period tile has about 30 epoch items, so extra SIMD groups
  idle.
- **Unrolling the per-period membership loop by 2 or 4** (two interior_bin
  chains interleaved): 16% and 40% slower over all ranges; the extra
  registers cost more occupancy than the instruction-level parallelism gains.
- **Staged screen.** One thread per (period, bin) with each epoch's cadence
  window copied to threadgroup memory. It was exact, but 1.6 times slower than
  the window fold: two barriers per epoch with one threadgroup per core.
- **Duration chunks of 4–8 per tile.** These cut bytes per visit but double
  membership tests and reductions. They were 7–21% slower.
- **Two-level window maximum before per-cell reduction.** Within noise.
- **A shared lower envelope for the denominators**, which would need
  1 + m + 1 integers per bin instead of 1 + 2m. The denominator profiles of
  different durations differ by a median factor of 1.2–1.4, so bounds would
  loosen by 20–40%.

## CPU: period sampling and grids

- Upstream's `kep_period_sampling` is a Python loop over 270,000–620,000
  periods, run in every Dump construction, so twice per case: 0.3 s each. Its
  recurrence (`d_dur = d · 13/24 · pow(P/365, 1/3) · pow(ρ, −1/3)`, then
  `P += 4 d_dur (P / baseline)`) uses only IEEE operations and libm `pow`,
  the function CPython and NumPy scalars call. `kernels.c v3_period_sampling`
  repeats it operation for operation without contraction. The compiled
  engine uses it while constructing a Dump; the vendored file is unchanged.
  It is bitwise upstream on six baseline/density/step settings
  (`tests/test_compiled_search.py`), and `make_dump` fell from 0.32 s to 0.04 s.
- `search_grid` (a periods × durations boolean matrix) was built twice per
  pass with identical arguments. It is now kept per argument content,
  read-only.
