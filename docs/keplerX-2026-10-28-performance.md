# V3 performance toward three seconds

The three-second target is **not achieved**. This checkpoint removes provably
unused detector work while retaining native scientific outputs. One-worker
complete CLI time fell from 17.32 to 15.74 seconds. The first ten-worker ABBA
shows 9.05% less process CPU but only 1.46% less elapsed time, smaller than its
run-to-run variation. It does not establish a ten-worker elapsed-time gain.

## Fixed-work measurements

Machine: Apple M1 Pro, 10 CPU cores, 16 GiB unified memory. Baseline: main commit
`17c0acff3`, including the preceding stable-rank optimization. Candidate: the
search-only changes to `detector/compiled_search.py` and `detector/program.py`.

Each case is the same complete prepared KIC 12644822 positive control: 59,829
cadences, 137,312 trial periods, 26 original durations, two detection passes,
one candidate, full vetting, native Q1 pixel localization and BOI assembly.
The ten-case cohort repeats that control under ten distinct case IDs. This
measures software concurrency on a declared fixed workload; it is not a
population recovery or heterogeneous-campaign measurement.

ABBA means baseline, candidate, candidate, baseline. Both versions use the same
manifest and configuration, the same worker count, one numerical thread per
worker, nice 19, and the same pinned runtime. External elapsed time includes
CLI imports, coordinator/worker startup, all stages and shutdown. Numerical
experiments used a shared compute lease to avoid competing with each other.

| Workers / cases | Baseline elapsed, s | Candidate elapsed, s | Baseline total CPU, s | Candidate total CPU, s |
| --- | --- | --- | --- | --- |
| 1 / 1, repeats | 17.3977, 17.2502 | 15.7514, 15.7285 | 14.2912, 14.3559 | 12.8606, 12.8429 |
| 1 / 1, means | 17.3239 | 15.7400 | 14.3236 | 12.8518 |
| 10 / 10, repeats | 52.5371, 53.7354 | 50.1419, 54.5818 | 172.9443, 182.8774 | 160.9729, 162.6363 |
| 10 / 10, means | 53.1363 | 52.3619 | 177.9109 | 161.8046 |

The one-worker reductions are 9.14% elapsed and 10.28% CPU. Ten-worker CPU
reduction is 9.05%; elapsed results overlap. No comparison between different
worker counts is presented as an engineering speedup. Dividing ten-case batch
time by ten would be throughput, not single-case latency or a three-second pass.
All eight complete CLI runs retained exactly matching scientific JSON and
mask/pixel array bytes, including every one of the 40 ten-worker case executions.

For 406,000 curves, extrapolating this measured ten-worker throughput gives
`406000 / 10 * 52.361887 / 86400 = 24.61` days for the candidate, versus
24.97 days for the paired baseline, at uninterrupted operation. This is a
conditional workload projection, not a measured campaign ETA. The ten cases
repeat one prepared positive control; population curve lengths, detector pass
counts, candidate counts, pixel coverage, ingestion and availability differ.
Adam clarified that only roughly 10% of the 406,000 curves receive numerical
vetting. The all-positive projection above is therefore not the campaign mix;
see [the stock-to-current system comparison and 90/10 probe](keplerX-2026-10-28-comparison-with-stock-upstream.md).
The 1.46% apparent elapsed gain is too small relative to this probe's variation
to promise a campaign-time saving. No matched ten-worker full-pipeline baseline
was measured for the original uncompiled upstream engine.

Receipts are under `KeplerX/results/performance-engineering/20261028/astra-3seconds/`:

- `complete-cli-abba-1/summary.json` and `complete-cli-abba-10/summary.json`;
- `paired_pipeline.py`, the exact harness, and the frozen `baseline-17c0/` and
  `search-only-production/` source trees;
- `control-{1,10}.jsonl` and `config-{1,10}.json`, the fixed inputs/settings;
- per-run logs, process/memory snapshots, stage latencies and GPU service time.

Both benchmark source digests include Python, C, C++, Objective-C, Metal and
configuration sources. Baseline digest:
`a6e1f8b601553e425905d50ab48f3fce9c167384af39ad7bab0c059ecb76f1dd`.
Candidate digest:
`de4fa9169ed0edc349c64cfb683430da3552369967fc7da0e096838bc94e2036`.
The receipts contain manifest/configuration digests and unchanged-source checks.

## Why skipping this work preserves the result

The upstream duration bank is built using the light-curve baseline, while the
actual period grid is restricted to the configured search range. On the long
controls, five or six bank durations cannot be admitted by any searched period.
The stage now calculates the union using the exact existing `search_grid` and
actual `dur_range`. It skips whitening those rows, retaining every original
period, the complete duration bank for padding, and the original row indices.
Every row consumed by folding, candidate cleaning and single-event vetoes keeps
its original arithmetic. The stage also skips CDPP diagnostics that it never
exports or consumes.

This is private to the compiled stage's search-only mode. Ordinary `FastDump`
callers keep complete whitening arrays and CDPP appends. Partial cache entries
have distinct keys, are never used as complete diagnostics, and are cleared
from the object's public SES/N/D attributes after the search. A subsequent
`calc_ses=False` request rebuilds complete arrays. High-single-event removal,
custom whitening overrides and raise/call/log NumPy policies use the complete
path. `detection.prune_unused_durations = false` selects the original stage path.

Six frozen native controls cover 18 detector passes, including five- and
seven-pass searches. The final path matched native candidate values, every
per-pass search result, all search-consumed SES/N/D bits and all final masks.
For each pass the verifier also recalculated complete public whitening and
checked its full native hashes. Focused tests cover full-bank padding, complete
diagnostic cache isolation, iterative masks/candidates, changed gates and
subsequent cached searches. The final software suite passed all 158 tests,
recorded in `pytest-search-only-final.log`. These controls establish software equivalence,
not population recovery or candidate validation.

## Bottlenecks and different-method experiments

A fresh warm positive-control profile before this change measured 16.60 s:
10.78 s detection, 5.09 s vetting and 0.73 s localization. Two whitening calls
used 6.68 s; transit refinement alone used 2.42 s. This was a diagnostic with
imports preloaded, not the complete-CLI speed comparison above. In the repeated
complete-CLI tests, two searches consumed about 3.04 GPU seconds per case; ten
workers share that device. Reducing CPU work alone cannot remove that service
demand.

The following remain isolated experiments, not production changes:

- Float64 hybrid FFT whitening replaces broad direct convolutions but retains
  templates, windows, wavelets and cadences. On six controls/all 18 passes,
  candidate identities, clocks, durations, iteration counts and masks matched;
  the largest returned MES difference was 1.78e-15. Full whitening bits differ.
  A first-pass ABBA against the earlier full-whitening implementation measured
  approximately 39% less whitening CPU. This is not a measured whole-pipeline
  gain and cannot be added to the production reduction. Receipts:
  `hybrid-firstpass-ratio12.json`, `hybrid-six-native-summary.json`.
- An alternate disk quadrature for nonlinear limb darkening saved 0.75 s on
  the positive control with very small flux differences. It failed broader
  verification: one of nine fits exhausted 12,000 evaluations, several fitted
  parameters shifted materially, and aggregate CPU rose from 1.445 to 1.865 s.
  **Rejected.** See `fitting/DISK-QUADRATURE.md` and its pinned gold receipt.
- Int16 GPU screening saved device time but widened bounds enough to more than
  double exact CPU resolution on the CPU-heavy control. **Rejected as a global
  default.** Duration-striped lanes were also slower.
- Persistent per-pass Metal input buffers retained all bounds/scientific bits
  on 17 edge cases and two captured controls. Serialized search means improved
  only 2.3564 to 2.3284 s and 3.3256 to 3.3125 s. No production promotion: the
  measured opportunity is small and does not address the shared GPU bottleneck.
- Conservative B4 coarse phase bounds with a cheap incumbent passed 17 GPU
  edge fixtures and exact captured search comparisons. The GPU implementation
  nevertheless regressed 26.8% on 56c4 and 33.8% on 6815: the extra geometry
  and reductions cost more than the skipped fine work. **Rejected.** See
  `gpu/coarse-summary.json`. Experiments stopped at Adam's request.

## DeepSeek and MLX lessons applied

V3 currently invokes Metal directly; it does not use MLX. Framework migration
alone would not eliminate its arithmetic. [MLX unified-memory documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
and [custom-kernel guidance](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
explain the shared memory pool and reusable kernel objects. Explicit buffer
copies and dependent CPU/GPU synchronization still cost time.

[DeepGEMM's scheduler](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/gemm.cuh)
and [FlashMLA's kernel analysis](https://github.com/deepseek-ai/FlashMLA/blob/395cb3afbb331fef7b1b21e9340861b9436b42d8/docs/20250422-new-kernel-deep-dive.md)
are useful for organizing work around locality, arithmetic intensity and
register limits. [DeepEP's allocation planner](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/allocator.py)
and [DeepInfer's scheduler](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/scheduler.hpp)
provide patterns for buffer reuse and bounded asynchronous scheduling. These
informed the isolated experiments above. Their NVIDIA-oriented implementation
details are not direct M1 tuning prescriptions. The production gain here comes
from proving which computations the science pipeline never consumes.
