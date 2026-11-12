# KeplerX compiled engine

The compiled classes run the pinned upstream Detector and Vetter code with
their hot loops replaced by kernels that repeat the same floating-point
operations in the same order. Upstream files stay unmodified; the compiled
classes subclass each stage's own upstream `Dump` and `Vetter`.

Since 2 November 2026 they are the only Detector: `config.json` holds the
search's five parameters and no engine or option switches, and the Detector
departs from upstream where KeplerX measured it should (whitening summed by
FFT, segments, templates, period grid, veto, cleaning, aliases; see
[keplerX.md](keplerX.md)). The bit-for-bit argument below now applies to the
golden tests' two upstream references (`upstream_adapter.make_dump`):
`reference="compiled"` runs these kernels with upstream's direct
convolutions, single-event veto and harmonic cleaning, and must return the
same bits as `reference="native"`, upstream's own Python search, on the same
template bank, grid, segments and repeat rule: candidate tables, per-pass
search outputs, search masks and vetting metrics. The compiled harmonic
cleaning (`v3_harmonic_window_max`, `v3_harmonic_dominated`) and the
direct-convolution bank (`v3_ses_bank`) served only those settings and were
removed then; the references call upstream's `clean_tces_of_harmonics` and
the per-duration path.

## What is replaced

| Upstream code | Replacement | Why the result is identical |
| --- | --- | --- |
| `Search_for_TCEs_in_all_Tdur_models` period loop (`np.mod`, three `fast_histogram.histogram1d` calls per duration, `num/sqrt(den)`, `np.max`/`np.argmax`, strict `> max_mes` update) | `kernels.c` `v3_fold_search` | `np.mod(t, P)` for `t >= 0` is `fmod`; its exact value `t - kP` is representable, so `fma(-k, P, t)` returns it. Bins use fast_histogram 0.14's formula, read from its machine code: `(int)(x * (nbins / P))`, with an index equal to `nbins` moved to the last bin. Each bin adds its cadences in time order. The duration gate, `nbins = int(P // texp) + 1` and the update rule are evaluated exactly as upstream. |
| `check_tce_caused_by_single_event` and `np.linspace` peak location | `v3_single_events` | Same remainder arithmetic (NumPy's sign rule; Sterbenz-exact subtraction). Cadences farther than `dur + 1e-6` d from a predicted transit cannot pass the test, since rounding is about `1e-13` d. |
| `running_median_insort` | Rank-count and bitset order statistic `v3_running_kth` | The median is an element of the window. The same reflect padding, warm-up and upper-median indices select the identical double. |
| `scipy.signal.convolve(..., mode="same")` | `fast.convolve_same` | scipy's method choice depends on shapes alone. FFT branch: the same `scipy.fft` calls, with the template spectrum cached. Direct branch: NumPy's correlate calls OpenBLAS `ddot`. The bundled OpenBLAS selects core `armv8` on M1. Its kernel (blocks of four, even/odd lanes unfused, pairwise lane sum, fused tail) is reproduced over the template's nonzero span, at the kept output positions only. A skipped product is ±0, which never changes a lane. |
| `ses.ocwt` (`pywt.swt` "db2", norm, trim_approx) | `native.ocwt` / `v3_swt_level` | pywt 1.9.0 convolves with the zero-upsampled filter. Its wrap-around branch steps one entry at a time, so cost grows as 4^level: 1.9 s for 131,072 samples at level 15. Its binary uses fused multiply-adds only. The replica applies the nonzero taps in pywt's per-branch order (increasing taps, or decreasing taps up to the wrap and then increasing), with `fma`; zero taps add exact zeros. It takes 0.01 s, and matched bit for bit at every length 2²–2¹⁷. |
| Vetter `chi2_tests` | `vetter.accelerated_vetter.FastVetter` | The channel and temporal statistics call `tce_masked_num_den` with identical arguments. It is computed once, with the kernels above, and upstream's arithmetic is applied to each statistic. |

Templates are cached per (depth, duration, padded length), and running-median
ranks are shared across templates of a segment. Both are deterministic inputs.
For direct whitening, one C call performs all levels' convolutions and ordered
weighted sums, reusing temporary storage and avoiding repeated Python/ctypes calls.
The running median reads reflected rank indices directly in C, avoiding a
padded index array and its gathered copy on every window; the selected
samples and window multisets are unchanged.
When padding spans complete reflection cycles, the median initializes their
exact multiplicities directly; no output is requested during that prefix.

The compiled detection stage uses a private search-only whitening path. It
forms the union of durations admitted by the actual period grid and the
unchanged duration gate, then skips only rows outside that union. The original
bank still determines padding and duration indices. CDPP diagnostics are not
computed in this private path because the stage consumes only candidates and
masks. General `Dump` callers retain complete arrays and CDPP; a segment's
search-only whitening is never reused for a complete request (its options differ). Single-event removal,
custom whitening overrides and exceptional NumPy error policies retain the
complete path. See [the fixed-worker measurements](keplerX-2026-10-28-performance.md).

From 28 October to 2 November 2026 that private path whitened a finite
segment's admitted durations in one native call, `v3_ses_bank` (-18.8% with
eight whitening threads on six native controls, bit for bit with
`calculate_ses`; receipts in
`KeplerX/results/performance-engineering/20261028/verify/`). The
Detector now sums its bank by FFT with the same weights, computed by
`v3_mad_pairs` with `calc_var_stat`'s operations (running median of |x|, or
for windows of 15,002 points and more NumPy's `linspace`/`interp`
decimation); wherever `calc_var_stat` would warn, replace zero weights, or
meet a non-finite value, the per-duration Python path runs.

A search's grid is computed once per `Dump` (`SearchGrid`): the trial periods,
longest first, their bin counts, and each period's admitted durations, an
interval of the sorted bank found by three binary searches (`min_dur <= tdur
<= max_dur`, `tdur < P/3`) instead of a periods-by-durations comparison
matrix. Each GPU chunk's dispatch layout is built when the chunk is first
screened and kept on the grid by chunk number, so later passes reuse it.
Whitening is reused the same way: each segment keeps its last input (its
cadences, time and flux bytes, and options) and result, and a pass whose
mask leaves a segment's cadences unchanged takes that result instead of
whitening again. These replace three dictionaries keyed by SHA-1 digests of
whole arrays (grid, GPU layouts, whitening), with the same reuse and bounded
memory (2 November 2026; `tests/test_golden_detector.py` and paired
multi-pass detections identical).

With `candidate_cleaning = "epoch"` the outlier veto and the epoch-aware
cleaning run as one greedy pass over the screen's winners in descending MES
(`candidate_ephemerides.vetted_clean`). Accepted detections sit in a flat
log-period bucket table (a head index per bucket, a next index per
detection), so a winner is compared only with accepted ones in its
same-period window and its 18 harmonic windows, then by the unchanged
`_shared`; a winner that an accepted one suppresses needs no veto, and one
with fewer than 48 transits takes the cheaper veto first. Only accepted
winners suppress and the stable sort keeps the vetted winners' order, so the
survivors are those of vetoing every winner and then cleaning: identical on
18 recorded passes and on 10 paired cases, veto and cleaning 16.4 s -> 0.76 s
on the recorded passes. A noisy curve screened at MES 6.6 passes over 200,000
winners; the comparison of each with every accepted one had cost 5-8 s per
pass (2 November 2026).

`execution.numerical_threads` bounds compiled whitening threads within each
case. Values above one process the existing segments independently, sharing
the deterministic template tables and merging arrays and CDPP values in
the original order. Segment boundaries and per-segment calculations are
unchanged. The native reference retains serial execution.
Paired production whitening of the long control measured 3.81/3.80 s elapsed
on one thread and 2.36/2.17 s on two, with identical arrays and CDPP values.
CPU time increased from about 3.80 s to 4.11–4.40 s under load, so thread and
case-worker counts must be budgeted together.

`native.library()` compiles `kernels.c` once per source digest with
`-O3 -ffp-contract=off -fno-fast-math`. At load it compares the direct
convolution with `np.convolve`, and falls back to NumPy if the OpenBLAS core
or the result differs. Inputs with NaN, non-positive denominators or a
negative threshold take the general (NaN-propagating) branches.

## GPU screening

`detector/gpu_screening.metal` (Metal, safe math mode) bounds each phase bin's
MES from integer sums of quantized numerators and denominators. Each duration
uses power-of-two quantization steps; the bound includes half a step per
cadence, float conversion and the final MES arithmetic. Bin edges
`k P + b P/nbins` are evaluated in double-float arithmetic. Their error is
below `2^-46 max(t)` days (2.3e-11 d over four years), and any cadence
within 2e-10 d of an edge marks its period as uncertain.

Screening tiles cover up to 16 admitted durations per dispatch; periods with
up to five admitted durations use the existing coalesced histogram kernel.
Duration selection and all bound/phase arithmetic are unchanged. Across six
saved native first passes, all three search arrays retained their exact bits.
On the CPU-heavy case, screening GPU time measured 2.50 s before and 2.01 s
after; complete search elapsed time measured 2.66 s and 2.16 s. Other long
controls had smaller gains. Overflow tests exercise both shader paths.

Screen tiles use up to 128 bins and fold groups use 512 threads. This changes
dispatch shapes only. Paired warm first passes on all six controls retained
all three exact search arrays; the four long controls used about 8–9% less
GPU time, with no GPU failures or CPU fallback chunks.

The integer pairs are stored duration-pair-major (two durations per 16-byte
word, one copy per first-duration parity), so a SIMD group's consecutive
cadences read consecutive words; the cadence-major rows made each lane touch
its own cache line. The histogram kernel now takes runs with up to 16
durations whose histograms fit 7,680 threadgroup integers. In the fold
kernel, a cadence well inside its bin is classified by four fused
multiply-adds; others use the double-float classifier
([derivation](keplerX-2026-10-28-phase-fold-fma-theory.md)). The screen's bin-end test
uses the double-float distance: the float distance rounds to the width up to
ulp(w)/2 (~9e-10 d) before the end, which previously dropped such cadences
from both bins without a flag. With that fix the screen and the histogram
kernel produce byte-identical cells and flags on the three long controls.
Paired, chunk-interleaved first passes used 19–29% less GPU time than the
fixed cadence-major shader, with identical cells and flags in every chunk.

Since 1 November 2026 the gather `screen` kernel is replaced by `window_fold`,
the histogram kernel over windows of bins, which visits only each epoch's
cadences of a window; runs whose histogram groups would hold one period use it
too. Both kernels receive the threshold and give cells that cannot exceed it
only their upper bound. Device time per long pass fell 23%, with identical
search results ([window fold](keplerX-2026-11-01-gpu-window-fold.md)).
Since 3 November each SIMD group of a window tile finds 32 epochs' cadence
ranges at once and the tile's threads take its cadences back to back; each
class of tiles reserves the threadgroup histogram it uses, and tiles from
3,072 bins hold 128 bins: 8.5% less device time, cells and flags bitwise
identical ([study time](keplerX-2026-11-03-study-time.md)).

The CPU then reproduces the exact result:
- A period whose upper bound does not exceed the threshold has no winner, exactly as upstream.
- Uncertain periods (about 0.1%) run the exact kernel.
- For every other period, cells whose bound reaches the largest lower bound are resolved exactly. A single bin is summed with `v3_bins_mes` when no other bin can reach that bound; otherwise the whole cell is recomputed.

Exact bin sums traverse epochs over blocks of 64 independent bins, improving
cache reuse while each bin retains its original cadence order and arithmetic.
On the CPU-heavy control, paired search CPU time fell from 2.72/2.74 s to
1.76/1.95 s. Single-worker elapsed time remained GPU-bound near 2.63 s;
the change frees CPU capacity for concurrent cases.

A pass is split into 16,384-period chunks; on the GPU the longest-period
chunk goes as eight blocks of 2,048 periods, which keeps each dispatch's
partial buffer to tens of MB. Workers queue for the shared GPU
(`runtime/cache/gpu.lock`), keeping CPU cores available for whitening and vetting.
At most KEPLERX_GPU_SLOTS (default 3 since 3 November 2026, was 2) chunks
are on the device at once
([study time](keplerX-2026-11-03-study-time.md)). The claim covers device work only: each chunk's layout and period records
are built before it, and a pass's cadence inputs are copied to the device
once per Context and reused by its chunks
([live measurements](keplerX-2026-10-29-edge-cases.md)).
Unsupported inputs and unavailable/failed GPU screening use the exact CPU
kernel. The claim ends immediately after screening, so
another worker can use the GPU during exact CPU resolution. Setting `V3_GPU=0`
disables screening.

Within a worker, one dispatch thread screens up to eight pending chunks while
the main thread resolves completed chunks on the CPU. Chunks execute from
short to long periods to balance the two devices; results are stored in their
original slices, retaining native winner and veto order. The screening buffers
are bounded, and CPU-only execution retains its original chunk order.

Integer sums are bounded using actual bin occupancy. Bins with 4,096 or more
cadences receive an infinite upper bound and exact CPU evaluation, avoiding
int32 overflow even when multiple cadences fall in one epoch's bin.

## Evidence

Since 1 November 2026 KeplerX keeps only golden and end-to-end tests: `tests/test_golden_detector.py` runs whole searches of TrES-2, a quiet lane K dwarf with injected planets and a curve of degenerate segments, the native reference against the compiled one bit for bit; the Detector GPU-screened against CPU-only, bit for bit; and the Detector's FFT whitening against upstream's convolutions, to 1e-12 of each template's largest value. The unit tests that first established the equivalence, now removed, covered:

- `tests/test_compiled_search.py`, on a synthetic Kepler-cadence light curve with gaps, a transit and a strong single event:
  - running median and direct convolution against upstream and NumPy;
  - whitening, bit for bit;
  - the per-period search state at thresholds 7.1, 4 and −1e300 against a loop over upstream `calc_mes`;
  - the full search output at three thresholds;
  - every veto against upstream;
  - a 3-pass `Iterative_TCE_Search`;
  - two native failure modes, which raise identical exceptions;
  - the GPU-screened search against the CPU kernel at three thresholds.
- `tests/test_fast_vetting.py`: every Vetter metric for the 9-signal failing-vetting fixture, including its numerical failure, and exact masked whitening after sparse-gap padding repair.
- `tests/test_runner.py`: worker and process isolation give identical candidates and metrics.
- Golden native runs of 2a69, 56c4, 6815, 737f, fc6d and d8ba (1, 1, 2, 2, 7 and 5 passes) matched exactly: candidate tables, every pass's search output and the final search mask. On case 56c4:
  - the full 280,925-period search matched upstream at thresholds 7.1, 5 and 4;
  - 64,719 and 100,392 vetoes matched at thresholds 7.1 and 5;
  - whitening matched across all 58,342 × 19 arrays.

## Measured runtime

Idle M1 Pro, one core, long case 56c4 (58,342 cadences, 280,925 periods,
19 templates, one pass):

| Part | Native | Compiled |
| --- | --- | --- |
| Whitening (`Calculate_SES_by_Segment`) | 53.9 s | 8.6 s, later 6.6 s |
| Period search, CPU kernel | 406 s | 57 s |
| Period search, GPU screened | — | 10 s (GPU) + 0.7 s (CPU) |
| Single-event vetoes | 15 s | 0.6 s |
| Whole detection pass | 460 s | 90 s (CPU), about 17 s (GPU) |

Vetter: 31 s per signal natively on a long curve, and 11.5 s compiled.
`chi2_tests` falls from 15.7 s to 1.0 s. The remaining lmfit/curve_fit fits
(local morphology, transit, cosine, odd/even: about 9.5 s) are upstream.

The batch runner's `isolation = "worker"` keeps stages in long-lived worker
processes. This avoids several seconds of imports per stage, which dominated
short light curves. Localization rewraps the native Gaia query with the same
cache keys and 30-day expiry in each case's own directory, so native imports
are reused without sharing case caches. The query itself is served by the ESA
Gaia archive (`localizator/gaia_cone_search.py`), which returns VizieR's table exactly. On the two-case real software control,
localization took about 1.1 s per case instead of 3.25 s in a fresh interpreter.

During native joint pixel fitting, each PRF retains three recent native splines
keyed by the exact coordinate bytes and stamp shape. Amplitude differences can
reuse a spline while coordinate checks and model evaluation still run normally.
The temporary method bindings and extra splines are released after fitting.
Paired warm real-control localization used 0.97/0.96 s CPU before and
0.76/0.76 s after, about 22% less, with identical native metrics and pixel arrays.
Synthetic joint-fit fixtures with three and nine repeated quarter maps used
3.81 → 1.81 s and 21.61 → 5.66 s CPU respectively. Native fit state,
covariance and residual array bytes matched exactly.

The further 75% elapsed-time target uses commit `49dc48d5` as its baseline.
Engineering gains must hold the case-worker and numerical-thread counts fixed.
The predeclared 16-case software cohort repeats one real prepared light curve
and its native Q1 pixels. Every case includes detection, vetting, native pixel
localization, BOI evidence and handoffs. External timing covers complete CLI
entry to exit, including coordinator/worker imports and shutdown.
Engineering uses nice 19 and one thread per numerical library while the
concurrent statistical-validation jobs retain their existing priorities.

With one case worker and one whitening thread in both versions, complete CLI
elapsed time fell from 403.38 s to 321.61 s: **20.27% less**. Total process CPU
time fell from 314.93 s to 265.53 s: **15.68% less**. The optimized serial run
has one measurement, so this is not a replicated estimate. Scientific JSON
and mask/pixel array bytes matched exactly. This is the applicable measured
engineering gain against the frozen baseline.

| Version | Case workers | Whitening threads/case | Complete CLI elapsed |
| --- | ---: | ---: | ---: |
| `49dc48d5` | 1 | 1 | 403.38 s, 409.56 s |
| `a82599d2`, serial diagnostic | 1 | 1 | 321.61 s |
| `a82599d2` | 4 | 2 | 101.52 s, 102.67 s |
| `a82599d2`, profile experiment | 4 | 1 | 104.39 s |
| `a82599d2`, profile experiment | 5 | 1 | 103.10 s |

All completed cases matched in scientific JSON and mask/pixel array bytes,
including all 16 cases in both complete CLI repeats.
The four-worker results include increased concurrency: one to four case
workers and one to two whitening threads per case. Their paired mean of
406.47 s to 102.09 s (74.88% less) is a throughput measurement with additional
workers and is excluded from the 75% engineering target. The earlier summary
incorrectly foregrounded that comparison as progress toward the target.
The **75% engineering target is not achieved**. Population recovery and
representative survey performance are unmeasured. Historical runner-only intervals omit
initial coordinator imports and final exit and are labeled separately in the
engineering receipts.

The batch coordinator now imports only its dispatcher. Workers retain their
normal scientific stage imports, and help output retains the complete parser.
Paired import-only interpreter runs took 2.19/1.69 s with all stages and
0.087/0.086 s with the dispatcher alone.

Vetter's direct morphology fits batch each finite-difference gradient in C.
The parameter transforms, representable steps, residual reductions and optimizer
remain identical. Nonfinite evaluations and partial evaluation-budget exhaustion
retain the ordinary path. Paired full vetting of the real software control used
6.96/6.84 s CPU with batching disabled and 6.22/6.21 s enabled, about 10% less,
with identical metrics. Gradient bits, counters and abort state are tested.
Exact float64 saturation tails avoid redundant exponential calls, and gradient
steps retain bound transforms for variables that did not change. A further
paired control measured 6.47/6.42 s CPU before and 6.25/6.13 s after, with
identical metrics; two primitives also matched over a million test inputs.
The objective and its unchanged finite differences now share one C call,
avoiding repeated Python step construction and dispatch. A paired full vetting
control measured 5.81/5.78 s CPU before and 5.58/5.49 s after, about 4.5% less,
with identical metrics. Evaluation counts, signed zeros, large representable
steps, nonfinite failures and partial-budget aborts retain the ordinary behavior.

Local morphology now finds each event's strict time bounds with binary searches
on finite, sorted float64 times, retaining copied fit inputs. Unsupported arrays
retain the original mask path. The maximum time is computed once per call.
One worker and one numerical thread, paired control vetting used
5.62/5.53 s CPU before and 5.08/4.99 s after: **9.7% less CPU time**.
All 357 optimizer states and scientific metrics matched; the nine-signal
failure fixture and native window-boundary/failure tests also matched.
Complete-CLI ABBA on that single positive control used 21.07/20.91 s before
and 20.32/20.26 s after: means **20.99 → 20.29 s, 3.3% less elapsed time**.
Both versions used one worker and one numerical thread, nice 19, while fits
continued. Scientific JSON and mask/pixel array bytes matched exactly.
This single-control measurement does not establish representative campaign speed
or achievement of the approximately 12 s average target.

Direct whitening convolution also uses SIMD for four scattered output positions
on the same side of the array centre. Each vector lane retains one output's
original even/odd sums, scalar edges and fused remainder. Cross-centre groups
and nonfinite inputs retain the previous path. The real control's first pass
used 4.17/4.20 s CPU before and 3.79/3.77 s after: **9.6% less CPU time**.
All SES, numerator, denominator and CDPP bytes matched. The full 143-test suite,
405 additional exceptional-value checks and six native detector cases passed;
all 18 native passes retained exact whitening bytes, search results and masks.
Loading each output's two adjacent products once, then transposing their lanes,
further reduced first-pass control CPU from 4.03/4.09 s to 3.96/3.73 s
(5.4% less against the first SIMD version). The 1,165 primitive checks, full
143-test suite and all 18 native passes retained exact outputs.
Independent exact-bin lookup windows also prepare two rows with SIMD, retaining
the original division, floor and cadence accumulation order. On captured 6815
inputs, paired bin CPU fell from 1.440/1.439 s to 1.405/1.388 s (about 43 ms).
The complete pipeline timing for these convolution and lookup changes is pending.

The lookup's usual single skipped cadence is now peeled with a conditional
increment before its original while loop; longer/dense scans retain that loop.
On the same captured 6815 inputs, paired bin CPU fell from 1.614/1.620 s to
1.304/1.307 s (19.2% less); full CPU resolution fell by about 0.33 s.
The 24 native histogram comparisons and complete captured search arrays matched,
including dense/duplicate cadences and long scans. Worker/thread counts stayed fixed.

The subsequent complete-CLI ABBA, against the unchanged pre-morphology code,
used 23.18/22.82 s before and 20.91/21.46 s after: **23.00 → 21.19 s,
7.9% less elapsed time** (8.9% less process CPU). Both versions used one
worker/thread while the same nine fit workers retained priority. Scientific
JSON and mask/pixel array bytes matched exactly. This remains a single positive
control, with a fresh baseline under that load; it is not a campaign average.

An additional whitening shortcut directly loads adjacent output pairs instead
of transposing them, while the same helper handles arbitrary gaps. Against the
preceding transpose version, paired first-pass control CPU used 4.04/3.88 s
before and 3.76/3.78 s after (4.9% less). The GPU screen similarly peels the
usual single skipped cadence before its unchanged scan. Paired 6815 device
times used 1.889/1.898 s before and 1.818/1.798 s after (4.5% less device time).
Dense/duplicate-edge probes retained exact raw bounds, flags and final search
arrays. These latest changes passed all 143 tests and all 18 native passes;
their additional complete-pipeline timing is pending.

GPU eligibility also guards scaling extremes. A nine-cadence software test
previously selected duration 1 instead of the native winner, duration 0, when
a float32 scale overflowed. Screening now requires normal factors with finite
lower-bound headroom and a denominator floor below every occupied true sum.
Tiny requested thresholds use the existing CPU path. The requested scientific
threshold is unchanged. Eleven native-result regressions cover overflow,
underflow, cancellation and scale/floor boundaries; ordinary inputs retain GPU
screening. All 154 tests and all 18 native passes match after this repair.

The screen uses standard correctly rounded reciprocal square root followed by
multiplication for its conservative float32 bounds. The existing enclosure and
rounding margins still cover the replacement; float64 resolution is unchanged.
Paired 6815 device times fell from a mean 1.7991 s to 1.7614 s (2.1% less).
All bound inequalities, scientific array bytes, 154 tests and 18 native passes
matched. This is device time; its complete-pipeline effect is measured separately.

The subsequent complete-CLI ABBA against the same pre-morphology baseline used
21.45/20.80 s before and 18.72/18.66 s after: means **21.12 → 18.69 s,
11.5% less elapsed time** (11.3% less process CPU). Both versions used one
worker/thread, nice 19, alongside the same two fit workers. Scientific JSON
and mask/pixel bytes matched exactly. An earlier repeat overlapped automatic
Git maintenance and regressed 8.1%; its receipt is retained. The fresh paired
comparison excludes that job on both sides. This is still one positive control,
not a campaign average or achievement of the approximately 12 s average target.

Stable ranking now sorts records containing each value and its original position.
The position breaks ties, retaining NumPy's exact ranks, signed zeros and sorted
bytes. Unsupported arrays and nonfinite values retain NumPy's stable sort.
One-worker control first-pass ABBA used 3.489/3.524 s CPU before and
3.320/3.349 s after: **4.9% less whitening CPU**. The integrated build passes
155 tests and all 18 native detection passes, including the five- and seven-pass
cases. Its complete-pipeline effect is measured separately.

Under the concurrent validation workload on 27 October, paired first-pass
whitening checks measured about 4% less CPU time from batching direct
convolutions, with identical array hashes. The 6815 two-pass replay took
17.34 s wall / 12.27 s CPU and retained its exact native candidate table and
mask. Its 900 s native reference predates the combined later changes;
the full reduction is not attributable to this batching change.

## Harmonic cleaning and degenerate segments

Upstream's cleaning cost grows with the square of the detections in a pass.
The saved 163,994-detection pass of calibration curve 7ce647 took 941 s
upstream and takes 0.07 s compiled, returning the same 7,542-row frame.
`tests/test_harmonics.py` compares adversarial frames (grid periods, exact
harmonics and their neighbouring doubles, tied mes, duplicated rows), exact
window-edge pairs and a dense period grid with upstream; a further 500
fuzzed frames matched, and mutating any window inequality fails the
tests. Such passes came from
degenerate segments, which both engines now drop; see
[the edge-case note](keplerX-2026-10-29-edge-cases.md).

## Native failures

Unadapted native Detector failed during pass-1 whitening on 52 development cases:
- 42: upstream reflect gap filling indexes outside a short segment.
- 8: upstream drops the last cadence when the final segment exceeds 13 days, which breaks a boolean mask.
- 2: `pywt` (odd or single-sample segments).

The adapter now repairs those boundary/padding failures in both native and
compiled modes, retaining real cadences and leaving successful native
calculations unchanged. It keeps upstream's segmentation (split at gaps over
0.5 d after 13 d). All 100 saved development inputs now complete first-pass
whitening with the correct array sizes and finite values. This is a software
check, with no detection or recovery scoring. Compiled Vetter shares the
padding repair. An opt-in `detection.segmentation = "quarters"`, which
passed each Kepler quarter's first usable time as upstream's `sector_dates`,
was never adopted and was removed on 2 November 2026, when the Detector also
began ending a segment at every gap longer than 30 days.
