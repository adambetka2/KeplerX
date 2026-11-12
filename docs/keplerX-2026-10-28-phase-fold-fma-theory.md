# Guarded fused phase classification: theory before engineering

28 October 2026. Baseline main `28bee5bf0` (production kernels unchanged
since `3077ee8cf`). This note precedes any new kernel, benchmark, or numerical
experiment for this idea.
The model below is a falsifiable cost hypothesis, not a measured speedup.

## Selection and rejected alternatives

The primary objective is at least 30% less detector elapsed time at the same
ten-worker workload, with one numerical thread per worker. Also report GPU
service, CPU work, one-worker latency, and memory. Never substitute a kernel
speedup for a detector speedup. Preserve native periods, durations, thresholds,
passes, winners, vetoes, and masks.

FFT whitening does not pass the production-resource gate. In the saved long
null replay, detector elapsed was 4.714 s, CPU was 2.385 s and GPU service was
2.379 s. Ten CPU workers share one GPU. A CPU-only change leaves the device
service demand intact. The older 39% whitening reduction would also give only
0.39 * 6.68 / 10.78 = 24.2% reduction in the earlier one-worker detection
profile, before numerical certification costs. No new FFT experiment follows.

Shared bounds across neighboring periods are also rejected before coding.
For an epoch k and a fixed phase-bin count B, a bin's boundaries are
(k + b/B)P and (k + (b+1)/B)P. Across a period interval of width dP, its
uncertain boundary width grows as k*dP. A numerator bound based on positive
optional samples loses cancellation: for m events and an optional fraction
f, the standardized uncertainty can grow as f*sqrt(m), even for noise.
Short periods have large m. There is no defensible 30% gain from this bound
without an additional mechanism to keep it tight. No kernel is warranted.

The selected experiment replaces always-expensive phase classification in
the short-period `fold` kernel with an inexpensive, certified interior test.
The existing double-float classifier remains the fallback. The long-period
`screen` kernel, quantization and exact CPU resolver initially stay unchanged.

## Arithmetic and correctness

Let u = 2^-24, the binary32 unit roundoff. A stored double-float represents
x as x_h + x_l, with representation error approximately bounded by u^2|x|
(allow additional binary64 rounding when constructing the split). All
arithmetic below is normal finite arithmetic; exceptional ranges fall back.
Use Metal safe math and explicit `fma`; do not enable reassociation/fast math.

For time t, period P and bin width w=P/B, form tentative integers

    k = floor(t_h * inverse_period)
    r_h = fma(-k, P_h, t_h)
    r_l = fma(-k, P_l, t_l)
    b = floor((r_h + r_l) * inverse_width)
    d_h = fma(-b, w_h, r_h)
    d_l = fma(-b, w_l, r_l)
    d_hat = fl(d_h + d_l).

The mathematical distance is d = t - kP - bw. The tentative quotients need
not be correct: only an interior interval certificate can accept them.
Otherwise execute the existing classifier, including its corrections.

Why FMA matters at the bit level: a binary32 significand has 24 bits. A
separate multiplication rounds the product to 24 bits *before* subtracting
two large, nearby numbers. FMA forms the product and sum with one final
rounding; the error scales with the small residual, not the large product.
The low component restores the contribution below the high component's ulp.
We do not require the residual's last bits to match native arithmetic: we
require an interval wholly inside the same native phase bin.

Within the existing support contract, t/P < 4096, B < 2^24, positive normal
P,w and finite split inputs. The float quotient estimate is at most one epoch
off. Hence |t_h-kP_h| <= 2P + O(uT), where T is the last cadence. Standard
one-rounding bounds on the four FMAs and additions give a conservative

    |d_hat - d| <= 4uP + 8uw + 32u^2 T.

This deliberately overcounts the low-part and split errors. The existing
host support requires edge >= 4*2^-46*T = 16u^2*T. Choose

    E = 16u(P_h + w_h) + 4*edge,

with a further upward rounding cushion in its implementation. This covers
the expression above, native edge construction/modulo rounding, and the
existing edge guard. Accept only when

    0 <= b < B,  d_hat > E,  d_hat < w_h - E.

These strict inequalities prove that the cadence is farther than the
existing ambiguity margin from either boundary. All uncertain points,
including t=0, use the original function. Existing flags, integer histogram
sums, MES bounds and exact float64 resolution are then unchanged.

Integer accumulation already uses unsigned 32-bit modular addition. This
experiment changes none of those operations or their quantization. Identical
cadence-to-bin assignments imply identical integer histograms regardless of
thread interleaving; the existing overflow guards still apply. Raw GPU bounds
and flags should match the baseline, not merely final candidate identities.

Before promotion, test this derivation against adversarial boundaries, the
support limits, duplicate times, tiny widths, overflow occupancies, and saved
native controls. A successful test is supporting evidence, not a substitute
for the interval argument.

## Cost model and explicit rejection gates

Let q be the fraction of device time in the changed fold kernel; g its
fraction spent classifying phases; r the guarded classifier's cost relative
to the old one, including fallback divergence. Then

    GPU_new/GPU_old = 1 - q*g*(1-r).

For 30% less device work, q*g*(1-r) must exceed 0.30. For detector wall time
with unaffected fraction h, require

    (1-h)*q*g*(1-r) >= 0.30.

Thus if r=0.20 and h=0.20, q*g must exceed 0.46875. If h=0.40 it must
exceed 0.625. These are requirements, not fitted assumptions.

The existing function expands several compensated additions and products
for every cadence-period pair. Each compensated addition alone has roughly
20 scalar add/sub operations; the common path has three of them plus two
compensated products. The proposed common path has four explicit FMAs,
two quotient estimates, additions, and comparisons. This gives a plausible
classifier operation ratio of 0.15-0.25 before atomics, memory and fallback.
Static source-operation counts do not equal Apple GPU instruction timings.

Under approximately spread phases, the fallback fraction is about 2E/w,
dominated by 32uB. At B=400 this is 0.000763 per lane; a 32-lane group has
at most about 2.44% probability of any fallback under that distribution.
Adversarial phases may all fall back; there is no distribution-free runtime
guarantee. Scientific correctness is independent of this distribution model.

The native period spacing is proportional to P^(4/3), so period density is
proportional to P^(-4/3). For the 0.5-100 d range, periods below 7.5 d comprise

    (0.5^(-1/3)-7.5^(-1/3))/(0.5^(-1/3)-100^(-1/3)) = about 72%.

This is an illustrative range of the short-period folding path, not a
measured kernel-time fraction. Duration gates and histogram capacity set the
actual boundary. The changed path has substantially more phase arithmetic
per cadence than the long-period gather path. A conditional example q=0.80,
g=0.75, r=0.20 predicts 48% less GPU service and 38.4% less detector wall if
h=0.20. It clears the engineering hypothesis gate, but proves no speedup.

Engineering is now justified as a small isolated falsification experiment:
first exactness; then same-input ABBA device/search timings; then, only if
credible, fixed ten-worker detector ABBA with pinned sources. If measured
detector improvement is below 30%, reject this as the requested solution
and derive a new method before engineering that method. Do not promote a
small gain merely because the kernel is faster. No clocks are changed.

## Completion: the bound, the equivalence and the measurements

28 October 2026, after the other runs stopped. The shader under
test is the `fma2` variant: helpers `fold_margin` and `interior_bin` before
`fold`, called before the unchanged `fold_bin`. Sources, harnesses and
receipts are in `KeplerX/results/performance-engineering/20261028/fma-probe/`.

### The bound

Every operation the test trusts is a correctly rounded FMA or addition
(the Metal specification's `fma`; safe math). With round to nearest,
|fl(x) - x| <= u|fl(x)|, plus at most 2^-126 absolute if a subnormal is
flushed. Writing e1..e5 for the five roundings,

    d = D + e1 + e2 + e3 + e4 + e5,
    D = (t_h + t_l) - k(P_h + P_l) - b(w_h + w_l),

with |e1| <= u|r_h|, |e2| <= u|r_l|, |e3| <= u|d_h|, |e4| <= u|d_l| and
|e5| <= u|d|, where d_h and d_l are the two FMA results. D is the exact
distance after the start of bin b of epoch k for the same double-float
inputs that `fold_bin` reads. The magnitudes need bounding only on
acceptance (0 <= b < nb, d < w_h). With |t_l| <= u t_h, |P_l| <= u P_h,
|w_l| <= u w_h, k <= 4096 and T < 4096 P (the existing epoch limit):
|r_l|, |d_l| <= 8195 u P_h; |d_h| <= w_h + 8195 u P_h; and
|r_h| <= nb w_h + 8196 u P_h <= P_h (1 + 8200 u). Hence

    |d - D| <= u (P_h + 2 w_h) + 32800 u^2 P_h <= 1.002 u P_h + 2 u w_h.

No accuracy is required of `1/P_h`, `nb/P_h` or `floor`: a wrong tentative
(k, b) can only fail the test. (On accepted pairs r_h and d_h are in fact
exact, so the true error is near u w_h; the bound does not use this.)

`fold_margin` returns at least 16u(P_h + w_h) + 4 edge (its final factor
1 + 2^-20 covers its three roundings), and forming w_h - margin adds at most
u w_h. On acceptance, therefore,

    D > 14.99 u P_h + 13.99 u w_h + 4 edge,
    (w_h + w_l) - D > 14.99 u P_h + 11.99 u w_h + 4 edge.

The earlier conservative expression 4uP + 8uw + 32u^2 T also lies inside E.

### Why outputs are identical

`fold_bin`'s existing contract (`gpu.py`): its double-float distances differ
from D by at most 2^-46 T <= edge/4 plus a relative u of the value, and its
float estimates of k and b are within one and corrected. The last bin ends at
kP + nb w, within about 2^-47 P of (k + 1)P, far below `edge`. A cadence with
D more than 2 edge inside both edges of bin b (hence also inside both epoch
ends) therefore receives bin b from `fold_bin` and is not flagged. The
acceptance region lies inside that region by more than 1.9 edge + 14 u P_h.
So `interior_bin` only reproduces an unflagged `fold_bin` answer, and every
other cadence runs `fold_bin` as before. Each bin receives the same integers;
unsigned addition is order-independent; each period's flag is the same OR
over the same flagged cadences. Raw cells and flags are identical by
construction, and agreement with native float64 bins remains `fold_bin`'s
unchanged contract.

### Verification

- Per-pair GPU check (`verify_classifier.py`, `classify.m`). The helpers are
  extracted verbatim from the `fma2` shader, compiled with `MTLMathModeSafe`
  as in `gpu.m`. Every first-pass cadence of controls 56c4, 6815 and d8ba is
  checked against every period of its grid, not only fold-path periods:
  41,689,198,413 pairs, of which 41,648,306,041 were accepted (99.90%).
  There were **0 violations**, meaning no accepted pair for which `fold_bin`
  returned another bin or set a flag.
- Adversarial edges: float64 kP + bP/nb and epoch ends for 66 periods at each
  of three cadences (0.0204, 0.00068, 0.0833 d), epochs up to 4,000, offsets
  from 0 through 1e-14 to 0.5 w and 0.5 to 2 margins on both sides. This
  covers 35,900,898 pairs with 0 violations. No accepted pair differs from
  native min(int(mod(t, P) * (nb / P)), nb - 1), and no unflagged `fold_bin`
  pair does either. Fallback here is 1.4 to 4.3% by construction.
- Complete GPU screens of the three controls' first passes gave cells and
  flags with SHA-256 identical to production in all six repeats.

### Measured cost terms

GPU device seconds per first-pass screen, means of six repeats (three for
`free` and `nofold`). `free` replaces the classifier with a cheap, wrong
formula and `nofold` returns at once; both are for timing only.

| Control | base | fma2 | free | nofold | q | g | q*g | r | GPU saved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 56c4 | 2.151 | 1.844 | 1.667 | 1.061 | 0.51 | 0.44 | 0.23 | 0.37 | 14.3% |
| 6815 | 1.791 | 1.619 | 1.490 | 1.141 | 0.36 | 0.46 | 0.17 | 0.43 | 9.6% |
| d8ba | 1.799 | 1.563 | 1.495 | 1.176 | 0.35 | 0.49 | 0.17 | 0.23 | 13.1% |

q = (base - nofold)/base, g = (base - free)/(base - nofold) and
r = (fma2 - free)/(base - free). The receipt is `cost-terms.json`.

The gate fails. Its requirement q*g > 0.47 is missed by roughly a factor of
two: even a free classifier (r = 0) would save only 17 to 23% of GPU service.
The measured r of 0.23 to 0.43 is above the 0.15 to 0.25 static estimate,
because the fallback branch and the loads remain.

**Verdict.** The guarded classifier is exact. It can be adopted as an
increment worth roughly 10 to 14% less GPU service. By the rule above it is
not the 30% solution. The remaining device time is the long-period `screen`
kernel, 1.06 to 1.18 s (50 to 65%), plus `fold`'s histogram atomics and
loads, 0.31 to 0.61 s. The next method must reduce the screen kernel's
epoch-by-cadence scan.

### Detector replay

Six-control detector replays (`packed3s/replay.py`), interleaved B C F F C B
at each thread count, every one of the 36 runs native-equal. B is main
`28bee5bf0`, C is production with the whitening bank, F is C plus the `fma2`
shader. Means of the six-control mean wall time, in seconds:

| Whitening threads | B | C | F | C vs B | F vs B | F vs C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 5.876 | 4.772 | 4.349 | -18.8% | -26.0% | -8.9% |
| 1 | 6.911 | 6.445 | 6.170 | -6.7% | -10.7% | -4.3% |

Per-case GPU service fell from 3.01 to 2.70 s with one thread and from 3.29
to 2.84 s with eight. These are one-worker latencies; ten-worker throughput
was not measured. Receipts: `verify/abba-*.json`, `abba-means.json`.
