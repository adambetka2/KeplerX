# Fast TRICERATOPS

`fast_triceratops/` makes the pinned TRICERATOPS runtime (`d20875c`) and
`TRICERATOPS_KEPLER_V3` run faster on the Mac without changing their science.
Its exact mode returns the same `calc_probs` outputs, bit for bit: every
scenario's lnZ, the probability table, FPP and NFPP. The exact static engine
was selected for queued campaigns on October 26; see the
[adoption record](../results/boi-validation/engine-adoption-20261026.json).

## Guarantees

- **Exact mode** (`accelerated_v3()`). Bitwise equal to upstream for the
  outputs that `calc_probs` keeps. Each model flux still comes from
  pytransit's own compiled kernel, because pytransit compiles its point model
  with fastmath and a re-inlined copy can round differently.
- **Certified mode** (`accelerated_v3(certified=True)`). Also drops draws
  whose weight is below e^-50 of the scenario's largest. With N draws, this
  changes the evidence by at most N e^-50 relative (2e-16 for N = 10^6). That
  is below one ulp of lnZ, so lnZ changes only when it lies within 2e-16 of a
  rounding boundary, about one scenario in 10^4. Every comparison so far was
  bitwise equal. The bound that drops a draw assumes pytransit's measured
  accuracy (step 4), so adopt this mode separately from exact mode.
- Of each scenario's `res`, `calc_probs` keeps only the first row and lnZ. The
  other 99 "best" draws can differ and are discarded upstream.

## Static experimental engine

`engine()` now runs the verified accelerations as code in the copied scenario
functions. It no longer rebuilds those functions at runtime. The copy keeps
the original draw order, priors and attempted-draw denominator, uses the exact
joint likelihood and pruning, shares deterministic flux calculations, and
allocates constant likelihood inputs only for selected rows. Radius and
temperature share their identical spline basis and clipping pass. The fitted
photometry and installed/frozen validation code remain unchanged.

The source-branch structure of the copied upstream functions is retained with
explicit inherited complexity exceptions. New helpers pass the limit of ten.
The copied engine checks the pinned upstream modules and the Mac FITPACK
binary used to qualify its FMA arithmetic.

The [workflow map and runner](fast-triceratops-workflow.md) cover the complete
path and separate experimental repeat directories. The
[native fitting experiment](fast-triceratops-native.md) accelerates orbit
evaluation without rerunning an accepted fit. The native kernel remains a
separate experiment; queued probability campaigns use the exact static engine.

## How it works

1. **Fused likelihood.** Rows go through pytransit in chunks, and a strict
   numba kernel reproduces numpy's pairwise row sum. Eclipsing binaries
   compute their 25-point secondary first and model the primary only if it
   passes the 1.5 sigma veto.
2. **Exact pruning.** Upstream keeps a scenario's lnL only for
   `argsort()[0]` and `_log_mean_exp`, whose `logsumexp` sees each row only
   through `exp(logw - max)`. That value is exactly 0 for gaps beyond 745.13.
   Each row's cost is a residual plus nonnegative per-point terms. The terms
   at a few probed points, modelled exactly, give a lower bound. Rounds probe
   2, 6 and 24 points, in order of information, for rows still within 760 of
   the best exact cost. Rows above the bound are deferred and return a
   sentinel cost. `fast_log_mean_exp` adds each deferred row's prior to its
   bound and models exactly any row that could still carry weight. A tied
   maximum disables deferral for that call.
3. **Finite proof.** A deferred row must stay in `logsumexp`'s finite set. On
   adversarial grids, pytransit's point model is non-finite only near the
   contacts z = 1 ± k and near z = 0. `kernels.finite_row` replays
   pytransit's window and exposure samples and requires every sample to stay
   1e-7 from those values. Rows it cannot prove are modelled exactly; this
   is 0.1–10% of rows, mostly binaries with e ≥ 0.95.
4. **Certified bound.** `kernels.cheap_row` bounds a row's cost without
   pytransit. Over each half of an exposure, z stays within the orbit's speed
   bound of its midpoint. The exact occulted flux decreases with z and lies
   between the uniform-disc overlap times the limb-darkened intensity at the
   overlap's outer and inner radius. Against an adaptive quadrature of the
   exact occultation, pytransit's model is within 6.4e-8 outside its two
   special branches. Inside them, |z - k| < 1e-4 (z + k) and
   |z + k - 1| < 1e-4, it is within 1.3e-4. The bound allows 2e-7 and 5e-4
   (`ETA`, `ETA_BAND`). Rows are modelled in order of the bound's midpoint
   guess while their bound lies within 50 of the best exact cost.
5. **Reference scenario rebuild.** The `accelerated_v3()` wrapper recompiles
   the 14 V3 scenario functions from
   their hash-pinned source with four exact substitutions:
   - the evidence call;
   - a partial top-100 ranking in place of the full argsort;
   - a per-pair limb-darkening lookup in STP and SEB in place of a loop over
     every draw;
   - a vectorized limb-darkening lookup in BTP and BEB in place of a loop over
     every background star.

## Evidence

- Bitwise equality, stock against both modes, on every third V3 campaign on
  the Mac: 59 of 59 runnable campaigns at 20k draws (the other 3 stop in
  `post_boi`'s scene check before any likelihood). Their single-thread CPU
  speedup at this size is median 1.45x exact and 2.34x certified.
- Bitwise equality with two numba threads.
- Saved production repeats (10^6 draws, `bench reproduce`): 15 repeats of 6
  campaigns (11 exact, 4 certified), including two M/K cases. Every
  scenario's lnZ, `scenarios.csv`, FPP and NFPP match the campaign's own files
  byte for byte.
- A full V3 `calc_probs` on the Kepler-10 fixture with a declared neighbour
  (18 scenarios) in both modes.
- Pruned evidence against upstream under none, flat, noisy and adversarial
  priors.
- Finite proof:
  - 2.37 million samples at and beside every edge are all finite where the
    proof declares them safe.
  - 96,000 mixed rows: no proven row was ever non-finite in pytransit.
- Certified bound: in 900,000 randomized rows over depths, noise, exposures,
  sub-sample counts, hosts and the test generator's special rows, the bound
  never exceeded the cost.
- Final folder checks: 101 passed in the stock runtime, with 53 tests skipped
  for unsupported V3/native contexts; all 122 applicable V3 tests passed.
  Stock-only source guards are exercised in the stock runtime.

## Measured speed

These are single-thread CPU times at nice 19, measured beside the validation
campaigns. Throughput at the same total CPU budget remains unmeasured.
Reference-wrapper V3 repeats (`bench v3`, first final seed, bitwise for both modes,
logged in `fast_triceratops/hill-climb.csv`):

| Case (depth/bin noise) | Draws | Stock | Exact | Certified |
| --- | --- | --- | --- | --- |
| BOI 2100325529865507712 (95) | 50k | 30.7 s | 4.3 s (7.2x) | 2.1 s (14.7x) |
| M/K 2104058096963805696-4 (2.2) | 200k | 10.8 s | 6.1 s (1.8x) | 4.6 s (2.4x) |
| BOI 2127169041106093184 (4.5) | 200k | 17.9 s | 10.8 s (1.7x) | 4.7 s (3.8x) |
| BOI 2076383384993192576 (18) | 10^6 | 28.1 s | 13.3 s (2.1x) | 9.1 s (3.1x) |

At 10^6 draws on one thread a repeat peaks at about 0.7 GB in either mode,
against 1.0 GB for the stock V3 route. In the certified 10^6-draw repeat above,
pytransit takes 17% of the time and the bound 13%. Upstream's own sampling,
priors and splines take most of the rest, and no mode changes them.

### Static engine closeout, October 25

The final copied engine was replayed on BOI 2076383384993192576 at the original
1,000,000 attempted draws per scenario, using two original independent seeds.
Both the stock replay and the copied engine matched the saved campaign's lnZ
bits, complete scenario and star tables, FPP and NFPP. The copied engine also
matched the stock replay's serialized target object. All source campaigns
remained read-only.

| Complete quarter calculation | Stock CPU | Engine CPU | Speedup |
| --- | --- | --- | --- |
| First seed, cold compilation | 44.44 s | 24.51 s | 1.81x |
| Second seed, warm worker | 31.73 s | 14.57 s | 2.18x |

Including input/source verification, saved-evidence comparison and output
preservation, the warm repeat used 31.87 versus 14.70 CPU seconds (2.17x).
Observed calculation wall times were 36.23 versus 17.39 seconds on the loaded
Mac; CPU placement and frequency were not pinned. These are paired observations
on one case, not a population-wide guarantee. The frozen benchmark source,
outputs and hashes are recorded in
`results/fast-triceratops/closeout-20261025.json`.

An earlier static integration on the deeper BOI 2100325529865507712 at only
20,000 draws used 13.44 versus 1.73 warm CPU seconds (7.77x). It is a different
case and draw count. The native likelihood separately improved by 1.42x; these
stage gains are not multiplied. Total elapsed time from acquisition through
fitting and final review has not been benchmarked.

### Cost of a complete 20-repeat campaign

The 2.17x warm-repeat CPU ratio is not a measured 20-repeat elapsed speedup.
Counting one observed cold repeat, worker initialization and 19 repeats at the
observed warm cost projects 650.70 stock versus 311.06 engine CPU seconds for
this case: 10 min 51 s versus 5 min 11 s, or 2.09x less total CPU work. This is
an extrapolation from two seeds, not a completed paired 20-seed benchmark.

The original production coordinator already assigns five numerical threads to
each serial repeat. Its preserved first-attempt through final-audit timestamps
span 323.29 seconds for all 20 repeats. Those historical elapsed times were
recorded under a different machine load from the new one-thread replay.
Therefore the one-thread CPU ratio cannot be applied directly to that elapsed
baseline, and five worker processes do not add a further unconditional 5x.

A simple scheduling model for the engine, retaining the first-two screen and
charging cold initialization for each new worker, gives these ideal elapsed
times with available dedicated cores:

| One-thread worker processes | Ideal model for all 20 |
| --- | --- |
| 1 | 5 min 11 s |
| 2 | 2 min 44 s |
| 5 | 1 min 45 s |

The model excludes contention, memory bandwidth limits, all-core frequency
changes and coordinator startup. A working target is about 2–3x shorter elapsed
time for this case's entire 20-repeat stage using the same five-core budget.
It remains a target until the original coordinator and experimental pool are
compared on all 20 seeds under the same resource conditions. The projection,
assumptions and all 20 original result receipts are saved in
`results/fast-triceratops/campaign-projection-20261025.json`.

Native fitting happens before these 20 repeats. Its measured 1.42x likelihood
gain is separate and does not establish the elapsed gain of a complete fit.
If the probability stage occupies fraction f of a candidate's original total
time, accelerating only that stage by S gives total speedup
`1 / (1 - f + f / S)`. At S = 2, a stage taking half the total time yields
1.33x overall; a stage taking 80% yields 1.67x.

### First adoption check, October 26

At 00:56 UTC, the checked survey and M/K case trees contain one engine replay:
two original million-draw seeds of KIC 5458397, with one worker. Both match
the original lnZ bits, complete scenario and star tables, FPP and NFPP. All
referenced output receipts were rechecked. The new launch took 747.53 seconds,
including startup, beside production jobs; the calculation phases used 313.00
and 300.40 CPU seconds. This was an equivalence check, not a matched speed test.

The original five-thread serial campaign's 20-repeat interval was 2235.26
seconds (37 min 15 s). Its first two repeats span 207.38 seconds through the
next attempt's start. The original records do not expose CPU time, and machine
load and scheduling priority differ, so the replay cannot establish a CPU or
elapsed speedup against them.

This workload differs materially from the earlier demonstration: 30 versus
21 scenarios, 200 versus 158 bins, and a 1.41-day versus 51.53-day period. Its
depth/bin-noise ratio is 2.72 versus 17.96. The observed later engine repeat
cost is about 300 CPU seconds instead of 15. Reusing the earlier scheduling
model gives about 1541 seconds (26 minutes) for 20 repeats with five dedicated
workers, preserving the first-two gate and worker startup. That is only a
projection; relative to the historical interval it is about 1.45x before
contention, not evidence for a universal 2–3x campaign speedup.

No completed five-worker 20-repeat engine campaign was present at this check.
The two active queue jobs were native fits, which do not use the new native
kernel. Preserve the earlier case-specific results, but obtain a complete
equal-budget comparison before making a general throughput claim. The
[review record](../results/fast-triceratops/first-production-review-20261026.json)
contains the inventory, output comparisons, timestamps and projection limits.

### Measured campaigns, October 26 afternoon

The five-worker engine has now finished 32 queued campaigns with 262 primary
repeats. Eleven completed all 20 independent million-draw repeats; 21 stopped
after the prescribed first-two screen. Full-campaign launch times, including
worker startup and result saving, are:

| Completed allocation | Campaigns | Minimum | Median | Maximum |
| --- | --- | --- | --- | --- |
| All 20 repeats | 11 | 2 min 24 s | 5 min 1 s | 16 min 17 s |
| First-two futility stop | 21 | 34 s | 51 s | 2 min 2 s |

All full campaigns used five worker processes. Output receipts, frozen engine
manifests, source plan/freeze receipts, seeds, attempted-draw counts, first-two
allocation, probability normalization and original-audit statistics were
checked. All 262 attempt directories have saved complete results, with no extra
failed or interrupted attempts and no recovery diagnostic repeats.

These establish actual elapsed times for complete probability campaigns. They
exclude earlier native fitting, acquisition and preparation, and later physical
closeout. None of these frozen plans has saved stock repeats for the identical
inputs; therefore they do not establish an original-versus-engine 20-repeat
speedup. Their worker CPU usage averaged a median 3.27 CPU seconds per elapsed
second, including startup and the first-two gate; this is utilization, not a
counterfactual speedup. Preserve that distinction when comparing cohorts.

The [measurement table](../results/fast-triceratops/campaign-measurements-20261026.csv)
lists every campaign. The
[measurement record](../results/fast-triceratops/campaign-measurements-20261026.json)
contains file receipts and verification scope.

### Historical stock comparison, October 26

Twenty historical stock V3 campaigns completed all 20 final-clock repeats at
1,000,000 attempted draws per scenario and a five-core budget. Their preserved
attempt and matching-audit timestamps support a production-cohort comparison
with the 11 completed engine campaigns:

| Comparison | Stock campaigns | Engine campaigns | Stock median | Engine median | Ratio of medians |
| --- | --- | --- | --- | --- | --- |
| All completed V3 sets | 20 | 11 | 13 min 1 s | 5 min 1 s | 2.59x |
| Same 20-sample exposure integration | 17 | 11 | 12 min 20 s | 5 min 1 s | 2.46x |
| M/K programme only | 8 | 10 | 11 min 24 s | 5 min 25 s | 2.10x |

All groups use the same nominal five-core budget. The all-stock group includes
three more expensive 40- or 80-sample exposure integrations; the restricted
comparison removes those. First-two exits, old V1/V2 protocols, controls and
duplicate stock-reference folders without matching final audits are excluded.
All 400 historical repeats were checked against seeds, draw counts, freeze
identity, output receipts, normalization and complete audit vectors. No large
inter-seed gaps were flagged by the recorded three-times-median gap check.

This historical comparison supports roughly 2.1–2.6x shorter median campaign
times in the observed production cohorts. The targets and machine loads differ,
so the ratio does not isolate implementation alone. Historical stock timing
starts at the first preserved attempt and excludes earlier coordinator startup;
engine timing includes launch startup. An identical-input paired run remains
useful for attribution, but its absence does not prevent this cohort comparison.

The [comparison table](../results/fast-triceratops/historical-stock-comparison-20261026.csv)
contains all 31 campaign timings. The
[comparison record](../results/fast-triceratops/historical-stock-comparison-20261026.json)
preserves inclusion rules, verification receipts and timing boundaries.

## Adoption

The queue now runs `engine()` through `fast_triceratops.workflow`, with five
single-thread workers within the frozen CPU budget. It writes separate
`campaign-engine/` directories and uses the original `post_boi.campaign.finish`
for the audit. Campaigns already started on the original coordinator finish
there. The runner's experimental scope labels predate the adoption and remain
in its frozen outputs; the adoption record describes their use.

The reference wrappers and the more aggressive certified mode remain separate
choices. Their earlier integration proposal was:

1. In `post_boi.campaign.calculate_quarters`, run `calc_probs` inside
   `fast_triceratops.accelerated_v3(certified=...)`. Declare the mode in the
   candidate, and freeze `fast_triceratops`'s source with the campaign.
2. In `resume()`, run the first two repeats as two single-thread processes.
   Then run the remaining 18 as single-thread processes, as many at a time as
   the CPU budget allows. Each repeat's attempt, seed and records stay as
   they are; only wall time changes.

Once pruning removes most of the transit-model work, upstream's own serial
sampling, priors and splines dominate a repeat. Single-thread repeats running
in parallel therefore use the cores better than five threads per repeat.

## Work log

The work on October 25, 2026 proceeded in two parts:

- The likelihood acceleration, bounds, reference benchmarks, unmodified engine
  copy and initial spline replica.
- The static engine integration, shared spline basis and flux calculations,
  native orbit experiment, workflow, controls and performance records.

Existing campaign folders and external datasets remain read-only. Adam
authorized internal Mac scratch on October 25; the current workspace is
`other/fast-triceratops-trial-20261025`.

## Limits

- At low signal-to-noise most planet draws carry nonzero weight, so exact
  mode must model them with pytransit's own code. The measured reference
  wrapper gains are near 2x for the M/K examples. Certified mode goes further.
  The static engine also shares deterministic spline and flux work, while
  preserving the original sampled priors.
- Apple GPUs are ruled out: they have no float64, and a float32 likelihood
  biases lnZ by +0.2 to +0.36.
