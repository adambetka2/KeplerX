# KeplerX Validator: native fit and preparation speed

2 November 2026. New stage `KeplerX/validator/` (`preparation.py`,
`native_fit.py`, `native_likelihood.py`, `transit_likelihood.c`,
`kernel_bindings.py`, `benchmark.py`, `juliet_reference.py`, vendored dynesty
3.1.0 in `upstream/`), `KeplerX/upstream.json`, tests
`test_golden_validator.py` and `test_e2e_validator.py` with the fixture
`validator-m-dwarf-2126055064030395392-1` and `golden/validator.json`.
Record: `KeplerX/results/2026-11-02-validator-native-fit-benchmark.json`
(runs in `KeplerX/results/validator-benchmarks/`). `post_boi/` and every
accepted study are unchanged.

## Why

TODO item 13: before large-scale v3 statistical validation, target a 50x
speedup of native fitting and preparation on the M1 MacBook Pro, measured end
to end against a frozen baseline, keeping the adopted models, priors, sampler
requirements and validation criteria, and verified on fitted posteriors,
prepared inputs and downstream results.

## Result

On eight of the nine production cases fitted in full (142 to 12,758 native
photons, 0.27 to 50 days, well-behaved and runaway posteriors; fourteen
complete fits counting the day's reruns over successive kernel revisions),
the Validator reproduces
production's posterior **sample for sample**: every 11-parameter sample bit
for bit, the same number of likelihood calls, the same review clock (period,
epoch, duration), and the prepared arrays bit for bit. For those, the
probability campaign's inputs are production's, so its scenario evidences,
FPP, NFPP and dispositions are the qualified baseline's by construction.

On the ninth, the largest (the PSF-campaign fit, 38,815 photons), the two
runs share their first 14,250 of 22,487 dead points (ordered by likelihood)
and differ after that, deep in the posterior: a near-tie comparison went the
other way.
There the Validator's likelihood still equals juliet's to 3e-10 on 300 of its
own posterior samples and the evidence agrees (0.09 combined sigma), but the
posterior quantiles moved by up to 0.29 posterior widths (one tail quantile by
0.89) and the review clock with them (duration 0.2045 to 0.2089 d, epoch by
2.7 minutes). A third run, the Validator with a new seed, settles what that
is: it differs from production by 0.27 posterior widths rms in its medians
(0.60 at most) and from the same-seed Validator run by 0.32 (0.64), more than
the 0.16 (0.29) between production and the same-seed run. The divergence is
this runaway fit's ordinary run-to-run scatter, which production with another
seed would show as well; dynesty's own estimates understate it about tenfold
(the three lnZ are 209,639.72, 209,639.68 and 209,642.04 with stated errors
of 0.25-0.29). Across the three runs the review clock spans 0.1976-0.2089 d
in duration and 2.7 minutes in epoch. Production held this fit at native
review (protection), so no probability campaign used its clock.

Elapsed time fell 16.6-34.7x on seven fits against production's own
records, 14-16x in paired reruns of production's
commands on the same machine under the same load, and 3.3x and 7.5x on the
ultra-short-period and PSF fits, whose workers got about half a core each
beside two other jobs. CPU per fit fell 5-29x (14.3x weighted by
production's CPU over the nine). The 50x target was not reached: exact
evaluation of the adopted model needs one batman-formula evaluation per
exposure sample in transit, and the fits that dominate production's cost
(runaway, grazing or ultra-short-period posteriors whose transit fills the
native window) are made of such samples. Per likelihood call the gain is
14-36x on well-behaved posteriors and 6-16x on those (10.3x weighted by
production's time over ten representative cases).

## Baselines

- **Production** (the frozen baseline): the stored native fits of 25-28
  October, run by `tools/run_native.sh` in the frozen V3 runtime (juliet
  2.2.10+spacelab.1, batman 2.5.3+spacelab.2 with the production orbit kernel
  enabled by its sitecustomize, dynesty 3.1.0, Lightkurve 2.6.0), five
  workers (three in the PSF campaign), often three fits at a time. Elapsed
  time (preparation included) and calls come from each case's `native.log`.
  The 150 completed dynamic-dynesty fits took 128.3 hours (median 1,228 s,
  90th percentile 8,569 s, longest 34,389 s); 68% of it was in the 29 fits
  with more than 5,000 native photons.
- **Paired rerun**: `validator.benchmark --baseline` reruns production's
  exact commands (preparation without the orbit kernel, the fit with it)
  just before the Validator on the same case. Both reruns reproduced the
  stored production posteriors bit for bit, so the runtime is deterministic
  and the comparison is like for like.
- **Per call**: production's juliet log-likelihood (orbit kernel on) and the
  Validator's, in one process on 40 production posterior samples each.

The Mac was shared with another 8-worker job for most of the
measurements (load averages 7-49 on 10 cores). Elapsed times are pessimistic
for both sides of a pair; against production's records (run beside up to two
other fits) they are roughly comparable. CPU times include time on efficiency
cores.

## Results

| Case | Native photons | Calls | Production | Validator, prep + fit | Elapsed gain | Per call (CPU) | Posterior |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M/K signal, 50.4 d | 142 | 250,589 | 147 s | 0.7 + 8.0 s | 17.0x | 35.8x | identical |
| Survey BOI, 11.3 d | 543 | 465,365 | 701 s | 0.7 + 37.9 s | 18.2x | 14.1x | identical |
| M/K signal, 2.79 d | 1,795 | 323,405 | 1,214 s | 1.3 + 33.7 s | 34.7x | 31.6x | identical |
| KIC 10528996 b, 13.7 d (validated) | 4,979 | 472,696 | 2,984 s | 0.8 + 137.6 s | 21.6x | 23.6x | identical |
| KIC 10528996 c, 8.56 d (validated) | 8,174 | 484,725 | 4,703 s | 1.9 + 282.1 s | 16.6x | 25.6x | identical |
| M/K signal, 5.74 d (posterior e 0.98) | 788 | 649,414 | 1,242 s | 0.6 + 69.2 s | 17.8x | 9.1x | identical |
| Survey 28.4 d, second fit (e 0.74) | 9,171 | 1,208,366 | 21,121 s | 0.7 + 1,035.8 s | 20.4x | 15.4x | identical |
| Survey USP, 0.272 d (grazing, a/R* 1.4) | 12,758 | 563,142 | 10,035 s | 0.7 + 3,008.1 s | 3.3x | 5.9x | identical |
| PSF campaign, 0.934 d (e 0.55) | 38,815 | 631,241 | 34,389 s | 0.8 + 4,567.8 s | 7.5x | 13.0x | diverged late (below) |

Validator elapsed is preparation plus fit, each a fresh process; production's
is its `native.log` from start to done (preparation included). "Identical":
every posterior sample, the call count and the review clock equal
production's. The six smaller cases were also fitted earlier the same day
with earlier kernel revisions, identical as well (each fit's `sampler.json`
names its kernel digest). The largest M/K second fit (21,892 photons, 26,202
s in production) was measured per call only. The PSF row's calls are the
Validator's (production made 631,706). CPU, production against the Validator
in table order: 254 against 19 and 1,140 against 98 CPU-s measured in the
paired reruns; then, estimated for production from its per-call cost (which
leaves out juliet's other overheads), 2,511 against 97, 8,659 against 361,
15,073 against 523, 3,835 against 209, 73,266 against 4,003, 29,293 against
6,132 and 128,180 against 6,810 CPU-s.



Paired reruns (same case, same load, preparation plus fit):

| Case | Production rerun | Validator | Elapsed | CPU |
| --- | ---: | ---: | ---: | ---: |
| M/K signal, 50.4 d (142 photons) | 43.2 + 92.3 s, 254 CPU-s | 0.7 + 8.0 s, 19 CPU-s | 15.7x | 13.1x |
| Survey BOI, 11.3 d (543 photons) | 15.9 + 537.2 s, 1,140 CPU-s | 0.7 + 37.9 s, 98 CPU-s | 14.3x | 11.7x |

Per likelihood call, CPU (production's juliet with its orbit kernel against
the Validator, same process):

| Case | Production juliet | Validator | Gain |
| --- | ---: | ---: | ---: |
| M/K signal, 50.4 d (142 photons) | 0.54 ms | 0.015 ms | 35.8x |
| Survey BOI, 11.3 d (543 photons) | 1.76 ms | 0.124 ms | 14.1x |
| M/K signal, 5.74 d (posterior e 0.98) (788 photons) | 5.70 ms | 0.630 ms | 9.1x |
| M/K signal, 2.79 d (1,795 photons) | 7.47 ms | 0.237 ms | 31.6x |
| KIC 10528996 b, 13.7 d (validated) (4,979 photons) | 17.67 ms | 0.748 ms | 23.6x |
| KIC 10528996 c, 8.56 d (validated) (8,174 photons) | 30.00 ms | 1.173 ms | 25.6x |
| Survey 28.4 d, second fit (e 0.74) (9,171 photons) | 58.66 ms | 3.817 ms | 15.4x |
| Survey USP, 0.272 d (grazing, a/R* 1.4) (12,758 photons) | 50.21 ms | 8.531 ms | 5.9x |
| M/K 4.13 d, second fit (a/R* 1.15) (21,892 photons) | 106.79 ms | 15.215 ms | 7.0x |
| PSF campaign, 0.934 d (e 0.55) (38,815 photons) | 196.07 ms | 15.046 ms | 13.0x |

CPU and batch throughput: the paired reruns measured 11.7x and 13.1x less
CPU; on the larger fits, production's CPU estimated from its per-call cost
(calls plus the serial posterior pass) is 18-29x the Validator's measured CPU
(18.3x on the 28-day second fit, 28.8x on KIC 10528996 c; the per-call
estimate leaves out juliet's other overheads, so it understates production).
The gains are computation, not more cores: each fit keeps production's five
proposals per queue and five workers, so fits per day on the Mac rise by the
same factor, two five-worker fits at a time as before.

Memory: a Validator fit's parent peaks at 210-380 MB and each worker at about
90 MB; production's juliet processes peaked near 310 MB each.

## What changed, and what did not

The Validator is a separate copy of `post_boi.prepare` and `post_boi.native`.
It keeps juliet's model (Keplerian orbit from P, t0, rho, b, e, omega;
batman's quadratic limb-darkening formulas with its Hastings K and E and
Bulirsch Pi; the exposure averaged over the declared 20 samples; juliet's
dilution, mean flux and jitter, and its rejections), the priors, the photons
(+/-3 delivered durations, other signals masked), the seed and every dynesty
setting (dynamic nested sampling, multi-ellipsoid bounds, random walks, 500
live points, dlogz_init 0.1, n_effective 5,000, the declared proposals per
queue, checkpoints, juliet's equal-weight resampling).

What production spent its time on, and what replaced it:

1. **Every photon on every call.** batman evaluates the orbit for all 20
   exposure samples of every native photon (263 ns per sample, 43 ms per call
   on KIC 10528996 c). Along the orbit d^2 = r^2 (sin^2 psi + cos^2 psi
   cos^2 i) exactly; bounding r from below on arcs of the front half, then on
   each side of conjunction, gives a window around each conjunction outside
   which d >= 1 + p for every sample, where batman's flux is exactly 1 and is
   left in place. Checked on 20,039 accepted random prior draws on three
   cases, a quarter of them at eccentricities from 0 to 0.99: all 56.8
   million skipped photons are exactly 1 when modelled everywhere.
   Well-behaved posteriors model 12-30% of the native photons.
2. **Per-sample orbit.** batman recomputes the periastron time and solves
   Kepler's equation from scratch for each sample with about 20
   transcendental calls. The kernel computes orbit constants once, takes the
   separation from (sin E, cos E) without the true anomaly, and solves each
   sample from a nearby solution (fourth-order Taylor predictor, sine and
   cosine carried by rotation, Newton) in groups independent of one another;
   batman's bracketed solver starts every window.
3. **Limb darkening.** batman's fully-inside and limb-crossing branches run in
   batches per photon, Bulirsch's iterations interleaved across the batch,
   with Accelerate's vForce logarithms and arc cosines.
4. **Sampler overhead.** juliet's Python likelihood; its prior transform (27
   us per call through scipy.stats, now 5 us through the same ufuncs and libm,
   bit for bit); its pool, which pickled the fit object (about 185 kB) into
   every proposal, replaced by dynesty's own caching pool; and juliet's serial
   pass recomputing every posterior sample's likelihood (40,547 samples took
   about 40 minutes on the 28-day second fit) now in the pool (31 s).
5. **Preparation.** Lightkurve's flatten in plain numpy/scipy: 0.6-1.9 s
   instead of 16-43 s (cold Lightkurve and astropy imports), arrays bit for
   bit.

The kernel's numbers agree with batman's to rounding, not bit for bit
(log-likelihoods to about 1e-10, model fluxes to 1e-14, and to 1e-11 on
grazing geometries). The nested sampler's comparisons resolve such
differences only at near-ties, so its path, and so the posterior, is
production's exactly unless a comparison falls within about 1e-9 of a
threshold. On eight fits none did; on the 38,815-photon PSF fit, whose
log-likelihoods agree to 1e-9 rather than 1e-10, one did after 14,250 dead
points. Bit-for-bit equality is the adoption test where it holds
(`tests/test_e2e_validator.py`); a diverged fit is judged statistically
against production's by the scatter of independent runs of its configuration:
for the PSF fit, a second seed (`results/validator-benchmarks/2026-11-02-native-fit-psf-second-seed`)
scatters as far from both runs as they lie from each other. The protocol takes
one seeded fit's clock, so that scatter reaches the probability campaign
unmeasured; on the well-behaved fits that validate it has not been measured.

## Cold setup, warm execution, caches

- The kernel compiles once per source digest (about 0.5 s) into
  `runtime/cache/compiled-kernels`; every later process loads it.
- A Validator fit starts its interpreter, numpy, scipy, dynesty and its
  spawned workers in 1-1.5 s; juliet's import chain (Lightkurve, astropy,
  matplotlib) took 5-40 s per process cold.
- Nothing is cached between fits or reused from production: every reported
  gain is computation and overhead removed.

## Projected cost of the v3 candidate sample

Production's 150 completed native fits took 128.3 hours of five-worker
time: 51 minutes per fit on average, 20 minutes at the median. With the
measured gains (14.3x in CPU over the nine fits, 10.3x per call over ten
cases, both weighted by production's cost), the same fits would take 9-12.5
hours of five-worker time: 3.6-5 minutes per fit on average and about one
minute at the median, on a machine not shared with other jobs.

For a v3 sample of N candidates with production's mix of fits, preparation
and native fitting then cost about 3.6-5 minutes of a five-worker slot per
candidate, against 51 minutes before, and the unchanged probability campaign
about 5 minutes at the median. The Mac runs two five-worker slots side by
side, so 1,000 candidates need about 30-42 hours of wall time for native fits
(about 18 days before) plus about 42 hours for probability campaigns. The
survey's vetting keeps candidates on 7.6% of curves (about 19,000 curves,
24,000 signals) before localization; only localized, vetted BOIs reach
validation, so the v3 sample is a fraction of that. Runaway fits remain the
costliest, and belong to the review of candidates that should not be fitted
again.

## The probability engine

Unchanged: campaigns keep `fast_triceratops.engine()` (exact mode), twenty
repeats of a million draws per scenario. Its measured campaigns (median 5 min
with five workers, 26 October) are now as large as a typical candidate's
native fit; any further gain there is separate work and was not measured here.

## Limits and what would go further

- The Validator supports juliet's one-planet `post_boi.native` model and the
  four prior distributions the campaigns use; anything else is refused.
- Small fits are bound by dynesty's own Python: about 15 us of bookkeeping
  per call in the parent and 40-60 us per proposal in a worker.
- Grazing, low-density and very eccentric runaway posteriors fill the native
  window and are the costliest fits. An early physical-plausibility stop for
  them would save more than any kernel, but it changes the protocol.
- Further exact gains would need NEON vectorization of the limb-darkening
  branches or a compiled random walk replicating numpy's generator.
  Tabulating limb darkening per call would be faster still but would end the
  bit-for-bit agreement of posteriors.
