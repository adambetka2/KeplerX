# KeplerX subsystems in detail

What each KeplerX subsystem does and how, moved here from the subsystems'
READMEs on 3 November 2026, when each README was cut to a 20-line map of
its files. File names are those in the subsystem's folder, other paths
start at `KeplerX/`, and linked records are in this folder.

## Detector (`detector/`)

Run the Detector's transit search on a prepared curve through the pinned
upstream Dump API.

Entry point from the campaign root:

```text
run.sh detect PREPARED OUTPUT
```

The Detector is one search with no option switches; `config.json` sets its
five parameters (period range, minimum observed transits, MES threshold and
passes). It subclasses the pinned upstream `Dump` (`compiled_search.py`) and
saves the returned candidates and the initial and final masks; internal
proposals are not exposed. Where it departs from upstream:

- Template bank (`upstream_adapter.py`): upstream's, from two cadences
  (0.98 h), with upstream's duration window of 0.25-1.5 times the reference
  duration. Templates of 6 hours and longer pass through KBonus's 2-day-knot
  spline flattening (`flattened_templates.py`), the shape such transits are
  delivered in, and their statistic is divided by its robust spread where
  the whitening underestimates its noise (`gap_padding.Segments._unit_spread`).
  The Vetter's whitening uses the same templates.
- Segments (`gap_padding.py`): upstream's, also ended at every gap longer
  than 30 days, which upstream merged and padded (65,536-point segments on
  survey curves), each whitened about its own mean (upstream's offset biased
  every MES low by 0.3-1.0). Boundary padding is repaired, and segments
  whose whitening is degenerate (non-finite, or a denominator above 1e12: a
  few reflected cadences with zero MAD) are dropped and cleared in the
  caller's mask; see keplerX-2026-10-29-edge-cases.md.
- Whitening (`compiled_search.FastSearch._fft_bank`): upstream's MAD noise
  weights, with each template's level convolutions summed by FFT (within
  5e-14 of upstream's sums). The search whitens only durations that some
  trial period admits.
- Period search: upstream's period grid at twice its step, folded in
  `search_kernels.c` and screened on the GPU where one is available
  (`gpu_screening.metal`: fold for short periods, window_fold for longer
  ones), keeping winners from MES 6.6 (`compiled_search.SCREEN_THRESHOLD`).
- Winners (`single_event_veto.py`, `candidate_ephemerides.py`): an outlier
  test on the strongest event replaces upstream's single-event veto, and
  cleaning by shared events its harmonic cleaning. Each detection's period,
  mid-transit epoch and duration are refined; the MES threshold applies to
  the refined value; the alias at P/2, P/3, 2P or 3P replaces it when its
  refined MES, with and without its strongest event, is at least 5% higher.
- Iteration (`repeat_rule.py`): a later detection at a known candidate's own
  period, or another detection's exact alias, is masked but not a new
  candidate, and the search ends after two passes that add nothing new.

`upstream_adapter.make_dump(..., reference=...)` also builds the golden
tests' two upstream references, never used in production: "native" is
upstream's own Python search, and "compiled" these kernels with upstream's
direct convolutions, single-event veto and harmonic cleaning. On the same
bank, grid, segments and repeat rule they must agree bit for bit
(tests/test_golden_detector.py). The equivalence argument and timings are in
keplerX-detector-compiled-search.md, and the evidence for each
departure in keplerX.md.

## Vetter (`vetter/`)

Compute metrics using the independently owned upstream copy.

Entry point from the campaign root:

```text
run.sh vet PREPARED DETECTIONS OUTPUT
```

Metrics remain UNCLASSIFIED; no custom acceptance-policy engine is installed.
Numerical metric failures retain their traceback and mark the signal UNRESOLVED.
Independent metrics and other signals continue without changing fitting budgets.

### Refinement weights

Upstream's `_transit_resids` divides residuals by `flux_err**2` in both of
its branches, so leastsq weights each cadence by sigma**-4. `FastVetter`
divides by sigma (`vetting.refinement_weight = "sigma"`, the default), so
the refinement minimizes chi-square; `"sigma_squared"` restores upstream
and the native comparisons use it. Effects on the 12 validation signals:
keplerX-2026-10-29-edge-cases.md.

### Compiled engine

The Vetter is `FastVetter` (`accelerated_vetter.py`), which subclasses the upstream class
and gives every measurement the same bits with less time,
except the transit model, which takes batman's limb-darkening integral
accurately instead of by batman's midpoint rule:

| Upstream code | Replacement | Why the result is identical |
| --- | --- | --- |
| `chi2_tests` | shared whitening | The channel and temporal statistics call `tce_masked_num_den` with identical arguments; it runs once, with the compiled running median and convolution of `detector.compiled_search`. |
| `lmfit.minimize` in `_update_tce`, `odd_even_transit_depths`, `bic_morphology_test` | `exact_fit.minimize` | An lmfit fit is fixed by the internal vector it hands SciPy, the array its residual wrapper returns per vector, and the SciPy call. exact_fit repeats them: `Parameter.setup_bounds` gives the internal start, `from_internal` is evaluated with the same libm `sin`/`sqrt`, values are clipped as the `value` setter does, residuals are coerced and NaN-checked as `coerce_float64` does, the reduce functions are lmfit's own, SciPy receives lmfit's arguments, and the statistics, leastsq covariance scaling and `create_uvars` repeat lmfit's arithmetic on lmfit's own `Parameters` copy. numdifftools is absent from the runtime, so lmfit estimates no covariance for the L-BFGS-B fits. |
| `box_residual`, `spsd_residual` (ramp and drop), `sine_residual`, `trap_residual` | `residual_kernels.c` through `kernel_bindings.Window` | Each kernel repeats NumPy's elementwise operations in order, with libm `exp` and `cos`, which NumPy and `scipy.special.expit` call on this platform (checked bitwise on 6 M inputs); `x**2.` is `x*x`. Reductions stay in NumPy. |
| `batman.TransitModel` in `_get_batman` and `fit_transit` | `transit_model.TransitModel` | Not identical: the flux is batman's own integral taken to 1e-12 (`v3_nonlinear_ld` in residual_kernels.c, Gauss rules in x below 0.9 and in s = mu^(1/2) above, closed form for the covered disc), where batman's rule stops at its `max_err` (6e-8 for the refinement, 3e-6 for planets larger than the star). Points are pruned as before: batman returns exactly 1.0 when `max(d - rp, 0) >= 1`; for a circular orbit with w = 90 deg and 0 <= inc <= 180 deg, `d >= a |sin f|` with batman's own `f = frac((t - t0)/P) 2 pi`, and `cos f <= 0` puts the planet behind the star. Points with `|f| >= asin((1 + rp + 1e-6)/a)` are out of transit; the rest go to batman's `_rsky` and the integral. The supersample average is batman's NumPy expression. Identical parameters return the previous curve. Other laws and orbits go to batman unchanged. `transit_model.batman_rule` restores batman's rule bit for bit. |

For calls with at least 2,048 points, `transit_model.INTERPOLATE` evaluates the
integral through cached piecewise Chebyshev interpolants in d where the flux
is analytic: inside the disc and in partial overlap, away from the contact
radii 1 -+ p and 0.99995 -+ p. Elsewhere the integral runs. The interpolants
agree with the integral to 3e-13, and to 6e-12 where the integral's own rule
switches at x = 0.9. Vetting a 0.95-day signal took 4.1 s instead of 29 s, and
a grazing 0.53-day eclipsing binary 8.5 s instead of 54 s.

Plot-only branches of the copied upstream functions are left out; `emcee`
refinement and plot values raise instead of diverging.

Evidence: `tests/test_golden_vetter.py` (with `batman_rule`, every metric of
the 9-signal failing-vetting fixture, including its numerical failure,
equals the native class; with the integral, every component's outcome is
kept). The part-by-part checks (kernels, transit model and fits against
their originals on random inputs, the integral against adaptive quadrature)
were unit tests, removed on 1 November 2026 when KeplerX kept only golden
and end-to-end tests. Effects of the integral on 12 signals:
keplerX-2026-10-28-vetting-and-localization-math.md. A scratch run of the whole
fixture with every fit executed by both lmfit and exact_fit compared 241 fits
and 133,678 evaluations: identical trajectories and fields.

Measured on a loaded M1 Pro (load 20-50 from other jobs; medians of
three runs for the fit rows), one injected signal on a 58,000-cadence curve.
Native is the upstream class on the native Dump; compiled is `FastVetter`
on the compiled Dump. The chi2, mes and secondary rows come from the compiled
whitening; the fit rows from the replacements above.

| Component | P = 1.9 d native / compiled | P = 15 d native / compiled | P = 44 d native / compiled |
| --- | --- | --- | --- |
| transit refinement (lmfit leastsq, batman) | 14.0 / 12.2 s | 1.38 / 1.05 s | 0.70 / 0.37 s |
| mes (one re-whitening) | 4.4 / 0.6 s | 3.9 / 0.5 s | 3.2 / 1.0 s |
| chi2 tests | 22.2 / 0.9 s | 21.8 / 1.0 s | 21.5 / 1.3 s |
| secondary (one re-whitening) | 3.9 / 0.5 s | 4.0 / 0.5 s | 4.6 / 0.6 s |
| odd-even depths (lmfit) | 2.8 / 1.5 s | 0.14 / 0.11 s | 0.57 / 0.53 s |
| cosine vs transit (curve_fit, batman) | 6.5 / 4.9 s | 0.93 / 0.84 s | 0.12 / 0.11 s |
| local morphology (lmfit L-BFGS-B per transit) | skipped | 6.22 / 1.82 s | 2.63 / 0.73 s |
| whole `measure()` | 49.6 / 25.2 s | about 36 / 6 s | about 32 / 5 s |

These timings predate the integral, which costs 0.38 us per point against
batman's 2.7 us (refinement step) and 11.6 us (cosine fit step); the KIC
12644822 control's whole `measure()` fell from 4.53 to 1.78 s. The morphology
test is skipped above 100 transits. A morphology-like fit of 1,672
evaluations takes 15.5 ms against 58 ms under lmfit (about 9 us per
evaluation), and a kernel residual on 40 points 2.5 us against 12.7 us.

## Localizator (`localizator/`)

Run the pinned native Localizator on detected clocks and explicit parent TPFs.

Entry point from the campaign root:

```text
run.sh localize PREPARED DETECTIONS VETTING REQUEST.json OUTPUT
```

The adapter calls Localize.from_tpf_info, build_design_matrix,
solve_transit_weights and MultiLocalize.fit_to_quarters for each signal.
It preserves pixel masks, depth maps, fit metrics and missing uncertainties.
Clocks come from Vetter's fitted period, epoch and duration by signal ID.
Unavailable fitted clocks and singular pixel solves remain unresolved.
Signals the vetting policy dismissed are not localized (`NOT_RUN`); with
none left, no TPF is opened. A quarter without the source's cadences, or
whose trend columns explain the transit (no information on where its light
comes from), is left out and listed (`quarters_without_coverage`,
`quarters_not_estimable`); other quarter failures leave the case UNRESOLVED.
A quarter whose prepared cadences come no closer than a transit duration to
any localized signal's mid-transit time is not read either
(`quarters_without_transits`): upstream's transit column is zero there, and
MultiLocalize drops such quarters, so results are unchanged (26 of 26
benchmark cases identical to the previous code).
Pixels come from the TPFs a request lists on disc, or with
`"archive": "mast"` from MAST: localizator/mast_pixels.py lists the parent's
folder in MAST's public S3 bucket and streams the quarters it needs into
memory over parallel connections (`localization.fetch_connections`, 8).
Nothing is written to disc; each file's URL, size and sha256 is kept in the
result (`pixels`). Each TPF is reduced on reading to its headers and the
time, flux and flux-error columns (the rest is freed): peak memory 0.70 to
0.51 GB on a 67 MB parent. Streamed and stored pixels localize identically
(26 of 26 benchmark cases).
localizator/accelerated_model.py replaces native arithmetic with faster mathematics for
the same model; results agree with native to tolerance, not bit for bit:
- The per-pixel regression drops the B-spline-redundant polynomial columns
  (native's matrix is singular and fails or succeeds by rounding) and solves
  each pixel once per quarter by Cholesky. Depths match QR to 1e-7 sigma.
  The B-spline columns' normal matrix is banded, so each pixel eliminates
  them by banded Cholesky and factors only a small dense remainder: the same
  depths to 5e-13 sigma, 4x faster on the benchmark's 316 quarters.
- lkprf's per-position spline fit becomes a sum of per-channel splines,
  cached on disk under runtime/cache/prf-basis.
- The joint fit caches PRF models by quarter and position.
- Gaia propagation runs once for all quarters, bit for bit.
localizator/gaia_cone_search.py serves upstream's Gaia DR3 cone search from the ESA Gaia
archive instead of VizieR (whose CDS services and mirrors were unreachable on
29 October 2026). It returns VizieR's I/355/gaiadr3 table exactly (columns,
units, masks, row order and published decimals), under upstream's cache key,
and keeps archive responses in runtime/cache/gaia-dr3. The KIC 12644822 control
localizes identically from either source.
The archive answers a cone in 10 s to over a minute, or not at all (a case
whose cone fails stays UNRESOLVED). localizator/gaia_field.py serves cones
from a local table of a whole field instead (config.json
`localization.gaia_field`): flat columns sorted by declination zone and
right ascension, one zone-offset array, two binary searches per zone, about
1 ms per cone; it returns the archive's table bit for bit (golden test, and
every stored archive cone). Building the Kepler field's table is one bulk
archive download (`python -m localizator.gaia_field build`).
Native pixel-recovery qualification remains open.

## Photometry (`photometry/`)

Read authenticated delivered PSF Flat from LC2 or native NPZ arrays.

Entry point from the campaign root:

```text
run.sh prepare REQUEST OUTPUT
```

Preserve native arrays, quality masks and constant quarter normalizations.

## Runner (`runner/`)

Execute independent stages and bounded manifest-driven batches.

Entry point from the campaign root:

```text
run.sh batch MANIFEST OUTPUT --workers N
```

New output generations preserve inputs, logs, source, resources and first failures.
Stage records include wall time and user/system CPU time.
execution.isolation "worker" (the default) runs every stage, localize
included, inside long-lived workers that import once and are renewed every
200 cases; "process" starts a fresh interpreter per stage. Records name the
isolation, and a worker records the source it imported. Localization keeps
its Vizier cache in each case's folder (localizator.program.native_localizer).
execution.dispatch "longest_first" starts the longest prepared curves first.
execution.retain_photometry "candidates" deletes the prepared arrays of a
completed photometry-request case without candidates (packet.json keeps their
SHA-256 so `prepare` reproduces them); "all" (default) keeps every packet.
The arrays are 90% of a case's 1.2 MB, 214 GB for the eligible dwarfs.
An optional localization request connects native pixels after detect and vet.
Localization consumes Vetter's fitted clocks. Each completed case writes
`bois.json` with provisional, unclassified evidence from the available stages.
A run's outputs live in OUT/results.sqlite (runner/outputs.py): tables jobs,
candidates, vetting and bois, plus every file each case wrote, zlib-compressed
with its SHA-256; a completed case's folder is removed once stored and verified,
and a failed case keeps its folder. Read a run with runner.outputs.RunOutputs.
stop_on_error controls dispatch; a failed case always retains its ERROR outcome.

## Calibration (`calibration/`)

Build the KBonus-realistic calibration sample and score pipeline runs on it.

Post-delivery injection does not measure extraction or flattening transfer.

The KBonus-realistic sample (`calibration-sample`, revision 5) has its own
builder: lane hosts in proportion under production preparation, physical
planets and binaries diluted by the host's Kepler-band share of the curve's
flux and passed through an emulation of KBonus's flattening, real TPF
leakage, real artifact curves, paired artifact copies, real planets, and a
habitable-zone stratum: the quietest K and M curves, each with small planets
drawn in its zone and weighted as every other planet is. Truth
carries each signal's information S/N, the ruler the detector's efficiency is
measured against.

- `sample_builder.py` builds, audits and registers it;
- `signal_injection.py` holds the signal models and the flattening emulator;
- `population.py` the stellar inputs, draws and weights;
- `lane_census.py` the lane census and the quality report;
- `artifacts.py` the artifact models measured on the census, their
  injection and their validation;
- `scoring.py` scores one or more runs, partition by partition;
- `study.py` runs the whole calibration as one study: one commit, one
  config, one manifest, resumable attempts, one score.

See keplerX-calibration-sample.md.

```text
uv run --locked python -m calibration.lane_census census REFERENCE.parquet OUT_DIR
uv run --locked python -m calibration.artifacts measure CENSUS_DIR
uv run --locked python -m calibration.artifacts validate
uv run --locked python -m calibration.sample_builder build|audit|register|manifests OUT
uv run --locked python -m calibration.lane_census report OUT --reference CENSUS_DIR/features.parquet
uv run --locked python -m calibration.study run NAME [--workers 10] [--partitions development [holdout]]
uv run --locked python -m calibration.scoring RUN [RUN ...] --sample SAMPLE --out OUT
```

Run from `KeplerX/` with `UV_PROJECT_ENVIRONMENT=runtime/pipeline-environment`.

## Validator (`validator/`)

The stage after the Localizator: a candidate's duration-aware preparation,
its native transit fit, the native review, the TRICERATOPS V3 probability
campaign and its closeout. Copies of `post_boi` (commit 819485bc2) and
`fast_triceratops` (commit 1c04e9be2), which stay unchanged for the accepted
v1/v2 studies, to be optimized here. Preparation and the native fit are
already rewritten and reproduce production: the same prepared arrays bit for
bit on every case tried, and on eight of nine full fits the same posterior
sample for sample (the ninth, the largest, diverged at a near-tie and agrees
statistically). The rest is copied with imports renamed only.

| Module | Holds |
| --- | --- |
| `preparation.py` | KBonus preparation (post_boi.prepare without Lightkurve), `fold` |
| `native_fit.py`, `native_likelihood.py`, `transit_likelihood.c`, `kernel_bindings.py` | The native transit fit and its compiled likelihood |
| `protocol.py`, `campaign.py`, `quarters.py`, `dependencies.py` | V3 protocol: declarations, clock review, freeze, 20 repeats, statistics; the quarter likelihood |
| `scene.py`, `aperture.py`, `psf_flat.py`, `psf_response.py`, `psf_scene.py` | Gaia scene, aperture and PSF responses |
| `protection.py`, `kbonus.py`, `ephemeris.py`, `occultation.py` | Protection review and closeout (flux closure, quarter noise, ephemeris, occultation) |
| `probability/` | The TRICERATOPS V3 engine (next section) |
| `records.py`, `benchmark.py`, `juliet_reference.py` | Receipts; the native-fit benchmark; the golden record's writer |
| `patches/`, `frozen-runtime-requirements.txt`, `candidate-template.json` | The reviewed dependency patches, the frozen V3 runtime's pins, the candidate record |
| `upstream/` | Vendored dynesty 3.1.0 and batman's licence |

Preparation and the native fit run in the KeplerX pipeline environment. The
probability campaign and closeout still need the frozen V3 runtime
(TRICERATOPS d20875c with `patches/triceratops-quarters.patch`, pytransit,
Lightkurve; `frozen-runtime-requirements.txt`). A campaign frozen by
`validator.protocol.freeze` copies the whole package, the probability engine
included, into its `source/validator`; that path has not yet run a campaign.

Entry points from the KeplerX folder:

```text
run.sh prepare-native CANDIDATE OUT                  (OUT/prepared, OUT/run.json)
run.sh fit-native CANDIDATE PREPARED OUT [--workers N]   (OUT/native, OUT/run.json)
```

or, in the pipeline environment, `python -m validator.preparation CANDIDATE
PREPARED`, `python -m validator.native_fit CANDIDATE PREPARED FIT` and
`python -m validator.benchmark CASES.json OUTPUT [--baseline FROZEN_V3_PYTHON
--kernel-site DIR]`.

`CANDIDATE` is a post-BOI candidate record (`post_boi/candidate-template.json`).
Outputs are post_boi's: `prepared/detrended.npz` and `preparation.json`;
`inputs.json`, `posterior.npz`, `summary.json` and `review.json`, plus
`sampler.json` (evidence, calls, iterations, stage timings, CPU, peak memory,
kernel digest) and `sampler/results.pkl` (dynesty's results).

### What is unchanged

- The preparation: transit-protected Lightkurve 2.6.0 Savitzky-Golay baseline
  per quarter (3 sigma, 3 iterations, break tolerance 5), errors divided by the
  trend (`preparation.flatten_trend`, the same numpy/scipy calls without
  astropy objects).
- The model: juliet 2.2.10+spacelab.1 with batman 2.5.3+spacelab.2: Keplerian
  orbit from (P, t0, rho, b, e, omega), quadratic limb darkening from
  (q1, q2) with batman's own Mandel & Agol formulas (Hastings K and E,
  Bulirsch Pi), the exposure averaged over the declared supersamples, juliet's
  dilution and mean-flux convention, its Gaussian likelihood with white-noise
  jitter, and its rejections (b > 1 + p, a non-transiting inclination).
- The priors (juliet's transforms, bit for bit), the photons (+/-3 delivered
  durations, other signals masked), the seed and dynesty 3.1.0's dynamic
  nested sampling with the declared settings: multi-ellipsoid bounds, random
  walks, live points, dlogz_init, n_effective, checkpoint cadence, `nthreads`
  proposals per queue, and juliet's equal-weight resampling.

### Where it departs

- **Only photons that can see the planet are modelled.** Along the orbit
  d^2 = r^2 (sin^2 psi + cos^2 psi cos^2 i) exactly (psi from inferior
  conjunction), and batman sets the back half to d = 100. An arc scan of the
  front half with the smallest orbital radius on each arc, then the same
  bound over the remaining hull, gives a time window around each conjunction
  outside which d >= 1 + p for every exposure sample: batman's flux there is
  exactly 1, which the kernel leaves in place.
- **One compiled likelihood** (`transit_likelihood.c`) per proposal instead of
  juliet's Python and batman's full-curve calls: orbit constants once per
  model, the separation from (sin E, cos E) without the true anomaly, Kepler's
  equation started from a nearby sample's solution (third-order predictor,
  sine and cosine carried by rotation, Newton; batman's bracketed solver
  starts each run and catches failures), batman's limb-darkening branches for
  fully-inside and limb-crossing samples evaluated in batches with
  Accelerate's vForce logarithms and arc cosines.
- **dynesty's own pool** keeps the likelihood and prior transform in each
  worker; juliet pickled its fit object (about 185 kB) into every proposal.
  The posterior's per-sample likelihoods are recomputed in the pool, not
  serially.
- **No Lightkurve/astropy import** for preparation.

The model's numbers therefore agree with batman to rounding, not bit for bit:
log-likelihoods to about 1e-10 (1e-15 relative), model fluxes to 1e-14, and to
1e-11 for grazing geometries, where the limb steepens the separation's
rounding. The nested sampler's comparisons resolve that only at near-ties, so
the sampler follows production's path, the same calls and the same posterior
samples bit for bit, unless a comparison falls within about 1e-9 of its
threshold. That held on eight of nine production fits; on the
38,815-photon PSF fit one near-tie went the other way after 14,250 dead
points, and that fit's posterior is production's only statistically
([validator native fit](keplerX-2026-11-02-validator-native-fit-speed.md)).

### Checks

`tests/test_golden_validator.py` and `tests/test_e2e_validator.py` (an
M-dwarf signal of the M/K survey, 142 native photons): preparation bit for bit
against production's arrays; juliet's prior transform bit for bit; juliet's
log-likelihood of 100 posterior and 150 prior draws to 1e-8 (1e-9 relative)
with the same rejections; skipped photons exactly 1 when modelled everywhere;
and the whole fit with production's seed reproducing production's posterior
(sha256 of every sample), call count, evidence and review clock.
`golden/validator.json` was written once from production and juliet by
`validator/juliet_reference.py` under the frozen V3 runtime.

### Limits

- Supported: juliet's one-planet, one-instrument transit model as
  `post_boi.native` configures it (`dynamic_dynesty`, quadratic limb
  darkening, the prior distributions used by the campaigns); anything else is
  refused.
- The compiled kernel is qualified on the arm64 Mac with Apple Clang 17.
- Geometries whose posterior transit fills the native window (grazing,
  low-density or very eccentric runaway fits) model most photons, so their gain
  is the per-sample kernel's alone.

## Validator probability engine (`validator/probability/`)

A copy of `spacelab/fast_triceratops` (commit 1c04e9be2, 26 October 2026),
made on 2 November 2026 to be optimized inside KeplerX; the original stays
unchanged for the accepted campaigns that froze it. Imports name
`validator.probability` and the post-BOI modules `validator.*`; nothing else
changed in the numerics.

CPU acceleration of pinned TRICERATOPS `d20875c` and of `TRICERATOPS_KEPLER_V3`
with the same `calc_probs` outputs (lnZ, probabilities, FPP, NFPP). Float64
only: Apple GPUs have no float64, and the tested float32 likelihood biased
planet-scenario lnZ by +0.2 to +0.36.

- `accelerated()`: stock route. `kernels.py` takes every model flux from
  pytransit's own compiled kernel, vetoes binaries before their primary, and
  fuses dilution and cost; `scenarios.py` replaces the per-draw LDC loops.
- `accelerated_v3()`: V3 route (patched runtime). `joint.py` fuses the quarter
  likelihood and skips draws whose weight is provably exactly zero, resolved in
  rebuilt scenario functions; bitwise equal to upstream. `certified=True` also
  skips weights below e^-50 of the best (lnZ change below one ulp).
- `engine()`: static V3 scenarios adapted from TRICERATOPS (`engine/`)
  with exact kernels and shared splines; what queued campaigns use.
- `workflow`: experiment runner with preserved retries and receipts. A campaign
  frozen by `validator.protocol.freeze` holds the whole `validator` package,
  this engine included, in its `source/`; experiments run from that copy
  (the original ran two separately named snapshots, `post_boi` and
  `fast_triceratops`). This arrangement has not yet run a campaign.
- `bench.py` replays captured stock calls against a kernel change and logs it
  to `KeplerX/results/validator-probability-hill-climb.csv` (the original's
  history included); captures go to the runtime scratch folder.
- `gold.py`: the offline Kepler-10 b case (`tests/fixtures/validator-kepler10-triceratops`).

Not copied: the batman orbit-kernel experiment (`native.py`), superseded by
`validator/transit_likelihood.c`, and the original's unit tests. The check is
`tests/test_golden_validator_probability.py`: upstream TRICERATOPS against this
engine on Kepler-10 b as a three-quarter joint likelihood, every lnZ bit for
bit. It needs the frozen V3 runtime (TRICERATOPS with the V3 patch, pytransit):

```text
FROZEN_V3_PYTHON -m pytest --noconftest tests/test_golden_validator_probability.py
```

Method, controls and measurements of the original: `fast-triceratops*.md` in this folder.

## Tests (`tests/`)

KeplerX keeps two kinds of tests and no unit tests (Adam, 1 November 2026).

- **Golden:** a whole stage, run on fixed real inputs, must reproduce a
  reference that does not move with the code: the vendored upstream
  implementation (bit for bit, or to a stated tolerance where KeplerX
  adopted faster mathematics), a closed-form answer, or a stored golden
  record in `golden/`.
- **End-to-end:** the real pipeline on real fixture photometry, stage after
  stage, read back from its outputs, judged on scientific outcomes.

```text
run.sh check            (ruff, then every test)
```

### Golden

| Test | Reference | What must hold |
| --- | --- | --- |
| `test_golden_detector.py` | Upstream native Detector | The compiled kernels with upstream's convolutions, single-event veto and harmonic cleaning (the `reference="compiled"` search) return upstream's own Python search's candidates (`reference="native"`) bit for bit, both on the Detector's template bank, grid, segments, templates and repeat rule, on TrES-2, on two quarters of the quiet lane K dwarf with an injected strong and a near-threshold planet (iterative passes, harmonic removal), and on a curve cut into degenerate segments. On four quarters (the outlier veto, epoch cleaning, long periods folded in windows) the GPU-screened Detector returns the CPU-only Detector's candidates bit for bit. On TrES-2 and the two-quarter K dwarf the Detector's whitening, summed by FFT, equals upstream's convolutions to 1e-12 of each template's largest value. |
| `test_golden_vetter.py` | Upstream native Vetter | With batman's limb-darkening rule the compiled Vetter gives every metric and every metric error of the native one, bit for bit, on the nine stored signals of the failing curve (both on upstream's whitening). With the adopted integral, every component keeps its outcome and the transit counts and MES stay equal; on the Detector's whitening, summed by FFT, every component keeps its outcome, transit counts stay equal and MES agrees to 1e-9. |
| `test_golden_localizator.py` | Native Localizator, VizieR's Gaia table | On KIC 12644822's quarter-1 pixels and KOI K00791.01's clock, the accelerated localizer reproduces the native centre recorded on 27 October within 1e-4 of its error. The ESA archive's Gaia cone becomes VizieR's I/355/gaiadr3 table value for value, and a field table holding the cone's sources returns that table bit for bit. |
| `test_golden_preparation.py` | `golden/preparation.json` | Production preparation of TrES-2 and of the lane K dwarf keeps the same cadences, mask reasons and quarter levels. The vendored upstream files match their recorded checksums. |
| `test_golden_binary_gate.py` | `golden/binary-gate.json` | On the stored vetting metrics of six real signals the binary gate (`linear-score-binary-gate`) dismisses Kepler-840 b on corroborated evidence (a secondary deeper than a planet's occultation, held event by event, and a stellar companion's ellipsoidal variation), keeps Kepler-158 c with no piece (its fold secondary is no secondary event by event), keeps a weak synthetic planet whose odd/even t of 5.7 rests on four transits each (judged by Student's t), dismisses a blended binary by odd/even over a thousand transits each, keeps a 50-ppt giant planet found at three times its period on its deep primary alone, and dismisses a deep, grazing blended binary on its primary and secondary together. |
| `test_golden_calibration.py` | Closed forms, `golden/injection.json` | The information S/N equals the matched filter in white noise and adds quarters in quadrature with their own noise. The Kepler-band term and the Sun's habitable zone take their stated values. A fixed planet injected into the lane K dwarf through the flattening emulator gives the stored arrays and signal statistics. The efficiency fit recovers a known curve. |
| `test_golden_validator_probability.py` | Upstream TRICERATOPS d20875c with the V3 patch | The Validator's copy of the probability engine gives upstream's lnZ for every scenario bit for bit, and the same probabilities, star table, FPP and NFPP, on Kepler-10 b as a three-quarter joint likelihood with one neighbour's declared response. Needs the frozen V3 runtime (`FROZEN_V3_PYTHON -m pytest --noconftest ...`); the pipeline environment skips it. |
| `test_golden_validator.py` | Production's prepared arrays and juliet, `golden/validator.json` | On a production M-dwarf signal (142 native photons), the Validator's preparation gives production's prepared arrays bit for bit; its prior transform gives juliet's bit for bit; its log-likelihood gives juliet's for 100 posterior and 150 prior draws to 1e-8 (1e-9 relative) with the same rejections, and for batman's circular branch, very eccentric orbits and central transits to 1e-8; photons left outside the transit windows are exactly 1 when modelled everywhere. |

### End-to-end

| Test | Chain | What must hold |
| --- | --- | --- |
| `test_e2e_planet_ephemerides.py` | `detector.program.run` | Two planets injected into the quiet lane K dwarf, the second at 3/2 of the first's period on another epoch, are both found, each period within 2e-5 and mid-transit time within half a cadence of the truth (upstream's bin-edge epoch missed the first by 0.013 d). |
| `test_e2e_long_transit_planet.py` | `detector.program.run` | A planet with 0.31-day transits on a 70-day orbit, injected into the quiet lane K dwarf through the flattening emulator at information S/N 8.3, is found exactly (the scorer's match); upstream's bare templates found nothing on this curve. |
| `test_e2e_sap_lane.py` | `calibration.lane_twin.sap_lane`, `detector.program.run` | The quiet lane K dwarf's stitched SAP, prepared as the `sap_flat` view (flattened here as KBonus flattened PSF) and twinned as the SAP calibration sample twins a case, keeps every quarter; the long-transit test's planet, diluted by the star's share of the SAP flux and passed through the flattening emulator, is found exactly. |
| `test_e2e_known_planet.py` | `run.py batch`: prepare, detect, vet, BOI record | TrES-2 b is found on every observed transit with fresh-process and long-lived-worker stages alike; vetting measures its depth; the BOI record stays unresolved without pixels; outputs live in `results.sqlite` only. |
| `test_e2e_localization.py` | Vet, then localize with the parent TPF and Gaia | KOI K00791.01's transit light is centred on KIC 12644822 (its lane source), to well under a Kepler pixel, with a successful fit; the same signal, dismissed by vetting, opens no pixels. Requested from MAST (an offline bucket serving the same file), the quarter holding the source is streamed into memory and localizes exactly as the stored file, with its URL, size and sha256 kept; the parent's quarter without the source, its short-cadence file and its tarball are not fetched. |
| `test_e2e_failures.py` | Prepare, vet and packet loading on broken inputs | A missing input leaves an ERROR record and its output is never reused; a changed packet is refused; a curve whose fits fail keeps all nine signals with their errors and accepts none. |
| `test_e2e_calibration_study.py` | Build, register, `calibration.study` (batch, score) | A miniature sample on the lane K dwarf (its original, a detectable planet, a planet below threshold, an inverted copy) runs as a study: the detectable planet is found exactly, the weak one is not, the inverted copy holds no planet, and the study ends SCORED; running it again starts no new attempt. |
| `test_e2e_validator.py` | `validator.preparation`, then `validator.native_fit` | The same M-dwarf signal, prepared and fitted with production's seed and dynesty settings, reproduces production's posterior sample for sample (sha256), its sampler call count, evidence and review clock. |

### Fixtures

| File | What it is |
| --- | --- |
| `tres2-q0.*` | TrES-2's KBonus PSF Flat, quarter 0 (known planet) |
| `vetting-failure.*` | A stored curve whose vetting fits fail, with its nine detected signals |
| `lane-k-dwarf-2103903753019155968-four-quarters.*` | A quiet PSF Flat lane K dwarf (G 14.3, 106 ppm in 3 h), quarters 2–4 and 6 with the raw stitched flux |
| `binary-gate-signals.json` | Stored vetting metrics, with binary evidence, of Kepler-840 b, Kepler-158 c, a held-out weak synthetic planet, a held-out blended binary, a development giant planet found at three times its period and a deep, grazing development blended binary |
| `lane-k-dwarf-2103903753019155968-four-quarters-sap.*` | The same K dwarf's stitched SAP and its error on the same cadences (LC2 shard 0603) |
| `kic12644822-q1-target-pixels.*` | KIC 12644822's quarter-1 target pixel file and its transfer record |
| `kic12644822-q1-prepared.*` | The KBonus PSF Flat of KIC 12644822 (Gaia DR3 2133476080319167104), quarter 1, prepared |
| `gaia-dr3-12644822-*` | The Gaia DR3 cone around that TPF, as the ESA archive and VizieR return it |
| `validator-m-dwarf-2126055064030395392-1/` | A production post-BOI case (M/K SAP signal, 50.4 d): candidate record, native photons and priors |
| `validator-kepler10-triceratops/` | Kepler-10 b (KIC 11904151) as upstream TRICERATOPS's example: light curve, stars, background, pixel geometry |

These fixtures are software controls, not the calibration sample.
