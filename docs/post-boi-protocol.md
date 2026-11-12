# Standard TRICERATOPS follow-up

`TRICERATOPS_KEPLER_V3`, local implementation from October 21, 2026. V3 retains
quarter responses in one joint likelihood. Its KBonus preparation, known-object
calibration and ephemeris rule of October 25 are in
[quarter likelihood](quarter-likelihood.md). Production adoption requires a
threshold calibration at Kepler long cadence; until then a V3 numerical pass is
not a statistical validation.
Use one pipeline for survey and M/K BOIs. Historical V1/V2 campaigns remain
immutable and run only from their own frozen copies. Do not relabel their
results or rerun accepted studies.

## Scientific method

Authenticate source, exact run-scoped signal, native photons, masks, time system,
exposure and true extraction aperture. Keep the calibrated detector/vetting,
alias/harmonic, spacecraft, source-ownership and dated exact-signal literature
checks. A BOI is not a physical admission; a missing query result is not novelty.

Prepare native SAP or a documented source-corrected native flux using the same
code. Do not assume a PSF_FLAT label establishes its dilution convention.
Record the original raw photons and every normalization/correction. For SAP,
pass `dilution_corrected=False`; for already source-corrected flux, pass True.
Never apply CROWDSAP twice. The M/K detector's prepared array is provenance,
not a substitute for the authenticated native likelihood input.

Fit period and epoch jointly with standard juliet/batman. Inspect sampler
convergence, residuals, coverage, timing/aliases, prior boundaries and radius.
Adopt the accepted posterior-median clock before FPP. Native fitting uses a
window of +/-3 supplied durations; detrending protects +/-1 supplied duration
plus half an exposure. Choose a longer protected window when warranted by
clock/duration uncertainty before fitting. There is no duration/k/density box
in this wrapper. Review protection on actual observed photons and quantify
detrending transfer. An approximate contact outside the mask or a negligible
ingress loss is not an automatic hold. For a material effect, preserve the first
fit and prepare a separate duration/clock-aware generation. Do not use FPP to
choose a fit or mask. See [dependency and preparation repairs](validation-repairs-20261021.md).

The [measured proposal-domain census](../results/boi-validation/preparation-domain-census-20261020.json)
finds p01/p99 durations of 0.6204/12.4249 h in the delivered 125 survey signals and
0.84/7.20 h in 663 M/K BOI proposals. Full survey TRANSIT_LIKE proposals extend
to 13.76 h. Broad survey BOI labels also include unrelated/unknown and shallow
window hypotheses; their default durations are not transit measurements.
Percentiles describe preparation demand, not physical support or tail vetoes.

Supply candidate priors and their source explicitly. Use reported mass/radius
errors, including covariance/asymmetry where available. Missing errors need a
justified uncertainty model; no automatic 10%/6% replacement is installed.
`native.stellar_density` offers a first-order SI Gaussian approximation from
reported symmetric M/R errors; it has no density ceiling. A direct density
posterior/prior can instead be supplied to juliet. Priors for P, central epoch,
p=Rp/Rstar, b, rho, limb darkening, eccentricity and noise are recorded in its
ordinary prior file. Radius ratio can exceed 0.3 and density 50 rho_sun. Check the
physical body's radius separately; host size is not a substitute for that check.

Prepare +/-2 fitted durations and at most 200 common phase bins. Retain the
original contributing photons, quarter identity and reviewed native noise.
The joint Gaussian likelihood scores every source hypothesis in the supplied
observed flux units. It retains within-bin residuals and quarter response;
check the common-bin timing approximation on original cadences. Supply
`probability.noise` and its `noise_basis`; folded errors cannot substitute for
native uncertainties. Formal measurement errors alone need a noise-adequacy
review. The existing `native_review.probability_noise_sigma` and its basis
remain the stock scalar secondary-screen reference, with its limitations
disclosed. V3 introduces no covariance/GP, jitter multiplier or stellar bank.

## Minimal scene adapter

`KeplerScene` binds exact source rows and measured source-quarter geometry
to upstream public arrays. This avoids the upstream constructor resolving the
parent TPF owner as the host of a background source. A narrow upstream patch
passes the same sampled physical model through the joint likelihood. Priors,
samplers and physical curves remain unchanged. All
potentially signal-producing sources need explicit supported M/R/Teff; missing
values cause a clear readiness failure rather than silent scenario loss.
No TIC, Solar-metallicity measurement, TIC EBV, parallax-SNR 5 gate, 4096-particle
bank, paired simulations or MIST conversion is required by this wrapper.
A physical target distance/parallax is still needed by upstream bound priors.

Supply documented quarter fractions (`measured_kepler` for measurements,
`calibrated_kepler` for explicitly modeled PRF fractions). Each retained photon
uses its own quarter response. The source-capability screen uses the largest
response among retained photons as a conservative bound, so averaging cannot
remove a seasonally capable source. The quarter mean remains only in the
upstream scalar secondary-screen conversion and reference output fields.
No PRF wings or pointing prior is introduced. The only quarter rejection is the
KBonus flux-closure gate (`post_boi.kbonus`): a calibrated response must predict
the observed median SAP flux within 20%, and a case needs two passing quarters.
Review per-quarter noise with `kbonus.quarter_noise`. Adopt photon selection
independently of FPP; preserve excluded quarters and review timing, depth and
source ownership before validation.

Supply one actual upstream-compatible field-population file. Acquire it with
the official TRILEGAL/Gaia utilities, preserving the raw response, request,
coordinates, area and method. Retry failed downloads without changing scientific
inputs. A missing population must never cause background scenarios to disappear.
Check the service's actual coordinate units: TRILEGAL's equatorial RA field is
in hours; its Galactic longitude/latitude fields are in degrees. The current
upstream helper passes RA through unchanged. Convert it explicitly or submit
authenticated Galactic coordinates, and preserve the exact request. Choose the
faint limit from the observed aperture depth, target magnitude and dilution;
the helper's fixed T=21 limit can miss capable contaminants of faint BOIs.
Choose its scientific population model explicitly; current upstream's default
Gaia route assumes a dwarf sequence and Solar metallicity. Those are upstream
assumptions, not measured properties of every field star.

Parse accepted result links as HTML, including unquoted href attributes. Save the
first accepted output URL before any retry; poll that job rather than submitting
another population because a link parser failed. Preserve failed and unused
attempts explicitly and never select a population by FPP.

The `mission=Kepler` model uses upstream Kepler limb darkening and the supplied
measured exposure. Upstream retains TESS-based magnitude/companion-flux
approximations even in Kepler mode. Preserve that limitation; do not relabel Tmag
as measured Kepler magnitude or claim every crowded field is thereby modeled
exactly. Examine its relevance to the candidate. Real contrast curves keep their
measured filters; missing imaging supplies no measured companion exclusion.

## Candidate files and commands

Copy `post_boi/candidate-template.json` into a new internal campaign workspace.
Null fields are deliberately undecided. A researcher can complete the
review within Adam's authorized scope; a review file is not a new approval
ritual. Record evidence before filling accepted=True.

- `photometry`: NPZ with scalar string source_id, time_bkjd, flux, flux_err,
  quarter, and authenticated boolean valid mask; aligned arrays, actual photons.
- `scene`: CSV, target first, exact numeric Gaia/TIC IDs in ID, Tmag, mass, rad,
  Teff, plx, ra, dec, Jmag, Hmag, Kmag. Keep absent optional measurements blank.
  Preserve catalogue identifiers/measurement provenance alongside the table.
- `geometry`: NPZ with matching source_id order, quarter, and per-quarter
  xy_qN and aperture_qN arrays in absolute detector col,row. Use the source's
  aperture, not the parent's default mask. Proper motion/WCS provenance belongs
  in readiness. For measured_kepler/calibrated_kepler add normalized
  flux_fraction_qN for all rows, explicitly distinguishing measurements and models.
- `native.priors`: standard juliet JSON; direct p_p1/b_p1/rho parameters, free
  P_p1/t0_p1. Supply ordinary Kepler limb-darkening/noise/dilution parameters.
  Model/time units follow juliet: rho in kg/m3, jitter in ppm, BKJD_TDB in days.
- `native.fit_options`: explicit upstream sampler/options. Convergence criteria
  must match that sampler; do not treat correlated ensemble walkers as independent
  chains or claim Rhat for a sampler without chains. Preserve full upstream output.
  `native.starting_point` is supplied for MCMC as required by juliet, null for
  nested sampling. In juliet 2.2.10, use null nthreads for serial MCMC; a supplied
  nthreads allocates nthreads-1 pool processes and therefore must exceed 1.
- `native_review`: adopted P/epoch/duration and fitted observed-convention depth,
  folded-noise sigma/basis, accepted status and hash-bound evidence. `native` writes an undecided template;
  fill it after inspecting mixing, residuals, geometry and exposure accuracy.
- `readiness`: source_id, signal_id, accepted and evidence receipts containing
  path/sha256; cover spacecraft, aliases, scene/stellar/dilution/noise adequacy,
  extraction identity and dated catalogue checks. Missing physical follow-up is
  explicit and cannot be silently treated as clearance.
- `probability.noise`: NPZ with source_id and original-photon-aligned time_bkjd,
  sigma and applied_fraction. Sigma is reviewed native noise in the normalized
  observed flux convention. Applied fraction is one for raw SAP, or the actual
  fraction already divided out for corrected photometry. Record its provenance
  and noise assumptions in `probability.noise_basis` and readiness evidence.

`post_boi.aperture.physical_aperture` authenticates a saved KBonus mask through
cadencewise replay of accepted SAP values from the original parent pixels. It
checks both documented background conventions against a float32 arithmetic
bound. A source `TPF_COL/TPF_ROW` header can conflict with the saved array's actual
frame: preserve the header and bind the array to the replay-authenticated physical
origin. Gaia coordinates use that WCS's own physical origin. Never resolve this
by shifting to flux centroids or silently changing the photons. The full replay
must support the chosen mapping; matching stamp shapes/cadences alone is weaker.

Run through the pinned `post_boi/requirements.txt` environment:

```bash
uv venv --python 3.12.13 /path/to/new-runtime
uv pip sync --python /path/to/new-runtime/bin/python post_boi/requirements.txt
```

Install the existing dependency repairs, then apply the quarter patch to this
new, unused runtime only (through `uv run --python`):

```bash
python -c 'from post_boi.dependencies import patch_quarter_runtime; patch_quarter_runtime()'
```

The installer checks both original and patched source hashes. Frozen runtimes
are never patched; V1/V2 records keep their original source and interpretation.

The following commands use that runtime's Python (through `uv run --python`):

```bash
python -m post_boi.prepare candidate.json prepared
python -m post_boi.native candidate.json prepared native-fit
python -m post_boi.campaign freeze candidate.json new-campaign
cd new-campaign/source
python -m post_boi.campaign verify ../plan.json
python -m post_boi.campaign resume ../plan.json
```

Use a durable launchctl Interactive agent on the Mac, with unique stdout/stderr
logs and a sleep inhibitor tied to the coordinator. Set BLAS/OpenMP threads to 1
before importing numerical libraries. The stock vectorized integration receives
the declared Numba budget; no custom renderer or dynamic thread-sharing engine
is installed. Measure actual performance rather than equating allocated threads
with continuous saturation. Do not run competing candidates/benchmarks while
production is active. Server production is run separately.

## Counts, recovery and outcomes

Freeze 20 independent final-clock primary seeds and 1,000,000 attempted draws per
scenario. The wrapper calls the patched `calc_probs` with one likelihood across
quarters for each shared draw; no custom proposals, phase
interpolator, radius-CDF, covariance projector, clipping-bound engine or stellar
particle population is installed. Complete first 2; if both FPP>=.10 or both NFPP>=.01,
stop allocation with REMAINS_PC and retain both. Otherwise finish all 20 including
the first 2. These counts, thresholds and early allocation policy are unchanged.
Recovery diagnostics are normally 0, exactly 3 paired first seeds only for a frozen
justified timing sensitivity. Hold non-timing inputs fixed; never pool clocks.

Resume verifies frozen inputs/source/environment and saved complete outputs.
Upstream exposes no within-scenario/RNG checkpoint; an interrupted repeat starts
a new preserved attempt at its same seed, while completed repeats are reused.
Do not delete failures, alter seeds, omit alternatives or retry for a favourable
result. Use one coordinator per campaign, and do not resume after an explicit stop.

Save every upstream scenario table, lnZ array, result and raw object, including
failed-attempt logs. MeanFPP<.015 and mean NFPP<.001 require all 20; report every
repeat, mean, sample SD, range and unfavourable indices. Stock output does not
provide full per-draw weights, their ESS/concentration or the former custom
omitted-evidence upper bounds. Do not claim those audits or import old values.

Numerical success still requires native/radius, RUWE<=1.4, source localization,
dilution, companion/imaging and dated exact-signal closeout before statistical
admission. Run `python -m post_boi.ephemeris` on the adopted clock; a low-order
(1:1, 2:1, 1:2) match on another star keeps the candidate unresolved until a
pixel-level analysis explains it. Investigate the reference 1.5-arcsec companion association; RUWE alone
does not exclude binarity. An RV mass is unnecessary for statistical validation;
RV/dynamical confirmation is a separate same-source/signal orbit with appropriate
diagnostics. Retain REMAINS_PC when evidence is unresolved; deny FP only with a
reproduced mechanism. Publication/novelty requires the dated literature record.

## Sources and scope of revalidation

- [Official API](https://triceratops.readthedocs.io/en/latest/user/api.html) and
  [Kepler example](https://triceratops.readthedocs.io/en/latest/tutorials/kepler_example.html).
- [Pinned official source](https://github.com/stevengiacalone/triceratops/tree/d20875c27d6761349cf032728bb4a59ec732a84d), checked against remote HEAD.
- [Collier 2026](https://arxiv.org/html/2608.21560v3), duration-scaled preparation
  and standard native/probability methods; no six-hour universal cutoff.
- [TFAW K2 survey 2023](https://doi.org/10.1093/mnras/stac3087), folded K2 data,
  stellar/aperture inputs and imaging; its additional vespa/multiplicity analysis
  is not silently added to this programme.
- [juliet](https://juliet.readthedocs.io/en/latest/), with candidate priors and
  sampler controls rather than the old native wrapper's boxes/optimizer.

See the [149-row parameter decision record](../results/boi-validation/standard-v2-parameter-decisions-20261020.json)
and [verification report](../results/boi-validation/standard-v2-verification-20261020.json)
for exactly what changed. These controls establish adapter/preparation behavior
and agreement with upstream; they are not a new empirical occurrence/reliability
calibration of all faint KBonus/M/K populations.
