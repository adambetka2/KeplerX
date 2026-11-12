# Post-BOI preparation parameter audit

Audited on October 20, 2026. This report concerns the active local KIC4904094
generation, its shared scientific kernels and the local queue preparer. It does
not reopen the sealed detector/vetting study or change historical admissions.
No scientific code, input, count, seed or admission threshold was changed.

The complete machine-readable inventory is
[`preparation-parameter-audit-20261020.json`](../results/boi-validation/preparation-parameter-audit-20261020.json).
It records 149 input/control groups, source hashes and line locations, a
comparison for each group, and references. Exact catalogue values, the two
75-key TRILEGAL requests and all declared seeds are preserved in its linked
input snapshot. A separate static appendix records function defaults and numeric
literals, including inactive legacy functions; a literal in that appendix is
not automatically an active scientific requirement.

## Findings that affect the interpretation

The wrapper contains restrictions that are not TRICERATOPS requirements.
The most direct demonstrated problem is the fixed native duration/window/noise
scope. It prevented KIC5079515's 9.288-hour event from entering an appropriate
native fit. The event has not been rejected astrophysically.

The radius, density and clock bounds also need domain justification. For
example, the native upper limit k=0.3 permits only about 3.3 Earth radii around
a 0.1-solar-radius star. A 0.1-solar-mass, 0.12-solar-radius star has density
about 58 solar densities, beyond the native limit of 50. These are illustrative
calculations, not classifications of a specific M/K BOI. In the current fit,
14.29% of the retained radius-ratio samples lie above 0.27, below the fixed 0.3
ceiling. That establishes proximity to a prior boundary; its effect on FPP has
not been measured by a changed-model experiment.

The native preparer takes the first ExoFOP row containing mass and radius and
always assigns 10% mass and 6% radius errors. The scene constructor instead
uses reported errors when available. The native implementation therefore does
not implement a missing-error-only fallback. Current C12 lacks those measured
errors, so the fallback applies there; this does not justify its unconditional
use across the queue.

The larger methodological issue is that the probability calculation is a
custom TRICERATOPS-derived model. It changes stellar uncertainty treatment,
photometry, PRF response, likelihood, conditioning and integration. Numerical
agreement with that same model is useful but does not demonstrate agreement
with stock TRICERATOPS or empirical probability calibration on Kepler/M/K
controls. Inspected records contain primitive and internal parity checks; a
matched end-to-end external validation of the entire modified model was not
established. Bias from the untested choices could raise or lower FPP.

## Comparison with primary sources

[The official API](https://triceratops.readthedocs.io/en/latest/user/api.html)
expects transit-centred time, normalized flux, flux uncertainty, period and a
stellar field. It exposes sample count, priors, contrast-curve inputs and
exposure integration. It does not prescribe our separate native fitter.
[The official Kepler example](https://triceratops.readthedocs.io/en/latest/tutorials/kepler_example.html)
already supports Kepler, quarters and aperture pixels. Kepler support itself
does not require replacing the evidence engine.

[Collier et al., inspected v3](https://arxiv.org/html/2608.21560v3),
sections IV.1.3, IV.3, IV.6 and IV.8, provides these relevant examples:

- Duration-scaled fitting and probability windows; approximately 200 probability
  bins.
- Rhat 1.03 and epoch ESS 100 in its global recovery stage.
- A circular time-domain fit, measured stellar errors with conditional 10%/6%
  missing-error assumptions, and a quantile-width definition of radius-ratio SNR.
- Repeated probability integrations and mean FPP/NFPP criteria.

Those are an implementation in that study, not universal API requirements.
Our native SNR uses standard deviation; the reference uses half the 16th–84th
percentile width. Those definitions differ for skewed/grazing posteriors.
KIC3935011 fails both definitions (2.867 versus 2.817), so that difference does
not explain its hold. Current KIC4904094 passes both (6.045 versus 5.337).
Our fixed two-hour epoch box and detector-resolution period box also differ
from its timing priors. The user-approved twenty-primary/zero-or-three-clock
protocol remains a separate prospective policy.

[Giacalone et al. 2021](https://arxiv.org/abs/2002.00691), sections 2 and 6,
describes the scenario framework, stellar prescriptions and repeated FPP
calculations. The no-imaging 2.2-arcsec convention has an upstream origin;
it is not a measured completeness guarantee for every KBonus extraction.
[Pinned official code](https://github.com/stevengiacalone/triceratops/blob/d20875c27d6761349cf032728bb4a59ec732a84d/triceratops/triceratops.py)
also distinguishes aperture SAP from already dilution-corrected flux. That
distinction is essential when comparing our prepared data to stock results.

## Native inputs, bounds and preparation

| Parameter or choice | Executed setting | Assessment |
| --- | --- | --- |
| Source/signal | Exact Gaia source, authenticated KIC/TIC; local selector is survey Rust rank 1, PSF_FLAT proposal | Identity checks are necessary; selector is not a generic M/K signal reader |
| Photometry | Native SAP_QDIAG; PSF_FLAT kept separately | Preserve delivered SAP and dilution convention |
| Validity | QUALITY=0, finite time/flux/error, error>0 | Declared strict mask; inspect lost events |
| Other signals | Empty mask list in current single-signal case | Needs an explicit multi-signal implementation |
| Timing inputs | Supplied P, epoch, duration and detector period resolution | Measurements/provenance, not arbitrary constants |
| Event half-window | 0.3 d = 7.2 h | Unsupported as a universal window |
| OOT start | Absolute phase 0.12 d = 2.88 h | Contaminated by sufficiently long transits |
| Coverage | At least 8 event-window points; at least 4 on each side outside OOT boundary | Local quality rules, not physical limits |
| Primary overlap | At least one cadence within supplied duration/2 plus exposure/2 | Coverage bookkeeping |
| Quarter scale | Median valid flux | One normalization choice |
| Baseline | Multiplicative line, x=phase/0.3 d; proper Gaussian widths 0.1, 0.5 | Custom nuisance model |
| Additive jitter | MAD from OOT residuals; quadrature excess over formal error | Fixed white-noise approximation |
| Shared noise | Uniform ln(scale) between -1 and 1 | Custom finite noise prior |
| Duration | 0.0104–0.25 d, about 15 min–6 h | No documented universal justification |
| Radius ratio | Log-uniform k between 0.005 and 0.3 | Can truncate small-host/grazing systems |
| Density | 0.1–50 solar densities | Can exclude legitimate host classes |
| Impact | Uniform b/(1+k) between 0.001 and 0.999 | Geometric prior; grazing is permitted |
| Period | Uniform within twice the detector resolution on either side | Resolution is not a measured timing uncertainty |
| Central epoch | Uniform within two hours | Candidate-specific adequacy needs checking |
| Orbit | Circular, omega=90 degrees | Literature analogue, not universal adequacy |
| Native dilution | Unit dilution; apparent transit | Final physical size requires the scene |
| Limb darkening | Triangular q1,q2, each 0.001–0.999 | Established parameterization; numerical endpoints are ours |
| Density prior | Independent positive M/R distributions, 2049 log-density nodes; 1024/2048-node quadrature | Numerical check does not validate stellar assumptions |
| Native errors | Always 10% M and 6% R in preparer | Implementation mismatch when measured errors exist |

[Kipping 2013](https://arxiv.org/abs/1308.0009) supports triangular limb-darkening
sampling. It supplies no justification for our duration or density boxes.

## Inference and probability packet

| Parameter or choice | Executed setting | Assessment |
| --- | --- | --- |
| MAP initialization | k=0.08, density M/R³; impact fractions 0.5, 0.8, 0.94 | Numerical starts |
| Optimizer | Nelder–Mead, 2500 then up to 4000 iterations; xatol 1e-5, fatol 1e-4 | Local numerical choice |
| Native MCMC | Four independent 24-walker ensembles; equal Stretch/DE moves | Different from reference sampler, not inherently wrong |
| Endpoints | 3000, 6000, 12000, 24000; original seeds/RNG continuation | Compute policy, not evidence of nonplanetary nature |
| Burn-in | min(1000, steps/3) | Must be assessed with mixing |
| Adopted native gate | All rank Rhat<=1.03; central epoch bulk ESS>=100; median(k)/std(k)>3 | Numerical/precision gate; last statistic differs from reference |
| Timing profiles | 65x65 local period/epoch grid; quarter offsets ±1.5 h in 1.5-min steps | Conditional local diagnostics, not global alias clearance |
| Native exposure | Actual 270-frame acquisition; 54 grouped frame centres | Appropriate measured acquisition model with checks |
| Extra phase groups | 0.6-day chronological quarter cells, merged if fewer than three samples | Custom baseline/packet construction |
| Probability template | Accepted native MAP shape | Shape uncertainty not marginalized |
| Probability noise | Native MAP scale and quarter jitter fixed | A conditional likelihood, not full noise marginalization |
| Detrending | Same nonlinear affine-MAP transform for observed data and alternatives | Internal equality is necessary, not external calibration |
| Fold/window | Adopted final P/epoch; ±2 fitted durations | Duration-scaled preparation |
| Bins | 200 equal total phase bins, empty bins omitted; inverse-variance weights | C12 has 178 occupied bins |
| Covariance | Full pooled first-order Jacobian covariance | Includes induced transform correlations, not a stellar-activity model |
| Covariance projection | Keep eigenvalues >1e-6 of largest | Information compression needs sensitivity checks |
| Group reduction | Only complete independent groups with zero bin projection | Reviewed equivalence; Q1 no-contribution context retained |
| Exposure/rendering | 54 samples and 800 phase-grid nodes | Different approximation errors; C12 grid spacing is 204 s |
| Noise quadrature argument | 128 | Inactive with the selected fixed-noise likelihood |

Finite exposure effects are physical and must be modeled, especially for Kepler
long cadence. [Kipping 2010](https://arxiv.org/abs/1004.3741) explains the issue;
it does not prescribe exactly 54 samples or an 800-node interpolation grid.
The current internal renderer parity compares the retained grid calculation
with itself, so it does not establish that grid error is negligible everywhere.

The strict MAP JSON writer can fail when a rejected start has log posterior
minus infinity despite a different finite start. This bookkeeping defect must
be distinguished from unsupported physical likelihood. No serializer or model
was changed during this audit.

## Stellar scene, photometry and priors

| Parameter or choice | Executed setting | Assessment |
| --- | --- | --- |
| Resolved census | Complete retained scene; 31 sources for C12; 60-arcsec TIC query | Field completeness/outside-field response need review |
| Cross-identities | Exact Gaia DR3; DR2 association within 500 mas; explicit unique TIC duplicate pointers only | Keep identity safeguards |
| Target family | TIC in C12; Gaia/empirical alternatives implemented | Evidence-based choice before FPP |
| Catalogue bank | 4096 particles, seed 20261011 | Finite-bank uncertainty separate from evidence draw count |
| Catalogue distance route | Positive parallax SNR>=5 | Our route restriction, not a universal validation rule |
| Extinction route | TIC EBV required; Av=3.1 EBV | Custom BC requirement; missing EBV needs a legitimate alternative model |
| Catalogue M/R errors | Reported errors; missing-only 10%/6% | More faithful than unconditional native fallback |
| Missing other errors | Fixed conditional Teff/MH/EBV where only centres exist | Does not mean zero measured uncertainty |
| Gaia marginal shapes | Equal-half split Gaussians from median and 16/84 percentiles | Approximation, not the joint posterior |
| Stellar dependence | Independent M,R,Teff,MH,EBV,parallax | Missing covariance and consistency checks matter |
| Metallicity | TIC MH else Gaia GSP-Phot bounds; no silent Solar fill | Custom requirement; catalogue systematics remain |
| Distance conversion | Positive Gaussian parallax then inverse; high-SNR route | Not suitable as a universal low-SNR solution |
| Empirical fallback | TRILEGAL G width 0.1 mag, BP-RP width 0.15 mag, measured parallax likelihood | Custom smoothing, not measured photometric errors |
| Sparse empirical priors | Some ESS near 1–2; retained | More integration repeats do not create additional prior stars |
| Measured seismic override | One exact APOKASC3 Gold giant; 4096 particles | Source-specific evidence, with missing covariance explicit |
| BC source | Pinned MIST tables, linear four-dimensional interpolation | Correct passband names do not establish atmospheric accuracy |
| BC domain | Teff 2500–1,000,000 K, logg -4–9.5, [Fe/H] -4–0.75, Av 0–6 | Rectangular tabulated domain, including padding; not physical validation |
| Kepler colour | Observed Gaia G + BC_G - BC_Kp | Custom conditioning; physical-G residual remains diagnostic |
| Population colours | Exact 12-field physical pairing; Kepler AB minus Vega 0.122666 mag | Exact comparison protects identity; separate simulations may never pair |
| FPP limb darkening | Nearest Kepler table row, including nearest-edge lookup | Must inspect cool-star/gravity applicability |
| Composite flux | Observed Gaia G treated as total unresolved-component G | Custom conditioning; no composite SED refit |
| PRF core | Published channel-specific Kepler calibration, actual apertures/WCS | Appropriate instrument adaptation |
| PRF wings | Exponential median fit adopted; join 3.5–4.25 pixels | Extrapolated distant wings remain a scientific assumption |
| Pointing | Five equal quarter states: nominal, ±0.2 pixel in x/y | Conventional discrete prior, not measured pointing distribution |
| Unresolved source | Uniform-area disk, radius 2.2 arcsec | Upstream no-imaging convention, not measured exclusion |
| Spatial grid | 13x13, bilinear interpolation | Numerical approximation |
| PRF normalization | 64-pixel radius, 32/64/128 comparison, analytic tail | Converges the assumed tail rather than proving it |
| Imaging/MOLUSC | No candidate-specific contrast curve or MOLUSC population enters active C12 adapter | Missing follow-up does not become a measured non-detection |

[Gaia's own documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_gspphot.html)
describes joint inference and cautions about AP uncertainties/systematics.
Independent marginal draws are therefore an approximation, not Gaia's joint
stellar model. A hard failure for a metal-rich neighbor outside our BC grid
does not demonstrate that the neighbor or BOI is physically impossible.

The [published Kepler PRF](https://arxiv.org/abs/1001.0331) supports using actual
instrument response. It does not calibrate our remote exponential wings or
uniform five-atom pointing prior. Those need data comparisons.

## Population generation and probability controls

The exact TRILEGAL requests each have 75 fields, reproduced in the linked
snapshot, including otherwise inactive form defaults. The active selections
are TRILEGAL 1.6, 0.1 square degree, Gaia magnitude limit 22.5 or Kepler 26,
magnitude resolution 0.1, Chabrier lognormal IMF, inherited thin/thick disk,
halo/bulge prescriptions, binary_kind=0, and the measured line-of-sight dust
input. Retained binary fraction/mass-ratio form fields are not active binary
synthesis with binary_kind=0. Their presence must not be presented as a new
measurement of binarity.

| Parameter or choice | Executed setting | Assessment |
| --- | --- | --- |
| Scenario census | 15 target families + 3 per resolved neighbor | Retain all alternatives |
| Planet size prior | 0.5–20 Earth radii, breaks 3/6; host-mass-dependent slopes | Upstream prior, different from native k box |
| Inclination/periapse | Isotropic inclination; uniform periapse | Upstream statistical assumptions |
| Planet eccentricity | Beta(0.867,3.030) | Upstream population prior |
| Binary eccentricity | Power-law shape 0.2 if P<=10 d, 0.6 otherwise | Upstream population prior |
| Mass-ratio priors | Pinned empirical distributions; twin split q=0.95 | Full source constants in static appendix |
| Companion relation | Torres/cool branch at 0.63 solar masses; R floor 0.1, Teff floor 2800 K; cap at primary | Inherited limitations also need cool-star review |
| Background selection | Fainter/equal Kepler magnitude than target | Audit resolved bright blends and completeness |
| Background area | pi times radius squared; request area 0.1 deg² | Correct for declared circular spatial measure |
| Measured companions | Source-specific observational constraints, if actually available | No imaginary contrast curves or silent scenario exclusions |
| Importance mixture | 10% full prior, 90% targeted, with p/q correction | Numerical acceleration, not a physical veto |
| Planet targeting | Radius ±15%, duration ±0.2 h | Full-prior component preserves wider support |
| Binary targeting | Depth ±50%, duration ±0.3 h; 40 bisection steps | Proposal approximation, not final likelihood |
| Draws | 1,000,000 attempted per family | Count includes invalid and zero weights |
| Batch/render | 256/64 | Performance choices |
| Numerical bound | 48 log units; saved omitted-weight upper bounds | Does not cover prior, grid, model or all sampling error |
| Repeats | 20 final-clock primary seeds | Explicit user-approved prospective allocation |
| Recovery clock | Normally zero; exactly three if justified | Do not pool clocks |
| Early screen | Two full runs; both FPP>=0.10 or both NFPP>=0.01 stops allocation | REMAINS_PC, not a reproduced false-positive mechanism |
| Admission | Upper mean FPP<0.015 and NFPP<0.001, all20; physical closeout | Preserve every unfavorable repeat |
| Hardware | Ten renderer threads shared across live repeats; BLAS/OpenMP one | Allocation is not measured saturation |
| Integrity | Frozen closure, environment, evidence, checkpoints and deep weight replay | Preserve protections; no bypass for yield |

An unsupported population star cannot be clipped, removed or redrawn merely
to let the model run. The appropriate outcome is a model-support hold and a
reviewed alternative, separate from the frozen campaign. The failed
KIC3322939 cross-band join likewise needs a common physical population design,
not approximate identity matching.

## Implications for the M/K programme

The October19 handoff receipt records 663 BOI signal proposals on 473 sources,
with zero new native fits and zero probability integrations at that date. It
demonstrates source/signal/array compatibility, not a completed M/K stellar
adapter or a measured validation yield. This audit did not access the server
or establish the server's current progress. The local 125-signal/94-host
survey queue is a different population.

For prospective M/K preparation, inspect the actual run-scoped signals,
multi-signal masks, stellar radius/density support, cool-star limb darkening,
companion floors and activity/noise model. Where suitable unblended photometry
and distances exist, empirical alternatives include
[Mann et al. 2015 radii](https://arxiv.org/abs/1501.01635) and
[Mann et al. 2019 masses](https://arxiv.org/abs/1811.06938).
These relations have applicability limits and measurement requirements; they
are not permission to apply blended 2MASS magnitudes as clean component fluxes.

## Recommended order, before further rollout

1. Review a duration-aware native preparation and candidate-appropriate
   k/density/clock support. Keep old holds and every original input intact.
2. Correct catalogue/error selection and the generic run-scoped signal handoff
   in a new reviewed generation. Resolve the cross-band population workflow.
3. Test the full modified likelihood/scene on authenticated known planets,
   binaries and blends spanning faint Kepler/M/K, long transits and grazing
   geometry. Separate numerical accuracy from empirical scientific calibration.
4. Compare matched data/stellar inputs with stock mission="Kepler", with the
   correct exposure, SAP dilution convention and explicit background source.
   This isolates necessary KBonus adaptations from optional model changes.
5. Retain numerical integrity, the declared alternatives, all attempts and
   unfavorable results. Change a scientifically deficient model after review;
   do not tune a model to raise the number of admissions.

The active coordinator is intentionally held for this user-requested audit.
Its two already-running workers may finish without starting eighteen more.
No next candidate or changed-model experiment was launched during the audit.
