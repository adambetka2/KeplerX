# Kepler validation follow-up — October 19, 2026

Subsequent decision: Adam accepted the three existing studies at their adopted
inputs and adopted the [prospective final-primary protocol](post-boi-protocol.md).
The timing-count observations below remain historical findings; they do not
mandate additional runs for those three.

This follow-up narrows the previous audit's model-change concern. Historical
scientific source hashes match the later frozen copies: nine inspected files
for KIC 6699887 and five for KIC 8745833. The second target also has an identical
complete v2 closure hash. A fresh check of the stopped server predecessor also
found all 453 shared archived scientific dependencies unchanged. The hardware
rewrite preserved the scientific model.
The remaining timing work is a matching final-ephemeris probability census;
final periods and epochs have already been fitted.

The [structured result](../results/boi-validation/boi-method-followup-20261019.json)
binds source comparisons, new bounded controls, timing summaries and lookup limits.

## Exact odd/even and secondary definitions

Our odd/even statistic uses inverse-variance weighted per-event depths. Let
`z = abs(d_odd-d_even)/sqrt(sigma_odd²+sigma_even²)`. Its two-depth BIC gain is
`z²-ln(N)`, and the orbital conflict margin is greater than 10. `N` includes
local before/inside/after sample counts. For N=1,000 the implied z threshold is
about 4.11. Event errors include excess scatter and duration-bin noise inflation.

The reference requires agreement between BLS formal errors and a robust
red-noise depth estimator at greater than 3 sigma; when the robust measurement
has insufficient parity coverage it uses BLS greater than 10 sigma.
Our method has one depth comparison and no identical two-estimator agreement
condition. Neither procedure is uniformly stricter in every noise/coverage case.

Our phase-half secondary conflict requires SNR at least 7 and secondary depth
above 50% of the primary. The reference uses SNR above 3 and depth above 10%.
For the same measurements, our secondary screen misses more weak/shallow cases.
A 20%-deep secondary at 5 sigma illustrates the difference. This is not a
finding of an eclipse in any of the studied planets. These thresholds should
be compared using reproduced measurements, because their error definitions differ.
See [reference orbital screening](https://arxiv.org/html/2608.21560v3#S4.SS2).

## What was verified for Kepler

| Component | Evidence and limit |
|---|---|
| Passband | The adopted kernel uses Kepler_Kp and the upstream Kepler limb-darkening table. Four cached/reference lookups agree exactly; MIST color arithmetic agrees within 1e-15 mag. |
| Population colors | All 7,260 new-field Gaia population rows uniquely match the twelve printed physical fields in the Kepler simulation. Native color includes extinction once, with explicit 0.122666-mag AB-to-Vega conversion. |
| Exposure integration | Actual Kepler timing headers specify 270 frames, 6.538751429414-second frame spacing, 6.01980290327-second live integration and 0.518948526144-second read time. The adopted 54-point coadd approximation has preserved 270-frame/live-width checks. |
| Independent transit curves | Six circular planet controls spanning radius ratios 0.025–0.1 and impact parameters 0.2/0.8 agree with batman at the same exposure midpoints; maximum discrepancy 0.045824 ppm, below the predeclared 5-ppm control tolerance. This checks the occultation kernel, not every eccentric EB, phase-grid error or scene probability. |
| Detrending and covariance | All three adopted discovery packets reproduce the observed bins and reconstruct covariance. Nonzero-residual chi-square values agree with independent covariance pseudoinverses. Fixed MAP baseline/noise and first-order covariance remain assumptions. |
| Sampling | Radius-CDF inversion agrees with the upstream sampler at 297 quantiles/masses. Defensive prior support, p/q correction and all-attempt denominators are explicit. Existing scalar/vector binary-prior checks passed. Sparse likelihood weights remain a precision limitation. |

These are positive implementation checks. They do not establish empirical
calibration of the whole modified probability model. Stellar-bank construction,
composite Gaia flux conditioning, PRF wings, counterfactual background response
and fixed-MAP noise treatment need their own known-object or paired sensitivity
controls. A population host prior for low parallax SNR is a declared scientific
choice, not automatically required by the Kepler passband. The same distinction
applies to adding measured seismic priors for identified neighbors.

## What changed and when

1. The October 12 archive contains final-timing FPP estimates for the first two
   targets, alongside discovery-timing estimates and retained sensitivities.
   Their shared physical kernels, prepared likelihood and proposal source
   hashes match the later copies. The larger common source closure is also
   identical for KIC 8745833. This is more specific than a generic concern that
   software versions might have changed.
2. The broader October 13 server campaign included both clocks. Its preserved
   status records a user-requested stop on October 14, with 13 of 160 runs
   completed. Actual result files confirm one TIC/final run for KIC 6699887
   and two for KIC 8745833. Each includes every family at one million attempts
   and already uses πr². Their FPPs are 0.420612%, 0.699336% and 0.761247%;
   all also pass NFPP. All 453 shared archived scientific dependencies match
   the current copies. This is substantial existing final-clock evidence.
3. Before the October 17 production freeze, unresolved-background area changed
   from r² to πr², matching the explicitly circular position prior. This is a
   real prior-normalization amendment. The later count protocol also requires
   20 independent repeats with one million attempts in every family; historical
   results used smaller baselines plus selected-family refinements.
4. The later production plan deliberately accepts only TIC discovery timing.
   Its runner rejects other ephemerides. That scope omitted the reference's
   same-input final-timing closeout. Describing final timing as unable to qualify
   the validation was an incorrect reading of the reference procedure.
5. The October 18 hardware generation changed caching, fused evaluation and
   scheduling. The plan changed only the output root. Preserved reference
   replays passed support/bound parity and the existing posterior L1 tolerance.
   This was not a new physical model or a new clock.
6. KIC 4829494 is a separately frozen field with its own population host prior,
   exact paired colors and two seismic neighbors. It uses discovery timing in
   every production repeat; the native fitted timing was never sent through
   a complete competing-scenario probability calculation.

An analytic diagnostic changes only the historical final estimates' area
factor to π. Both still pass: KIC 6699887 FPP 0.448227%, NFPP 0.000352160%;
KIC 8745833 FPP 0.839235%, NFPP about 8.47e-48%. This reconstructs saved
scenario probabilities without new draws. It does not create a new 20-repeat
census. π also scales the intended-host DTP family, so total FPP need not rise.

Historical raw pool paths no longer resolve locally, and the targeted read-only
server lookup did not find those exact files. This comparison uses the committed
October 12 ledger's full per-family marginals/log probabilities and recorded
source hashes; it does not claim fresh verification of unavailable raw pools.
No legacy result is overwritten or reclassified.

## Fitted timing versus probability calculation

| Target | Discovery period, days | Native fitted median period, days | Preserved fitted-clock FPP evidence | Matching 20-repeat fitted-clock FPP |
|---|---:|---:|---|---|
| KIC 6699887 | 15.113702872318 | 15.113694597194 | Smaller-count studies plus 1 full-count repeat | Not completed |
| KIC 8745833 | 4.566536246517 | 4.566494216965 | Smaller-count studies plus 2 full-count repeats | Not completed |
| KIC 4829494 | 16.105469174934 | 16.105601406596 | Not completed | Not completed |

KIC 4829494's original-cycle fitted epoch is
120.832457289493 BKJD TDB, derived jointly from the saved posterior.
The independent native fit exists; no new sampler was run for this audit.
The reference's final-clock test is a scenario-probability calculation with
other inputs held fixed, not simply a successful period/epoch fit:
[section IV.6.2](https://arxiv.org/html/2608.21560v3#S4.SS6.SSS2).

No final-clock campaign or next candidate was launched. A new production plan
needs a separately reviewed derived generation; frozen receipts and inputs
remain unchanged.
