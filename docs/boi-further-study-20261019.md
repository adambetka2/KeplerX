# KIC 4829494 statistical follow-up — October 19, 2026

The completed study records **STATISTICALLY_VALIDATED** for BOI-2101047904706267264-1.
All 20 independent repeats include every one of the 135 scenario families and
1,000,000 attempted draws per family: 2.7 billion attempted draws in total.
The deep replay checked all 2,700 weight files. Every unfavorable repeat remains.

This separately frozen Kepler adaptation uses an empirical population prior
for the target, conditioned on its photometry and parallax. Its parallax SNR
of 3.49 does not support the strict TIC/Gaia distance route. Two bright giants
use exact-identity APOKASC3 Gold priors. The discovery clock, full resolved and
unresolved scenario support, and original measurement groups remain fixed.
The TIC native shape fits are separately identified sensitivity evidence.

See the [complete numerical result](../results/boi-validation/boi-further-study-20261019.json)
and [all 20 repeats](../results/boi-validation/boi-further-study-repeats-20261019.csv).

| Probability | Mean, % | Sample SD, percentage points | Minimum, % | Maximum, % | Mean cut | Outcome |
|---|---:|---:|---:|---:|---|---|
| FPP | 1.3325336 | 1.2696967 | 0.1227402 | 4.1345785 | <1.5% | PASS |
| NFPP | 0.0930834 | 0.0372867 | 0.0482406 | 0.1805543 | <0.1% | PASS |

Individual repeats at or above the cuts: FPP 7/20; NFPP 7/20. All remain in the means.
Target-planet likelihood-weight effective draws range from 2.681 to 8.445 per repeat; the largest normalized single-draw weight is 0.56902.
This concentration explains why individual repeats can scatter substantially;
the mean-rule decision does not imply precise individual probabilities.
The maximum omitted-likelihood posterior-mass bound is 1.14754e-22.
That numerical omission bound does not bound Monte Carlo error or model uncertainty.

## Physical and numerical scope

Both native SAP and PSF shape fits pass the declared convergence and numerical checks. RUWE is 0.98885196.
The retained calibrated pipeline triage passed shape, orbital, instrument,
localization and contamination stages. Auxiliary UNRELEASED wrappers remain
separate; they are not silently promoted into passed assessments.

The adopted target-planet-family radius is 7.510 Earth radii, with 16th/84th percentiles [6.656, 8.562].
This uses the full scene posterior, including population stellar priors and
dilution, rather than a unit-dilution native radius ratio. It establishes no mass
or composition. The posterior-average construction and concentration diagnostics
are retained in the JSON; observed Monte Carlo scatter is not hidden by the mean.

The largest resolved contribution is `NEB:2101047909004685440`; its mean posterior contribution is 0.0896776%.
Independent spatial and secondary-eclipse diagnostics remain supplementary and
are not multiplied into FPP or used to remove a scenario. Recorded aperture and
background modes reproduce in all 16 quarters; the full counterfactual background
and correlated-noise response limitations remain explicit.

October 19 exact-host catalogue/companion queries and the dated retained RV
archive coverage are bound in the record. No usable exact-host RV time series
or RV mass is established. Query emptiness is not universal absence or novelty.

The original 40-run campaign and its stopped predecessor remain untouched;
all 470 and 466 dependencies respectively reverified with unchanged receipts.
Original inputs, priors, seeds, failed endpoints, logs, checkpoints and weights
remain preserved. No scientific input, draw count or validation threshold was changed after freezing.

Preserve the audited statistical admission; improve physical characterization or obtain RV mass separately if pursued.
