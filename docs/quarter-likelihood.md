# Joint quarter likelihood

The October 21 change retains each source's quarter response in the primary
transit likelihood. With the October 25 KBonus preparation and known-object
calibration below, Adam adopted it for statistical validation on October 25,
2026; see the [adoption record](../results/boi-validation/v3-adoption-20261025.json)
and the [list of changes](triceratops-v3.md).

For observed normalized flux y_i, reviewed uncertainty sigma_i and response
a_is, source s predicts y_i = 1 + a_is (m_b(theta) - 1). For raw SAP,
a_is is the source's aperture fraction. For corrected flux it is divided by
the actual fraction already removed from that photon. All scenarios use the
same observed data and Gaussian normalization.

Within each common phase bin, `quarters.py` stores the weighted mean of the
model deficit, its precision and the remaining observed residual sum. This is
algebraically equal to summing the photon likelihoods for the same binned model.
The residual term retains information about changes between quarters. Each
parameter draw is scored jointly before evidence integration; quarter-level
evidences are never multiplied. The existing 200-point grid and exposure
integration remain. Check its timing approximation against native cadences.

The upstream patch adds an optional likelihood callback to the existing
scenario functions. It changes no random draws, population priors, orbital
models or scenario probabilities outside that likelihood. The unchanged route
remains the stock gold reference. The Gaussian normalization is in common
observed units; agreement with a host-renormalized stock call therefore includes
the analytically known normalization offset, not arbitrary evidence rescaling.
The raw output retains the joint data. The upstream mean-dilution `plot_fits`
method is disabled for these outputs; inspect quarter-resolved curves.

The upstream 1.5-sigma secondary screen is retained, including its scalar noise
and mean-response approximation. Tmag-based unresolved component fluxes and
fixed supplied stellar properties also remain upstream assumptions. The change
adds no empirical covariance model, stellar posterior distribution or composite
SED conditioning. Those require separate evidence and controls.

The focused controls compare direct photon likelihoods for both hosts, raw and
corrected flux, zero response, exact photon/phase preservation, and injected
host responses using the frozen Kepler-10 example as a shape. The latter is a
synthetic host control, not an independent known-object validation. The original
478-row example is reduced to the prescribed 200-point grid for this control.

New campaigns use `TRICERATOPS_KEPLER_V3` and require reviewed native noise and
applied-dilution arrays. Keep all frozen V1/V2 studies and stopped campaigns.
Twenty seeds, one million attempted draws per scenario, first-two stopping and
physical closeout requirements remain unchanged.
The seed namespace remains V2 so the same source/signal keeps its prescribed
seed sequence; the new protocol identity does not select different draws.

## KBonus preparation, October 25, 2026

Each V1 preparation element was re-examined against KBonus data and Kepler
literature. V3 keeps the rules below; `post_boi/kbonus.py` and
`post_boi/ephemeris.py` implement them and reproduce every overnight review
exactly. The [overnight record](../results/boi-validation/v3-kbonus-overnight-20261025.json)
lists every input, outcome and hold.

The V2 standard study did not fail for want of light-curve preparation. Without
imaging, most of its false-positive probability came from planets or binaries on
unresolved bound companions and from resolved neighbours. Across 12 BOI scenes,
the target's PRF fraction of its delivered SAP aperture also changed between
spacecraft orientations by a median factor of three and up to 21, which a mean
dilution cannot represent.

### Flux-closure gate

Quarter responses use the calibrated Kepler PRF (`lkprf 1.1.1+spacelab.1`) at the
replay-authenticated aperture and Gaia/WCS/proper-motion positions, with Gaia G as
the Kepler-band proxy. Each scene source's flux is the target's median PSF flux
scaled by its relative Gaia G flux; the modeled throughputs must reproduce the
median SAP flux. Use a quarter only if predicted over observed lies within
0.8–1.2. Exclude failing quarters from the native and probability photons before
any fit, preserve them, and hold a case with fewer than two passing quarters.

Over 226 quarters of 17 sources, the PRF response closes within 20% in 54% of
quarters; the delivered `CROWDSAP`/`FLFRCSAP` keywords close in 29%, so they are
only a cross-check. The validated planet Kepler-565 b fails closure in all 17
quarters (1.48–1.61) and cannot be assessed: the gate trades completeness for a
response that matches the data. A scene source without Gaia G uses its TIC Tmag
flux relative to the target's Tmag, the flux proxy of upstream TRICERATOPS
(Adam's decision, October 25, 2026; [record](../results/boi-validation/tmag-rule-20261025.json));
a source with neither holds the case. Every
source capable of the observed depth needs mass, radius and effective
temperature; otherwise hold the case.

### Noise, integration and field population

`kbonus.quarter_noise` scales the formal detrended errors per quarter by the
robust out-of-transit scatter of the normalized residuals (never below one) and
by the time-averaging red-noise factor over bins from one cadence to one transit
duration (Pont et al. 2006; Winn et al. 2008). The scalar for the upstream
secondary screen is the larger of the propagated 200-bin error and the
out-of-transit bin RMS.

Integrate over the 29.4-minute exposure with the smallest of 20, 40 or 80 points
whose model differs from twice as many by less than 0.1 times the folded bin
noise, chosen before any FPP. The known-object calibration used 20 points.

Draw one TRILEGAL 1.6 field in the TESS/2MASS system: 0.1 deg², single stars and
A_V(∞) = 3.1 E(B−V) from the TIC. The faint limit reaches a fully eclipsed blend
at the observed depth: `Tmag − 2.5 log10(depth)`, rounded up.

### Ephemeris matching

TRICERATOPS does not model optical ghosts, CCD crosstalk or another star's
eclipses reaching the aperture. `post_boi.ephemeris` compares each adopted clock
with all DR25 TCEs and cumulative KOIs at period ratios m/n ≤ 5 (Coughlin et al.
2014) and reports the chance expectation. A low-order (1:1, 2:1 or 1:2) match on
another star leaves the candidate unresolved until a pixel-level analysis
explains it. Among the known objects below, such a match occurs for 0 of 24
confirmed planets, 13 of 22 centroid-offset false positives, including both V3
false validations, and 4 of 16 eclipse-only false positives.

### Known-object calibration

Known KOIs with Kp 15–17.5, model SNR 15–300 and periods below 50 days were drawn
by declared seeds before their results. Each used its KBonus photometry, its
catalogue clock, the BOI gate, noise, population and runtime, and two seeds of
10⁶ draws per scenario rather than 20 repeats. Confirmed planets include
statistically validated ones, so their labels are not fully independent of FPP
methods.

| Class | Completed | Mean FPP < 0.015 and NFPP < 0.001 |
|---|---:|---:|
| Confirmed planets | 30 | 10 |
| Centroid-offset false positives | 17 | 2 |
| Eclipse-only false positives | 13 | 1 |

V3 is conservative for planets: most confirmed planets fail, often through high
NFPP where weak seasonal leverage leaves a neighbour viable. Two false
validations, K07102.01 and K03979.01, are DR25 halo-ghost and ephemeris-match
contaminants, which the ephemeris rule flags. The third, K00895.01, is an
eclipsing binary with a 162 ppm secondary (DR25 `MOD_SEC_DV`, implied albedo
1.8). TRICERATOPS fits only the folded primary and its secondary screen cannot
use an observed secondary, and the survey rule rejects only secondaries deeper
than half the primary, so neither catches it.

### Occultation measurement

`post_boi.occultation` measures the phase-0.5 secondary in the valid native
photons, event by event against local linear
baselines with the reviewed noise. A positive secondary at SNR ≥ 3 occurs for 0
of 30 confirmed planets and for all three V3 false validations (SNR 9.5, 3.8 and
3.5). This threshold was chosen after these results. The even half of a fourth
sample tested it and the ephemeris rule prospectively: neither flagged a confirmed
planet, and the ephemeris rule flagged the one false pass, K00256.01. Its odd half
was stopped on October 25 by Adam's decision.

### Controls and admission

All post_boi tests pass in the patched runtime, including the official Kepler-10
stock reference. Synthetic transits injected into KIC 8878385's SAP photons with
its calibrated responses (two seeds, 10⁶ draws per scenario):

| Injected host | FPP | NFPP | Probability on injected host |
|---|---|---|---|
| Target | 0.0002, 0.0033 | 0, 0 | 1.000, 1.000 |
| Bright neighbour (Tmag 14.5) | 1.000, 1.000 | 1.000, 1.000 | 1.000, 1.000 |
| Target, measured flux ratios | 0.0001, 0.0065 | 0, 0 | 1.000, 1.000 |

These test the likelihood and attribution; the closure gate tests the response.
Statistical validation needs no imaging or RV mass; imaging only tightens the
companion constraints where it exists. The FPP < 0.015 and NFPP < 0.001
thresholds come from TESS data, and Giacalone et al. (2021) report unreliable
FPP below SNR 15. The known-object sample above skipped detection and vetting;
BOIs reach TRICERATOPS only after both.

Stock TRICERATOPS reruns (every authenticated quarter, formal errors, stock
likelihood) are diagnostics, not an admission requirement. On the 75 known
objects with two seeds per arm, V3 passes 11 of 34 confirmed planets and stock
11 or 12; V3 passes 4 of 41 false positives and stock 3 (McNemar p = 1.0). The
ephemeris or occultation rule flags every false pass. V3 rejects the stock false
pass K02567.01, whose signal is on a neighbour, with FPP 1.0.

PRF wings beyond the calibrated footprint, stellar particle banks, custom
companion priors, composite-flux conditioning, custom importance sampling and
MOLUSC/RUWE companion limits are not restored. None is calibrated on KBonus
photometry, and RUWE/MOLUSC limits are not calibrated for G 16–19 hosts.
