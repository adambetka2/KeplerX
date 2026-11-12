# Detector intelligence: inputs and algorithm — October 13, 2026

This report audits what enters the shallow-transit detector and how the detector
decides, on the frozen development cohort: 200 planet signals (172 synthetic
injections into faint G 16–18 KBonus M- and K-dwarf SAP curves, 28 real Kepler
planets) and 102 controls. The incumbent is the selected design
`physical_local_floor05_coherence_duration_dedup_sap_source_flux_global_fundamental`.
Every number comes from post-search diagnosis against the cohort clocks. No
sealed outcome was opened, vetting is unchanged, and adoption stays with the
native owner. The knot-1.5 round is scored; the knot-1.0 round is running.

## Summary

The search finds the planets. The detector's own consistency gates and seed
budget throw them away, and the input representation hides the rest.

- 46 of 200 planets pass a nominal white-noise screen with the current input
  (matched-filter SNR ≥ 7.1 with at least 3 events). The selected design
  recovers 23 of the 46; the knot-1.5 round recovers 26, the knot-1.0 round
  31.
- 13 of the 23 detectable-but-lost planets fail the physical-stage odd/even,
  harmonic or inverse test. All three tests are miscalibrated on faint,
  variable dwarfs (section 4.1).
- 6 fail in the coherence budget, the physical seed quota or the period grid.
  On real multi-planet hosts, aliases of strong signals occupy 45% of the
  physical seed quota (section 4.2).
- 154 of 200 planets fail that screen. This does not prove that a detector or
  extraction change cannot recover them. The measured noise, support, clocks
  and model assumptions must be checked before interpreting the number as a
  sensitivity bound (section 3).
- The scorer clocks mis-measure 37 of 200 planets and hide one true recovery
  (section 3.6).

References to the "ceiling" below retain the original diagnostic grouping;
they do not establish a physical information limit. The follow-up protocol is
in `photometry-information-experiment.md`.

Ranked changes, one mechanism per paired round. Effects are upper bounds on the
200-planet denominator unless a round has measured them.

| Rank | Change | Where | Planets | Controls | Section |
|---|---|---|---|---|---|
| 1 | Odd/even and harmonic significance from empirical per-event scatter; harmonic as a nested joint fit; inverse search on outlier-clipped residuals | detector | ≤ +13 | measure | 4.1 |
| 2 | Alias-family seed dedup before the quota; mask-and-repeat search | detector | +2 to +5 | −28% proposals expected | 4.2 |
| 3 | Nuisance knots 3.0 d → 1.0 d (scored; 1.5 d scored) | detector | +7 net (1.5 d: +3), 1 protected loss at 7.07 in each | 116 → 103 proposals, 27 → 21 active; ×2.7 CPU | 4.3 |
| 4 | Detrending window per duration band: 1.0 d below 2.5 h, 2.0 d above | input | +2 at the ceiling; ×1.1–1.3 SNR above 2.5 h | measure | 3.2 |
| 5 | Period grid to 300 d with the 3-event rule | detector | +2 real (153 d, 178 d) | measure | 4.4 |
| 6 | Clocks over the data span | scoring | +1 hidden, 1 unscorable | none | 3.6 |
| 7 | Quality-bit audit; keep bits that do not raise residual scatter | input | ≤ +5% SNR | measure | 3.3 |
| 8 | Per-target view choice from a label-free noise metric | input | unknown | measure | 3.4 |

## 1. Evidence

Server root `S=/srv/kbonus/dwarfs`;
analysis root `A=$S/lanes/detector-loss-20261013`.

- Loss ledgers: `$A/diagnostics/ledger-selected/`, `$A/diagnostics/ledger-knot15/`
  (`ledger.csv`, `summary.json`; one row per planet, first loss stage, white-noise
  SNR, physical-stage statistics).
- Physical-stage audit: `$A/diagnostics/physical-stage-audit-{selected,knot15}.csv`
  (seed ranks, alias counts, odd/even by stage, final proposals).
- Detrending retention: `$A/diagnostics/biweight-retention-v1.{csv,json}`.
- Residual noise at transit timescales: `$A/diagnostics/binned-noise-v1.{csv,json}`.
- Rounds: `$A/rounds/knot15-development200-20261013-v2/paired-review.json`,
  `$A/rounds/knot10-development200-20261013-v1/` (running).
- View information audit: `$S/analysis/overnight-20261011/photometry-signal-information172-v1`.
- Scripts: `$A/loss_ledger.py`, `$A/physical_stage_audit.py`,
  `$A/biweight_retention.py`, `$A/binned_noise.py`, `$A/paired_review.py`.

## 2. Where the planets are lost

First loss stage per planet. "Ceiling" counts only planets with white-noise
SNR ≥ 7.1 and at least 3 observed events (46 planets).

| First loss stage | Selected, all 200 | Selected, ceiling 46 | Knot 1.5, all 200 | Knot 1.5, ceiling 46 |
|---|---:|---:|---:|---:|
| Recovered | 24 | 23 | 27 | 26 |
| Fewer than 3 observed events | 25 | — | 25 | — |
| Coarse search below threshold | 38 | 0 | 38 | 0 |
| Coarse depth outside 10–6000 ppm | 13 | 0 | 13 | 0 |
| Coherence refinement budget | 37 | 4 | 37 | 4 |
| Box stage below 7.1 | 42 | 1 | 43 | 2 |
| Physical seed quota | 4 | 2 | 4 | 2 |
| Physical odd/even, harmonic or inverse test | 13 | 12 | 10 | 7 |
| Distinct-proposal overlap after a test failure | 1 | 1 | 2 | 2 |
| True period outside the 0.5–100 d grid | 2 | 2 | 2 | 2 |
| Proposed, clock mismatch (section 3.6) | 1 | 1 | 1 | 1 |

Recovery by white-noise SNR (planets with at least 3 events):

| White-noise SNR | Planets | Synthetic | Real | Recovered, selected | Recovered, knot 1.5 |
|---|---:|---:|---:|---:|---:|
| < 3 | 104 | 104 | 0 | 0 | 0 |
| 3–5 | 15 | 15 | 0 | 0 | 0 |
| 5–7.1 | 10 | 8 | 2 | 1 | 1 |
| 7.1–10 | 5 | 4 | 1 | 1 | 1 |
| 10–15 | 12 | 9 | 3 | 5 | 5 |
| 15–25 | 14 | 4 | 10 | 9 | 10 |
| 25–50 | 12 | 2 | 10 | 7 | 7 |
| ≥ 50 | 3 | 1 | 2 | 1 | 3 |

Two conclusions follow. The box search is close to the white-noise ceiling:
only one ceiling planet fails the box floor. The losses that a detector change
can recover are concentrated after the box stage, in the physical template
stage and its gates. The cohort itself has little power: 119 of 175 observable
planets are below SNR 5 and are undetectable by construction (section 3.5).

## 3. Data representation

### 3.1 What enters the detector

- SAP flux per quarter after the constant-neighbour source-flux transform:
  `flux − (1 − CROWDSAP) × quarter median`, absolute error unchanged. Relative
  noise therefore scales as 1/CROWDSAP.
- Cadences with any set quality bit are removed: about 10% of finite cadences.
  Bits 8192, 16, 128, 262144, 131072 and 256 dominate. Quarters with fewer than
  100 cadences are dropped.
- Biweight detrending, cval 5: 0.5 d window for the 0.5–30 d band, 1.0 d for
  the 30–100 d band.
- Noise: local-point MAD scale in 5 d windows, with 0.5 d gap breaks, scaled
  within 0.5–2.0 of the prior.
- Physical stage: cubic B-spline nuisance with knots every 3.0 d (1.5 d in the
  knot-1.5 round), diagonal weights from the same noise scale, limb-darkened
  profiles integrated over the 0.0204 d exposure with 11 samples.

### 3.2 Noise budget

Across the 200 planets, the first-difference scatter divided by the pipeline
error has median 1.07 and 90th percentile 1.39. The pipeline error has median
2540 ppm per cadence at G 16.5–17.5. This agreement describes the delivered
extraction's reported uncertainty; it does not establish a photon limit or
bound the gain from a different extraction. A physical budget must separate
target, contaminant and sky photon noise, read noise, calibration uncertainty
and model error, and compare extraction operators on the same pixels.

Residual noise at transit timescales, measured label-free on the 287 prepared
cases: after the 0.5 d biweight, the robust scatter of binned residuals divided
by the white expectation from the first-difference scatter has medians 0.97
(1 h), 0.97 (2 h), 0.93 (3 h), 0.78 (6 h) and 0.61 (12 h). The median case is
white or whiter at every transit timescale, because the window suppresses
power at 3–12 h; that is the same suppression that removes long transits. The
90th-percentile case carries 1.37, 1.50, 1.39 and 1.25 at 1, 2, 3 and 6 h. One
case in ten keeps 25–50% correlated excess at transit timescales, and those
cases drive the over-confident statistics of section 4.1. Positive outliers
above 5σ occupy 0.10% of cadences in the median case and 1.3% at the 90th
percentile; negative outliers 0.02% and 1.35%.

The detrending window removes transit signal from the coarse-search series.
The physical refinement fits original prepared flux jointly with its nuisance
splines, so the quoted attenuation does not describe its final-fit input.
Retention is the least-squares
amplitude of the whitened injected-minus-original difference against the true
loss profile (164 synthetic planets; 8 zero-support cases excluded). The
matched-filter SNR gain uses the detector's per-quarter MAD noise.

| Duration | Planets | Retention 0.5 d | 1.0 d | 2.0 d | SNR gain 1.0 d | SNR gain 2.0 d | Noise rise 2.0 d |
|---|---:|---:|---:|---:|---:|---:|---:|
| < 1.5 h | 55 | 0.89 | 0.94 | 0.96 | ×1.02 | ×0.99 | +4% |
| 1.5–2.5 h | 47 | 0.84 | 0.91 | 0.95 | ×1.05 | ×1.07 | +7% |
| 2.5–4 h | 43 | 0.73 | 0.85 | 0.92 | ×1.09 | ×1.13 | +5% |
| 4–6 h | 18 | 0.62 | 0.80 | 0.89 | ×1.24 | ×1.34 | +16% |
| > 6 h | 1 | 0.51 | 0.74 | 0.86 | ×1.39 | ×1.59 | +5% |

By period, the 1.0 d window gains ×1.02 (P < 5 d), ×1.07 (5–30 d) and ×1.20
(30–100 d). Planets at or above the 7.1 floor: 14 with 0.5 d, 16 with 1.0 d,
14 with 2.0 d. The window `max(0.5 d, 3 × duration)` equals 0.5 d for every
duration under 4 h and is not a real alternative.

Recommendation: whiten each duration band with its own window, 1.0 d for the
bands below 2.5 h and 2.0 d for the longer bands. The search is already
partitioned by duration, so this is one mechanism. K03096.01 (7.2 h, 80 ppm,
coarse SNR 4.9) is the archetype that the 0.5 d window loses.

### 3.3 Quality mask

Removing every flagged cadence costs about 10% of the data, a 5% SNR ceiling.
Not every bit marks damaged aperture photometry. Measure, per bit, the robust
scatter of the whitened residual on cadences carrying only that bit against
unflagged neighbours, and keep bits whose scatter is indistinguishable. This is
label-free and one mechanism.

### 3.4 Crowding and view

Quarter CROWDSAP across the 231 cohort cases with crowding records: 10th,
25th and 50th percentiles 0.47, 0.71 and 0.93; 12.6% of cases have median
CROWDSAP below 0.5 and 30% below 0.8. For these targets the source-flux
transform doubles or worse the relative noise, which is correct physics: the
neighbour photons are in the aperture.

The view audit on the 172 injections measured the projected signal norm per
view: SAP source flux 0.21, 0.86 and 3.09 at the quartiles against PSF 0.13,
0.59 and 1.42, with 0.94–0.98 signal retention in every view and more rows in
SAP (median 32,048 against 24,412). SAP is the right default. For the crowded
tail the PSF view may carry more information per target. A per-target choice
is legitimate only through a label-free metric: the predicted matched-filter
noise of each view, computed from its own residual scatter and dilution before
any search.

### 3.5 Cohort design and the habitable-zone target

104 of 200 planets have white-noise SNR below 3 and 119 below 5. The synthetic
depths have quartiles 58, 179 and 729 ppm at periods 0.9, 6.2 and 45.7 d; these
injections are hopeless at 2500–3500 ppm per cadence and give no information
about detector changes. The next development cohort should inject at the
detectability edge, white-noise SNR 5–15, where every change in section 4 acts.

The same arithmetic bounds the science goal. With 3200 ppm per cadence, a
1421 d span and 90% duty, the white-noise SNR of a habitable-zone transit is:

| Host | HZ period | Duration | 1 R⊕ | 1.5 R⊕ | 2 R⊕ | 3 R⊕ |
|---|---:|---:|---:|---:|---:|---:|
| Mid M dwarf, 0.3 R☉ | 24 d | 2.4 h | 4.6 | 10.4 | 18.5 | 42 |
| Early M, 0.5 R☉ | 46 d | 4.1 h | 1.6 | 3.6 | 6.4 | 14 |
| K dwarf, 0.7 R☉ | 126 d | 7.0 h | 0.7 | 1.5 | 2.6 | 5.8 |

At G 16–18 the reachable habitable-zone population is planets of 1.5 R⊕ and
larger around mid-M dwarfs and 2.5 R⊕ and larger around early-M dwarfs. The
K-dwarf habitable zone is out of reach at this brightness with any detector;
it opens at G ≤ 15, where the per-cadence noise is near 1000 ppm. Durations of
4–7 h also sit where the 0.5 d window keeps only 0.6 of the signal, and the
K-dwarf periods sit outside the 100 d grid.

### 3.6 Scorer clocks

37 of 200 planet clocks have a scoring window that does not cover the data
span or contains no transit centre. Synthetic 2100406000372590720 is proposed
exactly (21.6863 d, epoch 137.8659) with 61 observed events, yet its clock
spans 120.5–130.2 and it is scored lost. Synthetic 2101919370749488256 has 0
centres inside its window and 31 in the data and cannot be scored. The 8
zero-support cases are correctly excluded. Define clocks over the prepared
array span; this changes the denominator, not the detector.

## 4. Detector algorithm

### 4.1 The consistency tests are miscalibrated

The physical stage fits each seed jointly with the spline nuisance and then
applies three gates: odd/even depth difference below 3.5σ, harmonic
Δlog L ≤ 0, and dip-to-inverse SNR ratio ≥ 1.5. Ceiling planets lost to them,
selected design:

| Planet | Kind | P (d) | Depth (ppm) | Dur (h) | White SNR | Physical SNR | Odd/even σ | Harmonic Δlog L | Inverse ratio | Knot 1.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2127399526231743360 | synthetic K | 2.000 | 64 | 1.8 | 13.9 | 17.5 | 10.4 | −84 | 2.65 | recovered |
| 2079714943881015936 | synthetic M | 7.393 | 893 | 2.3 | 71.3 | 56.9 | 8.4 | −1171 | 5.64 | recovered |
| synthetic M | synthetic M | 3.546 | 150 | 1.7 | 15.1 | 15.4 | 0.2 | +1036 | 0.98 | lost, same |
| synthetic M | synthetic M | 27.07 | 519 | 2.8 | 13.5 | 12.6 | 1.0 | −24 | 0.60 | lost, same |
| synthetic K | synthetic K | 0.959 | 230 | 1.0 | 13.9 | 15.5 | 1.3 | +1.5 | 1.97 | lost, harmonic |
| synthetic K | synthetic K | 3.072 | 584 | 2.3 | 13.5 | 15.8 | 1.6 | +33 | 3.15 | lost, harmonic |
| synthetic K | synthetic K | 58.86 | 920 | 5.9 | 14.2 | 13.6 | 1.2 | −92 | 1.07 | lost, inverse |
| K00635.01 | real | 16.72 | 523 | 3.3 | 79.0 | 60.8 | 5.3 | +3880 | 2.10 | recovered |
| K00294.01 | real | 34.44 | 425 | 5.8 | 39.5 | 46.8 | 6.7 | −959 | 3.11 | lost, odd/even |
| K01314.01 | real | 8.575 | 161 | 8.7 | 46.7 | 22.4 | 4.7 | +19 | 2.90 | lost, overlap |
| K01337.01 | real | 1.923 | 260 | 1.9 | 37.6 | 26.3 | 4.8 | −318 | 4.54 | lost, overlap |
| K01835.03 | real | 6.767 | 249 | 1.7 | 17.7 | 11.1 | 5.7 | +122509 | 0.50 | recovered |
| K02482.01 | real | 45.09 | 436 | 4.6 | 22.4 | 15.9 | 10.4 | −81 | 1.61 | lost, odd/even |

Each test fails for a specific reason.

**Odd/even.** The statistic is |d_odd − d_even| / √(var_odd + var_even −
2 cov) with the covariance of the weighted linear fit. The weights are the
local-point MAD per cadence, so the formal error assumes white residuals.
Injected planets have identical odd and even depths by construction, yet 3 of
25 fitted synthetic planets exceed 3.5σ (up to 10.4σ) and 9 of 22 real planets
do. On the same hypotheses the box-stage odd/even exceeds 3.5σ for 0 of 25 and
1 of 22. Knots at 1.5 d reduce the counts to 1 of 25 and 4 of 22, which
confirms that nuisance misfit, not the planet, drives the statistic. The stage
already fits every event separately (`per_cycle`), so the empirical scatter of
per-event depths is available at no cost. Normalize the odd/even difference by
that scatter, or by a bootstrap over events.

**Harmonic.** The test fits a sinusoid at the orbital period to all cadences
and compares its Δlog L with the transit's. The two models are not nested, so
the test measures residual variability at the period, not whether the signal
is sinusoidal. A 3.0 d knot spacing cannot follow rotation signals below about
10 d (fewer than three knots per cycle), so active M and K dwarfs, the target
population, leave residual power that the sinusoid absorbs. K01835.03 shows
Δlog L of +122509 at 3.0 d knots and passes at 1.5 d. Replace it with the
nested test: fit nuisance, sinusoid and transit jointly, and reject only when
the transit term loses significance.

**Inverse.** The test runs a box search on the negated residual at the same
period and duration, over all epochs with oversample 10, and rejects when the
dip SNR is below 1.5 times the best bump SNR. The maximum over hundreds of
epochs of a bump statistic in residuals that still contain variability and
flares reaches the dip SNR: four ceiling planets show ratios 0.50–1.07, and
positive 5σ outliers occupy up to 1.3% of cadences (section 3.2).
Restrict the comparison to the eclipsing-binary hypothesis (a bump at the same
epoch and at phase 0.5), or compare the dip against the distribution of bump
SNRs at random periods, and clip positive outliers before the search.

All three gates are detector-internal. Changing them does not touch vetting.
The 13 planets they cost are the largest recoverable block above the ceiling.

### 4.2 Alias seeds, the quota and multi-planet hosts

The physical stage fits at most 64 seeds per band, sorted by seed SNR, with a
dedup that removes only clocks within 0.1% in period and 90% window overlap.
Integer multiples and fractions of a strong signal survive the dedup and fill
the quota. K00952.04 (Kepler-32 e, 2.896 d) is found in every box band at
coherence rank 9–23 with SNR 13–19, but its physical seed rank is 98: 56 of
the top 70 seeds are k×P or P/k members of the system's three strong planets.
Its final proposal list holds 10 integer-multiple aliases among 13 proposals.
The proposal overlap rule already divides by the smaller window set, so every
multiple of an accepted proposal is removed. The aliases survive because the
three fundamentals fail a consistency gate, and coprime multiples k ≥ 5 of a
rejected fundamental share at most 20% of their transits with each other.

Cohort census of final proposals: real planets 23 of 74 are aliases (9 cases),
catalogue false positives 23 of 51, inverted controls 7 of 33, instrumental
controls 3 of 19; 33 of 116 control proposals (28%) are aliases. Alias seeds
occupy 814 of 1792 top-64 slots (45%) on real planets and 1486 of 10432 (14%)
on synthetic cases. Alias-family dedup before the quota brings K00952.04 to
rank 28 and moves 5 of 32 never-fitted synthetic planets inside the quota.

Recommendations, in order of simplicity: dedup alias families before the
quota with the same smaller-window overlap rule; after the top proposal is
accepted, mask its in-transit cadences and repeat the seed search. Masking
removes alias flooding at its root and is the standard multi-planet strategy.
204 of 287 cases already hit the 128-seed cap with a median pool of 858, so
dedup also lowers cost.

### 4.3 Nuisance stiffness

The knot-1.5 round changed one number. Planets 24 → 27: gains K00635.01,
K01835.03, 2079714943881015936 and 2127399526231743360; loss of protected
2105767081631335552 (K dwarf, 664 ppm, G 17.4) at physical SNR 7.07 against
the 7.1 floor. Controls: any-activity 27 → 26, proposals 116 → 103,
catalogue-FP plus native-EB any-proposal 9 → 9, no eligibility change. Every
SEQUENTIAL_DEVELOPMENT200 criterion passes; only the protected-identity rule
fails. The physical SNR scale moves with the nuisance model, so a floor tuned
at 3.0 d knots is not transferable, and a loss by 0.03 at a hard floor is a
threshold artifact, not a detector regression. Re-measure the floor against
the control null when the nuisance model changes.

The knot-1.0 round changed the same number again. Planets 24 → 31 against
the selected design and 27 → 31 against knot 1.5: gains over knot 1.5 are
2105111016786174080, 2086711514332691584, K00294.01 and K01337.01, and the
knot-1.5 loss 2105767081631335552 returns; the one loss is protected
2104994846510734720 (K dwarf, 67 ppm, P 0.71 d, 0.9 h, white-noise SNR 5.8)
at physical SNR 7.07 against the 7.1 floor (7.22 at 3.0 d, 7.38 at 1.5 d).
Controls: any-activity 27 → 21, proposals 116 → 103, catalogue any-proposal
9 → 9, no eligibility change. All 31 recoveries sit at the white-noise
ceiling, and the consistency-gate losses at the ceiling fall from 13 to 5
with the tests unchanged: finer knots shrink the nuisance residual that the
formal errors of 4.1 ignore. The longest recovered transits (10.1, 8.4 and
7.1 h) survive at 1.0 d knots. The cost is the problem: 38.4 CPU hours per round
against 14.3 at 1.5 d and 13.3 at 3.0 d (section 4.7). Knot 0.75 d is
prepared to find where the gain stops, and the combination rounds build on
the 1.0 d base.

A per-target knot spacing chosen from the residual autocorrelation time, with
a lower bound of three times the longest duration searched, would resolve the
rotation of active dwarfs without absorbing transits. That is a later round.

### 4.4 Search grid and budget

- Period grid: 0.5–30 d with 20,000 points, long band 30–100 d. Two real
  planets at 153 d and 178 d are outside the grid; the K-dwarf habitable zone
  is above 100 d. Extending to 300 d with the 3-event rule costs about one
  more long band.
- Coherence budget: 37 planets are lost in refinement, 4 above the ceiling
  (white SNR 7.5–12.0, periods 2.7–75 d). Raising `max_refinements_per_duration`
  from 64 is cheap relative to the physical stage and should be a measured
  round.
- Seed quota: see 4.2. Fit the quota after dedup, not before.

### 4.5 Noise model and thresholds

The box and physical SNR are both scaled by the local-point MAD, which is a
cadence-level noise. Residual correlated noise at 1–6 h inflates every
statistic in the same way, so the 7.1 floor and the 3.5σ gate are not at the
significance their names suggest. Two consistent options: whiten at transit
timescales before the search (per-band detrending plus a per-quarter
first-difference scale), or keep the statistics and calibrate every threshold
against the 102 controls and the inverted and scrambled nulls. The second is
label-free, costs nothing, and is required whenever the nuisance model changes.

### 4.6 Architecture: score, then gate once

The current detector applies hard gates at five points: coarse SNR 4.0, box
floor 7.1, seed quota, three consistency gates, and the overlap rule. Each gate
loses planets at a calibration that depends on the nuisance residual. A single
ranking statistic (the physical Δlog L against a calibrated null) with the
consistency tests recorded as ledger flags, and one final gate measured on
controls, keeps the same control burden with fewer silent losses. The protocol
already separates discovery from vetting; the detector should not vet.

### 4.7 Cost

Per case the selected design takes 5, 150, 420 and 613 s at the 10th, 50th,
90th percentiles and maximum, 13.3 CPU hours per 287-job round. The physical
stage dominates; alias dedup and masking reduce it. Knot 1.5 d costs 14.3
hours and knot 1.0 d 38.4 hours with a longest job of 24.9 minutes: the
profiled nuisance solve, a sparse LU of the weighted normal matrix per
quarter and trial, grows faster than the knot count. Scaled from the cohort
mean, the 406,547-source survey costs about 54,000 CPU hours at 1.0 d
against 19,000 at 3.0 d. A banded solve over the per-quarter spline basis
makes the cost linear in knots and is the engineering item that pays for
the gain at survey scale.

## 5. Round order

1. Consistency-test calibration (4.1), single mechanism: empirical-scatter
   odd/even, nested harmonic, clipped and phase-restricted inverse. Measure
   each sub-change separately if the combined round moves controls.
2. Alias-family dedup before the quota (4.2).
3. Knot 0.75 d, and knot 1.0 d combined with the consistency
   recalibration, alias dedup and the native branch's per-event rescue and
   veto (4.3).
4. Per-band detrending window (3.2).
5. Coherence refinement budget (4.4).
6. Period grid to 300 d (4.4).
7. Combination of the accepted mechanisms, then the mixed-1000 cohort.

Fix the clocks (3.6) before the next scoring. Build the next development
cohort at white-noise SNR 5–15 (3.5).

## 6. Limits

The 46-planet ceiling set is small; a single planet is 2 percentage points.
Synthetic injections have identical odd and even depths and no secondary, so
they test calibration, not the gates' discriminating power; the 14 catalogue
false positives and 14 native EBs carry that burden and must stay flat in
every round. Real-planet hosts are brighter than the injection hosts and
dominate the high-SNR bins. The habitable-zone table assumes white noise and
no limb darkening.
