# Rejecting eclipsing binaries that survive vetting (2 November 2026)

The task: the smallest justified addition to the lean KeplerX stack that
rejects the eclipsing binaries the vetting policy keeps, without losing
planets, and never dismissing on a missing measurement or on a secondary
eclipse alone. Code: `KeplerX/vetter/binary_evidence.py` (commit
f153a36b7) and the `linear-score-binary-gate` policy in
`KeplerX/vetter/policy.py` (commit 37383f002). Scripts and tables:
`KeplerX/results/calibration-audits/2026-11-02-binary-evidence/`. All
evidence is development partition and the survey benchmark: the sealed
holdout was not touched.

## The binaries that survive

On calibration-sample revision 5's development-quick set, scored with the
final Detector, production vetting (`linear-score`) keeps:

- **On-target binaries:** 32 of 43.
- **Blended binaries:** 11 of 13.
- **Leaked binaries:** 25 of 51. These are real Kepler eclipsing binaries
  that leak from the TPF target into a lane curve.
- **Leaked KOI false positives:** 40 of 52.

Many have secondary eclipses at MES 30 to 3,000, and eclipses of tens of
percent. The score's only binary feature is the strongest other dip over
the detection's MES. A deep binary's primary MES runs to the hundreds, so
its secondary hardly moves the score: one kept binary has a 44% eclipse and
a secondary at MES 969 against a primary at 1,426.

## What other vetters test

| Test | DR25 Robovetter (Thompson et al. 2018) | Earlier spacelab campaigns (`pipeline/vetting/`) | KeplerX before | KeplerX now |
| --- | --- | --- | --- | --- |
| Secondary eclipse | Model-shift MS1–MS3: significance over red noise, uniqueness against the next dip and the brightest bump | Phase-half secondary depth; eccentric phases not covered | Strongest other dip / MES, a score feature | Significance and uniqueness over the fold's noise, every other signal masked |
| Planet occultation | Significant secondary kept as a possible planet if albedo < 1 | none | none | Albedo a planet would need, (d₂/d₁)(a/R*)², dilution-free |
| Twin eclipses at half period | Secondary's depth against the primary's | none | none | A half-phase secondary as deep as the transit is not evidence |
| Odd/even | Model-shift odd/even | Weighted event depths, significance | Upstream trapezoid fits, box depths (not in the score) | Per-transit depths, errors from their own scatter |
| Orbital modulation | none | Coherent harmonic fit, BIC | none | Ellipsoidal amplitude, flattening-corrected, halves agreeing, mass ratio |
| Shape (V-shape) | Flag, not a dismissal | Competing models | Model misfit (score) | unchanged |

DR25 Robovetter dismissed on a significant, unique secondary unless a
planet's occultation could explain it. The earlier campaigns' coherent
harmonic fit is the right tool for ellipsoidal variation, but KBonus's
flattening removes it beyond a few days, which they did not correct for.

## The measurements

`vetter/binary_evidence.py` measures, last among the Vetter's components
(it whitens again, so no earlier metric changes):

- **Secondary eclipse.** The strongest dip of the MES folded at the
  detection's period and duration, phases within two durations of the
  transit excluded. The fold masks every other signal of the curve, not
  only the stronger ones as upstream does, because in a multi-planet
  system a sibling's transits are a "secondary". Detections at the same
  period or at twice, three times, half or a third of it stay unmasked: a
  binary's secondary is found as a repeat at its period, on another epoch.
  The dip is measured against the fold's noise (the MAD of the MES), the
  next dip and the brightest bump.
- **Half period.** A secondary at half phase, within two durations, and
  within 3 in MES of the transit is the planet at half the period; it is
  not evidence.
- **Occultation albedo.** A planet's occultation is at most its transit
  depth × geometric albedo × (R*/a)². The geometric albedo a planet would
  need is (d₂/d₁)(a/R*)², with a/R* from the period and the host's
  catalogue density. Both depths are diluted alike, so the ratio, and the
  albedo, hold for blends.
- **Odd/even.** Depths of the individual transits, odd against even, with
  errors from each set's own scatter. On box depths with white errors, one
  systematic dip under one transit put a single planet at 15σ.
- **Ellipsoidal variation.** -A cos(4π phase), fitted out of eclipse with
  the cos and sin of 2π phase beside it. The amplitude is corrected for
  KBonus flattening's transfer, measured with the calibration's emulator
  on the lane K dwarf: 0.97 at a 2-day period, 0.93 at 3, 0.48 at 4,
  nothing beyond 6. So it is fitted only while P/2 ≤ 3 days, and the two
  halves of the baseline must agree, a coherent orbital signal rather than
  spots. The mass ratio follows from A = 1.37 q (R*/a)³ (Morris 1985).

## The first gate, and what it did to real planets

The first gate dismissed on any one piece, whatever the score:

- a secondary at significance ≥ 7.1, uniqueness ≥ 3, not at half period,
  needing an albedo above 1;
- odd/even ≥ 5σ with at least three transits each;
- an ellipsoidal variation ≥ 7σ, agreeing in both halves, of mass ratio
  above 0.05.

On development-quick it dismissed 31 of the 68 binaries the score keeps
(46%) and 17 of the 40 KOI false positives, and no synthetic, known or
validated planet. The out-of-sample check said otherwise. On the 493
confirmed-planet hosts of the survey benchmark (`data/survey-benchmark`,
which the calibration never used), it dismissed five of the 586 confirmed
planets the score keeps:

- **Kepler-158 c (28.6 d) and Kepler-1789 b (80.2 d).** A "secondary"
  0.9 and 1.6 days from the transit, at MES 9.6 and 13.9. These are not
  eclipses: transit-timing variations, the transit's wings under
  flattening, or a sibling.
- **Kepler-1662 b (134.5 d).** A "secondary" at MES 11.2 in a fold of three
  or four events: one large single event.
- **Kepler-493 b (3.0 d).** A 480-ppm "ellipsoidal" term, far beyond a
  planet's, agreeing in both halves: the transit's flattened surroundings
  or the star's own variability.
- **Kepler-840 b (2.5 d).** A secondary at half phase, 3.7% of the transit
  and needing an albedo of 2.05. It is confirmed by statistical
  validation, but DR25's Robovetter calls it a false positive on this
  secondary (`MOD_SEC_ALT, PLANET_OCCULT_ALT, HAS_SEC_TCE`).

So a lone secondary, or a lone ellipsoidal term, is not safe on real
curves. The secondary alone also breaks the task's rule.

## The gate

`linear-score-binary-gate` keeps the production score. It dismisses,
whatever the score, only on decisive or corroborated binary evidence:

- odd and even transits differing by 5σ, with three transits each (a
  planet's transits have one depth);
- or two independent pieces together: the secondary, the ellipsoidal
  variation, and odd/even counting from 3σ.

A lone piece never dismisses. It is recorded on the kept signal
(`disposition.binary_evidence`) for further assessment, and a missing
measurement is never evidence.

| Kept by `linear-score`, development-quick | Dismissed by the gate | Lone piece recorded |
| --- | ---: | ---: |
| Binaries (on-target, blended, leaked), 68 | 7 | 25 |
| Leaked KOI false positives, 40 | 4 | 15 |
| Unexplained false alarms, 251 | 3 | 17 |
| Synthetic, known and validated planets, 451 | 0 | 3 |

- **Confirmed planets.** On the 500-day search before its masking change
  the gate dismisses none of the 586 the score keeps on the survey
  benchmark, and records a lone piece on 5 of them (0.9%): the five above.
  On the committed Detector (37f62ddb9) it dismisses one of 583:
  Kepler-840 b (2.50 d, 2.5% deep). Its secondary eclipse (MES 42 beside
  the transit's 1,200) would need a geometric albedo of 2, and its 930-ppm
  ellipsoidal
  variation, now coherent across the two halves of the mission (it was
  not on the earlier run), implies a companion of mass ratio 0.28. DR25
  calls it a false positive; its 2016 statistical validation predates
  both measurements. The gate reads it as a binary.
- **The record.** Of the 60 kept candidates with a lone piece on
  development-quick, 40 are binaries or KOI false positives and 3 are
  planets. It marks what later assessment (localization, statistical
  validation) should look at first.
- **Cost.** The measurements take 0.12 s of CPU per candidate (median
  0.125 s, 95th percentile 0.22 s; 374 s for all 3,207 development-quick
  candidates), 6–10% of vetting. Every binary dismissed spares its native
  fit before statistical validation: production fits took 254 and 1,140
  CPU-s in two paired reruns, the Validator's 19 and 98
  (`docs/keplerX-2026-11-02-validator-native-fit-speed.md`).

## The held-out check, the correction and adoption

Before the gate could replace `linear-score` it ran on the 1,962
development cases outside development-quick, where none of its thresholds
were chosen (`development-check.jsonl`): every development binary, blend
and contamination case, the known planets, and 400 synthetic planets, 300
of them under 10 days, where the binary tests act. All ran on the committed
Detector (37f62ddb9) with production vetting, in three runs scored together
(`development-check-run/`, `-run-planets/`, `-run-others/`;
`development_check.py`, `gate-check.json`).

**One planet lost, to how the odd/even test was read.** On the first 609
cases the gate dismissed one planet: a synthetic planet of 0.475 days found
at twice its period (0.951 d, MES 7.8, four odd and four even transits in
its curve). The odd/even test, the difference of the mean depths over the
error from their own scatter, gave 5.7, which the gate read as Gaussian
sigma; the Vetter's odd/even depth statistic on the same fold gave 1.2.
With four transits each the scatter is itself uncertain: the test is
Welch's t, which reaches 5.7 by chance about once in a thousand at those
degrees of freedom (once in a hundred million were it Gaussian). Without
the decisive odd/even rule the gate was safe but nearly powerless (2 of 49
binaries, 1 of 23 KOI false positives).

**The correction** (`vetter/policy.gaussian_sigma`): the t is judged as the
Gaussian significance of the same chance, at the fewest degrees of freedom
it can have (the smaller of the odd and even counts less one). Over a
thousand transits nothing changes; on four transits each a t of 5.7 is
2.5σ, too little even to corroborate. Re-judged from the stored metrics,
with no other change:

| Kept by `linear-score` | Dismissed by the gate before | after |
| --- | ---: | ---: |
| Development-quick, where the thresholds were chosen: binaries 68, KOI false positives 40, planets 440 | 7, 4, 0 | 7, 3, 0 |
| The first 609 held out: binaries 49, KOI false positives 23, planets 517 | 9, 4, 1 | 9, 4, 0 |

**Then the other 1,353 held-out cases ran**, a check the correction had not
seen. The rule set in advance: production takes the gate only if it
dismisses none of the planets the score keeps there.

| Kept by `linear-score`, held out | Fresh 1,353 cases | All 1,962 |
| --- | ---: | ---: |
| Binaries dismissed (on-target, blended, leaked) | 34 of 355 | 43 of 404 |
| Leaked KOI false positives dismissed | 28 of 169 | 32 of 192 |
| Unexplained false alarms dismissed | 2 of 201 | 2 of 295 |
| Known, validated and synthetic planets dismissed | 0 of 524 | 0 of 1,041 |

Lone pieces, recorded on kept signals for further assessment: 146 of the
404 binaries, 63 of the 192 KOI false positives, 32 unexplained and 8
planets.

**Adopted on 3 November 2026.** Production vetting is
`linear-score-binary-gate` (`config.json`). Of the 583 confirmed planets
the score keeps on the survey benchmark it dismisses one, Kepler-840 b,
on the corroborated evidence above (a secondary needing an albedo of 2 and
a companion of mass ratio 0.28; DR25 calls it a false positive). The
golden test `tests/test_golden_binary_gate.py` holds the gate's verdicts on
four real signals: Kepler-840 b dismissed, Kepler-158 c kept with its lone
secondary recorded, the weak held-out planet kept, a blended binary
dismissed by odd/even over a thousand transits.

**3 November, later: a fourth piece tried, not adopted.** A deep or grazing
primary (at least 50 ppt, or 10 ppt at impact parameter 0.8) as a
corroborating piece would catch 150 of 414 binaries the score keeps (the
gate: 45) and 75 of 207 KOI false positives (33), with no development or
confirmed planet lost, but it dismissed one fresh planet in each of two
tests fixed in advance, one through a defect of the secondary piece (a
planet found at three times its period shows its own transit at phase 1/3):
docs/keplerX-2026-11-03-sap-dilution-and-calibration.md.

**3 November, evening: strengthened and adopted.** The secondary is now
measured event by event and recognised as the planet's own transit at
phases j/k; so guarded, the deep or grazing primary (from 30 ppt)
corroborates a secondary or an ellipsoidal variation. The gate removes 36.5%
of the binaries and 38.2% of the KOI false positives the score keeps, no
planet lost, and passed a fresh test fixed in advance (0 of 580):
docs/keplerX-2026-11-03-binary-gate-strengthened.md.
