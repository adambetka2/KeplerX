# SAP's dilution, decrowding and calibration (3 November 2026)

The task (Adam, 3 November): make sure the SAP lane's calibration sample was
prepared correctly (its signals on the same footing as its light curves);
make the Vetter lane-aware and fit a second set of weights on SAP's
calibration sample until SAP meets the calibration targets; decide whether
rescoring the two lanes' detected signals together catches more planets;
say whether SAP runs on all sources; strengthen the binary gate (30-50% more
binaries caught for at most the one confirmed planet it costs now); then
performance-engineer the new subsystems. Development data, the survey
benchmark and Gaia DR3 eclipsing binaries only; the sealed holdout was not
touched. Scripts and tables:
`KeplerX/results/calibration-audits/2026-11-03-sap-sample-audit/`,
`KeplerX/results/calibration-previews/2026-11-03-sap-calibration/`.

## SAP's dilution was wrong, quarter by quarter

**What KBonus's stitched SAP is.** Each LC2 cadence carries the stitched
SAP, and each quarter extension of a KBonus file its own raw SAP. The
stitched SAP is the raw SAP rescaled to one level for all quarters with its
slow trend divided out: on one twin host the raw quarter levels run from
3,840 to 15,490 e-/s (10-pixel apertures in one season take in a
neighbour), the stitched levels are 6,940 in every quarter, and the
stitched/raw ratio drifts smoothly within a quarter while the relative
scatter is the raw quarter's. Measured on Gaia eclipsing binaries
(`sap_dilution_gaia_binaries.py`), stitched and raw SAP eclipse depths agree
to 0.1% (median 0.999, interquartile 0.992-1.005; 0.98-0.99 above 5 days).
So the stitching keeps each quarter's relative depths, and a transit on the
star loses its depth times **the star's share of that quarter's aperture
light**, which changes with the season.

**What the twin did.** The first SAP twin (3 November, overnight) diluted
every quarter by min(1, expected / stitched level), the star's expected
Kepler-band flux on PSF Flat's G zero point over the stitched level: one
share for all quarters, and a level that is no quarter's.

**The test.** 1,025 Gaia DR3 eclipsing binaries are lane sources; 580 with
the eclipse present at Gaia's period and Gaia's depth error under 20% give
6,306 quarters with an eclipse S/N above 8. Each quarter's depth
(max-to-min of the fold, as Gaia's derived depth is defined) over Gaia's,
against each share model (`analyse_dilution.py`, `dilution-summary.json`):

| Share model | Depth / (Gaia x share), median (IQR) | Spread across curves | Spread between a curve's quarters | Slope between quarters |
| --- | --- | ---: | --- | ---: |
| The twin's: expected / stitched level | 0.77 (0.45-0.98) | 0.46 | 0.37 -> 0.37 | none |
| expected / raw quarter level | 0.79 (0.50-0.94) | 0.34 | 0.37 -> 0.21 | 0.87 |
| FLFRCSAP x expected / raw level | 0.84 (0.51-0.99) | 0.33 | 0.37 -> 0.19 | 0.92 |
| KBonus's CROWDSAP | 0.66 (0.29-0.97) | 0.71 | 0.37 -> 0.29 | 1.45 |
| **FLFRCSAP x PSFFRAC x expected / raw level** | **1.10 (0.91-1.26)** | **0.24** | **0.37 -> 0.12** | **0.98** |
| PSF Flat's own model on PSF Flat (check) | 1.05 (0.88-1.28) | 0.28 | | |

(spreads are robust sigmas of the natural log). PSFFRAC is KBonus's
fraction of the star's PSF inside the TPF; its median on these binaries is
0.39, because most lane sources sit at the edge of another target's
postage stamp, so a fixed aperture there holds a fraction of their light.
FLFRCSAP is the aperture's capture of that light. With both, the depth
follows the share between a curve's quarters with slope 0.98, the spread
between quarters falls to a third, and Gaia's depth times the share is met
within 10% at every share (1.18, 1.13, 1.13, 1.14, 1.05, 0.93 from share
below 0.2 to above 0.95; PSF Flat's own model measured the same way: 1.05).
On the strongest eclipses (S/N above 50) the model's own spread between
quarters is 0.075.

**How wrong the twin was** (`twin_share_error.py`, 667 twin hosts): an
injected planet's S/N under the measured shares over the twin's is 0.82 on
the median host (10th-90th percentile 0.46-1.38); 45% of hosts were
overstated by more than 20% and 13% understated by more than 20%. The mean
share was 0.31-1.0 in the twin and 0.15-0.62 measured, by G bin. Everything
measured on the first twin (the lane pair, the SAP gains by coverage, the
habitable-zone count) and the pre-search lane measurement, which diluted
SAP the same way, overstated SAP; they are superseded below.

PSF Flat is unaffected: the PSF fit models the whole PSF, and its own
dilution model, validated on the same binaries on 31 October, holds (1.05
here).

## What changed

- **The sap_flat view decrowds** (`photometry/star_share.py`,
  `photometry/program.py`): after normalising and flattening, each
  quarter's deviations from its level are divided by the star's share of
  that quarter's aperture, as Kepler's PDC removes crowding, so a planet
  keeps one depth in every season and the noise grows where the star's
  light is a small part of the aperture's. The share needs the quarter's raw
  SAP level, FLFRCSAP and PSFFRAC (LC2 quarter and header tables, or the
  KBonus FITS's quarter extensions) and G and BP-RP (the KBonus primary
  header). A quarter without them keeps its flux (share 1); a quarter whose
  share is below 0.01 is excluded (exclusion bit 128). PSF Flat's
  expected-flux constants moved there from the sample builder unchanged.
- **The twin injects as the sky does** (`calibration/lane_twin.py`): bodies
  go into the raw stitched SAP diluted by each quarter's share times the
  model's own error between quarters (lognormal, 0.075, seeded by the
  case), and the curve is then prepared exactly as the view prepares a real
  one (the real smoother, decrowding by the modelled share). Synthetic
  binaries on the host are twinned too; blends are not (their share depends
  on where the neighbour sits against the aperture), real leaked binaries
  and KOI false positives are. Hosts whose every quarter falls below the
  share floor carry no usable SAP of the star and are not twinned (23 of
  1,666 development-quick cases, 18 of 1,720 of the binary gate's check,
  8 of 1,900 fresh).
- **The end-to-end test** prepares its K dwarf with the shares recorded in
  its fixture (0.82-0.95), injects the 70-day planet through the twin's
  path and checks decrowding gives the depth back against the local
  baseline (local transfer 0.985). At 2 Earth radii that planet's SAP S/N
  is 7.9 once each quarter keeps its own share, on the detection threshold,
  so the test now uses 2.5 Earth radii (S/N 12).
- **The Vetter knows the lane** (`vetter/program.py`, `vetter/policy.py`):
  it reads the packet's view, scores with that lane's own weights
  (`LANE_SCORES`, PSF Flat's are the policy's), refuses a lane with no
  fitted score, and records the lane on every vetting row (it labelled every
  row `delivered_psf_flat` before). Scoring applies each lane's score.
- **The Dump no longer samples periods it throws away**
  (`detector/compiled_search.py`, `detector/upstream_adapter.py`): 26 ms
  less per Dump in the Detector, the Vetter and the cross-lane measurement,
  bit for bit the same (the new subsystems' cost, below).

## A stronger binary gate

**Outcome: production keeps its gate.** A fourth piece (a deep or grazing
primary) would catch about three times the binaries, but it dismissed one
fresh planet in each of two tests whose rules were fixed in advance; one of
those losses exposed a defect in the secondary piece to fix first.

**Later the same day:** the secondary piece was fixed (measured event by
event, own transits at j/k recognised) and the deep primary adopted at 30
ppt: 36.5% of binaries and 38.2% of KOI false positives removed in PSF
Flat, 46% and 44% in SAP, no planet lost, a fresh test passed
(docs/keplerX-2026-11-03-binary-gate-strengthened.md). The binaries and KOI
false positives this record counts as kept are the morning gate's.

Scripts: `KeplerX/results/calibration-audits/2026-11-03-binary-gate-strengthening/`
(`evidence_frame.py`, `evidence.parquet`). Data: every candidate the score
keeps on the current Detector's PSF Flat runs of the gate's development
check (1,962 cases) and half of development-quick (924), and the 583
confirmed planets of the survey benchmark the score keeps.

**What the gate misses.** It removes 45 of the 414 binaries the score keeps
(10.9%) and 33 of 207 KOI false positives (15.9%). Of the binaries it
misses, 137 of 369 (and 56 of 174 KOI false positives) show a significant,
unique secondary deeper than any planet's occultation, but a secondary alone
would also dismiss 7 development and 5 confirmed planets, which is why the
gate wants two pieces (Adam: never on a secondary alone). Odd/even (6) and
ellipsoidal (7) evidence is rare among them.

**A fourth piece: a deep or grazing primary.** A primary eclipse deeper than
30 ppt, or deeper than 10 ppt with a fitted impact parameter of at least
0.8. Binaries the score keeps have median depth 54 ppt and b 0.82; planets
0.63 ppt and 0.52. Planets do reach it around small stars (Kepler-45 b, a
hot Jupiter on an M dwarf, is 37 ppt deep), so it never decides alone; it
counts as one of the two pieces.

| Kept by the score | Production gate | With the primary piece |
| --- | ---: | ---: |
| Synthetic binaries (194) | 19 | 83 |
| Blended binaries (90) | 7 | 26 |
| Real binaries leaking into lane curves (130) | 19 | 50 |
| KOI false positives (207) | 33 | 79 |
| Development planets (1,299) | 0 | 0 |
| Confirmed planets (583) | 1 (Kepler-840 b) | 1 (Kepler-840 b) |
| Unexplained candidates (383) | 4 | 22 |

Binaries caught: 45 -> 159 of 414 (3.5 times); real binaries and KOI false
positives together: 52 -> 129 of 337.

- **Thresholds.** No development or confirmed planet is lost anywhere on a
  grid of depth 30-100 ppt, b 0.7-0.9 and grazing depth 10-30 ppt; the
  catch runs from 118 to 167 binaries across it. The chosen point is the
  most inclusive one that still names a physical case (3% deep, or 1% deep
  and grazing).
- **Margin.** 99 of the 1,882 planets show the piece (Kepler-45 b the one
  confirmed); none shows a second. If the pieces were independent, about
  one planet in 2,200 would carry both by chance.
- **Fresh tests, rules fixed in advance: not adopted.** Development
  planets no run had used, PSF Flat on the current Detector, the rule each
  time that no planet the score keeps may be dismissed:
  - 1,100 planets (600 habitable-zone, 500 synthetic; 201 kept by the
    score): at 30 ppt it dismissed one, a 1.6 Jupiter-radius planet on a
    1-day orbit of a G 18.6 K dwarf, 3.9% deep, whose fold holds a
    systematic dip at a quarter phase that passes for a secondary. Deep
    giants around small stars do reach 30 ppt; the rule moved to 50 ppt
    before the next set ran (with no loss on any development, first-fresh or
    confirmed planet at 50 ppt; 150 binaries and 75 KOI false positives).
  - 1,000 more synthetic planets (375 kept): at 50 ppt it dismissed one, a
    1.4 Jupiter-radius planet on a 126-day orbit of a G 18.5 K dwarf, 7.2%
    deep, found at three times its period. Its own transit at phase 1/3 of
    that fold passed for a unique secondary because the transit at 2/3 fell
    in a gap: the secondary piece recognises the planet's own transit at
    half phase (a detection at twice the period) but not at a third.
  Over both sets the 50 ppt rule loses 1 of 576 planets the score keeps
  (0.17%; weighted by occurrence 0.02%), and no confirmed planet beyond
  Kepler-840 b. It failed the rule fixed for it, so production keeps the
  three-piece gate; the trial policy and its piece were removed (they are in
  commit d49057e74). The way back: make the secondary piece recognise the
  planet's own transits at phases j/k of a detection at k times its period,
  then test the 50 ppt piece the same way on the 1,074 development planets
  no run has used. It would catch 3.3 times the binaries (150 against 45
  of 414) and 2.3 times the KOI false positives (75 against 33).
- **Code.** The gate's pieces are one function each now; production
  verdicts are unchanged on all 12,611 scored PSF Flat candidates checked
  (development, both fresh sets, confirmed planets).
- **Not taken: a secondary at half the orbit.** Counting a secondary
  within 0.03 in phase of half the orbit (a circular binary's) as a second
  piece would catch 176 of 414 binaries and 82 of 207 KOI false positives
  with no development planet lost (the planets' false secondaries sit at
  random phases; only Kepler-840 b's is at half). It is still a dismissal
  on the secondary alone, which Adam's rule forbids; it is his to decide.

## SAP's own vetting, reading PSF Flat at the ephemeris

Scripts: `KeplerX/results/calibration-audits/2026-11-03-sap-vetting-fit/`
(`fit_sap_cross.py`, `fit-development-quick.json`, `judge_sap.py`,
`lane_pair.py`) and `…/2026-11-03-cross-lane-rescoring/` (`cross_lane.py`,
the Vetter's own measurement). Fit set: the rebuilt SAP twin of
development-quick, 1,643 cases (all complete), 9,614 candidates, 339 null
originals; vetting metrics only, scores applied afterwards.

**What PSF Flat sees at a SAP candidate's ephemeris.** SAP holds everything
PSF Flat holds from the same pixels, plus its aperture's systematics. At
SAP's planets PSF Flat's MES (scaled by the share of SAP's transits it
observed) is about SAP's own: median log10 ratio -0.04, 10th percentile
-0.50. At SAP's unexplained candidates it is mostly nothing: median -0.85
(-0.82 on null originals). The other way round it does not work: PSF Flat's
own false alarms show in SAP too (median SAP MES 3.9 at their ephemerides),
so SAP's agreement cannot rescue PSF Flat planets its score dismisses: on
the binary gate's development check, keeping a dismissed PSF Flat candidate
when SAP's MES there reaches 7.1 would bring back 29 of 39 dismissed planets
and 1,065 unexplained candidates and 188 binaries and KOI false positives
(`psf-rescue.json`). The measurement is one-way: PSF Flat informs SAP's
vetting only.

**Two SAP scores.** `calibration.vetting_policy`'s penalised logistic score
on the eight PSF Flat features, fitted on SAP; and the same with the
cross-lane ratio, for candidates whose source has PSF Flat transits at the
ephemeris (88% of the fit set's candidates; every planet). Out of fold by
source star, with the production binary gate on top, efficiency after
vetting at S/N 8.5 and 12 (targets 50% and 90%):

| Budget (planets the cut may dismiss) | Score | Planets kept | Efficiency at 8.5 / 12 | Null originals keeping a false alarm |
| ---: | --- | ---: | --- | ---: |
| (PSF Flat's own score on SAP) | | 92.1% | 46.8% / 85.4% | 45 of 339 |
| 0.02 | with PSF Flat | 96.8% | 57.3% / 90.8% | 108 |
| 0.03 | with PSF Flat | 95.8% | 56.8% / 90.5% | 75 |
| 0.04 | with PSF Flat | 94.7% | 55.1% / 89.5% | 59 |
| 0.06 | with PSF Flat | 93.1% | 51.1% / 87.0% | 32 |
| 0.08 | with PSF Flat | 91.6% | 50.8% / 87.5% | 23 |
| 0.03 | SAP alone | 96.3% | 55.7% / 90.0% | 153 |
| 0.06 | SAP alone | 93.1% | 50.1% / 87.9% | 63 |
| 0.10 | SAP alone | 88.9% | 42.5% / 80.8% | 26 |

- PSF Flat's measurement removes a third to a half of SAP's false alarms at every budget for the
  same planets.
- Meeting 90% at S/N 12 takes budget 0.03: then 22% of null originals keep
  a false alarm, against 3.8% in PSF Flat's own lane on the same hosts.
  Matching PSF Flat's rate takes budget 0.08, where 90% efficiency moves to
  about S/N 12.6. At the false-alarm level of PSF Flat's score on SAP
  (budget 0.05), SAP's own score is better on both targets (about 53% and
  88%).
- The surviving false alarms sit at long periods (80 of 98 at 30-500 days,
  where 76 of 363 kept planets are) and 40% of them have no PSF Flat
  transit at their ephemeris; a stricter cut for those would cost the twin
  no planet, but the twin's planets were drawn where PSF Flat sees them and
  under-samples the planets only SAP's coverage reaches, which such a cut
  would hit, so both scores keep one budget.
- **Confirmed planets** (`confirmed_sap.py`; the decrowded SAP lane on the
  survey benchmark's 493 confirmed-planet hosts, 632 planets, with PSF
  Flat's measurement from the hosts' PSF Flat packets; every planet SAP
  finds exactly has it). SAP finds 599 exactly (PSF Flat's own lane 600) and
  keeps, by budget: 593 (99.0%) at 0.03, 591 (98.7%) at 0.04, 580 at 0.06,
  575 at 0.08, 560 at 0.10; with PSF Flat's score instead, 585. PSF Flat's
  own lane keeps 582 of its 600 (97.0%); the two lanes together keep 605,
  604, 600, 597 and 596.
- **What a stricter cut costs in the merged survey** (`lane_pair.py`, the
  924 development-quick cases both lanes ran, each lane with its own
  score): SAP's added planets hardly move (either lane keeps 173, 172, 170
  and 169 injected planets at budgets 0.03, 0.04, 0.06 and 0.08, PSF Flat
  alone 152), while SAP's null originals keeping a false alarm fall from
  25% to 13%, 8% and 5% (PSF Flat 3.8%): the planets SAP adds are the
  strong ones where it has coverage PSF Flat lacks.
- **Held out** (`judge_sap.py`, `judge-development-check-subset.json`): the
  SAP twins of 699 of the binary gate's development check cases, which
  SAP's fit never saw (every twinned binary, 200 leaked real signals, 120
  known and 120 synthetic planets; 5,112 candidates, 93% with PSF Flat's
  measurement), with the installed scores: efficiency after vetting 52.2% at
  S/N 8.5 and 93.0% at S/N 12 (both targets met); detectable planets found
  exactly and kept 86.2% of 123 (target 35%); planets kept 95.5%
  (synthetic 93.2% of 118, known 97.1% of 173, leaked KOI planets 17 of
  17); kept: 74% of synthetic binaries, 42% of leaked binaries, 69% of KOI
  false positives, 2.7% of unexplained candidates.
- **Installed (`vetter/policy.py` LANE_SCORES): the score with PSF Flat's
  measurement at budget 0.04, PSF Flat's own; SAP's plain score at 0.06.**
  The first meets Adam's 98% vetting completeness on confirmed planets
  (98.7%; every confirmed planet SAP finds exactly has PSF Flat's
  measurement), 50% efficiency at S/N 8.5 (55.1%) and misses 90% at S/N 12
  by half a point out of fold (89.5%; PSF Flat's own lane misses it on half
  the set too, at 12.3), and halves SAP's false alarms against budget 0.03
  for one confirmed planet fewer in the merged lanes. The plain score judges
  the SAP-only sources and the candidates PSF Flat never saw in transit. At
  0.04 it left 18.6% of real SAP-only curves keeping a candidate (below); at
  0.06, 11.9%, for no planet on any set: out of fold on development-quick
  (`installed_mix_out_of_fold.py`) planets kept and efficiency stay 94.7%,
  55.1% and 89.5% while null originals keeping a false alarm fall from 59
  to 47 of 339; the check subset, the fresh set and both lane pairings keep
  the same planets as with 0.04 for both (fresh null originals with a false
  alarm: 29 to 27 of 185), confirmed planets 591 either way. Out of fold
  on every fit candidate, as if none had PSF Flat, the plain score at 0.06
  gives 50.1% at S/N 8.5 and 87.9% at 12; the twin has no SAP-only hosts
  yet to measure it where it applies (plan item 1).

## Combining the lanes below threshold: still no

Rerun of last night's pre-search measurement with SAP as production now
prepares it (`KeplerX/results/calibration-audits/2026-11-03-lane-information-decrowded/`,
`hosts-600-summary.txt`; 593 hosts, 7 without a quarter holding 1% of the
star's light in SAP; 1,777 transits injected into each lane's raw flux,
measured with the Detector's whitening at the true ephemeris):

- **SAP is less sensitive than PSF Flat, not equal.** SAP's MES over PSF
  Flat's: median 0.77, 0.77, 0.79 and 0.78 for transits under 4, 4-8, 8-13
  and over 13 hours (10th-90th percentiles about 0.5-1.2); SAP is the
  better lane for 18-24% of transits (last night, with the old dilution:
  about 1.0 and half).
- **The noise correlation stands:** 0.80, 0.80 and 0.72 at 2, 6 and 12 hours.
- **Combination:** of 925 transits PSF Flat leaves under MES 7.1, SAP alone
  lifts 77 over it, the equal-weight combination 85, and SAP or the
  combination 98: 21 more than SAP alone (2.3%), before any false-alarm
  cost; over 13 hours, 1 more of 58.
- **Rescoring detected signals across lanes** (above): PSF Flat informing
  SAP's vetting removes a third to half of SAP's false alarms; SAP cannot
  rescue PSF Flat's dismissals (43 false positives per planet).

So a middle stage that promotes signals below 7.1 by combining lanes is not
built; the lanes' evidence is combined where it pays, in SAP's vetting.

## The rebuilt SAP sample, checked

`KeplerX/results/calibration-audits/2026-11-03-sap-sample-audit/twin_checks.py`
(`twin-checks.json`), on the rebuilt development-quick twin:

- **Same photometry for signal and curve.** Each host's G and BP-RP in its
  truth (Gaia DR3) against the KBonus primary header's GMAG and BPMAG -
  RPMAG that its shares were computed from: median difference 0.00 mag,
  90th percentile 0.02 mag in G; colours identical.
- **Injected depths are in the curve.** For injected planets with S/N
  above 20, the depth measured in the prepared SAP against its local
  baseline over the recorded delivered depth: median 1.09 (IQR 1.04-1.15);
  the PSF Flat sample, measured the same way: 1.06 (1.02-1.15). The
  recorded depth is taken against 1, the measured one against the ring
  around the transit, which the 2-day spline lowers a little; the
  convention is the same in both lanes.
- **The S/N ruler means the same in both lanes.** The Detector's MES over
  the recorded information S/N for planets found exactly: SAP 0.92 (IQR
  0.80-1.06), PSF Flat 0.93 (0.81-1.10); on the same planets, SAP's ratio
  over PSF Flat's: 1.04 (0.95-1.12).
- **A residual: decrowded SAP depths run 5-7% deep.** Real known planets
  (S/N above 15) are 1.07 times as deep in decrowded SAP as in PSF Flat
  divided by its star's share (IQR 1.03-1.13), and the Gaia binaries gave
  1.10 for SAP's model against 1.05 for PSF Flat's: SAP's shares run about
  5% low. The twin injects with the modelled share, so its planets are that
  much weaker than real ones of the same radius: efficiency by S/N is
  unaffected, efficiency by radius is conservative, and a SAP radius is
  about 3% large. A fitted SAP scale for the share is in the plan.

## The new subsystems' cost

Measured on the quiet machine (one process, single-threaded numerics; CPU
time, and wall clock where the disc enters), after first figures taken
beside the day's batches ran 1.1-2.5 times higher; scripts and
`quiet-timing.json` in
`KeplerX/results/performance-engineering/2026-11-03-new-subsystem-costs/`.

| Piece | Cost | Over the survey |
| --- | --- | --- |
| SAP preparation: LC2 cadences, the shares' inputs (quarter and header tables, the source row), normalising, flattening, decrowding | 63 ms CPU to load (92 ms wall with the disc read; 80 ms CPU before the source row was reused, commit 3c43f4f0d, bit for bit the same on 30 sources) and 22 ms to prepare, per source; 120 random sources, all prepared | about 6 CPU-hours on the 262,905 SAP sources, 0.5% of SAP's detection and vetting (about 17 CPU-seconds per source) |
| PSF Flat at a SAP candidate's ephemeris (vetter/cross_lane.py: the Detector's dump and whitening at the candidates' durations only) | 0.21 s per source, 29 ms per candidate (300 random twins, 2,167 candidates; 0.24 s before the Dump fix below); PSF Flat's packet loads in 14 ms | about 6 CPU-hours on the 107,281 partly covered sources; 73% of it is whitening, which the survey job can take from PSF Flat's own detection |
| Binary evidence (vetter/binary_evidence.py: the fold recomputed with every other signal masked) | 71 ms per candidate on 30 SAP twins (203 candidates), 4.9% of their vetting CPU (3.1% on the first, mixed sample); nearly all of it the one whitening at the candidate's duration with the other signals masked, which the measurement needs; the fold and the odd/even and ellipsoidal fits take under 1 ms | about 10 CPU-hours in PSF Flat, about 100 in SAP (more candidates per curve) |
| The Dump's construction (each Detector, Vetter and cross-lane run builds one) | 63 -> 37 ms: upstream's constructor sampled about 320,000 trial periods (28 ms) that make_dump replaced at once; it no longer does (`detector/compiled_search.deferred_period_sampling`; periods, durations, limb darkening and a whitening bit for bit the same on 48 dumps of both stages) | about 8 CPU-hours saved over both lanes |
| The Detector's transit requirement and masking of rejected folds | about 1 ms per curve of 1.96 CPU-seconds of detection | negligible |
| Lane-aware scoring | one dictionary lookup per candidate | negligible |

None is a bottleneck. The survey is bound by the Detector's GPU screen (SAP
5.9 GPU-seconds per curve, PSF Flat 3.4-4.4; plan item 6: the whole study
in four days) and, behind it, by vetting CPU, which SAP's extra candidates
per curve raise. In that CPU the Vetter whitens three times per candidate,
each with its own mask (upstream's weak-secondary fold, its MES-shape fold
and the binary evidence): 16% of SAP's vetting CPU on the 30 twins. Sharing
one whitening between folds whose masks agree is the lever there, if
vetting CPU comes to bound the survey (plan item 6).

## Held out on fresh planets, and where SAP runs

**Fresh development set** (`judge-development-fresh-subset.json`; SAP
twins of 1,100 cases no run had used: all 600 habitable-zone planets, 250
synthetic planets, 250 null originals): efficiency after vetting 51.4% at
S/N 8.5 and 90.6% at S/N 12 (both targets met), detectable planets found
exactly and kept 73.9% of 111 (target 35%), planets kept 90.2%, null
originals keeping a false alarm 27 of 185 (14.6%). Habitable zone: 16 found
and kept of 660 searchable (12 of 22 detectable in SAP); PSF Flat's own lane
on the same cases keeps 33 (32 of 40 detectable).

**Both lanes case for case** (`lane-pair-development-fresh.json`, 972
paired signals; and the development-quick half above), each with its own
score:

| PSF Flat covers | Injected planets kept, PSF Flat -> either lane (fresh) | Same (development-quick half) | Habitable zone, PSF Flat -> either (fresh; half) |
| --- | --- | --- | --- |
| < 50% of the SAP cadences | 17 -> 24 | 50 -> 62 (+22.7% weighted) | 0 -> 0; 5 -> 5 |
| 50-90% | 19 -> 20 | 23 -> 27 (+9.6%) | 3 -> 3; 3 -> 3 |
| >= 90% | 81 -> 85 (+1.3%) | 79 -> 83 (+0.9%) | 30 -> 31; 3 -> 4 |

**Real survey curves** (`real_curves_sap.py`; `real-curves-diverse.json`,
`real-curves-sap-only.json`; vetted metrics judged by the installed
scores). Real curves hold few planets, so the share of curves keeping a
candidate is mostly false alarms. Of 299 diverse survey-benchmark curves,
9.7% keep a SAP candidate and 11.7% a PSF Flat one (the final 500-day run):

| PSF Flat covers | Curves | PSF Flat | SAP | Either |
| --- | ---: | ---: | ---: | ---: |
| < 50% of the SAP cadences | 79 | 7.6% | 13.9% | 17.7% |
| 50-90% | 44 | 6.8% | 15.9% | 15.9% |
| >= 90% | 176 | 14.8% | 6.3% | 17.1% |

On 295 development SAP-only sources the plain score keeps a candidate on
27.8%, 23.4%, 18.6%, 11.9%, 6.4% and 5.1% of curves at budgets 0.02, 0.03,
0.04, 0.06, 0.08 and 0.10 (1.8 candidates per curve); 0.06 is installed.

**Decision: SAP on the SAP-only sources and where PSF Flat covers under 90%
of the cadences; not on the 143,642 fully covered sources for now.** On
fully covered sources SAP adds about 1% of planets weighted by occurrence
and two habitable-zone planets beyond PSF Flat's 34 across both sets
(near-threshold planets one lane happens to keep), for about 4.9 GPU-days,
as much again as the PSF Flat survey, and about ten points more null curves
keeping a false alarm; it is the least valuable SAP compute per planet,
and the whole study is to fit in four days (plan item 6). If that work
frees time, SAP on the fully covered K and M dwarfs, where the habitable-zone
planets are, is the variant to measure.
