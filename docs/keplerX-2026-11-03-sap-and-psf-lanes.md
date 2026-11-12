# The SAP and PSF lanes (3 November 2026)

**Superseded where it measures SAP.** The SAP twin and the pre-search lane
measurement here diluted every SAP quarter by the star's expected flux over
KBonus's stitched level, one share for all quarters; SAP's real dilution
changes from quarter to quarter with the season's aperture, and that error
overstated an injected planet's SAP S/N by 18% on the median host. The SAP
lane is now decrowded per quarter, and its twin and measurements were
rebuilt: docs/keplerX-2026-11-03-sap-dilution-and-calibration.md. The PSF
Flat and unflattened-PSF findings stand.

The task (Adam, 2 November): prepare the SAP lane from the ground up and
wire it through detection, vetting and localization, compute- and
disc-efficiently; decide whether the unflattened PSF adds signal worth a
lane of its own; and decide whether combining the lanes' evidence, without
a detector rerun, should promote signals below MES 7.1, weighing the
habitable-zone planets the survey most wants against runtime and false
positives. Everything here is development data and the survey benchmark;
the sealed holdout was not touched. Vetting throughout is `linear-score`,
production until the binary gate joined it later on 3 November. Scripts and
tables:
`KeplerX/results/calibration-audits/2026-11-03-lane-information/`,
`KeplerX/results/calibration-audits/2026-11-03-sap-lane/`,
`KeplerX/results/performance-engineering/2026-11-03-sap-lane-benchmark/`
and `KeplerX/results/calibration-previews/2026-11-03-development-quick-run-sap-twin/`.

## Decisions

- **The unflattened PSF: no lane.** It covers exactly PSF Flat's cadences
  (on twelve partially covered sources the finite counts are identical),
  its noise follows PSF Flat's (correlation 0.76–0.86), and flattened
  gently it is no more sensitive (0.83–1.04 of PSF Flat's MES). As a third
  lane in a combination it lifts 0.6% more of the transits below threshold.
  Its search would cost a second PSF Flat survey for nothing measurable.
- **Combining the lanes to promote signals below 7.1: no.** Beyond SAP
  searched on its own, the combination lifts 2.8% more of the transits
  below threshold (26 of 935), in information alone and before any
  false-alarm cost; for habitable-zone durations, 2 of 60. The lanes' noise
  correlates at 0.7–0.8, and SAP's systematics would enter the joint
  statistic. What the other lane shows at a signal's ephemeris belongs in
  vetting instead: a transit on the star appears in both lanes, while
  three quarters of the detections kept on random survey curves appear in
  one only.
- **SAP: a lane, where it adds coverage, once its vetting is its own.**
  Where both lanes are searched, together they keep 26% more planets
  (population-weighted) than PSF Flat alone, and twice as many where PSF
  Flat covers only part of the mission. SAP is the only photometry of
  155,624 sources and the only long baseline of another 88,993, where a
  planet with a period of a year can show three transits at all. But its
  systematics make three times the candidates, most of its kept
  detections are its own, and the vetting score fitted on PSF Flat
  dismisses real planets on its dirtier curves (on confirmed planets the
  lanes trade 19 for 17). So:
  1. Fit the vetting on the SAP lane's own calibration sample, with what
     PSF Flat shows at the ephemeris as a feature where it exists; SAP
     reaches both efficiency targets before the survey searches it.
  2. Search SAP on the 155,624 SAP-only sources and on the 107,281 PSF
     Flat sources whose PSF Flat covers less than 90% of their cadences:
     576 CPU-hours beside PSF Flat's 385, for three quarters of SAP's gain
     (+20% of the planets where both lanes exist; the SAP-only sources'
     yield is unmeasured).
  3. Not now on the 143,642 fully covered sources: +8% planets for 569
     CPU-hours, 1.5 times the PSF Flat survey, and twice the false alarms.
     Revisit once SAP's own vetting is fitted.

Limits: 924 cases, half of the development-quick set; the twin's planets
were drawn for PSF Flat, and its injection's emulated flattening fits all
finite cadences where SAP's own flattening fits only the usable ones (on
the test curve the surviving depths agree within 1%). The habitable-zone
gain (11 to 13 planets of 139) is too small a count to measure.

## What each lane holds

KBonus delivers, per cadence of the LC2 corpus, the stitched SAP, the
stitched PSF and KBonus's flattened PSF (PSF Flat). Of the 406,547
background sources (`data/calibration-sample-design/lane-inventory.parquet`):

- **155,624 have SAP only.** They are faint (median G 17.9 in a random
  thousand) and short: median 4 quarters, a quarter of them one. They
  carry 3.39 billion SAP cadences, against 7.58 billion PSF Flat cadences
  on the other 250,923.
- **PSF Flat often covers part of a source's quarters.** For the median
  source it covers all of them, but for 68,166 (27%) less than half and
  for 39,297 (16%) less than a quarter. SAP adds 40% to the PSF Flat
  cadences of these sources.
- **Long baselines exist mostly in SAP.** 93,287 sources have 12 or more
  quarters of PSF Flat. Another 88,993 reach 12 quarters only in SAP
  (32,869 of them without any PSF Flat); for 9 quarters the numbers are
  104,902 and 47,244. A one-year planet needs about that span for three
  transits.

## Preparing SAP

`photometry/program.py`, view `sap_flat` (commit 8b3f19e8b): the stitched
SAP gets production preparation and is then flattened as KBonus flattened
PSF, with the emulator calibrated on 613 confirmed planets (unclipped cubic
B-spline, 2-day knots per quarter; `photometry/flattening.py`). A transit
therefore loses the same depth in SAP Flat as in PSF Flat, and the
calibration builder's injection through the emulated flattening applies
unchanged. A quarter whose trend reaches zero is excluded (bit 16) rather
than failing the curve. The view reads a source's LC2 shard or a Kepler
target's KBonus FITS.

## The SAP calibration sample

`calibration/lane_twin.py` (commit 3f6272bee) builds every case of a PSF
Flat manifest again on the host's SAP, keeping its case ID, host and truth:
the controls as they are (inverted and scrambled with the same seed), the
same synthetic and habitable-zone planets diluted by the star's share of
the SAP flux and passed through the flattening, and the same real signals
measured on the SAP curve. Each signal is judged on its own SAP statistics.
The development-quick twin has 1,615 cases on 727 hosts, built without a
failure (`data/calibration-sample-sap/`, 2.4 GB on discSSD; 1.8 MB per case
against 1.4 MB in PSF Flat, as SAP has more cadences).

Not twinned: synthetic binaries and blends, whose signal depends on how
each lane mixes a neighbour's light, and the artifact copies, whose models
were measured on PSF Flat. The twin's planets were drawn for PSF Flat:
their selection favoured planets detectable in PSF Flat's own coverage, so
in counts it under-samples the planets only SAP's longer coverage reaches
(the population weights stay unbiased).

## Information in each lane, before any search

`lane_information.py`: 600 development lane hosts with at least 12
quarters, three lanes built from one read of each source (PSF Flat; SAP
Flat; the stitched PSF flattened with 6-day knots, which keeps more of a
long transit than KBonus's 2-day knots), and 1,798 transits injected into
each lane's own raw flux (each lane's dilution, then its flattening),
measured with the Detector's whitening at the true ephemeris.

- **The lanes' noise is strongly correlated.** On the same cadences the
  whitened single-event statistics of PSF Flat and SAP correlate at a
  median 0.80 for 2- and 6-hour templates and 0.72 at 12 hours (10th to
  90th percentile 0.40–0.90); PSF Flat and the gently flattened PSF at
  0.76–0.86.
- **SAP is as sensitive as PSF Flat, star by star either way.** The median
  ratio of SAP's MES to PSF Flat's is 0.99–1.03 at every duration, with
  10th to 90th percentiles of 0.5–0.9 and 1.4–1.6. Three quarters of the
  spread is between stars, not between transits on one star, so it is a
  property of the star's crowding and noise in each lane.
- **The gently flattened PSF is no better than PSF Flat.** Its MES is
  1.04, 0.97, 0.89 and 0.83 of PSF Flat's (medians) for transits under 4,
  4–8, 8–13 and over 13 hours: keeping more of a long transit costs more
  in noise than it gains in depth. Its noise follows PSF Flat's closely,
  and as a third lane in the equal-weight combination it lifts 6 more of
  the 935 transits below threshold (0.6%): 10 more under 8 hours, 4 fewer
  above.
- **Combining lanes adds little to SAP on its own.** Of 935 transits below
  MES 7.1 in PSF Flat, SAP alone lifts 210 (22%) over 7.1. The unbiased
  equal-weight combination of PSF Flat and SAP lifts 209, and SAP or the
  combination 236: 26 more (2.8%) than SAP alone, before any false-alarm
  cost; all three lanes add a few more. The optimal combination's
  statistic, sqrt(m'C⁻¹m), is biased upward on noise and overstates it.
  For transits over 13 hours, the habitable-zone regime, SAP lifts 8 of
  60 and the combination 2 more.

## End to end on the twin

Half of the development-quick twins, stratified (every habitable-zone,
known and validated planet case; half of each other kind by a hash of the
case ID; 924 cases), ran in both lanes through the committed Detector
(0.5–500 days) and production vetting, side by side on the same cases
(`calibration-previews/2026-11-03-development-quick-run-sap-twin/` and
`-psf-flat-pair/`), each scored on its own sample and paired by
`lane_pair.py` (`lane-pair.json`). A planet counts when it is found at its
own period and kept.

| The same 924 cases | PSF Flat | SAP |
| --- | ---: | ---: |
| Detection: information S/N at 50% and 90% efficiency | 7.6, 11.6 | 7.6, 11.7 |
| Planets found that the vetting keeps | 94% | 90% |
| After vetting: S/N at 50% and 90% | 8.1, 12.3 | 9.3, 14.2 |
| Null original curves with a candidate (per curve) | 34% (1.6) | 71% (5.1) |
| Null original curves keeping a false alarm | 13 of 256 | 26 of 256 |
| Known planets in range found | 95.6% | 93.4% |

- **The Detector works on SAP.** At the same information S/N it finds
  planets as often in SAP as in PSF Flat.
- **The vetting does not.** The score, fitted on PSF Flat, keeps 90% of
  SAP's planets against 94%, and after it SAP misses both calibration
  targets (50% at S/N 8.5, 90% at 12). Of 150 planets found in both lanes,
  8 are kept in PSF Flat and dismissed in SAP, 3 the other way; the 8
  differ by the strongest other dip and the deepest bump over the MES, and
  by the scatter of the transits' depths, all of which SAP's systematics
  raise.
- **Together the lanes keep 26% more planets.**

| 552 injected planets, found and kept | PSF Flat | SAP | Either | Weighted gain |
| --- | ---: | ---: | ---: | ---: |
| All | 154 | 174 | 194 | +26% |
| PSF Flat information S/N under 7.1 | 17 | 34 | 42 | |
| 7.1–8.5 | 8 | 14 | 15 | |
| 8.5 and over | 129 | 126 | 137 | |
| PSF Flat covers under half the host's SAP cadences | 51 | 70 | 72 | +106% |
| covers 50–90% | 23 | 29 | 32 | +58% |
| covers 90% or more | 80 | 75 | 90 | +8% |
| Habitable-zone planets (139) | 11 | 10 | 13 | |
| Known and validated planets (98) | 92 | 89 | 92 | |

  The weighted gain uses the sample's population weights, which undo the
  draw's preference for detectable planets. The gain comes where SAP sees
  more of the star than PSF Flat: where PSF Flat covers only part of the
  mission, SAP's other quarters double the planets found. Where both lanes
  cover the whole mission, SAP adds 8%. Known planets, nearly all on
  fully covered hosts, gain nothing.
- **And more false alarms.** On the null original curves (lane-weighted)
  6.1% keep a false alarm in PSF Flat, 14.2% in SAP and 18.5% in either; by
  coverage, 6.6% become 29.5% where PSF Flat covers under half and 6.7%
  become 14.4% where it covers 90% or more. The hosts where SAP adds
  planets are the ones where it adds false alarms, as there it searches
  years of data PSF Flat lacks.
- **Selecting by the predicted S/N ratio** (`lane_noise.py`: each lane's
  6-hour noise, the star's share of its flux and its usable cadences) works
  about as well as selecting by coverage: hosts where SAP is predicted at
  over 1.2 times PSF Flat's S/N (a third of them) hold 31 of SAP's 40 added
  planets. Coverage is simpler: it is in the lane inventory already.
- **Neighbours.** SAP finds and keeps slightly more of the neighbours'
  signals than PSF Flat (eclipsing binaries 9 against 8, KOI false
  positives 15 and 15, KOI planets 10 against 8); localization sorts these.

## Real survey curves

`compare_lanes.py`: the survey benchmark's 896 diverse curves through the
committed Detector (0.5–500 days) and production vetting in each lane; PSF
Flat is the period search's final 500-day leg.

| 896 diverse curves | PSF Flat | SAP |
| --- | ---: | ---: |
| Detector + Vetter CPU | 4,974 s | 12,556 s |
| Candidates | 1,691 | 5,262 |
| Kept | 125 | 136 |
| Kept and explained by a KOI (parent, neighbour) | 8, 1 | 10, 1 |
| Kept that the other lane also made (any candidate, same or harmonic period) | 39 | 29 |
| Curves with a kept candidate | 100 | 108 |

- **The two lanes keep different things.** Only a quarter to a third of
  each lane's kept detections appear in the other lane at all, even as a
  dismissed candidate or at a harmonic; with both lanes, 174 curves have a
  kept detection instead of 100. A strong transit on the star shows in
  both lanes: 22 of the 28 PSF Flat detections kept at MES 15 or more do.
  Of SAP's 43, only 16 do, so most of SAP's strong kept detections are its
  own systematics or neighbours in its aperture, and most weak ones in
  either lane are that lane's own.
- **SAP's excess is at long periods.** SAP made 3,616 candidates at
  200–500 days against 971, 930 at 100–200 days against 303 and 480 at
  10–100 days against 189, while below 10 days the lanes made about as
  many (236 and 228). SAP's single events are heavier-tailed: its
  candidates' median MES is 9.3 against 8.1, so more of its strong dips
  line up into three- to seven-event folds. They are only mildly
  concentrated in time (the busiest 20 days hold 10% of them, 7% in PSF
  Flat), so masking field-wide epochs would not remove them.
- **The KOIs it explains are its parents'.** SAP keeps ten signals of the
  TPF target's own KOIs against eight in PSF Flat (all but one of PSF
  Flat's, and three more on two curves): light of the parent leaking into
  the background source, which an aperture separates less well than a
  PSF fit.
- **Where PSF Flat covers less than half the source's cadences** (244
  curves), SAP's CPU is 8 times PSF Flat's and its candidates 9 times; the
  two lanes keep 28 and 31, and 44 curves have one in either.

**493 confirmed-planet hosts** (Kepler targets, their SAP from the KBonus
FITS; `known-planets-compare.json`). Alone, SAP does as well as PSF Flat:
of the 621 planets below 200 days with three transits it finds 594 at
their own period and keeps 579 (PSF Flat 592 and 577), and of the 10 at
200–500 days 7 and 6 (8 and 6). As a second lane it keeps 19 planets PSF
Flat does not (Kepler-9 b among them, at MES 116 in SAP), while SAP's
vetting drops 17 that PSF Flat keeps, some at MES 21–28: together the
lanes keep 602 of the 631, 3.3% more than PSF Flat alone, on hosts both
lanes cover fully.

**1,000 SAP-only sources** (development, outside the calibration sample,
drawn by a hash of the source ID), by their quarters:

| Quarters | Sources in the corpus | Curves | Candidates per curve | Curves with a kept candidate | Detector + Vetter CPU per curve |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 51,645 | 356 | 0.03 | 2 (0.6%) | 0.07 s |
| 2–5 | 40,173 | 238 | 0.90 | 34 (14%) | 1.7 s |
| 6–11 | 30,937 | 207 | 2.9 | 36 (17%) | 5.8 s |
| 12–18 | 32,869 | 199 | 4.9 | 30 (15%) | 10.5 s |

With two or more quarters, 15% of SAP-only curves keep a candidate,
against 11% of PSF Flat survey curves; across the corpus that is about
16,000 SAP-only sources for localization, beside about 27,000 from PSF
Flat. How many are planets the SAP-only sample cannot say: it has no
injections, and these sources are not in the calibration sample.

## What it costs

Each class's mean Detector-plus-Vetter CPU per curve on these benchmarks,
times its sources in the corpus (the runs shared the Mac, so the ratios
are firmer than the hours):

| Search | Sources | CPU-hours |
| --- | ---: | ---: |
| PSF Flat, the survey as planned (0.5–500 days) | 250,923 | 385 |
| SAP on PSF Flat sources that PSF Flat covers under half | 68,166 | 230 |
| SAP on PSF Flat sources covered 50–90% | 39,115 | 180 |
| SAP on PSF Flat sources covered 90% or more | 143,642 | 569 |
| SAP on SAP-only sources | 155,624 | 166 |

- **Everything in SAP** adds 1,145 CPU-hours to PSF Flat's 385, three
  times the survey's compute, and nearly doubles the curves with a kept
  candidate to localize.
- **SAP where it adds coverage** (SAP-only sources and PSF Flat sources
  covered under 90%) adds 576.
- **SAP on fully covered sources** alone costs 1.5 times the whole PSF Flat
  survey, for curves PSF Flat already searches.
- **Disc.** A run that keeps a SAP curve's prepared photometry whenever
  it has a candidate (three quarters of SAP curves do) took 1.1 GB for the
  896 diverse curves, about 500 GB for the corpus. The preparation
  regenerates bit for bit from LC2, so a survey run needs to keep
  photometry only for kept signals, or none.

## How the survey should run SAP

What is done once per source and what once per lane:

- **Reading: once.** One read of the source's LC2 row group gives both
  lanes' columns (PSF Flat, stitched SAP and their errors, quality,
  flatten mask); a Kepler target's KBonus FITS carries both as well.
- **Preparation, detection and vetting: per lane.** Each reads its own
  flux. The noise of each lane at 2, 6 and 12 hours, and the star's share
  of each lane's flux, cost no search and are known before it
  (`lane_noise.py`), so a job can decide there which lanes to search.
- **Merging, then localization and validation: once per signal.** The two
  lanes' candidates are matched by ephemeris (same or harmonic period, one
  transit's epoch); a match becomes one signal carrying both lanes'
  measurements, and the lane it appears in is itself evidence (a
  transit on the star shows in both; most single-lane detections are that
  lane's own systematics or, in SAP, its aperture's neighbours).
  Localization reads the target pixels at the signal's ephemeris and, from
  the packet, only which cadences and quarters are usable (not the flux),
  so a SAP packet localizes as a PSF Flat one does, over SAP's longer
  coverage; it runs once per signal and source.
- **Outputs.** Candidates and vetting rows carry their lane; prepared
  photometry is kept only for kept signals (it regenerates bit for bit).
