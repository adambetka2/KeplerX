# Searching 0.5–500 days (2 November 2026)

The task: extend the Detector's period search from 0.5–200 to 0.5–500
days, check long-period duration coverage, alias handling and the
observed-transit requirement, and measure the added Detector-plus-Vetter
runtime on representative survey curves. Scripts, tables and runs:
`KeplerX/results/performance-engineering/2026-11-02-period-search-to-500-days/`.
Everything here is the PSF Flat lane; the SAP lane, built since, runs the
same Detector ([SAP and PSF lanes](keplerX-2026-11-03-sap-and-psf-lanes.md)).
The sealed holdout was not touched.

## What changed (commits 323c08cda and 37f62ddb9)

- **Period range.** `config.json` `detection.period_max_days` is 500.
- **Transit requirement.** The Detector drops a detection that rests on
  fewer events than the three required transits
  (`detector/candidate_ephemerides.py`, `carrying_events`), below.
- **What a dropped fold rests on is masked (37f62ddb9).** Its carrying
  events are masked like a detection's and the search goes on
  (`detector/repeat_rule.py`), below.
- **Scoring.** Calibration scoring's period bins reach the configured top,
  with 200 days kept as an edge.

## The plain extension

First the range alone, nothing else changed, on the survey benchmark:
paired legs, the same code side by side on 4 workers each.

**896 diverse survey curves:**

- **Compute.** Detector CPU rose 24%, Vetter CPU 49%, the two together
  39% (55% weighted to the survey). The trial periods rose about 4%; the
  extra candidates explain the rest.
- **Candidates.** 1,259 new ones beyond 200 days, 1.4 per curve, at a
  median MES of 8.4. Curves with a kept candidate rose from 10.7% to 13.5%.
- **Kept candidates.** Unexplained ones rose from 122 to 156. Those that a
  KOI explains (8 of the parent target, 1 of a neighbour) did not change.
- **The search below 200 days.** It changed too: 543 of the 1,155
  detections of the 200-day search were not made again, and 133 new ones
  appeared. Long-period detections take passes, masks and the
  ten-detection limit.

**493 confirmed-planet hosts:**

- **Planets below 200 days.** The same 591 of 621 found at their own
  period (one lost at MES 6.7, one gained), and two more kept.
- **Planets at 200–500 days.** 8 of the 10 found at their own period,
  where the 200-day search found none: 7 at a harmonic, 3 missed. Six are
  kept by vetting: Kepler-22 b (289.9 d), Kepler-849 b (394.6 d),
  Kepler-1536 b (364.8 d), Kepler-1690 b (234.6 d), Kepler-1632 b (224.2 d)
  and Kepler-1040 b (201.1 d). Kepler-1514 b (217.8 d) and Kepler-1635 b
  (469.6 d) are dismissed.
- **Kept detections below 200 days that went.** 45 were not made again.
  None is a confirmed planet. Examples: Kepler-849 b at half its period
  (now found at its own); a two-event fold at a third of 436.65 days on
  Kepler-321's host (below); an alias of a 354.8-day false-positive KOI; a
  second detection of Kepler-88 b, whose transit times swing by half a
  day.
- **Compute.** Detector plus Vetter CPU rose 36%.

The plain-extension legs shared the Mac with eight orphaned workers of a
stopped batch from 20:25 UTC, which loaded both legs alike.

The long-period candidates are mostly not periodic signals. Real KBonus
curves have 2–50 independent dips beyond 5σ each, where Gaussian noise
would give about 0.01. Folds at 200–500 days have three to seven events,
and any three such dips equally spaced make a three-event fold above MES
7.1. Below 200 days a fold has at least seven events, and the other four or
more dilute the dips.

## The transit requirement

The search already requires three observed transits, and its single-event
veto (`detector/single_event_veto.py`) drops a fold carried by one outlier
event. Two kinds of fold still passed:

- **Two deep single events a period apart, nothing at the other epochs.**
  One survey curve has events of single-event statistic 42.5 and 31.6, 85
  days apart. The 200-day search reported them as an 85.14-day "planet"
  with MES 24 over 17 transits, and vetting kept it. Its other 15 epochs
  read −2.5 to 2.7. On Kepler-321's host, events of 81.9 and 81.3 made a
  436.65-day fold with two flat epochs (2.3 and −1.3), kept by vetting
  there and at a third of the period by the 200-day search.
- **One deep event and two weaker ones.** The veto keeps such a fold when
  the two others alone reach MES 7.1. They are then two events, under the
  three required.

`carrying_events` reads each refined detection's events, one cadence per
transit as the search folds them. Where the (k + 1)-th strongest event is
below a fifth of the k-th, for k = 1 or 2, the fold rests on its k
strongest events. It is then kept only if its other events are a detection
by themselves: at least three of them at MES 7.1. A planet whose one
transit coincides with an artifact stays, and so does any fold with no
sharp drop among its three strongest events.

**Planets keep a wide margin.** Transits of one planet differ by noise,
coverage and whitening, so the rule's fraction is set well below where any
planet's events fall:

- **400 development planets** (calibration-sample revision 5, found exactly
  by the 200-day Detector). The nearest to the rule keeps 0.36 of its
  second event: a 13.6%-deep giant whose three transits whiten to 70, 44
  and 16.
- **120 injected planets at 200–500 days** (one per development lane host
  with 14 or more quarters; radii 2–14 R⊕, log-uniform periods). 36 are
  within reach; 32 are found at their own period and 31 kept, with the
  rule and without it. The nearest of the 33 found keeps 0.42 of its
  second event (SES 11.4, 11.2 and 4.7). On these curves the rule cuts the
  long-period candidates from 509 to 438 (to 347 before the masking
  below). The committed Detector finds the same 32 and keeps the same 31.

**Masking what a dropped fold rests on.** As first committed (323c08cda),
a pass whose refined folds all rested on one deep single event returned
nothing, and the search stopped there: every later pass would have found
the same event first. On the 0.5–200-day search Kepler-607 b (0.64 d) and
c (39.7 d) were lost behind such an event, though the 500-day search,
whose first folds differed, found both. Since 37f62ddb9 the carrying events
are masked like a detection's and the search goes on. A pass that made
only dropped folds counts as one that added nothing, and the repeat rule
allows one such pass in a row, as it does a pass of repeats.

**What it removes.** Applied to recorded events, it removes about a
quarter of the development set's unexplained detections that the veto
passes, and a third of the survey curves' long-period candidates. A fold
whose third event happens to be a 5σ outlier still passes. Vetting, and the
false-alarm calibration of the 200–500-day regime, deal with those.

## Template durations

The template bank is sized for the longest period with three transits in
the curve (its span over three, about 490 days), whatever the search
range, so the 200–500-day trials needed no new templates. On the 896
survey curves, 610 spans admit 200–500-day trials:

- **Dwarfs (density at least 0.1 solar), 554 curves.** The longest template
  covers the central-transit duration at every period to 500 days, and
  reaches at least 78% of the window top, 1.5 times that duration.
- **Evolved stars (density below 0.1 solar), 56 curves.** Only 10 are
  covered. Their 200–500-day transits last 1.5–4.5 days, beyond upstream's
  1.5-day template cap, and KBonus's 2-day-knot flattening erases them.

## Aliases

The Detector reports a detection at P/2, P/3, 2P or 3P when that alias
folds at least 5% more MES, also without its strongest event. At 500
days the long-period planets the 200-day search found at a harmonic
(Kepler-849 b at half its period, and six more) are found at their own.
In the plain extension no planet below 200 days moved to a harmonic: the
same 591 are found at their own period, and one at a harmonic in both
searches. On the committed Detector one does: Kepler-87 c (191.2 d),
whose transit-timing variations measured its mass (Ofir et al. 2014),
folds at twice its period (382.3 d) as well as at its own, and the alias
rule picks the longer one once it is in range (below). The one injected
planet within reach found at a harmonic (346.2 days, found at half) is so
with the rule and without it.

## The cost of 500 days

The committed Detector (37f62ddb9) at 200 and at 500 days, paired legs on
the same 896 diverse survey curves (`final/`; the legs before the masking,
on 323c08cda, are in `final-without-masking/`):

| | 0.5–200 d | 0.5–500 d | Ratio |
| --- | ---: | ---: | ---: |
| Detector CPU, s per curve (weighted) | 1.63 | 2.07 | 1.27 |
| Vetter CPU, s per curve (weighted) | 1.93 | 3.27 | 1.69 |
| Detector + Vetter CPU, all 896 curves | 3,788 s | 4,973 s | 1.31 |
| Detector wall / Vetter wall | | | 1.35 / 1.26 |
| Candidates (beyond 200 d) | 1,043 | 1,691 (971) | 1.62 |
| Kept candidates (beyond 200 d) | 107 | 125 (24) | 1.17 |
| Kept and explained by a KOI | 9 | 9 | |
| Curves with a kept candidate (weighted) | 8.8% | 10.6% | |
| Curves that reached the detection limit | 22 | 52 | 2.4 |

- **Runtime.** The 500-day search costs 31% more Detector-plus-Vetter CPU
  (50% weighted to the survey, whose faint curves carry more long-period
  candidates). The Vetter carries most of it: every extra candidate is
  vetted. Weighted to the survey, Detector plus Vetter take about 250
  CPU-hours at 200 days and 370 at 500 for the 250,923 lane curves. These
  legs shared the Mac with the SAP benchmark's six workers, which loads
  both legs alike but inflates CPU seconds; the quieter legs before the
  masking gave 224 and 290 CPU-hours.
- **What the masking costs.** Letting the search go on past a dropped fold
  raises the candidates by 8% at 200 days and 14% at 500 (by 23% beyond
  200 days), and lowers the kept ones from 118 to 107 at 200 days and from
  136 to 125 at 500; the nine KOI-explained kept candidates are the same in
  every run.
- **What the transit requirement did.** On the legs before the masking it
  removed 16% of the 200-day search's candidates (10% of its kept ones) and
  26% of the 500-day search's (37% of those beyond 200 days, 18% of kept
  ones).
- **Below 200 days.** Three detections kept at 200 days are dismissed at 500
  and eleven kept at 500 are new. One is a 40.0-day detection at MES 15.3
  on a crowded curve with nine to eleven detections: once a different
  sixth detection is masked before it, its vetting score falls 0.008 below
  the cut.
- **Confirmed planets** (493 hosts, the same legs). Of the 621 planets
  below 200 days with three transits, the 500-day search finds 592 at
  their own period (591 at 200 days) and keeps 577 (576). It finds
  Kepler-1928 b (19.6 d), which the 200-day search lost to its detection
  limit behind ten folds of 120–198 days, and keeps Kepler-607 b and
  Kepler-36 b, which vetting dismissed at 200 days. It loses two:
  Kepler-87 c (191.2 d) comes out at twice its period (382.3 d) once that
  is in range, and Kepler-1199 b (15.0 d) is found but dismissed once
  different detections are masked before it. Of
  the 10 planets at 200–500 days, 8 are found at their own period and 6
  kept, as before the masking. The masking itself, at 200 days, finds
  Kepler-607 b and c again (vetting dismisses both).
- **The detection limit binds on more curves.** The search stops after the
  pass in which it passes ten detections (upstream's limit). At 500 days 52
  of the 896 curves reach it (5.8%), against 22 at 200 days. On a rich
  synthetic system that cost one planet
  ([rich planetary systems](keplerX-2026-11-02-rich-planetary-systems.md)).
  On the confirmed-planet hosts it binds on 46 curves at 500 days against
  20 at 200, and which planet it costs is a lottery of which folds come
  first: Kepler-1928 b is lost at 200 days and found at 500 as the
  eleventh detection. Rerun with a limit of 15 (`detection_limit.py`,
  `detection-limit-15/`), the 98 curves that reach it at 500 days find
  the same 62 of their 68 confirmed planets and keep the same 60, for 27%
  more CPU on those curves (8% of the survey's, as they carry 29% of it)
  and six more kept candidates; the rich system's lost planet does come
  back. The limit stays at ten.
- **The long-period range is mostly false alarms.** 971 candidates beyond
  200 days, and 24 kept. A random survey curve has a detectable transiting
  planet there with a chance of order 10⁻⁴: giant planets at those periods
  orbit a few percent of Sun-like stars, and transit with a probability of
  about 0.5%. Recalibrating false alarms in the 200–500-day regime is the
  next calibration's task (TODO item 9).
