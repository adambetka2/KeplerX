# Seven-to-nine-planet systems through KeplerX (2 November 2026)

Can KeplerX recover rich planetary systems, a TRAPPIST-1-like seven-planet
chain among them, and when it cannot, is that the data or the software?
Script, packets and tables:
`KeplerX/results/calibration-audits/2026-11-02-rich-planetary-systems/`
(`simulate.py`, `period-500/`, `period-500-trappist-architecture/`).

## Design

Five systems after real architectures, each injected into four development
lane hosts of its class from calibration-sample revision 5. The hosts have
at least 14 quarters and sit at the class's 1st, 10th, 30th and 50th
percentile of the noise a transit on the star meets: the curve's 3-hour
noise over the star's share of its flux. Ranking by the curve's own noise
picked curves that are mostly a neighbour's light.

| System | Host class | Planets | Periods (d) | Radii (R⊕) | TTVs |
| --- | --- | ---: | --- | --- | --- |
| TRAPPIST-1-like (Agol et al. 2021) | M | 7 | 1.51–18.8 | 0.76–1.13 | 2–40 min |
| TRAPPIST-1 architecture, radii × 2.5 | M (same hosts) | 7 | 1.51–18.8 | 1.9–2.8 | 2–40 min |
| Kepler-90-like | G | 8 | 7.0–331.6 | 1.2–11.3 | 5 min–6 h |
| Compact chain of nine (ratios 1.38) | K | 9 | 1.8–23.7 | 1.0–3.0 | 3–30 min |
| Kepler-80-like and one outer planet | K | 7 | 0.99–27.1 | 1.2–2.7 | 1–35 min |

Planets are injected with the calibration builder's own `inject_system`.
That covers the star's share of the flux, KBonus's flattening emulated, and
exposure integration. Each planet is circular, with the system's impact
parameters and a sinusoidal TTV on the super-period of its nearest
near-first-order pair. Overlapping transits happen where they fall: a
median of 642 cadences per curve have two or more planets in transit, and
up to 3,017.

Detection and vetting are production's: the committed Detector with the
0.5–500-day search, its transit requirement and the masking of what a
dropped fold rests on (commit 37f62ddb9,
[period search](keplerX-2026-11-02-period-search-to-500-days.md); run
`committed-detector/`). The run before that masking
(`transit-requirement-without-masking/`) and the first runs, before the
transit requirement (`period-500/`, `period-500-trappist-architecture/`),
differ in a few planets, below. Each planet is classed as follows:

- **Outside the observing window:** fewer observed transits than the
  Detector requires, or a period beyond the range.
- **Below the data's sensitivity:** information S/N under 7.1, the ruler
  of the calibration.
- **Within reach:** everything else.

A system is recovered when every planet within reach is found exactly.
The same systems, on the same epochs, were also injected into each host's
SAP lane as its calibration twin is built (`sap-lane/`; below); the
unflattened PSF and combined lanes were not built
([SAP and PSF lanes](keplerX-2026-11-03-sap-and-psf-lanes.md)).

## Results

| System | Planets | Below sensitivity | Within reach | Found exactly | Kept by vetting |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRAPPIST-1-like | 28 | 26 | 2 | 2 | 2 |
| TRAPPIST-1 architecture × 2.5 | 28 | 12 | 16 | 16 | 16 |
| Kepler-90-like | 32 | 10 | 22 | 20 | 20 |
| Compact nine | 36 | 20 | 16 | 11 | 11 |
| Kepler-80-like | 28 | 18 | 10 | 9 | 9 |
| All | 152 | 86 | 66 | 58 (88%) | 58 |

**Recovery matches isolated planets.** In the first four systems, 42 of
50 planets within reach were found exactly. The development-quick
efficiency curve of single and few-planet systems predicts 43.5:

| Information S/N | Planets | Found | Expected |
| --- | ---: | ---: | ---: |
| 7.1–10 | 17 | 9 | 11.1 |
| 10–15 | 8 | 8 | 7.8 |
| above 15 | 25 | 25 | 24.6 |

Within this sample, siblings, their masks, overlapping transits and TTVs
of up to six hours cost no measurable recall; the one planet the search's
detection limit cost is below. Every planet within reach that was missed
has information S/N 7.35–9.36.

- **Whole systems.** Ten of the 16 systems with planets within reach
  were recovered completely; the other six each lost only planets below
  S/N 9.4. Every planet found exactly was kept by the production policy
  and by the binary gate
  ([binary rejection](keplerX-2026-11-02-binary-rejection.md)), so the
  gate's masking of other signals protects multi-planet systems.
- **The TRAPPIST-1 architecture.** At 2.5 times its radii it was
  recovered completely on all four hosts: every planet within reach,
  through the resonant chain, TTVs and overlaps. On the best host all
  seven planets were found.
- **TRAPPIST-1 itself is beyond KBonus's M dwarfs.** These are early M
  dwarfs of G 16–19, not M8 stars of 0.12 R☉. Earth-sized planets are
  within reach only on the quietest 1% of hosts, and there only the two
  innermost. On the 10th to 50th percentile hosts all seven are below the
  data's sensitivity. That limit is the data's, not the software's.
- **Long periods.** Kepler-90-like g (210.6 d) and h (331.6 d) were found
  and kept on all four G hosts; at the 200-day limit h is out of reach.
- **The detection limit binds on the richest curves.** The search stops
  after the pass in which it passes ten detections (upstream's limit; 11
  to 13 by then), and 7 of the 20 curves reached it. On the quietest
  Kepler-90-like host six planets came first (MES 13–97), then five
  folds of 308–481 days at MES 8.6–9.6, and planet b (7.0 d, S/N 9.4) found
  no place; vetting dismissed all five folds. Before the masking of what a
  dropped fold rests on, b was the eleventh detection (MES 7.5); the
  0.5–200-day search does not make such folds. On the 493 confirmed-planet
  hosts of the survey benchmark the limit binds on 46 curves at 500 days
  and 20 at 200, and costs confirmed planets at either range: Kepler-1928 b
  is lost at 200 days and found at 500
  ([period search](keplerX-2026-11-02-period-search-to-500-days.md)).
  With a limit of 15 (`detection-limit-15/`) b is found and kept, but on
  the real curves that reach the limit a higher one finds no further
  confirmed planet for 8% more survey compute, so the limit stays.
- **Kepler-80-like c, once a software miss.** It has S/N 21.9 and 3.07 d,
  near 3:2 with planet d's 4.64 d, with a 195-day TTV cycle. Before the
  transit requirement a spurious 351-day fold (MES 20.5) was detected
  ahead of it, and c came out only at twice its period (6.144 d, MES
  12.7). The requirement drops that fold, and c is found at its own
  period (MES 16.9) and kept. One TRAPPIST-1-architecture planet below
  the sensitivity ruler (S/N 6.85) comes and goes between runs by chance.
- **Spurious long-period candidates.** The extended search finds 200–500-day
  candidates in rich and noisy curves: 30 in the 16 curves of the first
  four systems (37 before the transit requirement, 23 with it before the
  masking, which lets the search go on), with three to seven transits,
  mostly two to four of them good. Coincident dips line up there. Vetting
  dismissed every one.

## The SAP lane

The 20 hosts' stitched SAP, prepared as the `sap_flat` view, carries the
same systems (`simulate.py --lane sap`). Each planet is classed by its own
SAP information S/N, which is higher than in PSF Flat on the compact-nine
hosts (1.0–1.5 times) and on two of the four Kepler-90-like hosts.

| | PSF Flat | SAP | Either lane |
| --- | ---: | ---: | ---: |
| Planets within reach | 66 | 75 | 75 |
| Found exactly | 58 | 59 | |
| Found exactly and kept | 58 | 48 | 64 |
| Complete systems | 10 of 16 | 7 of 17 | |

- **SAP adds five planets** that PSF Flat misses: Kepler-90-like b on the
  quietest host (lost there to the detection limit), two compact-nine
  planets whose SAP S/N is 1.5 times PSF Flat's, a Kepler-90-like planet
  near threshold in both lanes, and a Kepler-80-like planet at S/N 5.0 in
  PSF Flat and 8.7 in SAP.
- **SAP loses sixteen** that PSF Flat keeps. Six it does not find (SAP
  S/N 7.1–13.5). Ten it finds and vetting dismisses, all on three
  Kepler-90-like hosts, from S/N 11 to 188. On these curves SAP carries
  strong spurious dips: one host's first two SAP detections are folds at
  MES 49 and 45 where no planet is. The vetting score weighs the strongest
  other dip against the detection's MES, and the curve's noise, so real
  planets on a curve with such dips fall below the cut. The score was
  fitted on PSF Flat; SAP, which KBonus did not correct for spacecraft
  systematics, needs its own.
- **SAP searches longer.** 13 of the 20 SAP curves reach the detection
  limit (7 in PSF Flat), with up to 24 candidates on one curve.

## Readiness

KeplerX can recover rich planetary systems in KBonus wherever the data
carry them. Its efficiency per planet is that of isolated planets, whole
systems come back when every member is above about S/N 10, and the
architecture of TRAPPIST-1 is handled end to end. What limits it is the
KBonus data: Earth-sized planets of early M dwarfs sit below sensitivity
on almost every host. The test covers 20 curves and 152 planets: enough to
rule out large multiplicity losses, not to measure small ones. As a second
lane SAP adds planets where its S/N is higher, but only once its own
systematics and vetting are calibrated; the test should be repeated then.
