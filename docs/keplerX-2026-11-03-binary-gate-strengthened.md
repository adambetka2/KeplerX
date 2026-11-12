# The binary gate strengthened: the secondary event by event, and a deep primary (3 November 2026)

The task (Adam, 3 November): the binary gate removed 10.6% of the binaries
and 16.7% of the KOI false positives the vetting score keeps, for one
confirmed planet in 583 (Kepler-840 b, which DR25 calls a false positive);
bring it towards 30-40% without losing planets. Development data, the
survey benchmark's confirmed planets and a fresh development set fixed in
advance; the sealed holdout was not touched. Scripts and frames:
`KeplerX/results/calibration-audits/2026-11-03-binary-gate-strengthening/event-by-event/`
and `.../fresh-planets-3/`.

**Outcome: adopted.** The gate now removes 36.5% of the binaries and 38.2%
of the KOI false positives the score keeps in PSF Flat, and 46% and 44% in
SAP, with no planet newly dismissed anywhere; on 1,574 development planets
no run had used, with the rule fixed before any case ran, it dismissed none
of the 580 the score keeps.

| Kept by the score (PSF Flat, development) | Gate before | Gate now |
| --- | ---: | ---: |
| Synthetic binaries on the host (194) | 19 | 80 |
| Blended binaries (90) | 7 | 25 |
| Real binaries leaking into lane curves (130) | 19 | 46 |
| All binaries (414) | 45 (10.9%) | 151 (36.5%) |
| KOI false positives (207) | 33 (15.9%) | 79 (38.2%) |
| Development planets (1,299), and the earlier fresh sets' (201, 375) | 0, 0, 0 | 0, 0, 0 |
| Confirmed planets (583) | 1 (Kepler-840 b) | 1 (Kepler-840 b) |
| Unexplained candidates (563, all sets) | 8 | 25 |

In SAP (its own scores; the SAP twins of development-quick, of the gate's
check cases and of the first fresh set, and SAP on the confirmed planets'
hosts): synthetic binaries 21 -> 106 of 232 (46%), real leaking binaries
4 -> 17 of 59, KOI false positives 17 -> 49 of 112 (44%); no planet newly
dismissed of 761 development planets and 592 confirmed ones (the one SAP
development planet dismissed, by a 5-sigma odd/even, is dismissed by the
gate before too; Kepler-840 b as in PSF Flat). The unexplained candidates
newly dismissed are all 12-1,000 ppt deep with a secondary or an
ellipsoidal variation, several on the same host stars across cases: real
binaries in the hosts' own curves that the truth tables do not list.

## What was wrong with the secondary piece

The secondary is the strongest dip of the MES folded at the detection's
period with the transit and every other signal masked. Measured again in
the light curve's own events, each passage of that phase on its own
(`binary_evidence.transit_depths` at the secondary's epoch), the two kinds
of dip part cleanly:

- **Binaries' secondaries are in the events:** median 26 sigma over the
  events' own scatter (binaries) and 21 (KOI false positives); 135 of 148
  binaries and 60 of 62 KOI false positives with the piece hold 3 sigma over
  all and 2 sigma in each half of the baseline.
- **Planets' false secondaries are not:** of the 20 planets (development,
  both earlier fresh sets, confirmed) whose fold showed a secondary piece,
  16 sit below 2.1 sigma event by event, 12 of them carried mostly by one
  event (the strongest over half the summed depth, up to 98%), and one has
  a single event at its phase. The other three are two planets' own
  transits at half the period (found at twice it, depths equal within
  errors) and Kepler-840 b (17 sigma, a DR25 false positive).
- **A detection at k times the period** shows the planet's own transits at
  phases j/k of its fold. The piece recognised only k = 2, by MES; the
  second fresh test's loss was a 126-day giant found at 378.6 days, its own
  transit at phase 1/3 passing for a secondary because the one at 2/3 fell
  in a gap. It holds a single event at that phase.

## What changed

- **`vetter/binary_evidence.py`** measures the secondary event by event
  (`secondary_events`): `binary_secondary_event_sigma` (the mean depth at
  its phase over the events' scatter), `binary_secondary_halves_sigma` (the
  lesser of the same in either half of the baseline) and
  `binary_secondary_is_own_transit` (within two durations of phase j/k,
  k up to 4, and as deep as the transit within 3 sigma event by event).
  About 1 ms per candidate beside the evidence's 71 ms.
- **`vetter/policy.py` BINARY_GATE:** the secondary piece also needs 3 sigma
  event by event and 2 in each half, and not the planet's own transit; so
  guarded, its fold thresholds go from 7.1 and 3 to 5 and 2 (6 binaries and
  4 KOI false positives more, no planet). A fourth piece, a primary eclipse
  at least 30 ppt deep, or 10 ppt and grazing (fitted b at least 0.8),
  corroborates a secondary or an ellipsoidal variation, never odd/even
  alone, and never decides alone: planets reach it around small stars
  (Kepler-45 b, 37 ppt), and 168 of the 2,458 planets above show it.
  `binary_pieces` returns the pieces and whether they dismiss. A primary
  and a secondary are two measurements, so nothing is dismissed on a
  secondary alone (Adam's rule).
- **The golden test** (`tests/test_golden_binary_gate.py`) holds six real
  signals: Kepler-840 b dismissed (secondary, held event by event, and
  ellipsoidal variation), Kepler-158 c kept with no piece now (its fold
  secondary, a dip a day from its transits, is 1.1 sigma event by event),
  the weak synthetic planet kept, the blended binary dismissed by odd/even,
  the 126-day giant found at 378.6 days kept on its deep primary alone, and
  a deep, grazing blended binary dismissed by its primary and secondary.

## Margins, and what was not taken

- **Odd/even does not corroborate a deep primary.** Among the 168 planets
  with the deep piece the strongest odd/even reaches 2.84 sigma, against
  the corroborating 3; with that pairing the gate would catch 7 binaries and
  4 KOI false positives more (1.7 and 1.9 points), and a deep planet's
  chance odd/even would be one coincidence from a dismissal.
- **The secondary's own margin:** 39 planets show a fold secondary at the
  new thresholds; 4 hold event by event, 3 of them the planet's own
  transits, the fourth Kepler-840 b.
- **A long-duration piece** (a transit at least twice as long as a
  central transit on the star's catalogue density): paired like the deep
  primary it adds 6 binaries and 8 KOI false positives but dismisses a
  development planet; as a full piece, 20 and 17, for two development
  planets and one fresh one. Not taken.
- **An implied-radius piece** (the companion's radius from the depth and the
  star's radius, 1.8 or 2 Jupiter radii): in place of the depth piece it
  catches 99-102 binaries against 151, beside it at most one more. Not
  taken.
- **A secondary at half the orbit alone** remains not taken (Adam: never
  dismiss on a secondary eclipse alone).

## The fresh test, fixed in advance

The rule and the criterion were written into `fresh-planets-3/launch.sh`
before any case ran, with the working tree's Vetter saved as
`uncommitted.patch` beside the run's commit: the 1,574 development synthetic
planets in no run's results (every `results.sqlite` under `results/`),
PSF Flat with the production configuration on 6 workers; pass if no planet
the score keeps is dismissed by the gate, otherwise not adopted. All 1,574
cases completed without error in 32.5 minutes (`judge.py`, `verdict.json`):

- **Passed:** the score keeps 580 planets; the gate dismisses none (the gate
  before: none). 75 carry a lone deep primary and 2 a lone ellipsoidal
  variation, recorded on the kept signal; 4 of 174 unexplained candidates
  are dismissed (before: 0).
- **The measurement ran:** the event-by-event values are in the run's own
  metrics for 2,911 of its 3,137 candidates (the rest have no secondary or
  fewer than two events at its phase).
- **Margins on fresh data:** the 50 deep planets with an odd/even reach at
  most 2.13 sigma. Of 24 planets with a fold secondary at the new
  thresholds and an albedo above 1, 5 hold event by event: 3 are the
  planet's own transits at j/k (recognised), and 2 are dips 2.2 and 3.0
  days before the transits of planets 44 and 7 ppt deep, which hold 4.9 and
  9.3 sigma event by event. These are what KBonus's 2-day spline leaves
  beside a deep transit; the fold's uniqueness test stops them (-23 and -17
  against +2: the spline leaves a stronger bump beside the dip). For the
  44-ppt planet, which has the deep piece, that test is what keeps it, so
  the uniqueness threshold carries the gate's margin on deep transits.

## Where the numbers come from

- `event-by-event/binary_recompute.py` rebuilds each stored candidate's
  Dump and FastVetter from its prepared packet and the run's own
  detections and recomputes the binary evidence; with the code before the
  change it reproduced all 840 stored values of 60 sampled candidates, and
  the stored binary metrics of all 2,174 SAP candidates, exactly.
- `gate_explore.py`, `combos.py` (the piece combinations on the stored
  frames), `variants.py` (the event-by-event diagnostics), `gate_judge.py`
  and `compare_gates.py` (the gate before, `policy-before.py`, against the
  gate now), `sap_frame.py` (SAP's score-kept candidates), `build_fixture.py`
  (the golden test's metrics).
