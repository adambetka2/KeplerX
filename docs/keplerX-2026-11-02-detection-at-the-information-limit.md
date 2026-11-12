# KeplerX Detector: detection at the information limit

2 November 2026. Changes: `KeplerX/detector/candidate_ephemerides.py`
and `flattened_templates.py` (new), `compiled_search.py`, `gap_padding.py`,
`repeat_rule.py`, `upstream_adapter.py`, `config.json` (detection), tests. Receipts (scripts and
tables, local only) are in
`KeplerX/results/performance-engineering/2026-11-02-detection-at-the-information-limit/`.

## Why

The 1 November development-quick run (1,750 cases, revision 3) put the
Detector's efficiency at 50% for information S/N 8.76 and 90% at 12.37; the
targets are 8.5 and 12. Detector MES over information S/N was 0.81 (0.70
below 1 day), and one process saturates the GPU.

## Where the S/N goes

135 detectable synthetic planets with information S/N below 13 from that run.
Each case's injected signal was regenerated through the sample builder (its
injected flux matched the delivered case to 1e-9) and passed through the
Detector's own per-segment whitening (`scripts/snr_budget.py`).

| Quantity | Value |
| --- | --- |
| Noise-free MES at the true ephemeris / information S/N | median 0.92 (IQR 0.83–0.96); 0.71 for transits over 0.4 d; 0.81 at 1–3 d |
| Gain from templates at the transit's own sub-cadence phase | mean +2.3% (+5.3% under 1.2 h) |
| Cadence-bin fold at the true period / nearest-cadence sum | 1.00 |
| Self-whitening (weights with the transits present / host's own weights) | 0.78–0.89 for half the planets under 1 d; none above about 1.5 d |

A sub-cadence (polyphase) fold would double the short-template fold for
2–5%; it was not built. Self-whitening is real at short periods (a transit
train inflates the MAD of wavelet levels 2–4 by 1.6–2.3 times on short
curves) and remains open.

## Where detectable planets are lost

The 54 detectable synthetic misses of that run, traced through every pass
(`scripts/trace_search.py`):

- 39 never reach MES 7.1 within 0.5% of the true period;
- 5 do and are dropped by the outlier veto;
- 10 do, pass the veto, and are erased afterwards: by upstream's
  `clean_tces_of_harmonics` (a stronger detection within 2% or within 1% of
  any of 36 ratios, epochs ignored; a P = 8.69 d planet at MES 15.5 was
  removed by its own 2P alias, a P = 110.5 d planet at MES 21.6 by an
  artifact) or by `mask_highest_TCE_and_check_others` (85% of the MES with
  stronger detections masked, which a strong short-period signal's mask
  covering 30–40% of the cadences denies any planet; and a bin edge within
  half a duration of the epoch, one cadence for the shortest template);
- right-period candidates fail the scorer's epoch tolerance because the
  reported period is a trial period (13.9816 against 13.9822 d drifts the
  last transit 0.044 d against a 0.040 d tolerance).

## The MES bias

Quarters are normalised by their median and segments are whitened as
flux − 1. A segment's mean is then offset from zero; the coarsest wavelet
level carries it and the transit template's own mean projects onto it, so
every trial carries −offset × (template sum) / σ per transit. On 24 null
development curves, 300 random ephemerides each (`scripts/mes_bias.py`):

| Whitening | MES mean (2-cadence template) | over 200 transits | under 50 | MES spread |
| --- | ---: | ---: | ---: | ---: |
| Upstream | −0.29 | −0.62 | −0.14 | 1.05 |
| Coarsest level left out | −0.08 | −0.20 | −0.04 | 1.04 |
| About the segment mean | 0.00 | −0.02 | +0.01 | 1.03 |

With a 5-cadence template the upstream mean is −0.48 (−0.96 over 200
transits). Capping single-event statistics before the fold does not change the
spread (1.05 to 1.02 at a ±4 cap); the noise core is close to unit variance
and the bias was the systematic part.

## What changed

- `detection.segment_baseline = "mean"`: each segment is whitened about its
  own mean (both engines; it also reaches the Vetter's MES metrics, which
  whiten through the vetter-stage Dump).
- `detection.candidate_cleaning = "epoch"`: greedy suppression in descending
  MES by shared events only (same period within 2% and the same
  mid-baseline epoch, or an integer multiple or divisor whose ratio holds
  over the baseline with coinciding epochs); refinement of each surviving
  detection's period (±1 trial step), mid-transit epoch (±1.5 cadences,
  quarter-cadence steps) and duration on the nearest-cadence matched filter;
  then the check of weaker detections with stronger ones masked: an exact
  alias (ratio p/q, q ≤ 3, within 2e-4, epochs on the p/q spacing) is
  rejected, otherwise the detection stays if its depth on unmasked transits
  holds 85% of its depth on all of them and its MES there passes the
  threshold. Rejected along the way: preferring a fundamental over its
  stronger integer alias (four planets lost where the fundamental's extra
  events fall in gaps), a threshold on the unmasked depth scaled to all
  transits (more candidates), suppression by a shared strongest event (an
  artifact dip on a planet's transit made the planet its duplicate),
  upstream's fractional ratios (17/3 at 1% removed a known 7.38-d planet
  beside a 42.19-d artifact).
- `detection.period_step_factor = 2`, `detection.screen_threshold = 6.6`:
  half the trial periods, screened at 6.6; the refinement recovers the
  grid's loss and MES 7.1 applies to the refined value.

## Paired evaluation

348 development-quick cases: the 54 cases with detectable synthetic misses,
68 with near-threshold planets found, 150 nulls (100 originals, 30 inverted,
20 scrambled), 40 known-planet cases and 36 artifact copies
(`tables/eval_sets.json`). `scripts/eval_detect.py` runs the production
Detector and scores every truth signal with the scorer's `match_signal`; under
the previous settings it reproduces the 1 November run exactly (every
outcome, all 512 candidates). Four processes share the GPU.

| Settings | Detectable exact (148) | Gained / lost | Known (58) | Nulls with a candidate | Candidates per null | All candidates | GPU s / case | Wall s / case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous (1 November) | 94 | – | 57 | 28.0% | 0.89 | 512 | 2.59 | 7.46 |
| Segment mean only | 100 | +7 / −1 | 57 | 28.7% | 0.81 | 534 | 2.56 | 7.26 |
| Epoch cleaning only | 97 | +5 / −2 | 57 | 28.0% | 1.01 | 618 | 2.21 | 6.18 |
| Epoch cleaning, step 2, screen 6.6 | 100 | +10 / −4 | 57 | 28.7% | 1.14 | 625 | 1.00 | 4.49 |
| Production (all three) | 107 | +16 / −3 | 57 | 26.7% | 0.97 | 608 | 1.01 | 4.49 |

The isolating rows ran before the within-pass repeat flags (below). By
information S/N production finds 16/44 below 8.5 (was 13), 58/68 at 8.5–12
(49) and 33/36 above 12 (32). The three losses: two planets on curves where
12–16 long-period artifact detections (MES 20–54) outrank them and the
ten-detection limit ends the search, and one planet found before at MES 7.4.
Summed over the 348 cases the Detector took 1,632 s of wall time instead of
2,598 (95th percentile 11 s against 20 s per case, worst 23 s against 47 s)
and 345 GPU-s instead of 901.

The 348 cases are enriched in former misses. 300 further cases (200 holding
planets found on 1 November, 100 uninjected originals; `tables/eval_sets2.json`)
check losses without that selection: of their 193 detectable signals no
planet was lost; four contaminating binaries (S/N 62–636) went from exact to
a harmonic; originals with a candidate 20% against 19%. Candidates rose 37%,
mostly on binaries (contamination 236 against 141, synthetic binaries 59
against 34, planets' curves 204 against 177): the first pass now returns a
binary's secondary eclipse and the wings of strong eclipses on their own
epochs. Upstream's pass masks such detections and its repeat rule keeps them
out of later candidates; the same now holds within a pass (a detection at a
stronger one's period, or an integer multiple or divisor of it within the
repeat tolerance, is masked but not a new candidate), which brought the 57
contamination cases from 236 to 215 candidates with every outcome unchanged.
The rest, aliases of strong signals' residual wings on other epochs, stays
open (`KeplerX/TODO.md`).

## Development-quick run

The whole development-quick manifest of calibration-sample revision 4 (1,752
cases) ran through detection and vetting on 8 workers at commit `5de0587eb`
and was scored (`KeplerX/results/calibration-previews/2026-11-02-development-quick-run-new-detector`),
paired with the 1 November run (revision 3, 1,750 cases; revision 4 differs
only in its blends).

| | 1 November | 2 November |
| --- | ---: | ---: |
| 50% efficiency at information S/N (target ≤ 8.5) | 8.76 [8.34, 9.23] | **8.13** [7.75, 8.60] |
| 90% efficiency at information S/N (target ≤ 12) | 12.37 [11.22, 13.23] | **11.28** [10.26, …] |
| Efficiency at S/N 8.5 / 12 | 45.7% / 87.9% | 56.8% / 93.5% |
| Detectable planets exact (308), plain / weighted | 82.5% / 81.5% | 86.7% / 89.4% |
| Occurrence-integrated completeness | 81.0% [75.1, 86.7] | 87.3% [82.4, 91.9] |
| Exact at S/N 7.1–8.5 / 8.5–10 / 10–12 / 12–15 | 27.9 / 60.6 / 81.8 / 95.2% | 34.9 / 81.8 / 87.9 / 97.6% |
| Known planets in range / detectable on KBonus depth | 88.9% / 93.0% | 92.2% / 96.5% |
| Validated BOIs | 7 / 7 | 7 / 7 |
| Detector MES / information S/N (median) | 0.81 | 0.84 |
| Null originals with a candidate / candidates per case | 21.0% / 0.72 | 21.4% / 0.95 |
| Inverted / scrambled nulls with a candidate | 54.2% / 30.3% | 50.0% / 30.3% |
| Batch on 8 workers | 62.1 min | 39.0 min |

Both efficiency targets are met. On the cases' stage records detection wall
time fell to 0.47 of 1 November's and vetting's to 0.81 (more candidates,
offset by the Vetter's own speedups that day), while CPU per case stayed at
0.95 and 0.98: the run is now CPU-bound, not GPU-bound. Every period bin
improved except 50–100 days (one planet fewer of 29). The extra candidates
sit on curves that already had some (null originals with a candidate 21.0%
against 21.4%); blended binaries are found less often at their own period
(found 40% → 28%), plain binaries more often (69% → 76%).

## Aliases as repeats

The development-quick run carried 3,269 candidates against 2,425 on
1 November; 816 of the 844 extra were unexplained, and 50 cases held 461 of
them: binaries' quarter-phase aliases (a 0.864-d binary adding 1.08, 1.512,
2.376 d … = 5/4, 7/4, 11/4 … of its period, exact to 1e-5), residual aliases
of strong signals in later passes, and near-threshold forests on noisy
curves. `detector/repeat_rule.alias` now marks, against this pass's stronger
detections and every earlier pass's, a detection as a repeat (masked, not a
new candidate): the same period or an integer multiple (2–10) within the
repeat tolerance at any epoch; an integer divisor off the known one's events,
or on them but weaker in the same pass (a divisor found in a later pass on
them may be the fundamental); a fraction p/q (q ≤ 4, ratio ≤ 10) exact to
5e-5 with epochs on the spacing of the shorter period / q. Ratios above 10
are left out: a planet at 8.6873 d sat within 3e-5 of 67/4 of a 0.5187-d
variable, where the alias spacing (0.13 d) is shorter than the durations.

| Paired cases | Detectable exact | Known | Nulls with a candidate | Candidates per null | All candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| 348 (1 November / before / after) | 94 / 107 / 107 | 57 / 57 / 57 | 28.0 / 26.7 / 26.7% | 0.89 / 0.97 / 0.89 | 512 / 608 / 578 |
| 300 unbiased (1 November / before / after) | 191 / 187 / 186 of 193 | – | 19 / 20 / 20% | 0.92 / 1.02 / 0.92 | 517 / 707 / 610 |

The 60 cases with the most extra candidates in the run hold 525 instead of
718 (190 on 1 November). No planet changed outcome; on the unbiased set five
contaminating binaries (S/N 62–636) are reported at a harmonic instead of
their own period.

## Final state

The development-quick manifest again, on the combined state of the day
(`2b25ce598`: the new Detector, aliases as repeats, the compiled coarse period
grid, the Vetter's speedups and the parallel fused veto and bucketed
cleaning), in `KeplerX/results/calibration-previews/2026-11-02-development-quick-run-final-state`:

| | 1 November | 2 November, first run | 2 November, final |
| --- | ---: | ---: | ---: |
| 50% / 90% efficiency at information S/N | 8.76 / 12.37 | 8.13 / 11.28 | 8.13 / 11.28 |
| Detectable planets exact (plain / weighted) | 82.5 / 81.5% | 86.7 / 89.4% | 86.7 / 89.4% |
| Occurrence-integrated completeness | 81.0% | 87.3% | 87.3% |
| Known planets in range | 88.9% | 92.2% | 92.2% |
| Null originals with a candidate / candidates per case | 21.0% / 0.72 | 21.4% / 0.95 | 21.4% / 0.83 |
| Candidates (unexplained) | 2,425 (1,845) | 3,269 (2,661) | 2,908 (2,314) |
| Batch on 8 workers | 62.1 min | 39.0 min | 37.6 min |
| Detection wall / CPU per case | 12.26 / 4.33 s | 6.49 / 4.11 s | 6.59 / 3.71 s |
| Vetting wall / CPU per case | 4.68 / 3.52 s | 4.15 / 3.53 s | 3.67 / 2.97 s |

Workers now spend 10.3 s per case for 6.7 CPU-s: about 5 of the 10 cores
are busy on average and detection waits about 2.9 s per case, mostly in the
queue for the shared GPU claim. More workers than 8 may therefore pay off
now (the calibration study defaults to 10); untested.

## Templates through KBonus's flattening

After the final state, transits of 6 hours and longer were the largest
systematic loss. KBonus flattened each quarter with an unclipped
least-squares cubic B-spline with 2-day knots and did not mask transits
(`calibration/signal_injection.py` reproduces the surviving depths of 613
confirmed planets within 0.3%). A delivered transit is therefore the true one
minus the smoother's response to it: shallower, with shoulders on both sides
(the knot-averaged response is up to 49% of the depth for a 1.5-day transit
and still 1% six days away). On 21 detectable synthetic planets with transits
over 0.3 days, the exact injected signal used as its own template reaches
1.29 times the information S/N under the Detector's whitening. Upstream's
bare templates reach only 0.60 of that (0.17 for a 1.7-day transit), against
0.90 below 0.1 days.

`detection.template_flattening = 2.0` (`KeplerX/detector/flattened_templates.py`)
subtracts from every template of 6 hours or longer the least-squares fit of
that smoother, averaged over 16 knot phases and fitted within three knot
spacings each side: the expected delivered shape of an isolated transit.
Noise-free, it lifts those 21 planets' MES by a median 38% (to 0.85 of the
optimum; 2.6 times for the 1.7-day transit). Shorter templates stay bare: a
short transit on a short period is not isolated, because the spline absorbs
the train's mean level rather than each transit's shoulders, and the
flattened shape lost up to 11% at 1.9 days with 2.2-hour transits.

The whitening's noise model does not hold for the new shape. On 20 null
development curves the single-event statistics of flattened templates spread
1.06 (0.25–0.4 d), 1.09 (0.4–0.7 d) and 1.21 (over 0.7 d), against 0.89, 0.83
and 0.73 for bare ones, and null false alarms doubled. Each flattened row's
numerator and statistic are therefore divided by the robust spread (1.4826
MAD) of its statistic over the pass's cadences wherever it exceeds 1
(`gap_padding.Segments._unit_spread`). Nothing is boosted, so bare rows keep
their conservative scale. Both engines and the Vetter's whitening, which
builds its Dump from the same detection settings, use the flattened
templates. The compiled search stays native's bit for bit, and the GPU
screen stays the CPU's (`tests/test_golden_detector.py`, `test_golden_vetter.py`).

Evaluated with `scripts/eval_detect.py` on the 348 paired cases and on all
114 development-quick cases holding a detectable signal with transits of 6
hours or longer (126 detectable signals; production outcomes from the final
state's run):

| Templates | Detectable planets exact (148) | Known (58) | Nulls with a candidate | Candidates per null | All candidates (348) | Long-transit signals exact (126) | Wall s / case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bare (final state) | 107 | 57 | 26.7% | 0.89 | 578 | 95 | 4.55 |
| Flattened | 112 | 58 | 44.7% | 1.94 | 1,027 | 102 | 5.12 |
| Flattened, unit spread (production) | 114 (+7 / −0) | 58 | 33.3% | 1.17 | 701 | 104 | 5.09 |
| Bare, unit spread both ways (control) | 110 (+4 / −1) | 57 | 30.0% | 1.13 | 701 | 99 | 4.34 |

The control normalises bare long templates to unit spread as well, boosting
them where it is below 1. It isolates how much of the gain comes from the
calibration (an effective threshold of 7.1 instead of 8–10 robust sigma for
long bare templates) and how much from the shape. At the same 701 candidates
the flattened shape finds four more detectable planets and five more
long-transit signals, among them two known planets that neither bare variant
finds: Kepler-313 c (32.3 days, 0.36-day transits, information S/N 9.6) and
Kepler-443 b (177.7 days, 0.39-day transits, S/N 8.7, a habitable-zone
planet of a K dwarf).

On the unbiased 300 cases (`tables/eval_sets2.json`; their planets were all
found on 1 November, so nothing can be gained) 183 of 193 detectable signals
stay exact against 186. The three changed are high-S/N signals with long
transits (S/N 17.5–23.9), now reported at twice their period (MES 10.3–18.7
there). On one of them, a 46-day planet beside a 27-day one, flattening makes
the fundamental a first-pass winner (screen MES 12.3), but two of its
transits fall within 0.2 and 0.9 days of the stronger planet's transits, not
yet masked; the outlier veto removes it, while the double period, folding
only the other transits, is accepted, and in the second pass the
fundamental is that detection's alias. With bare templates it emerged only
in the second pass, after the mask. Originals with a candidate rise from 20% to 25% (0.97 against
0.92 candidates each), candidates from 610 to 652, and wall time is unchanged
(1,700 s against 1,693). The extra null candidates on the paired set are
long templates at 30–250-day periods (12 to 33 on null originals), half of
them on two messy curves.

The flattened templates' shoulders lengthen the direct convolutions at the
fine wavelet levels: whitening CPU rose 5–27% on three null curves, and
Detector wall time 12% on the paired set (5.09 against 4.55 s per case).
The vetting policies were fitted on bare-template MES metrics and need a
refit on a run with this setting.

The development-quick manifest on `42f0b594b`
(`KeplerX/results/calibration-previews/2026-11-02-development-quick-run-flattened-templates`),
against the final state:

| | Final state (bare) | Flattened templates |
| --- | ---: | ---: |
| 50% / 90% efficiency at information S/N | 8.13 / 11.28 | 7.66 / 10.41 |
| Detectable planets exact (plain / weighted) | 86.7 / 89.4% | 88.6 / 89.5% |
| Occurrence-integrated completeness | 87.3% | 88.4% |
| Known planets in range | 92.2% | 93.3% |
| Detectable planets exact by transit duration: under 0.1 d / 0.1–0.25 / 0.25–0.4 / over 0.4 d | 165/181, 131/142, 31/37, 30/41 | 165/181, 130/142, 34/37, 35/41 |
| Null originals with a candidate / candidates per case | 21.4% / 0.83 | 26.9% / 0.97 |
| Candidates (unexplained) | 2,908 (2,314) | 3,285 (2,673) |
| Batch on 8 workers | 37.6 min | 41.2 min |
| Detection wall / CPU per case | 6.59 / 3.71 s | 7.22 / 4.33 s |
| Vetting wall / CPU per case | 3.67 / 2.97 s | 4.00 / 3.27 s |

Seventeen detectable signals changed outcome: thirteen found that were
missed (Kepler-313 c, Kepler-443 b, a contaminating binary at 43.2 days and
ten synthetic planets with transits of 0.38–1.70 days), three reported at
twice their period (above), and two lost where long-period detections with
long templates (MES 8–54) now fill the search before a weaker signal: a known
planet at 2.62 days (MES 11.5 before) and an artifact copy's 17.5-day signal
behind fourteen artifact detections.
