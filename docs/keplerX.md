# KeplerX (third campaign)

The V3 campaign runs on KeplerX: Detector searches, Vetter vets and Localizator localizes. Each
component keeps a pinned, unmodified upstream copy in its `upstream/` folder
("native"), our adapters around it and our faster engines. The campaign also
prepares, runs and scores detection experiments.

## Components

| Folder | Purpose |
| --- | --- |
| `detector/` | Detector, the transit search: native upstream copy, a prepared-array adapter and its bitwise-equivalent compiled engine |
| `vetter/` | Vetter, the vetting metrics: its own upstream copy, compiled engine and numerical error recording |
| `localizator/` | Localizator, the pixel localization: native upstream copy, faster mathematics, Gaia DR3 from the Kepler field table (or the ESA archive), parent pixels from disc or streamed from MAST into memory, and a native-pixel adapter |
| `photometry/` | Read source-specific LC2 or native arrays; normalize PSF Flat, or flatten the stitched SAP as KBonus flattened PSF |
| `calibration/` | KBonus-realistic sample builder, injections, realism report and run scorer |
| `runner/` | Stage execution, bounded batch workers, logs and timing |
| `tests/` | Known-planet, failed-vetting, photometry, search, vetting and scorer tests |

**Provenance.** Detector and Vetter start from Robert F. Wilson's TRASHDUMP and RECYCLEBin
(github.com/robertfwilson/trashdump, commit
`5ce20ac8c82cb381db37783de647d62f3575986e`; the repository has no licence
file). Localizator starts from INCInERATOr by Julie Moquin, Robert F. Wilson, Jorge
Martínez-Palomera and Christina Hedges (github.com/juliemoqs/incinerator,
commit `6cbc23967e54986dabffc0e7935d3bfb91fae2da`, PyPI 1.1.2, MIT licence in
`localizator/upstream/LICENSE`). Outside this note the original names remain only in dated records. The
vendored copies carry recorded edits (`upstream.json`, `recorded_edits`, each
with the original file name and hash): file, class and report names follow
KeplerX, and Localizator fixes its version string, so no pip distribution is needed. `upstream.json` pins every copied file;
`doctor` verifies it against its original hash. No scientific monkeypatch is
applied.

## Execution

From the repository root:

```bash
bash KeplerX/run.sh setup
bash KeplerX/run.sh doctor
bash KeplerX/run.sh check
bash KeplerX/run.sh prepare REQUEST.json OUTPUT
bash KeplerX/run.sh detect PREPARED OUTPUT
bash KeplerX/run.sh vet PREPARED DETECTIONS OUTPUT
bash KeplerX/run.sh localize PREPARED DETECTIONS VETTING REQUEST.json OUTPUT
bash KeplerX/run.sh batch MANIFEST.jsonl OUTPUT --workers 1
```

Each command requires a fresh output directory. Outputs retain input receipts,
configuration, source hashes, logs and timings. Batch manifests explicitly name
each prepared input or preparation request; the runner does not select stars.

Runtime and caches default to `KeplerX/runtime/` on the internal Mac disk.
`run.sh` uses the locked environment at `KeplerX/runtime/pipeline-environment/`.

## Preparation

`prepare` keeps finite cadences with positive errors whose quality and
flatten flags are clear, and normalizes each quarter by its median. Since
29 October 2026 it also:

- retains cadences flagged only by bits 16, 128, 8192, 131072 and 262144
  (`photometry.retained_quality_bits`). On 800 lane curves these scatter like
  clean cadences (noise ratio median 1.05) and add 18% usable cadences;
  the October 13 SAP audit found the same five bits;
- drops a retained cadence that stands alone more than 5 sigma (clean
  point-to-point noise) from the median of itself and its two neighbours
  (`retained_outlier_sigma`, exclusion bit 32): about 10 per curve, while
  dips of two or more cadences are kept;
- masks the field-wide artifact epochs (`field_artifact_epochs`, exclusion
  bit 64): 70 windows, 17.5 days, where 10–31% of lane sources dip at once
  (`photometry/field-artifact-epochs.json`, from the 20,000-curve stack).

Net, usable cadences rise 16.9% on 300 lane curves; on the 20,000-curve
census every curve now has at least 435 usable cadences (191 had fewer than
100 under quality 0 alone). The calibration sample is built under these
settings.

**The SAP lane (3 November 2026).** The `sap_flat` view
(`photometry/program.py`) prepares the stitched SAP the same way, then
flattens each quarter as KBonus flattened PSF: an unclipped cubic B-spline
with 2-day knots (`photometry/flattening.py`), the emulator that reproduces
KBonus's surviving transit depth within 0.3% on 613 confirmed planets. A
quarter whose fitted trend reaches zero is excluded (bit 16). The Detector,
its flattened templates, the Vetter and the calibration builder's injection
then treat both lanes alike. Where SAP runs, and the evidence:
[SAP and PSF lanes](keplerX-2026-11-03-sap-and-psf-lanes.md).

## Detector behavior

`detect` constructs the upstream `LightCurve` and `Dump`, then calls
`Iterative_TCE_Search`. The adapter supplies prepared arrays, declared
exposure, stellar inputs and the five parameters of `config.json`'s
`detection` block: a 0.5–500-day period interval (100 days until 29 October
2026, 200 days until 2 November;
[the 500-day search](keplerX-2026-11-02-period-search-to-500-days.md)),
three nominal transits, MES 7.1 and at most ten iterations. High-SES removal
and the additional veto option are disabled.

Since 2 November 2026 the Detector is one search with no option switches;
every choice described below is code. Records dated before then name the
former switches (`detection.engine`, `segmentation`, `duration_window`,
`min_template_cadences`, `repeat_policy`, `repeat_passes`,
`single_event_veto`, `segment_baseline`, `candidate_cleaning`,
`period_step_factor`, `screen_threshold`, `template_flattening`,
`whitening`, `resolve_aliases`, `noise_window_days`,
`prune_unused_durations`); each now holds its production value in code, and
`noise_window_days`, which cost recall, is gone.

The template bank is upstream's from two cadences (0.98 h; upstream: three,
a 1.47 h shortest template) with upstream's duration window, 0.25–1.5 times
the reference duration 13 h (P/365 d)^1/3 rho^-1/3 (Adam, 29 October 2026):
template coverage of confirmed Kepler planets rises from 91.4% to 96.2%, and
of the calibration sample's detectable planets from 63.6% to 80.3%, for an
estimated 18% more period-duration cells (not yet timed on its own); the
period floor that left dense M dwarfs (density 13) without templates below
~2 d drops to ~0.6 d, which brings Kepler-42 b and d within reach. Its
duration restrictions and event counting apply. No depth ceiling is added.

Iteration follows upstream's passes (`detector/repeat_rule.py`): harmonic
removal, masks and the ten-candidate limit are unchanged. Two things differ:

- A later detection at a known candidate's own period, whatever its epoch,
  is not a new candidate. Upstream kept eclipsing binaries' secondary
  eclipses, ellipsoidal phases and eclipse wings as new candidates.
- After two consecutive passes that add nothing new, the search ends.

Other integer period ratios keep upstream's epoch test, because the search
can lock onto an alias first. The search also stops when the masks leave
fewer than 100 cadences, or when nothing whitens. A curve with fewer than
100 usable cadences, or whose grid admits no template duration or trial
period, is recorded as `unsearchable`, with no candidates.

Each refined detection must hold the transit requirement on its own events
(`detector/candidate_ephemerides.py`, `carrying_events`, since 2 November
2026): where its (k + 1)-th strongest single event is below a fifth of its
k-th, for k under the three required transits, it rests on its k strongest
events and stays only if its other events are a detection by themselves.
Two deep single events a period apart, with nothing at the other epochs,
are not a planet; folds like that crowded the 200–500-day range. The events
a dropped fold rests on are masked like a detection's and the search goes
on (`detector/repeat_rule.py`); a pass that made only such folds counts as
one that added nothing. Without that masking, a deep single event shadowed
every later pass and ended the search (Kepler-607 b and c were lost).

The adapter repairs upstream boundary failures: it retains the final cadence,
handles empty segments, repeats reflection beyond short padding buffers and
pads tiny segments sufficiently for the wavelet and transit bank. Successful
upstream padding and segment calculations remain numerically unchanged.
Compiled Vetter uses the same padding repair when whitening masked signals.
A segment whose repaired padding whitens degenerately (NaN, or denominators
above 1e12 from a zero MAD; it holds 1, 2, 4 or 8 cadences) is dropped from
that pass in the Detector and its references. Such a cadence otherwise
poisoned or flooded the search ([edge cases](keplerX-2026-10-29-edge-cases.md)).

The Detector runs the pinned upstream code with compiled period folding,
running medians and convolutions, and GPU screening. The golden tests' two
upstream references, `reference="native"` (upstream's own Python search)
and `reference="compiled"` (the kernels with upstream's convolutions,
single-event veto and harmonic cleaning), must agree bit for bit; see
[the compiled engine](keplerX-detector-compiled-search.md).

The wrapper saves every candidate returned by upstream and the initial/final
masks. Upstream does not expose every internal proposal; the wrapper does not
claim to preserve an unavailable search surface.

## Detection at the information limit (2 November 2026)

Four departures from upstream were measured that day. Evidence, scripts
and tables:
[keplerX-2026-11-02-detection-at-the-information-limit.md](keplerX-2026-11-02-detection-at-the-information-limit.md).

- Each segment is whitened about its own mean (`detector/gap_padding.py`).
  Quarters are normalised by their median, so a segment's mean is offset
  from 1; the coarsest wavelet level carries
  that offset and the transit template's own mean projects onto it. On null
  development curves upstream's MES at random ephemerides averaged −0.3 to
  −0.5, and −0.6 to −1.0 with over 200 transits: every short-period trial,
  planets included, was pushed down by up to a full MES unit. About the
  segment mean the average is 0.0 and the spread 1.0.
- Cleaning by shared events (`detector/candidate_ephemerides.py`) replaces
  upstream's harmonic cleaning, which never uses the epochs it computes,
  with a greedy suppression by events actually shared; refines each
  detection's period, epoch and duration on the matched filter (reported
  epochs are mid-transit times, not phase-bin edges); and replaces the
  re-fold check of weaker detections, whose 85%-of-MES rule rejects any
  planet under a strong short-period signal's mask, by a depth-consistency
  test with an exact-alias rejection.
- Twice upstream's period step, screened at MES 6.6: the period search
  screens half the trial periods and the refinement recovers the grid's
  loss, so MES 7.1 applies to the refined value.
- Templates of 6 hours and longer pass through KBonus's 2-day-knot spline
  flattening, the shape in which such transits are delivered
  (`detector/flattened_templates.py`), and each such template's statistic
  is divided by its own robust spread where the whitening underestimates
  its noise. The Vetter's whitening uses the same
  templates. On the 348 paired cases it adds seven planets and loses none,
  and on all development-quick cases with transits of 6 hours or longer it
  finds 104 of 126 detectable signals instead of 95 (Kepler-313 c and
  Kepler-443 b among them), at a cost of more null candidates and up to 27%
  more whitening CPU.

A detection that is another's exact alias (same period or integer
multiple at any epoch, an integer divisor off the other's events, or a
fraction p/q with q ≤ 4 exact to 5e-5 on the alias spacing), against the
pass's stronger detections and every earlier pass's, is masked but not a
new candidate (`detector/repeat_rule.alias`). On the 1,752-case
development-quick run these settings put 50% efficiency at information S/N
8.13 and 90% at 11.28 (targets 8.5 and 12; 8.76 and 12.37 before), 86.7%
of detectable planets exact (82.5%), 92.2% of known planets in range
(88.9%), and the batch at 37.6 minutes on 8 workers (62.1).

## Detector speed on survey curves (2 November 2026)

The Detector sums each template's wavelet-level convolutions in the
frequency domain with upstream's noise weights (SES within 5e-14 of
upstream's arithmetic), and also ends a segment at every gap longer than 30
days, which upstream merged and padded into segments of up to 65,536
points on survey curves. On the out-of-sample survey benchmark the Detector's
CPU falls 57% (stratum-weighted, 241 survey curves) and 63% (493
confirmed-planet hosts) with the same 589 of 620 confirmed planets found. In
a quiet 8-worker batch of 241 survey curves the Detector's CPU per curve
fell 52% and its wall time 29% (stratum-weighted; the whole batch 30%): the
shared GPU screen now bounds a batch. Alias resolution replaces a
detection by its P/2, P/3, 2P or 3P alias when that folds at least 5% more
MES, also without its strongest event: two confirmed planets reported at
harmonics are found at their own periods, none lost. A coarser period grid
(step 3) saved 32% of GPU time and looked better on the synthetic sets, but
missed five confirmed planets (two with unusually short transits) and was
not adopted
([keplerX-2026-11-02-detector-speed-on-survey-curves.md](keplerX-2026-11-02-detector-speed-on-survey-curves.md)).

## Calibration and vetting

Calibration uses the KBonus-realistic sample (`calibration/sample_builder.py`,
[keplerX-calibration-sample.md](keplerX-calibration-sample.md)) and scores
runs from their `results.sqlite` (`calibration/scoring.py`). The sample
remains conditional on delivered Flat photometry.

Batch joins vetting and localization evidence by signal ID. These are provisional BOI records,
with no acceptance rule. Vetting applies the dismissal policy that
`vetting.policy` names from `vetter/policy.py` (production:
`linear-score-binary-gate` since 3 November 2026: the `linear-score` of
2 November, on features of the signal's shape only, so a stronger signal is
never dismissed for its strength, and a binary gate that dismisses only on
decisive or corroborated binary evidence,
[binary rejection](keplerX-2026-11-02-binary-rejection.md)): a signal it dismisses
stays `DISMISSED` and is not localized; any other is `UNRESOLVED` until its
light is localized. Absent pixels remain
`NOT_RUN`; a numerical failure never dismisses and never accepts. Scoring
reports what every policy would keep on each run. With
`vetting.dismissed_metrics: "policy"` the dismissed signals skip the costly
odd/even fits and local morphology
([vetting speed and policy](keplerX-2026-11-02-vetting-speed-and-policy.md)).
The Vetter's MES and weak-secondary statistics are whitened by the
Detector's FFT sums, within 5e-15 of upstream's arithmetic.
Out-of-sample benchmarks of real survey curves and confirmed planets, and
the localization speed work (banded pixel regression, Gaia field table,
pixels streamed from MAST),
are in [out of sample and localization](keplerX-2026-11-02-out-of-sample-and-localization.md).

Known planets, validated BOIs and real leaks in the KBonus sample supply
these requests. The adapter has software checks; population pixel
recovery and disposition thresholds require the planned calibration.
