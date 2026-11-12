# Pixel reprocessing experiment

Test whether rebuilding initial photometry from calibrated Kepler target pixels
improves transit recovery over the delivered KBonus SAP extraction. This is a
new upstream experiment. The earlier fixed-operator and downstream preparation
pilot did not refit adaptive pixel processing after injections.

## Result

The new processing recovers useful information from noisy pixels, but this
pilot demonstrates no recovery improvement at 40–1,000 ppm. At 10,000 ppm,
the pixel arm recovers four Q1 injections missed by mission-only processing;
both corrected arms recover five Q2 injections missed by SAP. No detector or
extraction change is adopted from these five sources.

The controlled search completed 2,560 pixel-null realizations and 292 pixel
injections per arm. Every realization refits both adaptive extractors. For
nominal profiles, each depth has 30 probes across the same five sources:

| Nominal source depth (ppm) | Replayed SAP | Pixel correction | Mission vectors only |
| --- | ---: | ---: | ---: |
| 40, 100, 300 (each) | 0/30 | 0/30 | 0/30 |
| 1,000 | 5/30 | 4/30 | 4/30 |
| 3,000 | 17/30 | 18/30 | 18/30 |
| 10,000 | 20/30 | 29/30 | 25/30 |

The one-count difference at 1,000 ppm is a frozen clock-scoring boundary.
All three arms detect that signal with empirical p = 1/129 under both
calibrations. The corrected arms choose the adjacent period-grid cell,
failing the standalone accumulated-period-drift cut. Their actual supported
event centers remain within the stated timing tolerance once the initial
epoch offset is included. Keep the primary counts; do not interpret this
one-count change as loss of physical detectability. A future scoring protocol
should define agreement on supported event times before inspecting outcomes.

Four-hour source-equivalent scatter falls by an average 20.9% against SAP.
Q1 improves by 38%; its 10,000 ppm recovery rises from 1/6 to 5/6 while
mission-only processing remains at 1/6. Q2's large improvement is already
obtained by mission-only processing. Across five source clusters, superiority
over mission-only processing or delivered PSF is uncertain. The Q1 spatial
model is also the explicitly weaker mission-PRF prior.

The pixel arm retains 98.64–101.05% of the known-clock projected transit
amplitude after fully refitting across all tested profiles. Its combined-rule
held false alarms are 14/640
for one-day and 25/640 for two-day surrogates; the largest individual cell
is 8/128. These are conditional noise checks with broad uncertainty, not
independent survey reliability. The experiment does not establish a physical
noise floor, population completeness, or the programme's adoption gate.

## Inputs and baseline

Use the five previously selected, unsealed development sources with usable
native pixels and pointing. Retain their full available quarters. Selection
was based on metadata and local availability, before this experiment's outcomes.
This is a small feasibility experiment, not an estimate of survey completeness.

The starting material is the calibrated target-pixel time series, including
native background estimates, errors, quality flags and pointing corrections.
These are exposure-integrated detector measurements, not individual photon
arrival records. The experiment does not redo detector calibration from raw
digital counts or recover pixels that Kepler never downloaded.

Replay the delivered aperture and its background convention by cadence number.
For three sources, KBonus replaced the native background: add the native
background back to each aperture pixel and subtract delivered `SAP_BKG` from
the sum. Two sources retain the native background convention and have zero
delivered `SAP_BKG`. Preserve both conventions explicitly. Compare replayed
flux against the original delivered SAP before testing a new method.

The delivered PSF light curve is an observational benchmark. A signal added
directly to that light curve is not a replay of adaptive KBonus PSF extraction.
Only claim a pixel-level PSF comparison if its extraction is actually replayed.

## New extraction

Learn an instrumental correction from the observed pixel time series using
regularized regression. Project the target's spatial response out of the
predictors so that its brightness variation is protected. Protect the first
spatial derivatives as well. Use blocked temporal cross-fitting and a guard
interval when predicting held-out cadences, with predeclared regularization.
Record all choices in the executable design before scoring.

Include four mission cotrending vectors and pointing terms as nuisance
predictors. Compare three arms: replayed SAP, the new pixel correction, and
the same learned correction using only mission vectors and pointing. Fully
calibrate and inject all three. The third arm separates gains from additional
pixel information from gains obtainable with existing mission regressors.

Use the published empirical KBonus FFI shape models for four sources. The
Q1/channel-65 model is absent from that archive; predeclare the mission PRF
as this source's spatial prior and flag its weaker calibration. Preserve raw
normalization constants and the finite support used to normalize each profile.
Test alternative response models and displaced profiles separately. Twenty
pixels in a small stamp cannot independently identify the broad empirical
PSF calibration that KBonus learned across many stars and pixels.

Inject target-only, exposure-integrated transit losses into the pixels before
the new extraction. Rerun every adaptive extraction step for each injection
and each pixel null. In the controlled comparison, use the same downstream
detector, cadence support and scoring rules for all arms. Preserve original
pixels, background conventions, injections, random seeds, model fits, masks,
thresholds, every proposal and the first failure.

The original mission calibration and delivered KBonus background convention
remain fixed inputs. Refit the new pixel correction completely. This tests an
additional adaptive correction from original pixels; it does not reconstruct
the original multi-TPF KBonus background and PSF-perturbation fits.

## Decision evidence

Measure blind-scored injection recovery at thresholds calibrated from separate
pixel nulls.
Apply each null transformation to the entire simultaneous pixel vector so
that spatial covariance is preserved. These remain conditional surrogates,
not independent astronomical observations or a survey reliability estimate.

Report transit-amplitude transfer, blind period and epoch recovery, held-out
false alarms and source-level paired results. Multiple injections on one star
are dependent. A reduction in light-curve scatter is insufficient for adoption.
Neither fitted source amplitudes nor agreement with reported errors establishes
a photon-noise limit. Keep sensitivity to source flux and spatial profile explicit.

Search 9.75–10.25-day periods with 2-, 4- and 6-hour boxes using the same
Astropy BLS search after a fixed 1.5-day cubic nuisance fit. Set period spacing
from the full observed span and phase oversampling to ten. This restricted
blind search measures the extraction comparison, not production completeness.
Inject 40, 100, 300, 1,000, 3,000 and 10,000 ppm nominal source depths, with
two phases per duration. Use the delivered corrected-PSF median as the source
rate. Retain the original noise when subtracting exposure-integrated signals.
Nominal depth is the squared planet-to-star radius ratio; limb darkening
changes the central transit depth. These imposed-duration probes are not a
stellar-population simulation.

For each one- and two-day null-block choice, use 128 calibration and 128
separate held realizations. Recovery requires significance under both
calibrations, the correct clock and at least three supported events. Report
held false alarms for this exact combined rule as well as individual-block
diagnostics. Independent-profile, displacement and width probes check the
sensitivity of 1,000/3,000 ppm four-hour injections to the spatial assumptions.

## Production detector endpoint

All 45 searches completed without runtime or search failures. At 1,000 ppm,
each arm recovered 1/5 injections. At 3,000 ppm, SAP recovered 3/5 and both
corrected arms recovered 4/5. The added Q2 recovery uses identical cadence
support and is shared with mission-only processing. This endpoint therefore
shows no recovery gain attributable specifically to pixel predictors at the
tested depths. The original light curves produced two, one and three total
proposals respectively; none matched the injected clock. Those five original
curves per arm do not provide an independent false-alarm estimate.

Separately run 45 searches: five sources, three extraction arms, and original,
1,000 ppm and 3,000 ppm inputs. Refit extraction after each pixel injection.
Use period 9.9731 days, epoch native start plus 2.731 days and four-hour
duration. Freeze these choices before production searches. Preserve the
unchanged selected detector configuration: 0.5/1-day search windows and
three-day physical-fit spline knots. Supply the same aperture errors for all
arms and use the normal production preparation. Snapshot the engine and save
every proposal before scoring. This is a small endpoint check, not a substitute
for the separate false-alarm experiment or the programme's 10,000-case gate.

This endpoint evaluates the frozen detector snapshot from commit `9fb7652`,
not the concurrent survey changes introduced in `f7e78fd`. Normal preparation
removes one additional Q12 cadence in both corrected arms (3,263 versus 3,264
for SAP); every resulting mask and signal-support count is preserved.
Integration found that the concurrent survey change retired the earlier
configuration files. `native-designs.json` now preserves the exact historical
experiment configurations explicitly; the completed evidence is unchanged.

The methodological references are the
[KBonus extraction paper](https://arxiv.org/html/2310.17733v1),
[linearized field deblending](https://arxiv.org/abs/2106.08411), and
[EVEREST's pixel decorrelation and injection tests](https://arxiv.org/abs/1607.00524).
This experiment is a local test inspired by those methods, not a reproduction
of the complete KBonus or EVEREST pipelines.

## Evidence location

The committed [result](../results/experiments/pixel-reprocessing-20261014.json)
contains the controlled summaries, all 45 endpoint scores, boundaries and
hash-bound references to the full evidence. The independent checks reproduce
input masks, SAP replay, selected refitted null searches, aggregate scoring
and the production endpoint's preparation and scoring. All 26 experiment
tests, Ruff and the complexity check pass.

Keep generated evidence under `dump/pixel-reprocessing-20261014/` on the
external disc. Preserve the earlier `dump/photometry-information-20261014/`
outputs unchanged. No sealed benchmark, survey result or running server job
is reopened.
