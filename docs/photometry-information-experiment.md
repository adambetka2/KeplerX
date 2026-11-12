# Photometry information experiment

This development pilot isolates pixel extraction, preparation and first-search
losses, with a separate paired check of the full detector. It does not estimate
survey completeness or validate planets.
Inputs are explicitly selected, unsealed single-quarter light curves. Retain
the first 45 days on their native timestamps and retain all raw input arrays.

## Paired comparison

Subtract the same exposure-integrated, quadratic-limb-darkened transit loss
from raw source flux before quality masking or normalization. Where supplied,
use the extraction operator's source-flux response, preserving dilution.
Otherwise the experiment assumes the delivered median is source flux.
Record this distinction in the manifest. Apply the production quality mask,
quarter normalization, upper-outlier rejection, sorting and cadence deduplication.

Compare exact production biweight windows of 0.5 and 1 day with a 1.5-day cubic
spline jointly projected from flux and every search template. This third arm
tests joint scoring at the initial search. The production physical refinement
already jointly fits nuisance splines to original, pre-biweight flux. The
pilot does not claim to introduce joint fitting or reproduce that full detector.
The research `window2x` configuration also changes physical spline spacing;
the preparation experiment isolates window size. The separate native experiment
compares all four combinations of the two window and spline-spacing settings.

## Pixel operators and budget

Select six G 16–18 K-dwarf sources by a frozen metadata hash order and local
pixel availability, without detector results. Five have sufficient usable
pointing-supported cadences; the sixth has only one and remains an explicit
preflight failure. Use calibrated target pixels, the delivered source aperture,
catalogue positions and the mission PRF. Include the original Kepler parent
star in the scene. Fit nonnegative source amplitudes and a signed constant
background on the uninjected median image, then freeze the extraction weights.

Compare aperture sums with a target-PRF operator that projects out listed
neighbors and a constant background. Translate the target PRF with pointing.
Clip small negative interpolation lobes and normalize on full PRF support
before cropping; retain the correction diagnostics. Subtract target-only
transits before extraction. Check all 36 physical loss profiles against both
operators, preserving a full pixel injection and the other replayable losses.
All 360 pixel-versus-extracted transfer checks pass on the five valid scenes.
This verifies the frozen linear extraction, not an adaptive photometry pipeline.

Two aperture replays match delivered SAP flux and propagated errors; three
differ by 27–46 electrons/s. Native pixel and delivered timestamps differ by
up to seven seconds. Therefore these are paired native aperture experiments,
not a claim of byte-identical delivered-SAP replay.

The subsequent [pixel reprocessing experiment](pixel-reprocessing-experiment.md)
resolved these three offsets: KBonus had replaced the native background.
Its new full-quarter replay agrees with delivered SAP to float32 rounding.
The earlier pilot's inputs and outcomes remain unchanged.

Use `INT_TIME * NUM_FRM` for live exposure. Propagate target, neighbor and sky
Poisson variance through each operator. Read variance is
`NUM_FRM * READNOIS**2 / live_seconds**2` per pixel, conditional on independent
frame reads. Retain supplied background-model uncertainty separately to avoid
unverified double counting. Preserve signed unexplained variance. Also compare
observed total flux plus sky Poisson variance and read variance without assigning
the observed flux to catalogue sources. This does not establish an ultimate bound.

## Frozen search and injections

Before results, save the executable hash, dependency hashes, package versions,
input manifest and design. Search 81 periods uniformly from 9.75 to 10.25 days,
256 phases each and 2-, 4- and 6-hour boxes. Use the identical complete bank for
all arms and every null. Inject off-grid periods uniformly between 9.8 and
10.2 days, two phases for each duration and nominal area-ratio depth of 40,
100, 300, 1,000, 3,000 and 10,000 ppm. The last two are positive controls.
Use 11 exposure samples over 0.02043361111111111 days. Imposed durations are
transfer probes; they are not a stellar-density-consistent planet population.

The smallest box is wider than the maximum endpoint period drift plus half
the phase spacing over 45 days. Off-grid physical gold injections independently
test localization with the actual bank. Known-clock, paired depth transfer is
a separate diagnostic and is never used to choose a blind search proposal.
This transfer includes the small cross-term from separately normalizing the
original and injected curves. Joint projection is exactly linear on fixed
normalized arrays, but the complete raw-input transfer need not be exactly one.
Its formal score uses reported errors and the injected normalization; it is
not a photon bound or an independently calibrated detection significance.

## Conditional false alarms

Fit a spline to each uninjected curve. Independently change residual signs in
one-day blocks, restore the spline trend, and replay preparation and the full
bank. Repeat with two-day blocks as a sensitivity check. These are conditional
wild-bootstrap nulls, not independent observations or real-survey reliability.
Preserve raw simulated flux and selected masks. Use 128 calibration and 128
disjoint held realizations for each block length, with source- and quarter-based
seeds shared across extraction operators. Keep the original uninjected maximum.

Use plus-one empirical ranks of the complete search maximum and a predeclared
5% threshold. Report held false alarms and conditional binomial intervals.
The one-day calibration is the primary recovery criterion; report two-day
sensitivity separately. Count recovery only with significant, correctly
localized period and phase. Report source-level dependence: phase, depth and
duration probes on the same source are not independent population trials.

## Full detector comparison

Use the five native aperture curves, each uninjected and with nominal 1,000
and 3,000 ppm, four-hour probes at 9.9731 days. Preserve the production period
bands, thresholds, local noise model, refinement and consistency gates.
Compare the selected 0.5/1-day windows and three-day spline spacing against
window-only 1/2 days, knot-only 1.5 days, and the combined research design.
Snapshot code and inputs, run all 60 searches before scoring, and retain every
proposal, first failure and periodogram. Use the production clock matcher;
report observed injected-event support and original-curve proposals separately.
These five originals are controls, not an empirical reliability calibration.

## Findings and next experiment

Numerical results and evidence references are in
`../results/experiments/photometry-information-20261014.json`. The 0.5-day
window still attenuates long shallow signals; its attenuation applies to the
coarse-search series. The physical refinement already receives original
prepared flux. Rebuilding LC2 or the preparation arrays is unnecessary for a
window-only change. Changing quality masks or extraction requires a new input.

The pilot supports testing joint nuisance projection or duration-aware
detrending at the initial search. It does not support adopting the experimental
PRF extraction or promoting a detector: recovery gains are small, one held
noise cell exceeds the intended false-alarm rate, and all four full-detector
configurations have identical recovery counts on these probes.

Before scaling, calibrate the complete search on independent original,
inverted and simulated controls with enough tail samples. Use fresh source
clusters and full available baselines. Any pixel re-extraction proposal must
first fit a calibrated scene and pointing response: the stationary mission-PRF
scene has substantial residuals, so its source amplitudes are conditional.
Reported-error agreement, algebraic amplitude retention, an informal opinion and
an immutable manifest are not substitutes for measured recovery at controlled
false-alarm rates.

## Execution

Run `dwarfs/experiments/preparation.py MANIFEST OUTPUT` through the task's `uv`
runtime. Each manifest entry specifies `name`, `source_id` and an NPZ `path`.
The NPZ contains `time`, `flux`, `error`, `quality`, `quarter` and `cadence`,
optionally `source_flux`. Use one process and one BLAS thread. Keep all outputs,
Numba caches, test caches and logs under `dump/photometry-information-20261014`.
Preserve the first failure and fix expectations as well as implementation when
the injected gold tests fail. No sealed cohort or survey outcome is reopened.

`pixels.py --manifest INPUT --out OUTPUT --prf-cache CACHE` builds paired inputs.
`native.py prepare OUTPUT --manifest INPUT`, then `run OUTPUT` and `score OUTPUT`,
runs the separate full-detector comparison. `summarize.py OUTPUT RUN...` creates
JSON, CSV and figures from completed preparation runs. Pin Python 3.12 and
`dwarfs/experiments/requirements.txt`; the exact runtime freeze is in the dump.

The budget follows the
[Kepler Data Processing Handbook, section 5.3.1](https://archive.stsci.edu/kepler/manuals/KSCI-19081-003-KDPH.pdf).
Pixel injection and recovery follow the measurement principle in
[Christiansen et al. (2013)](https://arxiv.org/abs/1303.0255).
