# PSF_FLAT post-BOI work

## Current calibration work — October 27

Adam deleted the recurring monitor and asked for the PSF Flat version to be
calibrated directly in this task. Do not recreate the monitor or treat an old
pending SciPy question as unresolved authorization. All seven prepared native
fits have finished; their production probability gates remain null.

The [first calibration report](../results/boi-validation/psf-flat-calibration-20261027-v1.json)
binds the fixed original 75-object cohort and its prepared original Flat
photons. The first physical Q5 experiment completed two 1-million-attempt
seeds per scenario for eight labelled false-positive controls on the exact
`fast_triceratops.engine()` path. None passes both numerical thresholds on its
two-seed mean. These are correlated catalogue controls, not a measurement of
global survey reliability or eight new candidate dispositions.

A metadata-selected original-cohort control, K01505.01 (KIC9813499), gives FPP
0.0239841 and 0.0231488 on Q5 alone, mean 0.0235665, with NFPP zero. Keep this
non-pass. Its old SAP calibration used multiple quarters and cannot supply a
causal detector or extraction improvement factor. The matched K01485.01 control
has no valid archived Flat photons in Q5; retain that coverage failure. The
next positive control is the highest published-S/N original confirmed object
with complete existing scene properties and Q5 coverage: Kepler-682 b,
K00791.01/KIC12644822, catalogue S/N277.2. Its full published Q5 channel56 batch5
is staged separately before probability calculation. Selection does not remove
any earlier control or alter thresholds.

These remain conditional instrument experiments. No accepted method/native/
candidate review is fabricated to make calibration executable. The independent
known-object runner reuses the standard preparation, noise, folding, signed
likelihood, physical-scene guards and exact fast engine; its outputs cannot
satisfy production acceptance. Physical source completeness, archival transfer,
normalization/uncertainty and candidate noise/ownership still need support.
The initial two confirmed controls in the first batch lack stellar properties
for capable physical neighbours, so their probability runs were not allocated.
Do not omit those neighbours or invent their properties.

Preserve the first worker-harness failure: a Python submodule import replaced
the same-named `engine` package attribute after a worker's first calculation.
The separate completion generation caches the context function once per
process, retains all four completed repeats and computes only the missing
repeats, with unchanged inputs, seeds and draw counts. New transfer diagnostics
also retain their earlier failures. Their post-flattening selection must
intersect the original SAP_QUALITY=0 photons with the flatten mask; the
flattener's replacement `quality` column alone does not authenticate that
intersection. Earlier transfer diagnostics with incomplete quality handling
remain research evidence and are not production qualification.

The [current runtime record](../results/boi-validation/psf-flat-runtime-status-20261027-v7.json)
records Adam's reiterated bug-fix authorization and the completed full-batch replay
v4. The two-call SciPy repair is applied only in a preserved research source
copy. The archived-flattening and fixed-operator controls completed; physical
contaminant-response qualification remains open. All four prepared native jobs
finished. Keep stopped attempts intact and do not park more BOIs for a shared
technical issue or ask again for routine compatibility-repair permission.

Adam authorized a separate PSF_FLAT version on October 26, 2026. The initial
selection contains 7 shortlisted shallow signals, 8 additional named shallow
signals and the 33 original Rust exclusions. Seventeen historical SAP native
significance failures have positive PSF_FLAT event-depth S/N above 10 and are
scheduled for review after that first wave: 65 groups in total. The two S/N
definitions differ; this is evidence for revisiting the fits, not a measured
improvement factor or proof that SAP caused each failure.

The [queue](../results/boi-validation/psf-flat-priority-queue-20261026.json)
binds the supplied signal identities and original clocks. All prior studies,
stopped attempts, frozen detector/vetting and accepted validations are retained.
New evidence uses the authorized internal disk under
`other/boi-v3-20261025/psf-flat-priority-20261026-v1/`.

The [October 27 readiness update](../results/boi-validation/psf-flat-source-readiness-20261027.json)
supersedes the October 26 counts: eight supported scientific dispositions and
57 groups still under review. Four proposed clocks are non-independent aliases:
KIC8745833 at 2.73993 days, KIC6699887 at 5.03801 days, KIC8296927 at 0.98462 days
and KIC8234517 at 6.22401 days. Their referenced clocks are 4.56654, 15.11370,
2.95362 and 18.67073 days, respectively. Masking the referenced transit photons
in a separate diagnostic reduces the proposed-primary depth S/N from 43.85,
16.94, 59.01 and 16.90 to 0.05, −0.02, −0.78 and 0.22. Half-period-shifted control
masks preserve the signals. All four original FITS products, masks, arrays and
exposures are authenticated. Do not allocate independent fits/FPP to those four
clocks. Keep the referenced physical signals, all selected catalogue records and
accepted historical studies intact; these are not astrophysical FP denials.
The full-orbit morphology pages retain all 65 supplied clocks. The unchanged
survey SWEET-style diagnostic also ran on the 57 continuing clocks in a
preserved cached survey environment; two import-only failures remain recorded
and production is unchanged. KIC5194722, KIC5896090 and KIC9226435 show broad
coherent modulation despite available TIC stellar errors. Those inputs alone
do not justify expensive transit fitting; inspect an isolated source-owned
transit first. Modulation can coexist with a transit, so it is not a new veto.

## Photometry and native fitting

`post_boi.psf_flat` authenticates the original FITS hash, Gaia identity, checksums,
TDB time convention, exposure and electron-per-second units. It reads only
`LIGHTCURVE_STITCHED.PSF_FLAT_FLUX` and its errors. The accepted mask requires
finite time/flux/error, positive error, `SAP_QUALITY=0` and `FLATTEN_MASK=0`.
It retains every delivered row and flag and applies no CROWDSAP correction.

The existing preparation and juliet/batman native fitter receive these photons
through the same input interface. The inherited candidate clock, stellar
uncertainties and sampling settings remain explicit; local noise-prior widths
are recomputed on PSF_FLAT photons. New conditional native fits retain a separate
scientific review gate before probability allocation. A Kepler-565 b preparation
control runs alongside the first supplied candidates. Its historical RUWE does
not make it a new admission control or a new validation.

The original Kepler V3 native kernel and exact `fast_triceratops` engine remain
the required runtime. The existing coordinator owns native/FPP allocation and
measures CPU availability before additional starts. Do not start another
coordinator or change sampling budgets to fill idle cores.

## Probability qualification still required

PSF_FLAT is the flattened, rescaled output of a joint source-amplitude fit.
The delivered `APERTURE_Q*` masks describe SAP photometry; neither those masks nor
CROWDSAP establishes the PSF extraction response. The current V3 `response.py`
and its flux-closure gate model SAP. Do not reuse their response fractions for
PSF_FLAT, assume perfect deblending, or omit nearby-source alternatives.

Qualify the PSF extraction and archived flattening response, including leakage
and target amplitude, before FPP allocation. Preserve the original delivered
photons, characterize observed-cadence transfer and noise, and use known planet
and contaminant controls. Native shape/clock inference alone does not close this
gate. The normal V3 scene, literature, timing, body-radius, physical-closeout,
first-two and 20-repeat requirements remain in force.

## First flattening control

The [qualification record](../results/boi-validation/psf-flat-method-qualification-20261026.json)
preserves a Kepler-565 quarter-5 replay. All 4,487 cadences and 21 break-mask
flags agree. Removing one diagnostic quarter-normalization factor leaves a
flux residual RMS of 0.000000008 ppm. This establishes the published flattening
operation for this quarter, conditional on the supplied native PSF flux. It
does not identify the full cross-quarter normalization or extraction operator.

The authors' Lightkurve 2.3.0 dependency raises a unit error when adding model
variance to dimensional photon variance. An isolated one-line repair uses its
already constructed model-error Quantity. Physical-unit and original
dimensionless calculations agree, including a flux-unit scaling control.
The production runtime is unchanged and both failed replay attempts remain.

Nine paired box perturbations on these observed cadences, with a 4.244-day
period, 3-hour duration and depths of 300–3,000 ppm, transmit 91.4–99.3% of the
input amplitude through flattening. The result depends on amplitude and phase;
it cannot supply a universal correction. These tests precede the required
target/contaminant pixel injections and do not qualify FPP calculations.

The inspected public workflow calls a `prior_sigma` argument absent from the
unmodified PSFMachine 1.1.2 tag named in the delivery headers. Its public tree
contains FFI shape models but no TPF batch map or archived fitted time operator.
Reconcile that release and extraction response before probability allocation.

The author's `use-at-nas` fork provides a stronger reconstruction candidate:
commit `b04570c` (July 8, 2022) reports version 1.1.2 and accepts `prior_sigma`.
A single-stamp probe retains four fitted sources and 24 pixels for the control.
It drops the actual 0.91-arcsecond fainter companion under its documented
one-arcsecond cleaning rule. Retain that companion in the physical scene.
The probe does not establish full-batch or time-operator parity. Its isolated
runtime has disclosed dependency metadata exceptions; production is unchanged.

The author's separate KBonus repository supplies a complete quarter-5 lookup
table with 166,270 entries. Its channel-13 batch 4 contains the control and 251
other TPFs. Replaying the public batch-sorting code changes four assignments
among the channel's 1,381 files. Stage the published table unchanged for the
first full-batch control. Preserve the membership discrepancy and establish
archive parity before treating this as the delivered operator. Input staging
uses two download workers; extraction waits for measured CPU/memory capacity.
All 252 staged files passed compressed-byte binding, gzip CRC, raw FITS checksum,
KIC/quarter/channel/module/output and published physical-origin checks. These
authenticate the replay inputs; they do not establish the fitted PSF response.
Read-only decompressed copies preserve their payload hashes and match the actual
Lightkurve reader in flux, errors, time, quality, cadence, footprint and WCS. The
official DR25 Q5 CBV file also passes identity and checksum checks. A separate
replay runner retains the dated extraction prefix through its global scale,
using four CBVs and 40 smoothing knots. After both samplers stopped, measurements
supported a separate six-thread generation alongside the two single-core native
postprocessors. Its guards required no sampler workers or FPP jobs, at least
6.5 idle cores and 5 GiB of free/inactive memory. Production stayed unchanged.

The [latest qualification record](../results/boi-validation/psf-flat-method-qualification-20261027-v5.json)
retains the first wrapper-namespace failure and its separate correction. Replay
v3 then reached post-extraction TPF bookkeeping and stopped: SciPy 1.15 returns
a scalar mode where the authors expect an array. Both affected calls have a
proposed isolated repair. Tests against unchanged SciPy 1.8.1 code reproduce all
524 integer-index, tie and empty-array controls. Adam reiterated permission to
fix bugs on October 27. The exact reviewed repair is now applied in a separate
package/source copy for replay v4; the original installed sources and failed
attempts remain intact. These are API controls, not extraction/FPP qualification.

Replay v4 completed with 585 fitted sources, 10,300 pixels and 4,487 cadences.
The raw PSF time series differs from the archive by 2,277.8 ppm RMS after a
diagnostic constant scale. Published flattening reduces that difference to
22.48 ppm RMS, with no mask mismatch. At the existing known-object clock the
local transit depth is 1,146.98 ppm in the archive and 1,148.93 ppm in the replay.
The diagnostic scales are not accepted corrections or replacements for archive
photons. This one-quarter agreement supports further response work; it does not
establish exact extraction parity or a candidate's FPP method.

The fixed authored operator also reproduces its saved target coefficients at
three preselected cadences. Twelve paired source-amplitude perturbations agree
with its signed response. The first arithmetic control is retained separately:
float64 perturbations changed the float32 baseline, so the corrected experiment
subtracts a zero-perturbation baseline of identical dtype and tightens tolerance
to the injected amplitude. Negative pixel weights and small negative fitted-
source responses demonstrate why normalized SAP fractions are unsuitable.
These are same-model algebra controls. The physical projection and paired
controls below address neighbor leakage and archived processing; broader
domain and likelihood qualification remain required before probability allocation.

An [independent calibrated-PRF sensitivity control](../results/boi-validation/psf-flat-physical-response-control-20261027.json)
now evaluates the actual target and omitted 0.91-arcsecond companion through
those signed pixel weights. It uses the existing reviewed Kepler PRF calibration,
the saved footprint, propagated positions and observed pointing corrections at
the same three cadences. For equal intrinsic photon-flux changes, the companion
response is 95.7–96.2% of the target response. This is not its flux fraction or
an identification of the transiting host. It demonstrates material leakage in
this control and rules out assuming perfect deblending. The following paired
controls extend this projection through extraction and archived flattening;
broader uncertainty and likelihood controls remain open.

The [physical input-control review](../results/boi-validation/psf-flat-physical-injection-review-20261027.json)
now preserves actual periodic photon deficits on both locations across 48 native
pixels and four augmented mission-background pixels. Each trial recomputes the
authored background, shape/source masks, time model, negative-source prior
selection and flux fit before
published flattening. Native reader/quality selection and the observed noise are
held fixed. The two equal absolute deficits produce 989.74/949.50 ppm before
flattening, 933.59/892.53 ppm after it, and 979.81/939.39 ppm through unchanged
standard transit-protected preparation. No native candidate fit was rerun.
The prepared companion/target response ratio is 0.95875. This remains a control,
not the companion's flux fraction, a host assignment or a validation.

An exact three-source, 15-pixel connected-component solve reproduces the full
585-source fixed operator over all 4,487 cadences. Its physical predictions
differ from the full refits by 0.019 ppm RMS. This enables an 18-case phase/depth
grid without repeating all extraction fits. After archived flattening and the
standard preparation, Flat/raw amplitude ratios span 0.9750–0.9949. At the two
fully refitted entries the prepared-depth differences are 0.0164 ppm and
−0.000035 ppm. These are one-quarter controls, not universal corrections.

The separate signed-response adapter now connects preparation, physical scene,
frozen inputs and the existing probability coordinator. Its software controls
include the unchanged Kepler-10 stock gold test, direct observed-photon Gaussian
likelihoods, an exact aperture limit and bitwise evidence equality between the
pinned upstream and exact fast engine for a declared signed, nonnormalized
response. These are software controls. Candidate-specific response, normalization,
uncertainty, noise and physical gates remain open; no BOI FPP has been allocated.
All 58 package tests pass, including a frozen PSF two-seed software benchmark
through the existing fast coordinator. The first benchmark completed its two
calculations but a test misread its aggregate-statistics field; that failed
assertion and the successful calculations remain preserved. The corrected test
uses the existing repeat-record field. The
[adapter record](../results/boi-validation/psf-flat-probability-adapter-20261027.json)
binds the code generation and controls.

The first candidate-specific response generation uses KIC8573780's published
Q5 channel-67 batch 6, containing 197 TPFs and the authenticated parent KIC8573783.
All 197 compressed inputs and read-only payloads are authenticated, paired
original-reader controls pass, and the FFI shape, mission background and actual
module's shared DR25 CBVs are authenticated. The
[candidate response record](../results/boi-validation/psf-flat-first-candidate-response-inputs-20261027.json)
binds a separate reconstruction under measured six-thread admission. Its authored
scientific extraction prefix matches the completed control. Archive parity and
physical response remain under review; no new native transit fit or BOI FPP runs.

### Signed-response input contract

New probability generations declare `flux_convention: psf_flat` and
`aperture_response: signed_psf_flat`. The `psf_response` block names `data`,
`input_review` and `review` files. Neither delivered photons nor prepared photons
receive a CROWDSAP correction. Existing native generations stay unchanged.

The response NPZ contains exact string `source_id`, original `original_index`,
`time_bkjd`, `quarter`, a photon-by-source `coefficient` matrix and source-order
`secondary_sigma`. Coefficients map an intrinsic relative source change to the
prepared observed Flat change, including the source/baseline conversion once.
They may be signed and need not sum to one. Physical geometry supplies actual
`footprint_q*` pixels and `xy_q*` positions for the complete physical scene.
It supplies no SAP aperture fractions.

`secondary_sigma` records intrinsic-source scalar noise for the unchanged
upstream secondary-eclipse screen. The upstream field named `fluxratio` carries
only the ratio of declared observed noise to this reviewed scalar. The primary
likelihood uses the signed photon coefficients. These bookkeeping values are
not physical flux fractions. The scalar's adequacy for time-varying signed
response requires explicit evidence; no mean or maximum response is substituted.
Missing values remain NaN, and any potentially signal-producing source with
missing secondary noise or stellar properties stops allocation.

The candidate review binds the exact clocks and receipts for response, reader
review, photons, preparation, scene, geometry and noise. It records normalization,
transfer, complete-scene, secondary-noise and uncertainty bases, scientific
evidence and a separate accepted method review. Freeze verifies these receipts,
retains their evidence and checks the actual photon rows and physical scene.
Unsupported method or candidate acceptance blocks allocation. The original
20-seed, first-two, upstream-prior and physical-closeout requirements remain.

## First completed BOI fit

The [partial native review](../results/boi-validation/psf-flat-first-native-review-20261026.json)
for KIC10528996's 13.7447-day signal reports radius-ratio S/N 33.75, effective
sample size 5,355 and matching stock likelihood replays. Its assumed-host
apparent radius is 2.84 Earth radii. Additional preparation transfer is within
0.05% of unity in amplitude. These results do not release the probability gate.
One folded out-of-transit outlier is dominated by a retained positive cadence;
noise adequacy, archived response and physical source review remain open.
At Q8 cadence 31765 the authenticated parent pixels show an 87.5-electron-per-
second excess in one pixel (38.8 formal difference errors), coincident with a
delivered cosmic-ray correction of 137.7 electrons per second at that pixel.
This supports a residual pixel-event explanation. No cadence was removed and
the noise/probability gates remain open.

The [second partial native review](../results/boi-validation/psf-flat-second-native-review-20261027.json)
for KIC8573780 reports radius-ratio S/N 27.64, effective sample size 5,456 and
matching likelihood replays. Its assumed-host apparent radius is 2.14 Earth
radii. Additional preparation gains are 0.99838 and 1.00055; individual residuals
and the folded transit still require review. The preserved KOI/DR25 screen finds
no low-order other-star matches for either completed posterior clock. Other
ratio matches remain recorded. This query result does not establish source
ownership, novelty or physical validation.

The [Gaia scene census](../results/boi-validation/psf-flat-scene-readiness-20261026.json)
covers all 62 selected stars, with 3,367 unique catalogue sources in their
60-arcsecond cones. Two selected stars have companions within 1.5 arcseconds at
the catalogue reference epoch. Exact quarter propagation and imaging remain
necessary; these flags are not automatic exclusions.

Reported TIC mass, radius and errors are available for three more targets:
KIC5194722, KIC5896090 and KIC9226435. Exact Gaia associations and the raw
responses are preserved. The latter has a nearby catalogue-clock warning;
stellar model support and source ownership still precede fit admission. The
Gaia supplementary query supplies 23 reported logg intervals but no additional
spectroscopic FLAME mass/radius pairs for the 30 queried stars. No derived
stellar prior or missing uncertainty has been substituted.

The [additional stellar-readiness record](../results/boi-validation/psf-flat-stellar-readiness-20261027.json)
retains Gaia astrophysical-property queries for every one of the 3,367 scene
sources. Reported Teff/logg/radius values exist for 1,251 and FLAME masses for
418; missing values remain missing. Two existing isoclassify solutions are
available for host review. Of the 25 other hosts still lacking reported pairs,
nine meet that established inference route's input rule and seven have usable
colour inputs staged. This is not a universal post-BOI parallax cut. No new
stellar fit or prior adoption has started. Gaia source 2127702609183750528 has
pseudocolour outside the zero-point calibration range; its retained boundary
correction needs explicit support review before fitting.

The completed BOIs' standard phase-0.5 occultation measurements are 12.0 ± 11.7
ppm for KIC8573780 and −10.8 ± 12.0 ppm for KIC10528996's 13.7447-day signal.
Neither crosses the three-sigma phase-0.5 rule. Scan maxima and event-to-event
scatter remain recorded; this does not close noise, timing or source ownership.
The native residual reviews average model values on the actual contributing
cadences and reproduce the saved likelihoods. They leave every photon intact
and release no probability gate.

The [third BOI review](../results/boi-validation/psf-flat-third-native-review-20261027.json)
reports KIC10528996's 8.55572-day signal: native ESS 5,360, matching likelihood
replays and an assumed-host apparent radius of 1.91 Earth radii. Its radius-ratio
precision metric is 12.44; the separate established matched-filter S/N is 23.95
on the independent native window. Additional preparation gains are 0.99936 and
1.00001. The phase-0.5 secondary is −8.7 ± 10.5 ppm. Noise, source ownership,
timing/aliases and extraction response remain open.

The [same-source review](../results/boi-validation/psf-flat-same-source-review-20261027.json)
preserves a paired event-masking experiment on the authenticated original Flat
photons. The 13.74458- and 8.55572-day clocks retain local event-depth S/N of
46.15 and 23.64 after masking the other clock, versus 47.53 and 24.90 before.
Equal-width shifted masks retain 45.35 and 24.03. These formal-error diagnostics
support separate repeated dips; they do not close the independence or noise gate.
Using the existing preparation and measured noise, event-depth reduced chi-square
is 2.01 and 7.97. One Q2 window centred at BKJD 245.92921 contributes 85.3% of
the weaker signal's event scatter. Its large ramp and abrupt drop already occur
in delivered Flat and remain after preparation. The mechanism is unresolved;
no photon, mask, prior, native fit or probability gate changed. A failed receipt-
path comparison before measurement is retained separately from the successful
content-hash and array-equality check.

The [Kepler-565 control review](../results/boi-validation/psf-flat-native-control-review-20261027.json)
recovers the known clock, with ESS 5,118 and matching likelihood replays. Its
high RUWE and close companion remain limitations. The unchanged 10-thread
Kepler-10 stock-evidence gold test passed after allocation became free; together
with the earlier 47 package tests, all 48 collected tests passed. This verifies
the reader/software route, not the new PSF probability method.

Sources: [KBonus methods and file definitions](https://arxiv.org/html/2310.17733v1),
[archived public stitching implementation](https://github.com/jorgemarpa/kepler-workflow/blob/cf9dda6b398d050cf46d62ad01b2459570f492cd/kepler_workflow/do_fits_stitch_nas.py).
