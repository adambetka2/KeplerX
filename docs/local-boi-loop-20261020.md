# Local BOI follow-up loop

Historical V1 execution record. New candidates use `docs/post-boi-protocol.md`
and `TRICERATOPS_KEPLER_V2`; do not launch these legacy helpers from the current
working tree. Frozen V1 evidence and runtimes remain unchanged.

Adam authorized sequential follow-up of further BOIs on October 20, 2026 and
specified the MacBook only. The server is reserved for other work. This loop
starts from the retained 125-BOI queue; it does not resume the three accepted
studies or claim to validate every candidate.

Apply [the full protocol](post-boi-protocol.md): exact-source catalogue/literature
and Kepler spacecraft checks, native fit, complete scene/likelihood preparation,
review and freeze, twenty primary probability integrations, deep replay and
physical closeout. Save a candidate that needs further evidence with that reason
and advance to the next eligible target. Keep all failures and unfavorable runs.

## Active candidate

KIC 4904094 (Gaia DR3 2103348117391603840, TIC 120316276) is running its first
two complete probability integrations on the Mac. Its native fit passed at
24,000 steps per ensemble: Rhat 1.01194, epoch ESS 738.97 and radius-ratio SNR
6.04490. The adopted period is 35.89368458751281 days, central epoch
953.1132745473803 BKJD TDB and median duration 4.07249 hours. Zero recovered-clock
integrations are assigned: the maximum 13.16-minute clock displacement is below
one exposure and small relative to the 244.35-minute duration. Weak, undercovered
Q7 timing sensitivity remains explicit.

The [execution record](../results/boi-validation/local-kic4904094-readiness-20261020.json)
binds 1,390 dependencies in `probability-v2`, freeze
`259c8824e50f59223d1bad337601642f71eeb849767447361508bc383b0060c7`.
All 105 scenario families and 31 sources remain. Direct reference replay gives
posterior L1 5.59e-17 with identical thread-mask results; eleven protocol tests
pass. The two initial runs receive five renderer threads each. All 4,760 Gaia
field rows pair uniquely with Kepler counterparts; all potentially selected
background hosts pass the model-domain check.

Twelve measured quarterly pixel stacks prefer the target in both retained
baseline windows. Q1 has nine raw primary cadences, all excluded by the original
quality mask; no transit localization is claimed there. The original pixel
assertion and phase-preparation failure remain preserved. Q1's 1,420 valid
out-of-transit photons have exactly zero coefficients in both clocks' fixed
phase-bin operators. The reviewed derived preparation projects only those
independent null-quarter groups before normalization, preserving every original
array and mask. Full-photon comparisons with two diagnostic-only Q1 noise and
normalization choices give exactly equal observed, model and covariance results.
Production invents no missing-quarter noise and removes no contributing group.

The exact APOKASC-3 Gold match for neighbor KIC 4904086 now supplies 4,096
supported measured stellar particles using the existing reviewed constructor;
its median physical Gaia-G residual is 0.0034 mag. Neighbor KIC 4904106 has a
detection-only entry without measured mass or radius. Sparse empirical priors,
native grazing/radius-bound influence and physical closeout remain explicit.
No planet admission is claimed while the probability campaign is running.

KIC 5079515 (Gaia DR3 2103418559153168128, current TIC 120255498) is held before
native sampling. Its original 9.288-hour duration exceeds the retained six-hour
model maximum and overlaps the fixed noise region starting at absolute phase
2.88 hours. Its [model-scope hold](../results/boi-validation/local-kic5079515-native-domain-hold-20261020.json)
also preserves the MAP writer's rejection of two nonfinite initial-attempt entries;
the third initial geometry is finite, so this is not an all-infinite likelihood.
No MCMC or FPP was launched. A reviewed general long-duration native model is
needed before this case can resume; cropping the event or bypassing serialization
does not close that requirement. All 4,567 field rows paired, every background-host
domain check passed, and the catalogue/source data remain preserved.

The neighbor Gaia DR3 2103418657935149952 has two TIC aliases. Official TIC v8.2
metadata marks 1716376627 as DUPLICATE and points to corrected SPLIT entry
120255477, with identical Gaia identity, astrometry and stellar fields. Both raw
records and all Gaia sources remain preserved. The inventory explicitly resolves
that pointer; it does not count a duplicate catalogue identifier as another star
or use potentially blended original 2MASS photometry. The
[MAST release notes](https://outerspace.stsci.edu/spaces/TESS/pages/111023212/TIC%2Bv8.2%2Band%2BCTL%2Bv8.xx%2BData%2BRelease%2BNotes)
describe this correction. The target's own identity is unique.

KIC 3935011 completed all four native endpoints with the reviewed ten-worker
runtime. At 24,000 steps per ensemble, Rhat 1.02667 and epoch ESS 467.92 pass,
but radius-ratio median/std 2.86720 does not exceed three. The
[native hold](../results/boi-validation/local-kic3935011-native-hold-20261020.json)
preserves every chain and endpoint. No timing was adopted, no extra sampling is
allocated, and no FPP run was started. The queued diagnostics correctly exited
without starting a stage. This is unresolved precision, not a reproduced false
positive. Its population pairing and all 27 stellar routes passed; that evidence
does not waive the native precision gate.

KIC 3322939 passed its native fit at 12,000 steps: Rhat 1.01795, epoch ESS 322.02
and radius-ratio SNR 6.59. Its six-quarter pixel review also passed. However,
none of its 5,100 Gaia field-population rows matches a Kepler physical tuple.
The strict crossband constructor stopped before any probability plan, freeze or
production run existed. Both authenticated outputs, failed logs and native chains
are preserved in the [input hold](../results/boi-validation/local-kic3322939-input-hold-20261020.json).
The cause of the independent remote realizations is unestablished. Do not redraw
populations to pass or replace exact pairing with an approximate match. A changed
population or radiometry route requires review and separate revalidation. Its
grazing body-size degeneracy and sparse bright-neighbor priors remain unresolved.

KIC 8805426 is held for a reproduced background-model boundary. Its two seeds
each preserved 13 completed scenario families, then stopped in BEB because a
background star has log g 9.57, beyond the frozen BC maximum 9.5. Exact saved-RNG
replay reproduces both failures. There is no complete FPP result. Preserve the
[model hold](../results/boi-validation/local-kic8805426-model-hold-20261020.json)
and every checkpoint; do not resume, clip, redraw the population or drop the star.
A scientific amendment requires explicit review and revalidation. Its accepted
native fit and reference checks remain preserved, without planet admission.

KIC 10196617 stopped after its first two complete primary integrations, with FPP
0.5384760920 and 0.6568040464. Both exceed the predeclared 0.10 allocation limit.
The [result](../results/boi-validation/local-kic10196617-futility-20261020.json)
preserves both repeats and every earlier generation. Do not resume the remaining
18. This is nonadmission and `REMAINS_PC`, not a reproduced stellar mechanism.

KIC 7300018 is held for unsupported neighbor metallicity particles. Preserve its
partial native chains; do not clip its prior or omit the neighbor. KIC 3330404 and
KIC 7729457 are deferred for coherent host/distance modeling. These holds do not
deny a planet. Adam abandoned KIC 8540098; no further work is authorized there.

For eligible new probability campaigns, run the first two predeclared seeds.
Stop if both FPP values >= 0.10 or both NFPP values >= 0.01, preserving all results.
Otherwise finish all twenty, including the first two. Admission requires all
twenty and unchanged numerical and physical checks. Share the reviewed ten-core
budget within this one BOI. Monitor about hourly and wait before advancing;
do not run competing candidates or unrelated numerical work.

## Execution and record

- Workspace: `other/boi-validation-20261017/local-loop-20261020-v1/` on the Mac's
  already authorized internal campaign volume. External light-curve discs stay
  read-only. No server compute or server job management is authorized here.
- `loop-plan.json` records the initial eight preparation priorities and points to
  the complete [retained queue](../results/boi-validation/boi-server-priorities-20261019.json).
  `loop-status.json` is operational state, not a scientific admission.
- Each source directory retains original photometry, query responses, spacecraft
  diagnostics, native inputs, logs and frozen receipts. Never overwrite a failed
  diagnostic with a corrected one; record a new version and its recovery cause.
- Native sampling uses an independent launchctl Interactive job and the retained
  Python 3.12 environment. Four ensembles of 24 walkers use their predeclared
  seeds and 3,000/6,000/12,000/24,000-step endpoints. Preserve each endpoint.
  These native-fit steps do not add or replace any of the twenty FPP repeats.
- Before recovery, check the label and actual full process arguments. Allow
  surviving workers to finish; preserve state and exact chain/RNG progress.
  An exited supervisor is not permission to restart live workers or reseed chains.

## First candidate

KIC 8690308 is Gaia DR3 2126585921988564224, TIC 63208643. Its parent pixel file
belongs to KIC 8690292. The exact recovered period is 39.52711869808169 days.
The [snapshot](../results/boi-validation/local-loop-20261020.json) records the
current candidate and preserved reviews; the first source's status is
`2126585921988564224/native-supervisor-status.json`.

All 18 parent products passed full stored-HDU checksum and identity checks;
nine also match retained frozen pixel hashes. Native photometry retains the dip
in all four seasonal channels. One Q16 window contains a rolling-band flag;
the original quality mask excludes that cadence. Paired local pixel diagnostics
show no material row/column/uniform preference, but that single event does not
identify a unique host. Keep this and the quarter-response review explicit before
FPP production. The initial conditional native mode is grazing; it is not an
adopted radius or probability result.

The launchctl label is
`local.spacelab.job.boi-native-2126585921988564224-20261020`.
The supervisor preserves each endpoint, checks unchanged native source hashes,
runtime pins, baseline parity and exposure integration, and stops before FPP
production for the candidate-specific readiness review.

The native posterior passed the declared checks at 12,000 steps: maximum rank
Rhat 1.02095, reference-epoch ESS 452.98 and radius-ratio median/std 3.38565.
The retained stricter diagnostic did not pass; both definitions remain saved.
The posterior acquisition-quadrature check passed.

Conditional quarter timing profiles then found material departures from a single
linear clock: Q3 prefers -19.5 minutes and Q7 +25.5 minutes relative to that fit.
These fixed-shape/noise profiles are not calibrated TTV significances, but the
timing adequacy is unresolved. `readiness-hold.json` retains the reason, grazing
radius limitations and sparse neighbor population support. No probability run
started. The source remains a candidate pending timing/source investigation;
three recovered-clock probability runs cannot repair an inadequate timing model.

## Second candidate

KIC 8540098 is Gaia DR3 2105418399005125376, TIC 123417514. The parent pixel file
belongs to KIC 8540105. Original photometry and all 18 parent products passed
integrity checks. The catalogue refresh retains a distinct 524.217-day parent TCE
alongside the target's 6.950815509597985-day signal; it is not an exact signal
match or a low-order alias.

The conditional native fit completed on the Mac under
`local.spacelab.job.boi-native-2105418399005125376-20261020`, using the same checked
kernels and predeclared sampling plan. Its status is
`2105418399005125376/native-supervisor-status.json`. Fresh inputs include the
complete 27-source scene, four seasonal PRFs, dust and field populations.
Q10 stacked pixel diagnostics favor the target PRF over the tested spatial
artifact patterns. The isolated flagged-event comparisons also cross no retained
material artifact threshold. This review does not measure false-alarm probability
or identify a unique host. The accepted native fit and conditional quarter timing
review supported its numerical census. Adam subsequently abandoned this case; no further closeout or recovery is authorized.

The bright aperture-owner neighbor has a supported TIC bank. Several faint
neighbors have sparse population support; a predeclared tenfold-area population
comparison is retained without adopting it for production. The original plan's
misidentification of the sparse star as the bright neighbor is explicitly corrected
in `population-discretization-identification-correction.json`; all identities and
bank arrays remained correct. No FPP was used to select these input diagnostics.

## Preserved KIC 10196617 readiness

KIC 10196617 (Gaia DR3 2130658616133595264, TIC 158322476) is frozen for twenty
final-clock primary runs and zero recovered-clock diagnostics. Its native fit
passed at 6,000 steps. The maximum clock shift is 6.85 minutes, against a
29.42-minute exposure and 202.17-minute fitted duration. Modest conditional
quarter timing offsets remain explicit model-adequacy diagnostics.

The bright neighbor KIC 10196630 initially lacked a TIC mass and received an
effectively single empirical prior particle. That hold and the draft generation
remain preserved. An authenticated exact Gaia/KIC match in APOKASC-3 Gold supplies
a coherent measured mass, radius, temperature and metallicity. The same retained
seismic prior constructor now gives 4,096 supported particles and a median
physical Gaia-G residual of 0.019 mag. Missing joint covariance and the measured
alpha enhancement on the retained scaled-Solar BC grid remain limitations.

All 96 families passed direct-reference comparison with posterior L1 5.84e-13.
`2130658616133595264/probability-v5/` binds 865 dependencies and supersedes the
draft readiness hold for numerical calculation. The [queued record](../results/boi-validation/local-loop-next-20261020.json)
names its freeze and launcher. Its first two primary seeds subsequently completed
and crossed the allocation limit; the terminal futility outcome is recorded above.
Physical planet admission remains separate from numerical passage.

The preserved v2 cold-worker check found a missing adapter-directory import path.
V3 changes that operational lookup and output paths only; all seeds, configurations,
prior mathematics and packet bytes remain identical. It passed repeat reference
checks and the actual runner's module/context/QoS check from its runtime directory.
Test new adapters from the worker's working directory before declaring them ready.

V4 retains that science and fixes static-result serialization to match the original
driver: sampled `target_kp_mag` stays runtime state, not a static model parameter.
The frozen allocation supervisor implements the prospective two-run budget rule.
The preserved previous generations must not be launched.

V5 preserves all 61 completed families from each first seed, their original weight
bytes and exact RNG states. It shares a ten-core budget across active repeats:
two runs receive five Numba threads each, with tail allocations redistributed.
All 96 families passed bitwise weight/bound parity at every thread count 1–10.
All scientific kernels remain unchanged. V4 is intentionally stopped and must
not be resumed; its outputs and transition records remain preserved.


## Preserved KIC 8805426 input recovery and native runtime

All 17 parent products passed checksum and identity checks. Exact Gaia/TIC
identity and the dated catalogue queries are saved. The released instrument rule
finds no material conflict in retained event evidence. Adopted-clock stacks of
all 17 quarters favor the target in 15 quarters; none crosses the retained
artifact BIC margin. Noisy Q1/Q2 other-source preferences and missing early/late
rolling-band metrics remain explicit. All 18 sources remain in the census.
Covered checks do not supply an instrumental false-alarm probability.

Two detached input readers stalled inside mounted-volume file opens. Their logs,
process samples and exact internal SQLite index copies are preserved. The same
reader succeeded in the authorized foreground tool context, verifying the raw
FITS hash and all eleven stitched and seven native column arrays. The continuation
verified that completed acquisition and ran only the remaining input stages.
No volume permissions, original photons or integrity criteria changed.

The native runtime evaluates the original likelihood in ten persistent processes
and returns proposal results in order. Two independent 64-step chains followed
by 16 resumed steps matched the serial reference exactly, including coordinates,
log probabilities and saved RNG states. The measured sampling time improved from
18.47 to 3.94 seconds. The review also binds input arrays, MAP, environment pins,
baseline and exposure checks, supervisor and posterior numerical check. Serial
initialization and diagnostics can still leave some cores idle.


## Full background-host support check

Before substantial native or scenario work, inspect every potentially selectable
background-binary host against the actual frozen flux domain. Preserve the
existing exact cold-binary branch. A short stochastic reference replay can miss
a rare unsupported particle, as KIC 8805426 demonstrated. Record the unsupported
row and hold the case for model review; never prune or redraw the population.
The read-only preflight changes no scientific prior, input, seed or draw count.

Also require every Gaia field-population row to have exactly one native Kepler
physical-tuple counterpart before expensive native fitting. A completed download
and valid stellar-domain scan do not establish crossband identity. Preserve
unmatched or ambiguous outputs and hold the case without resubmitting to pass.

Check the actual native preparation and likelihood scope too. The retained kernel
supports durations from 0.0104 to 0.25 days, uses fixed windows of +/-0.3 days and
estimates noise outside absolute phase 0.12 days. An original event extending into
that noise region or beyond the model duration cap needs a reviewed derived native
method. This is a software-support hold, not an astrophysical exclusion or a new
radius/duration veto. No documented scientific justification was found for six
hours as a universal cutoff; carrying that earlier implementation choice across
the BOI queue was a preparation-design error. Preserve the signal and original
inputs while a general duration-aware method awaits review and revalidation.
