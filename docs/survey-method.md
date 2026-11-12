# Rust V6.1 search method

## Scope and frozen identity

The Rust detector is an event-seeded deterministic search derived from the
Forgotten Stars V1 archive. Its development protocol is
`fsc-kepler-deterministic-v6.1`. Each run searches one declared discovery view:
`psf_flat`, `psf_qdiag` or `sap_qdiag`.
The periodic lane uses a BLS-like statistic. The event lane retains intervals
without assigning a period or requiring three epochs.

The current discovery runner also runs the shallow BLS detector on the same
prepared input and sends both detectors into one shared vetting queue. Its
method and development limits are in `../pipeline/shallow/README.md`. The V6.1 controls
below apply to the Rust detector and do not validate the combined release.

The archive is bound to repository commit
`2850e1a9681311fb337e950408ca1e609332a381`. `run.py` retains the original
SHA-256 identities of `src/main.rs`, `Cargo.toml`, `Cargo.lock`, and the
configuration in every detector contract. The current executable,
configuration, input transformation, cohort, and output files receive separate
hashes for each run.

V6 has passed bounded development recovery and ordinary-source checks. These
checks do not establish population completeness or reliability. Detector outputs
are proposals; the five existing vetting gates determine BOI designation.

## Search

The detector subtracts a 97-cadence running median within each quarter and
divides the residual by that quarter's robust sigma for discovery only. It
finds local downward box excursions, retains at most 240 events, and proposes
periods from event-pair spacings and integer harmonics. Event coherence and 16
logarithmic period buckets retain at most 512 core proposals. Seven refinements
per proposal feed a binned two-level box search. The calibrated core retains at
most 20 period-distinct rows from 0.5 to 100 days using durations through 12
hours. Separately budgeted short- and long-period extensions each retain at
most 64 proposals and four rows; they cannot displace a core proposal or row.
The upper period bound is the lesser of 1,000 days and the light-curve baseline
divided by two, the theoretical three-epoch limit. Extension periods may use
duration templates through 24 hours. Periodic `search_mes` records the
statistic on the whitened discovery copy. `mes`, every reported depth, and the
remaining candidate diagnostics are measured on untouched relative flux.

After an initial periodic candidate clears the dip-versus-brightening control,
the detector runs one residual pass. It removes discovery samples within one
fitted duration of each predicted transit centre, reruns the same proposal and
box search, and retains at most four period-distinct rows not already present.
Those rows carry `search_pass = 2`; the original rows remain immutable and keep
their separate budgets. Residual search is not triggered by a non-surviving
ordinary-source peak.

The event lane retains at most 10 high-scoring intervals per source. An event
must contain at least three long-cadence samples, exceed S/N 8 in the whitened
search and in untouched flux, and be a dip relative to both the preceding and
following local baselines. Its two-sided MES must exceed the strongest
two-sided brightening by a ratio of 1.25 and an absolute difference of 3.
Quarter rotation is not used as a sparse-event null because rotation preserves
an isolated event. Periodic windows supersede overlapping sparse windows only
when the detector-to-vetter queue is built; off-ephemeris events remain.

This is not exhaustive BLS or Kepler TPS. `config.json` is authoritative for
all numerical limits. Its periodic search range is:

- Period: 0.2 days to the three-epoch limit, capped at 1,000 days.
- Duration: 0.5 to 24 hours; the preserved 0.5–100-day core uses at most 12
  hours.
- MES-like proposal threshold: 8.
- Minimum coverage: three epochs across two quarters.
- Depth: 40 to 50,000 ppm.

Periodic `TRANSIT_LIKE` also requires local transit contrast, consistent odd and
even events, no strong phase-0.5 secondary, limited red noise and event
dominance, and separation from brightening and phase-scramble controls.
Survivor-only tests check event-duration consistency and leave-one-quarter-out
prediction. Untestable survivor checks remain `NOT_TESTABLE`; they do not
silently reject the signal. Phase-scramble searches run only after a signal
survives the brightening control; this short-circuit does not change any
`TRANSIT_LIKE` result.

## Input views

Prepare four correlated measurements from the same pixels. Run independent
discovery and vetting for `psf_flat`, `psf_qdiag` and `sap_qdiag`. Within each run,
also measure supplied ephemerides in the other available views:

| View | LC2 values |
|---|---|
| `psf_flat` | `cadence.parquet`: flattened PSF flux and error |
| `psf_qdiag` | `qdiag.parquet`: corrected PSF flux and error |
| `sap_qdiag` | `qdiag.parquet`: SAP flux and error |
| `psf_nova` | `qdiag.parquet`: mean-PSF flux and error without scene-motion correction |

For every view, retain finite timestamps, finite fluxes, positive finite
errors, and cadences with `SAP_QUALITY == 0`. Apply `FLATTEN_MASK == 0` only to
`psf_flat`. Normalize each quarter by its median, preserve negative
excursions, remove only upper excursions above 12 robust sigma, sort by time
and cadence, and reject duplicate cadences.

Do not sum view scores or treat views as independent confirmations. Preserve
every discovery row and every forced-view measurement. Discovery-view
`TRANSIT_LIKE` and `EVENT_LIKE` rows enter follow-up. Shared vetting reconciles
overlapping periodic and sparse windows, retains unresolved timing hypotheses,
and evaluates available extraction and pixel evidence. One view passing all five
gates is sufficient for BOI designation; a held or dismissed signal in another
view does not veto it. Cross-view matching links detections for counting and browsing.

KBonus photometry definitions follow
[Martínez-Palomera, Hedges, and Dotson (2023)](https://arxiv.org/html/2310.17733)
and the [KBonus data specification](https://archive.stsci.edu/hlsp/kbonus-bkg).

## Control-audit boundary

The periodic configuration was selected using confirmed Kepler hosts and a
1,000-source ordinary KBonus control. Those data are development evidence, not
a blind release test. V5 event thresholds reuse the periodic S/N and
dip-versus-brightening gates. Any future independent control audit requires a
parent-disjoint frozen cohort; it is not an additional BOI designation gate.

Keep sealed control outcomes closed until their cohort, truth and audit scope
are frozen and Adam separately authorizes opening them. Survey catalogue checks
remain part of the current pipeline.

The V2 release-control plan hash-orders sources that do not occur in any opened
planet, false-positive, ordinary, 50,000-source, 10,000-source, or V1 release
pool. It freezes ordered pools of 15,000 ordinary, 25,000 periodic-injection,
4,000 sparse-injection, and 5,000 strong-TTV source identities. Materialization
accepts the first eligible identities in pool order until it reaches targets of
5,000, 5,000, 1,000, and 1,000, respectively. Separate source-pure KOI pools
reserve 500 catalogue candidates and 500 catalogue false positives from 1,000
ordered KIC identities each; the ten candidate KICs inspected in the abandoned
V1 probe are excluded.

Periodic injections use exposure-integrated physical transit shapes with
predeclared period strata of 0.2--0.5, 0.5--100, and 100--500 days. Radius
ratio, stellar density, impact parameter, phase, and limb darkening are drawn
without using detector output. Strong-TTV controls displace every event by a
deterministic sinusoid of one to three transit durations; sparse controls place
one physical event on an observed interior cadence. The generator records
truth, eligibility failures, base-input hashes, code hashes, dependency
versions, and compact additive flux deltas. An interrupted materialization may
resume only when that identity is unchanged.

Flux-level injections measure discovery and flux-vetting behavior, not pixel
localization completeness. Catalogue candidates test real-signal regression
but are not confirmed-planet truth. The full local confirmed-planet cohort was
opened during development; it remains useful regression evidence but cannot be
represented as an independent release audit.

Repeated-event vetting associates the immutable detector alerts before fitting.
It selects one central event unit per observed quarter and then evenly spaced
units in time, up to 64 for the joint shape fit and 18 for engineering
diagnostics. The fit remains tied to the detector-supplied timing hypothesis;
sampling only limits repeated measurements and does not create a new search.
A periodic ephemeris that passed the legacy detector gates but failed only the
duration-consistency or quarter-leave-one-out survivor check enters a distinct
`SURVIVOR_REVIEW` lane. Its `SYSTEMATIC_LIKE` classification and failed check
remain immutable. The lane permits complete shared vetting to adjudicate the
signal; it does not relabel the row, weaken Bonus Object of Interest requirements, or
change detector-only survey proposal counts.
A dip-like signed box is a transit-family approximation, not a competing
false-alarm model. Out-of-eclipse variability is evaluated on the canonical
PSF-flat view with a [Kepler DR25 SWEET-style](https://arxiv.org/abs/1710.06758)
test. It fits coherent sinusoids at half, equal to, and twice the supplied
period across quarter-normalized flux, with independent quarter baselines and
shared phase and amplitude. A stellar-eclipse conflict requires a supplied
period below 10 days, sinusoid S/N above 50, amplitude above 5,000 ppm, and
amplitude below the measured transit depth. Weak or quarter-incoherent
variability is not an eclipsing-binary mechanism.

In a ten-source physical-injection diagnostic, six of eight injections with
expected white-noise MES at least 8 were recovered at the exact period. All six
passed shape, orbital, and instrument gates and stopped at localization because
the flux-only control deliberately leaves pixels unchanged. The complete run
retained 3,418 input alerts and took 141.8 wall seconds, 237.4 CPU seconds, and
619 MB peak memory. This small opened diagnostic does not measure release
completeness or reliability.

The frozen release materialization contains 5,000 periodic, 1,000 sparse, and
1,000 strong-TTV injections plus a separate 5,000-source ordinary control.
There are 3,070 periodic injections with expected white-noise MES at least 8.
Materialization evaluated 35,000 pool rows in 1,147.1 wall seconds and used
847 MB peak memory. These counts and resource measurements establish the
control input only; recovery is measured by the subsequent frozen detector
and vetter runs.

Pixel-level morphology controls use a separate parent- and raw-input-disjoint
partition. It reserves 40 opened development and 160 unopened evaluation
identities for each of the sparse-event and strong-TTV morphologies at expected
white-noise MES of at least 8. The evaluation size measures recovery near 80%
to a 95% Wilson half-width of about 6.2 percentage points. Flux eligibility is
not sufficient pixel eligibility: a source must also have a public
quarter-matched TPF with material PRF support at its propagated Gaia position.
Resolve that structural support without reading detector, vetter, or
localization outcomes before treating a morphology partition as a final
pixel-level control. Missing pixel support is an observational exclusion, not
a detector miss or a false-positive disposition.

The support preflight is bound to the partition freeze, base flux-injection
materialization, implementation, Gaia query, selected source order, public TPF
bytes, and PRF calibration bytes. It writes one atomic, resumable record per
source. Only exhausted quarter-matched public-pixel searches are recorded as
structurally unavailable; transport, checksum, identity, calibration, and
implementation failures stop the run. This keeps observational eligibility
separate from all recovery and classification outcomes.

The complete opened development preflight found material public-pixel support
for 39 of 40 sparse identities and all 40 strong-TTV identities. The TTV set
covered 436 quarter products. One TTV origin parent placed the propagated Gaia
position more than four pixels outside every required stamp; applying the same
two-pixel boundary to origin and spatial parents recovered complete support
from nearer public TPFs without changing the PRF criterion. The unsupported
sparse identity is an observational exclusion.

The final support candidate preflight checked 220 identities per morphology.
It found 219 supported sparse sources across 219 quarters and 219 supported
strong-TTV sources across 2,578 quarters. Independent warm replays reproduced
all source, Gaia, support, preflight, and progress artifacts byte-for-byte. The
support-qualified freeze then selected 40 development and 160 evaluation
identities per morphology, rejected one otherwise eligible TTV trial for an
actual resolved-parent collision, and retained 19 sparse plus 18 TTV supported
reserves. The 400 selected controls use 404 distinct resolved parents with no
sharing between trials or overlap with the periodic development parents.
Evaluation detector and vetter outcomes remain unopened.

## Development evidence

The 2026-09-20 replay used only the primary PSF-flat search. It recovered an
ephemeris match for 1,548 of 2,716 testable confirmed Kepler signals (57.0%);
975 (35.9%) were `TRANSIT_LIKE`. The preceding four-view V3.1 audit recovered
1,073 of 2,720 at any class (39.4%) and 811 (29.8%) as `TRANSIT_LIKE`.

On 1,000 ordinary KBonus sources, 23 sources produced 69 `TRANSIT_LIKE` rows after
all detector controls. The optimized detector completed that cohort in 124.5
seconds on the development machine, which extrapolates to about 14 hours for
406,000 primary-view searches. Preparation, forced-view measurement, and
shared vetting are excluded from that extrapolation. These opened cohorts are
development evidence and cannot enable automatic Bonus Object of Interest assignment.

The 2026-09-22 V5 event replay recovered both S/N-qualified sparse signals in
the three-signal Kepler holdout. On 918 ordinary holdout sources it emitted 226
event rows from 141 sources; the combined periodic and event detector took
136.4 seconds on six threads. On 141 catalogued-false-positive sources it
emitted 220 event rows from 56 sources. These are detector workload rates, not
false-planet rates.

The V6 domain probe preserved the controls-off search-100 core and raised
0.5–100-day held-out recovery from 532/687 to 542/687 through additional
accepted harmonics. On 47 held-out signals above 100 days, active-depth
`TRANSIT_LIKE` recovery rose from 6 to 10 and exact-period recovery from 0 to 10.
The 567-source broad-domain replay took 103.8 wall seconds and 143 MB peak
memory, compared with 80.3 seconds and 135 MB for its core-only comparator.
These opened development controls do not close the release gate.

On the frozen 200-source ordinary probe, V6 took 37.3 wall seconds and 64 MB
peak memory on six threads. V5 and V6 each emitted `TRANSIT_LIKE` rows from two
sources and sparse rows from 39 sources. Linear extrapolation is about 21 hours
for detector-only search of 406,000 sources on that setup.

The frozen V6.1 throughput pilot processed 10,000 unique eligible KBonus
sources in 20 batches on a 10-core M1 Pro. Detector scans used 1,880.0 seconds;
the batch median was 94.9 seconds, the p95 was 130.5 seconds, and total
preparation plus scan time was 1,989.9 seconds with 714 MB peak resident
memory. The corresponding 406,000-source projections are 21.2 detector hours
and 22.4 preparation-plus-detector hours. Cadence-normalizing against the
earlier 50,000-source cohort also gives 22.4 detector hours. The exact pilot
cohort averaged 25,260 retained cadences per source, so these are measured
planning estimates for this machine and frozen implementation, not universal
runtime guarantees. Artifact hashes and every source, row, classification,
and proposal count reconciled.

The pilot placed 130 sources in the periodic proposal queue and 1,670 in the
sparse-event queue; 54 appeared in both. These 1.3% and 16.7% source rates are
downstream workload estimates, not false-BOI rates or candidate dispositions.

The survey runner commits complete batches with cumulative source and row
counts and the size of every output table; a resume drops bytes beyond those
sizes and reruns the interrupted batch. An interrupted run can resume only when
its detector contract, implementation hashes, selection, batching, thread
count, frozen executable, and frozen configuration still match. A 100-source
interruption test retained the first 50-source checkpoint and reproduced the
uninterrupted cohort and proposal files byte for byte.

On 133 held-out multi-signal host light curves, the staged residual pass raised
ephemeris recovery from 198/331 to 205/331, `TRANSIT_LIKE` recovery from 86 to 90,
and exact-period `TRANSIT_LIKE` recovery from 79 to 84. Runtime rose from 99.4 to
174.8 wall seconds on six threads. On the frozen 200-source ordinary probe,
only six sources triggered the second pass, no additional source became
`TRANSIT_LIKE`, and runtime was 45.3 seconds in the final paired run.

An end-to-end probe selected 32 ordinary sources from a 200-source event
screen. Shared vetting dismissed 13 through a preferred non-transit shape;
18 remained unresolved at the shape gate. The only event that passed shape,
orbital, and instrument gates was dismissed after network-enabled pixel
localization reproduced a multiple-source conflict. This selected probe had no
all-gate survivor, but it is too small and selection-biased to establish
population reliability.

Catalogue timing is assessed across the complete associated signal. One
identified catalogue source must match at least two event units and at least
half of the signal before timing alone establishes a competing-source
conflict. An isolated timing coincidence remains a diagnostic; it does not
reproduce a contamination mechanism. Concurrent neighbour flux and an
explicitly established mechanism remain independent material conflicts.

Exact-target catalogue identity is assessed separately. A coherent match to a
catalogued eclipsing binary or false-positive signal contributes to the
contamination gate, but does not by itself assign a final false-positive
disposition. A high-fraction association supported by only one event remains
unresolved. Known planet and candidate entries are retained and handled by the
novelty layer. This separation follows the role of ephemeris matching in the
[Kepler Robovetter](https://arxiv.org/abs/1401.1240) and its parallel use with
centroid and stellar-eclipse tests in
[DR25](https://arxiv.org/abs/1710.06758).

Difference-image PRF centroids are fitted over the full physical stamp
footprint, from half a pixel before the first pixel centre through half a pixel
after the last. Restricting a centroid to pixel-centre indices clipped a real
pixel-level injection at the edge row in four quarters and manufactured a
coherent source offset. The repaired fit retains the same PRF likelihood,
pixel regression, residual-block covariance, quality qualification, and robust
quarter aggregation; it changes only the valid centroid domain.

The automatic instrument assessment also compares each difference image with
uniform, row, and column patterns. It records the median penalized model-score
advantage over the fitted source PRF across represented event units. A strong
instrumental spatial preference blocks automatic Bonus Object of Interest assignment;
missing pixel-shape evidence remains unassessed. Its calibration evidence comes
from opened development controls, without an independent reliability measurement.

## Known limitations

- Period refinement can accumulate about 10 hours of timing drift at the
  longest baselines and narrowest events.
- The 2,048-bin cap cannot represent the shortest requested durations
  separately at the longest periods.
- Detection uses a robust quarter scale and equal point weights. Supplied
  errors do not define the detection statistic, and the red-noise ratio is
  diagnostic.
- Covered epochs can contain sampled but flat points. Missing odd/even or
  secondary coverage can produce zero-valued diagnostics.
- `EB_LIKE` records threshold evidence; it does not reproduce an eclipsing
  binary mechanism.
- The 20 core rows, up to four rows from each extension, and four residual rows
  can include harmonics of one event train. One residual pass is not an
  exhaustive multiplanet search.
- Sparse discovery has only three parent-disjoint Kepler controls, of which two
  are S/N-qualified. Its completeness is not calibrated.
- The robust periodic localization rule requires three quality-qualified
  quarters. Single events use a single-quarter rule calibrated on opened
  pixel-planet and synthetic contaminant quarters, not on sparse events.
  Repeated-event signals with fewer than three good quarters, and single events
  without a qualifying quarter, remain unresolved rather than ineligible.
- A catalogued false-positive parent can emit additional unmatched hypotheses.
  Retain those hypotheses and report their source-level control outcomes;
  a BOI designation does not establish reliability.
- Adjacent event spacings preserve repeat and possible TTV hypotheses, but the
  detector does not run an exhaustive QATS-like search.
- The ordinary-source event queue is intentionally broad: 15.4% of the frozen
  holdout sources emit at least one interval before shared vetting.
- Inverted and rotated controls do not establish real-data recovery or
  population reliability.
- Forced views share pixels and systematics; they are consistency diagnostics,
  not independent trials or confirmations.

Transit-grid sampling follows the concern described by
[Ofir (2014)](https://arxiv.org/abs/1307.7330). Correlated-noise limitations
follow [Pont, Zucker, and Queloz (2006)](https://arxiv.org/abs/astro-ph/0608597).

## Claim boundary

`TRANSIT_LIKE` and `EVENT_LIKE` are detector proposals, not Bonus Objects of Interest,
validation, or confirmation. Neither Rust `search_mes` nor `mes` is Kepler
MES, a Gaussian significance, or a false-alarm probability. Multiple view
detections are not independent confirmations. A completed batch, fixture
success, or null result does not establish completeness, reliability, novelty,
or the absence of planets.

A signal passing all five current triage gates receives a BOI designation. Any
one discovery view is sufficient. Correlated view detections are linked for
catalogue browsing and counting, without requiring corroboration or changing
the gate history. Unknown-period single events remain eligible.

Representative parent-disjoint controls describe exact-signal recovery and
false-class retention; they do not change this study's BOI endpoint. Flux-only
injections cannot establish pixel localization performance. Preserve inputs,
calculations and thresholds. Physical validation and confirmation are separate
follow-up work. See [the survey protocol](survey-protocol.md).
