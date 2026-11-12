# Forgotten Stars BOI survey protocol

Search the frozen 406,547-source KBonus LC2 cohort. Original inputs stay read-only.
Freeze source identities, input hashes, code, configuration and catalogue snapshots;
retain accounting for every ineligible, scanned, queued and vetted source. Generated
inputs, pixels, checkpoints and evidence belong under `dump/<run>/`.

## Discovery and measurements

The original PSF_FLAT run completed on 10 October 2026 with Rust periodic and
sparse-event search plus shallow BLS. Its results are sealed and retain that scope.

Adam authorized the remaining full PSF and SAP runs on 9 October with the explicit
\`rust-transit-only-v1\` profile: only final main Rust TRANSIT_LIKE proposals with
at least three detected transits enter validation. Shallow BLS and separate sparse
EVENT_LIKE emission are disabled; SYSTEMATIC_LIKE survivor-review proposals are
excluded individually, including those sharing a source with an admitted transit.
Keep all periodic core, extension and residual searches, quality and flattening
masks, normalisation, grids, thresholds and scientific vetting unchanged. Each
view has its own complete pipeline run. The views are correlated measurements.

The original flat detector lanes shared one follow-up queue. Preserve every raw
row, detector classification, supplied event and hypothesis, including strong
systematic and rejected detections. Bounded surveys retain all Rust rows in
`detections.csv`, including sources that never enter the follow-up queue. Reconcile bounded native detector replays
against the frozen survey and preserve the shared survey-wide alert index.

Measure the original relative flux, exposure-integrated transit and nuisance
fits, orbital behaviour, instrument context, difference-image localization,
contamination, host properties and catalogue identities. Missing or conflicting
evidence remains explicit. Supplied sparse intervals may form repeat, alias or
TTV hypotheses; no new unrestricted period search is introduced.

## Final labels

A **BOI — Bonus Object of Interest** is a transit signal remaining ACTIVE after
all five ordered gates: shape, orbital, instrument, localization and contamination.
The BOI designation adds no threshold and does not change a measurement or fit.
Preserved flat results include single-event BOIs under the original protocol.
The PSF/SAP continuation requires at least three detected transits in the admitted
main Rust proposal. One passing photometry view does not mean one transit event.

The first non-PASS gate determines HELD or DISMISSED, including its stage and
reason. Later measurements in record mode preserve that first failure. A BOI in
one view remains eligible when another view is HELD or DISMISSED. One passing
view is sufficient; agreement between views is not required.

Keep detector classifications such as TRANSIT_LIKE, EVENT_LIKE, EB_LIKE,
SYSTEMATIC_LIKE and NO_SIGNAL alongside final signal labels. The complete
signal ledger makes retained, held and dismissed proposals available for later
follow-up. An astrophysical false-positive label requires a reproduced mechanism.
Reviews are attributed annotations and do not override the automatic gates.

## Catalogue and study boundary

New runs use one frozen catalogue search in the shared vetting for both detectors.
Check the complete cumulative KOI catalogue including false positives, DR25 TCEs,
confirmed-planet literature models, the current Kepler EB catalogue and the
published KBonus candidate ephemerides. Freeze retrieval receipts, independent
row counts, positions, time references and normalized-row hashes. Parent-aperture
KIC equality does not establish the Gaia target's identity.

At the existing contamination gate, a source-compatible catalogue signal with
coherent period and event timing is HELD as a recovery; a nearby timing association
is HELD pending source attribution. Retain both in the signal ledger and separate
export lists. Neither verdict is an astrophysical false-positive denial. Another
signal on a known host remains eligible when its ephemeris is not associated.
Genuine missing coverage
remains explicit and HELD; missing fields in one literature model can be covered
by another qualified model of the same object, with its row references retained.
Do not substitute an unqualified epoch or treat a centroid proxy as source
ownership. Unknown period alone does not veto a preserved single-event proposal.

This catalogue amendment applies to new run identities. Preserve earlier results
with their original code, configuration, catalogue bytes and gate histories.

Publish pipeline BOIs, their photometry views, signal identities, ephemerides,
measurements, gate histories, failure reasons and referenced evidence. Retain
all proposals and unavailable measurements; software completion does not establish
population completeness or catalogue reliability.

Link matching signals across views for counting and browsing without requiring a
match for eligibility. Preserve individual detections and uncertain identities;
correlated views and alternate ephemerides must not inflate a unique-signal count.

The study ends with the BOI catalogue and visual presentation of its measured
signals. BOI designates an object of interest. Physical validation, confirmation,
RV suitability and novelty require additional evidence and separate follow-up.
The number of real planets is not established by the pipeline designation.

Preserve the frozen benchmarks and their immutable evidence under server `dump`.
Control measurements may describe exact-signal recovery, attenuation, false-class
retention and limitations, without altering the frozen survey gates. Do not open
sealed control outcomes or add follow-up stages during release preparation.
