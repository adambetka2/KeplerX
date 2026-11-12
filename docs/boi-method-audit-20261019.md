# Statistical validation method audit — October 19, 2026

Our workflow is directionally aligned with the reference Bayesian validation
framework. **Full replication of the paper is not established.** Two concrete
closeout/enforcement gaps and a broader model-equivalence gap need attention
before extending the same claimed method to another candidate.

This review corrects the earlier description of KIC 4829494 as fully closed
under the reference procedure. Its 20-run mean-cut pass and every original
result remain preserved. This audit establishes no astrophysical false-positive denial.

The reference is [Collier et al. (2026), v3](https://arxiv.org/html/2608.21560v3),
with [v1](https://arxiv.org/html/2608.21560v1) checked for version differences,
and the [foundational TRICERATOPS study](https://arxiv.org/html/2002.00691).
The [structured audit](../results/boi-validation/boi-method-audit-20261019.json)
binds the inspected code, source receipts, controls and target records.

## Reference sequence compared with SpaceLab

The table covers candidate selection, observations and the complete method
sequence. The reference's imaging-removal and observability analyses are
separate study products. TESS-specific extraction and recovery choices need
Kepler adaptations; the final Bayesian model still needs a validated bridge.

| Step / reference section | Reference operation | Our implementation and assessment |
|---|---|---|
| P01 / II; III.1 | Select unresolved targets; authenticate source identities. | **adapted**. SpaceLab begins with a blind KBonus background-star search and a manually delivered 125-signal subset. |
| P02 / III.1 | Use uniformly processed, quality-filtered photometry. | **justified kepler adaptation**. LC2 PSF and SAP are different extractions of the same Kepler photons. |
| P03 / IV.1.1 | Remove trends without fitting away transits. | **adapted model needs external control**. Our discovery median filter and later local affine-baseline treatment differ. |
| P04 / IV.1.2 | Recover ephemerides; examine aliases and detection strength. | **directionally aligned different search**. The Rust event-seeded search and shallow BLS are intentionally different discovery algorithms. |
| P05 / IV.1.3 | Infer folded shapes and verify sampler diagnostics. | **adapted**. Our original vetting fits are optimized shape comparisons rather than posterior inference. |
| P06 / IV.1.3 | Refine timing when accumulated drift smears events. | **partial**. The pipeline preserves bounded timing refinements and follow-up fits period and epoch on native times. |
| P07 / IV.2.1 | Compare alternate events with independent noise treatments. | **different calibrated rule**. SpaceLab uses local excess-scatter and duration-bin noise inflation, then a weighted odd/even comparison with a BIC-gain cut of 10. |
| P08 / IV.2.2–IV.2.3 | Screen secondary eclipses; annotate grazing shapes. | **different rule no conflict in four checked cases**. Our survey secondary rule uses SNR at least 7 and depth ratio above 0.5. |
| P09 / IV.3 | Window and compress the validation light curve. | **adapted**. Our final original-clock packets have 200 phase bins, complete contributing baseline groups, and a fixed covariance projection. |
| P10 / IV.4 | Use pixel geometry to estimate source contributions. | **justified kepler adaptation with limits**. Kepler PRFs, proper motions, measured quarter masks and aperture responses replace TESS pixel geometry. |
| P11 / IV.4 | Assemble catalogued stars and simulated background populations. | **same concept different prior model**. Our Gaia/TIC scenes and TRILEGAL populations retain all modeled resolved neighbors and unresolved configurations. |
| P12 / IV.5 | Integrate competing planetary and stellar-eclipse configurations. | **directionally aligned not stock workflow**. The official pinned modules are genuine and unchanged. |
| P13 / III.2; IV.6.3–IV.6.4 | Use measured companion limits when available. | **aligned no fabricated constraint**. The studied targets have no adopted contrast curve. |
| P14 / IV.6.1 | Repeat independently and retain complete outcome distributions. | **aligned within adopted model**. The completed campaigns use 20 independent seeds and one million attempted draws per family. |
| P15 / IV.7 | Classify arithmetic means using the declared cuts. | **aligned**. We correctly compare the arithmetic means with FPP <0.015 and NFPP <0.001. |
| P16 / IV.7 | Check astrometry and unmodeled catalogued companions. | **aligned for studied targets**. The retained host RUWE values pass the declared 1.4 cut; current exact-identity companion queries are recorded. |
| P17 / IV.8 | Fit native-time parameters independently and report uncertainty. | **adapted diagnostics pass**. We use independent emcee ensembles, batman, proper local baseline priors and explicit stellar/noise assumptions. |
| P18 / IV.6.2; IV.7 | Recheck final timing with other inputs fixed. | **material gap**. The reference requires both clocks; this appears in both v1 and v3. |
| P19 / IV.7 | Review event coverage, timing ambiguities and prior claims. | **partial evidence retained**. Our aliases, event membership, coverage, source localization and exact-signal catalogue ledgers go beyond a bare FPP calculation. |
| P20 / IV.7–IV.8 | Derive sizes from the adopted physical model. | **different declared construction**. Our reported radii use scenario/dilution-weighted stellar distributions. |
| P21 / V.3; VI.2 | Study imaging impact and follow-up prospects. | **not a per candidate validation requirement**. An imaging-removal population experiment or JWST priority metric is not needed to establish our per-object numerical cut. |

## Findings and implications

### F01: Final-clock consistency is not closed

The current plans restrict the production census to discovery timing. The paper's final-clock requirement is present in both archived versions. Old passing final estimates are preserved, but matching their full model to the later freezes has not been demonstrated; the new target has no such census.

Earlier statements that final ephemerides cannot qualify the paper-based validation were too strong. Preserve all numerical outcomes and qualify the claim of complete paper replication.

Next: Prepare a separate reviewed final-clock generation with the adopted scene, priors, response, likelihood and all non-timing settings fixed. Bind usable historical checks where that identity can actually be established.

The requirement is explicit in [section IV.6.2](https://arxiv.org/html/2608.21560v3#S4.SS6.SSS2)
in both inspected versions. KIC 6699887 and KIC 8745833 have historical
passing original/final sensitivities, including different allocation and
model variants. Bind their exact relationship to the later adopted generation
before calling this current closeout complete. KIC 4829494 has only the
discovery-clock full census. A converged native period fit cannot substitute
for a competing-scenario calculation at that period.

### F02: The instrument triage gate checks availability, not its material-conflict assessment

A synthetic case with spatial-artifact BIC advantage 100 produces MATERIAL_CONFLICT from the released assessment, PASS from instrument_gate, and BOI from a complete otherwise-passing gate history.

The claim that this instrument assessment automatically blocks BOI assignment is not implemented. This gap does not demonstrate an artifact in a real target. All four studied targets separately reassessed as no material conflict.

Next: Enforce the existing assessment in a reviewed statistical-readiness layer and preserve original survey gate histories. Any frozen survey-code correction needs its own review and revalidation.

The real example is **KIC 9899369 / BOI-2080481745869519488-1**:
its spatial-null median BIC advantage is **47.8058**, above the recorded
margin of 10, while the original gate reason is `INSTRUMENT_MEASUREMENTS_AVAILABLE`.
Hold its future statistical-readiness work for an instrument/source review.
The original candidate disposition is retained. The separate current-rule
assessment finds no material instrument conflict for KIC 6699887, 8745833,
8878385 or 4829494. This is not a finding that those four are artifacts.

### F03: Custom model correctness is not established by internal replay alone

Seven vendored upstream Python modules exactly match the official pinned commit. The running engine nevertheless replaces the high-level evidence calculation and uses custom stellar banks, MIST/paired colors, quarter response, affine detrending covariance, importance sampling and numerical bounds. The 135-family gold comparison uses the same custom physical kernel on both sides.

The engine is directionally aligned and numerically reproducible. Those properties do not establish the same methods or empirical probability calibration as the reference pipeline.

Next: Add a matched-input stock-Kepler reference path on frozen known objects. First check identical physics/priors/likelihoods; then isolate and justify each deliberate amendment with paired, preserved results. Do not substitute current upstream defaults for the paper's explicit settings.

The seven vendored Python modules exactly match official commit
`d20875c27d6761349cf032728bb4a59ec732a84d` (September 2, 2026).
The official software already supports `mission="Kepler"`. Its current
default background source is Gaia; a stock invocation with defaults would
therefore not reproduce the reference's declared TRILEGAL configuration.
The paper does not identify an exact release of its complete wrapper pipeline
in the inspected article. Pin the library and every chosen setting explicitly.

Eight new bounded comparisons of our vectorized mass-ratio and bound-companion
prior functions against the official scalar implementations agree exactly
at the tested points. This provides a genuine external primitive check,
while leaving full-model empirical validation as a separate requirement.

Our altered likelihood has proper, fixed comparison constants and its
forward transformation reproduces the observed bins. This is useful algebraic
evidence, but not a known-object end-to-end probability-calibration result.
Likewise, cache/platform replay and saved-weight replay answer reproducibility
questions; they do not prove that two different astrophysical models agree.

### F04: Orbital screening is not the same decision procedure

The survey uses a BIC parity rule and a 7-sigma/0.5 secondary rule, with local duration-noise inflation. The paper uses different measurement/decision definitions. Existing measurements in the four studied cases show no nominal secondary or parity conflict in the checked views.

A BOI orbital PASS is not a certificate of exact paper-vetting agreement. Nor does a different calibrated rule by itself establish that the candidate is wrong.

Next: Retain the Kepler survey rules. Add a separately recorded reference-style cross-check for statistical readiness, preserving disagreements and incomplete coverage rather than relabeling old outcomes.

### F05: The newest probability estimate has weak effective-weight precision

KIC 4829494 has target-planet weight ESS 2.68–8.45 per repeat, maximum single normalized weight 0.569, and seven crossings for each probability. The mean margins are approximately 0.59 and 0.83 between-run standard errors.

The declared mean rule passes. It should not be described as a precise or robustly separated probability estimate. The omitted-likelihood bound does not bound sampling, grid, prior or model uncertainty.

Next: Preserve all 20 results. Diagnose proposal/support and numerical sensitivity under a reviewed new recipe, rather than selecting seeds or repeating until favorable. Keep precision diagnostics distinct from any admission criterion.

### F06: Evidence coverage is bounded, not uniformly complete for all 125 signals

The source-scene census only searched original PSF_FLAT records; nine delivered QDIAG signals remain unauthenticated by that scan. Exact-host catalogue and literature work has varying dates and depths. Published development controls and regression receipts do not measure global final-catalogue reliability.

These are scoped evidence gaps, not false-positive denials. They do not require blocking already authenticated first-batch sources while every other host is reviewed. This audit additionally found one selected QDIAG signal absent from an authenticated same-host PSF_FLAT record: host-level hash matching is insufficient for per-signal view authentication.

Next: Before each future candidate, authenticate its actual delivered view and exact signal, review alias-wide catalogues/literature and observed coverage, and record the relevant calibration boundary.

## Scope of the candidate record check

The read-only review examined 116 entries whose host file hashes matched the
retained PSF_FLAT survey records. It resolved **115 exact signals**: 114 have
no material instrument conflict under the separately applied rule, and one
has the KIC 9899369 conflict above. The four studied targets show no nominal
secondary or parity conflict in the checked measurements.

Nine delivered QDIAG records were already outside the earlier source scan.
An additional selected QDIAG signal, **KIC 7373485 / BOI-2077967368932890752-6**
at 0.6226501158 days, is absent from an authenticated same-host PSF_FLAT copy.
Thus ten selected signal/view bindings remain unverified in this audit. A
correct host file hash does not authenticate every signal or extraction on that host.

The reference-like secondary arithmetic flags BOI-2100442657917279616-1
(nominal SNR 3.519, depth ratio 0.221), which already remains contaminated/unresolved.
These are screens on retained measurements, not new denials or a reproduction
of the reference's full BLS plus red-noise measurement process.

## What remains correct

- FPP and resolved NFPP are normalized across the complete declared scenario census.
- The arithmetic mean rule, 20 independent seeds, per-family attempt counts and
  retained unfavorable repeats match the intended repeat protocol.
- The original two-target and new-target deep audits passed; no result, seed,
  draw count, scientific input or threshold was changed during this review.
- Proper long-cadence integration, exact source identities and measured aperture
  response matter for KBonus; directly substituting TESS products would be wrong.
- Native posterior convergence, numerical checks, host RUWE and companion
  evidence remain positive within their stated assumptions. The new native
  radius-ratio SNRs also pass the reference's quantile-width definition.
- Local duration-bin noise inflation exists in the survey orbital measurements.
  Its difference from the reference procedure should be measured, not erased.

The detector has published development recovery, opened controls and scientific
regression evidence. Those records support the implemented Kepler search; they
do not establish global final-catalogue completeness or reliability. This review
did not open unopened sealed benchmark outcomes.

## Before the next candidate

1. Record the current instrument assessment explicitly in validation readiness, with a preserved synthetic conflict control and real-candidate checks.
2. Build a pinned, matched-input reference comparison and document each deliberate Kepler likelihood/prior amendment.
3. Close final-clock consistency using exactly the adopted model; bind historical equivalents only when full identity is demonstrable.
4. Cross-check orbital measurements using the reference definitions and retain the existing calibrated survey decisions.
5. Review the new target's sparse-weight precision without changing mean cuts or selecting favorable repeats.
6. Authenticate the selected candidate's true view and complete its dated exact-signal/alias literature and companion record before launch.

The saved priority order remains a preparation recommendation. The new
readiness overlays identify the instrument case and view-binding case; they
do not rewrite the original queue or candidate ledger. Frozen-code changes
require review and a separate revalidated generation. No next candidate was launched.

No new universal radius cut, individual-repeat veto, confidence-interval rule,
mandatory new imaging or RV mass requirement is introduced. Numerical cut
outcomes, method adequacy, source identity and publication novelty remain
separate questions, with their evidence recorded explicitly.
