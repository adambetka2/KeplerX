# Standard post-BOI intake: 34 restored Rust and 70 shallow groups

The PI authorized standard readiness work and conditional queue admission for
all 104 groups on October 26, 2026. The schedule includes exactly 34 restored
Rust groups and the 70 shallow groups without matching Rust/M/K clocks from the
cheap screen. They cover 100 sources. New evidence uses the authorized internal
disc under `other/boi-v3-20261025/intake-104-20261026/`.

The [schedule](../results/boi-validation/standard-intake-104-20261026.json)
preserves each supplied signal and clock, all original discovery proposals and
views, and the original event units through the frozen catalogue binding. Its
selected discovery views are 98 PSF_FLAT, three SAP and three PSF. There is no
SAP-S/N admission veto. A weak SAP diagnostic does not remove a PSF_FLAT BOI;
source-corrected native photometry requires an authenticated flux convention,
dilution and source response through the standard preparation path.

## Preparation and queue

The strongest seven shallow signals lead the schedule. One sequential worker
queries fresh Gaia scenes and exact SIMBAD aliases/bibliography. Candidate
catalogue/literature evidence must be reviewed before substantial allocation.
Scoped absence is `NO_MATCH_IN_QUERIED_SOURCES`, never a novelty claim. Recorded
same-source subharmonics need event-independence review before a new campaign.

The acquisition adapter accepts the exact frozen proposal from the explicit
intake manifest instead of requiring membership in the historical manual
delivery. It verifies the original MAST file hash. Original pixel/SAP aperture
replay, physical scene construction, PRF responses, quarter closure, native
fitting, noise and population tools are retained. Missing reported stellar
uncertainty needs review, not an automatic uncertainty replacement.

Prepared cases append atomically to the existing `queue-4.json`; the existing
coordinator controls native/FPP compute slots. No second fitting coordinator is
introduced. Every new case has an explicit post-native scientific-review
checkpoint before FPP allocation. Complete measured detrending transfer,
convergence, timing/aliases, prior boundaries, native body radius, exposure,
noise, source/aperture/scene and exact-signal review before accepting that record.
This is a review within the authorized task, not a new approval
ritual. Existing studies without this intake checkpoint follow their original
declared orchestration. No native/FPP scientific threshold was changed.

## First pass

Three signals cleared preparation and entered the native queue:

| Host | Period, days | Run-scoped signal | Snapshot state |
| --- | ---: | --- | --- |
| KIC8573780 | 3.446824 | shallow-v1-2079714428492352768-1 | Native fitting |
| KIC10528996 | 13.744702 | shallow-v1-2127936912539397632-1 | Queued |
| KIC10528996 | 8.555780 | shallow-v1-2127936912539397632-4 | Queued |

The next case, KIC6113801, stopped acquisition at aperture authentication. In
quarters 3, 7, 9, 11, 13, 15 and 17 the native metadata declares NPIXSAP=7,
but the saved binary mask contains six pixels. Ten other quarters match the
declared pixel count. The original source and parent files, failed attempt and
logs remain preserved. No mask was edited, aperture guard bypassed or physical
false-positive label assigned.

Project rule 7 requires stopping affected work and asking the PI for a
software/instrument-method blocker, without parking further BOIs for that same
unresolved issue. The intake stopped and a separate preserved repair generation
was requested. The three queued native jobs and previous jobs can continue;
100 groups remain scheduled but unattempted. The reviewed dispatcher refuses to
restart this stopped intake. Hourly follow-up monitors running work and can
continue independent evidence review while awaiting the PI's direction.

## Evidence and checks

- [Progress and controls](../results/boi-validation/standard-intake-progress-20261026.json)
- [Aperture blocker](../results/boi-validation/standard-intake-aperture-blocker-20261026.json)
- [Recoverable source capsule](../results/boi-validation/standard-intake-source-20261026.json)

Real KIC8573780 proposal/scene input binding passed an end-to-end control;
wrong-signal and changed-clock inputs were rejected. Scientific scene, geometry,
checksum and transfer functions remain AST-identical to the reviewed helper.
The post-fit checkpoint preserves the existing main body and prevents unreviewed
FPP launch. The stopped-dispatch control creates no new acquisition or event.
Focused Ruff and complexity checks pass. These checks validate orchestration
and provenance; they do not validate a planet.
