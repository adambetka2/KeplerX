# Local PSF Flat retrieval pilot

On 14 October 2026, Adam requested a local PSF Flat experiment while the
server SAP survey continues. The real-source comparison initially awaited
the missing corpus disk. The later direct authorization and disk-independent
run are described below. The server checkout, workers and survey folders are unchanged.

## Directly authorized real comparison

On 15 October, Adam explicitly requested the brightest 5,000 usable real Flat
sources, replacing the earlier conditional 15% launch criterion. The frozen
selection is `results/psf-flat-real5000-20261014.csv`: 4,753 K and 247 M dwarfs,
G 12.173–16.399, in the server survey's brightness order. Do not replace targets.
The adopted detector source and configuration remain unchanged.

Adam also authorized internal MacBook outputs because only one PCIe disk can
be connected at a time. `flat_local.py` uses the policy in
`results/psf-flat-real5000-local-20261015.json`. It prepares available sources
with ten normal-priority workers, waits for the remaining exact local inputs,
then searches all 5,000 with ten workers. Outputs, raw curves, cadence masks,
runtime, receipts, numerical caches and compiled libraries live under
`/Users/adam/Other/spacelab-runs/psf-flat-real5000-20261015-v1/`.
The repository's `dump` symlink is unchanged. Corpus files remain read-only.

Every prepared record array and cadence count must reproduce the frozen CSV.
The full original receipts and pilot runtime reside on the disconnected
workspace disk; newly measured raw provenance and the rebuilt pinned runtime
are recorded separately. Hash equality of prepared records does not imply
independent verification of those unavailable raw receipts or old runtime.
This is a directly authorized real-source comparison; it does not establish
the programme's 10,000-case gate or classify proposals as planets.

Launch through `uv` with the policy's interpreter, using
`dwarfs/experiments/flat_local.py launch --policy
results/psf-flat-real5000-local-20261015.json`. A pre-existing output directory
prevents duplicate launches. Once launched, input disk availability is checked
every minute without changing the selection. An execution failure is preserved
and stops the run; the launcher does not restart it.

### Available-source search and vetting

A second mount scan on 15 October still found `discSSD` and the Windows volumes
`Nowy` and `Untitled`; the odd-shard corpus remained absent. Adam authorized
searching the available subset and then vetting its proposals. The available
count is 2,501, with 2,369 K and 132 M dwarfs; 2,499 are missing. The count
correction and use of all available sources were stated before launch.
The original waiting coordinator was stopped with its inputs and scope-change
receipt preserved. It will not restart when another disk is mounted.

`results/psf-flat-real2501-20261015.csv` is the exact intersection of the frozen
5,000-source selection with prepared inputs. `flat_subset.py` verifies every
inherited archive, record hash and cadence count, then searches with ten workers
under `available2501-v1/` in the internal run directory above. It preserves all
source errors and every emitted proposal. The detector source, numerical
runtime and configuration are unchanged.

`flat_vetting.py` snapshots the existing pipeline without editing its source.
A bound adapter registers `dwarfs_physical_shallow_v1` discovery provenance and
supplies complete 60 arcsec scene cones from all 406,547 original source-parent
identities. It copies each reported proposal's native Flat and diagnostic
arrays, verifies the Flat records against the detector input, and expands the
supplied ephemeris using the existing shallow event-window function. Unique
run-scoped identities preserve original proposal rows, clocks and event
membership. Only Flat is declared searched; diagnostic arrays remain available.

The five vetting gates, fits, thresholds, masks and first-failure behavior are
unchanged. The new disposition policy binds those settings and the private
pipeline source. Vetting uses its pinned Python runtime and ten workers after
all detector sources finish. The complete catalogue snapshot matches the
configured 12 October hash; two files were copied read-only from the server.
Legacy catalogues and the parent index reproduce the repository archives.
An internal copy of the corpus index excludes unavailable cached pixel paths,
retains every source identity, and allows the existing public-pixel retrieval.
The original index, corpus, repository `dump` link and server study are unchanged.

The first full vetting attempt stopped at pixel retrieval because its private
snapshot exposed the light curves but omitted the cached-pixel directory.
Adam authorized the repair and retry on 15 October. The corrected snapshot
links both collections read-only. `vetting2501-v2/` repeats vetting on the same
697 proposal sources; `vetting2501-v1/` preserves the original failure.
The repair receipt in `results/psf-flat-real2501-vetting-repair-20261015.json`
records matching inputs, proposals, scientific settings and pipeline files.

The detached follow-up waits for a completed detector catalogue, verifies all
results and preserves every proposal before running the ordinary vetting
stages. An execution failure is preserved and stops the follow-up. HELD and
DISMISSED signals retain their reasons; a BOI remains a pipeline designation.
Availability selects this subset, and shared-time coverage is this declared
source cohort. The run does not measure population completeness, confirmed
planet yield or novelty.

## Inputs

The running `mk-sap-20261014-v1` identity declares `sap_qdiag`, subtracting the
constant neighbour contribution inferred from CROWDSAP. It does not search
PSF Flat. Its source checkout is frozen at `f7e78fd`.

The hash-identical local and server `final10000-v7` cases and clocks describe
9,415 `psf_qdiag`, 385 `sap_qdiag` and 200 `psf_flat` cases. The cohort is not a
PSF Flat benchmark. Preserve it and prepare a derived experiment from the
same hosts, physical injection parameters and scorer clocks. Every derived
curve uses the delivered PSF Flat flux and error, QUALITY=0 and flatten-mask=0.
Missing or insufficient Flat data remain failures; another photometry product
does not substitute for them.

The completed detector-change pilot is
`dump/psf-flat-retrieval-20261014-v1/usable1000-v2/`. Its exact mix is:

| Kind | Cases |
| --- | ---: |
| Synthetic planet | 600: 300 M and 300 K |
| Real Kepler planet | 100 |
| Catalogue false positive | 20 |
| Native eclipsing binary | 25 |
| Instrumental artifact | 60 |
| Original, inverted, year-scrambled | 65 each |

Selection uses no detector outcomes. Preserve kind, M/K, depth-bin and original
preparation-eligibility quotas, then balance metadata CDFs by swapping within
those strata. Check magnitude, depth, period, duration, quarter count, stored
cadences and finite Flat cadences against all 10,000 cases. The largest
available marginal KS distance in the original selection v2 is 0.0462; for synthetic
signals it is 0.012. This checks marginal coverage, not joint equivalence or
noise equivalence. Archived planet/FP packets lack some magnitude and cadence
metadata; those comparisons remain explicitly unavailable.

The first selection is preserved in the parent folder. Its secondary coverage
audit found a 0.1123 Flat-cadence distance among year-scrambled controls.
Selection v2 adds cadence coverage to the balancing objective. Neither
selection was changed in response to detector results.

Selection v2 prepared 604 inputs; 363 had no usable Flat cadences and 33
failed exact year-scramble construction. Adam explicitly requested replacing
these cases to search 1,000 usable inputs. `usable_flat.py` keeps the 604
prepared cases and fills the remaining slots within kind/M/K and synthetic
depth-bin quotas. Replacement uses input eligibility only. It does not use
detector outcomes or filter by injection detectability. Preserve excluded
parent cases and failed replacement attempts separately. Recovery now
conditions on usable Flat data, with 700 signals and 300 controls.

The new distribution report compares the cohort with all 10,000 cases and
with the metadata availability prefilter. That prefilter is not a complete
preparation census. Year-null eligibility is checked on actual Flat records;
eligibility on an older full-PSF view must not exclude a valid Flat null.

Read selected background sources from the server corpus in one low-priority,
single-thread process without writing server files. Recover absent catalogue
inputs from canonical MAST products and require their original FITS hashes.
Keep acquired curves, original and injected records, cadence masks, physical
losses, thresholds, all proposal decisions and first failures on the external
disk. Source and numerical-runtime identities bind preparation and search.

## Unchanged detector: SAP versus saved Flat

Adam clarified that the intended comparison is the adopted server detector on
SAP and PSF Flat, with its centering, nuisance model and thresholds unchanged.
`sap_retrieval.py` searches only SAP in
`dump/psf-flat-retrieval-20261014-v1/usable1000-sap-v1/`. It reuses the adopted
detector's completed Flat results in `usable1000-v2`; it never searches Flat.
The alternative detector from the earlier experiment is excluded.

Keep the exact ordered 1,000 case IDs, physical injection parameters, control
recipes and original scorer clocks. Use the server's delivered `sap_qdiag`,
quarter CROWDSAP correction and preparation without the Flat mask. Inject the
same fractional physical loss against each corrected source-flux median.
The detector source, configuration and numerical runtime must match the saved
Flat baseline. The Mac's survey wrapper adds an `evolution.py` provenance
binding; its remaining syntax tree must equal the frozen server revision.

Read mounted sources locally. Acquire only selected unavailable sources in one
low-priority, single-thread, read-only server stream. Reuse the original
hash-bound FITS for catalogue anchors. Preserve delivered and corrected flux,
original and injected records, masks, injection arrays and first failures.
Unmodified records must reproduce their acquisition preparation hashes.

Do not replace any SAP-ineligible case. Report all 700 signals and 300 controls,
then a separate comparison conditional on both products preparing successfully.
Split recovery into both, SAP only, Flat only and neither, retaining the case
IDs and separating synthetic and real signals. Preserve the two inherited
Flat detector errors and distinguish new SAP errors. Saved Flat summaries must
reproduce the original report before the comparison is published.

Run `plan --flat .../usable1000-v2`, `acquire`, `prepare`, `run --workers 10`
and `report`, each with `--out .../usable1000-sap-v1`, through the same `uv`
runtime and single-thread BLAS environment below. The SAP lane uses the existing
cache-isolated search launcher. Both extractions remain available for audit.
The comparison is exploratory; conditional post-delivery injections cannot
measure upstream attenuation. Real confirmed planets probe retained signals.

All 1,000 SAP cases prepared successfully, yielding 997 unique searches.
K00492.01 recovered its exact clock in the worker preflight, and a native input
reproduced complete server preparation byte for byte. The ten-worker launch
and checks are recorded in `results/sap-flat-paired1000-20261014.json`.

## Earlier paired detector test

`dwarfs/experiments/flat_retrieval.py` runs the existing adopted design and
`dwarfs/shallow/evolve_psf_flat.py` on identical prepared Flat records.
The candidate uses quarter-median centering for seeds and one intercept per
quarter/gap segment for physical likelihoods. It retains the adopted duration
and period grids, physical profiles, local-noise model, coherence ranking,
harmonic conditioning and rejection thresholds. The incumbent retains its
biweight filtering and spline nuisance model. Private module instances keep
the candidate's operators from changing the incumbent.

Workers receive prepared records and configuration; scorer clocks remain in
the coordinator. Both searches finish before scoring. Score the frozen
period/epoch/duration clocks with the existing matcher, count zero retained
injection support as a miss, and keep every preparation and detector failure
in the usable cohort's 700-signal/300-control denominators. Report synthetic and real recovery
separately, M/K and depth strata, paired wins/losses and control proposal burden.
An unexpected search error preserves its partial evidence and fails admission.
The endpoint check caught a cached-module import mismatch; the failure and
source remain in selection v2. The corrected runner imports its snapshot's
engine. A frozen K00492.01 input recovered its exact clock with the candidate,
and all 20 archived Flat controls reproduced byte for byte. These endpoint
checks do not establish cohort performance.

The pilot's provisional admission requires at least 20% combined signal
recovery, no unexpected errors and no increase in active controls or control
proposals versus the incumbent on these same Flat inputs. This is a development
decision. It does not pass the programme's 10,000-case gate or establish
astrophysical classification. Injections occur after delivered flattening;
upstream transit attenuation remains unmeasured.

## Launcher repair

The first paired run remains under `usable1000-v1`. Its incumbent recovered
163/700 signals, with two preserved numerical rank errors. Every candidate
case failed while loading a Numba kernel from the incumbent's private module
namespace. These execution failures do not measure candidate sensitivity.

The launcher now gives each study and engine its own cache under
`pilot/numba-cache/<engine>`. It sets the cache before spawning workers and
restores the coordinator's environment after the pool closes. Detector
source, numerical libraries, input arrays, masks, configuration and scorer
clocks are unchanged. A regression switches between both engines in four
fresh worker processes, including warm restarts, and checks identical noise
arrays and actual cache hits.

The full worker preflight searches K00492.01 and the first saved synthetic
failure without errors. The known planet matches its exact frozen clock on
cold and warm starts. Candidate proposal values match across those starts;
the incumbent reproduces its original proposals exactly on both inputs.

The retry preserves the failed run and archives its preparation driver.
It inherits the byte-identical 1,000 prepared cases and reuses the incumbent's
997 completed input results after checking identical detector snapshots and
runtime. The two incumbent failures remain counted. The candidate searches
the same inputs again; scoring still compares the original 700 signals and
300 controls. The repair does not waive either admission gate.

## Run and inspect

Use the Python 3.11 runtime in the parent study folder through `uv`, with
single-thread BLAS, an external Numba cache and ten normal-priority workers.
The Mac has eight performance cores and two efficiency cores. The initial
two-worker launch shared the Mac with BOI work. After those jobs finished,
Adam requested full CPU use. The resumed launch preserved 530 completed
searches, their hashes, the frozen inputs, detector code and numerical runtime.
The original run's `launch-full.json` and `execution-full-v1/` record that
execution change. The retry has its own launch and inheritance receipts.
Run the stages in order: `acquire`, `prepare`, `run`, `report`. `prepare`
requires a new preparation; `run` can continue its unchanged source/runtime
snapshot and preserves failed results. Do not overwrite an existing study.

```bash
study="$PWD/dump/psf-flat-retrieval-20261014-v1/usable1000-v2"
runtime="$PWD/dump/psf-flat-retrieval-20261014-v1/runtime/bin/python"
export UV_CACHE_DIR="$study/../cache/uv"
export NUMBA_CACHE_DIR="$study/../cache/pilot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --no-project --python "$runtime" python dwarfs/experiments/flat_retrieval.py run --out "$study" --workers 10
uv run --no-project --python "$runtime" python dwarfs/experiments/flat_retrieval.py report --out "$study"
```

The initial plan and preparation remain under `selection-v2`.
`usable_flat.py --parent .../selection-v2 --out .../usable1000-v1` prepares
the replacement cohort. It can resume an unfinished input-only build; a
frozen cohort refuses extension. Inherited arrays remain byte-identical.

`representativeness.json` records selection checks; `prepared-cases.json`
records all input dispositions; `pilot/` holds the two blind searches;
`retrieval-report.json` holds the scored comparison. Logs and the local launch
receipt stay in the active study. No stage launches a real-source survey.

## Later real-source comparison

The pilot's original `real5000.csv` preserves the server's first 5,000 entries,
including sources without usable Flat data. It remains bound to the pilot's
original plan. Adam requested replacing unavailable sources. The active list
for the later real run is `results/psf-flat-real5000-20261014.csv`. Full
preparation receipts remain under
`dump/psf-flat-retrieval-20261014-v1/real5000-usable-v1/`.

`flat_sources.py` preserves server queue order and selects the first 5,000
sources that prepare at least 100 actual Flat records. The census provides
only a necessary cadence-count upper bound. Every selected source passes the
real finite-time/flux/error, positive-error, QUALITY=0, FLATTEN_MASK=0,
quarter-median, upper-12-MAD, deduplication and float32 preparation. One
low-priority reader checks server corpus rows without changing server files.
The list binds raw FITS, curve arrays, flatten masks, retained cadences and
prepared records to hashes. Keep the excluded prefix and failed preparations.
No detector outcome enters selection. Before local detection, reproduce each
prepared-record hash from the mounted corpus.

The frozen list contains 4,753 K and 247 M sources through queue rank 8,235.
Within that prefix, 3,230 entries fail the necessary cadence-count upper bound
and five fail actual preparation. All selected sources pass; they retain
132,193,500 cadences in total, with at least 179 per source. Twenty overlapping
cached sources reproduce the server-prepared record hashes byte for byte on
the Mac. `results/psf-flat-local-update-20261014.json` records the selection,
execution change and verification.

This bright-first usable subset is weighted toward K dwarfs and does not
represent the full faint population. When the disk is available and the pilot
is reviewed, run the unchanged adopted detector on Flat locally and join
source IDs to the server SAP results. Report eligibility, proposals, periods,
depths and proposal overlap. Without ground truth or vetting, differences in
real proposals do not measure planet recall or astrophysical false-positive rates. The direct
server/local comparison also changes photometry, processing and runtime;
the SAP-only pilot above keeps the detector and local runtime fixed.

## Conditional real-source launch

Adam authorized an hourly check and a conditional local run on 14 October.
Use the complete 700-signal scores: launch only if Flat recovery is at least
1.15 times SAP recovery. This is a relative 15% improvement. With the frozen
Flat reference of 163, SAP must recover at most 141. The 300 controls and all
recorded detector errors remain in the report; they are not additional gates.
This authorization is for the 5,000-source experiment, not the programme's
10,000-case admission gate.

The policy is `results/psf-flat-real5000-trigger-20261014.json`.
`flat_real.py check --policy ...` checks the final report and mounted inputs.
`flat_real.py launch --policy ...` starts once, only after the criterion passes
and all frozen sources are locally available. Use the same `uv` runtime.
The target folder is `dump/psf-flat-retrieval-20261014-v1/real5000-flat-20261014-v1/`.
It contains 4,753 K and 247 M sources from the previously frozen usable list.
No source replacement or server execution is authorized by this policy.

Ten workers reproduce every selected curve, flatten mask, retained cadence
sequence and prepared-record hash before detection. The unchanged adopted
detector then uses the existing cache-isolated launcher with ten workers.
Keep the original arrays, masks, configuration, source/runtime identity,
every proposal and first failure. An existing output prevents a duplicate
launch. A failure keeps its log and evidence; do not silently restart it.

The hourly monitor reports the pilot once. If it qualifies while a corpus disk
is unavailable, report the missing inputs once and keep checking hourly until
they are available. After launch, monitor the real run through its final
`catalogue.json` and report proposal counts and execution errors. If the
criterion fails, or a run fails, report that outcome and pause the monitor.
