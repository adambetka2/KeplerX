# Statistical validation performance

`fast_triceratops/` is an experiment on `main`. Existing detector, preparation,
fit, campaign and accepted planet records remain the scientific reference.
This page maps the path from a supplied BOI to statistical follow-up and tracks
performance work outside the transit kernel. Kernel work is described in
[Fast TRICERATOPS](fast-triceratops.md).

The exact static engine was selected for queued campaigns on October 26; the
[adoption record](../results/boi-validation/engine-adoption-20261026.json)
describes the separate `campaign-engine/` outputs and original-statistics audit.
The runner's experimental scope label predates that choice and remains frozen.

## Work map

| Stage | Entry point and work | Performance opportunity and constraint |
| --- | --- | --- |
| Authenticate the supplied signal | `post_boi.aperture.physical_aperture`, catalogue queries and candidate acquisition scripts bind source, clock, photons, masks and the source aperture. | Reuse authenticated downloads by request and content hash. Stream pixels and preserve the first failed request. Network latency needs separate timing from CPU. Detection and vetting are outside this experiment. |
| Prepare native flux | `post_boi.prepare.prepare` reads the authenticated arrays, protects every declared signal and runs Lightkurve flattening per quarter. | Preserve the supplied cadence mask, protection, window and polynomial. Profile NPZ decompression separately from the detrending refit. |
| Fit the native clock | `post_boi.native.fit` selects the native ±3-duration window, loads juliet once, fits period and epoch jointly with the supplied priors, and writes the posterior. | Measure initialization, sampler likelihood calls, transit integration and sampler communication. Reuse batman geometry within a likelihood only when its parameter dependencies allow it. Changing priors, supersampling or convergence budgets is a scientific change. |
| Review signal transfer and noise | `post_boi.protection.assess`, `post_boi.kbonus` and native review inspect observed-cadence detrending transfer, quarter flux closure and correlated noise. | Transfer trials refit detrending independently. Shared read-only inputs can be reused; distinct posterior trials, masks and first failures must remain identifiable. |
| Build the scene and population | Candidate acquisition supplies scene CSV, source-quarter geometry, measured constraints and one field population. `post_boi.scene.KeplerScene` authenticates their census and capability. | Cache parsing by immutable input content. Share one authenticated population across repeats. Do not omit scenarios, replace missing measurements or change the population to improve speed or FPP. |
| Freeze the calculation | `post_boi.protocol.freeze` copies source, folds to at most 200 bins while retaining photon membership, records the environment and hashes inputs, and declares 20 seeds. | This is once per generation. Folding and sufficient statistics can be reused across repeats after the freeze. Exact floating-point reduction order still needs paired verification. |
| Verify an attempt | `post_boi.protocol.verify` checks the complete environment and file manifest. `post_boi.campaign.attempt` checks saved outputs before reuse. | Keep these checks. Two measured warm-cache manifests contain 127 files and 10.3–13.4 MiB; hashing takes 0.024–0.031 s. Hash removal is not a useful optimization here. |
| Initialize the repeat | `calculate_quarters` loads geometry, builds `QuarterLikelihood`, constructs `KeplerScene` and prepares depths. Imports and Numba compilation can dominate a short repeat. | Persistent spawned workers amortize imports and compilation. Independent processes isolate the global RNG and temporary likelihood swaps. Do not fork initialized numerical runtimes. |
| Sample scenarios | `target.calc_probs` calls the target, companion, background and nearby-source scenario families. Upstream samples priors, evaluates stellar relations and selects limb darkening. | Preserve scenario order, attempted draw count and RNG sequence. Deduplicate deterministic lookups and avoid repeated parsing. Profile again after each kernel change because the dominant cost moves. |
| Evaluate physical curves | Pytransit generates finite-exposure models; the V3 callback applies the source-quarter Gaussian likelihood and the existing secondary veto. | Fuse model consumption and residual reduction to avoid draw-by-time temporaries. Exact mode must reproduce the reference reduction and nonfinite behavior. The static copy now integrates the verified kernels directly. |
| Integrate and normalize | Each scenario computes its log evidence, keeps its best model, then `calc_probs` normalizes the full scenario table into FPP/NFPP. | Preserve log-space arithmetic, the attempted-draw denominator, ties and degenerate outputs. A partial ranking can replace a full sort only when its retained outputs agree. |
| Allocate repeats | `post_boi.campaign.resume` currently runs each repeat serially with the declared Numba thread budget. | Run the first two independent seeds concurrently, wait for both, then apply the existing futility screen. Only then allocate the remaining 18. Single-thread worker processes can use serial sampling and spline work concurrently. Bound workers by the declared CPU and measured memory budget. |
| Preserve and close out | Attempts write logs, raw target objects, lnZ, scenario/source tables and hash receipts. `finish` records every repeat, mean, sample SD, range and unfavorable indices. Physical and exact-signal reviews remain separate. | Measure serialization and compression before changing them. A durable coordinator owns scheduling, failed attempts stay intact, and numerical results do not become physical validation through a faster runner. |

## Experiment boundary

The workflow experiment reads an existing frozen V3 plan and writes to a new,
separate output directory. It does not resume, patch or append to the source
campaign. The experimental engine and its workflow source are copied and hashed
separately. Replays preserve the source seed sequence, input identity and
first-two allocation screen. Reduced-draw profiling is explicitly a benchmark,
never a candidate integration or admission.

Exact comparisons include each scenario's lnZ, the complete scenario table,
best-model fields, FPP and NFPP. A rounding-level or empirically bounded mode
must be declared independently; it cannot silently replace the exact mode.

The existing serial coordinator already reuses its imports and compiled kernels
across repeats. A process pool pays that startup once per worker, rather than
once per seed. Its proposed throughput gain comes from running the serial
sampling and stellar-relation work concurrently. It needs a comparison against
the original coordinator with the same total CPU budget; single-thread kernel
speedups alone do not establish that gain.

## Run an experiment

Use the source campaign's pinned V3 Python, through `uv run --python`. Output
parents must already exist and have at least 2 GiB free. `PLAN` points to an
existing frozen `plan.json`; `NEW_OUTPUT` must be outside its campaign directory.

```bash
python -m fast_triceratops.workflow profile PLAN NEW_OUTPUT --mode engine --draws 20000
python -m fast_triceratops.workflow run PLAN NEW_OUTPUT --mode exact --workers 1 --repeats 2
python -m fast_triceratops.workflow resume NEW_OUTPUT/experiment.json
```

Each experiment copies the fast package, binds the original plan/freeze, and
executes the original frozen `post_boi` source with the copied fast package.
Numerical threads are set to one before imports. The process pool stays alive
across the first-two screen and later seeds. Phase packets are reused within
each worker as read-only arrays. Each dedicated worker enters its selected
engine once; repeated source inspection and scenario reconstruction are removed
from later seeds. Input and environment verification still runs
before every attempt. Resume verifies saved outputs; interrupted attempts are
preserved and restart at the same declared seed.

The default run has 20 primary seeds and the source plan's optional three
recovery diagnostics. `--repeats 2` is a two-seed benchmark. Reduced draws are
allowed only as an explicitly recorded performance experiment. Neither output
alone admits a candidate; the adopted queue separately writes the original
campaign audit and requires scientific closeout. A complete or explicitly stopped experiment is
not restarted; create a new directory for another independent measurement.

At the original draw count, available saved source repeats are checked against
their identity, file receipts and probability normalization. The replay then
compares lnZ bits, the complete scenario and star CSVs, FPP and NFPP. A mismatch
preserves the calculation and fails the experiment. Pending work is bounded by
the worker count; an exception stops further allocation. Already running
attempts retain their outputs. A `stop.json` in the experiment root also stops
new allocations and prevents resume.

`profile` writes CPU profiles in JSON alongside named verification, packet,
calculation, comparison and preservation timings. Profiling includes its own
instrumentation and cold compilation. Use unprofiled `run` measurements for
speed comparisons. Worker RSS is a process lifetime high-water mark, and the
launch record includes subprocess startup that repeat timings exclude.

A read-only 20,000-draw replay of BOI 2100325529865507712 on October 25 matched
all lnZ bits, scenario/best-model fields, star rows, FPP and NFPP for the copy
and existing exact mode. CPU times were 12.25 s stock, 12.61 s for the still
unmodified copy, and 1.89 s exact (6.46x). This is one loaded-Mac observation,
not a throughput measurement. Numba disk caching was disabled for this check.
Its separate CPU profile also found 42 source-inspection calls per exact-mode
entry, spending about 0.33 s in source extraction. Worker lifetime mode entry
avoids repeating that setup. All fourteen workflow controls pass, including a
real frozen-source Kepler-10 replay, preserved injected failure, same-seed retry,
verified completed-result reuse, changed-input rejection and explicit stop.

The final static engine was also checked against two saved original-million-draw
repeats. The warm repeat improved from 31.73 to 14.57 CPU seconds (2.18x), with
bitwise evidence and complete table equality. Including the measured verification,
comparison and preservation phases gives 31.87 versus 14.70 CPU seconds (2.17x).
See `results/fast-triceratops/closeout-20261025.json` for source and output receipts.

The first adoption check on October 26 contains two single-worker seeds of
KIC 5458397, both matching the original campaign bitwise. No completed
five-worker 20-repeat engine campaign was found at 00:56 UTC; the two active
jobs were native fits. This harder case has about 300 CPU seconds per later
engine repeat, and its idealized five-worker projection is about 26 minutes
against a historical 37-minute serial campaign. These are different resource
and load conditions, not a measured speedup. See the
[first-results review](../results/fast-triceratops/first-production-review-20261026.json).

By the afternoon check, 32 engine campaigns had completed: 11 full 20-repeat
sets and 21 first-two futility exits. Full campaign launch times ranged from
2 min 24 s to 16 min 17 s, with a 5 min 1 s median on five workers. All 262
primary repeats passed the record consistency checks. These are measured
campaign times; identical-input stock baselines are still absent. See the
[campaign measurements](../results/fast-triceratops/campaign-measurements-20261026.json).

The broader historical comparison includes 20 completed stock V3 sets on the
same nominal five-core budget. Median elapsed time is 13 min 1 s stock versus
5 min 1 s engine (2.59x); restricting exposure integration to 20 samples gives
2.46x, and the M/K-only comparison gives 2.10x. These describe different
production cohorts, rather than an identical-input experiment. See the
[historical comparison](../results/fast-triceratops/historical-stock-comparison-20261026.json).

## Remaining measurements

1. When the production CPU allocation is free, compare the original serial
   coordinator against a persistent pool at the same total core and memory
   budget. Measure two, five and more workers only as resources permit.
2. The native likelihood now has fixed-posterior and serial sampler controls;
   see the [native experiment](fast-triceratops-native.md). Cached callbacks and
   worker initialization in the parallel fitter remain unmeasured.
3. Only pursue preparation, scene parsing or evidence I/O rewrites when their
   measured contribution warrants them. Existing stock profiles put scene/data
   frame setup below 0.1%; accelerated runs need a fresh profile.

## Resources

On October 25, 2026, `discPCIe` had only 190 MiB free and `discSSD` 2.1 GiB.
Both are mounted. Do not write new benchmark artifacts to either until the
repository's 2 GiB reserve is available. Adam authorized the internal Mac disk
for this experiment; use `other/fast-triceratops-trial-20261025`. Active validation
owns the production CPU allocation; small controls use one numerical thread at
nice 19. Throughput
scaling needs a separate measurement when that allocation is free.
