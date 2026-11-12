# TIC statistical validation rerun

Adam stopped the original Mac run on October 18 for a hardware/runtime review.
Its partial evidence is preserved. The separate derived runtime, verification
and current stop state are documented in [Mac validation runtime](boi-hardware-runtime.md).

The active statistical-validation protocol covers BOI-2077554781499350144-1
(KIC 6699887) and BOI-2127186564572468352-1 (KIC 8745833). It follows the
published TRICERATOPS repeat test: exactly 20 independent TIC-based runs per
target, or 40 production runs in total. Each run attempts 1,000,000 draws for
every scenario family and uses the frozen discovery ephemeris.

The canonical plan is
`pipeline/vetting/statistical-validation.json`. The runner hard-codes the two
source IDs and the 20-run count. It rejects another catalogue, ephemeris,
target, draw count, threshold, or legacy model-matrix field. The admission
thresholds are FPP <1.5% and NFPP <0.1%.

Gaia remains supporting astrometric, localization, RUWE, and companion
evidence. Gaia stellar alternatives, final-ephemeris repeats, and modified
noise likelihoods are not members of the production census. They may be
reported separately as sensitivity evidence and cannot replace or veto the
declared TIC result.

## Historical stopped extension

The stopped October 13 campaign multiplied the repeat test across Gaia/TIC and
original/final inputs, creating 160 planned runs for the two targets. Its
service was `boi-validation-first-two-20261013.service` under
`/srv/kbonus/dump/boi-validation-first-two-20261013`. The separate paired
common-noise audit used `boi-validation-common-noise-20261014.service` under
`/srv/kbonus/dump/boi-validation-first-two-audit-20261013`.

Those folders, completed results, partial checkpoints, seeds, unfavorable
results, and service records remain historical evidence. They are not resumed
or mixed into the 40-run TIC census. In particular, the 1.8339% common-noise
FPP was a Gaia/original sensitivity result for KIC 8745833, not a paper-aligned
TIC admission run.

BOI-2127169041106093184-1 (KIC 8878385) remains paused and outside this
protocol.

## Execution and audit

The Mac launcher `pipeline.vetting.validation_campaign` uses ten processes,
one numerical thread per process, and keeps the Mac awake on AC power. The
longer 210-family jobs are queued first. It preserves a separate log for each
attempt, checkpoints completed scenario families, and stops dispatching work
after the first worker failure. Already running workers finish their work.
An explicit `resume` verifies completed results before continuing unfinished
seeds. It never adds replacement seeds or silently reruns completed results.

The prepared Mac workspace is `/Users/adam/Other/spacelab/other/boi-validation-20261017`.
Adam authorized internal storage on October 17 without a campaign size cap.
He subsequently moved the workspace into `spacelab/other/`. Active paths and
the freeze receipt were updated after verifying all 466 frozen file hashes.
The original plan and receipt are retained in `preparation/path-relocation-20261017/`.
The existing `dump` symlink and original datasets are unchanged. The workspace
holds restored inputs, the Python 3.12.13 runtime, caches, and campaign outputs.
Its environment exactly matches the archived 112-package Mac lock, including
NumPy 1.26.4, SciPy 1.13.1, PyTransit 2.2.0, and Numba 0.68.0.

All 497 restored dependency hashes match the archive. Both frozen 4,096-draw
reference censuses reproduced every log weight exactly across all 210 and 114
families. A ten-process representative likelihood check passed, with 3.76 GiB
summed peak worker RSS. These checks are preparation, not production seeds.

Check readiness from any directory:

```bash
bash /Users/adam/Other/spacelab/other/boi-validation-20261017/campaign.sh status
```

Only after Adam gives the start instruction:

```bash
nohup bash /Users/adam/Other/spacelab/other/boi-validation-20261017/campaign.sh start \
  >> /Users/adam/Other/spacelab/other/boi-validation-20261017/campaign/launcher.log 2>&1 &
```

The launcher uses the pinned runtime and automatically performs the deep audit
when all 40 runs finish. It reconstructs evidence bounds and category
probabilities, rejecting summaries that differ from retained weights. Output
is in `campaign/numerical-campaign-summary.json` and the corresponding CSV.
`campaign/campaign-status.json` records progress and the first worker failure.
Use the same wrapper with `resume` only to continue an interrupted campaign.
The Mac must remain on AC with its lid open; `caffeinate` prevents idle sleep.

For a monitoring assistant, the wrapper's `status`, `campaign-status.json`,
per-seed `checkpoint.json` files, and logs are sufficient to follow progress.
After a recoverable operational failure, let remaining workers finish and use
`resume` with appended launcher logs. Preserve the first error and every
checkpoint. Do not change the fixed 40-seed census, likelihood, data, priors,
or thresholds. Edits to frozen Python files invalidate the receipt and require
explicit review and revalidation; never bypass the hash checks or delete a
completed result to make a restart work. Completion of the numerical audit
still requires the scientific closeout described below.

Ten statistical-runner and audit regression tests pass. The full vetting suite
has 326 passes and three failures that also occur on the untouched branch:
two alias-context expectations and a singular-covariance expectation. The
changed files pass lint and complexity checks; folder-wide diagnostics are
unchanged from that baseline. Detailed verification logs are in `preparation/`.

The numerical census does not by itself establish statistical validation.
Native-fit convergence, radius, RUWE, localization, catalogue/literature, and
other physical-evidence gates remain required and separately reported.

Closeout binds the existing TIC native-fit quality and localization/vetting
evidence to each target, records the dated catalogue/companion search, and
reports mean, sample standard deviation, minimum and maximum FPP/NFPP over
all 20 runs. Retain unfavorable repeats and weight-concentration diagnostics.
Apply the declared mean thresholds without selective seed replacement.
Existing final-ephemeris and noise-model sensitivities remain separately
identified evidence. A failed statistical cut remains unresolved unless an
astrophysical false-positive mechanism has been reproduced. This campaign
does not establish RV confirmation or planet masses.
