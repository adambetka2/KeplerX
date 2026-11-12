# Mac validation runtime

Adam stopped the October 17 campaign on October 18 to investigate CPU use and
memory pressure. Its 110 completed scenario families, ten partial checkpoints,
logs, failures and original freeze receipt remain in `campaign/` under
`/Users/adam/Other/spacelab/other/boi-validation-20261017`.

The derived generation is `campaign-hardware-v1/` in the same workspace.
`audit/generation-lineage.json` binds the original receipt and checkpoint
hashes. Each derived checkpoint retains its seed, generator state and completed
rows; only its generation identity changes. Retained weight files are linked
without rewriting the originals.

## Changes

`validation_scene.py` caches the fixed particle CDFs and gathers pointing
responses together. Particle selection, random-stream consumption, scene
support and physical distributions remain unchanged.

`validation_models.py` fuses the original phase-grid interpolation, native
photon model, affine MAP detrending and CSR phase binning. It retains every
native photon, the original covariance whitening and positive-baseline checks.
Its bound calculation uses the same equations and cutoff. A sufficient
positive-baseline certificate avoids temporary arrays; uncertified cases use
the exact native-domain check. Compiled arithmetic disables fast math.

`validation_runtime.py` launches a Mac agent with application resource policy
and an explicit interactive process type. Every numerical worker requests and
verifies user-initiated scheduling. Each attempt has a separate launcher log;
there is no automatic service restart.

The scientific plan changes only its output directory. The two targets,
20 seeds per target, one million attempts per scenario, sampling batch 256,
render batch 64, discovery ephemerides, priors and thresholds remain fixed.

## Verification

The cached sampler reproduced all saved weights exactly across both 4,096-draw
reference censuses. The fused calculation preserved finite support and bound
flags. Posterior L1 differences were `6.72e-12` and `3.51e-10`, below the existing
`1e-8` replay tolerance. Fused weights have measured floating-point differences;
they are not described as byte-identical. The first exact-weight comparison
failures and their arrays remain in the audit directory.

Both targets passed 73 bound-domain cases, including unsafe baselines. Six
runtime regressions pass. Folder verification has 332 passes and the same three
pre-existing alias-context and singular-position-covariance failures. The
campaign environment remains pinned. Separate test-only packages and cache
routing accommodate the folder tests; they do not enter production.

Readiness, full-worker measurements and the remaining runtime uncertainty are
recorded in `results/boi-validation/hardware-runtime-readiness-20261018.json`.
The requested seven-to-eight-hour finish has not been demonstrated. Preparation
replays are excluded from the 40 production runs.

## Operation

The original campaign and monitor remain stopped while Adam chooses whether
to accept the measured runtime. Check the derived generation with:

```bash
bash /Users/adam/Other/spacelab/other/boi-validation-20261017/campaign-hardware.sh status
```

After the runtime decision, use the same wrapper with `resume`. It verifies the
new receipt before launching `local.spacelab.job.boi-validation-hardware-v1`.
Never resume the original and derived generations together. Do not edit frozen
code or receipts to recover a job. The original physical-closeout requirements
in `boi-validation-rerun.md` still apply.
