# Native fitting kernel experiment

`fast_triceratops.native.accelerated_native()` evaluates the existing batman
model with an experimental orbit kernel in a separate process. No accepted
native fit, installed runtime, preparation or probability campaign uses it.

## Measured cost

The inspected native fit for BOI 2076383384993192576 has 326 observed photons,
20 exposure samples and 11 fitted parameters. At 128 fixed, evenly indexed
accepted posterior points, batman's `_rsky` consumed 0.727 of 0.804 profiled
CPU seconds for 512 likelihood calls. Limb darkening consumed 0.035 seconds;
juliet's Gaussian likelihood consumed 0.007 seconds including its reductions.
The likelihood was evaluated directly; no inference sampler was started.

Moving orbit constants outside the cadence loop produced these alternating
512-call CPU measurements on the loaded Mac:

| Trial | Stock | Experimental |
| --- | --- | --- |
| 1 | 0.838 s | 0.648 s |
| 2 | 0.814 s | 0.549 s |
| 3 | 0.802 s | 0.575 s |

The ratio of medians is 1.42x. Every saved model-flux value and every
log-likelihood bit matched. This measures fixed-point likelihood execution,
not the wall time, convergence or posterior of a new fit. CPU scheduling and
frequency were not pinned. All trials are retained in the
[report](../results/fast-triceratops/native-likelihood-20261025.json).

## Implementation

`native_orbits.c` derives from the reviewed batman `2.5.3+spacelab.2` source.
Its bracketed `getE` solver, iteration limit, tolerances, circular branch and
front/back eclipse gates are unchanged. Conjunction/periastron conversion,
the eccentricity factor and sine of inclination are computed once per model
instead of once per exposure sample. Repeated sine values within a cadence
are reused with the same expression order.

The wrapper checks the reviewed Mac orbit binary's hash. Clang emits LLVM IR
to stdout, and llvmlite JITs it in memory. No wheel is installed and no build
directory is needed. The generated module uses the selected JIT target's
layout. Apple Clang's unsupported stack-probe attribute is removed only for
this bounded-stack numerical function; unknown probe ABIs fail explicitly.
Three loader failures before the first numerical call remain in
`results/fast-triceratops/native-orbit-*.json`.

The orbit code is adapted from batman (Laura Kreidberg); the engine is adapted
from TRICERATOPS (Steven Giacalone).

## Checks and use

The 49 native controls compare the pinned binary on 472,320 sky-position and
true-anomaly values. They cover short and long periods, both eclipse types,
angles, the 1e-5 circular-branch boundary and eccentricity approaching one.
Full batman curves over the Kepler-10 fixture also match, including the
long-duration, large-radius-ratio and high-density control domains. A separate
312,960-value exploratory comparison matched before these tests were added.
Three complete serial juliet/emcee controls also retain the same seeded
posterior bytes, prepared arrays and synthetic photons, including the three
physical-domain tails. These are new synthetic controls, not reruns of accepted
candidate fits.

The supported input is a contiguous float64 time vector on the arm64 Mac,
with one numerical thread and the reviewed binary. The context restores
batman even after an exception. Use it only around a separate experiment:

```python
from fast_triceratops.native import accelerated_native

with accelerated_native():
    model = batman.TransitModel(params, time, supersample_factor=20, exp_time=exposure)
    flux = model.light_curve(params)
```

The Clang/LLVM versions are qualified explicitly, and `provenance()` exposes
their versions, flags and generated-IR hash. Spawned fitter processes need
their own kernel initialization; a parent-process context alone does not enable
the kernel in a fresh child.

The production native fitter and its scientific review remain unchanged. A future
adoption needs a separate generation, source/environment freeze and the
appropriate sampler and known-object controls. The measurements establish
parity on the exercised inputs, rather than a formal proof over every possible
floating-point orbital input or future compiler.

## Other fitting opportunities

The installed juliet dynamic-dynesty path uses a standard multiprocessing
pool with bound likelihood/prior callbacks. The minimal fitter needed to
evaluate these same photons serializes to 185,166 bytes per bound callback;
512 pickle round trips consumed 0.072 CPU seconds before IPC. Dynesty already
provides a pool that caches callbacks in worker initializers. Its end-to-end
benefit and checkpoint behavior still need a paired experiment. Priors,
supersampling and convergence budgets must remain fixed during that comparison.
