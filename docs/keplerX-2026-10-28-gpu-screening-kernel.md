# V3 GPU screen kernel: profile, fix and measurements

28 October 2026. Baseline is main `28bee5bf0` (morning). Changes are
on main: `841f8e044` (FMA fold classifier), `305b021de` (screen bin-end fix),
`7d60f57e1` (pair layout, 16-duration fold), `27ddba1cb` (32-bit indices).
Receipts are under `KeplerX/results/performance-engineering/20261028/screen-probe/` and
`verify/`. Every timing below is either paired or interleaved: one
long-lived worker per variant, alternating chunk by chunk (`paired.py`), so
GPU clock drift (±15% between separate runs) cancels. Paired runs also assert
identical cells and flags in every chunk unless stated.

## Correctness fix

The screen left a bin when `rel`, the distance from the bin start rounded to
float, reached `width + edge`. Near the end of a bin float resolution is
ulp(w)/2, about 9e-10 d, larger than the 2e-10 d edge margin. A cadence
lying between those distances before a bin's end was dropped from both
bins and not flagged. The resulting bounds need not bracket the exact MES.

On 6815, period 5.0359 d, bin 106 has exact MES 14.544. The screen reported
bounds of [14.636, 14.646] and no flag; the fold path gave [14.539, 14.549].
Across the three long controls, 1, 6 and 13 cells per first pass were wrong,
with upper bounds off by up to 0.28. No final result changed on these controls. An
invalid lower bound can, however, raise a period's floor and exclude its true
best cell.

The end test now uses the double-float tail. `tests/test_gpu_bounds.py`
crafts such cadences in a cell's best bin. The old kernel reports 19.47 for
an exact 11.16; the fixed kernel brackets it. After the fix the screen and
the fold kernel classify identically: byte-identical cells and flags on
all three controls.

## Where screen time goes

Screen-only device seconds per first pass, current shader, 56c4: 0.79. Removing
parts in timing-only variants: pair accumulation 0.43 (loads 0.31,
adds and their registers 0.12), cadence loop 0.12, epoch arithmetic, table
lookups and reduction 0.24.

Instruments' Metal GPU Counters (`screen.trace`, screen only, 56c4) during
busy samples: ALU limiter 33%, last-level cache limiter 26%, MMU 20%, buffer
read limiter 13%, threadgroup store 0.5%, and **compute occupancy 22%**. No
unit is saturated. The kernel is latency-bound at low occupancy. Pipeline
thread caps fall from 1024 to 512 as durations per tile rise, consistent with
about 48 registers per thread at ten durations: one threadgroup per core.

| Experiment (screen only unless noted) | Paired effect | Outcome |
| --- | ---: | --- |
| Duration-pair-major pair words (total GPU) | -10 to -21% | adopted (mostly fold) |
| Fold up to 16 durations, 7,680 histogram ints (total) | -10 to -11% | adopted |
| 32-bit word indices (total) | -1.3 to -2% | adopted |
| 32 KB static threadgroup array made dynamic | +0.6 to +1% | rejected |
| int32 numerators + 16-bit denominators (load pattern) | +2% | rejected |
| Next-epoch table prefetch | -0.2 to -0.7% | rejected |
| Restructured loop with speculative pair loads (total) | +11 to +23% | rejected |
| 8- or 4-duration tiles (fewer registers, repeated scans) (total) | +16 to +29% | rejected |
| Exact edge path removed (timing only) | -4% | not applicable |

int16 numerators were measured earlier (`astra-3seconds/gpu/`). They widen
bounds enough to multiply whole-cell CPU resolutions 17-fold on 6815, so the
numerator stays int32.

## Measured totals

The six-control detector replay was interleaved B H H B, with all 24 runs
native-equal. B is main `28bee5bf0`; H is HEAD, which includes the
whitening bank. Mean of per-control wall seconds:

| Whitening threads | Morning main | HEAD | Change |
| --- | ---: | ---: | ---: |
| 8 | 5.393 | 3.263 | -39.5% |
| 1 | 6.449 | 5.305 | -17.7% |

GPU service per control fell from 2.79 to 1.88 s (-32.6%). Receipts:
`verify/final-*`.

Paired first-pass GPU time against the fixed cadence-major shader:
-29% (56c4), -23% (6815), -19% (d8ba). The screen remains 63-69% of
device time: 0.76-0.81 s against 0.37-0.45 s for the fold.

Ten-worker complete-CLI 90/10 probe (nine 56c4, one 6815 with vetting, nice
19), interleaved B C C B against `3077ee8cf`: wall 35.79 -> 26.51 s (-25.9%),
GPU service 21.99 -> 13.27 s (-39.7%), CPU 66.2 -> 64.3 s, with identical
candidates. A ten-case batch includes worker start-up and a vetting tail.
In steady state the GPU still binds: 1.33 GPU-s per case against about 0.64
s of CPU per case per core. Receipts: `verify/mixed-*`.

## Next structural option

The remaining screen cost comes from its design: one thread per (period,
bin) keeps 2m integer accumulators and rescans for every period. Three
options remain, none yet built:

- Stage each epoch's cadence window in threadgroup memory, so the up to four
  periods of a tile share coalesced loads. This needs the shared array made
  dynamic, and it carries a barrier per epoch.
- A phase-windowed fold with per-epoch cadence ranges. Measured atomic costs
  (about 4.75 ps per integer in the fold) suggest only about 10% per visit at
  ten durations, unless fold occupancy is also limiting. That is unprofiled.
- Fewer accumulators per thread, without rescans: for example two threads
  sharing one bin's scan through SIMD shuffles.
