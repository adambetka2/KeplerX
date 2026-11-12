# Exact division by the production lookup grid

Status: **new research hypothesis; no implementation or speed measurement yet.**

## Why this differs from an ordinary reciprocal optimization

The inspected `trashdump/kernels.c::v3_bins_mes` repeatedly computes `floor(low / grid) - 1`, including an ARM64 `vdivq_f64` route. `trashdump/gpu.py` sets the production `GRID_DAYS` to `0.004`. Earlier experiments using a simple reciprocal and conservative lookup start were slower. They did not test the following exact quotient algorithm.

Brisebarre, Muller and Raina's [2004 paper](https://perso.ens-lyon.fr/jean-michel.muller/DivIEEETC-aug04.pdf), Algorithm 2 and Theorem 2, supplies a sufficient condition for correctly rounded division with one multiplication and one FMA after a divisor is prepared. In round-to-nearest-even arithmetic, let:

```
zh = RN(1 / y)
zl = RN(1 / y - zh)   // exact difference before the final rounding
q1 = RN(x * zl)
q2 = FMA(x, zh, q1)
```

If the last significand bit of `y` is zero, the theorem establishes `q2 = RN(x / y)` in the paper's domain. It assumes no overflow or underflow. The explicit FMA in this replacement implements the division algorithm; it does not license contraction of other operations in the detector.

## The useful coincidence

The actual binary64 encoding of the decimal literal qualifies:

| Quantity | Exact binary64 representation |
|---|---|
| `y = 0.004` | `0x1.0624dd2f1a9fcp-8` |
| Bits of y | `0x3f70624dd2f1a9fc` |
| Last significand bit | **0** |
| `zh` | `0x1.f400000000000p+7` = 250 |
| `zl` | `-0x1.7700000000000p-48` |

Checked using only Python `struct` and exact `fractions.Fraction`: `zl = float(1 / Fraction(y) - Fraction(zh))`. This certifies the prepared constants, not the implementation or the theorem's complete applicability to v3.

The candidate scalar expression is therefore conceptually `fma(x, 250.0, RN(x * (-0x1.77p-48)))`. A NEON version would use a separately rounded multiply followed by explicit lane-wise FMA. Merely writing `x * 250` is a different algorithm.

## Minimal experiment for the engineering owner

1. Confirm this division still dominates a meaningful fraction of the latest captured CPU resolver. Recheck the live source; it is being edited concurrently.
2. Specialize only the exact binary64 value `grid == 0.004`. Keep the original path for other grids, unsupported rounding modes, signed zero, nonfinite inputs and values outside a proved normal-intermediate domain. For an initial bounded proof, `1e-12 <= abs(x) <= 1e6` keeps these products comfortably normal; tighter guards may be possible after inspection. Apply the paper's sign symmetry carefully rather than assuming unsigned inputs.
3. Preserve separately rounded `x*zl` and explicit FMA. Inspect emitted instructions and strict flags; do not enable fast math. The quotient must retain its original `floor`, int64 conversion and `-1`, and every cadence/bin check stays unchanged.
4. Compare division bits first on captured `low` values and adversarial values around `j*grid`, their neighboring floats, positive/negative zero, very small inputs, fallback grids and representable integer-quotient boundaries. Sampling supports the theorem; it does not replace the domain proof.
5. Compare the exact existing resolver's indices, sums and outputs on the same frozen replay. Time baseline/candidate/candidate/baseline at identical resources, counting preparation and branch costs. Reject if no repeatable component improvement; run the established full goldens/end-to-end gates only if it passes.

Do not replace the old rejected reciprocal result in the record. This experiment has a distinct proof and instruction sequence and needs a distinct receipt.

## Why this might still lose

[Dougall Johnson's Firestorm measurements](https://dougallj.github.io/applecpu/firestorm-simd.html) put FP64 vector divide at latency 10 cycles and reciprocal throughput 1 cycle, while FP64 vector multiply and FMA each have latency 4 cycles and reciprocal throughput 0.25 cycles. Two dependent operations have an ideal latency of 8 cycles before guard and load costs. The two operations also compete with other SIMD arithmetic. The current batched loop already overlaps many independent divides. There is no justified large speedup prediction.

[Reynolds' implementation note](https://marc-b-reynolds.github.io/math/2019/03/12/FpDiv.html) is a useful code-oriented entry point but uses float32 examples. The binary64 proposal must follow the original theorem. The follow-up [Louvet/Muller/Panhaleux FMA division paper](https://perso.ens-lyon.fr/jean-michel.muller/ASAP2010-a.pdf) is a reference for the proof conditions, not a reason to introduce a full software divider.

## Iteration 1: a proof that may avoid extra per-value guards

The original [supplementary proof](https://perso.ens-lyon.fr/jean-michel.muller/supplementary_117197.pdf), pages 7–8 (PDF indices 6–7), confirms the even-significand case through the absence of the dangerous rounding-breakpoint solution. No implementation receipt for this exact sequence was found in the owner's current messages or targeted research files at this check. That is a query result, not a claim about unrecorded work.

**New inference from the actual v3 expression:** the first proposed `abs(x) >= 1e-12` guard is unnecessarily restrictive for `v3_bins_mes`'s internally computed `low`. Both inspected routes compute

```
a = RN(RN(k * period) + start)
low = RN(a - epsilon), epsilon = binary64(1e-6)
```

The native flags are `-ffp-contract=off -fno-fast-math`; the NEON route also explicitly separates multiply/add/subtract. Under round-to-nearest-even, any finite `low` here is either **positive zero** or has magnitude at least `2^-73`:

- `2^-20 < epsilon < 2^-19`. For `epsilon/2 <= a <= 2*epsilon`, subtraction is exact by the standard factor-of-two subtraction lemma. Both operands are integer multiples of `2^-73`, so a nonzero difference has at least that magnitude.
- Outside that interval, the exact difference has magnitude at least `epsilon/2`, vastly larger than this bound; final rounding cannot make it tiny. This includes negative `a` and very small `a`.
- Zero is possible only for `a == epsilon`; the exact cancellation produces `+0` in this rounding mode. There is no negative-zero result or underflow-generated zero in this expression.

Consequently, within the **existing vector branch's** finite range `-1e6 <= low <= t[n-1] <= 1e6`, multiplying a nonzero `low` by `zl=-0x1.77p-48` has magnitude above `2^-121`. The main product and quotient are also normal and finite. For `low=+0`, the low product is `-0` and `FMA(+0, +250, -0)` is `+0`, matching division by positive `0.004`.

This suggests restricting the experiment to that already-guarded vector branch: select the exact-grid specialization once outside the epoch loop, retain ordinary division elsewhere, and avoid adding a per-lane tiny-value/zero test. The rounding-mode precondition still belongs to the overall numerical contract. The proof concerns these internally generated `low` values, not arbitrary external inputs and not unrelated divisions. It is a researcher-derived argument for the engineering owner to review, not an implementation verification or a speed claim.
