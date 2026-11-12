# Research decisions

## Iteration 1 — exact invariant division and guard cost

Question: can the two-operation exact quotient be worthwhile after guard overhead?

Read S01's original supplementary proof (S26), especially Theorem 2's proof on PDF pages 7–8. Confirmed the even-significand condition. Inspected the current `v3_bins_mes` expression and native strict flags at HEAD `cee659b139e4c3a05ac57fcb15677f9ad397bd28`. Targeted owner messages/files contain no exact-reciprocal-pair implementation receipt yet; ordinary reciprocal probes are distinct.

New actionable refinement: `low` is always a rounded subtraction of the fixed binary64 `1e-6`. Under the current RN-even/strict-operation assumptions, it is `+0` or has magnitude at least `2^-73`. Together with the existing vector range gate this proves normal intermediates for mul+FMA, and the `+0` case also matches division. Thus the initial proposed per-value magnitude/zero guards may be unnecessary inside that route. Full argument is appended to exact-division.md; engineering owner must review it.

Disposition: forward this narrower guard-free-vector experiment to the engineering owner; no production edit, numerical experiment or speed claim here. Cheapest falsification remains quotient/index comparison on the owner's captured replay, then current resource-matched ABBA. Keep fallback division outside the proven route.

Next question: can original Apple GPU operand-bandwidth/code-working-set research produce a distinct, provably equivalent expression/lifetime change in the current double-float screen, beyond the owner's already rejected occupancy/histogram probes?
