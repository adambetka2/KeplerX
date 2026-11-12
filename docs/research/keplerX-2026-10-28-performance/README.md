# V3 performance research dataroom

Research started 28 October 2026 for Adam's 406,000-light-curve campaign. This is a literature and experiment-design companion to the engineering work. Implementation and benchmark scheduling are tracked there.

## Read first

1. [An unusually concrete exact-division lead](exact-division.md): the production `0.004` divisor satisfies a sufficient condition in a 2004 arithmetic paper. This is a different algorithm from the rejected uncorrected reciprocal lookup.
2. [Sources and reading map](sources.md): 26 individually assessed sources, with applicability and architecture limits.
3. [Decision log](decisions.md): completed question-driven investigations and their dispositions.

## Verified local context

- `sysctl`: Apple M1 Pro; 16 GiB physical memory; 8 performance cores and 2 efficiency cores. Checked live.
- Read `docs/keplerX-detector-compiled-search.md`, the 27 October continuation notes, and the 28 October whitening/GPU research and proof notes. This research runs no numerical benchmarks.
- The 23.31-second breakdown is one prepared positive control with two detector passes, one candidate and Q1 pixels. It is not an estimate of the 406k population mean. Later engineering measurements use newer code and must be read in their own paired context.
- Existing implementation already includes exact NEON convolution, FFT-operand caching, MAD rank reuse, integer GPU screening with exact CPU resolution, finite-difference batching, sparse exact transit evaluation, and native pixel fitting. Many obvious wider-loop, reciprocal, histogram and caching proposals have already been tried.

Source snapshot inspected near Git HEAD `cee659b139e4c3a05ac57fcb15677f9ad397bd28`; the checkout was changing at the time. Line numbers and candidate novelty must be rechecked before implementation.

## Most useful research directions

| Direction | Specific v3 question | Evidence needed next |
|---|---|---|
| Exact division by an invariant | Can a two-operation, correctly rounded quotient replace repeated `low / 0.004`? | Theorem assumptions, signed-zero/underflow fallback, quotient/index bits, current replay ABBA. See exact-division.md. |
| Dependency-chain surgery | Which address or predicate chain actually limits the remaining native bin scan? | A short annotated assembly/dataflow graph; schedule independent work around it without changing each sum's order. S04–S06, S09–S10. |
| GPU register operand bandwidth | Is a shader limited by live operand fetches rather than occupancy or arithmetic count? | Inspect compiled shader and live ranges; reduce lifetimes or loads while preserving expressions. S07. Source spelling may compile identically. |
| Shared-cache interference | Can overlapping CPU whitening and GPU screening increase total time through shared memory/cache contention? | Same-work, same-worker comparisons of bounded overlap schedules, with separate CPU and GPU clocks. M1 cache research is a hypothesis source, not an M1 Pro guarantee. S08. |
| Instruction working set | Does unrolling or function specialization inflate the hot shader enough to counteract saved instructions? | Compiled code size and paired device timings, preserving exact outputs. S07, S13, S18. |
| Real population cost | Which paths dominate no-candidate, gapped, high-variability, multi-pass, short-period and grazing cases? | Existing real fixtures, path counts, stage times, p50/p95/max and weighted total CPU; no new calibration or survey required here. S19 and local receipts. |

The dense 16-output convolution already exposes many independent accumulators. An idealized Firestorm model gives a floor of about 8 cycles per two-tap middle iteration: 32 NEON FMUL/FADD instructions divided by four issue slots. That is a model of the inspected source, not a measurement; loads, branches, register moves and different paths can raise it. Use it to locate the gap before proposing another wider dense kernel. Do not apply it to the gapped four-output path.

## Experiment admission

Each proposed experiment should state: current baseline and exact code path; the limiting resource; the new dependency or data-layout change; the fraction of real work eligible; the numerical invariant; the cheapest falsification; and the end-to-end acceptance measurement. A source's impressive benchmark is never a v3 speed claim.

Keep worker and numerical-library thread counts fixed when attributing software improvement. Report throughput scheduling separately. Preserve the pinned operation tree, warnings/errors, edge behavior, masks and optimizer trajectories where required by the existing engine contract. Existing goldens and end-to-end replays are the principal verification evidence.

## Discovery method

Seed → inspect one precise question → read original material → follow an author/reference/code link → compare against current local work → record an experiment or rejection → rewrite the next question. Only material new findings are passed to engineering; the search stops after diminishing returns.

One article's linked quickstart could not be retrieved; the article's method itself was read. No claims depend on the inaccessible document.
