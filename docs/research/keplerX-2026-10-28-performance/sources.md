# Reading map: 26 assessed sources

Read on 28 October 2026. The descriptions are short research annotations, not transcriptions. `New lead` means new to this research comparison, not a claim of novelty in computer science. `Covered` means the active agents already cite or investigate closely related material. Links point to authors, original papers, maintainers or official documentation. Hardware results do not transfer automatically.

## Arithmetic and dependency-chain techniques

**S01 — New concrete lead.** Brisebarre, Muller, Raina, [Accelerating Correctly Rounded Floating-Point Division when the Divisor Is Known in Advance](https://perso.ens-lyon.fr/jean-michel.muller/DivIEEETC-aug04.pdf), 2004. Algorithm 2 / Theorem 2 is directly relevant to repeated division by `0.004`, whose binary64 significand is even. Read [exact-division.md](exact-division.md) before trying it. The paper's normal-range assumptions and rounding requirements are binding.

**S02 — Proof follow-up.** Louvet, Muller, Panhaleux, [Newton-Raphson Algorithms for Floating-Point Division Using an FMA](https://perso.ens-lyon.fr/jean-michel.muller/ASAP2010-a.pdf), 2010. Distinguishes mathematically equivalent iteration schemes that have different rounding behavior, and develops conditions for correct rounding. Useful when checking an arithmetic fast path; does not establish that a software divider beats M1 hardware.

**S03 — Code-oriented bridge to S01.** Marc B. Reynolds, [Floating point division with known divisor](https://marc-b-reynolds.github.io/math/2019/03/12/FpDiv.html), 2019. Gives unevaluated reciprocal-pair and corrected-quotient implementations and a practical caution about modern divider performance. Its float32 examples need deliberate translation to the pinned float64 engine.

**S04 — Obscure M1 worklog.** Dougall Johnson, [Reading bits with zero refill latency](https://dougallj.wordpress.com/2022/08/26/reading-bits-with-zero-refill-latency/), 2022. The useful technique is to issue work against an older valid state while preparing the next state, removing an avoidable dependency from a lookup chain. Inspect native bin-position and predicate chains for an equivalent dependency split. This is a scheduling analogy; no decoder code belongs in v3.

**S05 — Follow the actual dependency, not the source line count.** Johnson, [Faster zlib/DEFLATE decompression on the Apple M1 (and x86)](https://dougallj.wordpress.com/2022/08/20/faster-zlib-deflate-decompression-on-the-apple-m1-and-x86/), 2022. Detailed assembly, shifted loads, table layout and overlap of branch recovery with independent work. Its linked dataflow article led to S06. Compression gains are irrelevant unless an actual v3 input path spends time decompressing.

**S06 — Model-building foundation.** Fabian Giesen, [A whirlwind introduction to dataflow graphs](https://fgiesen.wordpress.com/2018/03/05/a-whirlwind-introduction-to-dataflow-graphs/), 2018. Draw loop-carried dependencies and calculate a plausible throughput limit before benchmarking more variants. In v3, preserve the sequential floating recurrences but move independent address, predicate or neighboring-output work off the critical chain. Its illustrative machine latencies are not M1 measurements.

**S09 — M1 queues and alias prediction.** Johnson, [Apple M1: Load and Store Queue Measurements](https://dougallj.wordpress.com/2021/04/08/apple-m1-load-and-store-queue-measurements/), 2021. Original experiments distinguish queues, scheduling and release behavior. Useful if annotated assembly shows a load/store issue limit despite resident data. Dense convolution currently has mostly read-only operands inside its hot middle; do not invent a store-alias bottleneck without evidence.

**S10 — Resource ceilings.** Travis Downs, [Performance Speed Limits](https://travisdowns.github.io/blog/2019/06/11/speed-limits.html), 2019. Works through execution ports, load throughput, dependency chains, buffers and register pressure. Use its methodology with Firestorm data, not its Intel-specific constants. It gives reasons a shorter or more vectorized loop can lose.

**S11 — Rearrange values without changing arithmetic.** Giesen, [SIMD transposes 1](https://fgiesen.wordpress.com/2013/07/09/simd-transposes-1/), 2013. Shows multiple equivalent interleave networks for AoS/SoA conversion. Relevant to gapped-output packing and batched independent model evaluations. Current whitening agents already implemented related transposition; new work must remove a measured shuffle/load cost rather than rediscover packing.

**S12 — Decomposing transpose networks.** Giesen, [SIMD transposes 2](https://fgiesen.wordpress.com/2013/08/29/simd-transposes-2/), 2013. Extends the decomposition to larger networks and even/odd layouts. Architecture-neutral lane mappings are useful, but old ARM32 two-register instruction examples need re-costing against actual AArch64 instructions.

**S13 — Static scheduling tool.** LLVM maintainers, [llvm-mca](https://llvm.org/docs/CommandGuide/llvm-mca.html). Can model resource pressure, dependencies and throughput of extracted assembly. Check that the installed build has a suitable Apple scheduling model; model quality bounds the answer. Useful for rejecting poor loop schedules before a full replay. It does not model all caches or replace hardware timing.

**S14 — Exactness constraints on vectorization.** LLVM maintainers, [Auto-Vectorization in LLVM](https://llvm.org/docs/Vectorizers.html), especially reductions and SLP. Distinguishes vectorizing independent expressions from reassociating a reduction. This supports the engine's current approach: pack independent outputs while retaining each dot product's original lane and addition order. Automatic vectorization diagnostics are supporting evidence, not proof of equivalence.

**S15 — Certified filters.** Jonathan Shewchuk, [Adaptive Precision Floating-Point Arithmetic and Fast Robust Predicates](https://www.cs.cmu.edu/~quake/robust.html), 1996–1997. The transferable pattern is cheap certified decisions with an exact fallback for uncertain cases. The current GPU screen already follows this philosophy. Any new filter must certify the pinned detector's comparisons, including ties, rather than merely the exact-real mathematical result.

## Apple GPU and memory details

**S07 — Covered source, deeper operand-level reading.** Philip Turner, [Metal Benchmarks](https://github.com/philipturner/metal-benchmarks), sections ALU Bottlenecks, Instruction Throughputs and On-Chip Memory. Original M1/M2 microbenchmarks discuss register-operand dependence and instruction working sets, beyond the usual occupancy advice. Candidate questions: shorten live ranges or remove repeated loads without changing expressions; check whether unrolling inflates code. Source variants may compile identically. Do not infer undocumented bank mappings from these results.

**S08 — Obscure cache-topology lead.** Xu, Ding, Fei, [EXAM](https://arxiv.org/html/2504.13385v1), 2025, sections 3.2–3.4 and Appendix A. Original reverse engineering studies CPU/GPU system-cache sharing and its indexing/inclusion behavior. Performance implication to test: overlap can introduce cache interference even when CPU and GPU clocks each look healthy. The experiments use a base M1 MacBook Air; M1 Pro topology must be checked independently. No cache-attack implementation is needed for v3.

**S16 — Covered, directly targeted to the M1 family.** Apple, [Scale compute workloads across Apple GPUs](https://developer.apple.com/videos/play/wwdc2022/10159/), WWDC22. Covers command gaps, dispatch ramp/tail costs, atomics and the memory-locality/divergence tradeoff. Current GPU agents have already explored histogram variants and queue depth. Use this to interpret a trace rather than repeat those probes.

**S17 — Covered generation-appropriate baseline.** Apple, [Optimize Metal Performance for Apple silicon Macs](https://developer.apple.com/videos/play/wwdc2020/10632/), WWDC20, shader-core section. Useful for register footprint and work scheduling. Its graphics-specific tile-memory examples are not directly applicable to the compute detector. Quantized sums require their current integer range; changing value precision needs a new proof.

**S18 — Useful with a generation boundary.** Apple, [Learn performance best practices for Metal shaders](https://developer.apple.com/videos/play/tech-talks/111373/). Function constants and shorter live ranges are worth understanding. Advice about Family 9 dynamic shader-core memory and its cache hierarchy applies to M3/A17 Pro, not this M1 Pro. The existing GPU agent already records this source; do not transfer the wrong-generation memory advice.

**S20 — Layout sensitivity.** Sergey Slotin, [Cache Associativity](https://en.algorithmica.org/hpc/cpu-cache/associativity/). Concrete stride experiments explain capacity lost to conflicting cache sets. Candidate work is on scratch-row strides and buffer placement only; changing scientific padding lengths would change the algorithm. Generic low-bit cache indexing is insufficient for Apple's SLC; S08 makes that limitation concrete.

## Measuring the whole program and the campaign

**S19 — Causal rather than descriptive profiling.** Curtsinger and Berger, [COZ: Causal Profiling](https://web.cs.umass.edu/publication/docs/2014/UM-CS-2014-010.pdf), original technical report. Shows why a large time attribution need not translate to end-to-end savings when another path becomes limiting. Use the question it asks to evaluate CPU/GPU overlap. The prototype described is Linux x86-64; this is not a proposal to install it on macOS.

**S21 — Followed directly from Adam's seed.** Horace He, [Strangely, Matrix Multiplications on GPUs Run Faster When Given Predictable Data](https://www.thonking.ai/p/strangely-matrix-multiplications). Original experiments explore input-dependent power effects on NVIDIA GPUs. It strengthens the case for real light-curve inputs and sustained measurements. It does not prove the same effect or magnitude on M1 Pro, and zero-filled synthetic kernels are not a campaign benchmark.

**S22 — Followed directly from Adam's seed.** Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM), 2022. Particularly useful: derive compute/data-movement floors, then change one layout/schedule at a time and keep code for each step. CUDA tiling constants, tensor cores and arithmetic reassociation do not port to the exact v3 CPU kernels.

**S23 — Seed.** Aleksa Gordic, [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul). Hardware → programming model → memory → assembly → measured kernel is the standard of technical depth for this search. Followed its Boehm and Horace references. The Feldmann shared-memory link timed out and is kept as unverified in the scratchpad.

**S24 — Survey ingestion caveat.** Astropy maintainers, [FITS FAQ](https://docs.astropy.org/en/latest/io/fits/appendix/faq.html), performance and table-reading sections. Header parsing, table conversion and full-image reading have different costs; memory mapping also has lifetime behavior. Relevant only after measuring the actual input loader. Do not replace the FITS backend based on an image benchmark or include new photometry work.

**S25 — Algorithm-selection boundary.** SciPy maintainers, [choose_conv_method](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.signal.choose_conv_method.html). The auto choice predicts performance from shape/type, and explicit measurement can choose another method. The current engine intentionally reproduces the pinned choice. Retuning direct versus FFT may change rounding and is not an admissible exact optimization by itself. Library version in this page is not asserted to equal the local pinned version.

**S26 — Original proof, followed from S01.** Brisebarre, Muller, Raina, [Supplementary material: tables, proofs and intermediate lemmas](https://perso.ens-lyon.fr/jean-michel.muller/supplementary_117197.pdf), 2003. Pages 7–8 contain the proof of Theorem 2, including why an even significand eliminates the problematic breakpoint equation. Read to check the exact-grid proposal. A separate local expression analysis now gives a normal-range bound for v3's generated `low` values; that bound is our inference, not a theorem attributed to this paper.

## Unverified / not admitted

- Feldmann's `https://feldmann.nyc/blog/smem-microbenchmarks` timed out. Do not summarize it from the seed's endorsement.
- The original Stabilizer site was inaccessible and the author's slide PDF returned 403. Third-party abstracts were discovered, but no claims here rely on them. Retry an author-hosted paper if layout bias becomes the active research question.
- A guessed OpenBLAS `dot_armv8.c` URL returned 404. The active whitening agent has already checked its pinned `dot.S` source; use that evidence instead.
- The pasted article's Google quickstart was inaccessible. The supplied full text establishes the requested research method.
