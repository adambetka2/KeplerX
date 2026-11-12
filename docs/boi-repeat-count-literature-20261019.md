# FPP repetition counts in recent validation papers

Primary-methods review, October 19, 2026. Adam subsequently adopted
[twenty primary final-clock repeats with conditional zero-or-three recovery diagnostics](post-boi-protocol.md).
The options below preserve the decision-support review; no new production census
was started.
The [structured review](../results/boi-validation/boi-repeat-count-literature-20261019.json)
preserves source URLs, retrieval hashes, scope and existing campaign statistics.
Years below are journal years, which can differ from preprint years.

## Explicit counts

These are repeated probability calculations, not transit-fit chains or iterations.
They must not automatically be described as final-fitted-ephemeris calculations.

| Year | Study | Repetitions explicitly reported | Scope |
| --- | --- | ---: | --- |
| 2022 | [Giacalone et al., 13 terrestrial planets](https://arxiv.org/html/2201.12661#S3) | 20 | Per TOI; mean and standard deviation. |
| 2022 | [Giacalone et al., HD 56414 b](https://arxiv.org/html/2208.06396) | 50 | Imaging-constrained TRICERATOPS; mean and standard deviation. |
| 2023 | [Wood et al., HD 109833](https://arxiv.org/html/2212.03266) | 20 | Per planet; includes companion constraints and separate multiplicity treatment. |
| 2024 | [Polanski et al., TESS–Keck XX](https://arxiv.org/html/2405.14786) | 20 | Mean and standard deviation; their admission requires three-sigma threshold clearance. |
| 2025 | [Kunimoto et al., HD 101581](https://arxiv.org/html/2412.08863) | 10 | Mean and standard deviation with MOLUSC companion inputs. |
| 2025 | [Gomez Barrientos et al., TRICERATOPS+](https://arxiv.org/html/2508.02782) | 10 | Explicit cost-based reduction from the recommended 20; comparisons add multicolour data. |
| 2026 | [Jiang et al., two temperate mini-Neptunes](https://arxiv.org/html/2512.06817) | 20 | Initial TRICERATOPS screen, followed by a separate multicolour Bayesian comparison. |
| 2026 | [Collier et al.](https://arxiv.org/html/2608.21560v3#S4.SS6) | 20, plus a final-clock recomputation | Initial count explicit; final-clock count not independently stated. |

No separate recovered/final repetition split is described in the inspected
methods of the first seven studies. That is a reporting observation, not proof
that no additional unpublished checks were performed. A comparison of data or
imaging configurations must not be relabeled as two timing sets.

[TOI-700 e](https://arxiv.org/html/2301.03617#S3.SS3) explicitly uses posterior
median period and transit time for VESPA; the independent repeat count is not
stated. [TOI-715 b](https://arxiv.org/html/2305.06206) also does not state an
independent repeat count in its statistical-validation section. Neither omission
establishes a one-run protocol. As an additional pipeline example, the
[2026 COUNTESS preprint](https://arxiv.org/html/2606.13789) uses 20 and a median;
it is not the principal precedent for strict validation here.

## Scientific interpretation

The principal probability result should use an independently justified, adequate
ephemeris. Prefer the final native-fit period/epoch when it is the best supported
timing model, rather than selecting the clock yielding the lowest FPP. This is
our scientific recommendation, not a claim that every reviewed paper uses that
exact handoff. The recovered clock supplies a sensitivity check. Both clocks
reuse photons, so their probabilities are reported separately, not pooled.

Repeated seeds assess numerical scatter conditional on fixed inputs. They do
not supply another observation or test an omitted false-positive mechanism.
With independent finite-variance estimates, the standard error of a mean scales
as 1/sqrt(N): ten repeats have about 41% larger standard error than twenty.
Sparse or heavy-tailed weights weaken that approximation. Draw count, proposal
efficiency, scenario census and observational constraints also differ between
studies, so equal repeat counts do not guarantee equal precision.

Ephemeris sensitivity matters: Collier excludes eight targets that fail its
final-clock check. Two fixed clocks still do not marginalize the full timing
posterior. The question is whether the adopted timing/model and numerical
precision support the decision, not whether an arbitrary total of forty is met.

For these three BOIs, the twenty discovery-clock repeats already exist. The
retained final-clock counts are one, two and zero for KIC 6699887, KIC 8745833
and KIC 4829494 respectively. A fixed ten-final-repeat option therefore needs
27 new integrations; ten, ten and twenty needs 37; twenty each needs 57.

The middle option is a reasonable cost/precision compromise. KIC 4829494's
existing FPP mean is 0.0133253 and NFPP mean is 0.000930834. Their approximate
standard errors are 0.00283913 and 0.0000833755, respectively, larger than their
distances to the declared cuts. Its mean rule still passes; this comparison
explains why numerical stability deserves attention before reducing repeats.
For future BOIs, twenty calculations at one adopted final ephemeris is a
literature-supported default; a separate recovered-clock budget is an explicit
robustness policy rather than a universal forty-run requirement.

Any revised count must be agreed and frozen before production. Retain all old
results, use predeclared seeds, keep full scenario/draw counts, preserve the
mean thresholds, and report scatter and bounds. Do not stop on a favorable
outcome or retroactively select a favorable subset.
