# BOI validation follow-up

## Current status — October 27, 2026

Ten planets are statistically validated: six on their targets and four
signals validated by multiplicity, host star undetermined. The four
signals validated by multiplicity are the three
M/K signals of KIC 6267058 ([M/K page](mk-statistical-validation.md#v3-campaign-october-25-2026))
and KIC 6699887's 47.69-day signal, a 2.5 R⊕ sibling of the validated KIC 6699887 b
(V3 mean FPP 2.12%, 0.39% after the boost; NFPP 3e-13;
[record](../results/boi-validation/kic6699887c-multiplicity-validation-20261025.json)). On
October 25, 2026, Adam adopted `TRICERATOPS_KEPLER_V3` ([changes](triceratops-v3.md))
for statistical validation and accepted KIC 8878385 b; see the
[adoption record](../results/boi-validation/v3-adoption-20261025.json). The three
October 19 studies are unchanged.

| BOI | Host | Period, days | Radius, R⊕ | Mean FPP | Mean NFPP | Protocol |
|---|---|---:|---:|---:|---:|---|
| 2077554781499350144-1 | KIC 6699887 | 15.11 | 2.66 | 0.696% | 0.0363% | V1 |
| 2127186564572468352-1 | KIC 8745833 | 4.57 | 4.97 | 0.455% | 2.67e-50% | V1 |
| 2101047904706267264-1 | KIC 4829494 | 16.11 | 7.51 | 1.333% | 0.0931% | V1 |
| 2127169041106093184-1 | KIC 8878385 | 5.905 | 3.45 | 0.276% | 5e-23% | V3 |
| 2127936912539397632-1 | KIC 10528996 b | 13.745 | 2.84 | 0.659% | 9e-99% | V3, PSF_FLAT |
| 2127936912539397632-2 | KIC 10528996 c | 8.556 | 2.05 | 0.574% | 7e-9% | V3, PSF_FLAT |

KIC 10528996 b and c (October 27) come from the shallow detector, so they are
not among the 125 survey BOIs below. They ran under V3 on KBonus PSF_FLAT
photometry with the psf-envelope-v2 neighbour response (v1 agrees), 20 seeds
each, and pass every closeout check: S/N 48 and 24, RUWE 1.05, no Gaia source
within 1.5 arcsec, radius below 8 R⊕, constant seasonal depth, and no KOI,
TCE, TOI, known planet or Bienias and Szabó 2025 match. The 7.1-arcsec
neighbour KIC 10528990 shows 53 ± 10 ppm at b's clock in its own Kepler light
curve, where hosting b needs 1,486 ppm. Adam accepted them on October 27: a
pass of the protocol is a validated planet. See the
[record](../results/boi-validation/kic10528996-psf-validation-20261027.json),
the [admission packet](../results/boi-validation/psf-admission-packet-2127936912539397632-20261027T161844Z.json)
and [PSF_FLAT post-BOI work](psf-flat-post-boi.md).

KIC 8878385 passes all 20 V3 repeats (FPP 0.13–0.52%) and its closeout: SNR 31,
RUWE 1.03, no Gaia source within 1.5 arcsec, no low-order ephemeris match on
another star and no phase-0.5 secondary (3.7 ± 11.8 ppm). Stock TRICERATOPS
fails it on NFPP (2.9% and 6.3%) from a Tmag 14.5 neighbour; stock reruns are
diagnostics, not an admission requirement. Statistical validation needs no
imaging or RV mass.

TRICERATOPS cannot validate objects of 8 R⊕ or more, because photometry cannot
separate planets from brown dwarfs (Gomez Barrientos et al. 2025). KIC 7373485
(12.8 R⊕, FPP 0.59%) and KIC 8293530 (16.2 R⊕, FPP 0.16%, a Bienias and Szabó
2025 candidate) are Possible Planets. The six other completed V3 campaigns, with
mean FPP 1.9–30% and NFPP below 0.1%, meet the same paper's Possible Planet
criteria; the adoption record lists them.

Of four further BOIs chosen on October 25, none reached FPP. KIC 3330404 and
KIC 9220619 stopped at the native radius-ratio rule (quantile SNR at least 3):
KIC 3330404 at 0.14, a small-planet versus grazing-body degeneracy, and
KIC 9220619 at 2.99. KIC 6699887's 47.69-day signal was held for a scene source
without Gaia photometry; with that source's TIC Tmag flux it passes flux closure
in all 17 quarters and is validated by multiplicity (above). KIC 7300018 runs under
the same rule. KIC 5303927 waits on a host model, since no catalogue gives its
mass or radius.
The other overnight BOIs completed above the thresholds, stopped after the
first two repeats, or were held before FPP; see the
[overnight record](../results/boi-validation/v3-kbonus-overnight-20261025.json).

### Dispositions of all 125 BOIs

Every delivered BOI has a current category in the
[disposition record](../results/boi-validation/survey-dispositions-20261025.json)
and its [table](../results/boi-validation/survey-dispositions-20261025.csv). None
is still queued under V3:

| Category | BOIs |
|---|---:|
| Validated (four on target, one by multiplicity) | 5 |
| Kepler-565 b recovered, and its two aliases | 3 |
| Denied through reproduced EB contamination (October 12) | 2 |
| Possible Planet | 12 |
| Stopped after the first two repeats | 5 |
| TRICERATOPS false-positive category | 4 |
| Alias of a deeper BOI on the same star | 28 |
| Held before FPP | 65 |
| Not applicable to V3 | 1 |

Of the 28 aliases, 27 have periods commensurate within 0.1% with a deeper BOI
on the same star, mostly P/3 to P/11 with diluted depths. In the other,
KIC 3975580's 1.46-day signal has no depth (−13 ppm) once the 4.41-day BOI's
transits are masked. Holds name the V3 step: 15 hosts still have no mass and
radius after the isochrone fits below, and 8 have no authenticated scene census.
The rest are aperture, closure, capable-source or native-fit limits. In two,
every native posterior sample has periastron inside the star, which excludes a
planet on that star. These are the 3.05-day BOI-2073285721453427712-1 on an
8.9 R⊙ giant (a/R* ≈ 1.1) and the second generation of the 28.43-day
BOI-2076476504190366336-1 on a 4.9 R⊙ subgiant. That fit wants a stellar
density of 0.7–1.2 kg/m³ against the FLAME 13.8 ± 7.4 kg/m³ and eccentricity
0.71–0.78. No other mechanism has been reproduced. The 105.37-day
BOI-2104484363877066752-1 on its isochrone subgiant (2.72 R⊙) fits a grazing body
(b 1.0–1.6) with radius ratio 0.36–0.90, 1.0–2.5 R⊙, as a grazing eclipsing
binary would. It is held at the radius-ratio rule. Nine more grazing fits with
unconstrained radius ratios are held the same way: BOI-2079536067092694016-1
(0.27 days), BOI-2100203999471640064-1
(3.14 days), BOI-2100123494603868416-1 (7.11 days), BOI-2076894554826018304-1
(7.65 days), BOI-2126354436132009088-1 (18.67 days), BOI-2130658616133595264-1
(32.95 days), BOI-2050231500928700032-1
(39.81 days), BOI-2073628180652139904-1 (55.52 days) and BOI-2101435138957164672-1
(80.72 days). The 0.27-day fit orbits at 1.32–1.42 stellar radii. A body on its
6.53-hour orbit needs a mean density of at least 3.7 g/cm³ to stay outside its
Roche limit (Rappaport et al. 2013), far above that of a gas giant of the fitted
18–37 R⊕. The first fit of
the 63.22-day BOI-2052563599452804096-1 found a 25-hour transit that matches its
9.3 R⊙ isochrone giant, and failed only protection. Its second generation turns
grazing at a density about 60 times below the host's, with a median 52-hour
transit. post_boi.native then stops at its geometry check. No third generation
was staged. The 11.258-day signal on
Gaia DR3 2100442657917279616 matches a contaminating binary 116–152 arcsec away,
outside the 60-arcsec V3 scene.

The repeats of two Possible Planets span both validation and false-positive
values: KIC 5115632 (FPP 0–79%, mean 18%) and KIC 4543190 (0 or 100%, mean 30%).
Their category rests on the mean of numerically unstable integrations.

The 22.13-day BOI-2102612750272489216-1 (9.2 R⊕) reached FPP through a second
generation. Its first fit failed protection by one cadence, because the
detector's 3.11-hour window was shorter than the fitted 4.75-hour transit. The
second generation protects the fitted transit and gives the same duration. Its
20 repeats give FPP 5.7–25% (mean 14.9%) and NFPP below 1e-97, a Possible Planet.
Its closeout has S/N 36, RUWE 1.04, no Gaia source within 1.5 arcsec and no
phase-0.5 secondary (−128 ± 204 ppm). Its three sibling BOIs are its P/3, P/5
and P/7 aliases.

Four BOIs stopped after their first two repeats in TRICERATOPS's statistical
false-positive category; none is a reproduced mechanism.

- BOI-2073296372971801728-1 (25.65 days) has FPP 94% and NFPP 74%. Its largest
  scenario is an eclipsing binary on Gaia DR3 2073296372962633600 (35%).
- BOI-2099197705814100480-1 (96.87 days) has FPP 100% and NFPP 0. Its largest
  scenario is an eclipsing binary on the target (84%), and its radius ratio of
  0.099 implies a 28.6 R⊕ body.
- BOI-2102918384442366592-1 (9.69 days) has FPP 100% and NFPP 0. Its native
  radius ratio of 0.71–0.96 implies a star-sized body. Its largest scenario is an
  eclipsing binary on the target at twice the period (50%).
- BOI-2103418559153168128-1 (198.16 days) has FPP 99.9% and NFPP 99.8%. Its
  largest scenario is an eclipsing binary on Gaia DR3 2103418559153166336 (89%).

On October 25, the survey preparation staged KIC 4845668 (Gaia DR3 2076468876327949568) with
a target-only scene. Its census row records a sealed-record hash mismatch and no
Gaia query. Its native fit was stopped after 17 minutes, before any
probability run. Survey preparation now holds such rows before acquisition.
The validated survey hosts all have exact-match census rows with queried scenes.

The eight scene censuses cannot be authenticated from the Mac. The census
builder searched only the sealed survey's `psf_flat` vetting batches, and four of
these BOIs come from qdiag views. The sealed records are on the kbonus server,
which no longer answers, and the frozen manual copies
(`dump/boi-manual-125-20261011`) are on neither connected disc.

### Isochrone hosts, October 26

On Adam's decision, hosts without Gaia FLAME or TIC mass and radius now get
isoclassify grid-mode solutions with Berger et al. (2020) inputs: the MIST grid,
the Combined19 dust map and the zero-point-corrected Gaia DR3 parallax. The
photometry is Gaia, 2MASS and Pan-STARRS1 griz (transformed with Tonry et al.
2012). The rules were set before fitting. A host needs parallax S/N of at least
5, RUWE of at most 1.4 and one clean colour. A solution is used when the mass and
radius half-intervals are at most 25%.

Two passes covered 41 held M/K and survey hosts: 18 qualified and 15 solved. The
giant and subgiant agree with TIC's radii (9.29 against 9.23 R⊙, 2.72 against
2.69 R⊙). Three hosts have no single-star solution: their colours and luminosity
fit no single model, as for a blend or unresolved binary. One second-pass host
came back with 0.6–0.9% mass and radius errors. The adopted errors therefore add
the model-systematic floors of Tayar et al. (2022) in quadrature: 5% in mass, 4%
in radius and 2% in Teff. This version 2 widens priors only and changes no
acceptance.

The solutions replace only the missing mass, radius and Teff, in both the native
prior and the scene. Seven of 21 cases were queued; six were re-planned with
version 2 before their fits started, and one had already been fitted with version 1.
The other 14 stop at later V3 checks: 12 fail flux closure, as blended hosts do,
one lacks TIC reddening and one has a neighbour without stellar properties. See
the [isochrone record](../results/boi-validation/isochrone-hosts-20261026.json).

### Faster probability runtime, October 26

Adam chose `fast_triceratops.engine()` ([notes](fast-triceratops.md)) for the
queued campaigns. The engine's exact mode replays two seeds of KIC 5458397's
completed M/K campaign bit for bit: every scenario's lnZ, the scenario and star
tables, FPP and NFPP. From 23:27 UTC on October 25, each new campaign runs through
`fast_triceratops.workflow` with five single-thread workers, the candidate's CPU
budget, in `probability-v3/campaign-engine/`. The frozen plan, seeds, first-two
screen and thresholds are unchanged; `post_boi.campaign.finish` writes the
audit. See the
[adoption record](../results/boi-validation/engine-adoption-20261026.json).

To use the whole M1 Pro (eight performance and two efficiency cores), the
scheduler keeps two five-core jobs running. It starts a third whenever measured
CPU idle time leaves at least three cores free. A native fit can run single-threaded
for hours, and engine campaigns use two workers during their first-two screen, so a
fixed two-job limit left four cores idle.

### Faster native fits, October 26

Adam asked for the prepared fast tools in every validation stage. Native fits
started after 01:58 UTC on October 26 run batman with the
[native orbit kernel](fast-triceratops-native.md), frozen from commit 2a7c8c86.
juliet's dynamic-dynesty fit spawns fresh pool processes, which a kernel enabled
in the parent never reaches. So `run_native.sh` puts a `sitecustomize` module on
the fit's `PYTHONPATH`; it enables the kernel as each process starts and records
it in `native-v3/native-kernel/`. A process that cannot enable the kernel stops
instead of falling back to stock batman.

A paired control replayed two completed production fits, BOI
2076383384993192576 and M/K 2102521666898377600-2, end to end with stock batman
and with the kernel. All three outputs (production, stock replay, kernel replay)
are bitwise equal in every posterior array, dynesty result field and text output.
CPU fell by 1.28x and 1.34x and elapsed time by 1.03x and 1.17x: the fits keep only
about three of their six processes busy, so dynesty's coordinating process paces
them. The first queued fit with the kernel, BOI-2073666564790289664-1, took 29
minutes; its 20 engine repeats then took 4 minutes (FPP 14%, a Possible Planet).
Probability campaigns keep the exact engine. Certified mode, which is not
exact by construction, stays unused. See the
[adoption record](../results/boi-validation/native-kernel-adoption-20261026.json).

## October 24, 2026 status

Three BOIs are accepted statistical validations. On October 19, 2026, Adam
accepted their studies at the adopted inputs; see the
[adoption record](../results/boi-validation/post-boi-protocol-adoption-20261019.json).
Each study completed 20 independent discovery-clock repeats with 1,000,000
attempted draws per scenario family, and passes the mean cuts FPP <1.5% and
NFPP <0.1%.

| BOI | Host | Period, days | Mean FPP | Mean NFPP | Study |
|---|---|---:|---:|---:|---|
| 2077554781499350144-1 | KIC 6699887 | 15.11 | 0.696% | 0.0363% | This page |
| 2127186564572468352-1 | KIC 8745833 | 4.57 | 0.455% | 2.67e-50% | This page |
| 2101047904706267264-1 | KIC 4829494 | 16.11 | 1.333% | 0.0931% | [October 19](boi-further-study-20261019.md) |

For the first two signals, these means come from the later 20-repeat campaign;
the October 12 nominal values below remain preserved. KIC 4829494 passes only
on the mean: 7 of 20 repeats are at or above each cut, and its target-planet
effective draws are 2.7–8.4 per repeat. None of the three has a complete
final-clock census; the [method follow-up](boi-method-followup-20261019.md)
records that the acceptance does not require one for these studies. All three
remain conditional statistical validations, without RV confirmation or a
novelty claim.

**KIC 8878385 (BOI-2127169041106093184-1) is not counted.** Its October 12
admission below remains preserved, but it never completed the 20-repeat
protocol and is not among the accepted studies. The October 21 standard study
held it before native fitting or any FPP run: in quarters 2, 3, 6 and 7, its
seasonal dilution and response allocation differ materially from the
calibrated response. Its disposition in that study is REMAINS_PC; no false
positive is established. See the
[hold record](../results/boi-validation/local-kic8878385-standard-hold-20261021.json).

No validation has been added since October 19. The local survey-BOI loop added
none; see the [local handoff](../results/boi-validation/local-24h-state-20261021.json).
The server's M/K campaign completed 112 probability integrations across 47 M/K signals:
46 stopped after the first two repeats, and one completed all twenty but failed
the numerical criteria; see the
[server totals](../results/boi-validation/server-continuation-20261021-v02.json).
Both efforts ended on October 21, 2026.

## October 12, 2026 follow-up

This follow-up statistically validates **three BOIs**, recovers **Kepler-565 b**,
and denies **two supplied waveforms** through reproduced eclipsing-binary
contamination. No archival RV confirmation or planet mass is established.
The server continuation has not admitted another validation or denial.

The [125-entry catalogue and harmonic rerun](../results/boi-validation/boi-catalogue-flags-20261012.json)
adds no flag to the three validated signals and changes no FPP or admission.
Sixteen of its 17 catalogue associations were already recorded in follow-up;
the additional nearby DR25 TCE association remains source-unconfirmed.

The October 11 historical selection contains 126 entries on 95 Gaia DR3 hosts. The
[historical delivery](../results/boi-validation/boi-delivery-selection-20261011.json) contains 125 entries on
94 hosts after Adam excluded BOI-2103289985012370304-1, original page 86.
This manual exclusion is not a scientific denial. Original inputs, masks,
fits, thresholds, and discovery labels remain preserved.

On October 26 the PI restored every main Rust BOI to current catalogue
membership: 147 groups, 156 proposals, 116 sources. The
[current selection](../results/boi-validation/boi-current-selection-20261026.json)
includes the 33 initial manual exclusions and the later removal named above.
The following tables describe the historical follow-up subset; they are not
the full Rust catalogue or its completeness denominator. Existing validation
outcomes remain unchanged. See [survey results](survey-results.md).

| Follow-up disposition | Current delivery | Original selection |
|---|---:|---:|
| Newly statistically validated | 3 | 3 |
| Previously validated planet recovered | 1 | 1 |
| Reproduced aliases of that planet | 2 | 2 |
| Supplied waveform denied through reproduced EB contamination | 2 | 2 |
| Unresolved with a reproduced contamination component | 2 | 2 |
| Other unresolved entries | 115 | 116 |
| Total | 125 | 126 |

The [CSV ledger](../dump/boi-validation-20261011/validation-ledger.csv) and
[structured ledger](../dump/boi-validation-20261011/validation-ledger.json)
retain every exact signal, disposition, evidence, and remaining requirement.
The current_delivery_included field selects delivered entries; page retains
the original page and current_delivery_page records the updated page.

## Newly statistically validated planets

All three decisions pass the unchanged FPP <1.5%, resolved NFPP <0.1%, host RUWE,
native-inference, and retained vetting requirements of the
[requested framework](https://arxiv.org/abs/2608.21560), under an audited Kepler
adaptation. Independent reviewers replayed scenario normalization, actual
per-family draw denominators, model identity, numerical checks, weights, units,
and input/source hashes. These are conventional **conditional statistical
validations**, with the declared astrophysical models and assumptions preserved.

| BOI | Host | Period, days | Intended-host planet | FPP | Resolved NFPP | Sampled EB |
|---|---|---:|---:|---:|---:|---:|
| 2077554781499350144-1 | KIC 6699887 / TIC 270864056 | 15.113695 | 99.5378% | 0.462225% | 0.0003993% | 0.0031544% |
| 2127186564572468352-1 | KIC 8745833 / TIC 158835390 | 4.566494 | 99.1156% | 0.884438% | <1e-45% | 0.0003753% |
| 2127169041106093184-1 | KIC 8878385 / TIC 158934005 | 5.905425 | 99.8283% | 0.171734% | 1.28e-27% | 1.10e-11% |

Intended-host planet probability is TP+PTP+DTP, or 1−FPP, within the adopted
scenario model. FPP includes planets on the wrong star and binary alternatives;
it is **not** EB probability or global nonplanet probability. Resolved NFPP
covers modeled resolved-neighbor alternatives. Unresolved off-target alternatives
remain in FPP. The extremely small second NFPP is a conditional finite-model
result, not proof that all possible blends have been excluded. Global
planet/nonplanet and instrumental false-alarm probabilities remain unestimated.
Unknown is not 50%.

Nominal models use each target's own SAP apertures, measured quarter responses,
composite observed Gaia flux, finite population/stellar priors, the declared
exponential PRF-wing model, and proper normalized affine-baseline/noise treatment.
Original and refined timing pass under both Gaia and TIC stellar sensitivities
for the first two signals. The third passes both own-TIC ephemerides using all
53,450 native photons and proper event-baseline/shared-noise marginalization.
Its earlier compressed original-TIC model failed with FPP 13.42% and NFPP 13.25%.
That failure remains preserved. The native likelihood is an explicit operator
and noise-model amendment, not a binning-only comparison; no Gaia-native scenario
census is claimed for that signal. Its Gaia/TIC native transit fits remain
correlated physical sensitivities.
Correlated PSF/SAP measurements are not independent planet probabilities;
no multiplicity boost is applied.

| BOI | Gaia original / refined FPP | TIC original / refined FPP | Complete scenario families |
|---|---:|---:|---:|
| 2077554781499350144-1 | 0.3043% / 0.3450% | 0.4343% / 0.4622% | 210 |
| 2127186564572468352-1 | 0.0910% / 0.2078% | 0.6462% / 0.8844% | 114 |
| 2127169041106093184-1 | No adopted native census | 0.1324% / 0.1717% | 99 |

The 15-day nominal FPP has an observed IID integration diagnostic of 0.0381
percentage points. The 4.57-day nominal diagnostic is 0.0711 percentage points;
its original-timing STP family remains sparse, with weight ESS 7.59 and FPP
diagnostic 0.2357 percentage points. All retained repeats pass the unchanged
threshold. Repeat ranges and ESS assess observed integration behavior; they
cannot certify unvisited modes or systematic model error. Shared-noise,
primary-flux-anchor, and wider-wing sensitivities for the second signal also pass.
The third nominal FPP has an IID integration diagnostic of 0.0569 percentage
points, with retained repeats spanning 0.1366–0.2169%. Its original-timing repeats
span 0.0864–0.2013%. The refined STP family has weight ESS 9.80, and several tiny
alternative families remain sparse. These numerical limitations are explicit;
the unchanged validation threshold still has a substantial observed margin.

Admission records bind the numerical evidence and limitations:

- [15-day admission](../dump/boi-validation-20261011/rv/kepler-route/207755-final-admission.v1.json)
  and [independent replay](../dump/boi-validation-20261011/rv/kepler-route/207755-independent-admission-numerics.v1.json).
- [4.57-day admission](../dump/boi-validation-20261011/rv/kepler-route/212718-final-admission.v1.json)
  and [independent replay](../dump/boi-validation-20261011/rv/kepler-route/212718-independent-admission-numerics.v3.json).
- [5.91-day admission](../dump/boi-validation-20261011/rv/kepler-route/212716-final-admission.v1.json)
  and [independent native replay](../dump/boi-validation-20261011/rv/kepler-route/212716-independent-native-numerics.v1.json).

![Observed own-SAP transit profiles](../dump/boi-validation-20261011/figures/admitted-transit-profiles.v1.png)

These two frozen observed profiles use the first two adopted FPP inputs. Error bars show
the covariance diagonal; inference retains the full covariance. The
[figure receipt](../dump/boi-validation-20261011/figures/admitted-transit-profiles.v1.receipt.json)
binds the data, units, admissions, and exported figures; the
[independent check](../dump/boi-validation-20261011/rv/kepler-route/admitted-transit-profiles-independent-review.v1.json)
verifies those bindings.

## Planet and host characterization

These weighted 16th/50th/84th percentile values come from the **same model as each
adopted FPP**, including actual quarter dilution. Unit-dilution native-fit radii
remain separate and are not substituted for the planet radii.

| Property | BOI-2077554781499350144-1 | BOI-2127186564572468352-1 | BOI-2127169041106093184-1 |
|---|---:|---:|---:|
| Planet radius, Earth radii | 2.659 [2.503, 2.861] | 4.967 [4.730, 5.243] | 3.377 [3.202, 3.569] |
| Radius description | Sub-Neptune-sized | Neptune-sized | Sub-Neptune-sized |
| Semimajor axis, AU | 0.10787 [0.10392, 0.11111] | 0.05595 [0.05411, 0.05780] | 0.06391 [0.06200, 0.06602] |
| Inclination, degrees | 89.46 [89.10, 89.83] | 88.69 [87.81, 89.66] | 88.76 [87.78, 89.62] |
| Prior-conditioned eccentricity | 0.085 [0.020, 0.240] | 0.074 [0.019, 0.213] | 0.070 [0.015, 0.214] |
| Host mass, Solar masses | 0.733 [0.655, 0.801] | 1.121 [1.014, 1.235] | 0.999 [0.912, 1.101] |
| Host radius, Solar radii | 0.764 [0.728, 0.803] | 1.213 [1.151, 1.273] | 1.270 [1.220, 1.329] |
| Host effective temperature, K | 4638 [4503, 4754] | 6121 [6008, 6251] | 5654 [5537, 5746] |
| Temperature-based stellar description | K-type | Late-F/early-G-type | G-type |
| Host metallicity [Fe/H], dex | +0.164 [+0.129, +0.208] | −0.288 [−0.375, −0.268] | −0.203 [−0.216, −0.188] |
| Host log g, cgs | 4.535 [4.475, 4.589] | 4.322 [4.265, 4.379] | 4.233 [4.180, 4.274] |
| Distance, pc | 859 [836, 883] | 1099 [1083, 1115] | 1871 [1795, 1945] |
| Model extinction Av, mag | 0.152 [0.120, 0.183] | 0.113 [0.103, 0.124] | 0.152 [0.142, 0.160] |
| Derived luminosity, Solar luminosities | 0.244 [0.208, 0.281] | 1.836 [1.644, 2.087] | 1.483 [1.311, 1.652] |
| Irradiation at semimajor axis, Earth units | 21.0 [17.9, 24.5] | 596 [515, 674] | 363 [322, 410] |
| Equilibrium temperature, K | 596 [572, 619] | 1375 [1326, 1418] | 1215 [1179, 1253] |

Luminosity uses modeled radius and effective temperature. Irradiation is evaluated
at the semimajor axis, not averaged over an eccentric orbit. Equilibrium
temperature assumes zero Bond albedo and uniform full-sphere heat redistribution;
it is not a measured atmospheric or surface temperature. Temperature-based
stellar descriptions are not measured spectral types.

The eccentricity distributions remain strongly prior-conditioned. These data
do not provide RV masses, bulk densities, compositions, atmospheric properties,
or reliable ages. Size alone cannot establish a rocky or gaseous composition.
Characterizations retain separate TP/PTP/DTP weights, posterior ESS, all finite
particle arrays, and quarter host-flux fractions:

- [15-day characterization](../dump/boi-validation-20261011/statistical/field-2077554781499350144/planet-characterization-v30-tic-own-sap-final65536.json), joint weight ESS 2702.7.
- [4.57-day characterization](../dump/boi-validation-20261011/statistical/field-2127186564572468352/planet-characterization-v25-tic-composite65536.json), joint weight ESS 278.4.
- [5.91-day native characterization](../dump/boi-validation-20261011/server-transfer/linux-v42_1/campaign/statistical/field-2127169041106093184/native-characterization-v42-tic-final65536.json), joint weight ESS 146.7.
- [Derived irradiation and temperature](../dump/boi-validation-20261011/admitted-planets-derived-properties.v2.json).

## Established identities and false-positive mechanisms

**BOI-2107574403864920576-1 is Kepler-565 b**, at the exact host and published
ephemeris. [Morton et al. (2016)](https://arxiv.org/abs/1605.02825) statistically
validated it; the [NASA archive](https://exoplanetarchive.ipac.caltech.edu/overview/Kepler-565)
retains it. The published model gives period 4.243747226 days, radius
3.64 (+0.51/−0.27) Earth radii, and a 5735 (+81/−75) K host with mass
0.98 (+0.043/−0.055) and radius 0.95 (+0.132/−0.070) Solar units. The derived
semimajor axis is about 0.051 AU. Mass and composition remain unmeasured here.
Current Gaia RUWE 1.989 fails the requested 2026 host gate, so this is a recovery
of a published validation. Its printed FPP 0e+00 is rounded table precision.

Entries **-2 and -3** fold that same transit train at P/5 and P/3. Masking the
known planet removes supplied depths in all three delivered views. PSF_FLAT
residuals are −4.6 ±6.5 and +2.8 ±5.0 ppm using quarter-scatter errors. They do
not establish additional planets. See the
[alias measurements](../dump/boi-validation-20261011/known-signal-alias.json)
and [published characterization](../dump/boi-validation-20261011/known-planet-characterization.json).

**BOI-2076055902327710720-1 and BOI-2076062434975364864-1 are denied for their
dominant supplied waveform.** Contemporaneous Kepler cadences reproduce KIC
8196180's primary eclipse, eccentric secondary, and modulation, including held
alternate cycles in Q6 and Q10. Target and source share CCD channels. The exact
optical/electronic transfer route remains unknown; residuals remain unclassified.
These denials do not establish that either star hosts no planets. See the
[reproduced mechanisms](../dump/boi-validation-20261011/evidence/recommended-transfer-dispositions.json).

**BOI-2100442657917279616-1 remains unresolved with contamination:** subtraction
of the reproduced Kepler-1647 binary contribution in Q5/Q9 leaves about 384–464
ppm. **BOI-2125801656663271552-1 also remains unresolved with contamination:**
an inverted RR Lyrae waveform is reproduced in Q1, while most event support lies
in other quarters. Neither partial reproduction denies the whole supplied signal.

## Unresolved signals and archival limits

Each unresolved entry has a finite record and reason in the
[126-entry feasibility snapshot](../dump/boi-validation-20261011/evidence/finite-feasibility-20261012-v2/finite-feasibility.json).
All-host RV, TESS, ZTF, catalogue, stellar, and harmonic checks retain missing
evidence explicitly. A large apparent radius, catalogue discrepancy, failed host
gate, query failure, or weak recurrence is not a false-positive denial.
No unsupported confidence percentage is assigned.

**BOI-2130460704041437824-1 remains unresolved with admitted conditional
probabilities.** Its coherent own-TIC, refined-timing, complete 66-family model
gives 69.04% intended-host planet, 30.95% modeled wrong-host planet, and
0.002284% sampled EB. Modeled any-host planet probability is 99.9977%; this
does not establish global planet confidence or identify the host. Resolved NFPP
is 29.46%, far above the 0.1% gate. The dominant alternative is a planet on Gaia
DR3 2130460669681699072. No reproduced EB mechanism supports a denial.

The nominal FPP is 30.957% with an IID integration diagnostic of 2.282 percentage
points. High-count repeats give 28.67–32.55%; original-TIC timing gives 50.48%
with diagnostic 2.403 percentage points. These assess observed numerical and
timing sensitivity, not formal confidence coverage. Tiny EB components remain
sparse, and systematic model error and unvisited modes remain unbounded. The
[held probability admission](../dump/boi-validation-20261011/rv/kepler-route/213046-held-probability-admission.v1.json)
retains the nominal model, all sensitivities, and independent reconstruction.

**BOI-2073226966298407552-1 remains unresolved with qualified conditional
probabilities.** Its complete 306-family own-TIC model gives **97.5517%**
intended-host category probability and **2.4483% FPP**, above the 1.5% gate.
The observed IID integration diagnostic is 0.3336 percentage points; selected
repeats give 2.0206% and 3.0469%, both above threshold. Dominant intended-host
families have weight ESS 295–329 and the principal wrong-host family ESS 55.9.
Original timing remains too sparse for probability admission. Tiny sampled EB
and NFPP values cannot establish precise exclusion bounds, and the finite model
does not resolve the material 13–80 Jupiter-mass alternatives. Global
planet/nonplanet confidence remains unknown. The
[held admission](../dump/boi-validation-20261011/rv/kepler-route/207322-held-probability-admission.v1.json)
retains those limits. An [audited retention check](../dump/boi-validation-20261011/statistical/eighth-characterization-retention.v69.json)
found running evidence moments and maximum-weight physical draws, but no full
per-draw weights or physical particles. Dilution-aware weighted characterization
requires a new likelihood replay; maximum-weight draws cannot substitute for it.

The literature crosscheck associates three hosts with published candidates in
[Bienias and Szabó (2025)](https://arxiv.org/html/2501.09152v1):
2100325529865507712, 2126910621514995968, and 2128959007378441472.
Three supplied primary clocks match; two extra clocks on the last host share
some primary events and remain unresolved. These are candidate associations,
not prior validations. The paper's parent-aperture KIC identifiers are distinct
from target-star identifiers. Five catalogue identity flags were corrected from
explicit Gaia DR3 fields; original records and timing calculations remain
preserved in the [independent review](../dump/boi-validation-20261011/evidence/catalogue-identity-and-bienias-cohort-independent-review.v1.json).

The [closed phase check](../dump/boi-validation-20261011/rv/kepler-route/210032-phase-feasibility-independent-review.v1.json)
for published candidate 2100325529865507712-1 does not establish a mass bound.
The SAP second harmonic is −66.6 ±69.7 ppm using the empirical quarter-mean
error; omitting Q11 changes the mean to −4.9 ppm. PSF_FLAT gives +12.9 ±13.2
ppm, conditioned on the archived flattening. All 306 signed models and 612
injection controls reproduce. Source dilution, activity, reflection and
substellar-mass alternatives remain unresolved.

**BOI-2078045782158649216-1 remains unresolved after completed native fits and
bounded scenario comparisons.** Its four Gaia/TIC PSF/SAP fits pass the declared
convergence and numerical gates. Its [authenticated inputs](../dump/boi-validation-20261011/evidence/native-2078045782158649216/native-ready.receipt.v1.json)
retain 44,436 valid rows in each original view, 14 quarter apertures and their
actual exposure headers. Different Gaia/TIC temperatures remain separate
sensitivities; these unit-dilution fits do not establish a planet radius or host.

Both complete 198-family, 4,096-draw TIC timing pilots retain competitive binary
and blend alternatives with sparse weights. Their FPP values remain unadmitted
diagnostics. Ten fixed native-mode comparisons and a separate Gaia extinction
control pass numerical review but do not supply marginal scenario probabilities
or reproduce a denial mechanism. The [reviewed endpoint](../dump/boi-validation-20261011/rv/kepler-route/207804-bounded-followup-independent-endpoint.v1.json)
preserves this hold. Further inference would require a declared complete native
comparison with common baseline/noise priors and measured noise adequacy;
larger identical primary-binned runs cannot close those gaps.

The server continuation tests one common live-exposure OU noise model on three
retained planet, blended-planet and binary witnesses. Independent dense controls
verify the likelihood and conditional simulator. A declared optimizer amendment
closes the initial guide failure; all 24,576 conditional weights are retained,
with per-seed effective counts about 2,400 and formal integration errors about
1.3%. All three witnesses fail the predeclared global predictive adequacy test
(rank 1/257), with unexplained residual correlations and quarter variability.
The [reviewed noise endpoint](../dump/boi-validation-20261011/evidence/server-continuation-20261012/method-review/ou-hyper-pilot-remote-metadata-v2/reviewed-endpoint.v1.json)
therefore holds further inference. These are conditional noise calculations;
the complete 198-family noise/physical integral, FPP and model probabilities
remain uncomputed. A deficient noise model does not deny the planet.

**BOI-2052194442729351680-1 remains unresolved after bounded original-photon
and source-response checks.** Its 2.42115379-day signal recurs in all seven
retained PSF_FLAT quarters: 224 primary windows give 2,515.9 ± 108.5 ppm from
event scatter, while the half-phase depth (−27.6 ± 108.1 ppm) and even-minus-odd
difference (−3.1 ± 217.5 ppm) show no conflict. SAP and PSF_QDIAG retain strong
full-phase residual correlation, so they cannot support the simple
diagonal-noise model or veto the passing PSF view. Target attribution is
compatible but unclosed: three of four selected maps prefer the target, the Q4
alternative remains viable under some response envelopes, and its negative
historical SAP aperture response cannot establish a physical dilution solution.
The bounded 80-source response diagnostic excludes no source from the archived
passing signal because the original PSF_FLAT extraction and flattening response
is unavailable.

The initial paired experiment stopped with zero fits when the Gaia density-prior
quadrature failed its unchanged 1e-5 tolerance. The server
[repair](../dump/boi-validation-20261011/evidence/server-continuation-20261012/prioritization/remote-metadata-v2/prior-repair-closure.v2.json)
preserves the original stellar probability law, bounds and grid; partitioned
integration agrees with independent scalar and doubled-order references.
The subsequent 16 native cases retain all original photons, proper event
baselines, the normalized full white-noise prior and authenticated live exposure.
Thirteen fits remain unconverged. Three pass the declared local stationarity
checks, but lie closer to a parameter boundary than the optimizer's resolution.
All 16 remain held; no posterior, FPP or body mass is admitted. The
[independent endpoint](../dump/boi-validation-20261011/evidence/server-continuation-20261012/runtime-readiness/205219-completion-endpoint-independent-review.v2.json)
preserves the first failures and numerical controls. Further inference needs
adequate physical and noise posteriors, the original PSFMACHINE deblend and
flattening response (or exact operator replay with signed pixel injections),
and Kepler-band target and neighbor flux constraints.

**BOI-2106817012155835904-1 remains unresolved after a reviewed common-photon
screen.** Its 4.51679789-day train recurs in 290 original-valid PSF_FLAT windows,
with mean signed depth 3405 ±112 ppm using event scatter. Quarter means vary;
their mean is 3499 ±386 ppm using quarter scatter. On the 56,431 cadences shared
by all three reductions, the primary remains in 276 usable windows. The two
short-only classes of the exact Rust 1.50559383-day proposal give −18 ±51 and
+10 ±52 ppm in PSF_FLAT. The overlapping class retains the main decrement.
These are fixed-clock diagnostics; the correlated shallow proposal is retained
but was not fitted in this screen. No additional-body or alias disposition is
admitted.

Independent reconstruction passed all 11,826 measured local windows and 104
full-quarter models. Full SAP and PSF retain strong residual correlation; their
diagonal-noise measurements do not admit confidence percentages. All 15 retained
pixel maps and 81 source models reproduce, preferring the target in every map,
but target chi-square per degree of freedom spans 1.22–14.36. Relative preference
does not establish NFPP or physical dilution. The
[reviewed endpoint](../dump/boi-validation-20261011/evidence/native-2106817012155835904/reviewed-endpoint.v1.json)
retains the failed retrieval/metadata checks, repaired authentication, complete
fits and independent reviews. Exact PSF extraction response, positive Kepler-band
source radiometry, and adequate local/native noise models remain required before
scenario probabilities; all current admissions and probability fields are unchanged.

![Observed fundamental and short-only event profiles](../dump/boi-validation-20261011/evidence/native-2106817012155835904/figures-v1/common-photon-profiles.v2.png)

The shared-cadence PSF_FLAT profiles retain unfitted shoulders. Error bars use
scatter among event means; the
[figure receipt](../dump/boi-validation-20261011/evidence/native-2106817012155835904/figures-v1/figure-release.v2.json)
binds all plotted photons and bin memberships.

The principal detailed holds are:

- **2130460704041437824-1:** all four native fits pass. Complete 4096-draw pilots
  and selected-family refinements retain the nearby-star planet alternative
  described above. Gaia-radius fits explicitly use a TIC mass because Gaia
  mass is missing. The original unsupported broad-metallicity prior and the
  separate measured-neighbor-metallicity model used for the admitted probabilities
  remain preserved; the hybrid original-timing sensitivity is still coarse.
- **2076248557386606336-1:** all four native fits pass. Eight complete 351-family
  full-native pilots are closed with proper baseline/noise treatment. The TIC
  refined FPP diagnostic is 1.0568% with 0.7597 percentage-point integration
  error; dominant weights remain too sparse for confidence admission. The
  [independent review](../dump/boi-validation-20261011/rv/kepler-route/207624-native-v65-independent-numerical-precision-review.v1.json)
  retains the hold. A bounded proposal refinement did not resolve the sparsity.
  The circular unit-dilution radius of 4.3–4.5 Earth radii is not a dilution-aware
  planet posterior; sampled intended-host modes can require much larger bodies.
- **2077554781499350144-3:** the 47.69-day signal has passing native fits, but
  its compressed original Gaia scenario pilot fails FPP/NFPP. It cannot inherit admission
  from the validated 15-day signal. Own compressed-TIC pilots also fail at least
  one threshold. The separately declared complete native-time pilots are closed;
  their [independent review](../dump/boi-validation-20261011/rv/kepler-route/207755-3-native-v46-pilot-independent-review.v1.json)
  retains a hold because original-timing binary weights are sparse. Native
  selected-family refinements are closed with all 210 scenario families retained.
  A server experiment adds two independent proposal repeats and recombines
  every one of the 210 families with its actual draw denominators. Pooling the
  two fresh repeats gives TIC diagnostic FPP 3.310% at original timing and 0.917% at
  refined timing, with formal Monte Carlo errors of 1.387 and 0.203 percentage
  points. The [reviewed endpoint](../dump/boi-validation-20261011/evidence/server-continuation-20261012/prioritization/207755-remote-metadata-v1/reviewed-endpoint.v1.json)
  retains the earlier 3.499%/0.744% diagnostics and Gaia sensitivities. Original
  timing fails the 1.5% threshold; the 206 shared older alternatives, sparse
  important weights and unclosed noise adequacy also prevent admission. These
  conditional diagnostics are not adopted planet confidence.
- **2076383384993192576-1:** the 51.5278568-day train recurs in 21 native events
  over 14 quarters. Seasonal depths vary strongly. Nine formal pixel comparisons
  prefer the target, but poor absolute fits and signed bright-neighbor residuals
  leave source ownership unresolved. The
  [native replay](../dump/boi-validation-20261011/rv/kepler-route/207638-minimal-native-independent-review.v1.json)
  establishes neither an EB mechanism nor a mass bound.
- **2128059130130482432-1:** the complete 58,153-photon SAP comparison favors the
  declared variability model over an added narrow transit template (delta lnZ
  −2.469). All 42 bounded controls, including injected transits, reproduce.
  The [independent review](../dump/boi-validation-20261011/rv/kepler-route/212805-complete-diagnostic-independent-review.v1.json)
  preserves the passing PSF view and unresolved source alternatives; this
  correlated SAP result does not veto the BOI or assign a planet probability.
- **2105418399005125376-1:** native fits pass with apparent radius about 9 Earth
  radii. A retained brown-dwarf witness occupies a material mass range absent
  from the current stellar-binary family; available phase data do not exclude it.
  The signal remains held without a fabricated mass bound or EB denial.
- **2076468876327949568-1:** a primary and eccentric secondary repeat in PSF/SAP.
  Conflicting density/luminosity constraints and unresolved source alternatives
  prevent a definitive mechanism classification.
- **2106383701495087232-1:** the free-density native experiment fails convergence
  and radius-significance gates; separately constrained stellar/blend models
  remain possible. A pointing/PRF envelope admits the faint neighbor, so nominal
  target preference is not NFPP.

Targeted own-SAP scenario pilots are closed for sources 2077209775366626944
and 2073226966298407552; their Gaia/TIC native target fits pass. The seventh
source's refined Gaia FPP diagnostic is 33.06% with sparse leading alternatives.
Its two unsupported TIC-neighbor metallicities remain explicit. Additional
clocks on the eighth host concentrate their depth in primary-coincident event
classes. An [independent replay](../dump/boi-validation-20261011/statistical/eighth-eventclasses-independent-review.v68_1.json)
reproduces all 8,172 event records, including missing-support records. One other PSF class retains a
504.6 ±178.7 ppm decrement; primary-masked coincident classes lack usable
in-transit support. These are fixed-clock diagnostics, without additional-planet
admission, alias classification, denial, or a multiplicity boost.
For 2130658616133595264, the TIC native fit passes and the Gaia endpoint remains
held. Its official Kepler population request completed, but the distinct Gaia
population needed by the current full scene model remains unavailable after
bounded requests. For 2099560128037101952, the TIC native endpoint remains held
and the explicitly mixed TIC-mass/Gaia-radius fit passes; both population requests
remain unavailable. These are finite retrieval and model-support holds, not
astrophysical denials or evidence that the original photons are absent.

For 2130460704041437824, six retained physical witnesses now pass an independent
comparison using all 35,256 native photons and authenticated live exposures.
Within original timing, the selected binary-minus-target-planet log likelihood
changes from −32.73 under the adopted bin operator to +15.86 under the native
operator. The operators use different photon sets and baseline/noise treatments;
this is joint sensitivity, not a compression-only effect or cross-data Bayes
factor. The [reviewed comparison](../dump/boi-validation-20261011/evidence/server-continuation-20261012/prioritization/213046-remote-metadata-v1/reviewed-endpoint.v1.json)
does not integrate the complete 66-family scene or exclude other modes of the
nearby-star alternative. No model probability, planet admission or EB denial
follows from these selected likelihood scores.

The [RV ledger](../dump/boi-validation-20261011/rv/rv_availability.json) covers all
95 original hosts. Queried ESO, HIRES, KPF, SOPHIE, DACE, APOGEE, LAMOST, Gaia,
and ExoFOP records yield no authenticated coherent exact-host RV series suitable
for the [requested RV framework](https://arxiv.org/abs/2606.20224). Single
low-resolution velocities and catalogue means cannot establish an orbit or mass.
This is a query result, not proof that no observations exist.

All-host TESS/ZTF coverage and targeted fixed-ephemeris photometry remain archived.
The two strongest hosts' latest native TESS measurements do not securely recover
or exclude their transits after measured event noise, crowding, and propagated
timing are considered. These data are not independent FPP factors. Non-detection
does not deny a planet without adequate sensitivity.

## Reproduction and interpretation

The [method audit](../dump/boi-validation-20261011/statistical-method.md), admission
records, source manifests, and complete ledgers define each adopted model.
Original numerical calculations and rejected experiments remain retained. Proper
normalized baseline/noise priors and defensive p/q importance sampling use all
samples, including nontransiting and zero-likelihood draws. No background scenario
is silently dropped, no unsupported stellar value receives a Solar default,
and no multiplicity boost is used.

The literal 20×1-million-draw benchmark was stopped administratively after
retaining complete/partial outputs and matched recipe comparisons. It is
not reported as completed. Admissions use separately audited importance
integration, complete scenario denominators, platform controls, independent
repeats, and unchanged thresholds. The stopped benchmark remains in its
[transfer receipt](../dump/boi-validation-20261011/server-transfer/linux-v2/stopped-benchmarks/transfer.receipt.json).

The accounting verifier checks exact 126/125 membership, frozen signal numbers,
ledger/source hashes, CSV/JSON agreement, aliases, and independent admission
receipts. Integrity checks support reproducibility; they do not independently
validate a planet. Catalogue absence remains only NO_MATCH_IN_QUERIED_SOURCES,
without an unconditional novelty claim. Additional observations may refine or
overturn these model-dependent conclusions.

## Before publishing statistical validations

- [ ] For each new statistical validation proposed for publication, complete
  exactly 20 independent TIC-based runs at the discovery ephemeris, with
  1,000,000 attempted draws per scenario family per run. Freeze the inputs,
  priors, source, configuration, and proposal recipe before the campaign.
  Preserve complete prior support, exact p/q weights, and every scenario
  denominator, including zero-likelihood draws. Do not promote Gaia stellar
  alternatives, another ephemeris, or a modified likelihood into another
  repeat census.
- [ ] Report the mean, standard deviation, minimum, and maximum FPP and NFPP
  across the 20 runs. Retain per-family effective sample sizes, integration
  errors, dominant weights, seeds, and complete outputs. Keep earlier adaptive
  pilots separate from these independent-run summaries.
- [ ] Review numerical precision and model adequacy before publication.
  Preserve all threshold crossings and failed checks; investigate sparse
  influential alternatives rather than selecting favorable runs. Require the
  published FPP <1.5%, resolved NFPP <0.1%, and other validation gates. Twenty
  completed runs alone do not close those gates.

These publication tasks are pending. The existing admissions and historical
calculations remain preserved; the modified sampling protocol must be described
explicitly in the publication.
