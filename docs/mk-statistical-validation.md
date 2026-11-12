# M/K statistical-validation campaign

Adam authorized server-only follow-up on October 20, 2026, aiming for 50 new
statistical admissions. The goal does not change the admission requirements or
promise that the candidate cohort contains 50 planets. Preserve every original
BOI label, proposal, unfavorable repeat and failure.

## V3 campaign, October 25, 2026

Adam chose `TRICERATOPS_KEPLER_V3` for M/K hosts on October 25, 2026. In paired
two-seed runs on three K dwarfs, stock TRICERATOPS failed NFPP (0.14–1.9%) in
every seed while V3 gave NFPP near zero, with similar FPP
([paired record](../results/boi-validation/mk-paired-stock2-20261025.json)).
Signals were ranked by declared vetting criteria: search SNR, odd/even below 3σ,
host RUWE at most 1.4 and parallax SNR at least 5, no false-positive catalogue
match, no survey-BOI alias, one signal per alias family.

| Host | Period, days | Radius, R⊕ | Mean FPP (20 repeats) | Outcome |
|---|---:|---:|---:|---|
| KIC 6267058 | 11.79 | 3.32 | 2.03% | Validated by multiplicity |
| KIC 6267058 | 16.74 | 2.82 | 2.32% | Validated by multiplicity |
| KIC 6267058 | 29.31 | 4.25 | 2.57% | Validated by multiplicity |
| KIC 9773466 | 10.81 | 2.16 | 0.71% | Numerical pass; fails the S/N ≥ 15 closeout (S/N 9.9) |
| KIC 3831969 | 15.31 | 3.66 | 1.81% | Possible Planet; matched-filter S/N 13.6, below 15 |
| KIC 7362934 | 14.61 | 2.62 | 2.03% | Possible Planet; matched-filter S/N 12.5, below 15 |
| KIC 10844387 | 12.03 | 2.45 | 2.21% | Possible Planet; matched-filter S/N 14.84, below 15 |
| KIC 3628504 | 7.54 | 3.03 | 2.24% | Possible Planet; unresolved ephemeris match |
| KIC 6146676 (M dwarf) | 40.29 | 2.62 | 2.71% | Possible Planet |
| KIC 10027832 | 2.08 | 2.62 | 2.68% | Possible Planet |
| KIC 10527790 | 46.55 | 2.76 | 3.01% | Possible Planet; phase-0.5 dip a planet cannot make |
| KIC 4740764 | 5.35 | 2.50 | 3.58% | Possible Planet |
| KIC 9474318 | 3.29 | 2.28 | 3.92% | Possible Planet; matched-filter S/N 14.47, below 15 |
| KIC 5458397 (M dwarf) | 1.41 | 3.29 | 4.27% | Possible Planet; period below 1.6 days |
| KIC 5689121 | 13.77 | 2.54 | 4.88% | Possible Planet; no sibling for multiplicity |
| KIC 5272880 | 6.11 | 2.16 | 5.50% | Possible Planet; matched-filter S/N 14.95, below 15 |
| KIC 7813322 | 4.08 | 2.01 | 6.03% | Possible Planet; no sibling for multiplicity |
| KIC 9827536 | 10.58 | 2.31 | 7.48% | Possible Planet; matched-filter S/N 13, below 15 |
| KIC 7120683 | 13.04 | 2.02 | 7.80% | Possible Planet; matched-filter S/N 13.1, below 15 |

Radii are the median TRICERATOPS planet-scenario best fits. KIC 9773466 is the only
single M/K host with a numerical V3 pass: mean FPP 0.71% (0.53–0.91%) and NFPP
0.005%. Its RUWE is 0.97, no Gaia source lies within 1.5 arcsec, and it has no
phase-0.5 secondary. But its matched-filter S/N of 9.9 fails the S/N ≥ 15 item of
the admission closeout (Giacalone et al. 2021), so it is not validated. The single K and M
dwarfs so far stop at FPP 1.8–7.8%. Without high-resolution imaging, most of it is the
scenario of a planet on an unresolved bound companion; the chance of a non-planet
is 0.01–1.3%. KIC 3831969, KIC 7362934, KIC 10844387, KIC 5689121, KIC 5272880,
KIC 7813322 and KIC 9827536 fail multiplicity only for want of a sibling: their
boosted FPPs would be 0.33%, 0.38%, 0.41%, 0.92%, 1.05%, 1.15% and 1.45%. KIC 10527790 has a dip at phase 0.5 of 830 ± 159 ppm (S/N
5.2, about 4.5 after scaling by the fit's χ²), 43% of its primary depth. As a planet
occultation that would need geometric albedo near 3,500, but a circular eclipsing
binary would give it. TRICERATOPS does not use secondaries, so this Possible Planet
needs a calibrated secondary test before any further claim.

KIC 6267058 has three independent candidates. All three meet the Lissauer et al.
(2014) and Rowe et al. (2014) conditions, with NFPP at most 1e-8 and a clean
closeout. With the smallest published multiplicity boost, 5.5, their FPP is
0.37%, 0.43% and 0.48%. They form a separate tier: planets of the KIC 6267058
system, host star (target or bound companion) not determined; see the
[multiplicity record](../results/boi-validation/mk-multiplicity-validation-20261025.json).

A sibling counts as independent only if it is not an alias of a stronger signal.
An alias has a commensurate period and S/N at most twice `S/N_strong/sqrt(m n)`,
or shares two or more events with that signal. Window overlap alone cannot reject
a 1/k sub-harmonic, which leaves 1 − 1/k of its windows clear. The system must also
be Hill-stable: adjacent candidates more than 2√3 mutual Hill radii apart
(Gladman 1993), with masses `M = R^2.06` (Lissauer et al. 2011). As in Rowe et
al. (2014, §5.9), the validated signal needs a fitted-model S/N above 10 (15 below
4 days), a period above 1.6 days and a transit that is not V-shaped
(b + σ_b + R_p/R⋆ ≤ 1); the four multiplicity validations pass (0.57–0.79). Every other
completed survey or M/K campaign with FPP below 50%, except KIC 6699887's
47.69-day signal ([BOI page](boi-validation.md)), fails: it has no other signal, its other signals are aliases, or its radius or boosted FPP is too
large ([sweep](../results/boi-validation/mk-multiplicity-sweep-20261025.json)).
KIC 3628504's signal matches a DR25 TCE on KIC 8458662 at 2:1, 12° away;
the expected number of such chance matches is 0.14, and pixel analysis is needed.

### Dispositions of all 663 M/K BOIs

The [disposition record](../results/boi-validation/mk-dispositions-20261025.json)
and its [table](../results/boi-validation/mk-dispositions-20261025.csv) give every
M/K BOI one category on October 25, 2026. Only the instrumental-date class asserts
a mechanism:

- **False positive, instrumental date (20).** One event on BKJD 200, 246 or 373
  carries at least half of the signal's S/N², and the other events fall below
  S/N 7.1 in local KBonus SAP. These dates show an event excess across all
  17,940 M/K proposals (z = 79, 72 and 13 in 0.5-day bins), so the detections
  are artifacts of those dates.
- **Period unsupported, single event (31).** One event carries the signal; the
  event may be real, but the period is not.
- **TRICERATOPS FP category (15).** NFPP 0.29–1.00 in both first repeats: the
  signal is more likely on a neighbour, chiefly as an eclipsing binary. For the
  7.09-day signal on Gaia DR3 2106616626158535296, KIC 10149205's 31.50-day signal,
  three signals (31.45, 39.31 and 53.46 days) on KIC 6197519 (Gaia DR3
  2101482490978833920), all led by its neighbour Gaia DR3 2101482387899618048, the
  32.58-day signal on Gaia DR3
  2077526743952548224, KIC 9605652's 32.00-day signal 10 and the 8.21-day signal on
  Gaia DR3 2127033079618738944, the leading scenario is a planet transiting a
  neighbour. This is
  statistical, not a reproduced mechanism. KIC 10149205's occultation scan peaks
  at phase 0.481 with 764 ± 138 ppm, 45% of the primary depth (S/N 5.5, 4.5 after
  scaling by the fit's χ², strongest of 112 trial phases), as an eccentric
  eclipsing binary would; a calibrated test is needed before it counts. KIC 9274180 (4.43 days) also shows a phase-0.5 dip of 261 ± 75 ppm
  (SNR 3.5, 2.7 after scaling by its event scatter). As a planet occultation it
  would need geometric albedo 42. KIC 9958491's 11.24-day signal, about 0.9 R⊕ on
  its K dwarf, failed only protection in its first fit. The second generation gives
  the same 3.0-hour transit, then FPP 85% and NFPP 83% in both first repeats. It is
  led by an eclipsing binary on Gaia DR3 2128518137575602688, a K dwarf 15.5 arcsec
  away and 0.5 mag fainter. Its matched-filter S/N is 4.1.
- **Stopped early (8).** KIC 10094106's 0.88-day signal fits a grazing body of
  0.49–0.94 stellar radii (b 1.41–1.88), which looks like an eclipsing binary;
  the first two repeats give FPP 0.58–0.75. KIC 8964194's 8.98-day signal first fit
  a grazing, unconstrained body. With its transit protected, the second generation
  fits radius ratio 0.0059–0.0096, about Earth-sized (1.1 R⊕). It then stops at FPP
  10% and NFPP 6%.
- **Transits too long for their hosts (10).** Ten native fits want transits 1.5
  to five times longer than a circular orbit allows on their hosts. They get there
  through high eccentricity or a density far below the host's. All but KIC
  8308249's get a duration-aware second generation. For the 1.19-day signal on Gaia DR3
  2101518671782975488, and for the 5.74-day signal 2 on Gaia DR3 2086861666381019648
  (8.8 hours at eccentricity 0.98, then 0.9 hours), the radius ratio then becomes
  unconstrained. Two fits
  lengthen again as protection widens. For the 13.38-day signal on Gaia DR3
  2126063585245659136 the transit goes from 4.8 to 11.6 to 30 hours. For the
  36.71-day signal on Gaia DR3 2078180571109230848 it goes from 6.45 to 20.2 to 93
  hours, at red-giant densities of 0.4–1.3 kg/m³ on an M dwarf. These four stay
  held without a further generation, because their durations do not converge.
  Signals 6 (31.45 days) and 11 (53.46 days) on the K dwarf Gaia DR3
  2101482490978833920 both need eccentricities near 0.5 at its FLAME density. With
  protection for their fitted transits, the durations shorten to 7.5 and 6.9 hours
  and both fits pass. Both campaigns then stop early with NFPP 97% and 99%. Each is
  led by a planet transiting the same neighbour, Gaia DR3 2101482387899618048: a K
  dwarf 3.4 arcsec away, 1.0 mag fainter and at a smaller parallax. The second
  generation of the 32.58-day signal on Gaia DR3 2077526743952548224 passes with a
  12.3-hour transit, still at eccentricity near 0.8 on an M dwarf. It stops early
  with NFPP 43%, led by a planet transiting a neighbour. The 4.13-day signal on Gaia
  DR3 2052149122233322368 needs eccentricity near 0.92 for an 8.1-hour transit, 4.4
  times circular on its M dwarf. Its second generation lengthens again, to a
  26-hour transit at a stellar density of 1.5–3.1 kg/m³ against the host's
  6,071 ± 1,798 kg/m³, so no posterior sample has its periastron outside the star.
  No third generation was staged. KIC 8308249's
  0.97-day signal (Gaia DR3 2078321583479039232) wants a 4.6–6.1-hour transit, 4.9
  times circular on its M dwarf, with its clock at the edge of its prior box. Its
  native photons cover two quarters. Quarter 3 shows the signal at the delivered
  clock (14,145 ± 367 ppm). Quarter 4, 2.1 times noisier, has no dip there
  (−2,339 ± 955 ppm) and a 30,264 ppm feature 2.4 hours later. The fit spans both,
  and post_boi.native stops at its geometry check. A duration-aware generation
  would keep both quarters and the clock prior, so none was staged. KIC 8012933's
  14.85-day signal (Gaia DR3 2105840847691288832) needs eccentricity 0.47–0.72
  for an 8.6-hour transit, 1.9 times circular. Its first preparation absorbs 24%
  of that transit's depth (6.9σ), so its second generation is queued.
- **Structure and holds.** 47 aliases or duplicates, 43 signals that are survey
  BOIs, and holds that name the V3 step and its evidence. Most holds are a
  response closure in fewer than two quarters, a capable neighbour without
  stellar properties, or a host without TIC mass and radius. KIC 9173600's
  12.70-day signal is held at protection without a second generation. Its native
  photons (quarters 5, 9, 13 and 17) show no dip at the delivered clock
  (30 ± 337 ppm). The fit moves to a period 0.0046 days shorter, inside its prior,
  and finds a dip of S/N 4.2 that quarter 17 contradicts.

Batches 6 and 7 prepared the remaining signals: 402 that failed only selection
checks (search S/N below 10, odd/even, astrometry or Hill stability) and the 21
strongest members of event-sharing families as single candidates. From October 26
their campaigns run on `fast_triceratops.engine()`, and hosts without FLAME or
TIC mass and radius get isochrone solutions; see the
[BOI page](boi-validation.md#isochrone-hosts-october-26).

On October 26 Adam stopped the queue after its higher-value cases and parked the
staged signals with search S/N below 10 before any fit (`NOT_RUN_LOW_SNR_PARKED`).
The 2–4% FPP of the eight finished single-host M/K campaigns is an observed
pattern, not a limit: a strong signal on a single host can pass, as KIC 8878385 b
did at 0.28% on a G star. Strong single-star signals therefore stay in the queue.
These include all with search S/N of at least 10, and eight parked signals whose
local SAP event S/N is at least 13 (at least 10 after the measured 1.3× error
underestimate). One more is KIC 10286954 (31.1 days, S/N 15), unblocked by an
isochrone solution for its capable neighbour. The remaining 238 resume from the
[parked list](../results/boi-validation/mk-parked-low-snr-20261026.json). Strong
signals still held are blocked by flux closure, aperture authentication, a scene
source without photometry, or a neighbour within 1.5 arcsec that lacks a usable
parallax.

Two event-review tests added on October 25 failed null calibration and decide
nothing. At six transit-free phase offsets, the local per-event errors are
1.3–1.5 times too small and per-quarter sums 2.5 times too small. As a result,
8.1% of null trials pass a 3σ within-quarter odd/even difference (8.2% at the
real phase), and 19% pass a quarter sign-reversal condition
([calibration](../results/boi-validation/mk-event-review-calibration-20261025.json)). The detector's odd/even flag
marks 6.8% of BOIs at 3σ, close to that null rate. It stays a conservative
selection requirement, not evidence of an eclipsing binary, and flagged
signals still run through V3.

## Select new signals

`post_boi/selection.py` compares the completed October 19 M/K proposals with the
full frozen 406,547-source BOI catalogue using its existing ephemeris matcher.
An epoch rebase does not make a signal new. Period-related proposals on an old
BOI source require review; a period ratio alone does not resolve identity.
The initial comparison has 663 proposals: 23 previous signals, 561 proposals on
417 previously absent BOI sources, and 79 proposals on previous BOI sources
requiring signal or alias review. The full baseline has 3,427 BOI groups on
2,907 sources; it includes both survey detectors and all retained BOI shapes.

Preparation priority uses source/timing review, available host properties,
localization, transit information and scene size. The first queue is provisional.
Update priorities using authenticated native coverage, current stellar records,
existing follow-up and complete scenario cost before freezing a production plan.
A catalogue non-match is not a novelty claim. The deferred detector calibration
gate remains open and separate from post-BOI follow-up.

## Review Kepler issues

`post_boi/spacecraft.py` authenticates native SAP against the saved detector views
and compares the original mask with explicit diagnostic alternatives. It uses the
bound run configuration: quality bitmask -1 and no flattening mask for SAP.
It measures local event depths, quality-flag overlap and module/output history.
The added all-quality comparison is diagnostic; it does not replace an input mask.
Per-event linear baselines and scalar excess-noise errors are limited diagnostics.

The campaign retains NASA's Data Characteristics Handbook KSCI-19040-005,
Data Processing Handbook KSCI-19081-003 and Certified False Positive Table
KSCI-19093-003. Review rolling bands, sensitivity dropouts, safe modes, pointing,
momentum management, thermal recovery, quarter boundaries, common extraction
responses, optical ghosts and detector/electronic contamination where supported.
Flag overlap does not reproduce a mechanism. Parent pixels, rolling-band severity,
release-note windows and native model residuals remain explicit follow-up work.

## Prepare and run

All work and heavy outputs stay on kbonus in
`/srv/kbonus/dump/mk-statistical-validation-50-20261020-v1`.
The native diagnostic ran from a frozen copy under a supervised systemd user
service with 32 workers and one numerical thread per worker. Fresh Gaia, native
archive, ExoFOP and complete known-signal catalogue responses retain dated receipts.
The obsolete ExoFOP endpoint response and corrected request remain preserved.

Prepare candidate-specific native fits, host/scene priors and supported models.
Refresh exact-signal literature, resolve ownership/timing contradictions, review
existing imaging/companions and run the appropriate reference controls. Freeze
and verify each adapter before `post_boi.campaign resume`. The probability protocol
remains KEPLER_BOI_FINAL_PRIMARY_V1: exactly twenty final-clock primary repeats,
one million attempted draws per family, and zero or three justified diagnostics.
A candidate counts only after deep numerical replay and physical closeout.

Keep numerical threshold failures, reproduced astrophysical false positives,
reproduced instrumental artifacts and unresolved candidates separate. Report
rates with their actual completed-follow-up denominator; keep rates null before
any candidate closes. A failed FPP threshold alone does not reproduce an EB.

## First conditional probability studies

The October 20 launch snapshot is
[`mk-conditional-primary-launch-20261020.json`](../results/boi-validation/mk-conditional-primary-launch-20261020.json).
It binds the separate campaign for Gaia DR3 2128845796339643392 and
2127732197218050304: twenty final-clock integrations each, with three paired
recovered-clock diagnostics only for the first target. Linux supervision uses
the available CPU affinity count and one numerical thread per worker.

The numerical results remain conditional on the declared stellar, aperture and
noise models. The hosts fit within the retained limb-darkening table; off-grid
companion approximations, source ownership, noise sensitivity and dilution-aware
radii need physical closeout before admission. A numerical pass does not increment
the fifty-object objective. Keep probability-criteria failures, reproduced false
positives, instrumental mechanisms and unresolved cases as separate outcomes.

## Readiness and model controls

The [next-readiness snapshot](../results/boi-validation/mk-next-readiness-20261020.json)
records corrected native priors, newly authenticated Q17 inputs and the
transit-first queue. Keep broad clock coincidences separate from narrow
phase/drift/duration flags; neither reproduces an instrumental mechanism.

A prospective physical quadratic limb-darkening prior passes mathematical
support and complete-census renderer controls. It remains outside production
until paired known-object calibration and model review pass. The 52 acquired
Kepler control products and aperture checks authenticate inputs; they do not
close empirical calibration or candidate physical admission.

The [prospective-control snapshot](../results/boi-validation/mk-prospective-controls-20261020.json)
records explicit wider native transit windows and stellar-density support. A
published Kepler-442 stellar prior exposed unresolved asymmetric quadrature;
splitting the integration at its discontinuities passes independent mass and
resolution checks. The real-photon wider-window likelihood also agrees with an
independent Gaussian marginalization. These are numerical controls; known-object
posterior recovery and empirical probability calibration remain separate gates.

A read-only receipt audit passes, and independent reconstruction reproduces both
active candidates' frozen aperture-response arrays exactly. Native-clock pixel
review places Gaia 2127732197218050304 on a physical source-ownership hold: its
robust offset is 6.21 arcsec at 4.44 sigma. Preserve its BOI label and conditional
probability repeats. Measure likely donors in authenticated native photons and
reproduce the aperture/pixel response before assigning a false-positive mechanism.

## Native reference controls and donor review

The [native-control and donor snapshot](../results/boi-validation/mk-native-controls-and-donor-review-20261020.json)
records converged Kepler-42 and Kepler-442 native fits and recovery of their
published parameters. Kepler-42 also passes posterior exposure-integration
checks. Keep these results separate from empirical planet/EB probability
calibration, which remains open. Its registered masks protect the other two
known planets consistently in both fitted and probability-input photons.

The cross-source clock screen includes periodic sealed BOI proposals, current
M/K proposals and preserved historical fitted clocks. It finds no local periodic
matches within 120 arcsec. Fourteen proposals have nearby older BOIs without
usable periodic clocks; preserve these for event-level review. No match, offset
or period ratio alone establishes an alias or false-positive mechanism.

Joint target/parent aperture measurements reproduce the native pixel sums and
shared-photon covariance. Fixed-source response profiles do not identify a unique
donor for Gaia 2127732197218050304, so its physical ownership hold remains open.
Six more K-star native preparations have exact current TIC identities. The strong
Gaia 2080571429081419520 fit uses TIC 1881690369; its Gaia/TIC temperature
disagreement still requires physical model review before probability admission.

The [next native-case snapshot](../results/boi-validation/mk-next-native-case-20261020.json)
records Gaia 2105665750463516800: accepted native convergence, posterior exposure
checks, coherent conditional quarter timing and stable parent-engineering
comparisons. Its free centroid is target-compatible; fixed-PRF attribution is
still limited. Retain the competing-BOI mask and explicit partial-quarter noise
inputs. Scene preparation does not supply probability or physical admission.

The known-binary reference exposed a broad-shape 54-point quadrature failure.
Preserve that failure. A separate native 270-frame refinement passes independent
live-frame checks; it does not close the probability renderer's 54-point gate.
The registered blinded model experiment remains separate from new-candidate
primary budgets and requires complete reference readiness before execution.

The [old-source event review](../results/boi-validation/mk-prior-source-event-review-20261020.json)
keeps all 79 potential additional or aliased signals on previous BOI hosts in
scope. Sixty share possible saved-event or predicted-clock windows with older
proposals; ten have incomplete older event evidence. These flags do not resolve
identity. All 23 previous exact matches pass the saved-event positive check.
Known-reference imaging and spectroscopy metadata are available, but a contrast
reported in another filter is not automatically a Kepler-band constraint.

## Recovered host priors and native readiness

The [recovered-host snapshot](../results/boi-validation/mk-recovered-host-readiness-20261020.json)
records 199 reciprocal Gaia/TIC recoveries and two unresolved identities. Fresh
mass and radius inputs restore 194 proposals to prior review. Four more targets
have authenticated native preparations; nine fresh fits are registered behind
the existing native coordinator with fixed sampling budgets.

Gaia 2126063585245659136 passes native convergence with the supplied TIC
uncertainties, resolving that specific prior-alignment hold. Its posterior
exposure checks pass, but conditional period-profile structure still needs
alias/timing review. Gaia 2131263686828359808 also passes the declared quantile
radius-ratio SNR gate and exposure checks. Its median/std SNR is 2.48, so preserve
the broad radius tail for physical review rather than presenting all uncertainty
summaries as passing.

The additional 11.79-day signal on Gaia 2104058096963805696 passes native
convergence with every older primary and half-phase window excluded. Its
identity remains unresolved. A broader catalogue screen finds a nominal 3:1
clock coincidence with distant Kepler-380 b; intervening-cycle photon tests and
detector geometry are needed to interpret it. Harmonic flags and different
channels alone do not establish or exclude every contamination mechanism.

The reference-evidence review distinguishes published validation from the
proposed comparison using less follow-up evidence. Raw imaging curves retain
their filters; a published model-dependent Kepler conversion is not a universal
contrast limit. The eighteen paired control repeats remain unlaunched, with
binary native convergence and probability-renderer accuracy still open.

Count probability progress from `checkpoint.completed_scenarios`.
`complete_scenario_names` is the scenario census, not completed work. No new
candidate has completed probability and physical admission; the count remains
zero and false-positive rates remain null.

The [additional-signal donor review](../results/boi-validation/mk-oldhost-native-donor-review-20261020.json)
records a coherent fitted 11.79-day clock and passing posterior exposure checks.
After the registered competing-signal masks, 27 usable current events have a
mean depth of 1082 +/- 95 ppm; 53 intervening Kepler-380 clock cycles have
5 +/- 66 ppm. Their 9.29-sigma difference uses the declared local error model
and disfavors a simple full-clock, equal-depth copy. All thirteen shared quarters
use different detector channels and modules. Selective transfer, other mechanisms,
source identity and physical admission still require their own evidence. The
27-event per-cadence masked result differs by construction from the earlier
24-event diagnostic that excluded overlapping windows in full.

The [preparation-limits snapshot](../results/boi-validation/mk-preparation-limits-20261020.json)
retains the two-eclipse binary reference's failure at its fixed 24,000-step
endpoint: maximum rank Rhat is 1.129, above 1.03. No extra sampling or probability
control repeat is authorized by that outcome. Five new IRSA field-extinction
requests failed; a bounded direct service diagnostic also timed out. Preserve
the failures and leave the affected population preparation open, without
substituting zero extinction, host extinction or a neighboring sightline.

## Eight-hour progress and five-run triage

The [eight-hour server snapshot](../results/boi-validation/mk-eight-hour-progress-20261020.json)
records zero complete full-census repeats and zero new admitted planets. Thirty-two
workers process the first two targets: twenty final-clock repeats and three
justified diagnostics for Gaia 2128845796339643392, and nine final-clock repeats
for Gaia 2127732197218050304. The original scheduler started many repeats together;
scenario-block progress does not provide an FPP or a reliable wall-time estimate.

Adam authorized a fixed first-five final-clock screen: stop further probability
work when the arithmetic mean point FPP is at least 0.03. Twenty repeats and all
physical checks remain mandatory for admission. Preserve the BOI label and all
unfavorable results; compute triage is not a reproduced false-positive mechanism.
The original coordinator received SIGINT only in its controlling Python process;
its executor drains the 32 surviving workers without submitting the remaining
jobs. A frozen triage supervisor waits on the same coordinator lock, then checks
complete results and replays weights. No integration physics, packets, priors,
seeds or running-worker files were changed. The eleven unstarted repeats for
Gaia 2127732197218050304 also remain held by its unresolved source-ownership
offset, regardless of FPP. Prefer another physically prepared target.

Three additional native fits pass convergence in the current batch. In a separate
registered comparison, an exact depth/limb-darkening coordinate transform reaches
the native convergence gates on Kepler-442, Kepler-42 and the binary fixture within
their original maximum budgets. The predecessor binary failure remains preserved.
The comparison's serialization-only recovery retains completed sampling without
repeating those steps. Published-parameter and posterior exposure checks for the
new coordinates and the probability-renderer accuracy gate remain separate;
production adoption and the eighteen blinded probability controls remain open.

The five failed extinction acquisitions now have authenticated official GET
responses, with a saved transport gold check reproducing the old service values
exactly. All ten associated official stellar-population files have arrived.
The canonical new inputs are `recovered-dust-v10`, `oldhost210405-dust-v4`,
`recovered-populations-v8` and `oldhost210405-populations-v2`. Earlier failed native
map lookups and service requests remain preserved. Actual image-pixel controls
also passed, but their partial-coverage alternative is superseded by direct
source-coordinate service responses. These inputs prepare scenes; they do not
close native, probability or physical admission by themselves.

## One candidate at a time

Adam's subsequent instruction supersedes the parallel preparation and two-target
schedule. The [single-target operation record](../results/boi-validation/mk-single-target-operation-20261020.json)
keeps only Gaia 2128845796339643392 active. Its twenty primaries and three assigned
diagnostics continue. The nine other probability workers and all competing native
and geometry jobs were explicitly stopped; their saved evidence remains. Do not
resume them automatically. Future triage plans must contain exactly one target.

The selected result observer checks the fixed first five as they complete. At mean
point FPP >= 0.03, it preserves partial checkpoint snapshots and interrupts the
unfinished selected workers. Otherwise it waits for the complete assigned set and
replays every saved weight. A numerical pass still requires timing and physical
closeout before admission. The initial observer generation failed path-integrity
verification before observing or signaling any worker; the corrected entry loads
the original frozen runner paths and passes the unchanged dependency checks.

A bounded profile on the selected target identifies quadratic eclipse rendering
as the main measured cost: 0.669 of 0.931 seconds in one far-neighbor profile,
versus 0.101 seconds in physical sampling. Its separate 4096-draw operational
profiles are not probability repeats or admission evidence. Removing contention
helps; the serial 132-scenario census also limits latency. Preserve all scientific
counts and compare any future execution optimization on identical physical draws.
Do not promise a two-hour result from block counts or this small profile.

## Rendering throughput and first-five priority

The [selected-target execution review](../results/boi-validation/mk-selected-execution-throughput-20261020.json)
records identical weights and flags across the complete 132-scenario census at
one, two, four and six renderer threads. All controls use the same physical draws.
Counterbalanced timings on fixed samples show six-thread rendering speedups of
3.72–3.94 for the three tested planet scenarios and 1.76 for the binary scenario.
These small measurements under concurrent load do not predict a full-candidate
completion time or establish physical validation.

The prospective scheduler reserves six CPU slots for each first-five repeat,
then two for each later primary or assigned diagnostic. It never reserves more
than the declared thirty-two slots, and it retains the fixed five-run screen.
The actual frozen worker boundary passes its environment, runtime-pin, full-freeze
and scientific-count checks in separate zero-draw controls. Those controls create
no probability result or primary repeat. Forty-two package tests pass, including
the real concurrent queue filling its slot budget without overcommit and keeping
later jobs behind the first-five decision. No current worker was restarted or
moved into a new science generation.

CPU inspection found four current first-five workers on efficiency cores. All
five now run on the fastest five distinct performance cores; the other eighteen
selected-target workers use the remaining three performance and fifteen
efficiency cores without sibling contention. All original PIDs, numerical thread
settings, seeds, checkpoints and frozen files remain. Only the selected target
continues to consume scientific CPU.

The first five complete target-planet scenario weight files give conditional
radius medians of 2.02–2.07 Earth radii, with effective weighted draws 375–520.
This uses saved photons, stellar/scene priors and dilution in the declared
prepared likelihood; no new draws were integrated. It is conditional on a planet
on the target and does not establish source ownership, reject binaries, supply a
complete FPP or replace the accepted native fit and scene/dilution review. Source attribution,
correlated-noise sensitivity, off-grid companion limb darkening and dated
literature still need physical closeout before any admission.

The [hardware and resolved-source review](../results/boi-validation/mk-selected-hardware-and-source-review-20261020.json)
records a two-second utilization measurement with twenty-three workers assigned
to distinct physical cores of the twenty-four-core, thirty-two-thread server.
All probability workers belong to the selected target. Memory, I/O and CPU
pressure remain negligible. This establishes high physical utilization; it does
not establish optimal end-to-end throughput or a candidate completion time.

A separate reuse of the selected target's saved difference-image centroids
reproduces the original target offset diagnostic before evaluating all forty
catalogued source positions. Seventeen quality-qualified events in six quarters
retain target compatibility at offset significance 1.72. The nearest brighter
neighbor, 6.71 arcseconds away, has offset significance 3.29 against the retained
threshold of three. Each alternative recomputes its own directional formal,
scatter and jackknife variance. All thirty-nine alternatives exceed that
threshold, but catalogue-epoch-only positions remain explicitly marked. These
are offset diagnostics, not independent source probabilities or contrast limits.
No new probability draws, astrophysical false-positive decision or admission
follows; unresolved physical requirements remain open in the record.

## Selected-star noise and literature review

The [selected noise and literature review](../results/boi-validation/mk-selected-noise-and-literature-review-20261020.json)
reproduces all twenty-eight original event measurements and both engineering
comparisons. Sixty fixed off-transit controls, protected from the primary signal,
have maximum signed SNR 2.22 with a local linear baseline and 2.10 with parent
pointing/background terms; the candidate gives 5.26 and 2.94 respectively.
Controls share data and are not independent trials or a calibrated false-alarm
rate. Their event counts range from twenty-five to thirty-one.

The candidate's 150 ppm reduction exceeds every paired control change. It is
mainly associated with pointing: pointing-only gives 241 ppm and SNR 3.25,
while background-only gives 370 ppm and SNR 5.11. A stable weighted-design SVD
reproduces the retained solver within 2.1e-9 ppm, and a fixed 300 ppm injected
template is recovered to numerical precision. This does not reproduce an
instrumental mechanism; the data-specific pointing sensitivity remains a
physical review concern before admission.

Fresh official NASA KOI/TCE/planet queries return no matching entries in the
declared source/parent/position coverage. Exact-alias arXiv metadata and SIMBAD
identifier/1.5-arcsecond queries return no indexed matches. ExoFOP authenticates
the target identity and lists no planet or candidate entries. The initial combined SIMBAD request timed out; separate
equivalent predicates completed and their empty union is retained with the
failure. ExoFOP lists no imaging, spectroscopy or time-series follow-up. These
are bounded dated queries, not a novelty or universal literature certificate.

The protocol requires accepted native timing/radius inference and consistent
scene/dilution treatment; it does not universally require a second native MCMC
fit that jointly samples dilution. Conditional apparent-radius and TP-only
radius summaries retain their labels. Missing imaging stays explicit within the
unresolved-scene review and does not automatically require new observations.
The original user services and workers were live at that review; system-manager
queries do not report these user-managed services. The later fixed-five closeout
records their completion and preserved stop decisions.

## First fixed-five rejection and next selected target

The [first-screen and next-native record](../results/boi-validation/mk-first-screen-and-next-native-20261020.json)
closes Gaia 2128845796339643392 with mean FPP 0.37333166 (sample SD 0.01298951)
and mean NFPP 0.33247268 (sample SD 0.01251031). All five fixed-prefix weight
sets and two previously completed later repeats pass full replay. Thirteen
later primaries and three recovery diagnostics were interrupted with checkpoint
snapshots after an audited prefix contribution guaranteed failure of the
unchanged three-percent cutoff. The remaining prefix finished. This rejects the
validation screen and preserves the BOI; no physical false-positive mechanism
was reproduced. One screen has failed; a population false-positive rate remains
unmeasured.

The observer could read an incomplete result snapshot just before the final
worker committed its result and exited. The prospective observer now reads
worker liveness before the result snapshot. Two event-order regressions fail
before the fix; all forty-five package tests and Ruff/complexity checks pass
after it. Original frozen runtime and observer files remain unchanged; a separate
full replay closes the completed screen.

The sole next host is Gaia 2131705076324568448 (KIC 10844387): forty-six measured
events, diagnostic SNR 10.96 and nineteen resolved neighbors. A fresh native
generation uses authenticated never-started inputs. Four independent
twenty-four-walker ensembles pass at 6000 steps: rank Rhat 1.01594, epoch ESS
332.06 and quantile-width radius-ratio SNR 12.45584. Eight serial-versus-parallel
chain/log-density comparisons match bitwise. Fork workers materialize immutable
prior arrays, preserving the earlier compressed-reader failure. Sampling used
eight evaluation workers per ensemble; short controls do not predict FPP timing.

Eighty posterior draws pass the frame-integration audit. All eight quarters'
geometry is prepared, with the original first-quarter arrays reproduced bitwise.
A separate metadata review assembles all five tail specifications without
altering arrays. The twenty-star scene uses this field's stellar priors and
official Gaia/Kepler populations. Probability preparation, final-clock pixel and
engineering review, model readiness and timing-budget freeze remain required.

The subsequent [source-hold and next-selection record](../results/boi-validation/mk-preparation-hold-next-selection-20261020.json)
holds Gaia 2131705076324568448 before any probability integration: its final-clock
centroid offset is 2.085 arcseconds with uncertainty 0.671 arcseconds, exceeding
the retained significance threshold (3.106 versus 3). Six of seven uncalibrated
fixed-PRF comparisons prefer the target; that does not silently override the
centroid conflict. Native and scene results remain preserved, with no denial
or change to the BOI.

The sole next source is Gaia 2053566049119753216 (KIC 5963572), using its already
completed 12000-step native posterior without resampling or resuming a stopped
job. Rank Rhat is 1.01150, epoch ESS 728.23 and radius-ratio SNR 23.06782. Its
posterior numerical audit passes; conditional timing profiles remain measurements
for review. The
previously prepared fifty-star scene is preserved; final-clock pixels are
reviewed before a new probability campaign.


## Current source holds and hardware use

The [preparation record](../results/boi-validation/mk-preparation-hold-next-selection-20261020.json)
also holds Gaia 2053566049119753216. Joint-quarter maps give a target offset
significance of 3.38 and compatibility with the bright parent KIC 5963556 at
1.06. This does not identify a transfer mechanism or reassign the signal.
Matched-cadence SAP agrees with the frozen native export, but its relationship
to the recorded aperture sum of parent pixels remains unresolved in several
quarters. Preserve the delivered data and this failed equivalence check.
Skip probability work on this case while selecting the next ready source.

The [live hardware audit](../results/boi-validation/mk-live-hardware-audit-20261020.json)
finds no probability workers or active science services, negligible CPU use and
about sixty GiB of available memory. The prospective first-five schedule reserves
thirty of thirty-two logical CPU slots for one target; later work can reserve all
thirty-two. These are budgets, not measured utilization. Small renderer and native
controls preserve numerical outputs and improve speed, but optimal full-campaign
throughput and a whole-candidate completion time have not been established.
The next admitted screen must retain actual CPU and scenario throughput evidence.


The next selected source is Gaia 2130921124535127424 (KIC 10527790). A fresh
output generation uses its authenticated, previously unsampled native
preparation. Four ensembles retain their original twenty-four walkers, seeds and
endpoint rules. Serial versus six- and eight-worker controls give identical
chains, log probabilities and acceptance counts. The selected eight-worker
configuration reserves thirty-two workers. A twenty-second live measurement
finds about forty-five percent logical CPU occupancy, equivalent to fourteen
point four busy CPUs. This exposes a utilization gap despite the worker budget;
execution overhead and full-scenario throughput still need measured review.
No other candidate is running and no probability result is claimed.

The selected native service subsequently completed successfully at the six-thousand
step endpoint. Its native numerical convergence gate passes; posterior numerical,
timing and physical review remain required. No probability integration is running
at this closeout. Keep the forty-five percent measurement tied to the sampling
phase rather than treating it as current idle-state utilization.

## KIC 10527790 probability screen

The [selected-source launch record](../results/boi-validation/mk-single-213092-probability-launch-20261020.json)
binds Gaia 2130921124535127424 to its fitted 46.55452963-day signal. The accepted
native fit has rank Rhat 1.02418, epoch ESS 310.51 and radius-ratio SNR 13.244.
Posterior exposure checks pass and conditional quarter timing has no strong
contradiction. The native medians determine adoption; assign zero recovered-clock
integrations.

The delivered SAP and errors are reproduced for all 58,404 retained photons
using the recorded aperture, matching cadence numbers and pixel origins. When
KBonus supplies a nonzero background, restore the archive pixel background and
subtract the recorded SAP background; zero-background quarters reproduce direct
pixel sums. The fixed-shape depth is 1,725 ppm in delivered SAP and 1,719 ppm in
the archive aperture sum. The correlated additive correction contributes about
9 ppm, with no significant transit-shaped amplitude. This authenticates the
scalar transformation; it does not independently establish source ownership or
calibrate every background-model uncertainty.

All 123 scenario families pass same-draw reference/accelerated comparisons and
one-, two- and six-thread weight equality. Independent covariance controls
respect the declared retained subspace: the final packet retains all 200 modes,
while the unused recovered-clock diagnostic packet retains 170 of 175 populated
bins. Preserve the original full-rank test failure and its corrected independent
eigenmode comparison. Neither the likelihood nor its projection threshold changed.

The five-worker mixed-workload benchmark includes physical sampling, rendering
and weight persistence. Six renderer threads per worker is fastest among one,
two, four and six: median slowest-worker time is 2.189 versus 2.831 seconds.
These short controls do not predict a complete-campaign time. Frozen production
reserves thirty logical slots for the first five and uses two renderer threads
per later repeat within the thirty-two-slot budget. A live launch check verifies
exactly five Python workers and their actual numerical thread environments.

The new campaign freezes 855 dependencies, twenty independent final-clock
primaries, one million attempts per family, fixed batch 256/render 64, phase grid
800 and 54-point Kepler exposure integration. One durable user service runs the
first five only. Mean point FPP at least 0.03 stops the budget; a pass releases the
remaining fifteen. Every full result, unfavorable draw and failure remains.

The centroid is target-compatible but broad: offset 2.893 arcseconds with
uncertainty 6.146 arcseconds. Do not describe this as unique localization. Keep
neighboring-source probabilities, dilution and physical ownership review open.
Pointing/background conditioning and rolling-band exclusion preserve the strong
transit. The fixed-shape odd/even difference is about 2.47 sigma; retain it and
the complete binary/doubled-period census.

Current NASA source/parent/position queries, SIMBAD identifier/cone queries and
arXiv indexed aliases return no matching rows. ExoFOP authenticates the source
and lists no planet, companion, imaging, spectroscopy or time-series entries.
These are bounded query results. Host limb-darkening support passes; cool or
off-grid companion approximations in 101 families remain explicit model
limitations before physical admission. No statistical validation, dynamical
confirmation, novelty or completed detector calibration follows from this launch.

## Source conflict, cool-host hold and next preparation

The [KIC 10527790 hold record](../results/boi-validation/mk-single-213092-physical-hold-20261020.json)
supersedes its conditional launch readiness. A separate gold-checked joint-quarter
measurement uses all accepted native events, native masks and multiple fixed
position starts. Ten quality-qualified quarters give a 16.731 ± 1.808 arcsecond
offset from the proposed source, with diagnostic significance 9.255. The bright
parent KIC 10527775 is compatible at 0.746 ± 1.803 arcseconds. This follow-up is
not a recalibrated shared vetting threshold or a source posterior probability.

The disjoint parent optimal-aperture pixels do not show the candidate dimming:
depth is -3.70 ± 5.68 ppm at the BOI epochs and -11.87 ± 6.13 ppm at intervening
half-period phases. The BOI aperture depth is about 1,725 ppm. Thus neither source
reassignment nor a simple uniform parent-eclipse transfer is established.
PRF residual mismatch, source ownership and instrumental alternatives remain open.

The probability generation was explicitly stopped after preserving its handles
and checkpoints. All five workers are confirmed absent. Fifteen completed
million-draw scenario blocks pass deep weight replay and the full input freeze
still verifies. There are no complete probability repeats or first-five FPP for
this case. Preserve its BOI, every result and failure; do not resume this stopped
generation automatically.

The [next-preparation record](../results/boi-validation/mk-source-model-holds-native-repair-20261020.json)
also holds Gaia 2103792461826460288 (KIC 5689121) before native sampling or
probability work. Its own thirty-star scene is prepared, but 19.043% of the
weighted host prior lies outside the retained Kepler atmosphere table. Preserve
that mass; do not clip the prior, substitute Solar composition or silently
accept extrapolation. Supported model development and review remain open.

The sole next candidate is Gaia 2052740174146478464 (KIC 3544862), with exact fresh
Gaia/DR2/TIC identity, own stellar specifications and authenticated native/parent
photons. It has fourteen diagnostic events, eleven quality localization quarters
and forty-nine neighbors. Complete its joint source follow-up before spending
probability repeats.

Its first native preflight exposed an operational map-recording defect: one
rejected optimizer start has negative-infinite density, which strict JSON cannot
represent. The post_boi.native_maps.write_map helper preserves that attempt with
an explicit outside-support state and rejects any nonfinite winning map. A
separate native output generation uses the unchanged likelihood, priors, bounds,
optimizer starts, seeds, walkers and endpoint budget. Finite-map output reproduces
an existing native gold byte-for-byte, serial/parallel chains match, and all
fifty-two package tests plus Ruff and complexity checks pass. No scientific
rejection or numerical admission follows from the recording repair.

## Native radius-information stop

The [native-information record](../results/boi-validation/mk-native-radius-hold-next-selection-20261020.json)
holds KIC 3544862 without any probability integrations. Its 12,000-step endpoint
meets parameter convergence: rank Rhat 1.01886 and epoch ESS 327.01. Radius-ratio
SNR is 1.69967, below the required three, and its broad posterior width is stable
across the 3,000-, 6,000- and 12,000-step endpoints. This is insufficient radius
information, not a failure to mix or a reproduced false-positive mechanism.

The already-started final sampling block was interrupted by SIGINT to the exact
four HDF writer processes. All four partial backends remain readable, with
23,318–23,345 committed iterations; preserve them alongside the exact saved
12,000-step posterior. The first closeout attempt failed while serializing a
NumPy shape integer. Its partial receipt remains, and a separate complete receipt
records the corrected primitive types. Do not overwrite or resume these chains.

Prospective native coordinators should stop at the first converged endpoint
whose radius precision fails the declared criterion. More Monte Carlo draws
cannot be treated as new transit-shape information. This changes computational
triage only; preserve the same physical likelihood, thresholds and maximum budget.

The next source for dated identity and prior review is Gaia DR3
2126998926041980032, provisionally associated with TIC 1882613763. It has 139
measured events, diagnostic SNR 14.19, a supplied transit lasting 6.26 Kepler
cadences and thirty-four neighbors. The nearby 2.37-arcsecond source and host
radius still need review. No KIC is assigned, and the parent TPF KIC must not be
substituted for this Gaia source.

The [sequential candidate ledger](../results/boi-validation/mk-sequential-candidate-ledger-20261020.csv)
summarizes this selection sequence without conflating computational rejection,
preparation holds and physical false positives. One complete five-run screen
fails its FPP budget threshold; five later cases are held on source, model or
native-radius information. The physical false-positive rate remains unknown.
The next Gaia source's fresh DR3/DR2/TIC identity is authenticated as TIC 1882613763;
its own stellar and native-photon preparation is separate from an accepted fit.

The new coordinator implements the first-converged radius-information stop.
Frozen prior endpoints verify continue, positive acceptance and radius-hold
decisions with unchanged thresholds. Its own native fit is running after exact
Gaia/TIC and photon authentication. Before probability production, review the
large Gaia/TIC temperature difference and the close neighbor. The official field
dust query gives E(B-V) 0.1037 and asymptotic A(V) 0.32147 for R(V) 3.1, whereas
Gaia reports A(G) 1.5235 and monochromatic A0 2.0235. These have different
definitions and spatial resolutions; the tension requires review and does not
automatically select a stellar prior. The conditional native fit retains its
declared inputs, with no probability integration or physical admission.

### Measured server throughput and source check

The October 20 audit does not support a full-capacity or globally optimal
throughput claim. The selected source's native unit used 8,449.064 CPU-seconds
over 622.281 wall-seconds: 13.578 logical CPU equivalents, 42.430% of 32.
Its short benchmark took 15.484 s with four serial ensembles,
7.995 s with 24 evaluation workers and 7.634 s with 32. For identical seeds,
saved chains, log densities and acceptance counts were identical. This short
control does not establish end-to-end scaling. Memory capacity and CPU quotas
did not constrain the job; active-job disk latency was not profiled.

For Gaia 2126998926041980032 / TIC 1882613763, the 12,000-step native endpoint
passed Rhat 1.01194, epoch ESS 538.30 and apparent-radius-ratio SNR 38.05.
Separate native-clock source measurements completed 140 individual events,
joint-quarter multistart fits and 35 catalogued source hypotheses. The
single-event gold replay passed. Five quality quarters place the transit
0.0341 +/- 0.1618 arcsec from the proposed source. The nearby parent is
displaced 14.36 sigma in this diagnostic. These measurements are not source
posterior probabilities or a false-positive mechanism.

Five quarter timing profiles touch their scan bounds. The period profile
also favors its boundary and the native duration approaches its declared
upper bound. These findings and the Gaia/TIC stellar discrepancy require
review before probability integration. No probability jobs are active and
the validated count remains 0/50. The lean hardware/source progress JSON
records the measurements; all original records, priors and freezes remain.

### Timing and stellar hold; next sequential native fit

Gaia 2126998926041980032 / TIC 1882613763 remains a BOI. Its native
numerical gates passed and five joint-map quarters localize the signal
to the target, but the period and duration posterior approach their
declared limits. Five conditional quarter timing scans touch their
bounds. The exact TIC record has a duplicate link to the nearby parent;
the Gaia temperature, metallicity and extinction solution conflicts
with TIC and the field dust estimate. Resolved PS1 counterparts for the
two Gaia components were associated at their observing epochs; their
photometric flags and shared mean/stack photons remain explicit.
The source is on a timing/stellar readiness hold, with zero probability
results and no astrophysical false-positive adjudication.

The next selection is Gaia 2052478387302513152 / KIC 4661212 /
TIC 138215498, signal 7. It is absent from the sealed survey BOI sample.
The native diagnostic measured 18 events with depth SNR 7.90 and
event-depth chi-square per degree of freedom 1.38. Its short duration
and 61 neighbors remain preparation risks. Fresh reciprocal source
identity, explicit stellar priors and native photon preparation were
saved. A preliminary 4,096-particle host catalogue bank lies inside
the retained Kepler limb-darkening table; final scene and companion
support still require review.

A source-specific 128-step comparison used the original four ensembles,
24 walkers, seeds, likelihood and density Jacobian. Four serial ensembles,
six/eight processes per ensemble and six/eight thread-local evaluators
produced bitwise-identical chains, log probabilities and acceptance arrays.
The complete group times were 4.587, 3.103, 2.857, 4.287 and 4.574 seconds
for serial, six processes, eight processes, six threads and eight threads.
Eight processes per ensemble were selected. Threading was slower and is
not selected for this fit; the comparison remains a recorded experiment.

The one-target supervisor now follows an accepted native endpoint with
posterior quadrature, timing profiles and individual/joint/multistart
source diagnostics automatically. Probability admission remains a
separate review. The successful NASA, SIMBAD and OpenAlex query records
remain bounded literature/catalogue evidence; arXiv returned HTTP 429
and is explicitly unavailable. No novelty claim follows from these queries.

KIC 4661212 reached its declared 24,000-step endpoint. Rhat 1.00847
and epoch ESS 750.51 passed, but radius-ratio quantile SNR 2.47143
failed the required value above 3. The supervisor terminated normally
without probability runs or a sampling extension. All four full chains,
posteriors and endpoint diagnostics remain preserved. This is a
radius-information hold; it does not establish an astrophysical false
positive.

The next sole target is Gaia 2079240642062921984 / KIC 9046816 /
TIC 268391058, signal 11. Its 0.21-day duration spans about 10.3 Kepler
cadences. The older source-specific preparation had no sampler checkpoints.
Fresh source identity and the exact same mass/radius prior parameters
were verified. Current corpus arrays reproduce every native export
column, all original parent-pixel hashes match, and the unchanged
density-prior preparation is reused in a separate generation. No old
coordinator or held inference was resumed. The native supervisor retains
the same fixed seeds, numerical controls and automatic review stages.
The validated count remains 0/50; the physical false-positive rate remains
undetermined.


KIC 9046816 reached its fixed 24,000-step endpoint with Rhat 1.38901
and epoch ESS 38.52. Its native fit remains unconverged, so no period
or epoch was admitted for probability integration. The complete chains
and original BOI remain preserved without an automatic extension.

Gaia 2079648041177607424 / KIC 8441616 passed the numerical native gate
at 12,000 steps: Rhat 1.01087, epoch ESS 512.53 and radius-ratio quantile
SNR 10.74. The 30.60-day signal has 41 measured events. Additional events
in its shorter 1/7 and 1/5 folds were consistent with no transit,
supporting one physical signal family. Every original BOI is retained.
Grouping those aliases prevents all 564 primary-transit cadences from
being erased by other-proposal masks.

The joint centroid is compatible with the target, although one resolved
neighbor remains compatible too. Two Q11 events prefer an approximately
one-hour early epoch. This persists with free event depths and in the
source aperture read directly from the original parent pixels. The
delivered background correction and subsecond Q11 timing differences
do not explain it. The source is held for a corrected timing model;
no probability run, physical false-positive denial or TTV confirmation
follows from this diagnostic.

The hardware audit does not establish full saturation or maximum
throughput. The complete KIC 8441616 native/control/source-diagnostic unit
used 3,358.10 CPU seconds over 317.65 wall seconds: 10.57 logical CPU
equivalents, or 33% of 32 logical CPUs. Its identical-chain 128-step
comparison took 3.987 seconds with 24 process evaluators and 4.188
with 32; thread variants were slower. All 16 comparisons preserved
chains, log densities and acceptance counts bitwise. These short timings
do not establish full-census probability throughput. Each probability
repeat still traverses its scenario census sequentially, checkpointing
a shared RNG stream. Serial preparation, diagnostics, external data
and readiness review also leave CPUs unused.

The next sole target is Gaia 2052124108331844608 / KIC 3354627 /
TIC 378013956. Its 3.17247-day signal has 313 measured events and native
depth SNR 9.51. Fresh reciprocal identity and exact-source NASA,
SIMBAD, arXiv and OpenAlex queries were reviewed before native sampling;
bounded nonmatches do not certify novelty. Its 4,096-particle preliminary
host bank lies within the retained Kepler limb-darkening table. Its
1.06-arcsecond neighbor and 97-neighbor field remain explicit source
and scene costs. One durable native supervisor runs the unchanged
four-ensemble fit, measured executor selection and automatic
posterior/timing/source diagnostics. Probability admission remains
separate. There are nine preparation holds, one fixed-five high-FPP
budget stop and zero new validations; the physical false-positive
rate is undetermined.

For KIC 3354627, all 16 seeded executor comparisons passed bitwise.
Its fastest tested complete group used 32 process evaluators: 11.796
seconds versus 12.346 with 24 and 25.567 with four serial ensembles.
A five-second active production-sampler sample averaged 18.61 logical
CPUs, or 58% of 32. It had no CPU quota, used about 1.04 GiB and ran
with performance governors. Worker counts therefore do not establish
full utilization, and these measurements do not identify the dominant
native bottleneck or establish full probability-census throughput.

## Aperture preflight and additional signals on old BOI sources

KIC 3354627 passed its numerical native gate but remains held for
quarter timing and shape inconsistencies after free-amplitude checks.
KIC 3241025 also passed numerical fitting, but its stored aperture and
single origin TPF fail to reproduce delivered SAP in all 17 quarters.
The declared PSFMachine v1.1.2 source permits multi-TPF source apertures;
this does not recover this source's complete extraction. Aperture
authentication now precedes new native fitting. Missing pixels are
never zero-filled.

KIC 7468536 authenticated all 18 quarter apertures and passed numerical
fitting. Its joint centroid conflicts with the proposed host at 4.93
sigma; the brighter extraction parent remains compatible. It is held
before probability integration, with no reproduced false-positive
mechanism. There are now 12 preparation holds.

The source-only queue had omitted 39 unmatched proposals on 33 previous
BOI hosts. The separate signal queue includes 38 measured proposals and
one explicit unavailable measurement. Each needs old-clock and photon
review; neither a different period nor an unmatched catalogue entry
establishes an additional planet. The sealed survey remains unchanged.

The sole selected signal is Gaia 2104058096963805696 / KIC 6267058 /
TIC 120318420, with fitted period 11.7927613 days. Its completed native
fit was reused after all frozen dependencies, current photon columns
and stellar priors were verified. Rhat is 1.02005, epoch ESS 341.05 and
radius-ratio quantile SNR 13.68. All 13 apertures authenticate. The
joint centroid is target-compatible at 0.66 sigma and conflicts with
the brighter parent at 6.64 sigma. Direct archive pixels, pointing and
background regressors, rolling-band exclusions and old-BOI exclusions
retain the signal. No alternate clock was selected using probabilities.

A new campaign freezes 93 complete scenario families, one million
attempted draws per scenario, twenty final-clock primaries and zero
recovered-clock diagnostics. The fixed first five stop further budget
at mean FPP at least 3%. All-family numerical and thread controls
passed. Two ordered complete-census throughput controls, at 1,024 draws
per scenario with five workers, selected six renderer threads per run;
this does not establish full-scale or globally optimal throughput.

Five workers are running under one durable coordinator. An initial
launch from the repository was rejected by the freeze before workers
started; the corrected launch uses the frozen runtime directory without
changing its plan or seeds. Retained off-grid companion atmosphere
approximations and other physical-closeout limits remain explicit.
There are still zero new validations, one failed fixed-five screen and
seven completed full probability repeats from that earlier case.
The physical false-positive rate remains undetermined.

The lean progress record is
results/boi-validation/mk-aperture-preflight-and-signal-scope-progress-20261020.json.

For KIC 6267058, the first five completed TP-family arrays already store
physical planet-radius draws and their original importance weights.
Replaying these saved arrays verified each one-million-attempt denominator,
weight hash, evidence, ESS and best-draw radius. No new likelihood or
probability integration was needed. The separate conditional TP radius
medians are 3.046, 3.049, 3.048, 3.063 and 3.061 Earth radii; their
16th--84th percentile intervals are approximately 2.82--3.36 Earth radii.
They remain conditional on TP and the retained model, with the full
scenario census still running. Keep every repeat separate and rebuild
the complete radius record after all twenty primary results.

A fixed 4,096-particle field-prior diagnostic gives target aperture-photon
fraction medians of 0.939--0.982 across the 13 quarters. The direct flux
calculation reproduces the frozen dilution kernel to floating precision.
These are prior-predictive resolved-scene fractions, not a dilution
posterior or an exclusion of unresolved companions. A five-second live
probability sample averaged 8.76 logical CPUs, or 27.4% of the 32 logical
CPUs. Six renderer threads was fastest in the bounded comparison, but
serial sampling and other phases still prevent full hardware saturation.

The interim radius and hardware record is
results/boi-validation/mk-6267058-radius-dilution-progress-20261020.json.
