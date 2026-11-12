# V3 calibration sample: built to look like KBonus

Design record for `calibration-sample`, first built 29 October 2026 at
Adam's request: a calibration sample as close to the real Kepler Bonus
background light curves as the data allow. It comes before detector and
vetting calibration, whose targets are 35% detection and 96% vetting
completeness on real planetary transits (`KeplerX/todo.md`).

**Revision 2 (31 October 2026).** Adam asked for the existing sample to be
made realistic in place, not replaced. Kept from the first build:

- every lane host, case ID and development/holdout partition;
- the known Kepler planets, the validated BOIs and the 799 leaks;
- the inverted controls.

Changed:

- **Preparation.** Every case and the 20,000-curve census use production
  preparation (retained quality bits, retained-outlier clip, field-artifact
  epochs). So does each host's noise ladder.
- **Detectability.** The expected MES carries the detector's measured scale
  of 0.80. "Detectable" means a scaled MES of at least 7.1 inside the search
  range.
- **Dilution.** Every synthetic signal on a host is diluted by the host
  star's share of the curve's flux; a blend by its own share of the whole
  curve.
- **Planets.** Redrawn on the same hosts, pulled toward the edges of the
  0.5–200-day range, with far fewer that have under three transits.
- **Real signals.** Known planets, validated BOIs and leaks are judged on
  their own depth in their own curve.
- **Set sizes.** 701 more leaks (1,500 in all), 300 more binaries (600) and
  400 fewer scrambled controls (400).
- **Artifacts.** 600 real artifact curves were reselected on what the
  production mask leaves, and 2,191 paired copies carry six measured
  artifact classes.
- **Weights.** Every negative carries a lane weight.

The overnight development run of 30 October ran on the first build; its
numbers do not carry over. That build and its re-prepared development copy
are in `~/.Trash/spacelab-calibration-sample-revision1-20261031/` until the Trash
is emptied.

**Revision 3 (built 1 November 2026 from commit fcbcdf0e1).** Adam's
decisions after a first-principles review:

- **Ruler.** Detectability is the information S/N of the delivered signal,
  not a scaled detector MES. The detector's efficiency is measured against
  it, as DR25 measured Kepler's pipeline.
- **Completeness.** C(S/N, P, R, class) is fitted unweighted to every
  searchable planet and then integrated against occurrence, so the
  integrated number has low variance. The efficiency target (50% at S/N 8.5,
  90% at 12) does not depend on occurrence.
- **Dilution.** The star's expected flux takes a Kepler-band colour term
  from Gaia BP − RP.

Same roster, case IDs and partitions as revision 2. 20,779 of 21,615 cases
have arrays identical to revision 2. The other 836 are on hosts redder than
BP − RP 1.75 (590 M, 113 unclassified, 98 K, 35 giants, subgiants and hot),
where the colour term changed the star's share and so the planets drawn.
Revision 2's bulk data went to `~/.Trash/spacelab-calibration-sample-revision2-20261101/`;
its plan, summary, quality, audit and failure records stay in
`build-logs/revision-2-records/` beside the build logs. Before the build,
`KeplerX/results/calibration-audits/2026-11-01-information-snr-of-sample-revision-2/` recovered the
information S/N of every revision-2 planet from the stored packets and
rescored the 31 October development-quick run on it (below).

**Revision 4 (built 1 November 2026 from commit 33eb24915).** Before the
first full study, Adam asked for the sample to be checked against the curves
it sits on (`KeplerX/results/calibration-audits/2026-11-01-pre-run-check-of-sample-revision-3/`).
Every kind of injected signal was measured in its real KBonus curve against
the same host's uninjected curve:

- **Matched their record and their G band:** planets (near the threshold,
  median depth ratio 1.0009, 2 of 299 off by more than 5%), binaries,
  leaks, artifacts and noise.
- **Did not match their curves:** the blends. 150 of 600 sat on curves
  without foreign light, and 184 gave more of the curve's flux than its
  foreign light.

Blends are now offered to the 800 plain lane hosts before the binaries and
built only on curves carrying foreign light (star share below 0.99). Each
takes between 0.1% and the lesser of 30% and that foreign light, from its
own random stream. The result is 600 blends again: 450 redrawn on their
hosts and 150 moved. The other 22,311 cases are identical to revision 3
(light-curve arrays, packets and truth, paths aside). Revision 3's bulk data
went to `~/.Trash/spacelab-calibration-sample-revision3-20261101/`; its
records stay in `build-logs/revision-3-records/`.

**Habitable-zone panel (added 1 November 2026).** Small planets in the
conservative habitable zone of the quietest K and M hosts, added to the
built sample as their own case kind (`hz_planet`).

**Revision 5 (built 2 November 2026 from commit 8b6a170c1).** Adam asked
for the habitable-zone panel to become part of one coherent sample, with
unified generation, truth records and scoring (TODO item 1). The one
`build` now makes it as the habitable-zone stratum, described under
Signals; the separate `hz-panel` command and its record are gone. The
changes:

- each stratum host gets its uninjected original, like every host (216
  new cases);
- panel planets carry population weights on the main sample's footing;
- every synthetic planet records its insolation and whether it lies in a
  conservative zone;
- stratum hosts join the quick sets by the common rule;
- scoring treats all synthetic planets as one set.

All other 22,911 cases are identical to revision 4: light-curve arrays,
and packets byte for byte. Their truth records differ only in the new
fields and in 12 main-sample weights, now zero, of primaries inside a
stratum host's box. Revision 4's bulk data went to
`~/.Trash/spacelab-calibration-sample-revision4-20261102/`; its records
stay in `build-logs/revision-4-records/`, and
`KeplerX/results/calibration-audits/2026-11-02-sample-revision-5-swap/`
holds the swap.

- Code (`KeplerX/calibration/`): `sample_builder.py` (builder),
  `signal_injection.py` (signal models, flattening emulator), `population.py`
  (stellar inputs, physical draws and weights), `lane_census.py` (lane census and
  the sample quality report), `artifacts.py` (artifact models, injection,
  validation), `scoring.py` (run scorer).
- Evidence: `KeplerX/data/calibration-sample-design/`, with the
  census in `lane-noise-census/` and the artifact models in `artifact-models/`.
- Sample: `KeplerX/data/calibration-sample/` (cases,
  truth, batches, `quality.json`, `audit.json`, `artifact-validation.json`).
  Since 3 November 2026 the sample, its design folder and the SAP lane's
  twin (`calibration-sample-sap/`, built by `calibration/lane_twin.py`) sit
  on discSSD in `/Volumes/discSSD/keplerX/data/`, behind symlinks of the
  same names in `KeplerX/data/`; a batch names the disc when it is not
  mounted.

```text
uv run --locked python -m calibration.lane_census census REFERENCE.parquet CENSUS
uv run --locked python -m calibration.artifacts measure CENSUS
uv run --locked python -m calibration.artifacts validate
uv run --locked python -m calibration.sample_builder build OUT --workers 8
uv run --locked python -m calibration.sample_builder audit OUT --workers 8
uv run --locked python -m calibration.lane_census report OUT --reference CENSUS/features.parquet
```

Run from `KeplerX/` with `UV_PROJECT_ENVIRONMENT=runtime/pipeline-environment`.

## Scope

- Lane: delivered PSF Flat (Adam, 29 October), which holds 250,923 of the
  406,547 KBonus background sources; the rest are SAP-only. Every statistic
  below refers to the PSF Flat lane unless it says otherwise.
- The earlier `kepler-diverse-20261028-v1` (10,000 cases, committed in d6e1b502c)
  is kept as a stress grid but is not representative:
  - stellar classes nearly uniform: K stars are 15% of its cases against 58.5%
    of the lane;
  - backgrounds a median 1.35 mag brighter than the lane;
  - a flat log-depth by log-period grid with circular orbits;
  - real hosts that are bright Kepler targets;
  - only 62 catalogue false positives, of two kinds.

## What real lane light curves look like

The census is a simple random sample of 20,000 lane sources
(`reference-20k.parquet`), measured the way the campaign prepares them.
Production preparation retains five quality flags, clips retained outliers
and masks the field-artifact epochs. Under it every census curve has at least
435 usable cadences; under quality 0 alone, 191 had fewer than 100.

| Quantity | 5% | 25% | 50% | 75% | 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaia G (mag) | 15.4 | 17.0 | 17.8 | 18.5 | 18.9 |
| Usable cadences | 439 | 3,514 | 27,007 | 51,402 | 64,015 |
| Quarters | 1 | 1 | 7 | 14 | 18 |
| Duty cycle | 0.24 | 0.50 | 0.85 | 0.90 | 0.92 |
| Masked fraction | 0.01 | 0.02 | 0.06 | 0.54 | 0.98 |
| Point noise (ppm per 29.4 min) | 321 | 890 | 2,007 | 4,101 | 8,953 |
| Noise of 3 h means (ppm) | 183 | 459 | 937 | 1,886 | 4,433 |
| Noise of 12 h means (ppm) | 141 | 321 | 588 | 1,105 | 2,882 |
| Red-noise ratio at 6 h | 0.95 | 1.08 | 1.18 | 1.44 | 3.70 |
| Half-day variability, 5–95% (ppm) | 533 | 1,123 | 1,976 | 3,722 | 11,483 |
| 5-sigma dips per 100 d | 0 | 0 | 0.4 | 8.1 | 113 |
| Raw-flux jumps per 100 d | 0 | 0 | 0.3 | 1.0 | 3.4 |

Against quality 0 alone, the noise ladder moves by 2–4% (the 3-hour noise
falls from 960 to 937 ppm at the median) and the usable cadences rise.

Stellar classes from Gaia Teff and logg:

| Class | Share of lane |
| --- | ---: |
| K | 58.8% |
| G | 14.6% |
| Unknown | 13.4% |
| M | 7.0% |
| Subgiant | 2.4% |
| F | 2.0% |
| Giant | 1.5% |
| Hot | 0.2% |

Each source's relation to its TPF target:

| Relation | Share of lane |
| --- | ---: |
| No KOI or eclipsing-binary target | 95.1% |
| KOI false-positive target | 1.85% |
| Kirk et al. (2016) eclipsing-binary target | 1.68% |
| KOI planet host | 1.37% |

None of the sources is its own TPF's Kepler target.

## Spacecraft and KBonus artifacts

### Where they are

The census stacked on the mission cadence clock in 3-hour bins, epochs left
in, finds field-wide dips. For a 3σ dip the field median is 2.1% of sources
at once.

| Field rate | Bins | Days | Masked by production | Left (days) |
| --- | ---: | ---: | ---: | ---: |
| ≥ 2× median | 1,835 | 225 | 140 | 1,695 (208) |
| ≥ 3× median | 782 | 96 | 140 | 642 (79) |
| ≥ 5× median | 170 | 21 | 140 | 30 (4) |

- **Named events:** at BKJD 246.2, 200.3 and 373.2, 28–31% of all sources dip
  together. These are the three dates behind the 20 instrumental M/K false
  positives of October. BKJD 373 is the Q4 PSF focus transient. The
  production mask covers them.
- **A Q2 comb:** BKJD 132–160 carries an artifact repeating every ~1.71 d,
  hitting about 20% of sources. A detector reads it as a periodic signal
  confined to one quarter.
- **The response is the curve's own:** no single field template exists. A
  curve's regression on the per-cadence field median over the active bins
  ranges from −5.8 to +8.4 (5–95%), dips for some curves and rises for
  others.
- **Momentum dumps** are mission-wide single missing cadences: 694 single
  and 22 double, most 146 cadences (2.98 d) apart. 28% of census curves
  have their dominant 1–40 d variability period at 2.8–3.0 d.
- **Gaps:** 77 mission-wide gaps of at least 12 cadences (monthly
  downlinks, safe modes, quarter ends). The raw stitched flux ramps by about
  ±5% (5–95%) after them and settles over about a day. KBonus's flattening
  removes nearly all of it; what is left sits in the first few cadences
  beside the gap.
- **Ringing:** the flattening rings for 1–2 days on each side of flux
  discontinuities it bridges.

All of these are in the real hosts, so the uninjected cases carry them at
their natural rates. Two further sets stress them.

### Real artifact curves

600 census curves (`control_artifact`) are chosen on what the production
mask leaves. The score is the curve's 3σ dips in the 1,695 active bins the
mask does not cover, beyond its own dip rate in quiet bins. The chosen
curves have a median 368 such dips (22% of those bins) against their own
3.6% elsewhere. Their point noise is low for their magnitude (median
478 ppm against 2,006 ppm at the same median G 17.8): curves carrying much
foreign light, which the focus and pointing events disturb most.

Events shared within one TPF (rolling band crossing several neighbours)
need whole TPF families to find. They are not in the selection.

### Artifact copies

Each of 2,800 lane hosts is assigned one copy of either its original or its
first planet case, with one artifact class added. Nothing else changes, so a
copy's candidates compare directly with its pair's. Each class is measured on
the census (`calibration/artifacts.py`). Where the census holds real
instances, a copy receives another real curve's own instance. That donor is
matched in noise, observed in the same quarters, and its instances are placed
at the same mission times. Where an instance cannot be told from noise one by
one, donors' instances are thinned to the measured excess over a null.

| Class | Model | Census curves holding it |
| --- | --- | ---: |
| `field_excursions` | a donor's ≥ 3σ 3-hour excursions (grown over ≥ 1.5σ neighbours, smoothed over 5 cadences) in the active bins outside Q2, each bin kept with the field's excess share, 1 − median rate / bin rate | 84.5% |
| `q2_comb` | the same inside BKJD 132–160 | 37.2% |
| `momentum_dumps` | a donor's mean residual against cadences since the last dump, per quarter, where its χ² exceeds 4 × both its degrees of freedom and the χ² on false dump times; applied at the host's own dumps | 15.4% |
| `rolling_band` | a donor's flagged runs (bits 131072, 262144; production retains them) of ≥ 3 cadences offset by ≥ 3σ of the curve's own scatter of means that long, thinned to 24%: 4.8% of flagged runs pass against 3.6% of unflagged twins 300 cadences away | 20.4% |
| `gap_ramps` | a donor's hook A exp(−k/κ) fitted to the 24 cadences beside a gap edge, where the mean of the 3 nearest departs by ≥ 6σ, thinned to 78%: 3.0% of edges pass against 0.7% of positions 300 cadences away (≥ 20σ: 0.5% against 0.06%) | 21.2% |
| `sensitivity_drops` | two steps in the raw flux, through the flattening emulator: size from the census's sharp persistent raw steps (5/50/95%: 0.06/0.26/2.6%), recovered fraction 0.3–0.9 and e-folding 0.1–2 d assumed | 61.0% |

Copies that no donor's instances land on (within 50 noise-matched donors)
are not kept: a Q2 comb on a curve never observed in Q2, or a dump profile or
rolling-band run on a short curve. The Q2 comb therefore gets a second block
of 400 hosts. Kept copies, of originals and of planet cases:

| Class | Originals | Planet cases |
| --- | ---: | ---: |
| `field_excursions` | 200 | 200 |
| `q2_comb` | 180 | 186 |
| `momentum_dumps` | 172 | 169 |
| `rolling_band` | 147 | 143 |
| `gap_ramps` | 199 | 195 |
| `sensitivity_drops` | 200 | 200 |

**What the census rules out.**

- *Gap ramps through the emulator:* injecting the raw ramps leaves 2.6
  times the residual KBonus left at the first cadence after a gap (5–95%
  ±10σ against ±3.7σ). The delivered residual is transplanted instead.
- *Rolling band in e⁻/s:* the measured run offsets scale with the curve's
  own noise (|offset| ∝ level^0.52), not with a fixed background. So they
  move between noise-matched curves in relative flux.
- *Clean sensitivity drops:* none could be isolated. Sharp raw drops are
  mostly eclipses (deep, recovered within hours, concentrated in 3% of
  curves); persistent steps are mostly raw trends.

**Validation** (`artifact-validation.json`, 800 census curves re-prepared
from LC2, each class injected):

| Class | Copy against census |
| --- | --- |
| `field_excursions` | 46,384 injected 3σ dip bins against 36,288 expected from the census excess per curve the class hits (×1.28); 48,796 rises |
| `q2_comb` | 4,794 against 7,087 (×0.68), on the 473 of 800 curves observed in Q2 |
| `momentum_dumps` | curves with their dominant period at 2.8–3.0 d: 39% → 48% (donors themselves 57%) |
| `rolling_band` | largest change 2.7σ at the median, 7.2σ at 90%, 26σ at 99% |
| `gap_ramps` | first cadence after a gap, 5–95%: ±3.6σ → ±6.0σ; 1–99%: −12.2/+10.2σ → −21.5/+20.0σ; six cadences on, unchanged |
| `sensitivity_drops` | a step of 1.9σ (median) leaves a −1.2σ residual dip in the Flat; the 10% largest leave −9.8σ |

The field excursions overshoot by about a quarter because donors observed in
all the host's quarters are preferred, and those tend to carry more
excursions. The Q2 comb undershoots by a third.

## The flattening transfer, measured and emulated

KBonus did not mask known transits when flattening. FLATTEN_MASK is set on
0.3% of the in-transit cadences of confirmed planets. Depth surviving in PSF
Flat, relative to the raw stitched PSF (both against the same local linear
baseline per event), for confirmed planets with KBonus light curves and raw
depth above 300 ppm:

| Duration | n | Real KBonus (median, IQR) | Emulator |
| --- | ---: | --- | ---: |
| ≤ 2 h | 48 | 0.997 (0.991–1.005) | 0.999 |
| 2–4 h | 119 | 0.998 | 0.999 |
| 4–6 h | 92 | 0.990 (0.975–1.001) | 0.996 |
| 6–8 h | 200 | 0.989 (0.964–1.003) | 0.993 |
| 8–10 h | 77 | 0.970 (0.953–1.002) | 0.986 |
| 10–13 h | 47 | 0.981 (0.935–0.997) | 0.967 |
| 13–16 h | 14 | 0.876 (0.806–0.912) | 0.956 |
| 16–24 h | 13 | 0.932 (0.841–0.951) | 0.910 |
| > 24 h | 3 | 0.856 | 0.779 |

**How the emulator was chosen.** The emulator was applied to the same 613
planets' real raw PSF, transit included. An unclipped cubic B-spline with
2-day uniform knots per quarter, which is KBonus's documented method, has a
median bias of +0.3%, and −0.6% for transits over 10 hours. The alternatives
fail:

- clipping at 3σ protects deep transits and keeps 1.5–3.7% too much depth
  above 10 hours;
- 1-day knots remove 5–18% too much.

KBonus's exact smoother could not be reconstructed (the emulated Flat differs
from the delivered one by a median 1.8σ per point), so the emulator is used
only differentially.

**How a signal is injected.** For a signal s on the star, first diluted by
the star's share f of the curve's flux (next section), s' = 1 + f · (s − 1),
times any raw artifact a of the whole curve:

    delivered Flat + s'a · E(raw) / E(raw · s'a) − 1 + c

E is the emulated smoother and c an artifact measured in the delivered Flat
itself. The noise, variability, artifacts and the real smoother's own
residuals stay exactly as delivered. The noise keeps its absolute level in
transit, as it does for background-limited faint stars.

**Accepted residual.** The emulator loses about 8% too little depth at
13–16 hours (14 real planets whose interquartile range does not contain the
emulator's value). As built, the sample keeps 0.49 of the depth above
24 hours, against 0.856 for the three real planets there.

## Faint curves carry light that is not their star's

The delivered PSF Flat is in e⁻/s. On the 832 hosts with G < 16 its level
follows Gaia G closely: 2.16 × 10⁵ e⁻/s at G = 12 (clipped fit over 654
hosts, scatter 0.047 dex). Fainter curves hold more flux than their G allows
(5,704 hosts of `calibration-sample`, measured on 30 October):

| G | Hosts | Level ÷ G-expected (25/50/75/90%) |
| --- | ---: | --- |
| 13–15 | 297 | 0.90 / 0.98 / 1.08 / 1.31 |
| 15–16 | 529 | 0.94 / 1.04 / 1.17 / 2.2 |
| 16–17 | 1,003 | 0.96 / 1.10 / 1.38 / 4.1 |
| 17–18 | 1,480 | 1.01 / 1.19 / 2.3 / 10.5 |
| 18–18.5 | 1,051 | 1.06 / 1.49 / 4.9 / 20 |
| 18.5–19.1 | 1,315 | 1.17 / 1.87 / 7.6 / 24 |

The excess is light of the TPF target or of neighbours taken up by the PSF
fit, or background left in. A transit removes part of the star's light
alone, so its depth in the curve is diluted by the star's share of each
quarter's level, f = min(1, expected / level).

**Expected flux is Kepler-band flux (revision 3, 1 November 2026).** Gaia G
runs to 1,050 nm, Kepler's response stops near 900 nm, so a red star
delivers less than its G implies. The builder takes the star's expected
flux from a Kepler-band magnitude on G's zero point:

Kp = G + 0.148 × max(0, BP − RP − 1.75)

- **Measurement:** delivered level over G-expected for 25,689 lane curves
  with a measured level (the 20,000-curve census and the sample's hosts),
  against Gaia DR3 BP − RP (fetched for all 406,547 inventory sources,
  97.5% with a colour). A quantile regression of the lower quartile, with a
  free offset per G bin to absorb the excess light, gives a slope of
  0.148 ± 0.016 mag per magnitude of colour above 1.75 and none below.
  Residuals stay within ±0.05 mag from BP − RP 0.5 to 3.5.
- **Size:** 9% less flux at BP − RP 2.5, 16% at 3.0, 24% at 3.5. The lower
  decile, the median at G < 17.5 and the fit above G 17.5 agree within the
  errors.
- **Independent check:** KIC Kepler magnitudes (MAST `kic10`) of 11,550
  lane sources give Kp − G rising by 0.08, 0.18, 0.30 and 0.43 mag over
  BP − RP 2.25–2.5, 2.5–2.75, 2.75–3.0 and 3.0–3.25: the same sign and
  shape, about 1.5 times steeper. The KIC's own magnitudes of faint red stars
  scatter by 0.6 mag (interquartile range), and the delivered flux is the
  quantity dilution needs, so the builder uses the flux fit.
- **Files:** `calibration-sample-design/gaia_colours.py` (ARI Heidelberg Gaia
  mirror; the ESA archive timed out), `gaia-colours.parquet` and `.json`
  (sha256 `ec7aaa76…`), `kic_kepler_magnitudes.py`, `kic-kepler-magnitudes.parquet` and
  `sample-host-levels.parquet`.

Without the term, M dwarfs were diluted too little, so their planets were
injected 9–24% too deep.

The builder applies f per quarter to every synthetic signal on a host
(planets and binaries) before the flattening emulator. It uses the host's
mean f when it draws planets toward the detection transition, and records f
(`star_light` in the truth record, `star_flux_fraction` on each signal and
case). A blend's dilution is already its share of the whole curve, so f does
not apply to it again. That share is part of the curve's foreign light, so
it never exceeds 1 − f (revision 4).

**Tested on Gaia eclipsing binaries (31 October 2026).** 613 Gaia DR3
eclipsing binaries are lane sources, and 456 have at least 2,000 usable
cadences. In 372 of those the eclipse is present at Gaia's period: the fold's
F statistic exceeds three times the largest of 30 at random frequencies. The
KBonus depth is the max-to-min of the fold of the raw stitched PSF flux at
the refined Gaia period, which is how Gaia's `derived_primary_ecl_depth` is
defined. Divided by Gaia's own depth, it follows the predicted share. The
table covers the 321 binaries whose Gaia depth error is below 20%:

| Share | Used / absent | Median G | Median share | KBonus ÷ Gaia depth | ÷ (1.043 × share) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–0.2 | 61 / 35 | 18.1 | 0.09 | 0.11 | 1.21 |
| 0.2–0.4 | 26 / 7 | 17.7 | 0.28 | 0.31 | 1.01 |
| 0.4–0.6 | 20 / 4 | 18.3 | 0.51 | 0.52 | 0.99 |
| 0.6–0.8 | 40 / 8 | 18.0 | 0.73 | 0.78 | 1.02 |
| 0.8–0.95 | 70 / 9 | 17.0 | 0.89 | 0.95 | 1.03 |
| ≥ 0.95 | 104 / 21 | 16.9 | 1.00 | 1.04 | 1.01 |

- **Fit:** depth scales as share^0.95 (Theil–Sen, 95% interval 0.91–0.99,
  where 1 is the model and 0 is no dilution). It holds at G 17–18 and 18–19
  alike, and the delivered Flat gives the same ratios.
- **Below share 0.2:** the measured depth is about 20% above the model.
  Absent eclipses also rise there, from 15% at share ≥ 0.95 to 32%, so part
  of that 20% is selection.
- **Files:** data, script and summary are in
  `data/calibration-sample-design/dilution-test-on-gaia-binaries/`. Gaia was queried
  by HEALPix source-ID ranges; the CSV's sha256 is `c389d31a…`.

The star's share across the built sample's hosts (10/25/50%):

| G | Share |
| --- | --- |
| < 16 | 0.57 / 0.88 / 0.98 |
| 16–17 | 0.24 / 0.73 / 0.91 |
| 17–18 | 0.10 / 0.43 / 0.84 |
| 18–18.5 | 0.05 / 0.21 / 0.67 |
| > 18.5 | 0.04 / 0.13 / 0.54 |

## Contamination by the TPF target is common and real

The target's ephemeris was folded on every lane source in its TPF:

| Target | Leaks at S/N > 7.1 | Leak depth (10/50/90%, ppm) |
| --- | ---: | --- |
| Kirk et al. (2016) eclipsing binary | 50.8% of 4,034 | 790 / 13,640 / 138,760 |
| KOI false positive | 37.4% of 6,125 pairs | 300 / 6,191 / 95,729 |
| Confirmed planet | 13.6% of 2,562 | 131 / 472 / 3,575 |
| KOI candidate | 8.3% of 1,925 | 53 / 486 / 6,675 |

A diluted planet leak (median 470 ppm) is indistinguishable from a planet on
the background star in its light curve; only localization can reject it.

The sample carries 1,500 real leakage cases, stratified in depth, with the
target's ephemeris as truth and false positive as the expected outcome:

| Label | First build | Added | Lane curves leaking (each once) |
| --- | ---: | ---: | ---: |
| Eclipsing binary | 450 | 250 | 2,050 |
| KOI false positive | 200 | 400 | — |
| KOI planet | 150 | 50 | — |

The KOI false-positive and KOI planet leaks that are not also binary leaks
together make 1,604 lane curves; in all, 3,654 lane curves leak above
S/N 7.1. The additions are drawn apart from the first 799, which stay
unchanged. Each leak is judged on its own depth in its curve. 556 of 700
binary leaks, 553 of 600 KOI false-positive leaks and 157 of 200 planet
leaks are detectable that way.

## Signals

### Hosts

**Selection.** Hosts are drawn at random from the lane in proportion to it,
with floors for the rare classes. The floors are M 500, subgiant 250, F 250,
giant 250 and hot 120. Host weights (lane count over sample count per class)
undo them. The partition into development and holdout is by TPF family, so
neighbours sharing systematics and leakage never straddle the split. It is
keyed on the first build's seed, so no family moves between builds.

**Stellar parameters.** The campaign's declared stellar inputs follow
`population.catalogue_stellar`:

- Gaia Teff and logg;
- the Mamajek dwarf mass at that Teff;
- the radius from mass and gravity;
- a solar surrogate when inputs are missing.

The injected truth scatters around those inputs as GSP-Phot errors do
(Andrae et al. 2023):

| | G 15 | G 19 |
| --- | ---: | ---: |
| Teff | 3% | 11% |
| logg | 0.12 dex | 0.32 dex |

Evolved stars scatter 30% more. Missing inputs scatter 600 K and 0.4 dex.
Vetting therefore sees a duration-density mismatch as large as the real one.

**Limb darkening** is Detector's own nonlinear Kepler grid at the true Teff
and logg. Vetting evaluates the grid at the catalogue values, so the profile
is wrong by exactly the stellar-parameter error. Scaling the four
coefficients instead would break the balance of the law: 81% of such
profiles came out unphysical.

### Planets

**Occurrence.** SAG13 (Kopparapu et al. 2018, period in years), with three
adjustments:

- below a break that moves out with radius (3, 7 and 10 days for small,
  sub-Neptune and large planets) the rate falls as P^1.5;
- a hot-Jupiter pile-up is added;
- cool dwarfs get twice the small planets and a third of the giants.

Integrated rates for FGK hosts, against the literature:

| Population | Model | Literature |
| --- | ---: | ---: |
| Ultra-short periods (0.25–1 d, 0.7–2 R⊕) | 0.75% | 0.5% (Sanchis-Ojeda et al. 2014) |
| Hot Jupiters | 0.66% | 0.4–0.6% |
| 1–4 R⊕ inside 50 d | 0.44 per star | about 0.5 |

**Orbits:**

- eccentricity from Van Eylen et al. (2019): a half-normal with σ 0.32 for
  singles and a Rayleigh with σ 0.083 for multis;
- tides circularise close orbits: circular below 2 days, the full
  distribution from 10 days;
- periastron at least three stellar radii;
- impact parameter uniform over [0, 1 + k], so grazing chords are included.

**Multi-planet systems and TTVs:**

- 27% of hosts get two or three transiting planets drawn from a ratio
  distribution like Kepler's (lognormal about 1.9), with radii within
  0.15 dex and 1.5° mutual inclinations;
- companions that do not transit are dropped, as nature drops them;
- near-first-order resonant pairs get sinusoidal TTVs of 1 minute to
  1.5 hours (Holczer et al. 2016), as do 3% of singles (an unseen
  perturber).

**The light curve model:**

- batman with nonlinear limb darkening;
- eccentric orbits;
- 29.4-minute exposure integration, supersampled to resolve ingress;
- a fixed-step fallback (below 1 ppm error) when batman's step search fails
  near the limb.

**Information S/N (revision 3).** The ruler is what the photons allow, not
what the detector does. For each transit event the builder takes the
delivered signal (diluted, after the flattening emulator, noise-free) on the
usable in-transit cadences, n of them. The event's information is
mean(loss²) / σ_q(n)², where σ_q(n) is the noise of a mean of n cadences in
the event's own quarter: the robust spread of adjacent-bin differences on
the host's uninjected curve, or the whole curve's ladder when a quarter
holds fewer than 20 full bins. Events add in quadrature, partial ones
included.

- In white noise this is the matched filter, Σ loss² / σ².
- For a box it is depth / σ(duration) per event.
- Real signals (leaks, known planets, validated BOIs) have no model, so they
  take a box at their measured central depth, which overstates them slightly.

A signal is detectable when it lies in the search range (0.5–200 d, at least
three observed transits) with information S/N ≥ 7.1. The detector's own
losses (MES estimator, whitening, template grid, cadence phase, vetoes) are
then measured against it as an efficiency curve, as DR25 measured Kepler's
pipeline against expected MES (Christiansen 2020). They no longer enter the
definition. Revision 2 used 0.80 × the ladder MES (central depth over the
whole-curve noise at the duration × √N), a detector measurement that the
development-quick run then put at 0.72.

The proposals are still steered with 0.80 × the ladder MES (below). That
only shapes where draws land, and keeping it means revision 3 draws the same
planets as revision 2 on every host whose star share did not change.

**Selection by importance resampling.** For each host, 20,000 proposals
(P 0.25–750 d and R 0.5–25 R⊕, both log-uniform) are weighted by
occurrence × transit probability × a boost g. g has three parts:

- it favours the detection transition, expected MES 3–40, from the host's
  production noise ladder, the star's share of the flux and the 0.80 scale;
- it moves draws toward the inner sides of the range's edges: periods below
  0.5 d × 0.2, beyond 200 d × 0.12, 100–200 d × 0.9 against 1–100 d; draws
  in the range with fewer than three expected transits × 0.06;
- inside the range, with enough transits and at MES ≥ 3, it favours long
  durations (×3 above 6 h, ×10 above 10 h), long periods, eccentric orbits
  and grazing chords.

One proposal is resampled. Its weight, host weight × Z_g / g, makes any
metric reportable for the occurrence-weighted lane population as well as per
stratum. A host with two independent planet cases shares its weight between
them.

**Injection and normalisation.** Each injected case is normalised per
quarter by the median of its usable cadences, exactly as the campaign
normalises delivered Flat. This removes the level shift that a deep,
frequent signal leaves after the smoother.

### Real planets

- **Known planets:** KBonus light curves of 999 confirmed Kepler planet
  hosts (1,499 planets); hard systems favoured; KOI candidates listed as
  known.
- **Validated BOIs:** the 11 internally validated planets on 7 lane sources,
  with ephemerides re-measured on PSF Flat.

Each real planet is judged on its own depth in its KBonus curve: the
baseline around its catalogue ephemeris less the mean of its transit centre.
That depth is a median 0.92 of the catalogue depth; the expected MES is then
computed as for injected planets. 1,375 of the 1,460 known planets in the
search range are detectable that way, as are all 11 validated BOI planets.

### Negatives

| Kind | Source | Cases |
| --- | --- | ---: |
| `control_original` | every lane host, uninjected | 5,709 |
| `contamination` | real TPF-target leakage (700 EB, 600 KOI false positive, 200 KOI planet; each source once) | 1,500 |
| `control_inverted` | DR25-style inversion (flares become dips) | 800 |
| `control_artifact` | real curves most affected by the field artifacts the production mask leaves | 600 |
| `synthetic_blend` | blended binary on a curve carrying foreign light, contributing from 0.1% to the lesser of 30% and that foreign light: planet-like depths, V shapes | 600 |
| `synthetic_binary` | eclipsing binary on the host, diluted by the host's share: primary, eccentric secondary, ellipsoidal; Kirk-like periods | 600 |
| `control_scrambled` | ~10-day blocks reordered within each quarter | 400 |
| `artifact_*` | paired artifact copies (above), half of originals and half of planet cases | 2,191 |

- **Known signals:** lane hosts that carry survey or M/K BOIs list them as
  `known_signals`, so a real signal on an uninjected host is never scored as
  a false alarm.

**Lane weights.** Each negative carries `lane_weight`, the number of the
lane's 250,923 curves it stands for, so a false-alarm rate can be stated for
the survey:

- originals: their host weight (summing to the lane);
- leaks: the lane's leaking curves of their label over the cases drawn;
- binaries: Kirk et al.'s 1.39% binary fraction of the lane (3,488 curves);
- blends: the same fraction of the 73.6% of curves carrying foreign light
  (star share below 0.99; 2,567 curves);
- real artifact curves: the lane over the census they top (12.5 each);
- null artifact copies: their class's census share of the lane.

Planet cases, planet-paired copies, transformed controls and real planets
carry none. The scorer (`scoring.py`) reports lane-weighted false-alarm
rates beside the plain ones.

### Habitable-zone stratum

Small rocky planets in the habitable zones of K and M dwarfs are a main
target of the study. The occurrence draws hold almost none above threshold
(Astra's starting audit, 1 November 2026: of 30 small planets in a K or M
habitable zone, none reached S/N 7.1), so the zone has its own stratum. It
was added to the built sample as a separate panel on 1 November; since
revision 5 (2 November 2026) the one build makes it, every planet in it is
weighted as the main sample's are, and scoring treats all synthetic planets
as one set.

**Where the information is.** An Earth at the middle of the conservative
habitable zone (catalogue star, central transit, the curve's noise ladder,
star share and expected transits), on the 20,000-curve census:

| Class | Census curves | Mid-zone period (median) | Earth S/N ≥ 1 | ≥ 2 | 1 R⊕ ≥ 7.1 | 1.5 R⊕ ≥ 7.1 | 2 R⊕ ≥ 7.1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G | 2,922 | 331 d | — | — | 0 | 0 | 3 |
| K | 11,780 | 180 d | 115 | 14 | 0 | 3 | 29 |
| M | 1,404 | 84 d | 111 | 30 | 2 | 14 | 36 |

- **G dwarfs.** The whole habitable zone lies beyond the 200-day search
  range, and an Earth stays below S/N 2.3 on every census curve.
- **K dwarfs.** For 82% of them the zone's outer edge is beyond 200 days.
- **Scale.** The census is 8% of the lane, so about 2,800 lane curves give
  a mid-zone Earth S/N ≥ 1 and about 550 give S/N ≥ 2.

**Design** (`PLAN["hz_panel"]`, `sample_builder.hz_jobs` and `make_hz_planet`):

- **Stratum.** K and M curves where a mid-zone Earth reaches S/N ≥ 1 (so a
  2 R⊕ planet reaches 4) and the zone's middle is inside 200 days
  (`in_hz_stratum`). The census is a simple random sample of the lane, so
  the stratum holds 1,408 M and 1,433 K lane curves: each class's census
  share in it times the lane's curves of the class.
- **Hosts.** Every census curve in the stratum that hosts nothing else in
  the sample: 108 M and 108 K. Each keeps its family's partition, its
  selection rank and pool, and, like every host, its uninjected original.
  79 main-sample hosts (51 M, 28 K) also fall in the stratum, measured on
  their own curves with the same function (`host_hz`).
- **Planets.** Six independent cases per stratum host. Radius is
  log-uniform in 0.5–2 R⊕ and the semi-major axis log-uniform between the
  conservative edges: runaway and maximum greenhouse, Kopparapu et al.
  (2014), one Earth mass, flux at the semi-major axis, true stellar
  parameters.
- **The 200-day edge.** Where a zone runs past the search range (most K
  dwarfs), only 5% of primaries are drawn beyond it, as structural controls.
  The first panel drew across the whole zone and put 17% of its planets
  past 200 days (Adam, 1 November: a bad design; it was rebuilt).
- **Multis.** 25% of systems get one companion and 10% two, drawn as in the
  main sample; TTVs as in the main sample.
- **Weights, on the main sample's footing.** A stratum host stands for its
  class's stratum curves over the class's stratum hosts (13.0 M, 13.3 K
  lane curves). Each of its cases weighs that over its six cases times
  the box rate the primary stands for: SAG13 occurrence × transit
  probability over the proposal's density in ln R and ln P, two-part where
  the zone is split at 200 days (`selection.box_rate`). Within the stratum
  the panel stands for that box. So a main-sample primary drawn inside a
  stratum host's box carries no weight (12 do), and every other
  main-sample planet keeps its weight. Stratum originals, from the panel
  and the main sample, share the stratum's lane curves of their class
  (`lane_weight`).
- **Every synthetic planet** records its insolation (`insolation_earth`)
  and whether it lies in its true star's conservative zone
  (`in_conservative_hz`).

**As built** (revision 5, `summary.json` → `habitable_zone`):

- 1,296 planet cases and 216 originals on the 216 stratum hosts; the
  planet cases split 720 development and 576 holdout.
- 1,343 planets drawn in a conservative zone, 72 of them beyond 200 days;
  57 detectable, 43 on M dwarfs and 14 on K dwarfs.
- The occurrence draws put 375 planets in a conservative zone, 142 of them
  detectable, almost all above 2 R⊕ (105 on K, 17 on M and 9 on G
  dwarfs). Four detectable ones are 2 R⊕ or smaller: three on M dwarfs
  and one on a star without a class.
- **Manifests.** Stratum hosts join `-quick` by the same rule as every
  host (20 development hosts, 140 cases); there is no separate manifest.
- **Scoring.** One set of synthetic planets feeds the efficiency curve, the
  completeness model and every breakdown. The planets in a conservative
  zone, from either draw, are also reported as a slice (`habitable_zone`):
  structural losses, detection of the detectable ones by class, radius,
  role and draw, efficiency by S/N beside the completeness model's
  prediction, the occurrence-integrated completeness of the panel's box
  (0.5–2 R⊕ in the conservative zone) over every lane K and M star, and
  what each vetting policy keeps of their exact detections.

## Composition and quality, as built

Revision 5 was built on 2 November 2026 from commit 8b6a170c1 (34
minutes on 8 workers, the machine shared): revision 4 with the
habitable-zone stratum built in. `summary.json`, `quality.json`,
`audit.json` and `artifact-validation.json` in the sample folder are the
record; `KeplerX/calibration/registered-sample.json` holds their receipts.
In all, 23,127 cases and 30.1 GB, split 11,667 development and 11,460
holdout. One known-planet host failed: its KBonus file has no finite PSF
Flat flux. The numbers below are revision 4's main sample unless marked;
revision 5 holds the same cases. With lane weights, under which the 295
stratum originals share the stratum, the hosts still match the lane:
worst |SMD| 0.007, worst KS 0.014, class shares within 0.1%. Revision 5's
synthetic planets are 11,498 (the zone's 1,453 included): 10,744 in the
search box and 3,910 detectable.

**Hosts match the lane.**

- Every one of 16 light-curve features matches the production census within
  sampling noise. Weighted, the worst |SMD| is 0.007 and the worst KS distance
  0.014 (3-hour noise).
- The weighted class shares equal the lane's within 0.4%.
- Absolute host counts: 500 M dwarfs, 250 giants, 250 subgiants, 1,315 with
  G > 18.5, 2,589 with fewer than six quarters, 577 in the lane's top decile
  of variability and 613 in its top decile of flux jumps.

**Planets.** 10,045 planets in 8,209 planet cases; 9,414 inside the search
range with at least three transits; 3,841 detectable (information S/N ≥ 7.1;
revision 2 had 3,405 on 0.80 × the ladder MES).

| Period (d) | < 0.5 | 0.5–1 | 1–10 | 10–50 | 50–100 | 100–200 | > 200 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First build | 429 | 906 | 4,095 | 2,688 | 1,341 | 259 | 259 |
| Revision 2 | 129 | 1,299 | 4,392 | 2,174 | 1,108 | 809 | 134 |
| Revision 3 | 138 | 1,323 | 4,378 | 2,170 | 1,088 | 819 | 129 |

Planets in the range with fewer than three observed transits: 1,198 in the
first build, 371 in revision 2, 364 in revision 3.

| Regime (revision 3) | Searchable | Detectable |
| --- | ---: | ---: |
| S/N 7.1–10 | 1,012 | 1,012 |
| S/N 10–15 | 932 | 932 |
| More than 100 transits | 2,749 | 1,166 |
| P > 50 d, at least three transits | 1,799 | 817 |
| Ultra-short period (P < 1 d) | 1,323 | 669 |
| Duration > 10 h | 1,008 | 433 |
| Duration > 16 h | 368 | 115 |
| e > 0.3 | 1,361 | 573 |
| Grazing | 1,131 | 562 |
| TTVs | 1,518 | 639 |
| Multi-planet systems | 3,194 | 1,383 |
| Depth < 200 ppm | 2,372 | 170 |
| Flattening loss > 5% | 509 | 173 |

Detectable planets by radius: 234 below 2 R⊕, 953 at 2–4, 665 at 4–8, 1,166
at 8–15 and 798 above 15. Host noise sets most of this. Lane hosts have a
median 3-h noise of 955 ppm, against 132 ppm on the Kepler-target hosts. The
smallest detectable radius (lane-weighted median, central transit, revision
2's ruler) built up as follows:

| | At 10 d | At 50 d |
| --- | ---: | ---: |
| Neither dilution nor the 0.80 scale | 2.2 R⊕ | 2.7 R⊕ |
| With the scale | 2.4 R⊕ | 3.1 R⊕ |
| With the scale and dilution | 3.5 R⊕ | 4.5 R⊕ |

The class floors add giant, subgiant, hot and F hosts, where only large
planets are detectable. Weighted by occurrence, 53% of the detectable planets
are below 4 R⊕.

**S/N spread of the searchable planets** (information S/N, revision 3):

| S/N | Planets |
| --- | ---: |
| < 1 | 1,451 |
| 1–3 | 1,165 |
| 3–5 | 1,676 |
| 5–7.1 | 1,281 |
| 7.1–10 | 1,012 |
| 10–15 | 932 |
| 15–20 | 513 |
| 20–40 | 882 |
| 40–100 | 363 |
| > 100 | 139 |

Weighted by occurrence, 7.8% of searchable transiting planets around lane
stars are detectable (7.0% in revision 2 on the scaled MES, 18.6% in the
first build, before dilution and the scale).

**What the photons allow, per set** (revision 3). The template match is the
detector's limit, not part of detectability; where no template matches, the
best keeps √(template/duration) of the S/N:

| Set | Planets | Outside range | < 3 transits | Below S/N | Detectable | Template matched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 10,045 | 267 | 364 | 5,573 | 3,841 (38.2%) | 2,988 |
| Development | 5,072 | 147 | 189 | 2,792 | 1,944 (38.3%) | 1,486 |
| Development quick | 813 | 27 | 29 | 449 | 308 (37.9%) | 242 |
| Holdout quick | 748 | 18 | 26 | 413 | 291 (38.9%) | 229 |
| Holdout | 4,973 | 120 | 175 | 2,781 | 1,897 (38.1%) | 1,502 |

Of the detectable planets, 439 are longer than the duration window and 411
shorter than the shortest template. Known planets detectable on their KBonus
depth (box S/N): 669 of 711 in development, 747 of 788 in holdout.

**Realised flattening transfer.** With the local-baseline estimator, the
sample keeps the following depth, against the real table: 0.999 at 2–4 h,
0.993 at 6–8 h, 0.985 at 8–10 h, 0.972 at 10–13 h, 0.946 at 13–16 h, 0.871
at 16–24 h and 0.49 above 24 h.

**Detectable planets against Kepler planets** (weighted medians; KOI planets
with P ≤ 200 d and at least three transits):

| Quantity | Sample | Kepler |
| --- | ---: | ---: |
| Radius (R⊕) | 3.3 | 1.94 |
| Period (d) | 4.7 | 10.6 |
| Duration (h) | 2.2 | 3.4 |
| Transits | 144 | 112 |
| Depth (ppm) | 1,290 | 342 |

The fainter, diluted and denser lane hosts need deeper, more frequent
transits to reach the same S/N.

**Power.** The binomial half-width is ±1.5% on 35% detection of the
detectable planets. If 35% are detected, about 1,340 planets measure the 96%
vetting retention to ±1.0%.

**Audit.** 727 cases from every lane kind, artifact copies and blends
included, were rebuilt from their truth record, the LC2 corpus, the artifact
models and the code alone. Every array came out identical (maximum flux
difference 0.0), in revision 2, in revision 3, in revision 4 (767 cases
with the panel) and in revision 5.

**Integrity.** No TPF family is in both partitions.

**Running it: one study.** The calibration runs as one study
(`calibration/study.py`): every case of the registered sample in one batch,
with one code commit and one config, then one score.

```text
uv run --locked python -m calibration.study run NAME
```

- **Layout:** the study lives in `KeplerX/results/calibration-studies/NAME/`:
  `study.json`, `config.json`, `manifest.jsonl`, `attempts/` and `score/`.
- **Checks before running:** it refuses uncommitted code. On every start it
  verifies each sample record against its registered receipt.
- **Interruptions:** a batch output is never reused, so running it again
  starts a new attempt with the cases no attempt completed. It refuses if the
  sample, config, code or partitions changed.
- **Scoring:** once every case is complete, the scorer takes all attempts
  together. It reports development and holdout separately, each with its
  efficiency, completeness and habitable-zone sections.
  `study score NAME` rescores into `score-2`, `score-3` and so on.
- **Partitions and workers:** by default the development half on all 10 of
  the Mac's cores. The holdout stays unrun until acceptance (Adam,
  1 November 2026), when `--partitions development holdout` runs both.

At the rate of the final Detector of 2 November (1,752 quick cases in 50
minutes on 8 workers, the machine shared):

| Batch (revision 5) | Cases | Time on 8 workers |
| --- | ---: | ---: |
| Whole sample | 23,127 | about 11 h |
| Development (`development-all`) | 11,667 | about 5.6 h |
| `development-quick` | 1,892 | about 54 min |

The quick sets keep whole hosts together with their paired originals and
copies, and always include the validated BOIs. The manifests are in
`batches/`. A run writes about 32 MB of outputs per 1,000 cases.

## Detector efficiency against the information S/N (revision 3)

`2026-11-01-development-quick-run`: the 1,750 development-quick cases, production
settings, 8 workers, 62 minutes (1 November 2026), scored with 200
TPF-family bootstraps. Its null controls give exactly the 31 October run's
numbers: 21.0% of uninjected curves with a candidate, 0.72 candidates per
case, the same MES quantiles. So the detector commits between the two runs
changed speed, not output, and the differences below come from the sample.

| Information S/N | Planets | Exact |
| --- | ---: | ---: |
| < 3 | 200 | 0% |
| 3–5 | 137 | 3% |
| 5–6 | 60 | 3% |
| 6–7.1 | 52 | 12% |
| 7.1–8.5 | 43 | 28% |
| 8.5–10 | 33 | 61% |
| 10–12 | 33 | 82% |
| 12–15 | 42 | 95% |
| 15–20 | 48 | 100% |
| 20–50 | 76 | 99% |
| ≥ 50 | 33 | 97% |

**Gamma fit:** 50% at S/N 8.76 (8.34–9.23), 90% at 12.37 (11.22–13.23).
That is 45.7% at 8.5 and 87.9% at 12, just short of both targets, within
the errors.

**Completeness and weighting:**

- 82.5% exact on the 308 detectable planets.
- Occurrence-integrated from C(S/N, P, R, class): 81.0% (75.1–86.7%).
- The plain weighted share is 81.5% on an effective 77 planets.
- 7.1% (5.2–10.7%) of all searchable transiting planets around lane stars
  are found.

**Strata:**

- Durations over 16 hours: 7 of 13 found.
- Giants: 8 of 16.
- Template-mismatched planets: 82%, matched 83%.
- Known Kepler planets: 93% of the 86 detectable on their KBonus depth.
- Validated BOIs: 7 of 7.
- Detector MES over information S/N: median 0.81, 0.70 at P < 1 day and
  0.74 under 1 hour.

**Habitable-zone panel, stopped run (preview).** The development half of the
first panel draw (`2026-11-01-habitable-zone-panel-stopped-run`) was stopped on Adam's instruction at
218 of 720 cases, so that the panel runs inside the full study. That draw
has since been replaced (the 200-day edge, above). The partial run was
scored with the quick run in `2026-11-01-habitable-zone-panel-stopped-run/score-with-quick-run`.

- **Selection bias.** The dispatcher runs the longest curves first, so the
  218 cases lean to long-baseline M-dwarf hosts and are not a random half.
- **Planets.** 256 in all: 38 beyond 200 days, 20 detectable.
- **Detection.** 19 of the 20 detectable planets were found exactly,
  including a 0.94 R⊕ planet at 64 days (S/N 8.8, detected at MES 7.13) and
  a 1.17 R⊕ planet at 45 days. The miss was a 1.16 R⊕ planet at 44 days
  (S/N 8.4, 31 transits of 1.5 hours). Below S/N 7.1 none of 198 was found.
- **Against the main sample's model.** The model predicts less than the
  panel delivered at S/N 7.1–15: 36% and 77%, against 4 of 5 and 8 of 8
  observed.
- **Vetting** computed metrics for all 19 detections with no metric errors.
  It has no decision policy yet, so nothing was dismissed or kept.
- **Null rate.** 23 of the 218 panel curves (11%) carry an unexplained
  candidate, against 21% of the main sample's uninjected curves.

## Detector efficiency against the information S/N (revision 2, rescored)

The information S/N of every revision-2 planet was recovered from the
stored packets: the planet case's native flux less its host original's, over
the original's quarter levels. A multi-planet change is shared between the
planets by their model losses. The 31 October development-quick run
(`2026-10-31-development-quick-run`: 816 synthetic planets, 761 searchable) was then
rescored with the revision-3 scorer functions. Script and outputs are in
`KeplerX/results/calibration-audits/2026-11-01-information-snr-of-sample-revision-2/` (`preview.json`).

**The ruler.** Information S/N is a median 0.86 of the old ladder MES
(0.84–0.92 by duration). Central depth exceeds the mean depth, and the
matched filter regains a little through the shape. Across the sample, 3,763
planets are detectable on it (1,893 development, 1,870 holdout), against
3,405 on 0.80 × the ladder.

**Efficiency, exact detections, unweighted:**

| Information S/N | Planets | Exact | 95% interval |
| --- | ---: | ---: | --- |
| < 3 | 218 | 0% | 0–2% |
| 3–5 | 137 | 3% | 1–7% |
| 5–6 | 54 | 4% | 1–13% |
| 6–7.1 | 46 | 13% | 6–26% |
| 7.1–8.5 | 40 | 30% | 18–45% |
| 8.5–10 | 35 | 69% | 52–81% |
| 10–12 | 35 | 80% | 64–90% |
| 12–15 | 42 | 98% | 88–100% |
| 15–20 | 45 | 100% | 92–100% |
| 20–50 | 79 | 99% | 93–100% |
| ≥ 50 | 30 | 97% | 83–99% |

Gamma CDF fit (c = 0.987, a = 13.7, b = 0.635):

- 50% at S/N 8.55 (case bootstrap 8.15–9.06);
- 90% at 12.04 (11.13–13.16);
- 49.1% at 8.5 and 89.8% at 12.

The detector sits on both targets, within the errors.

**Completeness and occurrence.** The model is
C = c · σ(α + β ln S/N + γ_P ln P + γ_R ln R + δ_class), with c = 0.987 and
β = 6.94. It is close to period-free at fixed S/N (γ_P = −0.01) and
slightly favours large planets (γ_R = 0.20). Giants (δ = −2.0) and hot stars
(−3.2) are found less often than their S/N says.

Its predicted and observed means agree within the counting noise in every
S/N, period, radius, class and template-match bin.

| Occurrence-integrated | Value | Interval |
| --- | --- | --- |
| Detectable planets, from C | 79.8% | 71.9–85.9% |
| All searchable transiting planets around lane stars, from C | 6.1% | 4.5–8.0% |
| Detectable planets, plain weighted share | 78.1% | (Kish effective size 93 of 306) |

**Detector MES over information S/N** for exact detections at S/N ≥ 12.5:

- median 0.83, interquartile range 0.73–0.91;
- 0.69 at P < 1 day, 0.75 under 1 hour of duration, 0.80 at 1–2 hours, and
  0.92 at 10–50 days.

This is where the detector loses S/N against the photons. The mismatched
templates' lower rate (30% against 37%) is explained by their S/N mix; the
model predicts 31% for them without a template term.

**Colour term on revision 2's hosts.** 593 of 7,962 planet cases would get
a star share lower by more than 1% (436 of them on M hosts). The median M
host keeps 0.973 of its share, the lowest tenth 0.90 or less.

## Limits

- **Pixels:** injections are made after extraction. Pixel-level effects on
  injected signals (PSF-fit dilution, centroid shifts) are not modelled, so
  localization is calibrated only on the real leakage and real planet cases.
- **Raw-to-catalogue depth:** KBonus depths of confirmed planets are a median
  0.92 of the catalogue depths. This is taken to be the estimator's geometry,
  not dilution; these bright targets carry no excess flux.
- **Dilution estimate:** f takes all flux above the expected level as
  foreign, and Gaia eclipsing binaries on lane stars confirm it (depth
  scales as share^0.95). Within the bright scatter (0.047 dex) it dilutes
  clean hosts by about 4% on average. Revision 2 diluted the reddest stars
  (BP − RP > 2.5) 9–24% too little; revision 3 takes the Kepler-band colour
  term (above). The term is fitted to 350 curves redder than 2.5 and 30
  redder than 3.5.
- **Known planets don't follow the model:** 28 with a predicted share below
  0.85 (median 0.75, G 12–16) keep 0.94 of the catalogue depth, while
  undiluted ones keep 0.92. Their catalogue depths come through the same 4″
  pixels, so light Kepler didn't know about would dilute both. The five Gaia
  binaries at G < 16.5 with a share below 0.6 do follow the model: they keep
  0.24–0.69 of Gaia's depth, close to their shares.
- **Information S/N:** per-quarter noise of means comes from one quarter of
  one curve, so each σ_q is good to about 5%; events from many quarters
  average it. The noise is measured on the host's uninjected curve, which for
  a real signal includes that signal (the robust spread resists it). Signal
  wings the flattening leaves outside the transit are not counted.
- **Occurrence:** SAG13 is extrapolated to giants, M dwarfs and periods
  beyond Kepler's reach. The occurrence-integrated completeness depends on
  it; the efficiency curve and the fitted C(S/N, P, R, class) do not.
- **Real planets** are on bright Kepler targets, except the 11 validated BOIs.
- **Artifacts:** sensitivity-drop recovery is assumed; rolling band shared
  within one TPF is not modelled; field excursions are about a quarter
  stronger, and the Q2 comb a third weaker, than the census excess per hit
  curve.
- **Unresolved binary companions:** these would dilute about 30% of hosts by
  roughly 10% (mean depth loss about 4%). They are not modelled; the
  stellar-radius errors already modelled are several times larger.
- **Blend and binary rates:** Kirk et al.'s fraction among Kepler targets is
  taken for the background lane.
- **Planet occultations** of hot Jupiters (tens of ppm) are not modelled.
- **Emulator:** about 6% too little depth lost at 13–16 h; about twice the
  real loss above 24 h (three real planets).

## Decisions for Adam

- **Targets (decided 1 November 2026).** Detectability is the information
  S/N ≥ 7.1 in the search box. The detector's exact-detection efficiency
  against it should reach 50% at S/N 8.5 and 90% at S/N 12. The 35% target
  is the occurrence-integrated completeness of the detectable planets,
  computed from C(S/N, P, R, class) fitted unweighted to every searchable
  planet. Both are in `KeplerX/todo.md`.
- **At least three transits (decided 1 November 2026, following DR25).**
  Two-transit and single events (100–750 d) stay controls here. They carry
  real information, and two-transit planets have been validated
  (Kepler-421 b, Kepler-167 e, Kepler-1654 b; TESS duotransits). Those were
  large, high-S/N planets whose aliases the coverage ruled out, so they need
  their own search and vetting.
