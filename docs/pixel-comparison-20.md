# Twenty-source pixel comparison

Test whether the existing target-protected pixel correction improves transit
recovery over delivered KBonus SAP, PSF and PSF Flat. Use 20 new background
stars, aiming for 10 photometric M-dwarf and 10 photometric K-dwarf candidates.
This is an independent development replication of the five-source experiment,
not a survey completeness estimate or an adoption decision.

## Result: no demonstrated improvement

All twenty sources completed on October 14, 2026: ten photometric M and ten
K candidates, one 89–98-day quarter each, twenty distinct parent cutouts.
The original five sources were excluded. The frozen two-worker run took
22.7 minutes. No scientific run failed or was rerun.

The pixel correction failed the predeclared recovery-superiority criterion.
Counts below are incremental recoveries, with 120 nominal probes per row.

| Depth (ppm) | SAP | Pixel correction | Mission vectors only | PSF | PSF Flat |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40 | 0 | 0 | 0 | 0 | 0 |
| 100 | 0 | 0 | 0 | 0 | 3 |
| 300 | 0 | 0 | 2 | 0 | 6 |
| 1,000 | 12 | 9 | 15 | 15 | 38 |
| 3,000 | 56 | 52 | 53 | 78 | 93 |
| 10,000 | 99 | 98 | 93 | 113 | 114 |

Across the primary 480 faint probes, pixel correction recovered 9 versus
12 for SAP, 17 for mission vectors alone, 15 for PSF and 47 for PSF Flat.
Pixel minus SAP was −0.625 percentage points, with a 95% source-bootstrap
interval of [−2.08, +0.42]. Pixel minus PSF Flat was −7.92 points
[−15.0, −2.5]. In paired trials, pixel correction alone recovered two
signals that Flat missed; Flat alone recovered forty that pixel correction
missed. These are conditional comparisons under the injection policies below.

On the selected M sources, faint counts were 0/1/1/2/30 for
SAP/pixels/mission/PSF/Flat, respectively; on K sources they were
12/8/16/13/17. Each class had 240 probes. The different parent-file sampling
prevents attributing the class difference to stellar type alone.

The correction preserved nominal projected amplitudes at 98.89–100.78%.
Its combined held false-alarm rates were 2.50% and 3.59% for the one- and
two-day noise families, respectively; Flat's were 2.93% and 3.75%.
Across all five arms and both families, pooled rates were 2.38–4.22%.
The ideal post-delivery control on our curve recovered ten faint probes,
versus nine with complete refitting. Thus attenuation during our refit
explains only one lost recovery in this check, not the overall deficit.
Profile-mismatch probes retained 96.64–104.13% of projected amplitude.

Mean per-source four-hour scatter was 3.0% higher than SAP, with uncertainty
spanning improvement and deterioration, and 45.4% higher than PSF. The earlier
five-source scatter gain did not generalize to this selected cohort.
The median pixel/SAP scatter ratio was 0.986, with improvement on eleven of
twenty sources; the median pixel/PSF ratio was 1.195. Mean ratios are sensitive
to outliers, so the figure and numerical record retain individual sources.
These results do not establish a photon limit or justify production adoption
or detector recalibration for this processing.

Independent checks reproduced all counts, combined false-alarm decisions and
source-bootstrap intervals. Predetermined M/K null and injection replays
matched saved curves, scores, proposals and fit hashes exactly. Input audits
checked original pixels, source identities, aperture replay, cadence masks,
calibration profiles and delivered products. All 34 experiments-folder tests
passed; lint and complexity checks passed with existing import-order
exceptions in unchanged modules.

The [numerical record](../results/experiments/pixel-comparison-20-20261014.json)
binds the complete external evidence. The
[comparison figure](../dump/pixel-comparison-20-20261014/analysis-v1/comparison.png)
shows every source's four-hour scatter, recovery by depth and held alarms.
The protocol below was frozen before outcome inspection; its original copy
is preserved in `validation-v1/`.

During this run, the author's public extraction and flattening workflow was
located. See [KBonus extraction reproduction](kbonus-extraction-comparison.md).
The current PSF comparison remains conditional because that historical
pipeline has not yet been replayed; public code availability makes this an
assembly and numerical-parity question, rather than a general absence of code.

## Selection

Select by a frozen hash order and input metadata before detector scoring.
Exclude the previous five sources and allow at most one source per original
Kepler target-pixel file parent. Require G between 16 and 18, all compared
products, at least 500 common good cadences, at least 30 days of coverage,
complete coverage of the delivered aperture and no more than 100 native
pixels. Prefer the quarter with the greatest common good cadence count, then
the lower quarter number, using only metadata and support. Apply this rule
identically to M and K selections. Preserve every attempted source and
technical exclusion. This tests the best-covered available quarter per source.

Reuse original local data read-only and acquire missing public mission inputs
on external storage when needed. The requirement for PSF Flat favors quarters
with sufficient captured starlight and removes some difficult cases admitted
by the earlier experiment. Neither this selection nor the photometric M/K
labels establish a representative or spectroscopically classified sample.
The K selection uses locally cached parent files. The M selection expands
to additional local-catalogue parents with newly acquired mission files to
obtain ten eligible sources. Report class results descriptively: the two
classes do not have identical parent-file sampling and class differences
cannot be attributed to stellar type alone.

## Compared processing

Compare five arms on the same observed cadence intersection: delivered SAP,
the unchanged pixel correction, its mission-vector-only ablation, delivered
corrected PSF and delivered PSF Flat. Reproduce delivered SAP from its original
pixels and background convention before using a source. Preserve masks and
all transformations between each product's units and intrinsic-source
fractional flux.

SAP and corrected PSF use native-quarter `qdiag` products. PSF Flat uses the
actual stitched, flattened PSF product. Normalize each delivered PSF product
by its own common-mask median. Normalize SAP and the two pixel corrections
by the native corrected-PSF target rate times the median aperture response;
pixel injections retain the cadence-dependent aperture response.

Keep the pixel method's spatial protection, ridge penalty, four-day folds,
half-day guard intervals, pointing predictors and four mission cotrending
vectors unchanged. Refit all its adaptive coefficients after every pixel
injection and noise realization. Original mission calibration and KBonus
background estimates remain fixed inputs.

The original KBonus PSF extraction and flattening are not rerun. Inject an
ideal fractional transit response into the delivered PSF and PSF Flat curves.
This deliberately grants those comparators perfect upstream preservation;
their measured recovery is conditional on the archived processing. A separate
post-processing injection check on our curve distinguishes residual-noise
performance from losses in the refitted pixel correction.

## Search and scoring

Freeze the executable protocol and source snapshots before scientific runs.
Use the same 1.5-day cubic nuisance fit and BLS search over 9.75–10.25 days
for every arm, with equal temporal weights in intrinsic-source units. Formal
error conventions must not determine the comparative significance. Search
two-, four- and six-hour durations. Inject 40, 100, 300, 1,000, 3,000 and
10,000 ppm nominal source depths at two phases per duration.
Use exposure-integrated transit shapes and retain the observed noise. These
imposed-duration probes compare measurement methods; they do not simulate a
stellar-density-conditioned planet population.

Apply common one- and two-day block signs jointly to the simultaneous pixel
vector, background and delivered curves. For each block family, use 128
calibration and 128 separate held realizations. Refit pixel processing for
each realization. Recovery requires significance under both calibrations at
the nominal 5% level, at least three supported events, timing agreement at
the observed events and the predeclared minimum captured transit energy.
Use observed-event timing instead of the earlier redundant full-span drift
veto. Record every proposal before aggregate scoring.

Noise realizations are conditional surrogates, not independent astrophysical
controls. Report the held false-alarm rate for the exact combined decision.
Do not interpret a lower scatter alone as increased detectability.

## Decision and scope

The primary recovery comparison is 40–1,000 ppm. Deeper injections are
positive controls. Report paired changes by source, pooled across 20 sources
and separately for the M/K selections, with source-cluster uncertainty.
Repeated injections on one source are not independent stars.

To describe the method as winning this pilot, require the lower 95% source
bootstrap bound of the faint-recovery difference to exceed zero against each
of SAP, PSF and PSF Flat. Also require the pixel arm's pooled held false-alarm
rate to remain at or below 5% under both null families and every nominal
probe's known-clock projected amplitude retention to remain within 90–110%.
Report profile-mismatch sensitivity separately. This is a strict pilot
decision under a common calibrated alarm rule; the realized false-alarm rates
may differ and must be shown. Report all comparisons, including losses and
uncertainty; do not select a favorable depth or product after seeing results.
Twenty sources cannot establish the programme's 10,000-case gate or survey
reliability.

Keep evidence under `dump/pixel-comparison-20-20261014/`. Preserve prior
experiments and running survey jobs. Production adoption, detector
recalibration and two full M/K survey runs remain subsequent decisions.
