# Reproducing the KBonus extraction

The extraction and final flattening code are public. The current limitation
is that we have not reproduced the exact archived processing, rather than
that the algorithms or all their code are unavailable.

## Papers and implementations

The relevant catalogue paper is [Martínez-Palomera, Hedges and Dotson
(2023)](https://arxiv.org/abs/2310.17733). It uses Linearized Field Deblending,
introduced by [Hedges et al. (2021)](https://arxiv.org/abs/2106.08411).
The [EXBA aperture paper](https://arxiv.org/abs/2112.06683) describes the
separate extended-background aperture dataset.

| Component | Public source |
| --- | --- |
| Example extraction and configuration | [KBonus notebooks](https://github.com/jorgemarpa/KBonus) |
| PSF fitting library | [PSFMachine](https://github.com/SSDataLab/psfmachine) |
| Rolling-band background model | [kbackground](https://github.com/SSDataLab/kbackground) |
| Extraction workflow | [make_lightcurves.py](https://github.com/jorgemarpa/kepler-workflow/blob/cf9dda6b398d050cf46d62ad01b2459570f492cd/kepler_workflow/make_lightcurves.py) |
| Flattening and stitching | [do_fits_stitch_nas.py](https://github.com/jorgemarpa/kepler-workflow/blob/cf9dda6b398d050cf46d62ad01b2459570f492cd/kepler_workflow/do_fits_stitch_nas.py) |

The inspected workflow revision is `cf9dda6`, dated May 1, 2023. Extraction
loads an FFI shape model, handles background, constructs a time-dependent
PSF using smoothed cotrending vectors, solves source fluxes and iterates
priors for negative solutions. These are exposed operations that can be
replayed once the required scene and calibration inputs are assembled.

## What our method changes

Our current experiment reuses mission-calibrated pixels, the archived
background convention and published KBonus PSF shapes. It keeps the SAP
aperture and subtracts a locally learned correction. Reference pixels are
projected away from the target profile and its position derivatives;
time-held fits use those references, pointing and mission vectors.

The KBonus workflow instead estimates source amplitudes jointly from a
scene model. A faithful corrected-PSF replay needs the surrounding batch,
not just one star's cutout. The catalogue paper describes batches of roughly
200 TPFs and warns that the full perturbation model needs many TPFs.
Our current twenty-source experiment compares actual delivered PSF products
conditionally; it does not yet measure their pixel-to-light-curve transfer.

## Release identity remains unresolved

All twenty original delivery headers record `ORIGIN=PSFMACHINE`,
`VERSION=1.1.2`, `HLSPVER=V1.0` and `PROCVER=2.0.9`. The last value comes from
the Lightkurve FITS writer, not a KBonus workflow release. The paper names
PSFMachine 1.1.4. Headers lack the exact workflow commit and flattening
parameters, so these differences require reconciliation rather than an
assumption about which implementation generated the archived arrays.
The stitching code copies the first quarter's primary header; its version
and date fields need not identify the later flattening runtime.

The paper's section III describes two-day flattening. The inspected public
workflow calls a 1.5-day spline with three-sigma regression clipping,
pointing/gap segmentation and additive correction. It aligns quarter
trends using a third-order polynomial and rescales quarter means. Its NAS
variant uses `PSFFRAC=0.4`, versus the paper's 0.5 selection. The two-day
smoothing of cotrending vectors is a separate processing stage.

## Bounded replay result

The first selected source, Gaia DR3 `2052281338504549120`, quarter 5,
was used for a baseline-only replay. Its official channel background file
was acquired to supply the author's pointing-break inputs. The unchanged
`fancy_flatten` function ran with its 1.5-day spline, physical flux units
and an empty design-matrix cache.

Lightkurve 2.0.9 failed during uncertainty propagation: the regression
corrector attempted to add dimensional flux variance to dimensionless model
variance. One alternate, Lightkurve 2.3.0 explicitly pinned by the authors'
KBonus requirements, reproduced the error. No author-function edit or unit
stripping was used to bypass it. Both failures and runtime snapshots are
preserved in `research/flat-replay/`; the final receipt is
`blocked-receipt-v2.json`. No numerical parity result is claimed.

This narrows the next step to historical reader units and dependency
compatibility. It does not establish a defect in the archived photometry.

## Next reproducibility check

First reproduce an unmodified delivered quarter and its PSF Flat output.
Use the complete native-quarter arrays, original masks, an identified
Lightkurve version and the public operator. Compare cadence support,
fluxes, uncertainties and break masks against the archive before injecting
anything. Preserve every attempted variant and residual; do not choose a
variant by its transit recovery.
Retain the accepted-quarter context when comparing final amplitudes: the
public stitching code rescales each quarter using cross-quarter means.

A quarter-to-Flat check can use the mounted LC2 arrays. The public code
obtains break locations from `POS_CORR1/2[:, 2000]` in channel background
FITS files; one such file was acquired for the bounded replay above. Exact
stitching also needs accepted-quarter selection and normalization. The workflow
references an untracked correlated-source exclusion table. Availability of
other public or archived copies remains to be checked.

After baseline parity, inject identical signals into original pixels and
refit both pipelines, including adaptive scene, background and flattening
steps. Score recovery with the same detector and calibrated false-alarm
rule. Frozen archive outputs establish baseline fidelity; blinded injected
signals and independent controls establish comparative performance.

Source snapshots, hashes and header receipts are under
`dump/pixel-comparison-20-20261014/research/` and `audit/`. This research did
not change the running twenty-source comparison or its frozen protocol.
