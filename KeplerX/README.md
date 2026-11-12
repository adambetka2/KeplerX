# KeplerX (third campaign)

KeplerX (Detector, Vetter, Localizator and Validator), input adapters and independent
KBonus calibration.

Entry point from this folder:

```text
run.sh setup; run.sh doctor; run.sh check
```

The survey sample is unset; synthetic and real-object controls are separate.
Detection calls native Detector through a prepared-array adapter.
Batch connects Detector, Vetter and native Localizator when pixels are supplied.
Runtime and outputs belong on the internal Mac disk.
Original mounted data and historical runs remain unchanged.
See ../docs/keplerX.md for schemas, parameters and evidence limits.

## Layout

Every name says what the thing holds or what it is for. One-off records
(runs, audits, probes) start with their date; living things (the sample, its
design inputs, the code) carry none. Vendored upstream files keep their
upstream names.

| Folder | Holds |
| --- | --- |
| `detector/`, `vetter/`, `localizator/` | The three pipeline stages; `upstream/` in each is the vendored original |
| `validator/` | Statistical-validation inputs: the post-BOI preparation and native transit fit (`validator/README.md`); `upstream/` holds the vendored dynesty |
| `photometry/` | Reading delivered PSF Flat light curves, and the stitched SAP flattened as KBonus flattened PSF (the `sap_flat` view) |
| `runner/` | Batches of cases over a manifest, and their output database |
| `calibration/` | The calibration sample's builder, its scoring and the calibration study |
| `tests/` | Golden, equivalence and end-to-end tests |
| `data/` (not in git) | Inputs the pipeline reads: `calibration-sample/` (light curves, truth, manifests, build logs), `calibration-sample-design/` (lane inventory, noise census, artifact models, Gaia colours and the measurements behind each choice), `calibration-sample-sap/` (the SAP lane's twin of the development-quick set), the survey and localization benchmarks and the Gaia field. The three calibration folders are symlinks to `/Volumes/discSSD/keplerX/data/`, so discSSD must be mounted |
| `results/` (not in git, except small records) | Everything runs produce; see `results/README.md` |
| `runtime/` (not in git) | `pipeline-environment/` (the locked Python environment `run.sh` uses), `cache/` and `scratch/` |

Environment variables: `KEPLERX_RUNTIME_ROOT` (default `runtime/`),
`KEPLERX_KERNEL_CACHE`, `KEPLERX_NUMBA_THREADS`, `KEPLERX_ISOLATION`,
`KEPLERX_GPU` and `KEPLERX_GPU_SLOTS`.
