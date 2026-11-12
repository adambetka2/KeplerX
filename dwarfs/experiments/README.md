# Photometry experiments

Bounded development experiments on explicitly selected, unsealed inputs.
`pixels.py` checks pixel injection transfer and records an electron-noise budget.
`preparation.py` compares biweight windows with joint projection in a blind search.

`native.py` runs paired configurations of the unchanged full detector;
`summarize.py` produces numerical summaries and figures. These are conditional
development pilots, not completeness estimates or planet validation.

`reprocess.py` learns a target-protected correction from original pixels;
`reprocess_validation.py` refits it for every injection and pixel null.
`comparison_inputs.py`, `compare_photometry.py` and `comparison_summary.py`
extend that paired test to 20 stars and SAP, PSF and PSF Flat products.
See `docs/pixel-comparison-20.md` for its frozen comparison and limits.

Run through `uv`; keep inputs, masks, proposals, failures and caches under `dump/`.
`flat_retrieval.py`, `usable_flat.py` and `sap_retrieval.py` prepare and compare pilots.
`flat_sources.py` freezes targets; `flat_real.py` and `flat_local.py` launch real runs.
`flat_subset.py` searches authorized subsets; `flat_vetting.py` runs their shared vetting.
