# TESS replication

Measurements of supplied BOI clocks after vetting, before statistical follow-up.
`uv run --python RUNTIME/bin/python -m tess export-mk VETTING_RUN INPUT.json`
`uv run --python RUNTIME/bin/python -m tess coverage INPUT.json OUTPUT`
`uv run --python RUNTIME/bin/python -m tess run INPUT.json OUTPUT`
Use `--source-ids` or `--sectors` for a bounded first pass.
Install `requirements.txt` into a separate runtime on external storage.
Inputs preserve Gaia identity, run-scoped signal, original clock and evidence.
MAST supplies native FFI cutouts; multiple signals share each host download.
Signed, exposure-integrated depths retain individual events and sectors.
Output includes odd/even, phase-0.5 secondary and pixel difference measurements.
No detector, vetting disposition, depth threshold or FPP is changed.
Missing clock covariance stays unknown; explicit clock trials are all retained.
Outputs resume only with the same input, settings, source code and FITS hashes.
See `docs/tess.md` for units, extraction choices and interpretation limits.
