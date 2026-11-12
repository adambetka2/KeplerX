# Shallow BLS detector

The survey's second detector. It reads the prepared light curves of one
discovery view (PSF_FLAT, full PSF or SAP), searches 40–1,000 ppm transits in
three budgeted period ranges (0.2–0.5, 0.5–30 and 30–100 days, each capped at
half the observed baseline) and emits at most five proposals per range and
source for the shared vetting queue. Method and limits: `../../docs/survey-method.md`.

- `run.py` is the detector: quarter-wise median detrending, empirical quarter
  scatter, box least squares on a period and duration grid, peak refinement,
  and the brightening, sinusoid and odd/even checks. It is also the common
  library of the research detector in `../../dwarfs/shallow/`.
- `bls.c` reproduces Astropy 7.2.2 `BoxLeastSquares.power(objective="snr")`
  bit for bit (adapted from Astropy's `bls.c`); `run.py` compiles
  it with the system `cc` into `dump/shallow-bls/`. `native-bls-oracle.npz`
  and `test_run.py` check it.
- `config.json` holds the frozen survey gates. `server.sh` runs the detector
  on the Linux server with the pinned `requirements.txt`.

```bash
uv run --no-project --with-requirements pipeline/shallow/requirements.txt \
  python pipeline/shallow/run.py INPUT_FSC_DIR dump/RUN --source-ids IDS.txt
```

Sources with too few usable quarters, too short a baseline or nonpositive
scatter are recorded as ineligible; unexpected errors stop the run.
