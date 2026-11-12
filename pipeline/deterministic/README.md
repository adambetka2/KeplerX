# Deterministic transit search

The survey runner. It prepares each KBonus light curve once, runs the Rust
detector (`src/main.rs`: event-seeded BLS-like periodic search, survivor-review
ephemerides and period-free `EVENT_LIKE` intervals) and the shallow BLS
detector (`../shallow/run.py`) on the same prepared input, and sends both
proposal sets into one shared vetting queue. Provenance label:
`rust_v6_shallow_v1`. Interpretation and limits: `../../docs/survey-method.md`.

- `run.py` prepares inputs, runs both detectors, builds the immutable proposal
  queue and launches vetting; `config.json` holds the Rust gates.
- `full_survey.py`, `vet_survey.py`, `multiview.py`, `source_ledger.py`,
  `release_artifacts.py` and `release_report.py` freeze, run, vet and release a
  bounded survey; `compare_runs.py` compares two runs field by field.
- `controls.py`, `control_*.py`, `catalogue_*.py`, `false_class_*.py`,
  `holdout_controls.py`, `injections.py`, `pixel_*.py`, `shallow_*.py`,
  `null_*.py` and `signal_contracts.py` freeze and replay the control,
  injection and null cohorts without opening outcomes early.
- `cloud*.py` with the `cloud-*.json` records replay the opened development
  controls from public inputs on another machine.
- `test_*.py` are the contract tests.

Commands run from the repository root with both discs mounted and Rust on `PATH`:

```bash
env_dir=dump/deterministic-env
uv venv --python 3.11 "$env_dir"
uv pip sync --python "$env_dir/bin/python" pipeline/vetting/requirements.txt
"$env_dir/bin/python" pipeline/run.py deterministic all --shards 200 201 --limit 1000 --run dump/RUN
"$env_dir/bin/python" pipeline/run.py deterministic survey --shards 228 229 --limit 1000 --batch-size 500 --threads 10 --run dump/SURVEY
"$env_dir/bin/python" -m pipeline.deterministic.vet_survey --survey dump/SURVEY --run dump/SURVEY-vetting --batch-size 50 --threads 10 --workers 4
"$env_dir/bin/python" -m pipeline.deterministic.source_ledger --source-ids SOURCE_IDS.json --out dump/LEDGER
"$env_dir/bin/python" -m pipeline.deterministic.full_survey --freeze-policy dump/RELEASE/psf_flat-policy.json --discovery-view psf_flat
"$env_dir/bin/python" -m pipeline.deterministic.full_survey --source-ids SOURCE_IDS.json --input-ledger dump/LEDGER --disposition-policy dump/RELEASE/psf_flat-policy.json --discovery-view psf_flat --survey dump/RELEASE/survey --vetting dump/RELEASE/vetting
PYTHONPATH=.:pipeline:pipeline/photometry:pipeline/vetting OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$env_dir/bin/python" -m pytest pipeline --import-mode=importlib --basetemp=dump/pytest
```

`all` runs `prepare`, the Cargo build (`--target-dir dump/RUN/build`), `scan`
and `vet` in turn; each stage also runs alone. Catalogue vetting needs the
frozen snapshot from `../vetting/catalogue_snapshot.py`. Every run freezes its
binary, configuration, input and catalogue hashes and refuses to resume under
a different hash. Outputs under the run folder: `raw.csv` (detector
diagnostics; `TRANSIT_LIKE` and `EVENT_LIKE` are detector labels, not BOIs),
`detections.csv` (every classification of a bounded survey),
`vetting/signal-ledger.json` (first failure of each HELD or DISMISSED signal),
`vetting/stage-timings.json` and `result.json` (counts, BOI dispositions and
processing status).
