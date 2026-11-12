# KeplerX results

Everything a KeplerX run produces lives here. Inputs are in `../data/`;
environments, caches and scratch files in `../runtime/`. Folders are not in
git; small top-level records are.

| Folder | Holds |
| --- | --- |
| `calibration-studies/` | One folder per calibration study (`calibration.study`): `study.json`, `config.json`, `manifest.jsonl`, `attempts/` and `score/` |
| `calibration-previews/` | Calibration runs before the study, one dated folder each with its `run/`, `score/`, config, launch script and logs |
| `calibration-audits/` | Dated audits of calibration samples and runs |
| `speed-benchmarks/` | Dated timing benchmarks |
| `software-checks/` | Dated software checks (known objects, native localizer and runner) |
| `performance-engineering/` | Speed probes and their scripts, one folder per day |
| `test-runs/` | Outputs of dated test runs during setup |
| `setup-records/` | Logs and checks from setting up the environments |
| `storage-records/` | Storage audits and moves, including `2026-11-01-descriptive-names.json`, every path renamed on 1 November 2026 |

A run writes `results.sqlite` (runner/outputs.py) with its config and logs.
