"""Run comparison must ignore timing and its derived hashes, and nothing else."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline")]

from deterministic.compare_runs import compare_runs


def write_run(path, depth, seconds, digest):
    record = {
        "source_id": "1",
        "signals": [{"contamination": {"depth": depth, "wall_seconds": seconds}}],
        "assessment": {
            "evidence_refs": [
                {"artifact": "/signals/0/contamination", "sha256": digest}
            ]
        },
    }
    (path / "vetting" / "sources").mkdir(parents=True)
    (path / "vetting" / "sources" / "1.json").write_text(json.dumps(record))
    return path


def compare(tmp_path, candidate):
    baseline = write_run(tmp_path / "baseline", 0.5, 1.0, "a" * 64)
    candidate = write_run(tmp_path / "candidate", *candidate)
    return compare_runs(baseline, candidate, [], 0.0, 0.0)


def test_timing_and_its_hash_are_ignored(tmp_path):
    report = compare(tmp_path, (0.5, 2.0, "b" * 64))
    assert report["records"]["identical"] == 1
    assert report["material_differences"] == {}


def test_referenced_value_difference_is_reported(tmp_path):
    report = compare(tmp_path, (0.6, 2.0, "b" * 64))
    assert list(report["material_differences"]) == ["signals/[]/contamination/depth"]
