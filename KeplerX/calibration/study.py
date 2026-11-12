"""One calibration study."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, config_file, digest, read, receipt, require, write
from runner.outputs import RunOutputs

REGISTRY = ROOT / "calibration/registered-sample.json"
STUDIES = ROOT / "results/calibration-studies"
# What a run imports or reads as settings; a study runs one commit of it.
CODE = (
    "calibration", "common.py", "config.json", "detector", "localizator", "photometry",
    "runner", "vetter", "run.py", "run.sh", "upstream.json", "uv.lock",
)  # fmt: skip


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def code_state():
    return {
        "commit": git("log", "-1", "--format=%H", "--", *CODE),
        "clean": not git("status", "--porcelain", "--", *CODE),
    }


def sample_state(registry=REGISTRY):
    """The registered sample, every record checked against its receipt."""
    path, registry = registry, read(registry)
    for name, record in {**registry["records"], **registry["manifests"]}.items():
        require(
            digest(ROOT / record["path"]) == record["sha256"],
            f"Sample record changed since registration: {name}",
        )
    return {
        "name": registry["name"],
        "root": str(ROOT / registry["root"]),
        "registry_sha256": digest(path),
    }


def folder_of(name, studies=STUDIES):
    return Path(studies) / name


def attempts(folder):
    return sorted(
        (p for p in (folder / "attempts").glob("*") if p.is_dir()),
        key=lambda p: int(p.name),
    )


def completed(folder):
    """Case IDs some attempt completed."""
    done = set()
    for attempt in attempts(folder):
        if (attempt / "results.sqlite").exists():
            rows = RunOutputs(attempt).database.execute(
                "SELECT case_id FROM jobs WHERE state = 'COMPLETE'"
            )
            done |= {case_id for (case_id,) in rows}
    return done


def start(folder, code, sample, partitions):
    folder.mkdir(parents=True)
    write(folder / "config.json", config_file())
    lines = []
    for part in partitions:
        lines += (
            (Path(sample["root"]) / "manifests" / f"{part}-all.jsonl")
            .read_text()
            .splitlines()
        )
    (folder / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    study = {
        "schema": 1, "name": folder.name, "state": "RUNNING", "started_utc": now(),
        "partitions": partitions, "code": code, "sample": sample,
        "config_sha256": digest(folder / "config.json"),
        "manifest": receipt(folder / "manifest.jsonl"), "attempts": [],
    }  # fmt: skip
    write(folder / "study.json", study)
    return study


def resume(folder, code, sample, partitions):
    study = read(folder / "study.json")
    require(
        study["code"]["commit"] == code["commit"],
        "The code changed since the study started",
    )
    require(study["sample"] == sample, "The sample changed since the study started")
    require(
        digest(folder / "config.json") == study["config_sha256"],
        "The study config changed",
    )
    require(
        digest(folder / "manifest.jsonl") == study["manifest"]["sha256"],
        "The manifest changed",
    )
    require(study["partitions"] == partitions, "A study keeps its partitions")
    return study


def run(name, workers, partitions, registry=REGISTRY, studies=STUDIES, bootstrap=200,
        require_clean=True):  # fmt: skip
    """Start or resume a study; score it once every case is complete."""
    folder, code = folder_of(name, studies), code_state()
    sample = sample_state(registry)
    require(
        code["clean"] or not require_clean,
        "Commit the code first: a study runs one commit",
    )
    exists = (folder / "study.json").exists()
    study = (resume if exists else start)(folder, code, sample, partitions)
    rows = [
        json.loads(line)
        for line in (folder / "manifest.jsonl").read_text().splitlines()
        if line
    ]
    pending = [row for row in rows if row["case_id"] not in completed(folder)]
    if pending:
        number = len(attempts(folder)) + 1
        manifest = folder / "attempts" / f"{number}.jsonl"
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text("".join(json.dumps(row) + "\n" for row in pending))
        study["attempts"].append(
            {"attempt": number, "started_utc": now(), "cases": len(pending)}
        )
        write(folder / "study.json", study | {"state": "RUNNING"})
        command = [
            "caffeinate", "-i", "-s", str(ROOT / "run.sh"), "--config", str(folder / "config.json"),
            "batch", str(manifest), str(folder / "attempts" / str(number)), "--workers", str(workers),
        ]  # fmt: skip
        exit_code = subprocess.run(command, cwd=ROOT).returncode
        study["attempts"][-1] |= {"ended_utc": now(), "exit_code": exit_code}
    remaining = len(rows) - len(completed(folder))
    study |= {
        "state": "INCOMPLETE" if remaining else "RUN_COMPLETE",
        "remaining_cases": remaining,
    }
    write(folder / "study.json", study)
    require(
        not remaining,
        f"{remaining} cases not complete; run the study again for a new attempt",
    )
    return score(name, bootstrap, workers, studies)


def score(name, bootstrap=200, workers=10, studies=STUDIES):
    """Score every attempt of a complete study together (score/)."""
    folder = folder_of(name, studies)
    study = read(folder / "study.json")
    require(
        study["state"] in ("RUN_COMPLETE", "SCORED"),
        "Score a study once its run is complete",
    )
    out = folder / "score"
    for n in range(2, 1000):
        if not out.exists():
            break
        out = folder / f"score-{n}"
    command = [
        sys.executable, "-m", "calibration.scoring", *map(str, attempts(folder)),
        "--sample", study["sample"]["root"], "--out", str(out),
        "--workers", str(workers), "--bootstrap", str(bootstrap),
    ]  # fmt: skip
    subprocess.run(command, cwd=ROOT, check=True)
    summary = read(out / "summary.json")
    headline = {
        part: {
            "detectable": s["detection_on_detectable_planets"],
            "efficiency": {k: s["efficiency_curve"]["gamma_fit"][k] for k in ("snr_at_50pct", "snr_at_90pct")},
            "occurrence_integrated": s["completeness_model"]["occurrence_integrated"],
            "habitable_zone_detectable": (s.get("habitable_zone") or {}).get("detectable"),
        }
        for part, s in summary["partitions"].items()
    }  # fmt: skip
    study |= {"state": "SCORED", "scored_utc": now(), "score": receipt(out / "summary.json"),
              "headline": headline}  # fmt: skip
    write(folder / "study.json", study)
    return headline


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="Start or resume a study, then score it")
    r.add_argument("name")
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--partitions", nargs="+", default=["development"],
                   choices=["development", "holdout"])  # fmt: skip
    s = sub.add_parser("score", help="Score a complete study again")
    s.add_argument("name")
    s.add_argument("--bootstrap", type=int, default=200)
    s.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.command == "run":
        result = run(args.name, args.workers, args.partitions)
    else:
        result = score(args.name, args.bootstrap, args.workers)
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
