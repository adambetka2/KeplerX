"""Read-only display extracts for frozen BOI event windows; no search or fit."""

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plot_key(job, view):
    value = [job["source_id"], job["signal_id"], job["discovery_view"], view]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def curve_key(source_id, view):
    return hashlib.sha256(json.dumps([source_id, view]).encode()).hexdigest()


@lru_cache(maxsize=4)
def frozen_inputs(path):
    return {row["source_id"]: row for row in json.loads(path.read_text())["sources"]}


def verify_inputs(raw, sid, path, reader):
    frozen = frozen_inputs(path)[sid]
    flatten = np.asarray(raw["flatten_mask"])
    measured = hashlib.sha256(
        str(flatten.dtype).encode() + flatten.tobytes()
    ).hexdigest()
    if measured != frozen["flatten_mask_sha256"]:
        raise ValueError(f"Frozen flatten mask differs: {sid}")
    for record in frozen["views"]:
        if (
            reader.curve_digest(reader.photometry(raw, record["view"]))
            != record["input_arrays_sha256"]
        ):
            raise ValueError(f"Frozen photometry differs: {sid} {record['view']}")


def full_curve(raw, view, config, reader):
    data = reader.photometry(raw, view)
    valid = reader.vetting_points(
        data, config["quality_bitmask"], reader.uses_flatten_mask(view, config)
    )
    arrays = {}
    for key, array in data.items():
        if key in reader.KEYS:
            arrays[key] = [
                None if isinstance(v, float) and not np.isfinite(v) else v
                for v in array.tolist()
            ]
    arrays["valid"] = valid.tolist()
    baselines = {}
    for quarter in np.unique(data["quarter"]):
        take = valid & (data["quarter"] == quarter)
        baseline = float(np.median(data["flux"][take])) if take.any() else None
        baselines[str(int(quarter))] = (
            baseline if baseline and np.isfinite(baseline) and baseline > 0 else None
        )
    return {"arrays": arrays, "quarter_baselines": baselines, "cadences": len(valid)}


def write_gzip(path, value):
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def windows(raw, units, view, config, reader):
    data = reader.photometry(raw, view)
    valid = reader.vetting_points(
        data, config["quality_bitmask"], reader.uses_flatten_mask(view, config)
    )
    result = []
    for unit in units:
        centre, duration = unit["centre_bkjd"], unit["duration_days"]
        half = max(config["minimum_context_days"], 4 * duration)
        context = (data["quarter"] == unit["quarter"]) & (
            abs(data["time"] - centre) <= half
        )
        take = context & valid
        time, flux, error = (data[k][take] for k in ("time", "flux", "error"))
        outside = abs(time - centre) > duration
        if outside.sum() < 4:
            result.append(
                {
                    "quarter": unit["quarter"],
                    "centre_bkjd": centre,
                    "status": "NO_BASELINE",
                }
            )
            continue
        baseline = float(np.median(flux[outside]))
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError("Nonpositive display baseline")
        result.append(
            {
                "status": "MEASURED",
                "quarter": unit["quarter"],
                "centre_bkjd": centre,
                "duration_days": duration,
                "time_hours": ((time - centre) * 24).tolist(),
                "flux_ppm": ((flux / baseline - 1) * 1e6).tolist(),
                "error_ppm": (error / baseline * 1e6).tolist(),
                "cadences": data["cadence"][take].tolist(),
                "baseline_flux": baseline,
                "excluded_context_points": int((context & ~valid).sum()),
            }
        )
    return result


def prepare_source(root, out, jobs, full=False):
    from pipeline.photometry import lc2

    sid = jobs[0]["source_id"]
    metadata, raw = lc2.load_source(root, sid)
    records = []
    for job in jobs[:1] if full else jobs:
        evidence = Path(job["evidence"])
        if digest(evidence) != job["evidence_sha256"]:
            raise ValueError(f"Frozen source evidence changed: {sid}")
        with gzip.open(evidence, "rt") as stream:
            source = json.load(stream)
        if metadata["raw_fits_sha256"] != source["raw_fits_sha256"]:
            raise ValueError(f"LC2 source identity differs: {sid}")
        signal = next(
            s for s in source["signals"] if s["signal_id"] == job["signal_id"]
        )
        config_path = evidence.parent.parent.parent / "vetting.json"
        verify_inputs(raw, sid, config_path.parent / "preparation.json", lc2)
        config = json.loads(config_path.read_text())
        for view in ("psf_flat", "psf_qdiag", "sap_qdiag"):
            key = curve_key(sid, view) if full else plot_key(job, view)
            path = Path(out) / f"{key}.json.gz"
            value = {
                "schema": 1,
                "source_id": sid,
                "signal_id": job["signal_id"],
                "discovery_view": job["discovery_view"],
                "view": view,
                "evidence_sha256": job["evidence_sha256"],
                "config_sha256": digest(config_path),
                "raw_fits_sha256": source["raw_fits_sha256"],
                "input_arrays_sha256": lc2.curve_digest(lc2.photometry(raw, view)),
                "display_rule": "Frozen vetting quality/flatten masks. Original event centres. +/- max(0.5 d, 4 durations); median outside one duration normalizes each window. No detrending, clipping, refit or new detection.",
            }
            if full:
                value.update(full_curve(raw, view, config, lc2))
                value["display_rule"] = (
                    "All original cadence arrays, including masked and nonfinite rows (nonfinite values encoded as null). Per-quarter median of frozen valid points provided for display normalization only. No detrending, clipping, search or fit."
                )
            else:
                value["windows"] = windows(raw, signal["events"], view, config, lc2)
            if path.exists():
                with gzip.open(path, "rt") as stream:
                    if json.load(stream) != value:
                        raise ValueError(f"Existing display extract differs: {key}")
            else:
                write_gzip(path, value)
            records.append(
                {
                    "key": key,
                    "file": path.name,
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                }
            )
    return records


def prepare_group(arguments):
    root, out, groups, full = arguments
    return [row for jobs in groups for row in prepare_source(root, out, jobs, full)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Extract complete source curves instead of event windows",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    plan = json.loads(args.plan.read_text())
    sources = defaultdict(list)
    for job in plan["jobs"]:
        sources[job["source_id"]].append(job)
    by_shard = defaultdict(list)
    with sqlite3.connect(
        f"file:{args.root / 'discSSD/data/index.sqlite'}?mode=ro", uri=True
    ) as database:
        for sid, jobs in sources.items():
            row = database.execute(
                "select volume, shard from sources where gaia_source_id=?", (sid,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Missing source: {sid}")
            by_shard[tuple(row)].append(jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    tasks = [
        (args.root, args.out, groups, args.full)
        for _, groups in sorted(by_shard.items())
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rows in enumerate(pool.map(prepare_group, tasks), 1):
            records.extend(rows)
            print(
                json.dumps(
                    {"shards": i, "total_shards": len(tasks), "files": len(records)}
                ),
                flush=True,
            )
    receipt = {
        "status": "COMPLETE_READ_ONLY_PLOT_EXTRACTS",
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "catalogue_sha256": plan["catalogue_sha256"],
        "plan_sha256": digest(args.plan),
        "reader_sha256": digest(args.root / "pipeline/photometry/lc2.py"),
        "extractor_sha256": digest(__file__),
        "sources": len(sources),
        "proposals": len(plan["jobs"]),
        "full_curves": args.full,
        "files": records,
        "bytes": sum(r["bytes"] for r in records),
        "scope": "Display only. Original result files and light curves opened read-only. All three photometries retained for every proposal; no scientific calculation rerun.",
    }
    write_gzip(args.out / "index.json.gz", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
