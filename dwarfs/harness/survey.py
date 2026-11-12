"""Run the M/K SAP survey: claim sources bright first, prepare, search, keep proposal evidence.

A worker prepares each source from the read-only corpus when it claims it: SAP flux minus the
(1 - CROWDSAP) share of the quarter's median, with CROWDSAP from the LIGHTCURVE_Qn header card,
then the unchanged survey preparation. The search runs the one design of current-designs.json
through evolution.scan_records under the "proposals" array policy. Every result is saved
atomically after its arrays, so a stopped run keeps each finished search and its proposals,
and a restarted worker continues the queue.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "shallow"))
sys.path.insert(0, str(ROOT))
import evolution as API

from pipeline.deterministic.run import InsufficientCadences, prepare_curve_records
from pipeline.photometry.lc2 import load_source, photometry, read_source_rows

PREPARATION_FILES = (
    ROOT / "pipeline/photometry/lc2.py",
    ROOT / "pipeline/deterministic/run.py",
    ROOT / "pipeline/deterministic/signal_contracts.py",
    HERE.parent / "shallow/evolution.py",
    Path(__file__).resolve(),
)
INPUT_CLASS = "NATIVE_SAP_CONSTANT_NEIGHBOUR_SOURCE_FLUX_V1"
PREPARATION = (
    "sap_qdiag view; per quarter the median M of finite time/flux/positive-error quality0 rows; "
    "source_flux = SAP_flux - (1 - CROWDSAP) * M with the delivered quarter CROWDSAP of the "
    "LIGHTCURVE_Qn header; then quality0/finite-positive-error/positive-quarter-median/"
    "upper12MAD/time-cadence-sort-and-dedup/FSCfloat32 records; mask_flatten=False"
)
QUEUE_FIELDS = ("source_id", "population", "phot_g_mean_mag", "quarter_count", "shard", "kic")
PROPOSAL_FIELDS = (
    "proposal_priority", "period_days", "t0_bkjd", "duration_hours", "depth_ppm", "search_snr",
    "search_band", "search_rank", "observed_events", "observed_quarters", "in_transit_cadences",
    "inverse_search_snr", "inverse_same_period_snr", "dip_inverse_snr_ratio", "odd_even_sigma",
    "harmonic_delta_log_likelihood", "positive_cycle_fraction", "profile_index",
    "physical_density_g_cm3", "physical_delta_log_likelihood",
)
SOURCE_FIELDS = (
    "source_id", "population", "phot_g_mean_mag", "quarter_count", "shard", "kic", "status",
    "first_failure", "rows", "usable_quality_cadences", "proposal_count", "search_seconds",
    "search_cpu_seconds", "elapsed_seconds", "evidence_bytes", "worker", "utc",
)
ANTICIPATED_BYTES = 4 * 1024**2  # trimmed JSON plus the evidence npz of a proposal source
CROWDSAP = re.compile(r"^CROWDSAP\s*=\s*([-+0-9.Ee]+)", re.MULTILINE)


def utc():
    return datetime.datetime.now(datetime.UTC).isoformat()


def quarter_crowding(shard, sid):
    columns = ["source_id", "extname", "quarter", "header_cards"]
    table = read_source_rows(shard / "header.parquet", sid, columns)
    crowding = {}
    for row in table.to_pylist():
        match = CROWDSAP.search(row["header_cards"])
        if row["extname"].startswith("LIGHTCURVE_Q") and match:
            crowding[int(row["quarter"])] = float(match.group(1))
    return crowding


def source_curve(curve, crowding):
    """Subtract the constant neighbour share of each quarter's median SAP flux."""
    flux = curve["flux"].astype(float)
    good = (
        np.isfinite(curve["time"])
        & np.isfinite(flux)
        & np.isfinite(curve["error"])
        & (curve["error"] > 0)
        & (curve["quality"] == 0)
    )
    scale, crowd = np.zeros(len(flux)), np.empty(len(flux))
    quarters = [int(quarter) for quarter in np.unique(curve["quarter"])]
    for quarter in quarters:
        use = curve["quarter"] == quarter
        value = crowding.get(quarter)
        if value is None or not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"Quarter {quarter} has no usable CROWDSAP: {value}")
        crowd[use] = value
        if np.any(use & good):
            scale[use] = np.median(flux[use & good])
    detail = {
        "quarter_crowding": {quarter: crowding[quarter] for quarter in quarters},
        "quarter_median": {
            quarter: float(scale[curve["quarter"] == quarter][0]) for quarter in quarters
        },
        "usable_quality_cadences": int(good.sum()),
    }
    return curve | {"flux": flux - (1 - crowd) * scale}, detail


def prepare_source(root, entry):
    sid = entry["source_id"]
    meta, raw = load_source(root, sid)
    curve = photometry(raw, "sap_qdiag")
    job = {
        "key": sid,
        "source_id": sid,
        "shard": meta["shard"],
        "kic": meta["kic"],
        "population": entry["population"],
        "phot_g_mean_mag": float(entry["phot_g_mean_mag"]),
        "photometry_view": "sap_qdiag",
        "input_class": INPUT_CLASS,
        "delivered_cadences": len(curve["flux"]),
    }
    try:
        crowding = quarter_crowding(Path(meta["shard_path"]), int(sid))
        corrected, detail = source_curve(curve, crowding)
        job.update(detail)
        records = prepare_curve_records(corrected)
    except (InsufficientCadences, ValueError) as error:
        return job, None, f"{type(error).__name__}: {error}"
    job.update(rows=len(records), records_sha256=hashlib.sha256(records.tobytes()).hexdigest())
    return job, records, None


def claim(folder, key, name):
    path = folder / f"{key}.claim"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"worker": name, "pid": os.getpid(), "utc": utc()}, handle)
    return True


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    return not cmdline.exists() or b"survey.py" in cmdline.read_bytes()


def summarize(entry, result, worker):
    job = result.get("input", {})
    arrays = result.get("arrays") or {}
    return {
        **{key: entry[key] for key in QUEUE_FIELDS},
        "phot_g_mean_mag": float(entry["phot_g_mean_mag"]),
        "quarter_count": int(entry["quarter_count"]),
        "status": result["status"],
        "first_failure": result.get("first_failure"),
        "rows": job.get("rows"),
        "usable_quality_cadences": job.get("usable_quality_cadences"),
        "search_seconds": result.get("search_seconds"),
        "search_cpu_seconds": result.get("search_cpu_seconds"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "evidence": arrays.get("path"),
        "evidence_bytes": arrays.get("bytes"),
        "proposals": result.get("proposals", []),
        "worker": worker,
        "utc": utc(),
    }


def prepare(args):
    out = args.out
    out.mkdir(parents=True)
    targets = API.rows(args.targets)
    queued = [row for row in targets if row["hopeless"] == "0"]
    queued.sort(key=lambda row: (float(row["phot_g_mean_mag"]), row["source_id"]))
    designs = API.read(args.design)
    (design,) = designs
    API.capacity(out, len(queued) * ANTICIPATED_BYTES)
    bindings = API.snapshot(out, designs) | API.bind_files(PREPARATION_FILES)
    identity = {
        "status": "RUNNING",
        "mode": "survey",
        "array_policy": "proposals",
        "root": str(args.root.resolve()),
        "design": design,
        "targets": {
            "path": str(args.targets.resolve()),
            "sha256": API.digest(args.targets),
            "rows": len(targets),
            "queued": len(queued),
            "hopeless_skipped": len(targets) - len(queued),
            "order": "phot_g_mean_mag ascending",
        },
        "preparation": {
            "photometry_view": "sap_qdiag",
            "input_class": INPUT_CLASS,
            "text": PREPARATION,
        },
        "source_bindings": bindings,
        "runtime": API.runtime_identity(),
        "prepared_utc": utc(),
    }
    API.save(out / "identity.json", identity)
    order = [{key: row[key] for key in QUEUE_FIELDS} for row in queued]
    API.save(out / "queue.json", {"order": order}, indent=None)
    for name in ("results", "summaries", "claims", "workers", "logs"):
        (out / name).mkdir()
    print(
        json.dumps(
            {
                "out": str(out),
                "design": design["name"],
                "queued": len(queued),
                "hopeless_skipped": len(targets) - len(queued),
                "brightest_g": queued[0]["phot_g_mean_mag"],
                "faintest_g": queued[-1]["phot_g_mean_mag"],
            }
        )
    )


def failure_result(status, error, job, start, policy):
    return {
        "status": status,
        "first_failure": f"{type(error).__name__}: {error}",
        "exception_traceback": traceback.format_exc(),
        "input": job,
        "elapsed_seconds": time.monotonic() - start,
        "array_policy": policy,
        "arrays": None,
    }


def search_source(root, entry, target, policy, start):
    job, records, failure = prepare_source(root, entry)
    if failure is None:
        API.scan_records(job, records, target)
        return
    result = {
        "status": "INPUT_INELIGIBLE",
        "first_failure": failure,
        "input": job,
        "elapsed_seconds": time.monotonic() - start,
        "array_policy": policy,
        "arrays": None,
    }
    API.save(target, result, indent=None)


def worker(args):
    identity = API.read(args.out / "identity.json")
    API.check_bindings(identity["source_bindings"])
    if API.runtime_identity() != identity["runtime"]:
        raise ValueError("Numerical runtime differs from the survey identity")
    design, policy = identity["design"], identity["array_policy"]
    source = args.out / "source/dwarfs/shallow"
    API.initialize(source, design["engine"], design["config"], array_policy=policy)
    root = Path(identity["root"])
    order = API.read(args.out / "queue.json")["order"]
    results, summaries, claims = (args.out / name for name in ("results", "summaries", "claims"))
    receipt = {
        "worker": args.name,
        "pid": os.getpid(),
        "nice": os.nice(0),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "started_utc": utc(),
        "counts": Counter(),
        "proposals": 0,
        "failures": [],
    }
    path = args.out / f"workers/{args.name}-{os.getpid()}.json"
    API.save(path, receipt)
    for entry in order:
        if (args.out / "STOP").exists():
            receipt["stopped"] = "STOP file present"
            break
        sid = entry["source_id"]
        target = results / f"{sid}.json"
        if target.exists() or not claim(claims, sid, args.name):
            continue
        start = time.monotonic()
        try:
            search_source(root, entry, target, policy, start)
        except Exception as error:  # noqa: BLE001 - one source must never stop the survey
            if not target.exists():
                API.save(target, failure_result("UNEXPECTED_ERROR", error, entry, start, policy))
            receipt["failures"].append({"source_id": sid, "error": f"{type(error).__name__}: {error}"})
        summary = summarize(entry, API.read(target), args.name)
        API.save(summaries / f"{sid}.json", summary, indent=None)
        receipt["counts"][summary["status"]] += 1
        receipt["proposals"] += len(summary["proposals"])
        line = {
            "source_id": sid,
            "g": summary["phot_g_mean_mag"],
            "status": summary["status"],
            "proposals": len(summary["proposals"]),
            "search_seconds": summary["search_seconds"],
            "elapsed_seconds": round(time.monotonic() - start, 1),
        }
        receipt["last"] = line | {"utc": utc()}
        API.save(path, receipt)
        print(json.dumps(line), flush=True)
    receipt["ended_utc"] = utc()
    API.save(path, receipt)
    return 1 if receipt["failures"] else 0


def read_summaries(out):
    return {path.stem: API.read(path) for path in (out / "summaries").glob("*.json")}


def status(args):
    out = args.out
    order = API.read(out / "queue.json")["order"]
    position = {entry["source_id"]: index for index, entry in enumerate(order)}
    done = {path.stem: path.stat() for path in (out / "results").glob("*.json")}
    evidence = sum(path.stat().st_size for path in (out / "results").glob("*.npz"))
    summaries = read_summaries(out)
    now = time.time()
    recent = {}
    for minutes in (20, 60):
        window = [
            summaries[sid]
            for sid, stat in done.items()
            if now - stat.st_mtime <= 60 * minutes and sid in summaries
        ]
        searched = [entry["search_seconds"] for entry in window if entry["search_seconds"] is not None]
        recent[f"last_{minutes}_minutes"] = {
            "completed": len(window),
            "median_search_seconds": round(median(searched), 1) if searched else None,
            "median_elapsed_seconds": round(median(e["elapsed_seconds"] for e in window), 1) if window else None,
        }
    hour = recent["last_60_minutes"]["completed"]
    workers = [API.read(path) for path in (out / "workers").glob("*.json")]
    live = sorted(entry["worker"] for entry in workers if "ended_utc" not in entry and alive(entry["pid"]))
    stale = []
    for path in (out / "claims").glob("*.claim"):
        if path.stem not in done:
            entry = API.read(path)
            if not alive(entry["pid"]):
                stale.append({"source_id": path.stem, "worker": entry["worker"], "utc": entry["utc"]})
    deepest = max((position[sid] for sid in done if sid in position), default=-1)
    remaining = len(order) - len(done)
    report = {
        "utc": utc(),
        "queued": len(order),
        "completed": len(done),
        "remaining": remaining,
        "reached_g": float(order[deepest]["phot_g_mean_mag"]) if deepest >= 0 else None,
        "by_status": dict(Counter(entry["status"] for entry in summaries.values())),
        "sources_with_proposals": sum(1 for entry in summaries.values() if entry["proposals"]),
        "proposals": sum(len(entry["proposals"]) for entry in summaries.values()),
        "recent": recent,
        "projected_hours_remaining": round(remaining / hour, 1) if hour else None,
        "live_workers": live,
        "stale_claims": stale,
        "failures": sum(len(entry["failures"]) for entry in workers),
        "results_gib": round(sum(stat.st_size for stat in done.values()) / 1024**3, 2),
        "evidence_gib": round(evidence / 1024**3, 2),
        "disk_free_gib": round(shutil.disk_usage(out).free / 1024**3, 1),
        "load_average": [round(value, 1) for value in os.getloadavg()],
        "stop_requested": (out / "STOP").exists(),
    }
    print(json.dumps(report, indent=2))


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def catalogue(args):
    out = args.out
    order = {entry["source_id"]: entry for entry in API.read(out / "queue.json")["order"]}
    summaries = read_summaries(out)
    for path in (out / "results").glob("*.json"):
        if path.stem not in summaries:
            summaries[path.stem] = summarize(order[path.stem], API.read(path), "catalogue")
            API.save(out / "summaries" / path.name, summaries[path.stem], indent=None)
    sources = sorted(summaries.values(), key=lambda s: (s["phot_g_mean_mag"], s["source_id"]))
    proposals = []
    for entry in sources:
        entry["proposal_count"] = len(entry["proposals"])
        for proposal in entry["proposals"]:
            proposals.append({key: entry[key] for key in QUEUE_FIELDS} | proposal | {"evidence": entry["evidence"]})
    write_csv(out / "sources.csv", SOURCE_FIELDS, sources)
    write_csv(out / "proposals.csv", (*QUEUE_FIELDS, *PROPOSAL_FIELDS, "evidence"), proposals)
    report = {
        "utc": utc(),
        "queued": len(order),
        "completed": len(sources),
        "by_status": dict(Counter(entry["status"] for entry in sources)),
        "by_population": {
            population: {
                "completed": sum(1 for e in sources if e["population"] == population),
                "with_proposals": sum(1 for e in sources if e["population"] == population and e["proposals"]),
            }
            for population in sorted({entry["population"] for entry in sources})
        },
        "sources_with_proposals": sum(1 for entry in sources if entry["proposals"]),
        "proposals": len(proposals),
        "search_cpu_hours": round(sum(e["search_cpu_seconds"] or 0 for e in sources) / 3600, 1),
        "files": {
            name: API.digest(out / name) for name in ("sources.csv", "proposals.csv")
        },
    }
    API.save(out / "catalogue.json", report)
    print(json.dumps(report, indent=2))


def reclaim(args):
    out = args.out
    removed = []
    for path in sorted((out / "claims").glob("*.claim")):
        sid = path.stem
        target = out / "results" / f"{sid}.json"
        if target.exists() or alive(API.read(path)["pid"]):
            continue
        removed.extend(unlink(path, target.with_suffix(".npz"), out / "summaries" / f"{sid}.json"))
    if args.failed:
        for sid, entry in read_summaries(out).items():
            if entry["status"] == "UNEXPECTED_ERROR":
                target = out / "results" / f"{sid}.json"
                removed.extend(
                    unlink(target, target.with_suffix(".npz"), out / "summaries" / f"{sid}.json", out / "claims" / f"{sid}.claim")
                )
    print(json.dumps({"removed": removed}, indent=2))


def unlink(*paths):
    removed = []
    for path in paths:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, run in (("prepare", prepare), ("worker", worker), ("status", status), ("catalogue", catalogue), ("reclaim", reclaim)):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, required=True)
        command.set_defaults(run=run)
        if name == "prepare":
            command.add_argument("--targets", type=Path, required=True)
            command.add_argument("--root", type=Path, required=True)
            command.add_argument("--design", type=Path, default=HERE.parent / "shallow/current-designs.json")
        if name == "worker":
            command.add_argument("--name", required=True)
        if name == "reclaim":
            command.add_argument("--failed", action="store_true", help="also clear UNEXPECTED_ERROR results for a retry")
    args = parser.parse_args()
    sys.exit(args.run(args) or 0)


if __name__ == "__main__":
    main()
