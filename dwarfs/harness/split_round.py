"""Run one evolution.py round as independent single-thread workers.

The round's cohort checks, source snapshot, job receipts, runtime identity and
scorer are reused unchanged; only the executor differs: each worker is one
process that claims jobs from a shared queue and runs scan_job for them
sequentially, so workers can start, stop and rejoin at any time.
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / "shallow"
sys.path.insert(0, str(CODE))
import evolution as API


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def completed(target):
    if not target.exists():
        return False
    try:
        result = API.read(target)
    except Exception:
        return False
    arrays = result.get("arrays")
    return bool(
        result.get("status") in ("SEARCHED", "DETECTOR_INELIGIBLE")
        and arrays
        and API.digest(target.with_suffix(".npz")) == arrays["sha256"]
    )


def claim(folder, key, name):
    path = folder / f"{key}.claim"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"worker": name, "pid": os.getpid(), "utc": utc()}, handle)
    return True


def reference_seconds(folder):
    identity = API.read(folder / "identity.json")
    (entries,) = identity["blind_results"].values()
    return {
        entry["key"]: API.read(entry["result"])["search_seconds"] for entry in entries
    }


def prepare(args):
    cases, cohort_bindings = API.cohort_inputs(args.cohort)
    designs = API.read(args.designs)
    if len(designs) != 1 or any(
        key in designs[0] for key in ("seed_prefix", "reuse_results")
    ):
        raise ValueError("One fresh design without prefix or reuse required")
    mode = API.round_mode(cases, designs, args.golden, False, args.sequential, False)
    if args.out.exists():
        raise ValueError("Fresh round directory required")
    jobs = API.jobs_for(cases, args.cohort)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    API.capacity(args.out.parent, args.reserve_gib * 1024**3)
    args.out.mkdir()
    bindings = API.snapshot(args.out, designs)
    seconds = reference_seconds(args.reference) if args.reference else {}
    order = sorted((job["key"] for job in jobs), key=lambda key: -seconds.get(key, 0.0))
    identity = {
        "status": "RUNNING",
        "mode": mode,
        "cohort": str(args.cohort.resolve()),
        "cohort_bindings": cohort_bindings,
        "source_bindings": bindings,
        "runtime": API.runtime_identity(),
        "cases": len(cases),
        "jobs_per_design": len(jobs),
        "workers": "single-thread workers on a claimed queue",
        "designs": designs,
        "admitted_bytes": args.reserve_gib * 1024**3,
        "blind_results": {},
        "result_reuse": {},
        "execution": {
            "driver": str(Path(__file__).resolve()),
            "driver_sha256": API.digest(Path(__file__).resolve()),
            "queue_order": "reference round search_seconds, longest first",
            "reference_round": str(args.reference) if args.reference else None,
            "prepared_utc": utc(),
        },
    }
    API.save(args.out / "identity.json", identity)
    API.save(args.out / "jobs.json", jobs)
    API.save(
        args.out / "queue.json",
        {
            "order": order,
            "reference_seconds": seconds,
            "reference_hours_total": sum(seconds.get(key, 0.0) for key in order) / 3600,
            "unreferenced_jobs": [key for key in order if key not in seconds],
        },
    )
    print(
        json.dumps(
            {
                "mode": mode,
                "cases": len(cases),
                "jobs": len(jobs),
                "reference_hours_total": round(
                    sum(seconds.get(key, 0.0) for key in order) / 3600, 2
                ),
            }
        )
    )


def worker(args):
    identity = API.read(args.out / "identity.json")
    (design,) = identity["designs"]
    source = args.out / "source/dwarfs/shallow"
    API.check_bindings(identity["source_bindings"])
    if API.runtime_identity() != identity["runtime"]:
        raise ValueError("Numerical runtime differs from the round identity")
    API.initialize(source, design["engine"], design["config"])
    jobs = {job["key"]: job for job in API.read(args.out / "jobs.json")}
    order = API.read(args.out / "queue.json")["order"]
    folder = args.out / design["name"]
    folder.mkdir(exist_ok=True)
    receipt = {
        "worker": args.name,
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "started_utc": utc(),
        "jobs": [],
        "failures": [],
    }
    path = args.out / f"worker-{args.name}.json"
    API.save(path, receipt)
    for key in order:
        target = folder / f"{key}.json"
        if completed(target) or not claim(folder, key, args.name):
            continue
        start = time.monotonic()
        try:
            API.scan_job((jobs[key], target))
            result = API.read(target)
            entry = {
                "key": key,
                "status": result["status"],
                "proposals": len(result["proposals"]),
                "search_seconds": result["search_seconds"],
                "elapsed_seconds": time.monotonic() - start,
            }
        except RuntimeError as error:
            entry = {"key": key, "status": "FAILED", "error": str(error)}
            receipt["failures"].append(key)
        receipt["jobs"].append(entry)
        API.save(path, receipt)
        print(json.dumps(entry), flush=True)
    receipt["ended_utc"] = utc()
    API.save(path, receipt)
    return 1 if receipt["failures"] else 0


def reclaim(args):
    identity = API.read(args.out / "identity.json")
    (design,) = identity["designs"]
    folder = args.out / design["name"]
    for key in args.keys.split(","):
        target = folder / f"{key}.json"
        if completed(target):
            raise ValueError(f"Completed job must not be reclaimed: {key}")
        for path in (folder / f"{key}.claim", target, target.with_suffix(".npz")):
            if path.exists():
                path.unlink()
                print(json.dumps({"removed": str(path)}))


def finish(args):
    identity = API.read(args.out / "identity.json")
    if identity["status"] != "RUNNING":
        raise ValueError("Only a running split round can be finished")
    (design,) = identity["designs"]
    folder = args.out / design["name"]
    jobs = API.read(args.out / "jobs.json")
    entries, missing = [], []
    for job in sorted(jobs, key=lambda row: row["key"]):
        target = folder / f"{job['key']}.json"
        if not completed(target):
            missing.append(job["key"])
            continue
        entries.append(
            {"key": job["key"], "result": str(target), "sha256": API.digest(target)}
        )
    if missing:
        raise ValueError(f"{len(missing)} jobs incomplete: {missing[:5]}")
    for path in folder.glob("*.claim"):
        if not completed(path.with_suffix(".json")):
            raise ValueError(f"Stale claim without result: {path.name}")
    API.check_bindings(identity["source_bindings"] | identity["cohort_bindings"])
    if API.runtime_identity() != identity["runtime"]:
        raise ValueError("Numerical runtime changed during the round")
    receipts = sorted(args.out.glob("worker-*.json"))
    identity["execution"]["workers"] = [API.read(path) for path in receipts]
    identity["execution"]["finished_utc"] = utc()
    identity["blind_results"][design["name"]] = entries
    identity["status"] = "BLIND_SEARCH_COMPLETE_UNSCORED"
    API.save(args.out / "identity.json", identity)
    print(json.dumps({"jobs": len(entries), "status": identity["status"]}))


def identical_arrays(produced, saved):
    with API.np.load(produced, allow_pickle=False) as a, API.np.load(
        saved, allow_pickle=False
    ) as b:
        if set(a.files) != set(b.files):
            return False, sorted(set(a.files) ^ set(b.files))
        different = [
            name
            for name in a.files
            if a[name].dtype != b[name].dtype
            or a[name].shape != b[name].shape
            or a[name].tobytes() != b[name].tobytes()
        ]
        return not different, different


def comparison(produced_path, saved_path, label):
    produced, stored = API.read(produced_path), API.read(saved_path)
    same_arrays, different = identical_arrays(
        Path(produced_path).with_suffix(".npz"), Path(saved_path).with_suffix(".npz")
    )
    report = {
        "label": label,
        "status": produced["status"],
        "saved_status": stored["status"],
        "proposals": len(produced["proposals"]),
        "saved_proposals": len(stored["proposals"]),
        "proposals_identical": json.dumps(produced["proposals"], sort_keys=True)
        == json.dumps(stored["proposals"], sort_keys=True),
        "arrays_identical": same_arrays,
        "different_arrays": different[:10],
        "search_seconds": produced["search_seconds"],
        "saved_search_seconds": stored["search_seconds"],
    }
    print(json.dumps(report), flush=True)
    return report


def compare(args):
    report = comparison(args.produced, args.saved, args.label)
    API.save(Path(args.produced).parent / "replay.json", report)
    return 0 if report["proposals_identical"] and report["arrays_identical"] else 1


def replay(args):
    cases, _ = API.cohort_inputs(args.cohort)
    (design,) = API.read(args.designs)
    API.initialize(CODE, design["engine"], design["config"])
    reference = API.read(args.reference / "identity.json")
    (saved_entries,) = reference["blind_results"].values()
    saved = {entry["key"]: entry["result"] for entry in saved_entries}
    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case_id in args.cases.split(","):
        row = next(r for r in cases if r["case_id"] == case_id)
        (job,) = API.jobs_for([row], args.cohort)
        folder = args.out / job["key"]
        folder.mkdir(exist_ok=True)
        receipt = API.scan_job((job, folder / f"{job['key']}.json"))
        report = comparison(receipt["result"], saved[job["key"]], case_id)
        API.save(folder / "replay.json", report)
        failures += not (report["proposals_identical"] and report["arrays_identical"])
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--designs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reserve-gib", type=int, required=True)
    p.add_argument("--reference", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--golden", action="store_true")
    mode.add_argument("--sequential", action="store_true")
    w = commands.add_parser("worker")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--name", required=True)
    c = commands.add_parser("reclaim")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--keys", required=True)
    f = commands.add_parser("finish")
    f.add_argument("--out", type=Path, required=True)
    k = commands.add_parser("compare")
    k.add_argument("--produced", required=True)
    k.add_argument("--saved", required=True)
    k.add_argument("--label", required=True)
    r = commands.add_parser("replay")
    r.add_argument("--cohort", type=Path, required=True)
    r.add_argument("--designs", type=Path, required=True)
    r.add_argument("--reference", type=Path, required=True)
    r.add_argument("--cases", required=True)
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    handlers = {
        "prepare": prepare,
        "worker": worker,
        "reclaim": reclaim,
        "finish": finish,
        "compare": compare,
        "replay": replay,
    }
    raise SystemExit(handlers[args.command](args) or 0)


if __name__ == "__main__":
    main()
