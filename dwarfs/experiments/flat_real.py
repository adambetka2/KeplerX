"""Conditionally launch the frozen local real-source Flat comparison after SAP scoring."""

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flat_retrieval as pilot

from pipeline.photometry.lc2 import read_source_rows, source_row

API = pilot.API
require = pilot.require
ROOT = pilot.ROOT


def utc():
    return datetime.now(UTC).isoformat()


def decision(progress, report):
    if progress.get("stage") != "report" or progress.get("state") != "COMPLETE":
        return {"status": "WAITING_COMPARISON"}
    require(
        report is not None and report["status"] == "SCORED_SAP_VERSUS_SAVED_FLAT",
        "Completed scored comparison required",
    )
    sap, flat = (report["summaries"][name] for name in ("sap", "flat_reused"))
    require(
        sap["planets"] == flat["planets"] == 700
        and sap["controls"] == flat["controls"] == 300,
        "Exact 700-signal and 300-control denominators required",
    )
    require(flat["recovered"] == 163, "Frozen adopted-detector Flat reference differs")
    passed = 20 * flat["recovered"] >= 23 * sap["recovered"]
    return {
        "status": "QUALIFIES" if passed else "CRITERION_NOT_MET",
        "flat_recovered": flat["recovered"],
        "sap_recovered": sap["recovered"],
        "signals": 700,
        "relative_gain": flat["recovered"] / sap["recovered"] - 1
        if sap["recovered"]
        else None,
        "criterion": "Flat recovery >= 1.15 * SAP recovery; integer comparison 20*Flat >= 23*SAP",
        "flat_errors": flat["statuses"].get("UNEXPECTED_ERROR", 0),
        "sap_errors": sap["statuses"].get("UNEXPECTED_ERROR", 0),
    }


def shard_folder(root, row):
    shard = int(row["shard"])
    volume = "discSSD" if shard % 2 == 0 else "discPCIe"
    return root / volume / f"data/kbonus/lc2/shard-{shard:04d}"


def selection(policy):
    API.check_bindings(policy["bindings"])
    rows = API.rows(policy["selection"])
    require(
        len(rows) == len({row["source_id"] for row in rows}) == 5000,
        "Exact frozen 5000-source selection required",
    )
    require(
        Counter(row["population"] for row in rows) == Counter({"K": 4753, "M": 247}),
        "Frozen M/K selection differs",
    )
    return rows


def availability(rows, root):
    missing = [
        row["source_id"]
        for row in rows
        if not all(
            (shard_folder(root, row) / name).is_file()
            for name in ("cadence.parquet", "source.parquet")
        )
    ]
    return {
        "selected": len(rows),
        "available": len(rows) - len(missing),
        "missing": len(missing),
        "missing_source_ids": missing,
    }


def check(policy):
    folder = Path(policy["comparison"])
    progress = API.read(folder / "progress.json")
    report = (
        API.read(folder / "retrieval-report.json")
        if (folder / "retrieval-report.json").exists()
        else None
    )
    result = decision(progress, report)
    if result["status"] != "QUALIFIES":
        return result
    rows = selection(policy)
    data = availability(rows, Path(policy["root"]))
    return result | {
        "status": "WAITING_LOCAL_DATA" if data["missing"] else "READY",
        "data": data,
    }


def native_records(root, row):
    folder = shard_folder(root, row)
    columns = [
        "source_id",
        "time_bkjd",
        "cadenceno",
        "quarter",
        "psf_flat_kbonus_flux",
        "psf_flat_kbonus_flux_err",
        "sap_quality",
        "flatten_mask",
    ]
    table = read_source_rows(folder / "cadence.parquet", int(row["source_id"]), columns)
    curve = pilot.photometry(
        {key: table[key].to_numpy() for key in columns}, "psf_flat"
    )
    records = pilot.prepare_curve_records(curve, mask_flatten=True)
    measured = {
        "raw_fits_sha256": source_row(folder, int(row["source_id"]))["raw_fits_sha256"],
        "curve_sha256": pilot.curve_digest(curve),
        "flatten_sha256": hashlib.sha256(curve["flatten"].tobytes()).hexdigest(),
        "stored_cadences": len(curve["time"]),
        "prepared_cadences": len(records),
        "records_sha256": hashlib.sha256(records.tobytes()).hexdigest(),
        "retained_cadence_sha256": hashlib.sha256(
            records["cadence"].tobytes()
        ).hexdigest(),
    }
    differences = {
        key: {"expected": row[key], "measured": value}
        for key, value in measured.items()
        if str(value) != str(row[key])
    }
    require(
        not differences,
        f"Frozen local Flat input differs: {row['source_id']}: {differences}",
    )
    return curve, records, measured


def prepare_one(task):
    root, out, row = task
    curve, records, measured = native_records(root, row)
    path = out / "prepared" / (row["source_id"] + ".npz")
    require(not path.exists(), "Prepared source already exists")
    np.savez_compressed(
        path,
        records=records,
        retained_mask=np.isin(curve["cadence"], records["cadence"]),
        **{"raw_" + key: value for key, value in curve.items()},
    )
    sha = API.digest(path)
    API.save(
        path.with_suffix(".json"),
        {
            "source_id": row["source_id"],
            "status": "PREPARED",
            "first_failure": None,
            "provenance": row,
            "verified": measured,
            "archive_sha256": sha,
        },
    )
    return row | {
        "case_id": "real_psf_flat:" + row["source_id"],
        "preparation_status": "PREPARED",
        "first_failure": "",
        "input_path": str(path),
        "input_sha256": sha,
        "array_key": "records",
        "receipt_path": str(path.with_suffix(".json")),
        "receipt_sha256": API.digest(path.with_suffix(".json")),
    }


def prepare(args):
    policy = API.read(args.policy)
    rows = selection(policy)
    out = Path(policy["out"])
    require(
        not (out / "prepared-cases.json").exists(), "Frozen real inputs already exist"
    )
    (out / "prepared").mkdir()
    bindings = policy["bindings"] | API.bind_files(
        [args.policy, Path(__file__), Path(pilot.__file__)]
    )
    API.save(
        out / "preparation-identity.json",
        {
            "bindings": bindings,
            "runtime": API.runtime_identity(),
            "view": "psf_flat",
            "mask_flatten": True,
        },
    )
    tasks = [(Path(policy["root"]), out, row) for row in rows]
    cases = []
    with ProcessPoolExecutor(
        policy["workers"], mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for case in pool.map(prepare_one, tasks, chunksize=1):
            cases.append(case)
            if len(cases) % 100 == 0:
                print(json.dumps({"prepared_sources": len(cases)}), flush=True)
    API.check_bindings(bindings)
    API.save(out / "prepared-cases.json", cases)


def catalogue(args):
    policy = API.read(args.policy)
    out = Path(policy["out"])
    identity = API.read(out / "pilot/identity.json")
    require(
        identity["status"] == "BLIND_SEARCH_COMPLETE_UNSCORED",
        "All real searches must finish",
    )
    API.check_bindings(
        identity["source_bindings"]
        | identity["case_bindings"]
        | identity["result_bindings"]
    )
    design = identity["designs"][0]
    rows = []
    for case in API.read(out / "prepared-cases.json"):
        path = out / "pilot" / design["name"] / (API.job_key(case) + ".json")
        result = API.read(path)
        rows.append(
            {
                key: case[key]
                for key in (
                    "source_id",
                    "population",
                    "phot_g_mean_mag",
                    "shard",
                    "server_queue_rank",
                )
            }
            | {
                "status": result["status"],
                "first_failure": result["first_failure"],
                "proposals": result.get("proposals", []),
                "result_path": str(path),
                "search_seconds": result.get("search_seconds"),
            }
        )
    require(len(rows) == 5000, "All frozen 5000 sources required")
    API.save(out / "sources.json", rows)
    API.save(
        out / "catalogue.json",
        {
            "status": "COMPLETE_REAL_PSF_FLAT_PROPOSAL_RUN",
            "completed_at": utc(),
            "sources": len(rows),
            "statuses": dict(Counter(row["status"] for row in rows)),
            "sources_with_proposals": sum(bool(row["proposals"]) for row in rows),
            "proposals": sum(len(row["proposals"]) for row in rows),
            "by_population": {
                population: {
                    "sources": len(group),
                    "with_proposals": sum(bool(row["proposals"]) for row in group),
                    "proposals": sum(len(row["proposals"]) for row in group),
                }
                for population in ("M", "K")
                for group in [[row for row in rows if row["population"] == population]]
            },
            "sources_sha256": API.digest(out / "sources.json"),
            "limits": [
                "Unvetted proposals are not confirmed planets or a measured recall",
                "Bright-first subset: 4753 K and 247 M sources",
                "Conditional 5000-source experiment does not pass the programme 10000-case gate",
            ],
        },
    )


def execute(args):
    policy = API.read(args.policy)
    out = Path(policy["out"])
    API.check_bindings(API.read(out / "plan.json")["bindings"])
    require(
        check(policy)["status"] == "READY",
        "Completed qualifying comparison and all local data required",
    )
    for stage in ("prepare", "run", "catalogue"):
        API.save(
            out / "progress.json",
            {"stage": stage, "state": "RUNNING", "workers": policy["workers"]},
        )
        try:
            if stage == "run":
                pilot.run(SimpleNamespace(out=out, workers=policy["workers"]))
            else:
                globals()[stage](args)
        except Exception:
            API.save(
                out / "progress.json",
                {"stage": stage, "state": "FAILED", "workers": policy["workers"]},
            )
            raise
    API.save(
        out / "progress.json",
        {"stage": "catalogue", "state": "COMPLETE", "workers": policy["workers"]},
    )


def launch(args):
    policy = API.read(args.policy)
    out = Path(policy["out"])
    if out.exists():
        return {
            "status": "ALREADY_STARTED",
            "out": str(out),
            "launch": API.read(out / "launch.json")
            if (out / "launch.json").exists()
            else None,
        }
    result = check(policy)
    if result["status"] != "READY":
        return result
    comparison = Path(policy["comparison"])
    identity = API.read(comparison / "pilot/identity.json")
    API.check_bindings(
        identity["source_bindings"]
        | identity["case_bindings"]
        | identity["result_bindings"]
    )
    require(
        identity["runtime"] == API.runtime_identity(),
        "Frozen comparison numerical runtime differs",
    )
    require(
        len(identity["designs"]) == 1 and identity["designs"][0] == policy["design"],
        "Unchanged adopted detector required",
    )
    API.capacity(out.parent, 32 * 1024**3)
    out.mkdir()
    API.save(
        out / "plan.json",
        {
            "bindings": policy["bindings"]
            | API.bind_files(
                [
                    args.policy,
                    comparison / "retrieval-report.json",
                    comparison / "pilot/identity.json",
                ]
            ),
            "trigger": result,
            "comparison": str(comparison),
            "sources": 5000,
        },
    )
    API.save(out / "designs.json", [policy["design"]])
    env = os.environ | {
        "UV_CACHE_DIR": str(out.parent / "cache/uv"),
        "NUMBA_CACHE_DIR": str(out / "coordinator-cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    command = [
        "/opt/homebrew/bin/uv",
        "run",
        "--no-project",
        "--python",
        policy["python"],
        "python",
        "-B",
        str(Path(__file__).resolve()),
        "execute",
        "--policy",
        str(args.policy),
    ]
    with (out / "execution.log").open("xb") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    wake = subprocess.Popen(
        ["/usr/bin/caffeinate", "-i", "-w", str(process.pid)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    receipt = {
        "status": "LAUNCHED",
        "out": str(out),
        "started_at": utc(),
        "pid": process.pid,
        "process_group": os.getpgid(process.pid),
        "nice": os.getpriority(os.PRIO_PROCESS, process.pid),
        "workers": policy["workers"],
        "caffeinate_pid": wake.pid,
        "trigger": result,
        "policy_sha256": API.digest(args.policy),
    }
    API.save(out / "launch.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "launch", "execute"))
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    args.policy = args.policy.resolve()
    if args.stage == "check":
        value = check(API.read(args.policy))
    else:
        value = globals()[args.stage](args)
    if value is not None:
        print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
