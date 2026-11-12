"""Run the directly authorized real Flat sample with outputs independent of corpus disks."""

import argparse
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flat_real as real

API, pilot, require = real.API, real.pilot, real.require


def selection(policy):
    API.check_bindings(policy["bindings"])
    require(
        policy["status"] == "AUTHORIZED_DIRECT_LOCAL_PSF_FLAT_RUN",
        "Direct authorization required",
    )
    rows = API.rows(policy["selection"])
    require(
        len(rows) == len({row["source_id"] for row in rows}) == 5000,
        "Exact 5000 sources required",
    )
    require(
        Counter(row["population"] for row in rows) == {"K": 4753, "M": 247},
        "Frozen M/K sample differs",
    )
    reference = API.read(policy["detector_reference"])
    require(
        policy["design"] == reference["design"],
        "Adopted detector configuration differs",
    )
    require(
        API.digest(policy["selection"]) == reference["bindings"][policy["selection"]],
        "Frozen selection differs",
    )
    return rows


def native_records(root, row):
    folder = real.shard_folder(root, row)
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
    table = real.read_source_rows(
        folder / "cadence.parquet", int(row["source_id"]), columns
    )
    curve = pilot.photometry(
        {key: table[key].to_numpy() for key in columns}, "psf_flat"
    )
    records = pilot.prepare_curve_records(curve, mask_flatten=True)
    require(
        len(records) == int(row["prepared_cadences"])
        and hashlib.sha256(records.tobytes()).hexdigest() == row["records_sha256"],
        f"Frozen Flat records differ: {row['source_id']}",
    )
    provenance = {
        "cadence_path": str((folder / "cadence.parquet").resolve()),
        "raw_fits_sha256": real.source_row(folder, int(row["source_id"]))[
            "raw_fits_sha256"
        ],
        "curve_sha256": pilot.curve_digest(curve),
        "flatten_sha256": hashlib.sha256(curve["flatten"].tobytes()).hexdigest(),
        "stored_cadences": len(curve["time"]),
        "prepared_cadences": len(records),
        "records_sha256": row["records_sha256"],
        "retained_cadence_sha256": hashlib.sha256(
            records["cadence"].tobytes()
        ).hexdigest(),
    }
    return curve, records, provenance


def prepare_one(task):
    root, out, row = task
    path = out / "prepared" / (row["source_id"] + ".npz")
    require(
        not path.exists() and not path.with_suffix(".json").exists(),
        "Prepared source already exists",
    )
    try:
        curve, records, provenance = native_records(root, row)
        API.capacity(out, 3 * 1024**3)
        np.savez_compressed(
            path,
            records=records,
            retained_mask=np.isin(curve["cadence"], records["cadence"]),
            **{"raw_" + key: value for key, value in curve.items()},
        )
        case = row | {
            "case_id": "real_psf_flat:" + row["source_id"],
            "preparation_status": "PREPARED",
            "first_failure": "",
            "input_path": str(path),
            "input_sha256": API.digest(path),
            "array_key": "records",
        }
        receipt = path.with_suffix(".json")
        API.save(
            receipt,
            {"case": case, "frozen_selection": row, "measured_provenance": provenance},
        )
        return case | {
            "receipt_path": str(receipt),
            "receipt_sha256": API.digest(receipt),
        }
    except Exception:
        API.save(
            path.with_suffix(".failed.json"),
            {"source": row, "first_failure": traceback.format_exc()},
        )
        raise


def prepare(args, policy):
    out = Path(policy["out"])
    rows = selection(policy)
    (out / "prepared").mkdir()
    API.save(
        out / "preparation-identity.json",
        {
            "bindings": API.read(out / "plan.json")["bindings"],
            "runtime": API.runtime_identity(),
            "view": "psf_flat",
            "mask_flatten": True,
            "verification": "Every prepared record byte and cadence count match the frozen selection",
            "raw_provenance": "Measured from mounted corpus; original full receipts are on the removed workspace disk",
        },
    )
    cases = {}
    while len(cases) < len(rows):
        pending = [row for row in rows if row["source_id"] not in cases]
        root = Path(policy["root"])
        ready = [row for row in pending if real.availability([row], root)["available"]]
        if ready:
            with ProcessPoolExecutor(
                policy["workers"], mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                tasks = [pool.submit(prepare_one, (root, out, row)) for row in ready]
                for task in as_completed(tasks):
                    case = task.result()
                    cases[case["source_id"]] = case
                    API.save(
                        out / "progress.json",
                        {
                            "stage": "prepare",
                            "state": "RUNNING",
                            "workers": policy["workers"],
                            "prepared": len(cases),
                            "sources": len(rows),
                            "updated_at": real.utc(),
                        },
                    )
        if len(cases) < len(rows):
            API.save(
                out / "progress.json",
                {
                    "stage": "prepare",
                    "state": "WAITING_LOCAL_DATA",
                    "workers": policy["workers"],
                    "prepared": len(cases),
                    "missing": len(rows) - len(cases),
                    "missing_source_ids": [
                        row["source_id"]
                        for row in rows
                        if row["source_id"] not in cases
                    ],
                    "updated_at": real.utc(),
                },
            )
            time.sleep(60)
    API.check_bindings(API.read(out / "plan.json")["bindings"])
    API.save(out / "prepared-cases.json", [cases[row["source_id"]] for row in rows])


def snapshot(out):
    folder = out / "pilot"
    folder.mkdir()
    designs = API.read(out / "designs.json")
    bindings = API.snapshot(folder, designs)
    build = out / "build"
    build.mkdir()
    # Only the new snapshot's private build link changes; the repository dump stays untouched.
    link = folder / "source/dump"
    require(link.is_symlink(), "Expected snapshot build symlink")
    link.unlink()
    link.symlink_to(build, target_is_directory=True)
    cases = API.read(out / "prepared-cases.json")
    jobs = API.jobs_for(cases, out)
    API.save(
        folder / "identity.json",
        {
            "status": "RUNNING",
            "source_bindings": bindings,
            "runtime": API.runtime_identity(),
            "designs": designs,
            "case_bindings": API.bind_files(
                [
                    out / "prepared-cases.json",
                    out / "preparation-identity.json",
                    *[Path(case["receipt_path"]) for case in cases],
                ]
            )
            | {job["path"]: job["sha256"] for job in jobs},
            "driver_sha256": API.digest(pilot.__file__),
            "internal_build_root": str(build),
        },
    )


def catalogue(args):
    real.catalogue(args)
    out = Path(API.read(args.policy)["out"])
    report = API.read(out / "catalogue.json")
    report["authorization"] = "Directly authorized brightest-5000 local comparison"
    report["limits"][-1] = (
        "This 5000-source experiment does not establish the programme 10000-case gate"
    )
    API.save(out / "catalogue.json", report)


def execute(args):
    policy = API.read(args.policy)
    out = Path(policy["out"])
    selection(policy)
    for stage in ("prepare", "run", "catalogue"):
        API.save(
            out / "progress.json",
            {
                "stage": stage,
                "state": "RUNNING",
                "workers": policy["workers"],
                "sources": 5000,
                "updated_at": real.utc(),
            },
        )
        try:
            if stage == "prepare":
                prepare(args, policy)
            elif stage == "run":
                snapshot(out)
                pilot.run(SimpleNamespace(out=out, workers=policy["workers"]))
            else:
                catalogue(args)
        except Exception:
            failure = traceback.format_exc()
            API.save(
                out / "first-failure.json",
                {"stage": stage, "traceback": failure, "at": real.utc()},
            )
            API.save(
                out / "progress.json",
                {
                    "stage": stage,
                    "state": "FAILED",
                    "workers": policy["workers"],
                    "first_failure": failure,
                },
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
        return {"status": "ALREADY_STARTED", "out": str(out)}
    rows = selection(policy)
    API.capacity(out.parent, 32 * 1024**3)
    out.mkdir()
    API.save(
        out / "plan.json",
        {
            "bindings": policy["bindings"] | API.bind_files([args.policy]),
            "authorization": policy["authorization"],
            "sources": len(rows),
            "population_counts": dict(Counter(row["population"] for row in rows)),
            "magnitude_range_g": [
                min(float(row["phot_g_mean_mag"]) for row in rows),
                max(float(row["phot_g_mean_mag"]) for row in rows),
            ],
            "initial_availability": real.availability(rows, Path(policy["root"])),
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
        "NUMBA_NUM_THREADS": "1",
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
            cwd=real.ROOT,
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
        "started_at": real.utc(),
        "pid": process.pid,
        "process_group": os.getpgid(process.pid),
        "nice": os.getpriority(os.PRIO_PROCESS, process.pid),
        "workers": policy["workers"],
        "caffeinate_pid": wake.pid,
        "policy_sha256": API.digest(args.policy),
        "out": str(out),
    }
    API.save(out / "launch.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("launch", "execute"))
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    args.policy = args.policy.resolve()
    value = globals()[args.stage](args)
    if value is not None:
        print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
