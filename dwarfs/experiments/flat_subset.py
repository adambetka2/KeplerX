"""Search an explicitly authorized subset of the frozen, internally prepared Flat sample."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flat_local as local

API, pilot, real, require = local.API, local.pilot, local.real, local.require


def inherited_cases(policy):
    API.check_bindings(policy["bindings"])
    parent = API.read(policy["parent_policy"])
    require(policy["design"] == parent["design"], "Adopted detector differs")
    original = {row["source_id"]: row for row in local.selection(parent)}
    selected = API.rows(policy["selection"])
    require(
        len(selected)
        == len({row["source_id"] for row in selected})
        == policy["sources"],
        "Frozen partial sample differs",
    )
    cases = []
    for row in selected:
        require(
            original.get(row["source_id"]) == row,
            "Source is outside original frozen sample",
        )
        receipt_path = Path(policy["prepared_directory"]) / (row["source_id"] + ".json")
        receipt = API.read(receipt_path)
        require(receipt["frozen_selection"] == row, "Prepared source selection differs")
        case = receipt["case"]
        require(case["preparation_status"] == "PREPARED", "Prepared source required")
        API.check_bindings({case["input_path"]: case["input_sha256"]})
        with np.load(case["input_path"], allow_pickle=False) as stored:
            records = stored[case["array_key"]]
            require(
                len(records) == int(row["prepared_cadences"])
                and hashlib.sha256(records.tobytes()).hexdigest()
                == row["records_sha256"],
                "Inherited Flat records differ",
            )
        cases.append(
            case
            | {
                "receipt_path": str(receipt_path),
                "receipt_sha256": API.digest(receipt_path),
            }
        )
    return cases


def catalogue(out):
    identity = API.read(out / "pilot/identity.json")
    require(
        identity["status"] == "BLIND_SEARCH_COMPLETE_UNSCORED",
        "All searches must finish",
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
    API.save(out / "sources.json", rows)
    report = {
        "status": "COMPLETE_REAL_PSF_FLAT_PROPOSAL_RUN",
        "completed_at": real.utc(),
        "sources": len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "sources_with_proposals": sum(bool(row["proposals"]) for row in rows),
        "proposals": sum(len(row["proposals"]) for row in rows),
        "by_population": {
            population: {
                "sources": len(group),
                "with_proposals": sum(bool(row["proposals"]) for row in group),
                "proposals": sum(len(row["proposals"]) for row in group),
                "statuses": dict(Counter(row["status"] for row in group)),
            }
            for population in ("M", "K")
            for group in [[row for row in rows if row["population"] == population]]
        },
        "sources_sha256": API.digest(out / "sources.json"),
        "limits": [
            "Unvetted proposals are not confirmed planets or measured recall",
            "Availability-selected subset of the frozen bright-first 5000; unavailable sources are not searched",
        ],
    }
    API.save(out / "catalogue.json", report)


def execute(args):
    policy = API.read(args.policy)
    out = Path(policy["out"])
    API.check_bindings(API.read(out / "plan.json")["bindings"])
    for stage in ("run", "catalogue"):
        API.save(
            out / "progress.json",
            {
                "stage": stage,
                "state": "RUNNING",
                "workers": policy["workers"],
                "sources": policy["sources"],
            },
        )
        try:
            if stage == "run":
                pilot.run(SimpleNamespace(out=out, workers=policy["workers"]))
            else:
                catalogue(out)
        except Exception:
            failure = {
                "stage": stage,
                "traceback": traceback.format_exc(),
                "at": real.utc(),
            }
            API.save(out / "first-failure.json", failure)
            API.save(
                out / "progress.json",
                {"stage": stage, "state": "FAILED", "first_failure": failure},
            )
            raise
    API.save(
        out / "progress.json",
        {
            "stage": "catalogue",
            "state": "COMPLETE",
            "workers": policy["workers"],
            "sources": policy["sources"],
        },
    )


def launch(args):
    policy = API.read(args.policy)
    out = Path(policy["out"])
    if out.exists():
        return {"status": "ALREADY_STARTED", "out": str(out)}
    cases = inherited_cases(policy)
    preparation = API.read(policy["preparation_identity"])
    require(
        preparation["runtime"] == API.runtime_identity(),
        "Prepared numerical runtime differs",
    )
    API.capacity(out.parent, 32 * 1024**3)
    out.mkdir()
    bindings = policy["bindings"] | API.bind_files(
        [args.policy, Path(policy["preparation_identity"])]
    )
    API.save(
        out / "plan.json",
        {
            "bindings": bindings,
            "authorization": policy["authorization"],
            "sources": len(cases),
        },
    )
    API.save(out / "designs.json", [policy["design"]])
    API.save(out / "prepared-cases.json", cases)
    API.save(
        out / "preparation-identity.json",
        preparation
        | {"bindings": bindings, "inheritance": policy["prepared_directory"]},
    )
    local.snapshot(out)
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
        "out": str(out),
        "pid": process.pid,
        "started_at": real.utc(),
        "process_group": os.getpgid(process.pid),
        "nice": os.getpriority(os.PRIO_PROCESS, process.pid),
        "workers": policy["workers"],
        "sources": len(cases),
        "caffeinate_pid": wake.pid,
        "policy_sha256": API.digest(args.policy),
    }
    API.save(out / "launch.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("launch", "execute"))
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    args.policy = args.policy.resolve()
    result = globals()[args.stage](args)
    if result is not None:
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
