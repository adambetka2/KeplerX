"""Launch the fixed 40-run census with ten independent Mac workers."""

import argparse
import concurrent.futures
import datetime
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline.vetting import statistical_validation as driver
from pipeline.vetting.validation_results import result_directory, validate_result

WORKERS = 10


def jobs(plan):
    # Longer 210-family jobs go first to reduce the tail of the overnight queue.
    return [
        (target_index, target, index)
        for target_index, target in enumerate(plan["targets"])
        for index in range(driver.RUNS_PER_TARGET)
    ]


def command(source_id, index):
    return [
        sys.executable,
        "-m",
        "pipeline.vetting.statistical_validation",
        "run",
        "--source-id",
        source_id,
        "--run-index",
        str(index),
    ]


def inventory(plan, receipt):
    complete, pending = [], []
    for target_index, target, index in jobs(plan):
        result = result_directory(plan, target["source_id"], index) / "result.json"
        if result.exists():
            validate_result(result, target, target_index, index, plan, receipt, False)
            complete.append(str(result))
        else:
            pending.append((target_index, target, index))
    return complete, pending


def execute(job, base, environment, attempt):
    _, target, index = job
    log = base / "logs" / f"{target['source_id']}-{index:02d}-{attempt}.log"
    with log.open("x") as stream:
        completed = subprocess.run(
            command(target["source_id"], index),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "source_id": target["source_id"],
        "run_index": index,
        "exit_code": completed.returncode,
        "log": str(log),
    }


def run_queue(pending, base, environment, attempt, state):
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        queue = iter(pending)
        active = {
            pool.submit(execute, job, base, environment, attempt)
            for job in [next(queue, None) for _ in range(WORKERS)]
            if job is not None
        }
        while active:
            done, active = concurrent.futures.wait(
                active, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                row = future.result()
                state["completed_jobs"].append(row)
                if row["exit_code"] and state["first_failure"] is None:
                    state["first_failure"] = row
                print(json.dumps(row), flush=True)
            if state["first_failure"] is None:
                for _ in done:
                    job = next(queue, None)
                    if job is not None:
                        active.add(
                            pool.submit(execute, job, base, environment, attempt)
                        )
            driver.write_json(base / "campaign-status.json", state)


def launch(plan, receipt, resume):
    base = Path(plan["output_root"])
    complete, pending = inventory(plan, receipt)
    if not resume and (complete or (base / "runs").exists()):
        raise ValueError("Existing campaign requires explicit resume")
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        NUMBA_CACHE_DIR=str(base.parent / "cache/numba"),
        MPLCONFIGDIR=str(base.parent / "cache/mpl"),
        XDG_CACHE_HOME=str(base.parent / "cache"),
        TMPDIR=str(base.parent / "tmp"),
    )
    for key in (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[key] = "1"
    (base / "logs").mkdir(exist_ok=True)
    attempt = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    state = {
        "attempt": attempt,
        "status": "RUNNING",
        "workers": WORKERS,
        "required_runs": 40,
        "already_complete": complete,
        "completed_jobs": [],
        "first_failure": None,
    }
    driver.write_json(base / "campaign-status.json", state)
    keep_awake = subprocess.Popen(["caffeinate", "-is", "-w", str(os.getpid())])
    try:
        run_queue(pending, base, environment, attempt, state)
        if state["first_failure"] is not None:
            raise RuntimeError("Worker failed; retained checkpoints and first failure")
        with (base / "logs" / f"deep-audit-{attempt}.log").open("x") as stream:
            subprocess.run(
                [sys.executable, "-m", "pipeline.vetting.validation_results", "--deep"],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = json.loads((base / "numerical-campaign-summary.json").read_text())
        if summary["complete_runs"] != 40 or not summary["deep_weight_replay"]:
            raise ValueError("Deep audit did not certify the complete 40-run census")
        state["status"] = "COMPLETE_NUMERICAL_CAMPAIGN_PENDING_PHYSICAL_AUDIT"
    except BaseException as error:
        state["status"] = "STOPPED_NO_ADMISSION"
        state["error"] = repr(error)
        raise
    finally:
        driver.write_json(base / "campaign-status.json", state)
        driver.write_json(base / "audit" / f"campaign-attempt-{attempt}.json", state)
        keep_awake.terminate()
        keep_awake.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "start", "resume"))
    args = parser.parse_args()
    plan, _ = driver.verify()
    base = Path(plan["output_root"])
    receipt = base / "freeze.json"
    if args.action == "status":
        complete, pending = inventory(plan, receipt)
        print(
            json.dumps(
                {"complete": len(complete), "pending": len(pending), "workers": WORKERS}
            )
        )
        return
    with (base / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch(plan, receipt, args.action == "resume")


if __name__ == "__main__":
    main()
