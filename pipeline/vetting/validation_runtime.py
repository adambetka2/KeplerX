"""Launch the Mac campaign and replay its retained reference weights."""

import argparse
import cProfile
import ctypes
import datetime
import json
import os
import plistlib
import pstats
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from pipeline.vetting import statistical_validation as driver
from pipeline.vetting.scenario_bounds import LikelihoodBound


def application_qos():
    library = ctypes.CDLL(None)
    apply = library.pthread_set_qos_class_self_np
    apply.argtypes = (ctypes.c_uint, ctypes.c_int)
    apply.restype = ctypes.c_int
    error = apply(0x19, 0)
    if error:
        raise OSError(error, "Cannot apply user-initiated scheduling policy")
    library.qos_class_self.restype = ctypes.c_uint
    if library.qos_class_self() != 0x19:
        raise RuntimeError("User-initiated scheduling policy was not applied")


def replay(index, output):
    plan, _ = driver.verify()
    target = plan["targets"][index]
    root = Path(plan["output_root"]).parent
    reference = root / "inputs/references" / (target["source_id"] + "-tic-original")
    saved = driver.restore(json.loads((reference / "result.json").read_text()))
    rows = saved["operators"]["grid800"]["scenario_evidences"]
    began, cpu = time.monotonic(), time.process_time()
    state = driver.context(plan, target, 256)
    if state[-1] != [row["scenario"] for row in rows]:
        raise ValueError("Reference scenario census differs")
    rng = np.random.default_rng(saved["seed_entropy"])
    bound = LikelihoodBound(state[3], False)
    timings = []
    old_arrays, new_arrays = [], []
    maximum_difference = 0.0
    for family_index, row in enumerate(rows):
        started = time.monotonic()
        path = reference / "grid800" / f"weights-{family_index:03d}.npz"
        if driver.digest(path) != row["weight_sha256"]:
            raise ValueError("Reference weights changed: " + str(path))
        with np.load(path) as arrays:
            expected = arrays["log_weight"]
            cutoff = float(arrays["bounded_log_weight_upper"])
            values, bounded = np.full(4096, -np.inf), np.zeros(4096, bool)
            for begin in range(0, 4096, 256):
                physical, ratio, _, scene, geometry, *_ = driver.physical_sample(
                    state, row["scenario"], 256, rng
                )
                values[begin : begin + 256], bounded[begin : begin + 256] = (
                    driver.evaluate_sample(
                        state, physical, ratio, scene, geometry, bound, cutoff, 64
                    )
                )
            same_support = np.array_equal(np.isfinite(values), np.isfinite(expected))
            if not same_support or not np.array_equal(bounded, arrays["bounded"]):
                np.savez_compressed(
                    output / f"first-failure-{index}-{family_index}.npz",
                    actual=values,
                    expected=expected,
                    actual_bounded=bounded,
                    expected_bounded=arrays["bounded"],
                )
                raise ValueError("Exact reference replay differs: " + row["scenario"])
            finite = np.isfinite(values)
            maximum_difference = max(
                maximum_difference,
                float(abs(values[finite] - expected[finite]).max(initial=0)),
            )
            old_arrays.append(expected.copy())
            new_arrays.append(values.copy())
        timings.append(
            {"scenario": row["scenario"], "seconds": time.monotonic() - started}
        )
        if family_index % 15 == 0:
            print(
                json.dumps({"target": index, "families": family_index + 1}), flush=True
            )
    old_norm = logsumexp([logsumexp(values) for values in old_arrays])
    new_norm = logsumexp([logsumexp(values) for values in new_arrays])
    l1 = sum(
        float(np.sum(abs(np.exp(new - new_norm) - np.exp(old - old_norm))))
        for new, old in zip(new_arrays, old_arrays, strict=True)
    )
    if l1 > 1e-8:
        raise ValueError("Frozen posterior L1 replay tolerance exceeded")
    result = {
        "source_id": target["source_id"],
        "status": "PASS_FROZEN_POSTERIOR_REPLAY",
        "maximum_log_weight_difference": maximum_difference,
        "posterior_L1_difference": l1,
        "posterior_L1_tolerance": 1e-8,
        "families": len(rows),
        "draws_per_family": 4096,
        "seconds": time.monotonic() - began,
        "cpu_seconds": time.process_time() - cpu,
        "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "timings": timings,
    }
    driver.write_json(output / f"reference-{index}.json", result)
    return result


def launch():
    plan, _ = driver.verify()
    base = Path(plan["output_root"])
    label = "local.spacelab.job.boi-validation-hardware-v1"
    existing = subprocess.run(
        ["launchctl", "list", label], capture_output=True, check=False
    )
    if existing.returncode == 0:
        raise RuntimeError("Remove only an exited launch agent before resuming")
    other = subprocess.run(
        [
            "pgrep",
            "-f",
            "^" + sys.executable + r" .*pipeline\.vetting\.validation_campaign",
        ],
        capture_output=True,
        check=False,
    )
    if other.returncode == 0:
        raise RuntimeError("Another validation campaign launcher is still live")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    logfile = base / "logs" / f"launcher-{stamp}.log"
    specification = {
        "Label": label,
        "ProgramArguments": [
            "/usr/sbin/taskpolicy",
            "-a",
            sys.executable,
            "-B",
            "-m",
            "pipeline.vetting.validation_campaign",
            "resume",
        ],
        "WorkingDirectory": str(Path(__file__).resolve().parents[2]),
        "ProcessType": "Interactive",
        "LowPriorityIO": False,
        "RunAtLoad": True,
        "KeepAlive": False,
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "StandardOutPath": str(logfile),
        "StandardErrorPath": str(logfile),
    }
    path = base / "audit" / f"launch-agent-{stamp}.plist"
    with path.open("xb") as stream:
        plistlib.dump(specification, stream)
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)], check=True
    )
    print(json.dumps({"label": label, "specification": str(path), "log": str(logfile)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("resume", "status", "replay"))
    parser.add_argument("--target-index", type=int, choices=(0, 1))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    application_qos()
    if args.action == "resume":
        launch()
        return
    if args.action == "status":
        from pipeline.vetting.validation_campaign import inventory

        plan, _ = driver.verify()
        complete, pending = inventory(plan, Path(plan["output_root"]) / "freeze.json")
        print(
            json.dumps(
                {"complete": len(complete), "pending": len(pending), "workers": 10}
            )
        )
        return
    if args.target_index is None or args.output is None:
        parser.error("Reference replay requires --target-index and --output")
    args.output.mkdir(parents=True, exist_ok=False)
    profile = cProfile.Profile()
    try:
        result = (
            profile.runcall(replay, args.target_index, args.output)
            if args.profile
            else replay(args.target_index, args.output)
        )
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "timings"}
            )
        )
    except BaseException as error:
        driver.write_json(
            args.output / "failure.json",
            {
                "utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "error": repr(error),
            },
        )
        raise
    finally:
        if args.profile:
            profile.dump_stats(str(args.output / "runtime.prof"))
            with (args.output / "profile.log").open("w") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumtime").print_stats(
                    40
                )


if __name__ == "__main__":
    main()
