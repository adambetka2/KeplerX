"""Time the Validator's preparation and native fit on accepted production cases."""

import argparse
import datetime
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def production_record(case):
    log = case / "native.log"
    text = log.read_bytes().decode(errors="ignore") if log.exists() else ""
    start = re.search(r"start (\S+)", text[:400])
    done = re.search(r"done (\S+)", text[-4000:])
    calls = re.findall(r"ncall: (\d+)", text[-4000:])
    elapsed = None
    if start and done:
        parse = datetime.datetime.fromisoformat
        elapsed = (parse(done.group(1)) - parse(start.group(1))).total_seconds()
    return dict(elapsed_seconds=elapsed, calls=int(calls[-1]) if calls else None)


def timed(command, env, cwd=ROOT, log=None):
    """Wall, CPU (the command and its workers) and the largest resident set of any process in it."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    with open(log or os.devnull, "w") as stream:
        subprocess.run(
            ["/usr/bin/time", "-l", *command], env=env, check=True, cwd=cwd,
            stdout=subprocess.DEVNULL, stderr=stream,
        )  # fmt: skip
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    peak = None
    if log:
        found = re.findall(
            r"(\d+)\s+maximum resident set size", Path(log).read_text(errors="ignore")
        )
        peak = int(found[-1]) if found else None
    return dict(wall_seconds=wall, cpu_seconds=cpu, peak_process_rss_bytes=peak)


def compare(case, prepared, native):
    a = np.load(case / "prepared" / "detrended.npz")
    b = np.load(prepared / "detrended.npz")
    prepared_equal = all(np.array_equal(a[k], b[k]) for k in a.files)
    p = np.load(case / "native" / "posterior.npz")
    q = np.load(native / "posterior.npz")
    same_shape = p["unnamed"].shape == q["unnamed"].shape
    posterior_equal = same_shape and np.array_equal(p["unnamed"], q["unnamed"])
    review = [
        json.loads((d / "review.json").read_text()) for d in (case / "native", native)
    ]
    clock = ("period_days", "epoch_bkjd", "duration_days")
    return dict(
        prepared_bitwise_equal=prepared_equal,
        posterior_bitwise_equal=bool(posterior_equal),
        posterior_samples=[len(p["unnamed"]), len(q["unnamed"])],
        loglike_max_abs_difference=(
            float(np.max(np.abs(p["loglike"] - q["loglike"]))) if same_shape else None
        ),
        review_clock_equal=all(review[0][k] == review[1][k] for k in clock),
        review_clock=[{k: r[k] for k in clock} for r in review],
    )


SPACELAB = ROOT.parent


def baseline_case(case, folder, python, kernel_site):
    """Production's preparation and juliet fit, as tools/run_native.sh ran them."""
    records = folder / "native-kernel"
    records.mkdir()
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1", NUMBA_NUM_THREADS="1", MPLBACKEND="Agg",
        MPLCONFIGDIR=str(folder / "matplotlib-cache"),
    )  # fmt: skip
    env.pop("PYTHONPATH", None)
    kernel = dict(
        env, PYTHONPATH=str(kernel_site), FAST_NATIVE_KERNEL_RECORDS=str(records)
    )
    prepared, native = folder / "prepared", folder / "native"
    candidate = str(case / "candidate.json")
    prep = timed(
        [python, "-B", "-m", "post_boi.prepare", candidate, str(prepared)], env, SPACELAB,
        folder / "preparation.log",
    )  # fmt: skip
    fit = timed(
        [python, "-B", "-m", "post_boi.native", candidate, str(prepared), str(native)],
        kernel, SPACELAB, folder / "native.log",
    )  # fmt: skip
    posterior = np.load(native / "posterior.npz")["unnamed"]
    production = np.load(case / "native" / "posterior.npz")["unnamed"]
    return dict(
        preparation=prep,
        native_fit=fit,
        kernel_records=len(list(records.glob("*.json"))),
        posterior_bitwise_equal_production=bool(
            posterior.shape == production.shape
            and np.array_equal(posterior, production)
        ),
    )


MOVED = ("/Users/adam/Other/spacelab/other/boi-v3-20261025/",
         "/Volumes/discSSD/spacelab-mac-archive/other/boi-v3-20261025/")  # fmt: skip


def portable_candidate(case, folder):
    """The case's candidate with its files named absolutely."""
    c = json.loads((case / "candidate.json").read_text())
    c["photometry"] = str(case / c["photometry"])
    c["native"]["priors"] = str(case / c["native"]["priors"])
    mask = c.get("protection_mask")
    if mask is not None:
        mask["path"] = str(case / mask["path"]).replace(*MOVED)
    path = folder / "candidate.json"
    path.write_text(json.dumps(c, indent=2) + "\n")
    return path


def run_case(case, output, workers, env, baseline=None):
    name = "-".join(case.parts[-3:])
    folder = output / name
    folder.mkdir(parents=True, exist_ok=False)
    candidate = portable_candidate(case, folder)
    stock = None
    if baseline:
        (folder / "baseline").mkdir()
        stock = baseline_case(case, folder / "baseline", *baseline)
    python = sys.executable
    prepared, native = folder / "prepared", folder / "native"
    prep = timed(
        [python, "-B", "-m", "validator.preparation", str(candidate), str(prepared)],
        env, ROOT, folder / "preparation.log",
    )  # fmt: skip
    command = [
        python,
        "-B",
        "-m",
        "validator.native_fit",
        str(candidate),
        str(prepared),
        str(native),
    ]
    if workers:
        command += ["--workers", str(workers)]
    fit = timed(command, env, ROOT, folder / "native.log")
    sampler = json.loads((native / "sampler.json").read_text())
    production = production_record(case)
    row = dict(
        case=str(case),
        native_points=sampler["native_points"],
        preparation=prep,
        native_fit=fit,
        sampler=sampler,
        production=production,
        calls_equal=production["calls"] == sampler["calls"],
        load_average=os.getloadavg(),
        baseline=stock,
        **compare(case, prepared, native),
    )
    total = prep["wall_seconds"] + fit["wall_seconds"]
    if production["elapsed_seconds"]:
        row["elapsed_speedup_vs_production_native"] = (
            production["elapsed_seconds"] / total
        )
    if stock:
        base = (
            stock["preparation"]["wall_seconds"] + stock["native_fit"]["wall_seconds"]
        )
        cpu = stock["preparation"]["cpu_seconds"] + stock["native_fit"]["cpu_seconds"]
        row["elapsed_speedup_vs_baseline"] = base / total
        row["cpu_speedup_vs_baseline"] = cpu / (
            prep["cpu_seconds"] + fit["cpu_seconds"]
        )
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--baseline", help="frozen V3 runtime's python")
    parser.add_argument(
        "--kernel-site", help="folder holding the production sitecustomize.py"
    )
    args = parser.parse_args(argv)
    baseline = (args.baseline, Path(args.kernel_site)) if args.baseline else None
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT)
    )
    rows = []
    for case in json.loads(Path(args.cases).read_text()):
        rows.append(run_case(Path(case), output, args.workers, env, baseline))
        (output / "benchmark.json").write_text(json.dumps(rows, indent=1) + "\n")
        r = rows[-1]
        print(
            f"{Path(case).parts[-2]} n={r['native_points']} prep {r['preparation']['wall_seconds']:.1f}s "
            f"fit {r['native_fit']['wall_seconds']:.1f}s prod {r['production']['elapsed_seconds']} "
            f"posterior_equal={r['posterior_bitwise_equal']} prepared_equal={r['prepared_bitwise_equal']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
