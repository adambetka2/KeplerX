"""Paired kernel benchmark on likelihood calls captured from a stock gold run.

    python -m fast_triceratops.bench capture CASE DRAWS
    python -m fast_triceratops.bench replay CASE DRAWS STEP "DESCRIPTION"
    python -m fast_triceratops.bench calc CASE DRAWS STEP "DESCRIPTION"
    python -m fast_triceratops.bench v3 CAMPAIGN DRAWS STEP "DESCRIPTION"
    python -m fast_triceratops.bench reproduce CAMPAIGN WORKERS REPEATS [certified]

CASE is kepler10 (the gold light curve) or typical (gold.typical_lightcurve).
capture runs stock calc_probs on that case and saves every lnL_*_p
call (inputs copied before upstream mutates inc) with its output and timing.
replay runs the fused kernels on the same inputs. calc times whole stock and
accelerated calc_probs runs in one process after a warm-up of each. Both
require bitwise-equal output and append one row to hill-climb.csv.
v3 runs the first final-clock seed of a frozen V3 campaign directory with the
patched runtime (post_boi.campaign.calculate_quarters, as a campaign repeat
does), stock, under accelerated_v3 in its exact and certified modes and under
engine() (the TRICERATOPS copy), and appends a row for each; all but certified
must be bitwise, certified is reported.
reproduce reruns the first REPEATS completed final-clock repeats of a V3 campaign
at its full draws under accelerated_v3, WORKERS single-thread processes at a
time, and compares each with the campaign's saved evidence.npz, scenarios.csv
and FPP/NFPP byte for byte; it prints one JSON line per repeat.
"""

import contextlib
import copy
import csv
import datetime
import pickle
import sys
import time
from pathlib import Path

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
LOG = HERE / "hill-climb.csv"
FIELDS = ["utc", "step", "description", "case", "draws", "threads", "calls", "rows", "stock_cpu_s",
          "stock_wall_s", "fused_cpu_s", "fused_wall_s", "cpu_speedup", "wall_speedup", "bitwise"]


def capture_dir(case, draws):
    return HERE / "scratch" / f"capture-{case}-{draws}"


def capture(case, draws):
    from triceratops import marginal_likelihoods

    from fast_triceratops.gold import case_probs
    from fast_triceratops.kernels import REPLACEMENTS

    out = capture_dir(case, draws)
    out.mkdir(parents=True, exist_ok=False)
    calls = []

    def recorder(name, original):
        def wrapped(*args, **kwargs):
            saved = copy.deepcopy((args, kwargs))
            c, w = time.process_time(), time.perf_counter()
            result = original(*args, **kwargs)
            calls.append({"name": name, "args": saved[0], "kwargs": saved[1], "result": result,
                          "cpu": time.process_time() - c, "wall": time.perf_counter() - w})
            return result
        return wrapped

    for name in REPLACEMENTS:
        setattr(marginal_likelihoods, name, recorder(name, getattr(marginal_likelihoods, name)))
    case_probs(1024, case=case)  # JIT warm-up, not recorded
    calls.clear()
    target = case_probs(draws, case=case)
    with (out / "calls.pkl").open("xb") as stream:
        pickle.dump({"calls": calls, "lnZ": target.lnZ, "threads": numba.get_num_threads()},
                    stream)
    print(f"captured {len(calls)} calls, stock likelihood cpu {sum(c['cpu'] for c in calls):.2f} s")


def stamp():
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(row):
    new = not LOG.exists()
    with LOG.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    print(row)


def row(step, description, case, draws, calls, rows, stock_cpu, stock_wall, cpu, wall, same):
    values = [stamp(), step, description, case, draws, numba.get_num_threads(), calls, rows,
              round(stock_cpu, 3), round(stock_wall, 3), round(cpu, 3), round(wall, 3),
              round(stock_cpu / cpu, 2), round(stock_wall / wall, 2), bool(same)]
    return dict(zip(FIELDS, values, strict=True))


def timed(function, *args):
    c, w = time.process_time(), time.perf_counter()
    result = function(*args)
    return result, time.process_time() - c, time.perf_counter() - w


def calc(case, draws, step, description):
    from fast_triceratops import accelerated
    from fast_triceratops.gold import case_probs

    def run(n):
        return case_probs(n, case=case)

    run(1024)
    with accelerated():
        run(1024)
        fast, fast_cpu, fast_wall = timed(run, draws)
    stock, stock_cpu, stock_wall = timed(run, draws)
    same = (np.array_equal(fast.lnZ.view(np.int64), stock.lnZ.view(np.int64))
            and fast.probs.equals(stock.probs) and (fast.FPP, fast.NFPP) == (stock.FPP, stock.NFPP))
    log(row(step, description, case, draws, "calc_probs", "", stock_cpu, stock_wall,
            fast_cpu, fast_wall, same))
    if not same:
        sys.exit("accelerated calc_probs differs from stock")


def replay(case, draws, step, description):
    from fast_triceratops.kernels import REPLACEMENTS

    with (capture_dir(case, draws) / "calls.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    calls = saved["calls"]
    for name in REPLACEMENTS:  # JIT warm-up on the first call of each kind, untimed
        call = next(c for c in calls if c["name"] == name)
        REPLACEMENTS[name](*copy.deepcopy(call["args"]), **copy.deepcopy(call["kwargs"]))
    cpu = wall = 0.0
    same = True
    for call in calls:
        args, kwargs = copy.deepcopy((call["args"], call["kwargs"]))
        c, w = time.process_time(), time.perf_counter()
        result = REPLACEMENTS[call["name"]](*args, **kwargs)
        cpu += time.process_time() - c
        wall += time.perf_counter() - w
        equal = np.array_equal(result.view(np.int64), call["result"].view(np.int64))
        if not equal:
            bad = np.flatnonzero(result.view(np.int64) != call["result"].view(np.int64))
            print(f"MISMATCH {call['name']}: {bad.size} of {result.size} rows, first {bad[:5]}")
        same &= equal
    stock_cpu = sum(c["cpu"] for c in calls)
    stock_wall = sum(c["wall"] for c in calls)
    log(row(step, description, case, draws, len(calls), sum(c["result"].size for c in calls),
            stock_cpu, stock_wall, cpu, wall, same))
    if not same:
        sys.exit("fused output differs from stock")


def v3(campaign, draws, step, description):
    import json

    from fast_triceratops import accelerated_v3, engine
    from post_boi.campaign import calculate_quarters
    from post_boi.protocol import declared_clocks

    campaign = Path(campaign)
    plan = json.loads((campaign / "plan.json").read_text())
    c = plan["candidate"]
    clock = declared_clocks(c, plan["clock"])["final"]
    with np.load(campaign / "final.npz") as archive:
        packet = {key: archive[key] for key in archive}

    def run(n, seed):
        np.random.seed(seed)
        return calculate_quarters(c, clock, packet, draws=n)

    modes = {"stock": contextlib.nullcontext, "exact": accelerated_v3, "engine": engine,
             "certified": lambda: accelerated_v3(certified=True)}
    results = {}
    for mode, context in modes.items():  # each warmed up, then timed, in one process
        with context():
            run(2048, 1)
            results[mode] = timed(run, draws, plan["seeds"][0])
    stock, stock_cpu, stock_wall = results.pop("stock")
    exact = True
    for mode, (fast, cpu, wall) in results.items():
        same = (np.array_equal(np.asarray(fast.lnZ).view(np.int64), np.asarray(stock.lnZ).view(np.int64))
                and fast.probs.equals(stock.probs) and (fast.FPP, fast.NFPP) == (stock.FPP, stock.NFPP))
        exact &= same or mode == "certified"
        log(row(step, f"{description} [{mode}]", campaign.parent.parent.name, draws, "calc_probs",
                "", stock_cpu, stock_wall, cpu, wall, same))
    if not exact:
        sys.exit("exact accelerated V3 calc_probs differs from stock")


def repeat(campaign, index, certified):
    """One saved final-clock repeat of campaign, recomputed under accelerated_v3."""
    import json

    from fast_triceratops import accelerated_v3
    from post_boi.campaign import calculate_quarters
    from post_boi.protocol import declared_clocks

    plan = json.loads((campaign / "plan.json").read_text())
    c = plan["candidate"]
    clock = declared_clocks(c, plan["clock"])["final"]
    with np.load(campaign / "final.npz") as archive:
        packet = {key: archive[key] for key in archive}
    with accelerated_v3(certified=certified):
        np.random.seed(plan["seeds"][index])
        target, cpu, wall = timed(calculate_quarters, c, clock, packet, plan["draws_per_scenario"])
    saved = json.loads((campaign / f"runs/final/seed-{index:02d}/result.json").read_text())
    files = {Path(item["path"]).name: Path(item["path"]) for item in saved["files"]}
    with np.load(files["evidence.npz"]) as archive:
        lnZ = archive["log_evidence"]
    same = {"lnZ": bool(np.array_equal(np.asarray(target.lnZ).view(np.int64), lnZ.view(np.int64))),
            "scenarios": target.probs.to_csv(index=False) == files["scenarios.csv"].read_text(),
            "FPP": float(target.FPP) == saved["FPP"], "NFPP": float(target.NFPP) == saved["NFPP"]}
    return {"campaign": str(campaign), "index": index, "certified": certified,
            "cpu_s": round(cpu, 2), "wall_s": round(wall, 2), "same": same}


def reproduce(campaign, workers, repeats, certified):
    import json
    import multiprocessing
    import os

    campaign = Path(campaign).resolve()
    done = [i for i in range(repeats) if (campaign / f"runs/final/seed-{i:02d}/result.json").exists()]
    os.environ["NUMBA_NUM_THREADS"] = "1"  # inherited by the spawned workers
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for result in pool.starmap(repeat, [(campaign, i, certified) for i in done]):
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    mode, case, draws = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if mode == "capture":
        capture(case, draws)
    elif mode == "reproduce":
        reproduce(case, draws, int(sys.argv[4]), sys.argv[5:] == ["certified"])
    else:
        {"replay": replay, "calc": calc, "v3": v3}[mode](case, draws, sys.argv[4], sys.argv[5])
