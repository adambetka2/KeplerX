"""Profile and replay frozen V3 inputs in a separate, preserved experiment."""

import argparse
import concurrent.futures
import contextlib
import cProfile
import datetime
import fcntl
import hashlib
import json
import multiprocessing
import os
import pstats
import resource
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from multiprocessing.util import Finalize
from pathlib import Path

MODES = ("stock", "engine", "exact", "certified")
THREAD_VARIABLES = (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
STATE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def write(path, value):
    """Publish complete JSON without replacing an existing record."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def stamp():
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")


def timed(name, timings, function, *args, **kwargs):
    cpu, wall = time.process_time(), time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        timings[name] = dict(
            cpu_s=time.process_time() - cpu, wall_s=time.perf_counter() - wall
        )


def create(
    plan_path,
    output,
    mode="engine",
    workers=1,
    draws=1_000_000,
    repeats=20,
    profile=False,
):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = read(plan_path)
    require(plan["protocol"] == "TRICERATOPS_KEPLER_V3", "Only frozen V3 inputs")
    require(mode in MODES, "Unknown engine mode")
    require(
        type(workers) is int
        and 1 <= workers <= min(plan["cpu_budget"], os.cpu_count()),
        "Workers exceed the source campaign's CPU budget",
    )
    require(
        type(draws) is int and 100 <= draws <= plan["draws_per_scenario"],
        "Draws must be between 100 and the frozen attempted-draw count",
    )
    require(
        repeats in (2, 20) and len(plan["seeds"]) == 20, "Two or twenty frozen seeds"
    )
    require(
        not output.is_relative_to(plan_path.parent)
        and not plan_path.is_relative_to(output),
        "Use a separate experiment directory",
    )
    source = Path(__file__).resolve().parent
    require(
        not output.is_relative_to(source),
        "Keep experiment artifacts outside the package",
    )
    require(output.parent.is_dir(), "Create the experiment's parent directory first")
    require(
        shutil.disk_usage(output.parent).free >= 2 * 2**30,
        "Maintain at least 2 GiB free on the experiment disk",
    )
    # validator.protocol.freeze copies the validator package into the campaign's source.
    frozen_source = plan_path.parent / "source" / "validator"
    require(
        (frozen_source / "probability" / "workflow.py").is_file(),
        "Freeze the campaign with validator.protocol.freeze",
    )
    output.mkdir(parents=True, exist_ok=False)
    config = dict(
        scope="PERFORMANCE_EXPERIMENT_ONLY_NOT_CANDIDATE_ADMISSION",
        source_plan=receipt(plan_path),
        source_freeze=receipt(plan_path.parent / "freeze.json"),
        mode=mode,
        workers=workers,
        threads_per_worker=1,
        draws=draws,
        repeats=repeats,
        profile=profile,
        seeds=plan["seeds"],
        created_utc=stamp(),
    )
    write(output / "experiment.json", config)
    files = sorted(
        p
        for p in frozen_source.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    write(
        output / "freeze.json",
        dict(files=[receipt(p) for p in files] + [receipt(output / "experiment.json")]),
    )
    return output / "experiment.json"


def verify_experiment(config_path):
    config_path = Path(config_path).resolve()
    config, frozen = read(config_path), read(config_path.parent / "freeze.json")
    for item in [*frozen["files"], config["source_plan"], config["source_freeze"]]:
        require(
            digest(item["path"]) == item["sha256"],
            "Experiment input changed: " + item["path"],
        )
    bound = {item["path"] for item in frozen["files"]}
    frozen_source = Path(config["source_plan"]["path"]).parent / "source" / "validator"
    require(
        all(str(p.resolve()) in bound for p in frozen_source.rglob("*.py")),
        "Unbound campaign source",
    )
    return config, digest(config_path.parent / "freeze.json")


def launch(config_path):
    """Run the campaign's frozen workflow so spawn imports the same immutable source."""
    config_path = Path(config_path).resolve()
    config, _ = verify_experiment(config_path)
    root = config_path.parent
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(config["source_plan"]["path"]).parent / "source"),
        NUMBA_CACHE_DIR=str(root / "numba-cache"),
        MPLCONFIGDIR=str(root / "matplotlib"),
    )
    env.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    command = [
        sys.executable,
        "-B",
        "-m",
        "validator.probability.workflow",
        "coordinate",
        str(config_path),
    ]
    started = time.perf_counter()
    result = None
    try:
        result = subprocess.run(command, cwd=root, env=env, check=False)
    finally:
        write(
            root / ("launch-" + stamp() + ".json"),
            dict(
                wall_s=time.perf_counter() - started,
                returncode=result.returncode if result is not None else None,
            ),
        )
    result.check_returncode()


def initialize(config_path, worker=False):
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    import numba

    from validator import campaign, protocol

    config, frozen = verify_experiment(config_path)
    source_plan = Path(config["source_plan"]["path"])
    require(
        Path(campaign.__file__).resolve().is_relative_to(source_plan.parent / "source"),
        "Execute validator from the source campaign's frozen copy",
    )
    require(
        Path(__file__).resolve().is_relative_to(source_plan.parent / "source"),
        "Execute the campaign's frozen workflow",
    )
    plan, source_frozen = protocol.verify(source_plan)
    numba.set_num_threads(1)
    STATE.update(
        config=config,
        frozen=frozen,
        source_frozen=source_frozen,
        plan=plan,
        root=Path(config_path).resolve().parent,
        packets={},
        calculations=0,
    )
    if worker:
        context = acceleration(config["mode"])
        context.__enter__()
        STATE["acceleration"] = context
        Finalize(None, context.__exit__, args=(None, None, None), exitpriority=10)
    STATE["initialization"] = dict(
        cpu_s=time.process_time() - started_cpu,
        wall_s=time.perf_counter() - started_wall,
    )


def acceleration(mode):
    from validator.probability import accelerated_v3, engine

    return {
        "stock": contextlib.nullcontext,
        "engine": engine,
        "exact": accelerated_v3,
        "certified": lambda: accelerated_v3(certified=True),
    }[mode]()


def packet(stage):
    import numpy as np

    if stage not in STATE["packets"]:
        path = Path(STATE["config"]["source_plan"]["path"]).parent / f"{stage}.npz"
        with np.load(path, allow_pickle=False) as archive:
            STATE["packets"][stage] = {key: archive[key] for key in archive}
        for array in STATE["packets"][stage].values():
            array.setflags(write=False)
    return STATE["packets"][stage]


def check_saved(saved, stage, index):
    from validator.campaign import verify_evidence

    require(
        (
            saved["freeze_sha256"],
            saved["seed"],
            saved["stage"],
            saved["index"],
            saved["draws_per_scenario"],
        )
        == (
            STATE["frozen"],
            STATE["plan"]["seeds"][index],
            stage,
            index,
            STATE["config"]["draws"],
        ),
        "Saved experiment repeat identity differs",
    )
    for item in saved["files"]:
        require(digest(item["path"]) == item["sha256"], "Saved repeat evidence changed")
    verify_evidence(saved)


def compare_source(target, stage, index):
    """Compare complete saved reference outputs only at their original draw count."""
    import numpy as np

    from validator.campaign import verify_evidence

    config = STATE["config"]
    source = Path(config["source_plan"]["path"]).parent
    path = source / "runs" / stage / f"seed-{index:02d}/result.json"
    if config["draws"] != STATE["plan"]["draws_per_scenario"] or not path.exists():
        return None
    saved = read(path)
    require(
        (
            saved["freeze_sha256"],
            saved["seed"],
            saved["stage"],
            saved["index"],
            saved["draws_per_scenario"],
        )
        == (
            STATE["source_frozen"],
            STATE["plan"]["seeds"][index],
            stage,
            index,
            config["draws"],
        ),
        "Source repeat identity differs",
    )
    for item in saved["files"]:
        require(
            digest(item["path"]) == item["sha256"], "Source repeat evidence changed"
        )
    verify_evidence(saved)
    files = {Path(item["path"]).name: Path(item["path"]) for item in saved["files"]}
    with np.load(files["evidence.npz"], allow_pickle=False) as archive:
        equal = dict(
            lnZ=bool(
                np.array_equal(
                    np.asarray(target.lnZ).view(np.uint64),
                    archive["log_evidence"].view(np.uint64),
                )
            )
        )
    equal.update(
        scenarios=target.probs.to_csv(index=False)
        == files["scenarios.csv"].read_text(),
        stars=target.stars.to_csv(index=False) == files["stars.csv"].read_text(),
        FPP=float(target.FPP).hex() == float(saved["FPP"]).hex(),
        NFPP=float(target.NFPP).hex() == float(saved["NFPP"]).hex(),
    )
    return equal


def profile_record(profiler):
    rows = []
    for (filename, line, function), (
        primitive,
        calls,
        own,
        cumulative,
        _,
    ) in pstats.Stats(profiler).stats.items():
        rows.append(
            dict(
                file=filename,
                line=line,
                function=function,
                calls=calls,
                primitive_calls=primitive,
                own_cpu_s=own,
                cumulative_cpu_s=cumulative,
            )
        )
    return sorted(rows, key=lambda r: r["own_cpu_s"], reverse=True)


def calculate(stage, index, trial, timings):
    import numpy as np

    from validator.campaign import calculate_quarters
    from validator.protocol import declared_clocks

    config, plan = STATE["config"], STATE["plan"]
    data = timed("load_packet", timings, packet, stage)
    clock = declared_clocks(plan["candidate"], plan["clock"])[stage]
    profiler = cProfile.Profile(timer=time.process_time) if config["profile"] else None
    require("acceleration" in STATE, "Initialize a dedicated numerical worker")
    np.random.seed(plan["seeds"][index])
    STATE["calculations"] += 1
    if profiler:
        profiler.enable()
    try:
        target = timed(
            "calculate_quarters",
            timings,
            calculate_quarters,
            plan["candidate"],
            clock,
            data,
            config["draws"],
        )
    finally:
        if profiler:
            profiler.disable()
            write(trial / "profile.json", profile_record(profiler))
    return target


def preserve(target, trial):
    import pickle

    import numpy as np

    target.probs.to_csv(trial / "scenarios.csv", index=False)
    target.stars.to_csv(trial / "stars.csv", index=False)
    np.savez_compressed(trial / "evidence.npz", log_evidence=target.lnZ)
    with (trial / "upstream-output.pkl").open("xb") as stream:
        pickle.dump(target, stream)
    return [
        receipt(p)
        for p in sorted(trial.iterdir())
        if p.suffix in (".csv", ".npz", ".pkl", ".json")
    ]


def attempt(stage, index):
    config, root = STATE["config"], STATE["root"]
    require(not (root / "stop.json").exists(), "Experiment explicitly stopped")
    require(
        shutil.disk_usage(root).free >= 2 * 2**30,
        "Maintain at least 2 GiB free on the experiment disk",
    )
    timings = {}
    timed("verify_inputs", timings, verify_experiment, root / "experiment.json")
    from validator.protocol import verify

    timed("verify_source", timings, verify, config["source_plan"]["path"])
    directory = root / "runs" / stage / f"seed-{index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    result = directory / "result.json"
    if result.exists():
        saved = read(result)
        check_saved(saved, stage, index)
        return saved
    trial = directory / ("attempt-" + stamp())
    trial.mkdir()
    write(
        trial / "identity.json",
        dict(
            seed=STATE["plan"]["seeds"][index],
            stage=stage,
            index=index,
            freeze_sha256=STATE["frozen"],
            draws_per_scenario=config["draws"],
        ),
    )
    with (
        (trial / "worker.log").open("x") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            target = calculate(stage, index, trial, timings)
            comparison = timed(
                "compare_source", timings, compare_source, target, stage, index
            )
            files = timed("preserve", timings, preserve, target, trial)
            saved = dict(
                status="COMPLETE_PERFORMANCE_EXPERIMENT_REPEAT",
                stage=stage,
                index=index,
                seed=STATE["plan"]["seeds"][index],
                freeze_sha256=STATE["frozen"],
                draws_per_scenario=config["draws"],
                FPP=float(target.FPP),
                NFPP=float(target.NFPP),
                files=files,
                timings=timings,
                source_comparison=comparison,
                worker_initialization=STATE["initialization"],
                calculation_in_worker=STATE["calculations"],
                pid=os.getpid(),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
            )
            write(trial / "calculation.json", saved)
            require(
                comparison is None or all(comparison.values()),
                "Replay differs from saved source evidence",
            )
            write(result, saved)
            return saved
        except Exception:
            write(trial / "failure.json", dict(traceback=traceback.format_exc()))
            raise


def batch(pool, stage, indices):
    require(not (STATE["root"] / "stop.json").exists(), "Experiment explicitly stopped")
    indices = iter(indices)
    pending = {
        pool.submit(attempt, stage, i): i
        for i in [next(indices, None) for _ in range(STATE["config"]["workers"])]
        if i is not None
    }
    results = {}
    try:
        while pending:
            done, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                index = pending.pop(future)
                results[index] = future.result()
                print(
                    json.dumps(dict(stage=stage, index=index, status="COMPLETE")),
                    flush=True,
                )
            for _ in done:
                index = next(indices, None)
                if index is not None:
                    require(
                        not (STATE["root"] / "stop.json").exists(),
                        "Experiment explicitly stopped",
                    )
                    pending[pool.submit(attempt, stage, index)] = index
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    return [results[i] for i in sorted(results)]


def schedule(pool, config, recovery_count):
    from validator.campaign import unfavorable

    first = batch(pool, "final", range(2))
    if unfavorable(first):
        return "STOPPED_EARLY_FUTILITY_REMAINS_PC", first, []
    if config["repeats"] == 2:
        return "COMPLETE_TWO_SEED_BENCHMARK", first, []
    primary = first + batch(pool, "final", range(2, 20))
    diagnostics = (
        batch(pool, "recovered", range(recovery_count)) if recovery_count else []
    )
    return "COMPLETE_PERFORMANCE_EXPERIMENT", primary, diagnostics


def coordinate(config_path):
    config_path = Path(config_path).resolve()
    config, _ = verify_experiment(config_path)
    root = config_path.parent
    with (root / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (root / "stop.json").exists(), "Experiment explicitly stopped")
        require(
            not (root / "summary.json").exists(),
            "Experiment finished; use a new directory for another measurement",
        )
        initialize(config_path)
        cpu, wall = time.process_time(), time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config["workers"],
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(str(config_path), True),
        ) as pool:
            status, primary, diagnostics = schedule(
                pool, config, STATE["plan"]["candidate"]["recovery_diagnostics"]
            )
        from validator.protocol import statistics
        from validator.records import finite_record

        write(
            root / "summary.json",
            finite_record(
                dict(
                    scope=config["scope"],
                    status=status,
                    primary=statistics(primary),
                    diagnostics=diagnostics,
                    repeats=primary,
                    mode=config["mode"],
                    workers=config["workers"],
                    wall_s=time.perf_counter() - wall,
                    coordinator_cpu_s=time.process_time() - cpu,
                )
            ),
        )
        print(status, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("run", "profile"):
        p = commands.add_parser(operation)
        p.add_argument("plan", type=Path)
        p.add_argument("output", type=Path)
        p.add_argument("--mode", choices=MODES, default="engine")
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--draws", type=int, default=1_000_000)
        p.add_argument(
            "--repeats",
            type=int,
            choices=(2, 20),
            default=2 if operation == "profile" else 20,
        )
    for operation in ("resume", "coordinate"):
        commands.add_parser(operation).add_argument("experiment", type=Path)
    args = parser.parse_args()
    if args.operation == "coordinate":
        coordinate(args.experiment)
    elif args.operation == "resume":
        launch(args.experiment)
    else:
        path = create(
            args.plan,
            args.output,
            args.mode,
            args.workers,
            args.draws,
            args.repeats,
            args.operation == "profile",
        )
        launch(path)


if __name__ == "__main__":
    main()
