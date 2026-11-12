"""Bounded process workers over an explicitly supplied manifest; never select targets."""

import concurrent.futures
import contextlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import ROOT, config_file, digest, read, receipt, require, write
from runner.bois import collect
from runner.outputs import DATABASE, connect, pack_case, store


def stage_commands(case, destination):
    require(
        ("photometry" in case) != ("prepared" in case),
        "Supply photometry or an existing prepared packet",
    )
    commands = []
    prepared = (
        Path(case["prepared"]).resolve()
        if "prepared" in case
        else destination / "prepared"
    )
    if "photometry" in case:
        request = destination / "request.json"
        write(request, case["photometry"])
        commands.append(("prepare", [request, prepared]))
    commands.extend(
        [
            ("detect", [prepared, destination / "detections"]),
            ("vet", [prepared, destination / "detections", destination / "vetting",
                     *(["--cross-lane", Path(case["cross_lane"]).resolve()]
                       if "cross_lane" in case else [])]),
        ]
    )  # fmt: skip
    if "localization" in case:
        request = destination / "localization-request.json"
        write(request, case["localization"])
        commands.append(
            (
                "localize",
                [
                    prepared,
                    destination / "detections",
                    destination / "vetting",
                    request,
                    destination / "localization",
                ],
            )
        )
    return commands


def run_process(stage, arguments, destination, config_path):
    """One stage in a fresh interpreter."""
    command = [
        sys.executable,
        "-B",
        str(ROOT / "run.py"),
        "--config",
        str(config_path),
        stage,
        *map(str, arguments),
    ]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (destination / f"{stage}.launcher.log").open("w") as stream:
        result = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT
        )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return result.returncode, before, after


_COMMAND_LINE = []


def run_in_worker(stage, arguments, destination, config_path):
    """One stage inside this long-lived worker: same parser, receipts and logs."""
    from runner import program

    if not _COMMAND_LINE:
        _COMMAND_LINE.append(program.command_line())
    parser, modules = _COMMAND_LINE[0]
    os.environ["KEPLERX_ISOLATION"] = "worker"
    before = resource.getrusage(resource.RUSAGE_SELF)
    code = 0
    with (destination / f"{stage}.launcher.log").open("w") as stream:
        try:
            args = parser.parse_args(
                ["--config", str(config_path), stage, *map(str, arguments)]
            )
            with contextlib.chdir(ROOT):
                program.execute(modules[stage], args, config_file(args.config))
        except (Exception, SystemExit):
            code = 1
            stream.write(traceback.format_exc())
    after = resource.getrusage(resource.RUSAGE_SELF)
    return code, before, after


def run_case(case, root, config_path, isolation="process"):
    destination = root / case["case_id"]
    destination.mkdir()
    stages = []
    for stage, arguments in stage_commands(case, destination):
        runner = run_in_worker if isolation == "worker" else run_process
        started = time.perf_counter()
        code, before, after = runner(stage, arguments, destination, config_path)
        stages.append(
            {
                "stage": stage,
                "exit_code": code,
                "isolation": isolation,
                "wall_seconds": time.perf_counter() - started,
                "cpu_user_seconds": after.ru_utime - before.ru_utime,
                "cpu_system_seconds": after.ru_stime - before.ru_stime,
            }
        )
        if code:
            return {"case_id": case["case_id"], "state": "ERROR", "stages": stages}
    result = {
        "case_id": case["case_id"],
        "state": "COMPLETE",
        "stages": stages,
        "bois": collect(destination),
    }
    execution = read(config_path)["execution"]
    if (
        "photometry" in case
        and execution.get("retain_photometry", "all") == "candidates"
    ):
        result["photometry_retained"] = release_photometry(destination)
    result["pack"] = pack_case(destination)
    return result


RETAIN_PHOTOMETRY = ("all", "candidates")


def release_photometry(destination):
    """Delete a completed case's own prepared arrays when it has no candidates."""
    detections = read(destination / "detections" / "detections.json")
    if detections["candidates"]:
        return True
    (destination / "prepared" / "photometry.npz").unlink()
    return False


def dispatch_order(rows, policy):
    """Manifest order, or longest prepared light curves first (shorter tail)."""
    if policy != "longest_first":
        return rows

    def cost(row):
        if "prepared" not in row:
            return 0.0
        packet = read(Path(row["prepared"]) / "packet.json")
        return float(packet["usable_cadences"]) * float(
            packet.get("baseline_days", 1.0)
        )

    costs = [cost(row) for row in rows]
    return [row for _, row in sorted(zip(costs, rows), key=lambda x: -x[0])]


class Dispatcher:
    """Keep `workers` cases running; a dead worker fails its in-flight cases."""

    def __init__(self, workers, isolation, out, config_path, stop_on_error):
        self.workers, self.isolation, self.out = workers, isolation, out
        self.config_path, self.database = config_path, connect(out)
        self.stop_on_error, self.stopped, self.ordinal = stop_on_error, False, {}
        self.completed = self.errors = 0

    def record(self, result):
        pack = result.pop("pack", None)
        if pack is not None:
            store(self.database, result["case_id"], pack, self.out / result["case_id"])
        self.database.execute(
            "INSERT INTO jobs VALUES(?,?,?,?)",
            (result["case_id"], self.ordinal.get(result["case_id"]), result["state"],
             json.dumps(result)),
        )  # fmt: skip
        self.database.commit()
        self.completed += 1
        self.errors += result["state"] == "ERROR"
        self.stopped |= result["state"] == "ERROR" and self.stop_on_error

    def pool(self):
        # Long-lived workers import the stages once; renew them periodically.
        return concurrent.futures.ProcessPoolExecutor(
            max_workers=self.workers,
            max_tasks_per_child=200 if self.isolation == "worker" else None,
        )

    def run(self, pending):
        pool, futures = self.pool(), {}
        while True:
            while not self.stopped and len(futures) < self.workers:
                case = next(pending, None)
                if case is None:
                    break
                arguments = (case, self.out, self.config_path, self.isolation)
                futures[pool.submit(run_case, *arguments)] = case["case_id"]
            if not futures:
                break
            done, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            if self.collect(done, futures):
                for case_id in futures.values():
                    self.record(
                        failed(case_id, "Worker pool failed while this case ran")
                    )
                pool.shutdown(wait=False, cancel_futures=True)
                pool, futures = self.pool(), {}
        pool.shutdown()

    def collect(self, done, futures):
        broken = False
        for future in done:
            case_id = futures.pop(future)
            try:
                self.record(future.result())
            except concurrent.futures.process.BrokenProcessPool as error:
                broken = True
                self.record(failed(case_id, f"Worker process died: {error}"))
        return broken


def failed(case_id, error):
    return {"case_id": case_id, "state": "ERROR", "stages": [], "error": error}


def unmounted(path):
    """The symlink on `path` whose target is gone."""
    for link in (Path(path), *Path(path).parents):
        if link.is_symlink() and not link.exists():
            return link
    return None


def run(args, config):
    rows = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    ids = [row["case_id"] for row in rows]
    require(rows and len(set(ids)) == len(ids), "Empty manifest or duplicate case IDs")
    missing = [
        r["prepared"]
        for r in rows
        if "prepared" in r and not Path(r["prepared"]).is_dir()
    ]
    link = unmounted(missing[0]) if missing else None
    require(
        link is None,
        f"{link} points to {os.readlink(link) if link else ''}, which is not mounted",
    )
    require(
        not missing,
        f"{len(missing)} prepared folders are missing, the first {missing[:1]}",
    )
    require(
        all(
            value and value.replace("-", "").replace("_", "").isalnum() for value in ids
        ),
        "Unsafe case ID",
    )
    workers = args.workers or config["execution"]["workers"]
    require(workers >= 1, "Invalid worker count")
    config_path = Path(args.out) / "config.json"
    write(config_path, config)
    execution = config["execution"]
    require(
        execution.get("retain_photometry", "all") in RETAIN_PHOTOMETRY,
        "Unknown retain_photometry",
    )
    isolation = execution.get("isolation", "process")
    require(isolation in ("process", "worker"), f"Unknown isolation {isolation}")
    order = dispatch_order(rows, execution.get("dispatch", "manifest"))
    dispatcher = Dispatcher(
        workers, isolation, Path(args.out), config_path, execution["stop_on_error"]
    )
    dispatcher.ordinal = {row["case_id"]: i for i, row in enumerate(order)}
    dispatcher.run(iter(order))
    stopped = dispatcher.stopped
    dispatcher.database.close()
    result = {
        "manifest": receipt(args.manifest),
        "config_sha256": digest(config_path),
        "scheduled_cases": len(rows),
        "isolation": isolation,
        "dispatch": execution.get("dispatch", "manifest"),
        "completed_cases": dispatcher.completed,
        "error_cases": dispatcher.errors,
        "stopped_on_error": stopped,
        "outputs": DATABASE,
    }
    write(Path(args.out) / "batch.json", result)
    require(
        not stopped,
        "Batch stopped after a stage error; first failure and running workers preserved",
    )
    return result


def arguments(parser):
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int)
