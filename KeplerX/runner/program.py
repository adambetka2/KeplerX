"""Execute isolated stages and retain status, scientific source and resource use."""

import argparse
import contextlib
import functools
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

from common import (
    ROOT,
    config_file,
    digest,
    json_value,
    output_directory,
    read,
    require,
    source_manifest,
    write,
)

STAGES = {
    "prepare": "photometry",
    "detect": "detector",
    "vet": "vetter",
    "localize": "localizator",
    "prepare-native": "validator.preparation",
    "fit-native": "validator.native_fit",
    "batch": "runner.batch",
}


def environment():
    names = (
        "numpy",
        "scipy",
        "astropy",
        "batman-package",
        "PyWavelets",
        "lkprf",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in names},
    }


@functools.cache
def provenance():
    """This process's source and environment, read once."""
    return source_manifest(), environment()


def shared(case, **records):
    """Each record as its digest, its JSON written once under CASE/provenance/."""
    references = {}
    for name, value in records.items():
        text = json.dumps(json_value(value), sort_keys=True)
        sha256 = hashlib.sha256(text.encode()).hexdigest()
        path = Path(case) / "provenance" / f"{sha256}.json"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
        references[name] = {"sha256": sha256}
    return references


def execute(module, args, config):
    out = output_directory(args.out, config)
    start = time.monotonic()
    source, packages = provenance()
    isolation = os.environ.get("KEPLERX_ISOLATION", "process")
    record = {
        "state": "RUNNING",
        "stage": args.stage,
        "arguments": vars(args),
        "config": config,
        "source": source,
        "environment": packages,
        "pid": os.getpid(),
        "isolation": isolation,
    }
    if isolation == "worker":
        # Batch workers share these records instead of 15 KB of source digests per stage.
        record |= shared(out.parent, config=config, source=source, environment=packages)
    write(out / "run.json", record)
    try:
        with (
            (out / "stdout.log").open("w") as stdout,
            (out / "stderr.log").open("w") as stderr,
        ):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = module.run(args, config)
        record.update(state="COMPLETE", result=result)
        return result
    except BaseException as error:
        record.update(
            state="ERROR",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - start
        record["process_peak_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        write(out / "run.json", record)


def doctor(config):
    import batman
    import numpy as np

    with contextlib.chdir(Path(config["runtime_root"]) / "cache"):
        from localizator.upstream.localize import Localize, MultiLocalize
    from detector.upstream.dump import Dump
    from vetter.upstream.vetting import Vetter

    return {
        "environment": environment(),
        "runtime_root": config["runtime_root"],
        "upstream": verify_upstream(),
        "components": [cls.__name__ for cls in (Dump, Vetter, Localize, MultiLocalize)],
        "batman_import": bool(batman.TransitModel),
        "numpy_import": np.__version__,
        "sample": config["sample"],
        "scientific_calibration": "NOT_RUN",
    }


def verify_upstream():
    origins = read(ROOT / "upstream.json")
    for stage, origin in origins.items():
        for name, expected in origin["files"].items():
            relative = f"{stage}/upstream/{name}"
            require(
                digest(ROOT / relative) == expected,
                f"Unrecorded upstream edit: {relative}",
            )
    return {name: row["commit"] for name, row in origins.items()}


def command_line(batch_only=False):
    parser = argparse.ArgumentParser(
        description="Isolated third-campaign stages; no implicit target selection"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("doctor")
    modules = {}
    commands = {"batch": STAGES["batch"]} if batch_only else STAGES
    for command, component in commands.items():
        module = importlib.import_module(
            f"{component}.program" if "." not in component else component
        )
        modules[command] = module
        module.arguments(subparsers.add_parser(command))
    return parser, modules


def batch_coordinator(argv):
    """The batch parent dispatches work; its workers import the scientific stages."""
    if "-h" in argv or "--help" in argv:
        return False
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config")
    selector.add_argument("stage", nargs="?")
    selected, _ = selector.parse_known_args(argv)
    return selected.stage == "batch"


def main():
    parser, modules = command_line(batch_only=batch_coordinator(sys.argv[1:]))
    args = parser.parse_args()
    config = config_file(args.config)
    result = (
        doctor(config)
        if args.stage == "doctor"
        else execute(modules[args.stage], args, config)
    )
    print(json.dumps(json_value(result), allow_nan=False))
