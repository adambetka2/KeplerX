"""Query coverage and measure each supplied signal; never change BOI dispositions."""

import argparse
import fcntl
import json
import platform
import shutil
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import astropy
import numpy as np

from tess.archive import coverage, cutouts, digest, read_pixels, write
from tess.measure import measure, summarize
from tess.timing import spacecraft


def validate_clock(clock):
    p, epoch, duration = [
        clock[k] for k in ("period_days", "epoch_bjd_tdb", "duration_days")
    ]
    if (
        not np.all(np.isfinite([p, epoch, duration]))
        or not 0 < duration < p
        or epoch < 2400000
    ):
        raise ValueError(
            "Require a finite period, BJD_TDB epoch and 0 < duration < period"
        )
    covariance = clock.get("covariance_t0_period_days2")
    if covariance is not None:
        c = np.asarray(covariance, float)
        if (
            c.shape != (2, 2)
            or not np.all(np.isfinite(c))
            or not np.allclose(c, c.T)
            or np.min(np.linalg.eigvalsh(c)) < 0
        ):
            raise ValueError(
                "Clock covariance must be positive semidefinite, in (t0, P) order"
            )


def validate(catalogue):
    hosts, signals = set(), set()
    for host in catalogue["hosts"]:
        if host["source_id"] in hosts or not host["source_id"].isdigit():
            raise ValueError("Require unique Gaia source IDs stored as strings")
        hosts.add(host["source_id"])
        if not 0 <= host["ra_deg"] < 360 or not -90 <= host["dec_deg"] <= 90:
            raise ValueError("Invalid source coordinates")
        for signal in host["signals"]:
            key = signal["signal_id"]
            if key in signals or not key.replace("-", "").replace("_", "").isalnum():
                raise ValueError("Invalid or repeated run-scoped signal ID")
            signals.add(key)
            for clock in [signal["clock"], *signal.get("clock_trials", [])]:
                validate_clock(clock)
    return catalogue


def export_mk(run, output):
    from pipeline.catalogue.annotate import load_run

    rows, bindings, _, _ = load_run(run)
    grouped = defaultdict(list)
    for row in rows:
        if row["label"] == "BOI":
            grouped[row["source_id"]].append(row)
    hosts = []
    for sid, group in grouped.items():
        row = group[0]
        evidence = Path(row["evidence"])
        source = json.loads(evidence.read_text())
        if source["source_id"] != sid:
            raise ValueError("Source evidence identity differs")
        bindings[str(evidence)] = digest(evidence)
        saved = {s["signal_id"]: s for s in source["signals"]}
        stellar = source.get("stellar", {})
        host = {
            "source_id": sid,
            "ra_deg": row["ra_deg"],
            "dec_deg": row["dec_deg"],
            "host_kic": row["host_kic"],
            "aperture_kic": row["aperture_kic"],
            "astrometry": None,
            "signals": [],
        }
        if all(
            stellar.get(k) is not None
            for k in ("source_id", "ra", "dec", "pmra", "pmdec")
        ):
            if str(stellar["source_id"]) != sid:
                raise ValueError("Gaia stellar identity differs")
            host.update(
                ra_deg=stellar["ra"],
                dec_deg=stellar["dec"],
                astrometry={
                    "epoch_jyear": 2016.0,
                    "pmra_mas_yr": stellar["pmra"],
                    "pmdec_mas_yr": stellar["pmdec"],
                },
            )
        for row in group:
            original = saved[row["signal_id"]]
            if original["period_days"] != row["period_days"]:
                raise ValueError("Saved signal period differs from ledger")
            host["signals"].append(
                {
                    "signal_id": row["signal_id"],
                    "discovery_view": row["discovery_view"],
                    "disposition": row["label"],
                    "evidence": row["evidence"],
                    "evidence_sha256": bindings[str(evidence)],
                    "event_ids": original["event_ids"],
                    "clock": {
                        "period_days": row["period_days"],
                        "epoch_bjd_tdb": row["epoch_bkjd"] + 2454833.0,
                        "duration_days": row["duration_days"],
                        "covariance_t0_period_days2": None,
                    },
                    "clock_trials": [],
                }
            )
        hosts.append(host)
    result = validate(
        {
            "schema": 1,
            "vetting_run": str(run.resolve()),
            "bindings": bindings,
            "hosts": hosts,
        }
    )
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    print(
        json.dumps(
            {"hosts": len(hosts), "signals": sum(len(h["signals"]) for h in hosts)}
        )
    )


def verify_files(root, result):
    for name, sha256 in result["files"].items():
        if digest(root / name) != sha256:
            raise ValueError(f"Changed completed output: {root / name}")
    for file in result.get("inputs", []):
        if digest(file["path"]) != file["sha256"]:
            raise ValueError(f"Changed FITS input: {file['path']}")


def sector_measurement(path, host, settings, root, timing):
    data = read_pixels(path, host, settings, timing)
    prefix = Path(path).stem
    series_path = root / (prefix + "-photometry.npz")
    np.savez_compressed(
        series_path,
        **{
            k: data[k]
            for k in (
                "time",
                "archive_time",
                "archive_timecorr",
                "target_timecorr",
                "exposure",
                "quality",
                "flux",
                "error",
                "good",
                "aperture",
                "background",
            )
        },
    )
    files, reports = {series_path.name: digest(series_path)}, []
    for signal in host["signals"]:
        other = [s["clock"] for s in host["signals"] if s is not signal]
        for trial, clock in enumerate(
            [signal["clock"], *signal.get("clock_trials", [])]
        ):
            result, arrays = measure(data, clock, other, settings["window_durations"])
            output = root / f"{prefix}-{signal['signal_id']}-clock{trial}.npz"
            np.savez_compressed(
                output, **arrays, median_image=np.nanmedian(data["pixels"], axis=0)
            )
            files[output.name] = digest(output)
            reports.append(
                dict(
                    signal_id=signal["signal_id"],
                    trial=trial,
                    clock=clock,
                    arrays=output.name,
                    **result,
                )
            )
    return {"metadata": data["meta"], "signals": reports}, files


def measure_sector(host, sector, settings, root, reserve_gib, timing):
    result_path = root / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        verify_files(root, result)
        return result
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < reserve_gib * 1024**3:
        raise OSError("Free-space reserve reached")
    paths = cutouts(host, sector, settings["cutout_size_pixels"], root / "pixels")
    products, files = [], {}
    for path in paths:
        product, output_files = sector_measurement(path, host, settings, root, timing)
        products.append(product)
        files.update(output_files)
    result = {
        "sector": sector,
        "products": products,
        "files": files,
        "inputs": [{"path": str(p.resolve()), "sha256": digest(p)} for p in paths],
    }
    write(result_path, result)
    return result


def bind_run(args, settings, kernel_hash):
    args.output.mkdir(parents=True, exist_ok=True)
    code = {
        p.name: digest(p)
        for p in sorted(Path(__file__).parent.glob("*.py"))
        if not p.name.startswith("test_")
    }
    binding = {
        "input_sha256": digest(args.input),
        "settings": settings,
        "sectors": args.sectors,
        "source_ids": args.source_ids,
        "code": code,
        "kernels_sha256": kernel_hash,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "astropy": astropy.__version__,
            "lkspacecraft": version("lkspacecraft"),
            "spiceypy": version("spiceypy"),
        },
    }
    path = args.output / "binding.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError("Run inputs/code changed; use a new output directory")
    write(path, binding)


def get_coverage(host, root):
    path = root / "coverage.json"
    if path.exists():
        return json.loads(path.read_text())
    result = coverage(host)
    if not isinstance(result["response"].get("results"), list):
        raise TypeError("Incomplete MAST footprint response")
    write(path, result)
    time.sleep(0.25)  # MAST permits at most five requests per second.
    return result


def run_host(args, host, settings, timing):
    root = args.output / host["source_id"]
    root.mkdir(exist_ok=True)
    footprint = get_coverage(host, root)
    available = sorted({int(r["sector"]) for r in footprint["response"]["results"]})
    selected = (
        available
        if args.sectors is None
        else [s for s in available if s in args.sectors]
    )
    result = {
        "source_id": host["source_id"],
        "available_sectors": available,
        "selected_sectors": selected,
        "sectors": [],
        "signals": [],
    }
    if args.command == "coverage":
        return result
    for sector in selected:
        result["sectors"].append(
            measure_sector(
                host,
                sector,
                settings,
                root / f"s{sector:04d}",
                args.reserve_gib,
                timing,
            )
        )
    for signal in host["signals"]:
        for trial, clock in enumerate(
            [signal["clock"], *signal.get("clock_trials", [])]
        ):
            events = [
                e
                for sector in result["sectors"]
                for product in sector["products"]
                for s in product["signals"]
                if s["signal_id"] == signal["signal_id"] and s["trial"] == trial
                for e in s["events"]
            ]
            result["signals"].append(
                dict(
                    signal_id=signal["signal_id"],
                    trial=trial,
                    clock=clock,
                    **summarize(events),
                )
            )
    write(root / "summary.json", result)
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_catalogue(args)


def run_catalogue(args):
    catalogue = validate(json.loads(args.input.read_text()))
    if args.source_ids is not None and not set(args.source_ids) <= {
        h["source_id"] for h in catalogue["hosts"]
    }:
        raise ValueError("Requested Gaia source is absent from the supplied catalogue")
    settings = json.loads(args.settings.read_text())
    if not (
        0
        < settings["aperture_radius_pixels"]
        < settings["background_inner_pixels"]
        < settings["background_outer_pixels"]
        < settings["cutout_size_pixels"] / 2
        and settings["window_durations"] > 0.5
        and settings["quality_bitmask"] >= 0
    ):
        raise ValueError("Invalid aperture, background, window or quality settings")
    timing, kernel_hash = (None, None)
    if args.command == "run":
        timing, kernel_hash = spacecraft(
            args.kernel_cache or args.output.parent / "tess-kernels"
        )
    bind_run(args, settings, kernel_hash)
    results, failures = [], []
    for host in catalogue["hosts"]:
        if args.source_ids is not None and host["source_id"] not in args.source_ids:
            continue
        try:
            result = run_host(args, host, settings, timing)
            results.append(result)
            print(
                json.dumps(
                    {
                        "source_id": host["source_id"],
                        "status": "COMPLETE",
                        "sectors": len(result["selected_sectors"]),
                    }
                ),
                flush=True,
            )
        except Exception:  # noqa: BLE001 -- retain the failure and return a nonzero cohort status.
            failure = {
                "source_id": host["source_id"],
                "utc": datetime.now(UTC).isoformat(),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            with (args.output / "failures.jsonl").open("a") as log:
                log.write(json.dumps(failure) + "\n")
            print(
                json.dumps({"source_id": host["source_id"], "status": "ERROR"}),
                flush=True,
            )
    write(
        args.output / "summary.json",
        {
            "status": "PARTIAL" if failures else "COMPLETE",
            "hosts": results,
            "failures": failures,
        },
    )
    return 2 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-mk")
    export.add_argument("run", type=Path)
    export.add_argument("output", type=Path)
    kernels = sub.add_parser("kernels")
    kernels.add_argument("cache", type=Path)
    for command in ("coverage", "run"):
        p = sub.add_parser(command)
        p.add_argument("input", type=Path)
        p.add_argument("output", type=Path)
        p.add_argument(
            "--settings", type=Path, default=Path(__file__).with_name("settings.json")
        )
        p.add_argument("--sectors", type=int, nargs="+")
        p.add_argument("--source-ids", nargs="+")
        p.add_argument("--reserve-gib", type=float, default=20)
        p.add_argument("--kernel-cache", type=Path)
    args = parser.parse_args()
    if args.command == "kernels":
        _, checksum = spacecraft(args.cache)
        print(json.dumps({"kernels_sha256": checksum}))
        return 0
    if args.command == "export-mk":
        export_mk(args.run, args.output)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
