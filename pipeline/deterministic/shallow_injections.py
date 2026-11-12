"""Screen ordinary KBonus sources for targeted shallow physical-pixel controls."""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]

from deterministic import injections
from deterministic.control_partitions import parent_sources
from deterministic.run import InsufficientCadences, prepare_curve_records
from lc2 import load_curve, source_catalogue

PLAN = Path(__file__).with_name("shallow-injection-plan.json")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv_column(path, name):
    with path.open(newline="") as stream:
        return {str(row[name]) for row in csv.DictReader(stream)}


def jsonl_column(path, name):
    return {
        str(row[name])
        for line in path.read_text().splitlines()
        if (row := json.loads(line))
    }


def all_cells():
    return [
        f"d{d}m{m}p{p}" for d, m, p in itertools.product(range(3), range(4), range(3))
    ]


def include(path, inputs):
    inputs[str(path)] = digest(path)
    return path


def existing_plan_exclusions(controls_root, inputs, parents, source_ids, raw_hashes):
    plan_path = controls_root / "pipeline/deterministic/planet-control-plan.json"
    plan = read(include(plan_path, inputs))
    for item in plan["excluded_parent_csv"]:
        parents.update(
            csv_column(include(controls_root / item["path"], inputs), item["column"])
        )
    for item in plan["excluded_parent_jsonl"]:
        parents.update(
            jsonl_column(include(controls_root / item["path"], inputs), item["field"])
        )
    for item in plan.get("excluded_parent_audits", []):
        audit = read(include(controls_root / item["path"], inputs))
        if audit["status"] != "PARENT_INDEPENDENT":
            raise ValueError("Existing resolved-parent audit is incomplete")
        parents.update(
            str(parent)
            for values in audit["source_parents"].values()
            for parent in values
        )
    for item in plan["opened_source_manifests"]:
        source_ids.update(
            csv_column(include(controls_root / item["path"], inputs), item["column"])
        )
    for item in plan.get("excluded_raw_csv", []):
        raw_hashes.update(
            csv_column(include(controls_root / item["path"], inputs), item["column"])
        )


def frozen_control_exclusions(controls_root, inputs, parents, source_ids):
    for folder in (
        "release-planet-control-partitions-v1",
        "release-planet-morphology-control-partitions-v2",
    ):
        path = controls_root / "dump" / folder / "parent-map.csv"
        freeze = read(include(path.with_name("freeze.json"), inputs))
        if digest(path) != freeze["artifacts"][path.name]:
            raise ValueError(f"Frozen parent map changed: {folder}")
        parents.update(csv_column(include(path, inputs), "parent_tpf"))

    false_path = controls_root / "dump/false-class-controls-v5/candidate-pool.csv"
    false_freeze = read(include(false_path.with_name("freeze.json"), inputs))
    if (
        false_freeze["status"] != "CANDIDATE_POOL_FROZEN"
        or digest(false_path) != false_freeze["artifacts"][false_path.name]
    ):
        raise ValueError("False-class candidate pool changed")
    with false_path.open(newline="") as stream:
        false_rows = list(csv.DictReader(stream))
    include(false_path, inputs)
    parents.update(row["parent_tpf"] for row in false_rows)
    source_ids.update(row["source_id"] for row in false_rows)
    source_ids.update(
        row["contaminant_source_id"]
        for row in false_rows
        if row["contaminant_source_id"]
    )


def pilot_exclusions(controls_root, inputs, parents, source_ids, raw_hashes):
    pilot = controls_root / "dump/shallow-pixel-feasibility-20260924"
    pilot_freeze = read(include(pilot / "partitions/freeze.json", inputs))
    support_path = pilot / "support.json"
    if digest(support_path) != pilot_freeze["support_sha256"]:
        raise ValueError("Opened shallow pilot support changed")
    support = read(include(support_path, inputs))
    for row in support["ledger"]:
        if row["status"] == "PIXEL_SUPPORTED":
            source_ids.add(row["source_id"])
            parents.update(row["resolved_parents"])

    for path in (
        controls_root / "dump/release-injections-v1/input-manifest.json",
        pilot / "pixels-development/input-manifest.json",
    ):
        manifest = read(include(path, inputs))
        raw_hashes.update(
            row["base_raw_fits_sha256"] for row in manifest["sources"].values()
        )


def resolve_source_parents(controls_root, inputs, parents, source_ids):
    index = controls_root / "discSSD/data/index.sqlite"
    mapped, _, metadata_files = parent_sources(
        controls_root, index, sorted(source_ids)
    )
    parents.update(mapped.values())
    inputs[str(index)] = digest(index)
    for item in metadata_files:
        path = controls_root / item["volume"] / "data" / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"Indexed source metadata changed: {path}")
        inputs[str(path)] = item["sha256"]


def exclusion_snapshot(controls_root):
    parents, source_ids, raw_hashes = set(), set(), set()
    inputs = {}
    existing_plan_exclusions(controls_root, inputs, parents, source_ids, raw_hashes)
    frozen_control_exclusions(controls_root, inputs, parents, source_ids)
    pilot_exclusions(controls_root, inputs, parents, source_ids, raw_hashes)
    resolve_source_parents(controls_root, inputs, parents, source_ids)
    return {
        "schema": 1,
        "status": "FROZEN_INPUT_EXCLUSIONS",
        "plan_sha256": digest(PLAN),
        "exclusion_code_sha256": digest(Path(__file__)),
        "parent_mapping_code_sha256": digest(
            Path(__file__).with_name("control_partitions.py")
        ),
        "controls_root": str(controls_root.resolve()),
        "parents": sorted(parents, key=int),
        "source_ids": sorted(source_ids, key=int),
        "raw_fits_sha256": sorted(raw_hashes),
        "inputs": dict(sorted(inputs.items())),
    }


def validate_exclusions(path):
    record = read(path)
    if (
        record["status"] != "FROZEN_INPUT_EXCLUSIONS"
        or record["plan_sha256"] != digest(PLAN)
        or record["exclusion_code_sha256"] != digest(Path(__file__))
        or record["parent_mapping_code_sha256"]
        != digest(Path(__file__).with_name("control_partitions.py"))
    ):
        raise ValueError("Exclusion identity changed")
    return record


def in_cell(value, bounds, index, axis):
    lower, upper = bounds
    if axis == "depth":
        return (value >= lower if index == 0 else value > lower) and value <= upper
    if axis == "mes":
        return value >= lower and (upper is None or value < upper)
    return value >= lower and (value <= upper if index == 2 else value < upper)


def try_cell(records, sigma, source, plan, d, m, p):
    sid, seed = str(source["source_id"]), plan["seed"]
    cell = f"d{d}m{m}p{p}"
    depth_bin = plan["depth_bins_ppm"][d]
    mes_bin = plan["mes_bins"][m]
    period_bin = plan["period_bins_days"][p]
    period = injections.log_uniform(seed, *period_bin, sid, cell, "period")
    epoch = 120.0 + period * injections.unit(seed, sid, cell, "phase")
    times = records["time"].astype(np.float64)
    cycles = np.rint((times - epoch) / period)
    phase = np.abs(times - epoch - cycles * period)
    max_duration = plan["signal"]["duration_hours"][1] / 24
    possible = int(
        np.count_nonzero(phase <= (max_duration + plan["signal"]["exposure_days"]) / 2)
    )
    mes_upper_bound = depth_bin[1] * 1e-6 * math.sqrt(possible) / sigma
    if mes_upper_bound < mes_bin[0]:
        return None, "MES_UPPER_BOUND_BELOW_CELL"
    if possible < 3:
        return None, "NO_CADENCE_SUPPORT"

    desired_depth = plan["target_mes"][m] * sigma * 1e6 / math.sqrt(0.75 * possible)
    desired_depth = min(max(desired_depth, depth_bin[0] * 1.04), depth_bin[1] * 0.96)
    centres = [
        value
        for _, value in injections.event_centres(
            period, epoch, float(times[0]), float(times[-1])
        )
    ]
    for geometry_try in range(2):
        density = injections.log_uniform(
            seed,
            *plan["signal"]["stellar_density_solar"],
            sid,
            cell,
            geometry_try,
            "density",
        )
        impact = injections.uniform(
            seed, *plan["signal"]["impact_parameter"], sid, cell, geometry_try, "impact"
        )
        radius_ratio = math.sqrt(desired_depth * 1e-6)
        for _ in range(4):
            try:
                scaled_axis, inclination, duration = injections.orbit(
                    period, density, impact, radius_ratio
                )
            except ValueError:
                break
            if (
                not plan["signal"]["duration_hours"][0]
                <= duration * 24
                <= plan["signal"]["duration_hours"][1]
            ):
                break
            geometry = (radius_ratio, scaled_axis, inclination, duration)
            loss = injections.transit_loss(
                times, centres, period, geometry, plan["signal"], epoch
            )
            depth = float(loss.max() * 1e6)
            mes = float(np.linalg.norm(loss) / sigma)
            if depth <= depth_bin[1] and mes > mes_upper_bound * (1 + 1e-8):
                raise ValueError(
                    f"Injected MES exceeds declared upper bound: {sid} {cell}"
                )
            events, quarters, points = injections.observed_events(
                records, centres, duration, period, epoch
            )
            if (
                in_cell(depth, depth_bin, d, "depth")
                and in_cell(mes, mes_bin, m, "mes")
                and events >= plan["signal"]["minimum_periodic_events"]
                and len(quarters) >= plan["signal"]["minimum_periodic_quarters"]
            ):
                return {
                    "source_id": sid,
                    "category": "periodic_injection",
                    "stratum": cell,
                    "cell": cell,
                    "period_days": period,
                    "shape_period_days": period,
                    "epoch_bkjd": epoch,
                    "expected_white_mes": mes,
                    "point_sigma_ppm": sigma * 1e6,
                    "depth_ppm": depth,
                    "duration_hours": duration * 24,
                    "radius_ratio": radius_ratio,
                    "scaled_semimajor_axis": scaled_axis,
                    "inclination_degrees": inclination,
                    "impact_parameter": impact,
                    "stellar_density_solar": density,
                    "observed_events": events,
                    "observed_quarters": sorted(map(int, quarters)),
                    "in_transit_cadences": points,
                    "ttv": None,
                    "source_parent_tpf": str(source["origin_tpf_kic"]),
                    "base_raw_fits_sha256": source["raw_fits_sha256"],
                    "source_shard": int(source["shard"]),
                }, "ELIGIBLE"
            if depth <= 0 or mes <= 0:
                break
            target_depth = plan["target_mes"][m] * depth / mes
            target_depth = min(
                max(target_depth, depth_bin[0] * 1.04), depth_bin[1] * 0.96
            )
            radius_ratio *= math.sqrt(target_depth / depth)
    return None, "NO_PHYSICAL_CELL_MATCH"


def evaluate(source, plan):
    try:
        curve = load_curve(ROOT, source["source_id"], "psf_flat")
        records = prepare_curve_records(curve, mask_flatten=True)
        sigma = injections.robust_point_sigma(records)
    except (InsufficientCadences, KeyError, ValueError) as error:
        return {
            "source_id": source["source_id"],
            "status": "INELIGIBLE",
            "reason": f"{type(error).__name__}: {error}",
        }
    rejects = {}
    for d in range(3):
        for m in range(3, -1, -1):
            for p in range(2, -1, -1):
                truth, reason = try_cell(records, sigma, source, plan, d, m, p)
                if truth is not None:
                    return {
                        "source_id": source["source_id"],
                        "status": "ELIGIBLE",
                        "truth": truth,
                        "cell_rejections": rejects,
                    }
                rejects[f"d{d}m{m}p{p}"] = reason
    return {
        "source_id": source["source_id"],
        "status": "INELIGIBLE",
        "reason": "NO_TARGETED_CELL",
        "cell_rejections": rejects,
    }


def screen(args):
    plan = read(PLAN)
    excluded = validate_exclusions(args.exclusions)
    parents = set(excluded["parents"])
    sources = set(excluded["source_ids"])
    raw_hashes = set(excluded["raw_fits_sha256"])
    metadata = [
        row
        for row in source_catalogue(ROOT, [args.shard])
        if str(row["origin_tpf_kic"]) not in parents
        and row["source_id"] not in sources
        and row["raw_fits_sha256"] not in raw_hashes
        and row["finite_psf_flat_kbonus"] >= 100
        and row["quarter_count"] >= 2
    ]
    metadata.sort(
        key=lambda row: hashlib.sha256(
            f"{plan['seed']}:source:{row['source_id']}".encode()
        ).digest()
    )
    if args.max_sources:
        metadata = metadata[: args.max_sources]
    shard_folder = (
        Path(metadata[0]["shard_path"])
        if metadata
        else ROOT
        / ("discSSD" if args.shard % 2 == 0 else "discPCIe")
        / f"data/kbonus/lc2/shard-{args.shard:04d}"
    )
    manifest = shard_folder / "manifest.json"
    identity = {
        "plan_sha256": digest(PLAN),
        "implementation_sha256": {
            path.name: digest(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("injections.py"),
                Path(__file__).with_name("run.py"),
                ROOT / "pipeline/photometry/lc2.py",
            )
        },
        "environment": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in (
                    "numpy",
                    "pyarrow",
                    "batman-package",
                    "astropy",
                    "scipy",
                )
            },
        },
        "exclusions_sha256": digest(args.exclusions),
        "shard_manifest_sha256": digest(manifest),
        "shard": args.shard,
        "max_sources": args.max_sources,
        "selected_source_ids_sha256": hashlib.sha256(
            "\n".join(row["source_id"] for row in metadata).encode()
        ).hexdigest(),
    }
    out = args.out / f"shard-{args.shard:04d}"
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    cell_counts = Counter()
    cell_rejections = Counter()
    for chunk_no, start in enumerate(range(0, len(metadata), args.chunk_size)):
        path = out / f"chunk-{chunk_no:04d}.json"
        rows = metadata[start : start + args.chunk_size]
        source_ids = [row["source_id"] for row in rows]
        if path.exists():
            saved = read(path)
            if (
                saved["identity"] != identity
                or saved["source_ids"] != source_ids
                or [row["source_id"] for row in saved["results"]] != source_ids
            ):
                raise ValueError(f"Resumable chunk identity changed: {path}")
            results = saved["results"]
        else:
            results = [evaluate(row, plan) for row in rows]
            save(
                path,
                {
                    "schema": 1,
                    "status": "COMPLETE_INPUT_ONLY_CHUNK",
                    "identity": identity,
                    "source_ids": source_ids,
                    "results": results,
                },
            )
        counts.update(row["status"] for row in results)
        cell_counts.update(
            row["truth"]["cell"] for row in results if row["status"] == "ELIGIBLE"
        )
        cell_rejections.update(
            (cell, reason)
            for row in results
            for cell, reason in row.get("cell_rejections", {}).items()
        )
    summary = {
        "schema": 1,
        "status": "COMPLETE_INPUT_ONLY_SHARD_SCREEN",
        "claim_boundary": "Physical injected truth and LC2 input availability only; no detector, vetter, or pixel support outcomes. A limited max_sources run is a feasibility sample, not full-cell capacity.",
        "identity": identity,
        "metadata_eligible_sources": len(metadata),
        "source_status": dict(counts),
        "candidate_cell_counts": {cell: cell_counts[cell] for cell in all_cells()},
        "candidate_cell_deficits_to_six": {
            cell: max(0, plan["trials_per_cell"] - cell_counts[cell])
            for cell in all_cells()
        },
        "cell_rejections": {
            cell: {
                reason: count
                for (key, reason), count in sorted(cell_rejections.items())
                if key == cell
            }
            for cell in all_cells()
        },
        "chunks": {
            path.name: digest(path) for path in sorted(out.glob("chunk-*.json"))
        },
    }
    save(out / "summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "metadata_eligible_sources",
                    "source_status",
                    "candidate_cell_counts",
                )
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    exclusions = command.add_parser("exclusions")
    exclusions.add_argument("--controls-root", type=Path, required=True)
    exclusions.add_argument("--out", type=Path, required=True)
    subset = command.add_parser("screen")
    subset.add_argument("--exclusions", type=Path, required=True)
    subset.add_argument("--shard", type=int, required=True)
    subset.add_argument("--out", type=Path, required=True)
    subset.add_argument("--max-sources", type=int, default=0)
    subset.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()
    if args.command == "exclusions":
        save(args.out, exclusion_snapshot(args.controls_root))
        print(
            json.dumps(
                {"status": "FROZEN_INPUT_EXCLUSIONS", "sha256": digest(args.out)}
            )
        )
    else:
        if args.max_sources < 0 or args.chunk_size < 1:
            raise ValueError(
                "Positive chunk size and nonnegative source limit required"
            )
        screen(args)


if __name__ == "__main__":
    main()
