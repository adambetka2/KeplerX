"""Enumerate scarce shallow-control cells for previously eligible LC2 sources."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]

from deterministic import injections, shallow_injections
from deterministic.run import prepare_curve_records
from lc2 import load_curve, source_catalogue

PLAN = Path(__file__).with_name("shallow-cell-enrichment-plan.json")
SCREEN = Path(__file__).with_name("shallow_injections.py")
INJECTION_PLAN = Path(__file__).with_name("shallow-injection-plan.json")


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


def validate_plan(plan, screen_root, exclusions):
    if plan["status"] != "INPUT_ONLY_DEVELOPMENT":
        raise ValueError("Enrichment plan is not an input-only development plan")
    for path, key in (
        (screen_root / "selection.json", "screen_selection_sha256"),
        (screen_root / "audit.json", "screen_audit_sha256"),
        (SCREEN, "base_generator_sha256"),
        (INJECTION_PLAN, "base_plan_sha256"),
        (exclusions, "exclusions_sha256"),
    ):
        if digest(path) != plan[key]:
            raise ValueError(f"Frozen enrichment input changed: {path}")
    audit = read(screen_root / "audit.json")
    target = sorted(
        cell
        for cell, count in audit["candidate_cell_counts"].items()
        if count < plan["candidate_pool_target_per_cell"]
    )
    if target != sorted(plan["target_cells"]):
        raise ValueError("Target cells differ from frozen input-only deficit")


def verified_source_metadata(shard, summary):
    volume = "discSSD" if shard % 2 == 0 else "discPCIe"
    folder = ROOT / volume / f"data/kbonus/lc2/shard-{shard:04d}"
    manifest_path = folder / "manifest.json"
    if digest(manifest_path) != summary["identity"]["shard_manifest_sha256"]:
        raise ValueError(f"LC2 shard manifest changed: {manifest_path}")
    manifest = read(manifest_path)
    for key in ("source", "cadence"):
        item = manifest["files"][key]
        if digest(folder / item["path"]) != item["sha256"]:
            raise ValueError(f"LC2 {key} bytes changed: {folder}")
    return {row["source_id"]: row for row in source_catalogue(ROOT, [shard])}


def enrich(item, source, target_cells, injection_plan):
    original = item["truth"]
    if (
        source["source_id"] != item["source_id"]
        or source["raw_fits_sha256"] != original["base_raw_fits_sha256"]
        or str(source["origin_tpf_kic"]) != original["source_parent_tpf"]
    ):
        raise ValueError(f"Screened source identity changed: {item['source_id']}")
    records = prepare_curve_records(
        load_curve(ROOT, item["source_id"], "psf_flat"), mask_flatten=True
    )
    sigma = injections.robust_point_sigma(records)
    first = original["cell"]
    recovered, reason = shallow_injections.try_cell(
        records,
        sigma,
        source,
        injection_plan,
        int(first[1]),
        int(first[3]),
        int(first[5]),
    )
    if reason != "ELIGIBLE" or recovered != original:
        raise ValueError(f"Frozen first cell cannot be reproduced: {item['source_id']}")
    extras = []
    for cell in target_cells:
        if cell == first:
            continue
        truth, _ = shallow_injections.try_cell(
            records,
            sigma,
            source,
            injection_plan,
            int(cell[1]),
            int(cell[3]),
            int(cell[5]),
        )
        if truth is not None:
            extras.append(truth)
    return {"source_id": item["source_id"], "extra_truths": extras}


def screen_shard(args):
    plan = read(PLAN)
    validate_plan(plan, args.screen_root, args.exclusions)
    selection = read(args.screen_root / "selection.json")
    if args.shard not in selection["selected_shards"]:
        raise ValueError(f"Shard not in frozen 64-shard selection: {args.shard}")
    screen_folder = args.screen_root / f"shard-{args.shard:04d}"
    original_summary_path = screen_folder / "summary.json"
    original_summary = read(original_summary_path)
    identity = original_summary["identity"]
    if (
        original_summary["status"] != "COMPLETE_INPUT_ONLY_SHARD_SCREEN"
        or identity["shard"] != args.shard
        or identity["implementation_sha256"]["shallow_injections.py"]
        != plan["base_generator_sha256"]
        or identity["plan_sha256"] != plan["base_plan_sha256"]
        or identity["exclusions_sha256"] != plan["exclusions_sha256"]
        or identity["max_sources"] != 0
    ):
        raise ValueError(
            f"Frozen base screen identity changed: {original_summary_path}"
        )
    metadata = verified_source_metadata(args.shard, original_summary)
    injection_plan = read(INJECTION_PLAN)
    out = args.out / f"shard-{args.shard:04d}"
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    total_sources = 0
    for chunk_name, original_sha in sorted(original_summary["chunks"].items()):
        original_path = screen_folder / chunk_name
        if digest(original_path) != original_sha:
            raise ValueError(f"Frozen base screen chunk changed: {original_path}")
        chunk = read(original_path)
        if (
            chunk["identity"] != identity
            or chunk["status"] != "COMPLETE_INPUT_ONLY_CHUNK"
        ):
            raise ValueError(
                f"Frozen base screen chunk identity changed: {original_path}"
            )
        eligible = [row for row in chunk["results"] if row["status"] == "ELIGIBLE"]
        record_identity = {
            "plan_sha256": digest(PLAN),
            "implementation_sha256": digest(__file__),
            "base_chunk_sha256": original_sha,
            "base_summary_sha256": digest(original_summary_path),
            "shard": args.shard,
        }
        path = out / chunk_name
        if path.exists():
            saved = read(path)
            if (
                saved["identity"] != record_identity
                or saved["source_ids"] != [row["source_id"] for row in eligible]
                or [row["source_id"] for row in saved["results"]] != saved["source_ids"]
            ):
                raise ValueError(f"Resumable enrichment identity changed: {path}")
            results = saved["results"]
        else:
            results = [
                enrich(
                    row,
                    metadata[row["source_id"]],
                    plan["target_cells"],
                    injection_plan,
                )
                for row in eligible
            ]
            save(
                path,
                {
                    "schema": 1,
                    "status": "COMPLETE_INPUT_ONLY_ENRICHMENT_CHUNK",
                    "identity": record_identity,
                    "source_ids": [row["source_id"] for row in eligible],
                    "results": results,
                },
            )
        total_sources += len(results)
        counts.update(truth["cell"] for row in results for truth in row["extra_truths"])
    if total_sources != original_summary["source_status"].get("ELIGIBLE", 0):
        raise ValueError(f"Enrichment source count differs: {args.shard}")
    summary = {
        "schema": 1,
        "status": "COMPLETE_INPUT_ONLY_CELL_ENRICHMENT",
        "identity": {
            "plan_sha256": digest(PLAN),
            "implementation_sha256": digest(__file__),
            "base_summary_sha256": digest(original_summary_path),
            "shard": args.shard,
        },
        "eligible_sources": total_sources,
        "extra_candidate_options": sum(counts.values()),
        "extra_candidate_cell_counts": {
            cell: counts[cell] for cell in plan["target_cells"]
        },
        "chunks": {
            path.name: digest(path) for path in sorted(out.glob("chunk-*.json"))
        },
        "claim_boundary": "Alternative physical input truths only; no detector, vetter, pixel localization, or sealed outcomes",
    }
    save(out / "summary.json", summary)
    print(
        json.dumps(
            {
                "shard": args.shard,
                "eligible_sources": total_sources,
                "extra_candidate_options": sum(counts.values()),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    screen_shard(parser.parse_args())


if __name__ == "__main__":
    main()
