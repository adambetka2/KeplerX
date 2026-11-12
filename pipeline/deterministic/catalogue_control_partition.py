"""Freeze catalogue controls that exclude every already opened probe identity."""

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ("catalogue_candidate", "catalogue_false_positive")


def read(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def opened_run_identities(path):
    path = Path(path)
    preparation_path = path / "preparation.json"
    preparation = read(preparation_path)
    reference = preparation["input_manifest"]
    manifest_path = Path(reference["path"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if (
        preparation["status"] != "PREPARED"
        or digest(manifest_path) != reference["sha256"]
    ):
        raise ValueError("Opened run preparation or input identities changed")
    manifest = read(manifest_path)
    source_ids = [str(row["source_id"]) for row in preparation["sources"]]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Opened run repeats a prepared source")
    identities = {"source_id": set(source_ids), "kic_id": set(), "fits_sha256": set()}
    for source_id in source_ids:
        metadata = manifest["sources"][source_id]["metadata"]
        if str(metadata["source_id"]) != source_id:
            raise ValueError("Opened input metadata source identity differs")
        identities["kic_id"].add(str(metadata["origin_tpf_kic"]))
        identities["fits_sha256"].add(str(metadata["raw_fits_sha256"]))
    inputs = {str(p.resolve()): digest(p) for p in (preparation_path, manifest_path)}
    return identities, inputs


def parent_map_identities(path):
    path = Path(path)
    identities = {"source_id": set(), "kic_id": set(), "fits_sha256": set()}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"source_id", "parent_tpf"} <= set(reader.fieldnames or []):
            raise ValueError("Exclusion map lacks source or parent identity")
        for row in reader:
            identities["source_id"].add(row["source_id"])
            for key in ("parent_tpf", "resolved_parent_tpf"):
                if row.get(key):
                    identities["kic_id"].update(row[key].split(";"))
            if row.get("raw_fits_sha256"):
                identities["fits_sha256"].add(row["raw_fits_sha256"])
    return identities, {str(path.resolve()): digest(path)}


def opened_identities(paths, runs=(), parent_maps=()):
    identities = {"source_id": set(), "kic_id": set(), "fits_sha256": set()}
    inputs = {}
    for path in map(Path, paths):
        freeze_path, truth_path = path / "freeze.json", path / "truth.jsonl"
        freeze = read(freeze_path)
        if (
            freeze.get("protocol") != "fsc-kepler-catalogue-controls-v2"
            or freeze.get("status") != "PROBE_COMPLETE"
        ):
            raise ValueError(
                f"Opened catalogue control is not a completed probe: {path}"
            )
        for row in read_jsonl(truth_path):
            for key, values in identities.items():
                value = str(row[key])
                if value in values:
                    raise ValueError(f"Opened catalogue controls repeat {key}: {value}")
                values.add(value)
        inputs[str(freeze_path.resolve())] = digest(freeze_path)
        inputs[str(truth_path.resolve())] = digest(truth_path)
    for source_paths, loader in (
        (runs, opened_run_identities),
        (parent_maps, parent_map_identities),
    ):
        for path in source_paths:
            additional, evidence = loader(path)
            for key, values in additional.items():
                identities[key].update(values)
            inputs.update(evidence)
    if not identities["source_id"]:
        raise ValueError("At least one opened catalogue probe or run must be declared")
    return identities, inputs


def select_evaluation(rows, targets, opened):
    selected = defaultdict(list)
    for row in rows:
        if row.get("status") != "ELIGIBLE":
            continue
        if any(str(row[key]) in opened[key] for key in opened):
            continue
        category = row["category"]
        if category not in targets or len(selected[category]) >= targets[category]:
            continue
        selected[category].append(row)
    if any(len(selected[category]) != target for category, target in targets.items()):
        raise ValueError("Eligible unopened catalogue-control reserve is insufficient")
    source_ids = [
        str(row["source_id"]) for category in CATEGORIES for row in selected[category]
    ]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Evaluation catalogue controls repeat a Gaia source")
    return selected


def freeze_partition(base, opened_paths, out, opened_runs=(), parent_maps=()):
    base, out = Path(base), Path(out)
    base_freeze_path = base / "freeze.json"
    base_freeze = read(base_freeze_path)
    if (
        base_freeze.get("protocol") != "fsc-kepler-catalogue-controls-v2"
        or base_freeze.get("status") != "FROZEN_INPUTS"
    ):
        raise ValueError("Base catalogue-control materialization is not frozen")
    for relative, expected in base_freeze["artifacts"].items():
        if digest(base / relative) != expected:
            raise ValueError(f"Base catalogue-control artifact changed: {relative}")
    if out.exists():
        raise ValueError("Output already exists")
    opened, opened_inputs = opened_identities(opened_paths, opened_runs, parent_maps)
    ledger_path = base / "eligibility.jsonl"
    targets = {
        category: int(base_freeze["targets"][category]) for category in CATEGORIES
    }
    selected = select_evaluation(read_jsonl(ledger_path), targets, opened)
    rows = [row for category in CATEGORIES for row in selected[category]]
    for row in rows:
        array_path = base / row["arrays"]
        fits_path = (
            base / "downloads" / row["category"] / f"kic-{int(row['kic_id']):09d}.fits"
        )
        if (
            digest(array_path) != row["arrays_sha256"]
            or digest(fits_path) != row["fits_sha256"]
            or row["metadata"]["raw_fits_sha256"] != row["fits_sha256"]
        ):
            raise ValueError(f"Selected catalogue input changed: KIC {row['kic_id']}")
    out.mkdir(parents=True)
    manifest = {
        "schema": 1,
        "measurement_scope": "Parent-disjoint unopened catalogue evaluation controls. Catalogue candidates are real-signal regression controls, not confirmed planets; false-positive labels retain their catalogue meaning.",
        "sources": {
            str(row["source_id"]): {
                "arrays": os.path.relpath(base / row["arrays"], out),
                "sha256": row["arrays_sha256"],
                "metadata": row["metadata"],
            }
            for row in rows
        },
    }
    source_ids = [str(row["source_id"]) for row in rows]
    save(out / "input-manifest.json", manifest)
    save(out / "source-ids.json", source_ids)
    for category in CATEGORIES:
        save(
            out / f"{category}-source-ids.json",
            [str(row["source_id"]) for row in selected[category]],
        )
    truth_path = out / "truth.jsonl"
    with truth_path.open("x") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "category",
                            "pool_rank",
                            "kic_id",
                            "source_id",
                            "fits_sha256",
                            "catalogue_signals",
                        )
                    },
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    outputs = [
        out / "input-manifest.json",
        out / "source-ids.json",
        out / "truth.jsonl",
        *(out / f"{category}-source-ids.json" for category in CATEGORIES),
    ]
    freeze = {
        "schema": 1,
        "protocol": "fsc-kepler-catalogue-evaluation-v2",
        "status": "FROZEN_UNOPENED_EVALUATION_INPUTS",
        "selection": "First eligible frozen-pool rows after excluding every declared opened source, KIC parent and raw FITS identity",
        "targets": targets,
        "selected": {category: len(selected[category]) for category in CATEGORIES},
        "opened_exclusions": {key: len(values) for key, values in opened.items()},
        "inputs": {
            str(base_freeze_path.resolve()): digest(base_freeze_path),
            str(ledger_path.resolve()): digest(ledger_path),
            str(Path(__file__).resolve()): digest(__file__),
            **opened_inputs,
        },
        "artifacts": {str(path.relative_to(out)): digest(path) for path in outputs},
    }
    save(out / "freeze.json", freeze)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--opened", type=Path, action="append", default=[])
    parser.add_argument("--opened-run", type=Path, action="append", default=[])
    parser.add_argument("--exclude-parent-map", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = freeze_partition(
        ROOT / args.base,
        [ROOT / path for path in args.opened],
        ROOT / args.out,
        [ROOT / path for path in args.opened_run],
        [ROOT / path for path in args.exclude_parent_map],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
