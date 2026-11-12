"""Prepare the declared catalogue-EB controls without detector or vetter outcomes."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))

from deterministic import catalogue_controls as inputs
from runtime import worker_count

CATEGORY = "catalogue_eclipsing_binary"
DEVELOPMENT = 100
EVALUATION = 450


def read(path):
    return json.loads(Path(path).read_text())


def control_pool(identity_capacity, remaining_capacity, catalogue):
    initial, remaining = read(identity_capacity), read(remaining_capacity)
    ordered = initial["ordered_identity_records"]
    keys = [str(row["kic_id"]) for row in ordered]
    retained = list(map(str, remaining["remaining_catalogue_eb_kics"]))
    if len(set(keys)) != len(keys) or retained != [
        key for key in keys if key in set(retained)
    ]:
        raise ValueError("Catalogue-EB reserve does not preserve its frozen order")
    if len(retained) < DEVELOPMENT + EVALUATION:
        raise ValueError("Catalogue-EB identity reserve is insufficient")
    with Path(catalogue).open(newline="") as stream:
        catalogue_rows = list(csv.DictReader(stream))
    declared = {row["kic_id"]: row["catalogue_records"] for row in ordered}
    signals = {}
    for kic in retained:
        original = [row for row in catalogue_rows if str(int(row["KIC"])) == kic]
        if declared[kic] != original:
            raise ValueError(f"Catalogue-EB ephemeris records changed: {kic}")
        signals[int(kic)] = original
    rows = [{"pool_rank": rank, "kic_id": kic} for rank, kic in enumerate(retained, 1)]
    return {"rows": rows}, signals


def validate_products(out, ledger):
    for row in ledger:
        if row["status"] != "ELIGIBLE":
            continue
        raw = out / "downloads" / CATEGORY / f"kic-{int(row['kic_id']):09d}.fits"
        if (
            inputs.digest(raw) != row["fits_sha256"]
            or inputs.digest(out / row["arrays"]) != row["arrays_sha256"]
        ):
            raise ValueError("Catalogue-EB input product content changed")


def restore_eligibility(out, data):
    ledger = inputs.read_eligibility(out / "eligibility.jsonl")
    pool_keys = {int(row["pool_rank"]): int(row["kic_id"]) for row in data["rows"]}
    if any(
        row["category"] != CATEGORY
        or pool_keys.get(row["pool_rank"]) != int(row["kic_id"])
        for row in ledger
    ):
        raise ValueError("Catalogue-EB eligibility differs from its frozen pool")
    validate_products(out, ledger)
    return ledger


def finalize_partitions(out, freeze):
    selected = read(out / "source-ids.json")
    if len(selected) != DEVELOPMENT + EVALUATION or len(set(selected)) != len(selected):
        raise ValueError("Catalogue-EB input selection is incomplete or repeated")
    for name, values in (
        ("development", selected[:DEVELOPMENT]),
        ("evaluation", selected[DEVELOPMENT:]),
    ):
        path = out / f"{name}-source-ids.json"
        if path.exists() and read(path) != values:
            raise ValueError(
                "Catalogue-EB role identities differ from their frozen order"
            )
        if not path.exists():
            inputs.save(path, values)
        freeze["artifacts"][path.name] = inputs.digest(path)
    freeze["artifacts"]["eligibility.jsonl"] = inputs.digest(out / "eligibility.jsonl")
    freeze["scientific_evaluation_run"] = False
    inputs.save(out / "freeze.json", freeze)
    return freeze


def materialize(
    out, identity_capacity, remaining_capacity, catalogue, parent_maps, resume, workers
):
    out = Path(out)
    paths = [
        Path(identity_capacity),
        Path(remaining_capacity),
        Path(catalogue),
        Path(__file__),
        Path(inputs.__file__),
        Path(inputs.parent_map_identities.__code__.co_filename),
        Path(inputs.prepare_curve_records.__code__.co_filename),
        ROOT / "pipeline/photometry/lc2.py",
        *map(Path, parent_maps),
    ]
    spec = {
        "schema": 1,
        "protocol": "fsc-kepler-catalogue-eb-controls-v1",
        "mode": "FROZEN",
        "categories": [CATEGORY],
        "minimum_free_bytes": inputs.MINIMUM_FREE_BYTES,
        "inputs": {str(path.resolve()): inputs.digest(path) for path in paths},
        "excluded_parent_maps": list(map(str, parent_maps)),
        "partitions": {"development": DEVELOPMENT, "evaluation": EVALUATION},
        "selection": "First usable independent parents in the original seed-bound catalogue-EB reserve order, before any detector or vetter outcomes.",
        "claim_boundary": "Catalogue-EB input controls with supplied catalogue ephemerides. Denial requires a reproduced mechanism. Synthetic EB challenge controls remain separately preserved; scientific policy and final outcomes are not frozen by this input preparation.",
    }
    if out.exists() and not resume:
        raise ValueError("Output exists; use --resume or a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "run-spec.json"
    if spec_path.exists() and read(spec_path) != spec:
        raise ValueError("Catalogue-EB run identity changed")
    if not spec_path.exists():
        inputs.save(spec_path, spec)
    data, signals = control_pool(identity_capacity, remaining_capacity, catalogue)
    exclusions = {name: set() for name in ("source_id", "kic_id", "fits_sha256")}
    for path in parent_maps:
        values, _ = inputs.parent_map_identities(path)
        for name, excluded in exclusions.items():
            excluded.update(values[name])
    if any(str(row["kic_id"]) in exclusions["kic_id"] for row in data["rows"]):
        raise ValueError("Frozen catalogue-EB reserve overlaps an excluded parent")
    ledger_path = out / "eligibility.jsonl"
    ledger = restore_eligibility(out, data)
    if (out / "freeze.json").exists():
        freeze = read(out / "freeze.json")
        if any(
            inputs.digest(out / name) != expected
            for name, expected in freeze["artifacts"].items()
        ):
            raise ValueError("Catalogue-EB frozen artifacts changed")
        return finalize_partitions(out, freeze)
    inputs.materialize_category(
        out,
        CATEGORY,
        data,
        signals,
        DEVELOPMENT + EVALUATION,
        ledger_path,
        ledger,
        workers,
        20,
        90,
        exclusions,
    )
    validate_products(out, ledger)
    freeze = inputs.freeze_outputs(
        out, spec, ledger, {CATEGORY: DEVELOPMENT + EVALUATION}
    )
    return finalize_partitions(out, freeze)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--identity-capacity", required=True, type=Path)
    parser.add_argument("--remaining-capacity", required=True, type=Path)
    parser.add_argument("--catalogue", required=True, type=Path)
    parser.add_argument(
        "--exclude-parent-map", action="append", required=True, type=Path
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=worker_count())
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("Catalogue-EB workers must be positive")
    freeze = materialize(
        args.out,
        args.identity_capacity,
        args.remaining_capacity,
        args.catalogue,
        args.exclude_parent_map,
        args.resume,
        args.workers,
    )
    print(
        json.dumps({key: freeze[key] for key in ("status", "selected", "partitions")})
    )


if __name__ == "__main__":
    main()
