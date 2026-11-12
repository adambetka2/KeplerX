"""Freeze parent-disjoint ordinary-control partitions without reading photometry."""

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SOURCE_IDS = ROOT / "dump/release-injections-v1/ordinary-source-ids.json"
INDEX = ROOT / "discSSD/data/index.sqlite"
SOURCE_POOLS = ROOT / "dump/release-controls-v2/source-pools.csv"
ELIGIBILITY = ROOT / "dump/release-injections-v1/eligibility.jsonl"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value):
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def eligible_reserve(source_pools, eligibility, accepted):
    with Path(source_pools).open(newline="") as stream:
        pools = {
            int(row["pool_rank"]): row["source_id"]
            for row in csv.DictReader(stream)
            if row["category"] == "ordinary"
        }
    results = {}
    with Path(eligibility).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["category"] != "ordinary":
                continue
            rank = int(row["pool_rank"])
            if rank in results or pools.get(rank) != row["source_id"]:
                raise ValueError("Ordinary eligibility and frozen pool differ")
            results[rank] = row
    eligible = [
        {"source_id": row["source_id"], "pool_rank": rank}
        for rank, row in sorted(results.items())
        if row["status"] == "ELIGIBLE"
    ]
    if [row["source_id"] for row in eligible[: len(accepted)]] != accepted:
        raise ValueError(
            "Frozen ordinary identities are not the first eligible pool rows"
        )
    return eligible[len(accepted) :]


def source_locations(index, source_ids):
    locations = {}
    with sqlite3.connect(f"file:{Path(index).resolve()}?mode=ro", uri=True) as database:
        for start in range(0, len(source_ids), 500):
            batch = source_ids[start : start + 500]
            marks = ",".join("?" for _ in batch)
            rows = database.execute(
                f"select gaia_source_id,volume,shard from sources "
                f"where gaia_source_id in ({marks})",
                batch,
            ).fetchall()
            locations.update(
                {
                    str(source_id): (volume, int(shard))
                    for source_id, volume, shard in rows
                }
            )
    if set(locations) != set(source_ids):
        raise ValueError("Some frozen ordinary sources are absent from the LC2 index")
    return locations


def verified_source_metadata(path, expected, identities):
    if not expected or digest(path) != expected:
        raise ValueError(f"Source metadata is unavailable or changed: {path}")
    return pq.read_table(
        path,
        columns=["source_id", "origin_tpf_kic", "raw_fits_sha256"],
        filters=[("source_id", "in", identities)],
    ).to_pylist()


def parent_sources(root, index, source_ids):
    locations = source_locations(index, source_ids)
    groups = defaultdict(list)
    for source_id, location in locations.items():
        groups[location].append(int(source_id))
    parents, raw_inputs = {}, {}
    metadata_files = []
    with sqlite3.connect(f"file:{Path(index).resolve()}?mode=ro", uri=True) as database:
        indexed_files = {
            (volume, path): sha256
            for volume, path, sha256 in database.execute(
                "select volume,path,sha256 from files "
                "where collection='kbonus' and path like 'kbonus/lc2/shard-%/source.parquet'"
            )
        }
    for (volume, shard), identities in sorted(groups.items()):
        relative = f"kbonus/lc2/shard-{shard:04d}/source.parquet"
        path = Path(root) / volume / "data" / relative
        expected = indexed_files.get((volume, relative))
        for row in verified_source_metadata(path, expected, identities):
            source_id = str(row["source_id"])
            parent = int(row["origin_tpf_kic"])
            if parent <= 0 or source_id in parents:
                raise ValueError("Invalid or repeated source-to-parent mapping")
            parents[source_id] = str(parent)
            raw_inputs[source_id] = row["raw_fits_sha256"]
        metadata_files.append({"volume": volume, "path": relative, "sha256": expected})
    if set(parents) != set(source_ids):
        raise ValueError("Some frozen ordinary sources lack parent TPF metadata")
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in raw_inputs.values()
    ):
        raise ValueError("Ordinary source metadata lacks a valid raw FITS identity")
    return parents, raw_inputs, metadata_files


def validate_raw_independence(development, evaluation, raw_inputs, parent_maps):
    selected = development + evaluation
    hashes = {raw_inputs[source_id] for source_id in selected}
    if len(hashes) != len(selected):
        raise ValueError("Ordinary control sources share a raw FITS identity")
    evaluation_hashes = {raw_inputs[source_id] for source_id in evaluation}
    for path in map(Path, parent_maps):
        with path.open(newline="") as stream:
            excluded = {row.get("raw_fits_sha256") for row in csv.DictReader(stream)}
        if evaluation_hashes & excluded:
            raise ValueError(
                f"Ordinary raw FITS identities overlap excluded controls: {path}"
            )


def parent_exclusions(paths):
    parents, inputs = set(), []
    for path in map(Path, paths):
        inputs.append({"path": str(path), "sha256": digest(path)})
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if "parent_tpf" not in (reader.fieldnames or ()):
                raise ValueError(f"Parent exclusions lack parent_tpf: {path}")
            for row in reader:
                values = [row["parent_tpf"], row.get("resolved_parent_tpf", "")]
                parents.update(
                    parent.strip()
                    for value in values
                    for parent in value.split(";")
                    if parent.strip()
                )
    return parents, inputs


def select_opened_parents(rows, count, excluded_sources, excluded_parents=()):
    identities = [row["source_id"] for row in rows]
    if (
        type(count) is not int or count < 1
        or len(identities) != len(set(identities))
        or any(
            not isinstance(row[key], str) or not row[key].isdecimal()
            or int(row[key]) <= 0
            for row in rows for key in ("source_id", "parent_tpf")
        )
    ):
        raise ValueError("Opened pool identities or requested count are invalid")
    excluded_sources, excluded_parents = set(excluded_sources), set(excluded_parents)
    selected, excluded, parents = [], [], set()
    for position, row in enumerate(rows, 1):
        source, parent = row["source_id"], row["parent_tpf"]
        if source in excluded_sources:
            reason = "DECLARED_SOURCE_EXCLUSION"
        elif parent in excluded_parents:
            reason = "DECLARED_PARENT_EXCLUSION"
        elif parent in parents:
            reason = "SELECTED_PARENT_DUPLICATE"
        elif len(selected) == count:
            reason = "BEYOND_REQUESTED_COUNT"
        else:
            selected.append(dict(row))
            parents.add(parent)
            continue
        excluded.append({**row, "original_position": position, "reason": reason})
    if len(selected) != count:
        raise ValueError(f"Opened pool supplies {len(selected)} distinct parents, needs {count}")
    return {"status": "NOT_FROZEN", "selection": selected, "excluded": excluded}


def nonempty_joint_identity(value):
    return isinstance(value, str) and bool(value.strip())


def full_component_witness(row):
    required = {"component_id", "trial_ids", "identity_map_members", "historical_stage_exposure"}
    if not isinstance(row, dict) or not required <= row.keys():
        return False
    return (
        nonempty_joint_identity(row["component_id"])
        and isinstance(row["trial_ids"], list)
        and bool(row["trial_ids"])
        and all(nonempty_joint_identity(trial) for trial in row["trial_ids"])
        and isinstance(row["identity_map_members"], dict)
        and nonempty_joint_identity(row["historical_stage_exposure"])
    )


def joint_metadata_rows(metadata):
    if (
        type(metadata.get("schema")) is not int or metadata["schema"] != 1
        or metadata.get("status") != "PORTABLE_METADATA_SCOPE_PREPARATION_ONLY"
        or metadata.get("joint_independence_complete") is not False
    ):
        raise ValueError("Joint component bindings require provisional scope metadata")
    trials, components = metadata.get("trials"), metadata.get("full_witness_components")
    if not all(isinstance(rows, list) and rows for rows in (trials, components)):
        raise ValueError("Joint trial and component witness rows are missing")
    identities = ("trial_id", "scope", "full_witness_component_id")
    if not all(
        isinstance(row, dict) and all(nonempty_joint_identity(row.get(key)) for key in identities)
        for row in trials
    ) or not all(full_component_witness(row) for row in components):
        raise ValueError("Joint trial identities or required full component witnesses are missing")
    return trials, components


def joint_component_index(metadata):
    trial_rows, component_rows = joint_metadata_rows(metadata)
    trials = {row["trial_id"]: row for row in trial_rows}
    components = {row["component_id"]: row for row in component_rows}
    members = [trial for row in components.values() for trial in row["trial_ids"]]
    if (
        len(trials) != len(trial_rows)
        or len(components) != len(component_rows)
        or sorted(members) != sorted(trials)
        or any(not row["trial_ids"] for row in components.values())
    ):
        raise ValueError("Joint component identities or complete membership changed")
    for component in components.values():
        if any(
            trials[trial]["full_witness_component_id"] != component["component_id"]
            for trial in component["trial_ids"]
        ):
            raise ValueError("Trial and full witnessed component bindings differ")
    return trials, components


def bind_joint_components(metadata, selected_trial_ids):
    """Preserve the entire witnessed dependency component of selected identities."""
    trials, components = joint_component_index(metadata)
    selected = set(selected_trial_ids)
    if not selected or len(selected) != len(selected_trial_ids) or not selected <= trials.keys():
        raise ValueError("Selected joint trial identities are missing or repeated")
    touched = {trials[trial]["full_witness_component_id"] for trial in selected}
    bindings = [components[key] for key in sorted(touched)]
    shared = {
        row["component_id"]: sorted({trials[trial]["scope"] for trial in row["trial_ids"]})
        for row in bindings
    }
    return {
        "status": "METADATA_COMPONENT_BINDINGS_ONLY",
        "joint_independence_complete": False,
        "selected_trial_ids": [trial for trial in trials if trial in selected],
        "full_witness_components": bindings,
        "cross_scope_components": {key: scopes for key, scopes in shared.items() if len(scopes) > 1},
        "holdout_overlap_witnesses": metadata["holdout_overlap_witnesses"],
        "qualified_external_connectivity": metadata["qualified_external_connectivity"],
        "unresolved": metadata["unresolved"],
        "boundary": metadata["boundary"],
    }


def partition(source_ids, parents, development_count, excluded_parents=()):
    if (
        development_count < 1
        or development_count >= len(source_ids)
        or len(source_ids) != len(set(source_ids))
        or set(source_ids) != set(parents)
    ):
        raise ValueError("Invalid ordinary-control identities or development size")
    opened = source_ids[:development_count]
    opened_parents = {parents[source_id] for source_id in opened}
    excluded_parents = set(excluded_parents)
    development, evaluation, excluded = [], [], []
    represented = set()
    for position, source_id in enumerate(opened, 1):
        parent = parents[source_id]
        if parent in represented:
            excluded.append(
                {
                    "source_id": source_id,
                    "parent_tpf": parent,
                    "original_position": position,
                    "reason": "DEVELOPMENT_PARENT_DUPLICATE",
                }
            )
        else:
            represented.add(parent)
            development.append(source_id)
    represented = set()
    for position, source_id in enumerate(
        source_ids[development_count:], development_count + 1
    ):
        parent = parents[source_id]
        if parent in opened_parents:
            reason = "DEVELOPMENT_PARENT_OVERLAP"
        elif parent in excluded_parents:
            reason = "EVALUATION_EXCLUDED_PARENT"
        elif parent in represented:
            reason = "EVALUATION_PARENT_DUPLICATE"
        else:
            represented.add(parent)
            evaluation.append(source_id)
            continue
        excluded.append(
            {
                "source_id": source_id,
                "parent_tpf": parent,
                "original_position": position,
                "reason": reason,
            }
        )
    development_parents = {parents[source_id] for source_id in development}
    evaluation_parents = {parents[source_id] for source_id in evaluation}
    if (
        len(development) != len(development_parents)
        or len(evaluation) != len(evaluation_parents)
        or development_parents & evaluation_parents
    ):
        raise ValueError("Control partitions are not parent-disjoint")
    return opened, development, evaluation, excluded


def replenish_evaluation(
    evaluation,
    excluded,
    reserve,
    parents,
    opened_parents,
    target,
    excluded_parents=(),
):
    represented = {parents[source_id] for source_id in evaluation}
    selected = []
    attempts = 0
    for row in reserve:
        if len(evaluation) + len(selected) == target:
            break
        attempts += 1
        source_id, parent = row["source_id"], parents[row["source_id"]]
        if parent in opened_parents:
            reason = "RESERVE_DEVELOPMENT_PARENT_OVERLAP"
        elif parent in excluded_parents:
            reason = "RESERVE_EXCLUDED_PARENT"
        elif parent in represented:
            reason = "RESERVE_EVALUATION_PARENT_DUPLICATE"
        else:
            represented.add(parent)
            selected.append(source_id)
            continue
        excluded.append(
            {
                "source_id": source_id,
                "parent_tpf": parent,
                "original_position": row["pool_rank"],
                "reason": reason,
            }
        )
    if len(evaluation) + len(selected) != target:
        raise ValueError("Eligible frozen reserve cannot replenish evaluation parents")
    return evaluation + selected, selected, attempts


def freeze(
    out,
    source_ids_path=SOURCE_IDS,
    index=INDEX,
    source_pools=SOURCE_POOLS,
    eligibility=ELIGIBILITY,
    development_count=500,
    evaluation_target=4500,
    parent_maps=(),
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source_ids_path, index = Path(source_ids_path), Path(index)
    source_pools, eligibility = Path(source_pools), Path(eligibility)
    source_ids = list(map(str, json.loads(source_ids_path.read_text())))
    reserve = eligible_reserve(source_pools, eligibility, source_ids)
    parent_ids = source_ids + [row["source_id"] for row in reserve]
    parents, raw_inputs, metadata_files = parent_sources(ROOT, index, parent_ids)
    excluded_parents, exclusion_inputs = parent_exclusions(parent_maps)
    opened, development, evaluation, excluded = partition(
        source_ids,
        {key: parents[key] for key in source_ids},
        development_count,
        excluded_parents,
    )
    evaluation, reserve_selected, reserve_attempts = replenish_evaluation(
        evaluation,
        excluded,
        reserve,
        parents,
        {parents[source_id] for source_id in opened},
        evaluation_target,
        excluded_parents,
    )
    if any(digest(row["path"]) != row["sha256"] for row in exclusion_inputs):
        raise ValueError("Parent exclusions changed during the partition freeze")
    validate_raw_independence(development, evaluation, raw_inputs, parent_maps)
    outputs = {
        "development-opened-source-ids.json": opened,
        "development-source-ids.json": development,
        "evaluation-source-ids.json": evaluation,
    }
    for name, values in outputs.items():
        save_json(out / name, values)
    with (out / "parent-map.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "source_id",
                "parent_tpf",
                "raw_fits_sha256",
                "original_position",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "source_id": source_id,
                "parent_tpf": parents[source_id],
                "raw_fits_sha256": raw_inputs[source_id],
                "original_position": position,
            }
            for position, source_id in enumerate(parent_ids, 1)
        )
    with (out / "excluded-sources.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=excluded[0])
        writer.writeheader()
        writer.writerows(excluded)
    artifacts = {
        path.name: digest(path) for path in sorted(out.iterdir()) if path.is_file()
    }
    reasons = {
        reason: sum(row["reason"] == reason for row in excluded)
        for reason in sorted({row["reason"] for row in excluded})
    }
    report = {
        "schema": 1,
        "status": "FROZEN_PARENT_DISJOINT_PARTITIONS",
        "protocol": "fsc-kepler-ordinary-control-partitions-v3",
        "generator_sha256": digest(__file__),
        "source_ids": {
            "path": str(source_ids_path),
            "sha256": digest(source_ids_path),
            "sources": len(source_ids),
        },
        "source_index": {"path": str(index), "sha256": digest(index)},
        "source_pools": {"path": str(source_pools), "sha256": digest(source_pools)},
        "eligibility": {
            "path": str(eligibility),
            "sha256": digest(eligibility),
            "eligible_reserve_sources": len(reserve),
        },
        "source_metadata": {
            "files": len(metadata_files),
            "indexed_files_sha256": json_digest(metadata_files),
            "status": "ALL_METADATA_CONTENT_HASHES_VERIFIED",
        },
        "raw_input_independence": {
            "development_and_evaluation_unique_raw_fits": len(development)
            + len(evaluation),
            "partition_overlap": 0,
            "excluded_control_overlap": 0,
            "excluded_control_overlap_scope": "evaluation",
            "boundary": "Raw FITS identities from hash-verified source metadata. Evaluation excludes historical exposure; development remains fixed. Raw FITS bytes and pixel values are not read by this partition freeze.",
        },
        "additional_parent_exclusions": {
            "parents": len(excluded_parents),
            "inputs": exclusion_inputs,
        },
        "development": {
            "opened_sources": len(opened),
            "independent_sources": len(development),
            "parents": len({parents[source_id] for source_id in development}),
        },
        "evaluation": {
            "sources": len(evaluation),
            "parents": len({parents[source_id] for source_id in evaluation}),
            "base_sources": len(evaluation) - len(reserve_selected),
            "reserve_sources": len(reserve_selected),
            "reserve_attempts": reserve_attempts,
            "status": "FROZEN_DETECTOR_AND_VETTER_OUTCOMES_UNOPENED",
        },
        "excluded_sources": len(excluded),
        "exclusion_reasons": reasons,
        "parent_overlap": 0,
        "artifacts": artifacts,
        "claim_boundary": (
            "This freeze reads frozen identities, prior input-eligibility decisions, "
            "and LC2 source metadata. It does not read detector or vetting outcomes."
        ),
    }
    save_json(out / "freeze.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-ids", default=SOURCE_IDS, type=Path)
    parser.add_argument("--index", default=INDEX, type=Path)
    parser.add_argument("--source-pools", default=SOURCE_POOLS, type=Path)
    parser.add_argument("--eligibility", default=ELIGIBILITY, type=Path)
    parser.add_argument("--development-count", default=500, type=int)
    parser.add_argument("--evaluation-target", default=4500, type=int)
    parser.add_argument("--exclude-parent-map", action="append", default=[], type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                args.out,
                args.source_ids,
                args.index,
                args.source_pools,
                args.eligibility,
                args.development_count,
                args.evaluation_target,
                args.exclude_parent_map,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
