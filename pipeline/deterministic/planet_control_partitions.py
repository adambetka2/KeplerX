"""Freeze disjoint pixel-level planet-control identities without reading outcomes."""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.deterministic.control_partitions import parent_sources

PLAN = Path(__file__).with_name("planet-control-plan.json")


def read(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def root_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def csv_values(path, column):
    with Path(path).open(newline="") as stream:
        return [str(row[column]) for row in csv.DictReader(stream) if row.get(column)]


def truth_by_source(path, categories):
    categories = set(categories)
    rows = [row for row in read_jsonl(path) if row["category"] in categories]
    mapping = {str(row["source_id"]): row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("Selected injection truth repeats a source identity")
    return rows, mapping


def validate_injections(folder, categories):
    report = read(folder / "materialization.json")
    if report.get("status") != "FROZEN":
        raise ValueError("Planet controls require the frozen injection materialization")
    for name in ("input-manifest.json", "truth.jsonl"):
        if report["artifacts"].get(name) != digest(folder / name):
            raise ValueError(f"Frozen injection artifact changed: {name}")
    manifest = read(folder / "input-manifest.json")
    if manifest.get("schema") != 2:
        raise ValueError("Unsupported injection manifest")
    truth, mapping = truth_by_source(folder / "truth.jsonl", categories)
    if set(mapping) - set(manifest["sources"]):
        raise ValueError("Injection truth lacks manifest records")
    return report, manifest, truth, mapping


def plan_categories(plan):
    if plan.get("schema") == 1:
        return ("periodic_injection",)
    if plan.get("schema") != 2:
        raise ValueError("Unsupported planet-control plan")
    categories = tuple(plan.get("categories", ()))
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("Planet-control categories are empty or repeated")
    return categories


def partition_quotas(plan, partition, category):
    specification = plan["partitions"][partition]
    if plan["schema"] == 1:
        return specification["strata"]
    categories = specification.get("categories", {})
    if set(categories) != set(plan["categories"]):
        raise ValueError(f"{partition} category quotas do not match the plan")
    return categories[category]["strata"]


def excluded_parents(plan):
    parents = set()
    inputs = []
    for item in plan["excluded_parent_csv"]:
        path = root_path(item["path"])
        parents.update(csv_values(path, item["column"]))
        inputs.append(path)
    for item in plan["excluded_parent_jsonl"]:
        path = root_path(item["path"])
        parents.update(str(row[item["field"]]) for row in read_jsonl(path))
        inputs.append(path)
    for item in plan.get("excluded_parent_audits", []):
        path = root_path(item["path"])
        audit = read(path)
        if audit.get("status") != "PARENT_INDEPENDENT":
            raise ValueError("Excluded resolved-parent audit is incomplete")
        parents.update(
            str(parent)
            for source_parents in audit["source_parents"].values()
            for parent in source_parents
        )
        inputs.append(path)
    opened_sources = set()
    for item in plan["opened_source_manifests"]:
        path = root_path(item["path"])
        opened_sources.update(csv_values(path, item["column"]))
        inputs.append(path)
    mapped, _, metadata_files = parent_sources(
        ROOT,
        ROOT / "discSSD/data/index.sqlite",
        sorted(opened_sources),
    )
    parents.update(mapped.values())
    return parents, inputs, metadata_files


def excluded_raw_hashes(plan):
    hashes = set()
    inputs = []
    for item in plan.get("excluded_raw_csv", []):
        path = root_path(item["path"])
        hashes.update(csv_values(path, item["column"]))
        inputs.append(path)
    return hashes, inputs


def validate_support_ledger(item):
    category = item["category"]
    preflight_path = root_path(item["preflight"])
    support_path = root_path(item["support"])
    preflight = read(preflight_path)
    if (
        preflight.get("status") != "COMPLETE_PIXEL_SUPPORT_PREFLIGHT"
        or preflight.get("category") != category
        or preflight.get("artifacts", {}).get("support.jsonl") != digest(support_path)
    ):
        raise ValueError(f"Invalid pixel-support preflight: {category}")
    rows = read_jsonl(support_path)
    if len(rows) != preflight.get("sources"):
        raise ValueError(f"Pixel-support source count changed: {category}")
    supported = set()
    resolved_parents = {}
    statuses = Counter()
    for row in rows:
        source_id = str(row["source_id"])
        if row.get("category") != category:
            raise ValueError("Pixel-support identity is mislabelled")
        status = row.get("status")
        statuses[status] += 1
        if status == "PIXEL_SUPPORTED":
            parents = {
                str(pixel["base_parent_kic"])
                for pixel in row.get("pixels", {}).values()
            }
            if not parents:
                raise ValueError("Supported source has no resolved pixel parent")
            supported.add(source_id)
            resolved_parents[source_id] = parents
        elif status != "PIXELS_UNAVAILABLE":
            raise ValueError(f"Unsupported pixel-support status: {status}")
    identities = [str(row["source_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Pixel-support identities are repeated")
    if dict(statuses) != preflight.get("support_statuses"):
        raise ValueError(f"Pixel-support status counts changed: {category}")
    return (
        supported,
        resolved_parents,
        identities,
        dict(statuses),
        [
            preflight_path,
            support_path,
        ],
    )


def validated_pixel_support(plan, categories):
    specification = plan.get("pixel_support")
    if specification is None:
        return None, {}, [], {}
    ledgers = specification.get("ledgers", [])
    if {item.get("category") for item in ledgers} != set(categories):
        raise ValueError("Pixel-support ledgers do not match the plan categories")
    supported = set()
    resolved_parents = {}
    inputs = []
    counts = {}
    seen = set()
    for item in ledgers:
        category = item["category"]
        selected, parents, identities, statuses, ledger_inputs = (
            validate_support_ledger(item)
        )
        if seen & set(identities):
            raise ValueError("Pixel-support identities repeat between ledgers")
        seen.update(identities)
        supported.update(selected)
        resolved_parents.update(parents)
        counts[category] = statuses
        inputs.extend(ledger_inputs)
    return supported, resolved_parents, inputs, counts


def select(
    rows,
    quotas,
    manifest,
    excluded_parent_ids,
    excluded_raw_hashes,
    resolved_parents=None,
):
    remaining = dict(quotas)
    selected = []
    parents = set(excluded_parent_ids)
    raw_hashes = set(excluded_raw_hashes)
    rejected = Counter()
    for row in rows:
        stratum = row["stratum"]
        if remaining.get(stratum, 0) == 0:
            continue
        metadata = manifest["sources"][row["source_id"]]["metadata"]
        parent = str(metadata["origin_tpf_kic"])
        trial_parents = (
            resolved_parents[row["source_id"]]
            if resolved_parents is not None
            else {parent}
        )
        raw_hash = metadata["raw_fits_sha256"]
        if trial_parents & parents:
            reason = (
                "RESOLVED_PARENT_OVERLAP"
                if resolved_parents is not None
                else "PARENT_OVERLAP"
            )
            rejected[reason] += 1
            continue
        if raw_hash in raw_hashes:
            rejected["RAW_INPUT_OVERLAP"] += 1
            continue
        parents.update(trial_parents)
        raw_hashes.add(raw_hash)
        selected.append(row)
        remaining[stratum] -= 1
    deficits = {key: value for key, value in remaining.items() if value}
    if deficits:
        raise ValueError(f"Planet-control strata lack disjoint trials: {deficits}")
    return selected, parents, raw_hashes, dict(rejected)


def select_partition(
    plan,
    name,
    categories,
    observable,
    manifest,
    parents,
    raw_hashes,
    include_ids=None,
    exclude_ids=(),
    resolved_parents=None,
):
    rows = []
    rejected = {}
    exclude_ids = set(exclude_ids)
    for category in categories:
        pool = observable[category]
        if include_ids is not None:
            pool = [row for row in pool if row["source_id"] in include_ids]
        pool = [row for row in pool if row["source_id"] not in exclude_ids]
        selected, parents, raw_hashes, category_rejected = select(
            pool,
            partition_quotas(plan, name, category),
            manifest,
            parents,
            raw_hashes,
            resolved_parents,
        )
        rows.extend(selected)
        rejected[category] = category_rejected
    return rows, parents, raw_hashes, rejected


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def opened_controls(plan, truth_map, manifest):
    configured = plan.get("opened_periodic_cohort")
    if not configured:
        return [], set(), set(), []
    path = root_path(configured)
    source_ids = csv_values(path, "source_id")
    if len(source_ids) != len(set(source_ids)) or set(source_ids) - set(truth_map):
        raise ValueError("Opened periodic cohort identities are invalid")
    parents = {
        str(manifest["sources"][source_id]["metadata"]["origin_tpf_kic"])
        for source_id in source_ids
    }
    raw_hashes = {
        manifest["sources"][source_id]["metadata"]["raw_fits_sha256"]
        for source_id in source_ids
    }
    return source_ids, parents, raw_hashes, [path]


def development_candidates(opened, supported):
    if supported is None:
        return opened or None
    return supported if not opened else supported & opened


def selected_parent_set(source_ids, manifest, resolved_parents):
    if not source_ids:
        return set()
    if resolved_parents:
        return set().union(*(resolved_parents[source_id] for source_id in source_ids))
    return {
        str(manifest["sources"][source_id]["metadata"]["origin_tpf_kic"])
        for source_id in source_ids
    }


def freeze(out, plan_path=PLAN):
    out, plan_path = Path(out), root_path(plan_path)
    plan = read(plan_path)
    categories = plan_categories(plan)
    injections = root_path(plan["injections"])
    injection_report, manifest, truth, truth_map = validate_injections(
        injections, categories
    )
    opened_ids, opened_parents, opened_raw_hashes, opened_inputs = opened_controls(
        plan, truth_map, manifest
    )
    base_excluded, exclusion_inputs, metadata_inputs = excluded_parents(plan)
    base_raw_hashes, raw_exclusion_inputs = excluded_raw_hashes(plan)
    supported_ids, resolved_parents, support_inputs, support_counts = (
        validated_pixel_support(plan, categories)
    )
    threshold = float(plan["minimum_expected_white_mes"])
    opened = set(opened_ids)
    observable = {
        category: [
            row
            for row in truth
            if row["category"] == category and row["expected_white_mes"] >= threshold
        ]
        for category in categories
    }
    parents, raw_hashes = set(base_excluded), set(base_raw_hashes)
    development_include = development_candidates(opened, supported_ids)
    development, parents, raw_hashes, development_rejected = select_partition(
        plan,
        "development",
        categories,
        observable,
        manifest,
        parents,
        raw_hashes,
        include_ids=development_include,
        resolved_parents=resolved_parents or None,
    )
    development_ids = {row["source_id"] for row in development}
    parents.update(opened_parents)
    raw_hashes.update(opened_raw_hashes)
    evaluation, parents, raw_hashes, evaluation_rejected = select_partition(
        plan,
        "evaluation",
        categories,
        observable,
        manifest,
        parents,
        raw_hashes,
        include_ids=supported_ids,
        exclude_ids=opened | development_ids,
        resolved_parents=resolved_parents or None,
    )
    evaluation_ids = {row["source_id"] for row in evaluation}
    development_parents = selected_parent_set(
        development_ids, manifest, resolved_parents
    )
    evaluation_parents = selected_parent_set(evaluation_ids, manifest, resolved_parents)
    development_raw = {
        manifest["sources"][source_id]["metadata"]["raw_fits_sha256"]
        for source_id in development_ids
    }
    evaluation_raw = {
        manifest["sources"][source_id]["metadata"]["raw_fits_sha256"]
        for source_id in evaluation_ids
    }
    if (
        development_ids & evaluation_ids
        or development_parents & evaluation_parents
        or development_raw & evaluation_raw
    ):
        raise ValueError("Planet-control partitions are not independent")
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("development", development), ("evaluation", evaluation)):
        write_json(out / f"{name}-source-ids.json", [row["source_id"] for row in rows])
        write_jsonl(out / f"{name}-truth.jsonl", rows)
    with (out / "parent-map.csv").open("x", newline="") as stream:
        fields = (
            "partition",
            "source_id",
            "category",
            "parent_tpf",
            "raw_fits_sha256",
            "stratum",
            "expected_white_mes",
        )
        if resolved_parents:
            fields += ("resolved_parent_tpf",)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for partition, rows in (
            ("development", development),
            ("evaluation", evaluation),
        ):
            for row in rows:
                metadata = manifest["sources"][row["source_id"]]["metadata"]
                record = {
                    "partition": partition,
                    "source_id": row["source_id"],
                    "category": row["category"],
                    "parent_tpf": metadata["origin_tpf_kic"],
                    "raw_fits_sha256": metadata["raw_fits_sha256"],
                    "stratum": row["stratum"],
                    "expected_white_mes": row["expected_white_mes"],
                }
                if resolved_parents:
                    record["resolved_parent_tpf"] = ";".join(
                        sorted(resolved_parents[row["source_id"]], key=int)
                    )
                writer.writerow(record)
    artifacts = {
        path.name: digest(path)
        for path in sorted(out.iterdir())
        if path.name != "freeze.json"
    }
    inputs = [
        plan_path,
        Path(__file__),
        Path(__file__).with_name("control_partitions.py"),
        injections / "materialization.json",
        injections / "input-manifest.json",
        injections / "truth.jsonl",
        *exclusion_inputs,
        *raw_exclusion_inputs,
        *support_inputs,
        *opened_inputs,
    ]

    def partition_report(name, rows, rejected):
        return {
            "status": plan["partitions"][name]["status"],
            "sources": len(rows),
            "categories": dict(Counter(row["category"] for row in rows)),
            "strata": dict(Counter(row["stratum"] for row in rows)),
            "rejected": rejected,
        }

    report = {
        "schema": 2,
        "status": "FROZEN_PLANET_CONTROL_PARTITIONS",
        "protocol": plan["protocol"],
        "minimum_expected_white_mes": threshold,
        "partitions": {
            "development": partition_report(
                "development", development, development_rejected
            ),
            "evaluation": partition_report(
                "evaluation", evaluation, evaluation_rejected
            ),
        },
        "disjointness": {
            "source_overlap": 0,
            "parent_tpf_overlap": 0,
            "raw_fits_overlap": 0,
            "excluded_opened_or_other_control_parents": len(base_excluded),
            "excluded_prior_control_raw_inputs": len(base_raw_hashes),
            "opened_periodic_parents": len(opened_parents),
            "opened_periodic_raw_inputs": len(opened_raw_hashes),
        },
        "pixel_support": {
            "required": supported_ids is not None,
            "supported_sources": 0 if supported_ids is None else len(supported_ids),
            "status_counts": support_counts,
        },
        "injection_materialization_sha256": digest(injections / "materialization.json"),
        "injection_generator": injection_report["generator"],
        "inputs": {str(path.relative_to(ROOT)): digest(path) for path in inputs},
        "source_metadata_inputs": metadata_inputs,
        "artifacts": artifacts,
        "claim_boundary": plan["claim_boundary"],
    }
    write_json(out / "freeze.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan", default=PLAN, type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(root_path(args.out), args.plan), indent=2))


if __name__ == "__main__":
    main()
