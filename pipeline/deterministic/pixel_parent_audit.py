"""Audit resolved target-pixel parents without reading scientific outcomes."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.deterministic import planet_control_partitions

PLAN = Path(__file__).with_name("planet-control-plan.json")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def root_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_materialization(folder):
    folder = root_path(folder)
    report_path = folder / "materialization.json"
    manifest_path = folder / "input-manifest.json"
    report = read(report_path)
    if report.get("status") != "FROZEN":
        raise ValueError(f"Pixel materialization is not frozen: {folder}")
    if report.get("artifacts", {}).get(manifest_path.name) != digest(manifest_path):
        raise ValueError(f"Pixel input manifest changed: {manifest_path}")
    manifest = read(manifest_path)
    if manifest.get("schema") not in {1, 2} or len(
        manifest.get("sources", {})
    ) != report.get("sources"):
        raise ValueError(f"Unsupported or incomplete pixel manifest: {manifest_path}")
    source_parents = {}
    for source_id, source in manifest["sources"].items():
        pixels = source.get("pixels", {})
        if not pixels:
            raise ValueError(f"Pixel manifest source has no products: {source_id}")
        parents = sorted({str(pixel["base_parent_kic"]) for pixel in pixels.values()})
        if any(not parent.isdigit() or int(parent) <= 0 for parent in parents):
            raise ValueError(
                f"Pixel manifest source has an invalid parent: {source_id}"
            )
        source_parents[str(source_id)] = parents
    return {
        "folder": folder,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "source_parents": source_parents,
        "parents": {
            parent for parents in source_parents.values() for parent in parents
        },
    }


def parent_owners(source_parents):
    owners = defaultdict(list)
    for source_id, parents in source_parents.items():
        for parent in parents:
            owners[parent].append(source_id)
    return {parent: sorted(source_ids) for parent, source_ids in sorted(owners.items())}


def select_owners(owners, parents, minimum=1):
    return {
        parent: owners[parent]
        for parent in sorted(set(parents) & set(owners))
        if len(owners[parent]) >= minimum
    }


def audit(materialization, plan_path=PLAN, references=()):
    plan_path = root_path(plan_path)
    current = load_materialization(materialization)
    reference_runs = [load_materialization(path) for path in references]
    excluded, exclusion_inputs, metadata_inputs = (
        planet_control_partitions.excluded_parents(read(plan_path))
    )
    owners = parent_owners(current["source_parents"])
    internal = select_owners(owners, owners, minimum=2)
    excluded_overlap = select_owners(owners, excluded)
    reference_results = []
    reference_parent_overlap = 0
    reference_source_overlap = 0
    for reference in reference_runs:
        parent_overlap = select_owners(owners, reference["parents"])
        source_overlap = sorted(
            set(current["source_parents"]) & set(reference["source_parents"])
        )
        reference_parent_overlap += len(parent_overlap)
        reference_source_overlap += len(source_overlap)
        reference_results.append(
            {
                "materialization": display_path(reference["folder"]),
                "materialization_sha256": digest(reference["report_path"]),
                "manifest_sha256": digest(reference["manifest_path"]),
                "parent_overlap": parent_overlap,
                "source_overlap": source_overlap,
            }
        )
    independent = not (
        internal
        or excluded_overlap
        or reference_parent_overlap
        or reference_source_overlap
    )
    inputs = {
        display_path(plan_path): digest(plan_path),
        display_path(current["report_path"]): digest(current["report_path"]),
        display_path(current["manifest_path"]): digest(current["manifest_path"]),
        display_path(Path(__file__)): digest(Path(__file__)),
    }
    inputs.update(
        {display_path(path): digest(path) for path in sorted(exclusion_inputs)}
    )
    return {
        "schema": 2,
        "status": "PARENT_INDEPENDENT" if independent else "PARENT_OVERLAP",
        "protocol": "fsc-kepler-resolved-pixel-parent-audit-v2",
        "repository_root": str(ROOT.resolve()),
        "materialization": display_path(current["folder"]),
        "sources": len(current["source_parents"]),
        "distinct_resolved_parents": len(current["parents"]),
        "source_parents": current["source_parents"],
        "checks": {
            "within_materialization": internal,
            "excluded_control_parents": excluded_overlap,
            "reference_materializations": reference_results,
        },
        "counts": {
            "within_materialization_parent_overlap": len(internal),
            "excluded_control_parent_overlap": len(excluded_overlap),
            "reference_parent_overlap": reference_parent_overlap,
            "reference_source_overlap": reference_source_overlap,
        },
        "excluded_parent_count": len(excluded),
        "source_metadata_inputs": metadata_inputs,
        "inputs": dict(sorted(inputs.items())),
        "claim_boundary": (
            "Checks identity independence using resolved pixel-parent KICs only; "
            "does not inspect detector, vetter, localization, or disposition outcomes."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plan", default=PLAN, type=Path)
    parser.add_argument("--reference", action="append", default=[], type=Path)
    args = parser.parse_args()
    report = audit(args.materialization, args.plan, args.reference)
    write_json(root_path(args.out), report)
    print(json.dumps(report, indent=2))
    if report["status"] != "PARENT_INDEPENDENT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
