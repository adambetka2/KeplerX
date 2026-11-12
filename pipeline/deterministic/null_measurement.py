"""Execute the frozen development null pair and assemble record-mode flux endpoints."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from pipeline.deterministic import null_exclusions as exclusions
from pipeline.deterministic import release_report, run, signal_contracts
from pipeline.runtime import dump_path, implementation_hashes, worker_count

SEED = 20261002


def measurement_inputs(args):
    base_ref = {"path": str(args.base_manifest.resolve()), "sha256": args.base_sha256}
    base = exclusions.development_base(args.base_manifest, args.base_sha256)
    prepared, prep_ref = exclusions.preparation_binding(
        args.original_preparation, args.original_preparation_sha256, base_ref, base
    )
    if len(prepared) != len(base["sources"]):
        raise ValueError(
            "Complete original development preparation is required; retain missing sources"
        )
    binding = {"path": str(args.exclusions.resolve()), "sha256": args.exclusions_sha256}
    index = signal_contracts.exclusion_document(binding)
    signal_contracts.validate_exclusion_index(
        index,
        {
            "role": "development",
            "null_construction": {"base_manifest": base_ref},
        },
    )
    if index["original_preparation"] != prep_ref or list(index["sources"]) != list(
        base["sources"]
    ):
        raise ValueError(
            "Exclusions changed the original preparation or ordered development population"
        )
    for sid in base["sources"]:
        signal_contracts.exclusion_references(
            {
                "role": "development",
                "null_construction": {"base_manifest": base_ref},
                "null_exclusions": binding,
            },
            sid,
        )
    return base_ref, binding


def construction_input(args, out, variant):
    supplied = getattr(args, "constructed_manifests", None)
    if supplied:
        index = ("inverted", "quarter_scrambled").index(variant)
        manifest_path = supplied[index].resolve()
        manifest = signal_contracts.exclusion_document(
            {"path": str(manifest_path), "sha256": args.constructed_sha256[index]}
        )
    else:
        constructed = out / variant / "construction"
        signal_contracts.construct(
            SimpleNamespace(
                input_manifest=args.base_manifest,
                variant=variant,
                seed=SEED,
                exclusions=args.exclusions,
                out=constructed,
            )
        )
        manifest_path = constructed / "input-manifest.json"
        manifest = run.read(manifest_path)
    contract = signal_contracts.null_manifest(run.ROOT, manifest_path)
    if contract["variant"] != variant or contract["seed"] != SEED:
        raise ValueError("Supplied null construction changed its variant or seed")
    if supplied and (
        contract["base_manifest"]
        != {"path": str(args.base_manifest.resolve()), "sha256": args.base_sha256}
        or manifest["null_exclusions"]
        != {"path": str(args.exclusions.resolve()), "sha256": args.exclusions_sha256}
    ):
        raise ValueError(
            "Supplied null construction changed its original base or exclusions"
        )
    return manifest_path, manifest


def execute_variant(args, out, variant, binary):
    manifest_path, manifest = construction_input(args, out, variant)
    (out / variant).mkdir(parents=True, exist_ok=True)
    source_ids = out / variant / "source-ids.json"
    run.save(source_ids, list(manifest["sources"]))
    measured = out / variant / "measurement"
    if not manifest["sources"]:
        measured.mkdir()
        run.save(
            measured / "empty-variant.json",
            {
                "status": "NOT_APPLICABLE_SHORT_COVERAGE",
                "input_manifest": exclusions.reference(manifest_path),
                "detectors_or_vetting_run": False,
            },
        )
        return measured
    run.prepare(
        SimpleNamespace(
            shards=sorted(
                {int(row["metadata"]["shard"]) for row in manifest["sources"].values()}
            ),
            limit=len(manifest["sources"]),
            source_ids=source_ids,
            input_manifest=manifest_path,
            disposition_policy=args.disposition_policy,
            out=measured,
        )
    )
    run.scan(
        SimpleNamespace(
            run=measured,
            binary=binary,
            threads=args.threads,
            disposition_policy=args.disposition_policy,
        )
    )
    run.vet(
        SimpleNamespace(
            run=measured,
            python=args.python,
            workers=args.workers,
            no_network=args.no_network,
            record_every_gate=True,
            vetting_config=args.vetting_config,
            disposition_policy=args.disposition_policy,
            pixel_cache_directory=getattr(args, "pixel_cache_directory", None),
            pixel_minimum_free_bytes=getattr(args, "pixel_minimum_free_bytes", None),
        )
    )
    return measured


def measurement_status(counts):
    return (
        "COMPLETE_DEVELOPMENT_NULL_MEASUREMENT"
        if counts["completed_sources"] == counts["sources"]
        else "INCOMPLETE_DEVELOPMENT_NULL_MEASUREMENT"
    )


def measure(args):
    if bool(args.constructed_manifests) != bool(args.constructed_sha256):
        raise ValueError(
            "Both constructed manifest paths and their two SHA256 pins are required"
        )
    base_ref, exclusion_ref = measurement_inputs(args)
    policy = run.read(args.disposition_policy)
    if (
        policy.get("status") != "DEVELOPMENT"
        or policy.get("boi_catalogue_enabled") is not False
    ):
        raise ValueError(
            "This measurement command requires an opened DEVELOPMENT policy"
        )
    if min(args.threads, args.workers) < 1:
        raise ValueError(
            "Positive detector thread and vetting worker counts are required"
        )
    pixel_cache = run.pixel_cache_configuration(
        args, run.read(run.vetting_config_path(args))
    )
    out = dump_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve() if args.binary else run.build_binary(out)
    runs = [
        execute_variant(args, out, variant, binary)
        for variant in ("inverted", "quarter_scrambled")
    ]
    entries = run.read(args.reports) if args.reports else []
    report = release_report.build_report(
        entries,
        args.input_graph,
        "development_replay",
        null_runs=runs,
        null_base=base_ref,
    )
    run.save(out / "development-report.json", report)
    receipt = {
        "status": measurement_status(report["null_measurement"]),
        "implementation": implementation_hashes(),
        "base_manifest": base_ref,
        "original_preparation": exclusions.reference(args.original_preparation),
        "exclusions": exclusion_ref,
        "selection_seed": SEED,
        "transform_seed": SEED,
        "runs": list(map(str, runs)),
        "report": exclusions.reference(out / "development-report.json"),
        "counts": report["null_measurement"],
        "detector_binary": exclusions.reference(binary),
        "pixel_cache_configuration": pixel_cache,
        "evaluation_arrays_read": False,
        "calibration_run": False,
        "quarter_scene_independence_certified": False,
    }
    run.save(out / "execution.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--original-preparation", type=Path, required=True)
    parser.add_argument("--original-preparation-sha256", required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--exclusions-sha256", required=True)
    parser.add_argument("--input-graph", type=Path, required=True)
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--constructed-manifests",
        type=Path,
        nargs=2,
        metavar=("INVERTED", "QUARTER_SCRAMBLED"),
    )
    parser.add_argument(
        "--constructed-sha256",
        nargs=2,
        metavar=("INVERTED_SHA256", "QUARTER_SCRAMBLED_SHA256"),
    )
    parser.add_argument("--vetting-config", type=Path)
    parser.add_argument("--pixel-cache-directory", type=Path)
    parser.add_argument("--pixel-minimum-free-bytes", type=int)
    parser.add_argument(
        "--disposition-policy", type=Path, default=run.DISPOSITION_POLICY
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--threads", type=int, default=worker_count())
    parser.add_argument("--workers", type=int, default=worker_count())
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args)
    print(json.dumps({key: result[key] for key in ("status", "report", "counts")}))


if __name__ == "__main__":
    main()
