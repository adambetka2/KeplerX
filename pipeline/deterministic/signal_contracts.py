"""Replayable null construction, exclusions and pre-vetting lane decisions."""

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path[:0] = [
    str(Path(__file__).resolve().parents[1]),
    str(Path(__file__).resolve().parents[1] / "photometry"),
]

from lc2 import QDIAG_FLUX, VIEWS, file_stamp, input_manifest, load_source, photometry
from runtime import ROOT, dump_path, json_sha256

NULL_CONTRACT = "upper-clipped-quarter-nulls-v1"
LANE_CONTRACT = "detector-lane-gates-v1"
FLUX_GATES = ("shape", "orbital", "instrument")
EXCLUSION_RULE = "verified-only-null-exclusions-v1"
EXCLUSION_COVERAGE = "COMPLETE_ACCOUNTING_VERIFIED_ONLY_UNKNOWN_RETAINED"
LANE_METRICS = {"periodic": "search_mes", "shallow": "search_snr", "sparse": "mes"}


def relative_curve(curve, mask_flatten=False, relative_input=False):
    time = np.asarray(curve["time"], dtype=np.float64)
    flux = np.asarray(curve["flux"], dtype=np.float64)
    error = np.asarray(curve["error"], dtype=np.float64)
    good = (
        np.isfinite(time)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (error > 0)
        & (curve["quality"] == 0)
    )
    if mask_flatten:
        good &= curve["flatten"] == 0
    indices = np.flatnonzero(good)
    time, cadence, quarter, flux, error = (
        array[good] for array in (time, curve["cadence"], curve["quarter"], flux, error)
    )
    relative, relative_error = np.empty_like(flux), np.empty_like(error)
    keep = np.ones(len(flux), dtype=bool)
    for value in np.unique(quarter):
        use = quarter == value
        if relative_input:
            relative[use], relative_error[use] = flux[use] - 1.0, error[use]
            continue
        median = float(np.median(flux[use]))
        if not np.isfinite(median) or median <= 0:
            keep[use] = False
            continue
        normalized = flux[use] / median - 1.0
        sigma = 1.4826 * np.median(np.abs(normalized - np.median(normalized)))
        relative[use], relative_error[use] = normalized, error[use] / median
        if sigma > 0:
            keep[use] &= normalized <= 12.0 * sigma
    arrays = [
        array[keep] for array in (time, relative, relative_error, quarter, cadence)
    ]
    indices = indices[keep]
    order = np.lexsort((arrays[4], arrays[0]))
    arrays = [array[order] for array in arrays]
    indices = indices[order]
    _, unique = np.unique(arrays[4], return_index=True)
    unique.sort()
    return indices[unique], [array[unique] for array in arrays]


def quarter_shift(source_id, quarter, seed, count):
    if count < 2:
        return 0
    identity = json.dumps([str(source_id), int(quarter), seed]).encode()
    return 1 + int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") % (
        count - 1
    )


def null_raw(raw, source_id, variant, seed):
    if (
        variant not in {"inverted", "quarter_scrambled"}
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("Invalid null variant or seed")
    discovery = photometry(raw, "psf_flat")
    _, arrays = relative_curve(discovery, True)
    quarter_count = len(np.unique(arrays[3]))
    if variant == "quarter_scrambled" and quarter_count <= 2:
        return None, quarter_count
    result = dict(raw)
    for view in ("psf", "sap", "psf_flat", *QDIAG_FLUX):
        try:
            curve = photometry(raw, view)
        except KeyError:
            continue
        indices, (_, flux, error, quarter, _) = relative_curve(
            curve, view == "psf_flat"
        )
        if variant == "inverted":
            flux = -flux
        else:
            for value in np.unique(quarter):
                use = np.flatnonzero(quarter == value)
                shift = quarter_shift(source_id, value, seed, len(use))
                flux[use] = np.roll(flux[use], -shift)
                error[use] = np.roll(error[use], -shift)
        columns = QDIAG_FLUX.get(
            view, (VIEWS[view] + "_flux", VIEWS[view] + "_flux_err")
        )
        for key, values in zip(columns, (1.0 + flux, error)):
            result[key] = np.full(len(curve["time"]), np.nan, dtype=np.float64)
            result[key][indices] = values
    return result, quarter_count


def null_manifest(root, path):
    if path is None:
        return None
    path, manifest = input_manifest(root, path)
    contract = manifest.get("null_construction")
    if contract is None:
        return None
    from evaluation_scope import reference

    binding = contract.get("base_manifest")
    if not reference(binding):
        raise ValueError("Null base manifest reference is malformed")
    base_path = (Path(root) / binding["path"]).resolve()
    return validated_null_manifest(str(Path(root).resolve()), str(path), file_stamp(path),
                                   str(base_path), file_stamp(base_path), binding["sha256"])


@lru_cache(maxsize=4)
def validated_null_manifest(root, path, stamp, base_path, base_stamp, checksum):
    _, manifest = input_manifest(root, path)
    contract = manifest["null_construction"]
    if (
        manifest["schema"] != 1
        or contract.get("id") != NULL_CONTRACT
        or contract.get("variant") not in {"inverted", "quarter_scrambled"}
        or type(contract.get("seed")) is not int
        or contract["seed"] < 0
        or contract.get("normalized_relative") is not True
    ):
        raise ValueError("Null input construction is invalid")
    from evaluation_scope import read_reference

    _, base = read_reference({"path": base_path, "sha256": checksum})
    validate_construction_ledger(manifest, base, contract["variant"])
    for sid, item in manifest["sources"].items():
        original = base["sources"][sid]["metadata"]
        if any(
            item["metadata"][key] != original[key]
            for key in ("source_id", "origin_tpf_kic", "raw_fits_sha256")
        ):
            raise ValueError("Null input changed its original identity")
        if contract["variant"] == "quarter_scrambled" and item["null_quarters"] <= 2:
            raise ValueError(
                "Quarter scrambling is unavailable for at most two quarters"
            )
    if file_stamp(path) != stamp or file_stamp(base_path) != base_stamp:
        raise ValueError("Null manifests changed during validation")
    return contract


def validate_construction_ledger(manifest, base, variant):
    ledger = manifest.get("construction_ledger", [])
    if manifest["role"] != base["role"] or [row["source_id"] for row in ledger] != list(base["sources"]):
        raise ValueError("Null construction changed its role or original source ledger")
    included = []
    for row in ledger:
        count = row["quarters"]
        if type(count) is not int or count < 1:
            raise ValueError("Null construction coverage is invalid")
        short = variant == "quarter_scrambled" and count <= 2
        status = "INVERTED_ONLY_SHORT_COVERAGE" if short else "CONSTRUCTED"
        if row["status"] != status:
            raise ValueError("Null construction violates the short-coverage rule")
        if not short:
            included.append(row["source_id"])
            if manifest["sources"][row["source_id"]]["null_quarters"] != count:
                raise ValueError("Null construction source coverage changed")
    if list(manifest["sources"]) != included:
        raise ValueError("Null construction omitted or duplicated source inputs")


def calibrated_lanes(policy, required=False):
    gates = policy.get("lane_gates")
    if gates is None or gates.get("status") == "PENDING_DEVELOPMENT_CALIBRATION":
        if required:
            raise ValueError("Frozen lane gates need opened-null calibration")
        return None
    thresholds = gates.get("thresholds", {})
    if (
        gates.get("contract") != LANE_CONTRACT
        or gates.get("status") != "CALIBRATED"
        or set(thresholds) != set(LANE_METRICS)
        or any(
            type(value) not in {int, float} or not np.isfinite(value) or value <= 0
            for value in thresholds.values()
        )
    ):
        raise ValueError("Lane thresholds are malformed")
    from evaluation_scope import read_reference

    _, calibration = read_reference(gates["calibration"])
    if (
        calibration.get("status") != "CALIBRATED"
        or calibration.get("measurement_scope") != "OPENED_DEVELOPMENT_NULLS"
        or calibration.get("lane_contract") != LANE_CONTRACT
        or calibration.get("thresholds") != thresholds
    ):
        raise ValueError("Lane thresholds differ from opened-null calibration")
    return gates


def lane_decision(identity, lane, row, gates):
    metric = LANE_METRICS[lane]
    value = float(row[metric])
    if not np.isfinite(value):
        raise ValueError("Lane statistic is not finite")
    threshold = gates["thresholds"][lane] if gates else None
    passed = threshold is None or value >= threshold
    return {
        "contract": LANE_CONTRACT,
        "proposal_id": identity,
        "lane": lane,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "decision": "PASS" if passed else "REJECT",
        "status": "CALIBRATED" if gates else "DEVELOPMENT_UNGATED",
        "calibration": gates["calibration"] if gates else None,
    }


def null_exclusion(alerts, references):
    from deterministic.planet_control_report import match_event_units

    units = [
        {
            "unit_id": alert["event_id"],
            "quarter": alert["quarter"],
            "centre_bkjd": (alert["start_bkjd"] + alert["end_bkjd"]) / 2,
        }
        for alert in alerts
    ]
    matches = []
    for reference in references:
        if reference["origin"] not in {"REAL_SIGNAL", "CATALOGUE_EPHEMERIS"}:
            raise ValueError("Unsupported null exclusion origin")
        measured = match_event_units(
            reference["events"], units, reference["duration_hours"]
        )
        if measured["status"] == "MATCHED":
            matches.append(
                {
                    "reference_id": reference["reference_id"],
                    "origin": reference["origin"],
                    "event_match": measured,
                }
            )
    return {"status": "EXCLUDED" if matches else "ELIGIBLE", "matches": matches}


@lru_cache(maxsize=64)
def cached_exclusion_document(path, stamp, checksum):
    from evaluation_scope import read_reference

    _, document = read_reference({"path": path, "sha256": checksum})
    if file_stamp(path) != stamp:
        raise ValueError("Frozen exclusion evidence changed while reading")
    return document


def exclusion_document(reference):
    from evaluation_scope import reference as valid_reference

    if not valid_reference(reference):
        raise ValueError("Null exclusion evidence reference is malformed")
    path = (ROOT / reference["path"]).resolve()
    return cached_exclusion_document(str(path), file_stamp(path), reference["sha256"])


@lru_cache(maxsize=64)
def cached_exclusion_file(path, stamp, checksum):
    with Path(path).open("rb") as stream:
        measured = hashlib.file_digest(stream, "sha256").hexdigest()
    if measured != checksum or file_stamp(path) != stamp:
        raise ValueError("Frozen null exclusion input changed")


def verify_exclusion_file(reference):
    from evaluation_scope import reference as valid_reference

    if not valid_reference(reference):
        raise ValueError("Null exclusion input reference is malformed")
    path = (ROOT / reference["path"]).resolve()
    cached_exclusion_file(str(path), file_stamp(path), reference["sha256"])


def verified_exclusion(item, origin, source_id, index):
    verification = item.get("verification", {})
    expected = "FROZEN_CATALOGUE_EPHEMERIS" if origin == "CATALOGUE_EPHEMERIS" else "PINNED_VERIFIED_ADJUDICATION"
    if item.get("origin") != origin or verification.get("status") != expected:
        raise ValueError("Null exclusion is not a verified catalogue or adjudicated signal")
    support = exclusion_document(verification["prepared_support"])
    physical = {key: item[key] for key in ("duration_hours", "events")}
    if (
        verification.get("base_manifest_sha256") != index["base_manifest"]["sha256"]
        or support.get("base_manifest_sha256") != index["base_manifest"]["sha256"]
        or support.get("source_id") != source_id
        or support.get("original_preparation") != index["original_preparation"]
        or support.get("status") != "PREPARED_DISCOVERY_SUPPORT"
        or support.get("supported_references", {}).get(item["reference_id"]) != physical
    ):
        raise ValueError("Verified exclusion differs from its original prepared-quarter support")
    from deterministic.planet_control_report import match_event_units

    match_event_units(item["events"], [], item["duration_hours"])
    if not item["events"] or any(
        not event["support"] or any(
            type(counts.get("psf_flat")) is not int or counts["psf_flat"] <= 0
            for counts in event["support"].values()
        ) for event in item["events"]
    ):
        raise ValueError("Verified exclusion has no observed prepared-quarter support")
    verify_exclusion_file(support["prepared_fsc"])
    if origin == "CATALOGUE_EPHEMERIS":
        if verification.get("snapshot") != index["catalogue_snapshots"][verification["catalogue"]]["reference"]:
            raise ValueError("Catalogue exclusion snapshot differs from the frozen lookup")
    else:
        review = exclusion_document(verification["adjudication"])
        evidence = exclusion_document(verification["source_evidence"])
        if not (
            review.get("status") == "VERIFIED_ASTROPHYSICAL_SIGNAL"
            and review.get("source_id") == evidence.get("source_id") == source_id
            and review.get("evidence") == verification["source_evidence"]
            and review.get("raw_fits_sha256") == evidence.get("raw_fits_sha256") == support["raw_fits_sha256"]
            and all(review.get(key) for key in ("reviewer", "rationale"))
        ):
            raise ValueError("Internal exclusion lacks pinned verified adjudication")


def exclusion_references(manifest, source_id):
    if "null_exclusions" not in manifest:
        raise ValueError(
            "Null follow-up requires frozen real-signal and catalogue exclusions"
        )
    evidence = exclusion_document(manifest["null_exclusions"])
    row = evidence["sources"][source_id]
    validate_exclusion_index(evidence, manifest)
    if row.get("coverage_complete") is not True:
        raise ValueError("Null source lacks complete verified-only accounting")
    references = []
    for name, origin in (
        ("real_signals", "REAL_SIGNAL"),
        ("catalogue_ephemerides", "CATALOGUE_EPHEMERIS"),
    ):
        source = exclusion_document(row[name])
        if (
            source.get("source_id") != source_id
            or source.get("coverage_complete") is not True
            or source.get("coverage_scope") != EXCLUSION_COVERAGE
            or source.get("unknown_proposals") != "RETAIN_IN_NULL_TRIAL_COUNT"
            or source.get("base_manifest_sha256") != evidence["base_manifest"]["sha256"]
        ):
            raise ValueError("Null exclusion reference identity or coverage differs")
        items = source["references"]
        if origin == "CATALOGUE_EPHEMERIS":
            validate_catalogue_accounting(source, evidence)
        for item in items:
            verified_exclusion(item, origin, source_id, evidence)
        references.extend(items)
    return references


def validate_exclusion_index(evidence, manifest):
    if (
        evidence.get("contract") != NULL_CONTRACT
        or evidence.get("exclusion_rule") != EXCLUSION_RULE
        or evidence.get("coverage_scope") != EXCLUSION_COVERAGE
        or evidence.get("unknown_proposals") != "RETAIN_IN_NULL_TRIAL_COUNT"
    ):
        raise ValueError(
            "Null exclusions lack complete verified-only accounting with unknown proposals retained"
        )
    base = manifest.get("null_construction", {}).get("base_manifest")
    if base is not None and (base != evidence["base_manifest"] or manifest["role"] != evidence["role"]):
        raise ValueError("Null exclusion base or role differs from the original construction")
    if evidence.get("original_preparation") is not None:
        exclusion_document(evidence["original_preparation"])
    for snapshot in evidence["catalogue_snapshots"].values():
        verify_exclusion_file(snapshot["reference"])
    specification = exclusion_document(evidence["catalogue_specification"])["files"]
    if len(specification) != 6 or set(specification) != set(evidence["catalogue_snapshots"]):
        raise ValueError("Null exclusions require all six frozen snapshot lookups")
    if any(snapshot["reference"]["sha256"] != specification[name]["sha256"]
           for name, snapshot in evidence["catalogue_snapshots"].items()):
        raise ValueError("Null snapshot identities differ from the frozen specification")


def validate_catalogue_accounting(source, index):
    lookups, snapshots = source.get("lookups", {}), index["catalogue_snapshots"]
    if set(lookups) != set(snapshots) or any(
        lookup["snapshot"] != snapshots[name]["reference"]
        or lookup["status"] not in {"SCOPED_SNAPSHOT_LOOKUP", "SOURCE_COORDINATES_UNRESOLVED"}
        for name, lookup in lookups.items()
    ):
        raise ValueError("Per-source null catalogue lookup accounting is incomplete")
    for item in source["references"]:
        verification = item["verification"]
        if verification["row_index"] not in lookups[verification["catalogue"]]["matched_rows"]:
            raise ValueError("Catalogue exclusion is absent from its per-source lookup")


def null_flux_decisions(path):
    if not path:
        raise ValueError("Null flux-gate ledger is missing, duplicated or reordered")
    stages = [item["stage"] for item in path]
    if stages != list(FLUX_GATES[: len(stages)]):
        raise ValueError("Null flux-gate ledger is missing, duplicated or reordered")
    decisions = []
    for item in path:
        allowed = {"PASS", "CONTINUE_INCONCLUSIVE"}
        if item["stage"] != "instrument":
            allowed.add("DISMISS")
        if item["decision"] not in allowed:
            raise ValueError(f"Null {item['stage']} flux-gate decision is invalid")
        decisions.append(item["decision"])
    if len(path) != len(FLUX_GATES) and decisions[-1] != "DISMISS":
        raise ValueError("Null flux vetting is incomplete")
    return decisions


def null_proposal_outcome(proposal):
    exclusion = proposal["exclusion"]["status"]
    if exclusion not in {"ELIGIBLE", "EXCLUDED"}:
        raise ValueError("Null exclusion is unmeasured")
    if exclusion == "EXCLUDED":
        return {"vetted": False, "false_candidate": False, "excluded": True}
    if "gate_paths" in proposal:
        paths = proposal["gate_paths"]
        if not paths:
            raise ValueError("A vetted null proposal lacks fitted signal gate paths")
        measured = [
            null_proposal_outcome(
                {"exclusion": proposal["exclusion"], "gate_path": item["gate_path"]}
            )
            for item in paths
        ]
        return {
            "vetted": True,
            "false_candidate": any(item["false_candidate"] for item in measured),
            "excluded": False,
        }
    decisions = null_flux_decisions(proposal["gate_path"])
    return {
        "vetted": True,
        "false_candidate": decisions == ["PASS"] * len(FLUX_GATES),
        "excluded": False,
    }


def null_variant_counts(row):
    declared = row["declared_proposal_ids"]
    actual = [proposal["proposal_id"] for proposal in row["proposals"]]
    if len(declared) != len(set(declared)) or actual != declared:
        raise ValueError("Null proposal ledger lost, reordered or duplicated an alias")
    counts = {name: row[name] for name in ("emitted_alerts", "adjudicated_alerts")}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Null alert counts are invalid")
    counts.update(trials=0, outcomes=0, excluded_proposals=0)
    for proposal in row["proposals"]:
        result = null_proposal_outcome(proposal)
        for name, field in (
            ("trials", "vetted"),
            ("outcomes", "false_candidate"),
            ("excluded_proposals", "excluded"),
        ):
            counts[name] += result[field]
    counts["identity_match"] = row["exact_identity_match"] is True
    counts["complete"] = (
        row["status"] == "COMPLETE_NULL_FLUX_LEDGER"
        and counts["emitted_alerts"] == counts["adjudicated_alerts"]
        and counts["identity_match"]
    )
    return counts


def null_case_variants(case):
    quarters = case["usable_quarters"]
    if type(quarters) is not int or quarters < 1:
        raise ValueError("Null coverage is invalid")
    expected = ["inverted"] + (["quarter_scrambled"] if quarters > 2 else [])
    if [row["variant"] for row in case["variants"]] != expected:
        raise ValueError("Null variants differ from the short-coverage rule")
    if type(case["scientific_abstention"]) is not bool:
        raise ValueError("Null abstention state is invalid")
    return case["variants"]


def null_population(cases, expected_sources=None):
    if expected_sources is not None and len(cases) != expected_sources:
        raise ValueError("Null source population differs from the frozen specification")
    totals = {
        "trials": 0,
        "outcomes": 0,
        "excluded_proposals": 0,
        "sources": len(cases),
        "false_candidate_sources": 0,
        "completed_sources": 0,
        "emitted_alerts": 0,
        "adjudicated_alerts": 0,
        "identity_mismatch_sources": 0,
        "scientific_abstention_sources": 0,
    }
    variants = {
        name: {"sources": 0, "trials": 0, "outcomes": 0, "excluded_proposals": 0}
        for name in ("inverted", "quarter_scrambled")
    }
    for case in cases:
        rows = null_case_variants(case)
        measured = [null_variant_counts(row) for row in rows]
        for row, counts in zip(rows, measured):
            variant = variants[row["variant"]]
            variant["sources"] += 1
            for name in ("trials", "outcomes", "excluded_proposals"):
                variant[name] += counts[name]
                totals[name] += counts[name]
            for name in ("emitted_alerts", "adjudicated_alerts"):
                totals[name] += counts[name]
        totals["completed_sources"] += all(row["complete"] for row in measured)
        totals["identity_mismatch_sources"] += not all(row["identity_match"] for row in measured)
        totals["false_candidate_sources"] += any(row["outcomes"] for row in measured)
        totals["scientific_abstention_sources"] += case["scientific_abstention"]
    totals["variants"] = variants
    totals["source_fraction"] = (
        totals["false_candidate_sources"] / len(cases) if cases else None
    )
    return totals


def construction_preflight(args, path, base):
    if base.get("null_construction") or base.get("role") not in {"development", "evaluation"}:
        raise ValueError("Null construction requires original scoped inputs")
    if base["role"] == "development":
        return None
    policy_path = getattr(args, "disposition_policy", None)
    if policy_path is None or json.loads(policy_path.read_text()).get("status") != "FROZEN_EVALUATION":
        raise ValueError("Sealed null construction requires a frozen scoped evaluation policy")
    from deterministic.run import evaluation_preflight

    sources = [dict(item["metadata"], source_id=sid) for sid, item in base["sources"].items()]
    recipe = {"id": NULL_CONTRACT, "variant": args.variant, "seed": args.seed}
    scope = evaluation_preflight(args, sources, path, null_construction=recipe)
    if scope is None or scope["evaluation_role"] != "final_replay":
        raise ValueError("Sealed null construction requires the final replay scope")
    return scope


def construction_exclusions(args, path, base):
    if not args.exclusions:
        return None
    binding = {"path": str(args.exclusions.resolve()), "sha256": hashlib.sha256(args.exclusions.read_bytes()).hexdigest()}
    index = exclusion_document(binding)
    base_ref = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    validate_exclusion_index(index, {"role": base["role"], "null_construction": {"base_manifest": base_ref}})
    if list(index["sources"]) != list(base["sources"]):
        raise ValueError("Null exclusions changed the original ordered base population")
    return binding


def construct(args):
    path, base = input_manifest(ROOT, args.input_manifest)
    scope = construction_preflight(args, path, base)
    exclusions = construction_exclusions(args, path, base)
    out = dump_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    sources, ledger = {}, []
    order = list(base["sources"])
    scheduled = (
        sorted(order, key=lambda sid: base["sources"][sid]["lc2_folder"])
        if base["schema"] == 3
        else order
    )
    for sid in scheduled:
        metadata, raw = load_source(ROOT, sid, path)
        transformed, count = null_raw(raw, sid, args.variant, args.seed)
        ledger.append(
            {
                "source_id": sid,
                "quarters": count,
                "status": "INVERTED_ONLY_SHORT_COVERAGE"
                if transformed is None
                else "CONSTRUCTED",
            }
        )
        if transformed is None:
            continue
        arrays = out / f"{sid}.npz"
        np.savez_compressed(arrays, **transformed)
        sources[sid] = {
            "arrays": arrays.name,
            "sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
            "metadata": metadata,
            "null_quarters": count,
        }
    positions = {sid: index for index, sid in enumerate(order)}
    ledger.sort(key=lambda row: positions[row["source_id"]])
    sources = {sid: sources[sid] for sid in order if sid in sources}
    contract = {
        "id": NULL_CONTRACT,
        "variant": args.variant,
        "seed": args.seed,
        "normalized_relative": True,
        "base_manifest": {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }
    result = {
        "schema": 1,
        "role": base["role"],
        "measurement_scope": "SEALED_NULL_EVALUATION" if scope else "OPENED_NULL_DEVELOPMENT_ONLY",
        "control_role": "SEALED_FINAL_NULLS" if scope else "DEVELOPMENT_EXTRA",
        "survey_context_complete": base.get("survey_context_complete", False),
        "null_construction": contract,
        "sources": sources,
        "construction_ledger": ledger,
    }
    if scope:
        result["construction_scope"] = scope
    if "scene_metadata" in base:
        result["scene_metadata"] = {
            "path": str((path.parent / base["scene_metadata"]["path"]).resolve()),
            "sha256": base["scene_metadata"]["sha256"],
        }
    if exclusions:
        verify_exclusion_file(exclusions)
        result["null_exclusions"] = exclusions
    (out / "input-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "source-ids.json").write_text(json.dumps(list(sources)) + "\n")
    print(
        json.dumps(
            {
                "status": "CONSTRUCTED",
                "sources": len(sources),
                "contract_sha256": json_sha256(contract),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument(
        "--variant", required=True, choices=("inverted", "quarter_scrambled")
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--disposition-policy", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    construct(parser.parse_args())


if __name__ == "__main__":
    main()
