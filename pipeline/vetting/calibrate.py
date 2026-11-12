"""Set the disposition policy on the opened development controls.

Every stored record-mode vetting record is replayed through the pipeline's own
triage gates and adjudication under each policy vector of a declared grid.
Coordinate descent maximizes planet recovery weighted to the sealed strata,
subject to exact binomial false-BOI caps; ties resolve toward fewer candidates.
"""

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/vetting"),
    str(ROOT / "pipeline/photometry"),
]

import vetting_triage
from runtime import (
    EVENT_MATCHING_CONTRACT,
    OUTCOME_CONTRACT,
    artifact_path,
    disposition_calibration_parameters,
    dump_path,
    implementation_hashes,
    worker_count,
)
from vetting import (
    frozen_assessment_rules,
    pipeline_contract,
    replay_case_ledger,
    replay_identity_sets,
)
from vetting_disposition import adjudicate

from pipeline.deterministic.catalogue_control_report import (
    emitted_candidate,
    exact_upper_bound,
    incomplete_disposition,
)

STAGES = ("signals", "orbital", "instrument", "localization", "contamination")
COMPARED = ("signals", "triage", "disposition")
RUN_CONFIGS = {}
DRAFT_SPECIFICATION_SHA256 = "72cf8759cd01b4c93560f0157ea2e170b93c764010f58f3caf4ac77388ef1370"
LAUNCH_FALSE_BOI_CAPS = {
    "null": 0.005,
    "native_eb": 0.05,
    "ordinary": 0.01,
    "background_contaminant": 0.03,
    "contamination_pair": 0.03,
    "guide_star_family": 0.03,
}
OBJECTIVE_METADATA = {
    "periodic_planet": ("planets", "periodic_planet", "periodic_injection"),
    "sparse_injection": ("sparse", "morphology", "sparse_injection"),
    "ttv_injection": ("ttv", "morphology", "ttv_injection"),
}


class NullCounts(NamedTuple):
    vetted: int
    false_candidates: int
    excluded: int


def null_outcome(proposals):
    counts = Counter()
    for proposal in proposals:
        if (
            set(proposal) != {"vetted", "false_candidate", "excluded"}
            or any(type(value) is not bool for value in proposal.values())
            or proposal["vetted"] == proposal["excluded"]
            or (proposal["false_candidate"] and not proposal["vetted"])
        ):
            raise ValueError("Null proposal endpoint is missing or inconsistent")
        counts.update({key: int(value) for key, value in proposal.items()})
    result = NullCounts(counts["vetted"], counts["false_candidate"], counts["excluded"])
    return result, result.false_candidates, False


def cohort_counts(case, hit):
    if case["truth_cohort"] != "null":
        return 1, hit
    if (
        not isinstance(hit, NullCounts)
        or any(type(value) is not int or value < 0 for value in hit)
        or hit.false_candidates > hit.vetted
    ):
        raise ValueError("Null calibration requires vetted-proposal counts")
    return hit.vetted, hit.false_candidates


def null_source_summary(cases, outcomes):
    null = [
        hit for case, (hit, _, _) in zip(cases, outcomes)
        if case["truth_cohort"] == "null"
    ]
    if not null:
        return {}
    return {"null_sources": {
        "trials": len(null),
        "vetted_sources": sum(hit.vetted > 0 for hit in null),
        "false_candidate_sources": sum(hit.false_candidates > 0 for hit in null),
        "excluded_proposals": sum(hit.excluded for hit in null),
    }}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def reference(path):
    path = resolve(path)
    return {
        "path": artifact_path(path, ROOT),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def verified(ref):
    data = resolve(ref["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError(f"Calibration input changed: {ref['path']}")
    return data


def development_cases(report):
    cases = [
        *report.get("cases", []),
        *report.get("morphology_cases", []),
        *report.get("diagnostic_cases", []),
    ]
    nulls = report.get("null_cases", [])
    if nulls:
        from pipeline.deterministic.signal_contracts import null_population

        measured = null_population(nulls)
        if measured["completed_sources"] != len(nulls):
            raise ValueError("Calibration requires complete null proposal execution")
    if (
        report.get("replay_role") != "development_replay"
        or report.get("status") != "OPENED_REPORT_ASSEMBLED"
        or report.get("outcome_contract") != OUTCOME_CONTRACT
        or report.get("event_matching_contract") != EVENT_MATCHING_CONTRACT
        or not cases and not nulls
        or not all(replay_case_ledger(case)["complete"] for case in cases)
    ):
        raise ValueError("Calibration requires the complete opened development replay")
    verify_report_inputs(report)
    replay_identity_sets(report)
    return [*cases, *nulls]


def verify_report_inputs(report):
    inputs = report.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("The assembled calibration report lacks input hash bindings")
    for path, checksum in inputs.items():
        verified({"path": path, "sha256": checksum})


def case_records(case):
    return case["variants"] if case.get("truth_cohort") == "null" else [case]


def run_directory(case):
    if case["source_evidence"]:
        return str(resolve(case["source_evidence"]["path"]).parents[2])
    return str(resolve(case["proposal_evidence"]["path"]).parents[1])


def development_runs(cases, base):
    if not base.get("pipeline_contract") or not base.get("implementation"):
        raise ValueError("Freeze the implementation and pipeline contract first")
    if base["implementation"] != implementation_hashes():
        raise ValueError("The implementation differs from the base policy")
    configs, detectors, recipes = {}, set(), []
    runs = {
        run_directory(record) for case in cases for record in case_records(case)
        if record["source_evidence"] or case.get("truth_cohort") == "null"
    }
    null_runs = {
        run_directory(record) for case in cases if case.get("truth_cohort") == "null"
        for record in case_records(case)
    }
    for run in runs:
        config_path = Path(run) / "vetting.json"
        config = json.loads(config_path.read_text())
        manifest = json.loads((Path(run) / "vetting/manifest.json").read_text())
        if (
            manifest["code_sha256"] != base["implementation"]
            or manifest["config_sha256"] != reference(config_path)["sha256"]
            or pipeline_contract(config) != base["pipeline_contract"]
            or config["triage_execution"] != "RECORD_EVERY_GATE"
        ):
            raise ValueError(f"Run is not a frozen record-every-gate replay: {run}")
        if run in null_runs:
            recipes.append(verify_null_run(run))
        detection = json.loads((Path(run) / "detection.json").read_text())
        detectors.add(detection["binary"]["sha256"])
        configs[run] = config
    if len(detectors) != 1:
        raise ValueError("Development runs used different detector binaries")
    if any(recipe != recipes[0] for recipe in recipes):
        raise ValueError("Null variants differ in base inputs, seed or exclusions")
    return configs, detectors.pop()


def verify_null_run(run):
    from pipeline.deterministic.signal_contracts import null_manifest

    reference = json.loads((Path(run) / "preparation.json").read_text())["input_manifest"]
    specification = json.loads(verified(reference))
    contract = null_manifest(ROOT, reference["path"])
    if specification["role"] != "development" or contract is None:
        raise ValueError("Calibration nulls require opened frozen construction inputs")
    return {
        "id": contract["id"], "seed": contract["seed"],
        "base_manifest": contract["base_manifest"],
        "null_exclusions": specification["null_exclusions"],
    }


def check_spec(spec, cases):
    if spec.get("status") in {"DRAFT_UNBOUND", "BOUND_METADATA_INPUTS_PENDING"}:
        raise ValueError("Calibration metadata bindings lack verified opened records")
    classes = {
        case["truth_cohort"]: "null" if case["truth_cohort"] == "null" else case["truth_class"]
        for case in cases
    }
    objective = {term["opened_cohort"] for term in spec["objective"]}
    capped = set(spec["false_boi_caps"])
    diagnostic = set(spec["diagnostic_cohorts"])
    if (
        objective & capped
        or objective & diagnostic
        or capped & diagnostic
        or set(classes) != objective | capped | diagnostic
        or any(classes[cohort] != "planet" for cohort in objective)
        or any(classes[cohort] == "planet" for cohort in capped)
    ):
        raise ValueError(
            "Each opened cohort must be exactly one of an objective planet cohort, "
            "a capped false cohort or a declared diagnostic"
        )
    if not 0 < spec["confidence"] < 1:
        raise ValueError("The false-BOI cap confidence must lie between 0 and 1")
    for path, values in spec["grid"].items():
        if (
            path.split(".")[0] not in {"policy", "triage"}
            or path.startswith("triage.assessment_rules")
            or not values
            or len(values) != len({json.dumps(value) for value in values})
            or spec["start"].get(path) not in values
        ):
            raise ValueError(f"Invalid calibration grid parameter: {path}")


def read_strata(ref):
    rows = [json.loads(line) for line in verified(ref).decode().splitlines() if line]
    strata = {str(row["source_id"]): row["stratum"] for row in rows}
    if len(strata) != len(rows):
        raise ValueError(f"Truth repeats a source: {ref['path']}")
    return strata


def declared_strata(spec, cases):
    strata = {}
    for term in spec["objective"]:
        opened = read_strata(term["opened_truth"])
        sealed = Counter(read_strata(term["sealed_truth"]).values())
        sources = {
            case["source_id"]
            for case in cases
            if case["truth_cohort"] == term["opened_cohort"]
        }
        if set(opened) != sources or not set(sealed) <= set(opened.values()):
            raise ValueError(
                f"Opened strata must cover the cohort and every sealed stratum: "
                f"{term['name']}"
            )
        strata[term["name"]] = opened, sealed
    return strata


def candidate(base, start, vector):
    policy = copy.deepcopy(base)
    policy.update(
        status="FROZEN_EVALUATION",
        _evaluation_authorized=True,
        boi_catalogue_enabled=True,
    )
    for rule in policy["assessment_rules"].values():
        rule["status"] = "RELEASED"
    triage = copy.deepcopy(policy["pipeline_contract"]["vetting"]["triage"])
    for path, value in {**start, **vector}.items():
        scope, *keys = path.split(".")
        target = {"policy": policy, "triage": triage}[scope]
        for key in keys[:-1]:
            target = target[key]
        if keys[-1] not in target:
            raise ValueError(f"Unknown calibration parameter: {path}")
        target[keys[-1]] = copy.deepcopy(value)
    triage["assessment_rules"] = policy["assessment_rules"]
    policy["pipeline_contract"]["vetting"]["triage"] = triage
    if not frozen_assessment_rules(policy["assessment_rules"]):
        raise ValueError("A calibration vector leaves a policy value undeclared")
    return policy, triage


def replay(row, config, policy):
    for signal in row["signals"]:
        signal["triage"] = {"status": "ACTIVE", "history": []}
    for stage in STAGES:
        vetting_triage.apply_gate(row, stage, config)
    if policy is not None:
        row["disposition_policy"] = policy
    row["disposition"] = adjudicate(row)


def outcome(case, row):
    if case.get("truth_cohort") == "null":
        raise ValueError("Null calibration requires replayed proposal flux endpoints")
    candidates = {
        item["signal_id"]
        for item in row["disposition"]["signals"]
        if item["disposition"] == "BOI"
    }
    incomplete = incomplete_disposition(row)
    if case["truth_class"] == "planet":
        # Fitted units come from the policy-independent signal shape, so the
        # report's event match holds under every vector.
        matched = {
            signal["signal_id"]
            for signal in case["signals"]
            if signal["fitted_event_match"]["status"] == "MATCHED"
        }
        hit = not incomplete and bool(candidates & matched)
    else:
        hit = emitted_candidate(row["disposition"])
    return hit, len(candidates), incomplete


def load_record(case):
    row = json.loads(verified(case["source_evidence"]))
    if row.get("reviews") or not all(
        signal.get("triage", {}).get("record_every_gate") is True
        for signal in row["signals"]
    ):
        raise ValueError(
            f"Calibration requires unreviewed record-every-gate records: "
            f"{case['source_id']}"
        )
    return row


def canonical(row):
    return json.dumps({key: row.get(key) for key in COMPARED}, sort_keys=True)


def init_worker(configs):
    RUN_CONFIGS.update(configs)


def null_records(case):
    from pipeline.deterministic.signal_contracts import null_case_variants

    for variant in null_case_variants(case):
        run = run_directory(variant)
        reference = json.loads((Path(run) / "preparation.json").read_text())["input_manifest"]
        specification = json.loads(verified(reference))
        proposal = json.loads(verified(variant["proposal_evidence"]))
        row = load_record({**variant, "source_id": case["source_id"]}) if variant["source_evidence"] else None
        original = specification["sources"][case["source_id"]]
        if (
            specification["role"] != "development"
            or specification["null_construction"]["variant"] != variant["variant"]
            or original["sha256"] != variant["input_sha256"]
            or original["null_quarters"] != case["usable_quarters"]
            or str(original["metadata"]["origin_tpf_kic"]) != case["parent_tpf"]
            or [original["metadata"]["raw_fits_sha256"]] != case["raw_fits_sha256"]
            or proposal["source_id"] != case["source_id"]
            or row is not None and row["source_id"] != case["source_id"]
        ):
            raise ValueError("Null source, variant or original input identity changed")
        yield variant, run, specification, proposal, row


def null_variant(case, variant, specification, proposal, row):
    from pipeline.deterministic.release_report import null_variant_record
    from pipeline.deterministic.signal_contracts import null_variant_counts

    result = null_variant_record(proposal, row, specification, case["source_id"], variant["variant"])
    if not null_variant_counts(result)["complete"]:
        raise ValueError("Replayed null proposal execution is incomplete")
    return result


def null_identity(case):
    for variant, run, specification, proposal, row in null_records(case):
        if row is not None:
            stored = canonical(row)
            replay(row, RUN_CONFIGS[run], row.get("disposition_policy"))
            if canonical(row) != stored:
                return False
        result = null_variant(case, variant, specification, proposal, row)
        if result != {key: variant[key] for key in result}:
            return False
    return True


def evaluate_null(case, candidates):
    from pipeline.deterministic.signal_contracts import null_proposal_outcome

    outcomes = [[] for _ in candidates]
    for variant, run, specification, proposal, row in null_records(case):
        for position, (policy, triage) in enumerate(candidates):
            if row is not None:
                replay(row, {**RUN_CONFIGS[run], "triage": triage}, policy)
            result = null_variant(case, variant, specification, proposal, row)
            if result["declared_proposal_ids"] != variant["declared_proposal_ids"]:
                raise ValueError("A policy vector changed the fixed null proposal population")
            outcomes[position].extend(null_proposal_outcome(item) for item in result["proposals"])
    return [null_outcome(items) for items in outcomes]


def identity_shard(shard):
    mismatches = []
    for _, case, run in shard:
        if case["truth_cohort"] == "null":
            if not null_identity(case):
                mismatches.append(case["source_id"])
            continue
        row = load_record(case)
        stored = canonical(row)
        replay(row, RUN_CONFIGS[run], row.get("disposition_policy"))
        if canonical(row) != stored:
            mismatches.append(case["source_id"])
    return mismatches


def evaluate_shard(shard, candidates):
    results = [[] for _ in candidates]
    for index, case, run in shard:
        if case["truth_cohort"] == "null":
            for target, value in zip(results, evaluate_null(case, candidates), strict=True):
                target.append((index, value))
            continue
        row = load_record(case)
        for position, (policy, triage) in enumerate(candidates):
            replay(row, {**RUN_CONFIGS[run], "triage": triage}, policy)
            results[position].append((index, outcome(case, row)))
    return results


def shards(cases, count):
    items = [
        (index, case, None if case.get("truth_cohort") == "null" else run_directory(case))
        for index, case in enumerate(cases)
        if case["source_evidence"] or case.get("truth_cohort") == "null"
    ]
    sizes = {
        index: sum(
            resolve(record[key]["path"]).stat().st_size
            for record in case_records(case) for key in ("source_evidence", "proposal_evidence")
            if record.get(key)
        )
        for index, case, _ in items
    }
    groups, loads = [[] for _ in range(count)], [0] * count
    for item in sorted(items, key=lambda item: -sizes[item[0]]):
        lightest = loads.index(min(loads))
        groups[lightest].append(item)
        loads[lightest] += sizes[item[0]]
    return [group for group in groups if group]


def summarize(cases, outcomes, spec, strata):
    if len(outcomes) != len(cases):
        raise ValueError(
            f"Calibration outcomes do not reconcile: "
            f"expected {len(cases)}, received {len(outcomes)}"
        )
    trials, hits = Counter(), Counter()
    for case, (hit, _, _) in zip(cases, outcomes):
        count, recovered = cohort_counts(case, hit)
        trials[case["truth_cohort"]] += count
        hits[case["truth_cohort"]] += recovered
    terms = {}
    for term in spec["objective"]:
        opened, sealed = strata[term["name"]]
        size = Counter(opened.values())
        recovered = Counter(
            opened[case["source_id"]]
            for case, (hit, _, _) in zip(cases, outcomes)
            if hit and case["truth_cohort"] == term["opened_cohort"]
        )
        terms[term["name"]] = {
            "opened_trials": len(opened),
            "opened_recovered": sum(recovered.values()),
            "sealed_trials": sum(sealed.values()),
            "expected_sealed_recoveries": sum(
                count * recovered[stratum] / size[stratum]
                for stratum, count in sealed.items()
            ),
        }
    false_boi = {
        cohort: {
            "trials": trials[cohort],
            "false_boi": hits[cohort],
            "upper_bound": exact_upper_bound(
                hits[cohort], trials[cohort], spec["confidence"]
            ),
            "cap": cap,
        }
        for cohort, cap in spec["false_boi_caps"].items()
    }
    incomplete = sum(item[2] for item in outcomes)
    candidates = sum(item[1] for item in outcomes)
    shortfall = sum(
        min(0.0, item["cap"] - item["upper_bound"]) for item in false_boi.values()
    )
    expected = sum(term["expected_sealed_recoveries"] for term in terms.values())
    feasible = shortfall == 0 and incomplete == 0
    summary = {
        "feasible": feasible,
        "expected_sealed_recoveries": expected,
        "objective": terms,
        "false_boi": false_boi,
        "diagnostic": {
            cohort: {"trials": trials[cohort], "candidate_cases": hits[cohort]}
            for cohort in spec["diagnostic_cohorts"]
        },
        "boi_signals": candidates,
        "incomplete_cases": incomplete,
        "score": [feasible, -incomplete, shortfall, expected, -candidates],
    }
    summary.update(null_source_summary(cases, outcomes))
    return summary


def search(evaluate, grid, start):
    current = {name: start[name] for name in grid}
    cache, trace, moves = {}, [], []

    def summaries(vectors, sweep, parameter):
        new = [v for v in vectors if json.dumps(v, sort_keys=True) not in cache]
        for vector, summary in zip(new, evaluate(new) if new else [], strict=True):
            cache[json.dumps(vector, sort_keys=True)] = summary
            trace.append(
                {
                    "sweep": sweep,
                    "parameter": parameter,
                    "vector": vector,
                    "score": summary["score"],
                    "false_boi": {
                        cohort: item["false_boi"]
                        for cohort, item in summary["false_boi"].items()
                    },
                }
            )
        return [cache[json.dumps(v, sort_keys=True)] for v in vectors]

    best = summaries([current], 0, None)[0]
    sweep, moved = 0, True
    while moved:
        sweep, moved = sweep + 1, False
        for name, values in grid.items():
            vectors = [{**current, name: value} for value in values]
            results = summaries(vectors, sweep, name)
            top = max(range(len(vectors)), key=lambda i: results[i]["score"])
            if results[top]["score"] > best["score"]:
                moves.append(
                    {
                        "sweep": sweep,
                        "parameter": name,
                        "from": current[name],
                        "to": vectors[top][name],
                    }
                )
                current, best, moved = vectors[top], results[top], True
            print(f"sweep {sweep} {name} score {best['score']}", file=sys.stderr)
    return (
        current,
        best,
        {
            "method": "coordinate_descent_strict_improvement",
            "sweeps": sweep,
            "vectors_evaluated": len(cache),
            "moves": moves,
            "trace": trace,
        },
    )


def reconcile_outcomes(cases, count, parts):
    expected = {
        i for i, case in enumerate(cases)
        if case["source_evidence"] or case.get("truth_cohort") == "null"
    }
    reported = [{} for _ in range(count)]
    for part in parts:
        if len(part) != count:
            raise ValueError("Calibration shard policy vectors do not reconcile")
        for target, items in zip(reported, part, strict=True):
            for index, value in items:
                if type(index) is not int or index not in expected or index in target:
                    raise ValueError(f"Unexpected or repeated calibration case: {index}")
                target[index] = value
    if any(set(target) != expected for target in reported):
        raise ValueError("Calibration shard outcomes omit a recorded case")
    return [
        [target.get(i, null_outcome([]) if case.get("truth_cohort") == "null" else (False, 0, False))
         for i, case in enumerate(cases)]
        for target in reported
    ]


def optimize(cases, configs, base, spec, strata, workers):
    candidate(base, spec["start"], {})
    parts = shards(cases, 4 * workers)
    with ProcessPoolExecutor(
        workers, initializer=init_worker, initargs=(configs,)
    ) as pool:
        mismatches = [item for part in pool.map(identity_shard, parts) for item in part]
        if mismatches:
            raise ValueError(f"Stored records do not replay identically: {mismatches}")

        def evaluate(vectors):
            candidates = [candidate(base, spec["start"], vector) for vector in vectors]
            outcomes = reconcile_outcomes(
                cases, len(vectors), pool.map(evaluate_shard, parts, repeat(candidates))
            )
            return [summarize(cases, items, spec, strata) for items in outcomes]

        return search(evaluate, spec["grid"], spec["start"])


def verify_specification(spec):
    from pipeline.deterministic.control_bindings import calibration_summary

    if spec.get("status") in {"DRAFT_UNBOUND", "BOUND_METADATA_INPUTS_PENDING"}:
        raise ValueError("Calibration metadata bindings lack verified opened records")
    metadata = spec.get("metadata")
    original = ROOT / "pipeline/deterministic/cloud-control-metadata.json"
    truth = spec.get("opened_truth_descriptor")
    descriptor = ROOT / "pipeline/deterministic/cloud-opened-source-metadata.json"
    if (
        spec.get("original_draft_sha256") != DRAFT_SPECIFICATION_SHA256
        or not isinstance(metadata, dict)
        or metadata.get("sha256") != reference(original)["sha256"]
        or not isinstance(truth, dict)
        or truth.get("sha256") != reference(descriptor)["sha256"]
    ):
        raise ValueError("Calibration requires the approved metadata-bound specification")
    calibration_summary(
        json.loads(verified(metadata)),
        {**spec, "status": "BOUND_METADATA_INPUTS_PENDING"},
        json.loads(verified(truth)),
    )


def calibrate(report_path, spec_path, base_path, out, workers):
    spec = json.loads(Path(spec_path).read_text())
    verify_specification(spec)
    report, base = (
        json.loads(Path(path).read_text()) for path in (report_path, base_path)
    )
    cases = development_cases(report)
    configs, detector = development_runs(cases, base)
    check_spec(spec, cases)
    strata = declared_strata(spec, cases)
    vector, best, record = optimize(cases, configs, base, spec, strata, workers)
    policy, triage = candidate(base, spec["start"], vector)
    result = {
        "schema": 1,
        "status": "CALIBRATED" if best["feasible"] else "NO_FEASIBLE_POLICY",
        "spec": reference(spec_path),
        "development_evidence": reference(report_path),
        "base_policy": reference(base_path),
        "detector_sha256": detector,
        "vector": vector,
        "outcomes": best,
        "search": record,
    }
    if best["feasible"]:
        result.update(
            parameters=disposition_calibration_parameters(policy), triage=triage
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


def bind_objective_metadata(metadata, term, folder):
    cohort, family, category = OBJECTIVE_METADATA[term["opened_cohort"]]
    sources = {row["source_id"] for row in metadata["opened_cohorts"][cohort]["rows"]}
    for role, partition in (("opened", "development"), ("sealed", "evaluation")):
        origin = metadata["strata"][f"{family}-{partition}-strata.jsonl"]
        rows = [row for row in origin["rows"] if row["category"] == category]
        identities = {row["source_id"] for row in rows}
        if (
            not rows or len(identities) != len(rows)
            or any(set(row) != {"source_id", "category", "stratum"} for row in rows)
            or (role == "opened" and identities != sources)
        ):
            raise ValueError(f"Objective strata do not reconcile: {term['name']} {role}")
        path = folder / f"{term['name']}-{role}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        term[f"{role}_truth"] = {**reference(path), "origin_metadata_sha256": origin["sha256"]}


def prepare_bindings(metadata_path, out):
    from pipeline.deterministic.control_partitions import select_opened_parents

    metadata = json.loads(verified(reference(metadata_path)))
    draft = metadata["draft_specification"]
    if (
        metadata.get("schema") != 1 or metadata.get("status") != "METADATA_ONLY"
        or draft["sha256"] != DRAFT_SPECIFICATION_SHA256
        or hashlib.sha256(draft["text"].encode()).hexdigest() != draft["sha256"]
    ):
        raise ValueError("Control binding metadata or original draft hash is invalid")
    spec = json.loads(draft["text"])
    if spec.get("status") != "DRAFT_UNBOUND" or spec.get("confidence") != 0.95:
        raise ValueError("Binding requires the unchanged launch draft")
    cohorts = metadata["opened_cohorts"]
    if set(cohorts) != {"ordinary", "planets", "native-eb", "sparse", "ttv", "background"} or any(
        len(item["rows"]) != item["sources"]
        or len({row["source_id"] for row in item["rows"]}) != item["sources"]
        for item in cohorts.values()
    ):
        raise ValueError("Opened metadata source counts do not reconcile")
    rows = [row for item in cohorts.values() for row in item["rows"]]
    if any(
        len({row[key] for row in rows}) != len(rows)
        for key in ("source_id", "parent_tpf", "raw_fits_sha256", "cloud_arrays_sha256")
    ):
        raise ValueError("Opened metadata source, parent or input identities repeat")
    folder = out.parent / (out.stem + "-strata")
    folder.mkdir(parents=True, exist_ok=True)
    for term in spec["objective"]:
        bind_objective_metadata(metadata, term, folder)
    excluded = set().union(*(set(item["source_ids"]) for item in metadata["sealed_exclusions"].values()))
    parents = {row["parent_tpf"] for item in cohorts.values() for row in item["rows"]}
    selections = {}
    for name in ("guide-star-opened-pool-current-sealed-safe.csv", "catalogued-parent-opened-pool-current-sealed-safe.csv"):
        selection = select_opened_parents(metadata["prospective_pools"][name]["rows"], 100, excluded, parents)
        selections[name] = selection
        parents.update(row["parent_tpf"] for row in selection["selection"])
    spec.update(
        status="BOUND_METADATA_INPUTS_PENDING",
        false_boi_caps=dict(LAUNCH_FALSE_BOI_CAPS),
        metadata=reference(metadata_path),
        opened_truth_descriptor=reference(ROOT / "pipeline/deterministic/cloud-opened-source-metadata.json"),
        original_draft_sha256=draft["sha256"],
        opened_cohorts={name: {key: value for key, value in item.items() if key != "rows"} for name, item in cohorts.items()},
        prospective_selections=selections,
        prerequisites={
            "opened_record_runs": "NOT_PROVISIONED",
            "prospective_cohorts": "NOT_FROZEN",
            "joint_sealed_parent_independence": "NOT_VERIFIED",
            "null_flux_gate_ledger": "OPENED_EXECUTION_RECORDS_NOT_PROVISIONED",
        },
    )
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-metadata", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--base-policy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=worker_count())
    args = parser.parse_args()
    if args.bind_metadata:
        if any((args.report, args.spec, args.base_policy)):
            parser.error("Metadata preparation cannot evaluate opened records")
        spec = prepare_bindings(args.bind_metadata, dump_path(args.out))
        print(json.dumps({"status": spec["status"], "out": str(args.out)}))
        return
    if not all((args.report, args.spec, args.base_policy)):
        parser.error("Calibration requires --report, --spec and --base-policy")
    result = calibrate(
        args.report, args.spec, args.base_policy, dump_path(args.out), args.workers
    )
    print(json.dumps({key: result[key] for key in ("status", "vector")}))
    if result["status"] != "CALIBRATED":
        sys.exit("No policy vector on the declared grid meets every false-BOI cap")


if __name__ == "__main__":
    main()
