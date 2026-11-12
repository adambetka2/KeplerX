"""Bind scoped evaluation execution to frozen source and input identities."""

import hashlib
import json
import re

from runtime import ROOT, disposition_evaluation_token

ROLES = ("development_replay", "final_replay")


def sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def reference(value):
    return (
        isinstance(value, dict)
        and set(value) == {"path", "sha256"}
        and isinstance(value["path"], str)
        and bool(value["path"])
        and sha256(value["sha256"])
    )


def role_binding(value):
    if not isinstance(value, dict) or set(value) != {"controls_sha256", "cohorts"}:
        return False
    cohorts = value["cohorts"]
    return (
        sha256(value["controls_sha256"])
        and isinstance(cohorts, list)
        and bool(cohorts)
        and all(reference(item) for item in cohorts)
        and len({item["path"] for item in cohorts}) == len(cohorts)
    )


def read_reference(item):
    if not reference(item):
        raise ValueError("Evaluation input reference is malformed")
    path = (ROOT / item["path"]).resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise ValueError(f"Frozen evaluation input changed: {path}")
    return path, json.loads(payload)


def authorize(policy, token):
    if policy.get("status") != "FROZEN_EVALUATION":
        return None
    role = next(
        (role for role in ROLES if token == disposition_evaluation_token(policy, role)),
        None,
    )
    if role is None or not role_binding(policy.get("evaluation_binding", {}).get(role)):
        raise ValueError("Frozen evaluation requires a valid scoped role token")
    return role


def source_identity(row):
    identity = {
        "source_id": row.get("source_id"),
        "parent_tpf": str(row.get("parent_tpf", row.get("origin_tpf_kic", ""))),
        "raw_fits_sha256": row.get("raw_fits_sha256"),
    }
    if (
        not isinstance(identity["source_id"], str)
        or not identity["source_id"].isdigit()
        or not identity["parent_tpf"].isdigit()
        or int(identity["parent_tpf"]) <= 0
        or not sha256(identity["raw_fits_sha256"])
    ):
        raise ValueError("Evaluation source identity is incomplete")
    return identity


def pixel_input_matches(cohort, binding):
    expected = cohort.get("pixel_inputs")
    if expected is None:
        return binding is None
    expected_path, _ = read_reference(expected)
    if binding is None:
        return False
    configured = {key: binding[key] for key in ("path", "sha256")}
    path, _ = read_reference(configured)
    role = {"development_replay": "development", "final_replay": "evaluation"}
    return (
        path == expected_path and configured["sha256"] == expected["sha256"]
        and binding["role"] == role[cohort["role"]]
    )


def cohort_input_matches(cohort, input_manifest, null_construction=None):
    configured = cohort["input_manifest"]
    recipe = cohort.get("null_construction")
    if configured is None:
        return input_manifest is None and recipe is None and null_construction is None
    expected_path, _ = read_reference(configured)
    if input_manifest is None:
        return False
    actual_path = (ROOT / input_manifest).resolve()
    if recipe is None:
        return actual_path == expected_path and null_construction is None
    from deterministic.signal_contracts import NULL_CONTRACT

    if (
        set(recipe) != {"id", "variant", "seed"}
        or recipe["id"] != NULL_CONTRACT
        or recipe["variant"] not in {"inverted", "quarter_scrambled"}
        or type(recipe["seed"]) is not int
        or recipe["seed"] < 0
    ):
        raise ValueError("Frozen null construction recipe is invalid")
    if null_construction is not None:
        return actual_path == expected_path and null_construction == recipe
    derived = json.loads(actual_path.read_text())
    contract = derived.get("null_construction", {})
    base = contract.get("base_manifest")
    if not reference(base):
        return False
    base_path, _ = read_reference(base)
    expected_role = {"development_replay": "development", "final_replay": "evaluation"}
    return (
        {key: contract.get(key) for key in recipe} == recipe
        and contract.get("normalized_relative") is True
        and derived.get("role") == expected_role[cohort["role"]]
        and base_path == expected_path
        and base["sha256"] == configured["sha256"]
    )


def cohort_sources(cohort, input_manifest, null_construction):
    expected = [source_identity(row) for row in cohort.get("sources", [])]
    if not expected or len({row["source_id"] for row in expected}) != len(expected):
        raise ValueError("Frozen evaluation cohort is empty or repeats a source")
    recipe = cohort.get("null_construction", {})
    if null_construction is not None or recipe.get("variant") != "quarter_scrambled":
        return expected
    derived = json.loads((ROOT / input_manifest).read_text())
    ledger = derived.get("construction_ledger", [])
    if [row["source_id"] for row in ledger] != [row["source_id"] for row in expected]:
        raise ValueError("Null construction ledger does not cover the frozen source population")
    for row in ledger:
        count = row["quarters"]
        if type(count) is not int or count < 1:
            raise ValueError("Null construction coverage is invalid")
        status = "CONSTRUCTED" if count > 2 else "INVERTED_ONLY_SHORT_COVERAGE"
        if row["status"] != status:
            raise ValueError("Null construction violates the short-coverage rule")
    return [row for row, coverage in zip(expected, ledger) if coverage["quarters"] > 2]


def verify_null_construction_scope(cohort, input_manifest, null_construction, matched):
    if (
        cohort.get("null_construction")
        and null_construction is None
        and matched["evaluation_role"] == "final_replay"
    ):
        derived = json.loads((ROOT / input_manifest).read_text())
        if derived.get("construction_scope") != matched:
            raise ValueError("Sealed null construction scope changed")


def match_cohort(
    policy, role, sources, input_manifest=None, subset=False, pixel_inputs=None,
    null_construction=None,
):
    actual = [source_identity(row) for row in sources]
    by_id = {row["source_id"]: row for row in actual}
    if len(by_id) != len(actual):
        raise ValueError("Evaluation execution repeats a source identity")
    binding = policy.get("evaluation_binding", {}).get(role)
    if role not in ROLES or not role_binding(binding):
        raise ValueError("Evaluation execution has no frozen cohort binding")
    matches = []
    for item in binding["cohorts"]:
        _, cohort = read_reference(item)
        if (
            cohort.get("schema") != 1
            or cohort.get("status") != "FROZEN_EVALUATION_COHORT"
            or cohort.get("role") != role
            or not isinstance(cohort.get("cohort_id"), str)
            or not cohort["cohort_id"]
            or "input_manifest" not in cohort
        ):
            raise ValueError("Evaluation cohort is not frozen for the authorized role")
        manifest_matches = cohort_input_matches(cohort, input_manifest, null_construction)
        if not manifest_matches:
            continue
        expected = cohort_sources(cohort, input_manifest, null_construction)
        expected_ids = {row["source_id"]: row for row in expected}
        if len(expected_ids) != len(expected):
            raise ValueError("Frozen evaluation cohort is empty or repeats a source")
        identities_match = (
            all(row == expected_ids.get(sid) for sid, row in by_id.items())
            if subset
            else actual == expected
        )
        pixels_match = null_construction is not None or pixel_input_matches(cohort, pixel_inputs)
        if identities_match and pixels_match:
            matched = {
                    "cohort_id": cohort["cohort_id"],
                    "cohort": item,
                    "evaluation_role": role,
                    "definition_sha256": policy.get("definition_sha256"),
                }
            verify_null_construction_scope(cohort, input_manifest, null_construction, matched)
            matches.append(matched)
    if len(matches) != 1:
        raise ValueError(
            "Actual evaluation source/input cohort does not match exactly one authorized scope"
        )
    return matches[0]


def queue_source_identities(source_ids, input_manifest):
    if input_manifest is not None:
        path = (ROOT / input_manifest).resolve()
        manifest = json.loads(path.read_text())
        return [
            source_identity({**manifest["sources"][sid]["metadata"], "source_id": sid})
            for sid in source_ids
        ]
    from deterministic.control_partitions import parent_sources

    parents, raw, _ = parent_sources(
        ROOT, ROOT / "discSSD/data/index.sqlite", source_ids
    )
    return [
        {"source_id": sid, "parent_tpf": parents[sid], "raw_fits_sha256": raw[sid]}
        for sid in source_ids
    ]
