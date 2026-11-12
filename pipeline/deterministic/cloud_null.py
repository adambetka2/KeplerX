"""Provision an unselected null identity pool without opening scientific inputs."""

import csv
import gzip
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = (
    "source_id", "shard", "origin_parent", "raw_fits_sha256",
    "metadata_dependency_component_id", "linked_existing_scopes",
    "linked_full_witness_component_ids", "detector_exposure",
    "vetting_exposure", "disposition_exposure",
)
FLAGS = (
    "scope_frozen", "cohorts_selected", "roles_frozen", "seeds_frozen",
    "null_arrays_generated", "scientific_outcomes_provisioned", "scientific_acceptance",
    "joint_independence_complete", "original_local_inputs_reverified",
)
EXPOSURE = {
    "detector_exposure": "COMMITTED_STOPPED_DEVELOPMENT_DETECTOR",
    "vetting_exposure": "UNRESOLVED_BY_IDENTITY_METADATA",
    "disposition_exposure": "UNRESOLVED_BY_IDENTITY_METADATA",
}
COUNT_FIELDS = (
    "sources", "origin_parents", "raw_fits_hashes", "metadata_dependency_components",
    "shards", "ordered_identity_metadata_sha256", "linked_scope_sources",
    "linked_scope_components", "component_scope_combinations",
    "no_existing_control_link_sources", "no_existing_control_link_components",
)
OBSERVED_EDGE_FIELDS = {
    "actual_raw_header_TPFORG", "historical_reservation_identity_rows",
    "historical_reservation_namespace_counts", "historical_stage_exposure",
    "joint_independence_complete", "known_protected_link_found",
    "must_enter_future_joint_null_connectivity_before_selection_or_freezing",
    "native_reader_contract", "original_reader_parent", "raw_file", "raw_fits_sha256",
    "raw_header", "scientific_acceptance", "source_domain", "source_id",
    "supplemental_identity_edge", "trial_id", "would_connect_previously_separate_metadata_components",
}
IDENTITY_MAP_NAMES = (
    "opened-resolved-parent-map.csv", "historical-opened-survey-parent-map.csv",
    "paused-survey-opened-parent-map.csv", "ordinary-expanded-exposure-reserved-parent-map.csv",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def bounded_integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def decoded_bundle(path, artifact, size_key, hash_key, limit):
    require(bounded_integer(artifact["bytes"], 1, 16 * 1024**2)
            and bounded_integer(artifact[size_key], 1, limit)
            and valid_hash(artifact["sha256"]) and valid_hash(artifact[hash_key]),
            "Invalid bounded metadata byte identities")
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size == artifact["bytes"] and digest(path) == artifact["sha256"],
            "Compressed metadata changed")
    with gzip.open(path, "rb") as stream:
        payload = stream.read(artifact[size_key] + 1)
    require(len(payload) == artifact[size_key]
            and hashlib.sha256(payload).hexdigest() == artifact[hash_key],
            "Decoded metadata changed")
    return payload


def check_provenance(provenance):
    require(set(provenance) == {"inventory", "program", "original_inputs", "joint_scope", "identity_audit"},
            "Null metadata provenance fields changed")
    require(isinstance(provenance["original_inputs"], list) and provenance["original_inputs"],
            "Original null metadata input bindings are missing")
    descriptors = [provenance[key] for key in ("inventory", "program", "identity_audit")]
    descriptors += provenance["original_inputs"]
    for row in descriptors:
        require(set(row) == {"origin_path", "bytes", "sha256"}
                and isinstance(row["origin_path"], str) and bool(row["origin_path"].strip())
                and type(row["bytes"]) is int and row["bytes"] > 0 and valid_hash(row["sha256"]),
                "Invalid original metadata provenance binding")
    scope = provenance["joint_scope"]
    require(set(scope) == {"specification_sha256", "gzip_sha256", "canonical_json_sha256"}
            and all(valid_hash(value) for value in scope.values()),
            "Joint scope provenance identities changed")


def check_counts(counts):
    require(set(counts) == set(COUNT_FIELDS), "Null metadata count fields changed")
    require(all(isinstance(counts[key], dict) for key in COUNT_FIELDS[6:9]), "Invalid null metadata count maps")
    scalars = [counts[key] for key in COUNT_FIELDS[:4] + COUNT_FIELDS[-2:]]
    scalars += [value for key in COUNT_FIELDS[6:9] for value in counts[key].values()]
    require(all(bounded_integer(value, 0, 200000) for value in scalars)
            and bounded_integer(counts["sources"], 1, 200000), "Invalid bounded null metadata counts")
    shards = counts["shards"]
    require(isinstance(shards, list) and all(bounded_integer(value, 0, 65535) for value in shards)
            and shards == sorted(set(shards)) and valid_hash(counts["ordered_identity_metadata_sha256"]),
            "Invalid null metadata shard or ordered identities")


def check_specification(specification, bundle):
    require(type(specification.get("schema")) is int and specification["schema"] == 1
            and specification.get("status") == "CLOUD_NULL_POOL_METADATA_ONLY"
            and all(specification.get(key) is False for key in FLAGS),
            "Null pool metadata boundary changed")
    require(specification["artifact"]["path"] == bundle.name
            and specification["csv_fields"] == list(FIELDS) and specification["exposure"] == EXPOSURE,
            "Null pool CSV schema or exposure changed")
    require(isinstance(specification["boundary"], str) and specification["boundary"].strip()
            and isinstance(specification["unresolved"], dict) and specification["unresolved"]
            and all(isinstance(value, str) and value.strip() for value in specification["unresolved"].values()),
            "Null pool unresolved boundaries are missing")
    check_counts(specification["counts"])
    check_provenance(specification["provenance"])


def trial_links(row):
    links = {(kind, str(row[field])) for field, kind in (
        ("source_id", "source"), ("raw_fits_sha256", "raw"), ("origin_parent", "parent"),
        ("contaminant_source_id", "source"), ("contaminant_raw_fits_sha256", "raw"),
    ) if row.get(field)}
    links.update(("parent", str(value)) for key in ("resolved_parents", "additional_parent_identities")
                 for value in row[key])
    links.update((kind, str(quarter[field])) for quarter in row["pixel_quarters"]
                 for field, kind in (("actual_parent_kic", "parent"), ("content_sha256", "tpf_content"))
                 if quarter.get(field))
    return links


def decimal_identity(value):
    return isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value) is not None and int(value) < 2**63


def observed_file_identity(row, limit):
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}
            and isinstance(row["path"], str) and row["path"].strip()
            and bounded_integer(row["bytes"], 1, limit) and valid_hash(row["sha256"]),
            "Invalid observed parent metadata file identity")


def observed_parent_witnesses(edge):
    counts, rows = edge["historical_reservation_namespace_counts"], edge["historical_reservation_identity_rows"]
    require(isinstance(counts, dict) and set(counts) == set(IDENTITY_MAP_NAMES)
            and all(bounded_integer(value, 0, 200000) for value in counts.values())
            and isinstance(rows, list), "Invalid observed parent historical witnesses")
    seen, actual = set(), Counter()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"namespace", "parent_tpf", "raw_fits_sha256", "row", "source_id"}
                and row["namespace"] in IDENTITY_MAP_NAMES[:3] and bounded_integer(row["row"], 2, 200001)
                and decimal_identity(row["source_id"]) and valid_hash(row["raw_fits_sha256"])
                and row["parent_tpf"] == edge["actual_raw_header_TPFORG"],
                "Invalid observed parent historical row binding")
        identity = row["namespace"], row["row"]
        require(identity not in seen, "Repeated observed parent historical witness")
        seen.add(identity)
        actual[row["namespace"]] += 1
    require(all(counts[name] == actual[name] for name in IDENTITY_MAP_NAMES[:3]),
            "Observed parent historical witness counts changed")


def observed_parent_header(edge):
    header = edge["raw_header"]
    require(isinstance(header, dict) and set(header) == {
        "GAIAID", "HLSPID", "HLSPVER", "KEPLERID", "TPFORG", "extensions",
        "actual_header_parent_matches_original_declared", "header_origin_binding", "original_declared_parent",
    }, "Observed parent primary header fields changed")
    require(header["GAIAID"] == f"Gaia DR3 {edge['source_id']}"
            and header["HLSPID"] == "KBONUS-BKG" and header["HLSPVER"] == "V1.0"
            and bounded_integer(header["KEPLERID"], 1, 2**63 - 1)
            and bounded_integer(header["TPFORG"], 1, 2**63 - 1)
            and bounded_integer(header["original_declared_parent"], 1, 2**63 - 1)
            and str(header["KEPLERID"]) == edge["original_reader_parent"]
            and str(header["original_declared_parent"]) == edge["original_reader_parent"]
            and str(header["TPFORG"]) == edge["actual_raw_header_TPFORG"]
            and header["actual_header_parent_matches_original_declared"] is False
            and header["header_origin_binding"] == "ORIGINAL_CATALOGUE_PRODUCT",
            "Observed parent primary header identity mismatch")
    require(isinstance(header["extensions"], list) and header["extensions"]
            and all(isinstance(value, str) and value for value in header["extensions"])
            and len(set(header["extensions"])) == len(header["extensions"]),
            "Invalid observed parent header extension inventory")


def bind_observed_parent(edge, trials):
    require(isinstance(edge, dict) and set(edge) == OBSERVED_EDGE_FIELDS,
            "Observed parent edge fields changed")
    require(decimal_identity(edge["source_id"]) and decimal_identity(edge["original_reader_parent"])
            and decimal_identity(edge["actual_raw_header_TPFORG"])
            and edge["original_reader_parent"] != edge["actual_raw_header_TPFORG"]
            and valid_hash(edge["raw_fits_sha256"])
            and edge["trial_id"] == f"native_eb:development:{edge['source_id']}"
            and edge["source_domain"] == "KBonus_LC1"
            and edge["native_reader_contract"] == "catalogue_controls.read_product, exact original21 arrays",
            "Invalid observed parent source, reader or raw identity")
    require(edge["joint_independence_complete"] is False and edge["scientific_acceptance"] is False
            and edge["must_enter_future_joint_null_connectivity_before_selection_or_freezing"] is True
            and type(edge["known_protected_link_found"]) is bool
            and type(edge["would_connect_previously_separate_metadata_components"]) is bool
            and edge["historical_stage_exposure"] == "UNRESOLVED_BY_STAGE_IN_ORIGINAL_IDENTITY_MAPS"
            and edge["supplemental_identity_edge"] == ["parent", edge["actual_raw_header_TPFORG"]],
            "Observed parent metadata boundary or edge changed")
    trial = trials.get(edge["trial_id"], {})
    expected = {"source_id": edge["source_id"], "source_domain": "KBonus_LC1", "cohort": "native_eb",
                "role": "development", "scope": "CURRENT_OPENED_DEVELOPMENT",
                "origin_parent": edge["original_reader_parent"], "raw_fits_sha256": edge["raw_fits_sha256"]}
    require(all(trial.get(key) == value for key, value in expected.items()),
            "Observed parent trial source, raw, reader, class or role mismatch")
    parents = trial.get("additional_parent_identities")
    require(isinstance(parents, list) and all(decimal_identity(value) for value in parents)
            and parents == sorted(set(parents)) and edge["actual_raw_header_TPFORG"] in parents,
            "Observed parent is missing from unique additional trial identities")
    observed_file_identity(edge["raw_file"], 512 * 1024**2)
    require(edge["raw_file"]["sha256"] == edge["raw_fits_sha256"], "Observed parent raw file hash mismatch")
    observed_parent_header(edge)
    observed_parent_witnesses(edge)


def check_observed_parent_edges(metadata):
    provenance = metadata.get("provenance", {})
    require(isinstance(provenance, dict), "Invalid observed parent provenance")
    if "observed_native_header_parents" not in provenance:
        return
    observed = provenance["observed_native_header_parents"]
    require(isinstance(observed, dict) and isinstance(observed.get("edges"), dict),
            "Missing observed parent edge supplement")
    supplement = observed["edges"]
    require(set(supplement) == {"schema", "status", "boundary", "edges", "inputs", "joint_independence_complete", "scientific_acceptance"}
            and type(supplement["schema"]) is int and supplement["schema"] == 1
            and supplement["status"] == "SUPPLEMENTAL_NATIVE_RAW_HEADER_PARENT_METADATA"
            and supplement["joint_independence_complete"] is False and supplement["scientific_acceptance"] is False
            and isinstance(supplement["boundary"], str) and supplement["boundary"].strip(),
            "Observed parent supplement schema or boundary changed")
    require(isinstance(supplement["inputs"], list) and supplement["inputs"], "Missing observed parent input bindings")
    for row in supplement["inputs"]:
        observed_file_identity(row, 64 * 1024**2)
    require(isinstance(supplement["edges"], list) and len(supplement["edges"]) == 6,
            "Observed parent supplement requires its six original edges")
    require(isinstance(metadata.get("trials"), list)
            and all(isinstance(row, dict) and isinstance(row.get("trial_id"), str) for row in metadata["trials"]),
            "Invalid observed parent trial membership")
    trials = {row["trial_id"]: row for row in metadata["trials"]}
    require(len(trials) == len(metadata["trials"]), "Repeated observed parent trial identity")
    seen = set()
    for edge in supplement["edges"]:
        bind_observed_parent(edge, trials)
        identity = edge["trial_id"], edge["source_id"], edge["raw_fits_sha256"]
        require(identity not in seen, "Repeated observed parent edge")
        seen.add(identity)


def scope_indices(metadata):
    require(type(metadata.get("schema")) is int and metadata["schema"] == 1
            and metadata.get("status") == "PORTABLE_METADATA_SCOPE_PREPARATION_ONLY"
            and metadata.get("joint_independence_complete") is False,
            "Null pool requires provisional full joint scope metadata")
    require(metadata.get("qualified_external_connectivity") == [],
            "Additional qualified external connectivity requires a new null dependency audit")
    check_observed_parent_edges(metadata)
    trials = {row["trial_id"]: row for row in metadata["trials"]}
    components = {row["component_id"]: row for row in metadata["full_witness_components"]}
    members = [trial for row in components.values() for trial in row["trial_ids"]]
    require(len(trials) == len(metadata["trials"])
            and len(components) == len(metadata["full_witness_components"])
            and sorted(members) == sorted(trials), "Joint witnessed membership changed")
    scopes, links = {}, defaultdict(set)
    for identity, component in components.items():
        require({"component_id", "trial_ids", "identity_map_members", "historical_stage_exposure"} <= component.keys(),
                "Incomplete full joint component witness")
        require(valid_hash(identity) and component["trial_ids"]
                and isinstance(component["identity_map_members"], dict)
                and isinstance(component["historical_stage_exposure"], str)
                and component["historical_stage_exposure"].strip(), "Incomplete full joint component witness")
        scopes[identity] = {trials[trial]["scope"] for trial in component["trial_ids"]}
        for trial in component["trial_ids"]:
            row = trials[trial]
            require(row["full_witness_component_id"] == identity, "Joint witnessed trial binding changed")
            for link in trial_links(row):
                links[link].add(identity)
    return scopes, links


def linked_scope(specification, scope_specification, scope_bundle):
    binding = specification["provenance"]["joint_scope"]
    specification_bytes = scope_specification.read_bytes()
    require(hashlib.sha256(specification_bytes).hexdigest() == binding["specification_sha256"],
            "Bound joint scope specification changed")
    spec = json.loads(specification_bytes)
    artifact = spec["artifact"]
    require(artifact["path"] == scope_bundle.name and artifact["sha256"] == binding["gzip_sha256"]
            and artifact["canonical_json_sha256"] == binding["canonical_json_sha256"],
            "Bound joint scope byte identities changed")
    payload = decoded_bundle(scope_bundle, artifact, "canonical_json_bytes", "canonical_json_sha256", 32 * 1024**2)
    return scope_indices(json.loads(payload))


def ordered_links(value):
    values = tuple(value.split(";")) if value else ()
    require(list(values) == sorted(set(values)) and all(values), "Null component links must be sorted and unique")
    return values


def row_identity(row):
    require(set(row) == set(FIELDS) and all(isinstance(value, str) for value in row.values()),
            "Null CSV row fields changed")
    require(all(re.fullmatch(r"[1-9][0-9]*", row[key]) and int(row[key]) < 2**63
                for key in ("source_id", "origin_parent"))
            and re.fullmatch(r"0|[1-9][0-9]*", row["shard"]) and int(row["shard"]) <= 65535
            and valid_hash(row["raw_fits_sha256"]) and valid_hash(row["metadata_dependency_component_id"]),
            "Invalid typed null source, shard, parent or content identity")
    require(all(row[key] == value for key, value in EXPOSURE.items()), "Null stage exposure changed")
    return {"source_id": row["source_id"], "shard": row["shard"],
            "parent_tpf": row["origin_parent"], "raw_fits_sha256": row["raw_fits_sha256"]}


def bind_row(row, scopes, links, components, owners):
    linked = ordered_links(row["linked_full_witness_component_ids"])
    declared_scopes = ordered_links(row["linked_existing_scopes"])
    require(all(identity in scopes for identity in linked), "Unknown full witnessed component link")
    expected_scopes = sorted({scope for identity in linked for scope in scopes[identity]})
    require(list(declared_scopes) == expected_scopes, "Full witnessed component scopes changed")
    tokens = (("source", row["source_id"]), ("parent", row["origin_parent"]), ("raw", row["raw_fits_sha256"]))
    known = {identity for token in tokens for identity in links.get(token, ())}
    require(known <= set(linked), "Direct known joint identities are missing from full component links")
    component = row["metadata_dependency_component_id"]
    binding = (linked, declared_scopes)
    require(components.setdefault(component, binding) == binding, "Null component rows disagree on witnessed links")
    for token in (*tokens, *(("full_witness", identity) for identity in linked)):
        require(owners.setdefault(token, component) == component, "Known null dependency split across components")
    return declared_scopes


def metadata_counts(identities, components, linked_sources, unlinked):
    linked_components, combinations = Counter(), Counter()
    for _, scopes in components.values():
        linked_components.update(scopes)
        combinations[";".join(scopes) or "NO_EXISTING_CONTROL_TRIAL_LINK"] += 1
    return {
        "sources": len(identities), "origin_parents": len({row["parent_tpf"] for row in identities}),
        "raw_fits_hashes": len({row["raw_fits_sha256"] for row in identities}),
        "metadata_dependency_components": len(components),
        "shards": sorted({int(row["shard"]) for row in identities}),
        "ordered_identity_metadata_sha256": hashlib.sha256(canonical(identities)).hexdigest(),
        "linked_scope_sources": dict(linked_sources), "linked_scope_components": dict(linked_components),
        "component_scope_combinations": dict(combinations),
        "no_existing_control_link_sources": unlinked,
        "no_existing_control_link_components": combinations["NO_EXISTING_CONTROL_TRIAL_LINK"],
    }


def check_csv(payload, specification, scope):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""), strict=True)
    require(reader.fieldnames == list(FIELDS), "Null CSV columns changed")
    identities, components, owners, linked_sources, seen = [], {}, {}, Counter(), set()
    previous, unlinked = (-1, -1), 0
    for row in reader:
        require(len(identities) < 200000, "Null metadata source count exceeds its bound")
        identity = row_identity(row)
        order = (int(identity["shard"]), int(identity["source_id"]))
        require(order > previous and identity["source_id"] not in seen,
                "Null source identities must be unique and in committed order")
        seen.add(identity["source_id"])
        previous = order
        scopes = bind_row(row, *scope, components, owners)
        linked_sources.update(scopes)
        unlinked += not scopes
        identities.append(identity)
    counts = metadata_counts(identities, components, linked_sources, unlinked)
    require(counts == specification["counts"], "Null metadata counts or ordered identities changed")
    return counts


def output_paths(dump):
    require(dump.is_dir(), "Null metadata requires an existing dump directory")
    path, receipt = dump / "cloud-null-pool.csv", dump / "cloud-null-pool-receipt.json"
    paths = path, path.with_suffix(".part"), receipt, receipt.with_suffix(".part")
    require(not any(item.is_symlink() for item in paths), "Null metadata outputs must not redirect writes")
    receipt.unlink(missing_ok=True)
    require(all(not item.exists() or item.is_file() for item in paths), "Null metadata outputs must be regular files")
    require(not any(item.exists() for item in (paths[1], paths[3])), "Stale null metadata .part file; inspect and remove it before retrying")
    return path, receipt


def publish(path, payload):
    temporary = path.with_suffix(".part")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
        temporary.chmod(0o444)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def check_input_stability(identities):
    require(all(digest(path) == expected for path, expected in identities),
            "Metadata specification or bundle changed during verification")


def provision(specification_path, bundle, scope_specification, scope_bundle, dump):
    path, receipt = output_paths(dump)
    specification_bytes = specification_path.read_bytes()
    specification_sha256 = hashlib.sha256(specification_bytes).hexdigest()
    specification = json.loads(specification_bytes)
    check_specification(specification, bundle)
    payload = decoded_bundle(bundle, specification["artifact"], "csv_bytes", "csv_sha256", 64 * 1024**2)
    counts = check_csv(payload, specification, linked_scope(specification, scope_specification, scope_bundle))
    joint_identity = specification["provenance"]["joint_scope"]
    identities = ((specification_path, specification_sha256), (bundle, specification["artifact"]["sha256"]),
                  (scope_specification, joint_identity["specification_sha256"]),
                  (scope_bundle, joint_identity["gzip_sha256"]))
    check_input_stability(identities)
    if path.exists():
        require(path.stat().st_size == len(payload) and digest(path) == specification["artifact"]["csv_sha256"],
                "Previously provisioned null metadata changed")
        path.chmod(0o444)
    else:
        publish(path, payload)
    result = {
        "status": "NULL_CANDIDATE_METADATA_VERIFIED", "path": str(path), "bytes": len(payload),
        "sha256": specification["artifact"]["csv_sha256"], "specification_sha256": specification_sha256,
        "counts": counts, "provenance": specification["provenance"], "exposure": EXPOSURE,
        "unresolved": specification["unresolved"], "boundary": specification["boundary"],
        **{key: False for key in FLAGS},
    }
    check_input_stability(identities)
    publish(receipt, json.dumps(result, indent=2, allow_nan=False).encode() + b"\n")
    return result
