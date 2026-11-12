"""Bind explicitly supplied native mechanisms and report opened record runs."""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

from pipeline.deterministic import catalogue_control_report as catalogue
from pipeline.deterministic import control_prospective_inputs as prospective
from pipeline.deterministic.control_bindings import (
    read_json,
    reference,
    require,
    verified_reference,
)
from pipeline.deterministic.control_eligibility import COHORTS, build_plan, save
from pipeline.deterministic.control_pair_inputs import (
    protection_sets,
    require_permitted,
)
from pipeline.runtime import OUTCOME_CONTRACT, dump_path

PROTOCOL = "native-prospective-controls-v1"
ORIGINAL_PLAN_SHA256 = (
    "1695054c8d594a0b1c9b75b43858b2bf3c1c739d4486cf4f7815b692e6bfd09b"
)
READY = "NATIVE_UNCHANGED_REPLAY_BOUND"
REQUIREMENTS = {"pixels", "prf", "scene", "siblings"}
REPORT_STATUSES = {
    "COMPLETE_NATIVE_PROSPECTIVE_CONTROL_REPORT",
    "PARTIAL_NATIVE_PROSPECTIVE_CONTROL_REPORT",
}
CONFIGURATION = Path(__file__).parents[1] / "vetting/config.json"


def identity(source):
    return {
        "source_id": source["source_id"],
        "cohort": source["cohort"],
        "parent_tpf": source["parent_tpf"],
        "raw_fits_sha256": source["pool"]["raw_fits_sha256"],
        "component_id": source["pool"]["metadata_dependency_component_id"],
        "role": "development",
    }


def permitted_identity(item, protection):
    require_permitted(
        {
            "source_id": item["source_id"],
            "parent_tpf": item["parent_tpf"],
            "pool": {
                "raw_fits_sha256": item["raw_fits_sha256"],
                "metadata_dependency_component_id": item["component_id"],
            },
        },
        protection,
    )
    require(
        item["role"] == "development"
        and item["component_id"] not in protection["protected_full_component_ids"],
        "Protected component or nondevelopment native input",
    )


def payload_path(receipt, protection):
    require(
        receipt["sha256"] not in protection["raw_or_pixel_content_hashes"],
        "Protected native payload hash",
    )
    return verified_reference(receipt)


def metadata(receipt):
    return read_json(verified_reference(receipt))


def same_reference(left, right):
    return (Path(left["path"]).resolve(), left["sha256"]) == (
        Path(right["path"]).resolve(),
        right["sha256"],
    )


def verify_original_plan(plan, locations):
    require(
        not set(locations) - set(plan["inputs"]), "Unknown original input relocation"
    )
    inputs = {
        key: locations.get(key, receipt) for key, receipt in plan["inputs"].items()
    }
    for key, receipt in inputs.items():
        require(
            receipt["sha256"] == plan["inputs"][key]["sha256"],
            "Relocated original input content changed",
        )
        verified_reference(receipt)
    rebuilt = build_plan(
        Path(inputs["opened_packet"]["path"]).parent,
        inputs["public_catalogue"]["path"],
        inputs["stopped_identity_pool"]["path"],
    )
    require(
        set(plan) == set(rebuilt)
        and all(
            plan[k] == rebuilt[k] for k in plan if k not in {"inputs", "implementation"}
        )
        and {k: r["sha256"] for k, r in plan["inputs"].items()}
        == {k: r["sha256"] for k, r in rebuilt["inputs"].items()},
        "Original native plan recipe, identity or input content changed",
    )


def original_scope(declaration):
    require(
        declaration["schema"] == 1
        and declaration["protocol"] == PROTOCOL
        and declaration["role"] == "development",
        "Native declaration requires the opened development contract",
    )
    require(
        declaration["original_plan"]["sha256"] == ORIGINAL_PLAN_SHA256,
        "Original prospective plan content changed",
    )
    plan = metadata(declaration["original_plan"])
    verify_original_plan(plan, declaration.get("original_input_locations", {}))
    require(
        Counter(s["cohort"] for s in plan["sources"])
        == {"guide_star_family": 100, "contamination_pair": 100},
        "Native declaration changed the original 100/100 denominators",
    )
    prospective.exact_index(
        declaration["cases"], [s["source_id"] for s in plan["sources"]]
    )
    for source, case in zip(plan["sources"], declaration["cases"]):
        require(
            case["identity"] == identity(source), "Original native identity changed"
        )
    return plan


def native_recipe(case, protection):
    permitted_identity(case["identity"], protection)
    require(case.get("recipe") is not None, "Missing native recipe")
    recipe = metadata(case["recipe"])
    require(
        recipe["schema"] == 1
        and recipe["identity"] == case["identity"]
        and recipe["mode"] == "native_unchanged_replay"
        and recipe["operation"] == "identity"
        and set(recipe)
        == {
            "schema",
            "identity",
            "mode",
            "operation",
            "truth",
            "quarters",
            "required_inputs",
        },
        "Unsupported native recipe; physical perturbations require a defined operation",
    )
    require(case.get("truth") is not None, "Missing native mechanism truth")
    require(recipe["truth"] == case["truth"], "Recipe truth reference changed")
    truth = metadata(case["truth"])
    require(
        truth["schema"] == 1
        and truth["identity"] == case["identity"]
        and truth["native_false_class_verified"] is True
        and isinstance(truth["mechanism"], str)
        and truth["mechanism"].strip()
        and isinstance(truth["mechanism_evidence"], list)
        and bool(truth["mechanism_evidence"]),
        "Missing verified native mechanism truth",
    )
    require(
        sorted(recipe["required_inputs"]) == sorted(REQUIREMENTS)
        and recipe["required_inputs"] == truth["required_inputs"],
        "Native mechanism physical requirements changed",
    )
    quarters = recipe["quarters"]
    require(
        bool(quarters)
        and quarters == sorted(set(quarters))
        and all(type(q) is int and 0 <= q <= 17 for q in quarters),
        "Native recipe requires distinct ordered mission quarters",
    )
    clock = metadata(truth["event_clock"])
    require(
        clock["identity"] == case["identity"]
        and clock["time_scale"] == "BKJD"
        and clock["independent_native_timing"] is True
        and clock["derived_from_detector_proposals"] is False
        and isinstance(clock["signal_id"], str)
        and clock["signal_id"]
        and bool(clock["events"]),
        "Native event clock is absent or belongs to another source",
    )
    for event in clock["events"]:
        require(
            type(event["quarter"]) is int
            and event["quarter"] in quarters
            and all(
                type(event[k]) in {int, float} and math.isfinite(event[k])
                for k in ("centre_bkjd", "duration_days")
            )
            and event["duration_days"] > 0,
            "Invalid native physical event clock",
        )
    for receipt in truth["mechanism_evidence"]:
        witness = metadata(receipt)
        require(
            witness["identity"] == case["identity"]
            and witness["native_false_class_verified"] is True
            and witness["mechanism"] == truth["mechanism"]
            and witness["event_clock"] == truth["event_clock"]
            and witness["scientific_outcomes_used"] is False,
            "Native mechanism witness does not establish the declared truth",
        )
        measured = metadata(witness["measurements"])
        require(
            measured["identity"] == case["identity"]
            and measured["event_clock"] == truth["event_clock"]
            and measured["status"] == "MEASURED"
            and isinstance(measured["native_inputs"], list)
            and bool(measured["native_inputs"])
            and isinstance(measured["measurements"], dict)
            and bool(measured["measurements"]),
            "Native mechanism lacks per-case measured input evidence",
        )
        required = (
            "pointing_pixel_coupling_measured"
            if case["identity"]["cohort"] == "guide_star_family"
            else "target_donor_attribution_measured"
        )
        require(measured[required] is True, "Native physical mechanism is unmeasured")
        for native_input in measured["native_inputs"]:
            require(
                native_input["sha256"] not in protection["raw_or_pixel_content_hashes"],
                "Protected native mechanism evidence",
            )
            verified_reference(native_input)
    return recipe, truth


def native_donor(source, truth, plan, protection):
    if source["cohort"] != "contamination_pair":
        require("donor" not in truth, "Guide truth acquired a contaminant donor")
        return
    original = next(
        r for r in plan["donor_bindings"] if r["source_id"] == source["source_id"]
    )
    candidates = original["public_donor_product_identities"]
    require(
        len(candidates) <= 1,
        "Original donor is ambiguous; no identity may be chosen",
    )
    donor = truth["donor"]
    require(
        donor["identity"]["source_id"] != source["source_id"],
        "Target and donor are the same source",
    )
    permitted_identity(donor["identity"], protection)
    clock = metadata(truth["event_clock"])
    require(
        clock["signal_source_id"] == donor["identity"]["source_id"]
        and clock["native_signal_reference"] == donor["reference"],
        "Contamination clock is not bound to its exact native donor signal",
    )
    require(
        donor["identity"]["parent_tpf"] == source["parent_tpf"]
        and donor["use"] == "DONOR_SIGNAL_REFERENCE_ONLY",
        "Native donor parent or reference use changed",
    )
    if candidates:
        require(
            donor["identity"]["source_id"]
            == str(candidates[0]["gaia_designation"]).split()[-1]
            and donor["domain"] == "KBONUS_PSF_NATIVE_SIGNAL",
            "KBonus donor identity or domain changed",
        )
        require(
            donor["reference"]["sha256"] == donor["identity"]["raw_fits_sha256"],
            "Native KBonus donor reference is not the declared original product",
        )
        kbonus_donor(original, donor, clock, protection)
    else:
        mission_donor(original, donor, clock, protection)


def kbonus_donor(original, donor, clock, protection):
    from pipeline.deterministic.control_donor_headers import (
        bind_donor_header,
        donor_product,
    )
    from pipeline.deterministic.control_eligibility_archive import local_headers

    provisioned = metadata(donor["provisioning"])
    status = provisioned["status"]
    if status == "CANONICAL_KBONUS_DONOR_BYTES_AND_IDENTITY_VERIFIED":
        job = provisioned["job"]
        prior = metadata(job["original_header_record"])
        require(
            job["kind"] == "DONOR"
            and job["target_source_id"] == original["source_id"]
            and job["native_source_id"] == donor["identity"]["source_id"]
            and job["global_parent"] == donor["identity"]["parent_tpf"]
            and job["canonical_url"] == donor_product(original)["url"]
            and job["product_bytes"] == prior["archive_product_bytes"],
            "Published native donor job differs from its original contract",
        )
        bound_original, bound_headers = prior["binding"], provisioned["native_headers"]
    elif status == "CANONICAL_DONOR_RAW_BYTES_AND_PREPARATION_BOUND":
        prior = metadata(provisioned["prior_archive_record"])
        bound_original, bound_headers = provisioned["binding"], provisioned["headers"]
    else:
        raise ValueError("Native KBonus donor input was not fully provisioned")
    require(
        bound_original == prior["binding"] == original
        and prior["status"] == "VERIFIED_DONOR_ARCHIVE_HEADERS"
        and same_reference(provisioned["payload"], donor["reference"]),
        "Missing exact provisioned KBonus donor input receipt",
    )
    path = payload_path(donor["reference"], protection)
    headers = local_headers(path)
    observed = bind_donor_header(original, headers)
    require(
        headers == prior["headers"] == bound_headers
        and path.stat().st_size == prior["archive_product_bytes"]
        and observed["archive_gaia_source_id"] == donor["identity"]["source_id"]
        and observed["global_extraction_parent"] == donor["identity"]["parent_tpf"]
        and {e["quarter"] for e in clock["events"]}
        <= {q["quarter"] for q in observed["quarters"]},
        "Native KBonus donor header, hash or event quarter changed",
    )


def mission_donor(original, donor, clock, protection):
    from pipeline.deterministic.control_eligibility_archive import local_headers
    from pipeline.deterministic.control_mission_reference import DOMAIN, bind_reference

    require(donor["domain"] == DOMAIN, "Mission SAP donor domain changed")
    contracts = metadata(donor["reference"])
    provisioned = metadata(donor["provisioning"])
    require(
        contracts["status"] == "EXPLICIT_MISSION_SAP_REFERENCE_CONTRACTS_ONLY"
        and same_reference(provisioned["original_contracts"], donor["reference"]),
        "Missing exact provisioned mission reference contract ledger",
    )
    selected = [
        (i, c)
        for i, c in enumerate(contracts["contracts"])
        if c["identity"]["target_source_id"] == original["source_id"]
    ]
    rows = [
        r
        for r in provisioned["contracts"]
        if r["identity"]["target_source_id"] == original["source_id"]
    ]
    require(
        bool(selected)
        and [r["original_contract_index"] for r in rows] == [i for i, _ in selected],
        "Missing, extra or reordered mission donor quarter",
    )
    for row, (_, contract) in zip(rows, selected):
        inputs = contract["inputs"]
        expected = bind_reference(
            original,
            inputs["original_donor"],
            inputs["identity_receipts"],
            inputs["header"],
            protection["source_ids"],
        )
        require(
            contract == expected
            and contract["identity"]["gaia_dr3"] == donor["identity"]["source_id"]
            and row["identity"] == contract["identity"]
            and row["mission_product"] == contract["mission_product"]
            and row["domain"] == DOMAIN
            and row["use"] == "DONOR_SIGNAL_REFERENCE_ONLY"
            and row["status"] == "EXACT_NATIVE_SAP_REFERENCE_AVAILABLE",
            "Mission donor is unavailable or differs from the exact catalogue contract",
        )
        path = payload_path(row["payload"], protection)
        require(
            path.stat().st_size == contract["mission_product"]["bytes"]
            and local_headers(path)[:2] == metadata(inputs["header"])["headers"],
            "Mission donor native payload or headers changed",
        )
    require(
        {e["quarter"] for e in clock["events"]}
        <= {c["mission_product"]["quarter"] for _, c in selected},
        "Mission donor lacks an independent event-clock quarter",
    )


def native_pixel(quarter, protection):
    from astropy.io import fits
    from lkprf.utils import channel_to_module_output

    from pipeline.deterministic.control_pixel_inputs import header_record

    pixel_path = payload_path(quarter["pixel"], protection)
    prf_path = payload_path(quarter["prf"], protection)
    header = header_record(
        pixel_path,
        {
            "parent_kic": int(quarter["parent_tpf"]),
            "quarter": quarter["quarter"],
        },
    )
    prf_header = fits.getheader(prf_path, 0)
    module, output = channel_to_module_output(header["channel"])
    require(
        int(prf_header["CHANNEL"]) == header["channel"]
        and int(prf_header["MODULE"]) == module
        and int(prf_header["OUTPUT"]) == output
        and prf_path.name == f"kplr{module:02}.{output}_2011265_prf.fits",
        "Native PRF and quarter pixel detector identities differ",
    )
    return {
        "path": str(pixel_path),
        "sha256": quarter["pixel"]["sha256"],
        "base_parent_kic": header["parent_kic"],
        "native_header_sha256": header["header_sha256"],
        "prf_calibration": {
            "filename": prf_path.name,
            "path": str(prf_path),
            "sha256": quarter["prf"]["sha256"],
        },
    }


def quarter_bindings(case, recipe, original, protection):
    require(
        case.get("physical_bindings") is not None,
        "Missing native quarter/pixel/PRF/scene/sibling bindings",
    )
    physical = metadata(case["physical_bindings"])
    require(physical["identity"] == case["identity"], "Physical binding source changed")
    quarters = physical["quarters"]
    require(
        [row["quarter"] for row in quarters] == recipe["quarters"],
        "Missing, extra or reordered native physical quarter",
    )
    observed = [r["quarter"] for r in original["metadata"]["quarter_parent_bindings"]]
    require(
        set(observed) == set(recipe["quarters"]),
        "Unavailable original quarter or discarded unchanged-replay quarter",
    )
    pixels, siblings = {}, {}
    members = native_parent_members(case["identity"])
    require(members is not None, "Original native sibling membership is unavailable")
    for quarter in quarters:
        parent = quarter["parent_tpf"]
        require(
            parent == case["identity"]["parent_tpf"],
            "Per-quarter sibling parent changes are unsupported by the current runtime",
        )
        require(
            quarter["parent_candidates"] == [parent]
            and parent not in protection["parent_identities"],
            "Ambiguous or protected quarter extraction parent",
        )
        witness = metadata(quarter["witness"])
        require(
            witness["identity"] == case["identity"]
            and witness["quarter"] == quarter["quarter"]
            and witness["parent_tpf"] == parent
            and witness["parent_candidates"] == [parent]
            and witness["extraction_parent_verified"] is True
            and witness["scene_complete"] is True
            and witness["siblings_complete"] is True
            and all(
                witness[k] == quarter[k] for k in ("pixel", "prf", "scene", "siblings")
            ),
            "Missing exact native parent/pixel/PRF/scene/sibling witness",
        )
        require(
            sorted(quarter["siblings"]) == list(members),
            "Native sibling bindings omit or add original parent-index members",
        )
        require(quarter["scene"] == physical["scene_metadata"], "Quarter scene changed")
        for sid, item in quarter["siblings"].items():
            permitted_identity(item["identity"], protection)
            require(
                item["identity"]["source_id"] == sid
                and item["identity"]["parent_tpf"] == parent
                and item["input"]["metadata"]["source_id"] == sid
                and item["input"]["metadata"]["raw_fits_sha256"]
                == item["identity"]["raw_fits_sha256"]
                and str(item["input"]["metadata"]["origin_tpf_kic"]) == parent,
                "Quarter sibling input identity changed",
            )
            validate_sibling(item, protection)
            payload_path(
                {"path": item["input"]["arrays"], "sha256": item["input"]["sha256"]},
                protection,
            )
            require(
                sid not in siblings or siblings[sid] == item, "Sibling identity drift"
            )
            siblings[sid] = item
        pixels[str(quarter["quarter"])] = native_pixel(quarter, protection)
    scene = catalogue.read_jsonl(verified_reference(physical["scene_metadata"]))
    scene_ids = [str(row["source_id"]) for row in scene]
    require(
        len(scene_ids) == len(set(scene_ids))
        and case["source_id"] in scene_ids
        and set(siblings) <= set(scene_ids),
        "Incomplete original scene identity inventory",
    )
    indexed = {str(row["source_id"]): row for row in scene}
    require(
        all(
            indexed[case["source_id"]][k] == original["metadata"][k]
            for k in (
                "source_id",
                "origin_tpf_kic",
                "raw_fits_sha256",
                "ra_deg",
                "dec_deg",
                "phot_g_mean_mag",
                "shard",
            )
        ),
        "Native scene changed the original selected-source metadata",
    )
    physical = dict(
        physical,
        scene_metadata={k: physical["scene_metadata"][k] for k in ("path", "sha256")},
    )
    return physical, pixels, siblings


def native_parent_members(item):
    from pipeline.vetting.parent_index import parent_members

    configuration = read_json(CONFIGURATION)
    return parent_members(
        catalogue.root_path(configuration["parent_index"]),
        configuration["parent_index_sha256"],
        item["source_id"],
        item["parent_tpf"],
    )


def validate_sibling(item, protection, decode=False):
    permitted_identity(item["identity"], protection)
    proof = metadata(item["original_input"])
    archive = metadata(proof["archive_record"])
    source = archive["source"]
    require(
        identity(source) == item["identity"]
        and proof["status"] == "ORIGINAL_RAW_READER_ARRAYS_AND_PREPARATION_BOUND"
        and proof["original_payload"]["sha256"] == item["identity"]["raw_fits_sha256"]
        and proof["metadata"] == item["input"]["metadata"]
        and same_reference(
            proof["arrays"],
            {"path": item["input"]["arrays"], "sha256": item["input"]["sha256"]},
        ),
        "Sibling input lacks its exact original native content proof",
    )
    if decode:
        verify_original_arrays(source, proof, protection)


def validate_case(source, case, original, plan, protection, decode=False):
    recipe, truth = native_recipe(case, protection)
    native_donor(source, truth, plan, protection)
    require(
        original["status"] == "ORIGINAL_RAW_READER_ARRAYS_AND_PREPARATION_BOUND",
        "Original unchanged input is unavailable",
    )
    require(
        original["source_id"] == source["source_id"]
        and original["cohort"] == source["cohort"]
        and original["component_id"] == case["identity"]["component_id"]
        and original["original_payload"]["sha256"]
        == case["identity"]["raw_fits_sha256"],
        "Original materialized identity changed",
    )
    physical, pixels, siblings = quarter_bindings(case, recipe, original, protection)
    runtime = {
        "arrays": original["arrays"]["path"],
        "sha256": original["arrays"]["sha256"],
        "metadata": original["metadata"],
        "pixels": pixels,
    }
    payload_path(original["arrays"], protection)
    if decode:
        verify_original_arrays(source, original, protection)
        for item in siblings.values():
            validate_sibling(item, protection, decode=True)
    return runtime, physical["scene_metadata"], siblings


def verify_original_arrays(source, original, protection):
    prospective.validate_source_proof(
        source,
        {
            "record": original["archive_record"],
            "payload": original["original_payload"],
            "preparation": original["preparation"],
        },
        original,
        protection,
    )
    prospective.validate_encoded_source(
        original,
        {
            "arrays": f"arrays/{source['source_id']}.npz",
            "sha256": original["arrays"]["sha256"],
            "metadata": original["metadata"],
        },
        Path(original["arrays"]["path"]).parent.parent / "input-manifest.json",
    )


def load_binding_inputs(declaration_receipt, materialization_receipt):
    declaration = metadata(declaration_receipt)
    plan = original_scope(declaration)
    protection = protection_sets(declaration["protected_inputs"])
    materialization = metadata(materialization_receipt)
    require(
        materialization["inputs"]["plan"] == declaration["original_plan"]
        and materialization["runtime_policy_required"] == "DEVELOPMENT"
        and materialization["final_scoped_role_authorization"] is False,
        "Original materialization scope changed",
    )
    rows = [metadata(r) for r in materialization["records"]]
    prospective.exact_index(rows, [s["source_id"] for s in plan["sources"]])
    return declaration, plan, protection, rows


def bind(declaration_receipt, materialization_receipt, policy_receipt, directory):
    require(not directory.exists(), "Preserve prior attempts; output already exists")
    declaration, plan, protection, originals = load_binding_inputs(
        declaration_receipt, materialization_receipt
    )
    policy = metadata(policy_receipt)
    require(
        policy["status"] == "DEVELOPMENT" and policy["boi_catalogue_enabled"] is False,
        "Native record inputs require the disabled development policy",
    )
    rows, sources, scenes, ancillary = [], {}, [], {}
    for source, case, original in zip(plan["sources"], declaration["cases"], originals):
        row = {
            "source_id": source["source_id"],
            "identity": identity(source),
            "case": case,
            "original_input": original,
            "status": "NATIVE_INPUT_REFUSED",
        }
        try:
            runtime, scene, siblings = validate_case(
                source, case, original, plan, protection, decode=True
            )
            require(
                not scenes or scene == scenes[0],
                "Different native runtime scene inventories",
            )
            directories = {
                str(Path(p["prf_calibration"]["path"]).parent)
                for p in runtime["pixels"].values()
            }
            previous = {
                str(Path(p["prf_calibration"]["path"]).parent)
                for item in sources.values()
                for p in item.get("pixels", {}).values()
            }
            require(
                len(directories) == 1 and (not previous or directories == previous),
                "Native PRFs need the same exact runtime cache directory",
            )
            for sid, item in siblings.items():
                require(
                    sid not in ancillary or ancillary[sid] == item,
                    "Sibling changed across cases",
                )
            sources[source["source_id"]] = runtime
            scenes.append(scene)
            ancillary.update(siblings)
            row.update(status=READY, runtime_input=runtime)
        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
            row.update(error=type(error).__name__, message=str(error))
        rows.append(row)
    for sid, item in ancillary.items():
        require(
            sid not in sources or sources[sid] == item["input"],
            "Target/sibling input conflict",
        )
        sources.setdefault(sid, item["input"])
    directory.mkdir(parents=True)
    vetting_configuration = runtime_configuration(sources, directory)
    save(
        directory / "bindings.json",
        {
            "schema": 1,
            "protocol": PROTOCOL,
            "role": "development",
            "declaration": declaration_receipt,
            "materialization": materialization_receipt,
            "policy": policy_receipt,
            "vetting_configuration": vetting_configuration,
            "binding_implementation": reference(Path(__file__)),
            "trials": rows,
            "ancillary": ancillary,
            "counts": dict(Counter(row["status"] for row in rows)),
            "final_scoped_role_authorization": False,
        },
    )
    manifest = {
        "schema": 1,
        "protocol": PROTOCOL,
        "role": "development",
        "measurement_scope": "OPENED_NATIVE_MECHANISM_RECORD_ONLY",
        "native_control_binding": reference(directory / "bindings.json"),
        "sources": sources,
    }
    if scenes:
        manifest["scene_metadata"] = scenes[0]
    save(directory / "input-manifest.json", manifest)
    jobs = {}
    for cohort in sorted(COHORTS):
        requested = [r["source_id"] for r in rows if r["identity"]["cohort"] == cohort]
        selected = [
            r["source_id"]
            for r in rows
            if r["identity"]["cohort"] == cohort and r["status"] == READY
        ]
        save(directory / f"{cohort}-requested.json", requested)
        save(directory / f"{cohort}-source-ids.json", selected)
        jobs[cohort] = {
            "declared_sources": 100,
            "selected_sources": len(selected),
            "requested_ids": reference(directory / f"{cohort}-requested.json"),
            "source_ids": reference(directory / f"{cohort}-source-ids.json"),
            "input_manifest": reference(directory / "input-manifest.json"),
            "disposition_policy": policy_receipt,
            "vetting_configuration": vetting_configuration,
            "record_every_gate": True,
            "argv": [
                sys.executable,
                "-m",
                "pipeline.deterministic.run",
                "all",
                "--input-manifest",
                str(directory / "input-manifest.json"),
                "--source-ids",
                str(directory / f"{cohort}-source-ids.json"),
                "--limit",
                str(len(selected)),
                "--shards",
                *map(
                    str, sorted({sources[sid]["metadata"]["shard"] for sid in selected})
                ),
                "--record-every-gate",
                "--vetting-config",
                vetting_configuration["path"],
                "--disposition-policy",
                policy_receipt["path"],
                "--run",
                str(directory / f"run-{cohort}"),
            ]
            if selected
            else None,
        }
    save(directory / "runtime.json", jobs)
    return metadata(reference(directory / "bindings.json"))


def runtime_configuration(sources, directory):
    configuration = read_json(CONFIGURATION)
    configuration["triage_execution"] = "RECORD_EVERY_GATE"
    caches = {
        str(Path(pixel["prf_calibration"]["path"]).parent)
        for source in sources.values()
        for pixel in source.get("pixels", {}).values()
    }
    require(len(caches) <= 1, "Native PRFs require one exact runtime cache directory")
    if caches:
        configuration["prf_cache_directory"] = caches.pop()
    save(directory / "vetting-config.json", configuration)
    return reference(directory / "vetting-config.json")


def runtime_binding(manifest):
    require(
        manifest["schema"] == 1
        and manifest["protocol"] == PROTOCOL
        and manifest["role"] == "development"
        and manifest["measurement_scope"] == "OPENED_NATIVE_MECHANISM_RECORD_ONLY",
        "Native runtime scope changed",
    )
    binding = metadata(manifest["native_control_binding"])
    verified_reference(binding["binding_implementation"])
    require(
        binding["protocol"] == PROTOCOL
        and binding["role"] == "development"
        and binding["final_scoped_role_authorization"] is False,
        "Native runtime binding scope changed",
    )
    declaration = metadata(binding["declaration"])
    require(
        declaration["original_plan"]["sha256"] == ORIGINAL_PLAN_SHA256,
        "Original prospective runtime plan content changed",
    )
    plan = metadata(declaration["original_plan"])
    prospective.exact_index(
        binding["trials"], [s["source_id"] for s in plan["sources"]]
    )
    prospective.exact_index(
        declaration["cases"], [s["source_id"] for s in plan["sources"]]
    )
    require(
        binding["schema"] == 1
        and declaration["protocol"] == PROTOCOL
        and declaration["role"] == "development"
        and Counter(s["cohort"] for s in plan["sources"])
        == {"guide_star_family": 100, "contamination_pair": 100},
        "Native runtime lost original scope or denominator",
    )
    for source, case, row in zip(
        plan["sources"], declaration["cases"], binding["trials"]
    ):
        require(
            row["case"] == case
            and row["identity"] == case["identity"] == identity(source)
            and row["status"] in {READY, "NATIVE_INPUT_REFUSED"},
            "Native runtime case ledger changed",
        )
    expected = {
        r["source_id"]: r["runtime_input"]
        for r in binding["trials"]
        if r["status"] == READY
    }
    for sid, item in binding["ancillary"].items():
        require(
            sid not in expected or expected[sid] == item["input"],
            "Native sibling collision",
        )
        expected.setdefault(sid, item["input"])
    require(manifest["sources"] == expected, "Native runtime input set changed")
    return binding


def runtime_source(manifest, sid):
    binding = runtime_binding(manifest)
    declaration = metadata(binding["declaration"])
    protection = protection_sets(declaration["protected_inputs"])
    policy = metadata(binding["policy"])
    require(
        policy["status"] == "DEVELOPMENT" and policy["boi_catalogue_enabled"] is False,
        "Native runtime policy changed",
    )
    if sid in binding["ancillary"]:
        item = binding["ancillary"][sid]
        permitted_identity(item["identity"], protection)
        plan = metadata(declaration["original_plan"])
        owners = [r for r in binding["trials"] if r["status"] == READY]
        verified = False
        for row in owners:
            source = next(
                s for s in plan["sources"] if s["source_id"] == row["source_id"]
            )
            _, _, siblings = validate_case(
                source, row["case"], row["original_input"], plan, protection
            )
            if sid in siblings:
                require(
                    item == siblings[sid],
                    "Ancillary sibling differs from native recipe",
                )
                verified = True
        require(verified, "Native ancillary input has no authorized owner")
        validate_sibling(item, protection, decode=True)
        payload_path(
            {"path": item["input"]["arrays"], "sha256": item["input"]["sha256"]},
            protection,
        )
        return
    row = next(r for r in binding["trials"] if r["source_id"] == sid)
    require(row["status"] == READY, "Native runtime selected a refused case")
    plan = metadata(declaration["original_plan"])
    source = next(s for s in plan["sources"] if s["source_id"] == sid)
    case = next(c for c in declaration["cases"] if c["source_id"] == sid)
    require(
        row["identity"] == identity(source) and row["case"] == case,
        "Native runtime truth/recipe identity changed",
    )
    runtime, scene, _ = validate_case(
        source, case, row["original_input"], plan, protection
    )
    require(
        runtime == manifest["sources"][sid] and scene == manifest["scene_metadata"],
        "Native runtime physical bindings changed",
    )


def trial_result(row, run, sources):
    result = {
        "source_id": row["source_id"],
        "category": row["identity"]["cohort"],
        "identity": row["identity"],
        "binding_status": row["status"],
        "truth": row["case"].get("truth"),
        "recipe": row["case"].get("recipe"),
        "status": "NATIVE_INPUT_REFUSED",
        "execution_complete": False,
        "boi_catalogue": None,
        "scientific_abstention": None,
        "signals": [],
        "proposal_evidence": None,
        "source_evidence": None,
        "message": row.get("message"),
    }
    if row["status"] != READY or run is None:
        if row["status"] == READY:
            result.update(
                status="NOT_EXECUTED", message="No native record run supplied"
            )
        return result
    sid = row["source_id"]
    proposal_path = run / "proposals" / f"{sid}.json"
    proposal = read_json(proposal_path)
    require(proposal["source_id"] == sid, "Native detector source changed")
    source = sources.get(sid)
    require(
        (source is not None) == bool(proposal["alerts"]),
        "Native proposal/source evidence mismatch",
    )
    complete = not catalogue.incomplete_disposition(source)
    require(
        source is None or source["alerts"] == proposal["alerts"],
        "Native alert identity changed",
    )
    result.update(
        status=("VETTING_COMPLETE" if source else "DETECTOR_NO_PROPOSAL")
        if complete
        else "EXECUTION_FAILED",
        execution_complete=complete,
        scientific_abstention=catalogue.scientific_abstention(source)
        if complete
        else None,
        signals=[catalogue.trial_signal(source, signal) for signal in source["signals"]]
        if source
        else [],
        proposal_evidence=reference(proposal_path),
        source_evidence=reference(run / "vetting/sources" / f"{sid}.json")
        if source
        else None,
        observed_candidate_emission=catalogue.emitted_candidate(source["disposition"])
        if source
        else False,
    )
    return result


def native_pixel_evidence(run, manifest, manifest_path, sources):
    if not sources:
        return {}
    path = run / "vetting/pixel-retrieval.json"
    retrieval = read_json(path)
    require(
        retrieval.get("status") == "EXPLICIT_EXPERIMENT_INPUTS"
        and retrieval.get("measurement_scope") == manifest["measurement_scope"]
        and isinstance(retrieval.get("input_manifest"), dict)
        and {"path", "sha256"} <= retrieval["input_manifest"].keys()
        and same_reference(retrieval["input_manifest"], reference(manifest_path)),
        "Native record did not use its explicit bound pixel inputs",
    )
    inputs = {str(path): reference(path)["sha256"]}
    for sid in sources:
        item = manifest["sources"][sid]
        parent = item["metadata"]["origin_tpf_kic"]
        path = run / "vetting/pixel-manifests" / f"{parent}.json"
        actual = read_json(path)["quarters"]
        require(bool(item.get("pixels")), "Native target has no bound pixels")
        for quarter, pixel in item["pixels"].items():
            expected = dict(
                pixel, path=str(Path(pixel["path"]).resolve()), status="AVAILABLE"
            )
            observed = actual.get(quarter)
            require(
                isinstance(observed, dict) and isinstance(observed.get("path"), str),
                "Native record pixel product differs from its physical binding",
            )
            observed = dict(observed, path=str(Path(observed["path"]).resolve()))
            require(
                observed == expected,
                "Native record pixel product differs from its physical binding",
            )
        inputs[str(path)] = reference(path)["sha256"]
    return inputs


def run_evidence(run, selected, binding_receipt, binding):
    from pipeline.deterministic.run import validate_preparation

    require(bool(selected), "No native cases were authorized for this run")
    preparation = validate_preparation(run)
    _, detection, results, sources = catalogue.validate_run(
        run, selected, "development_replay"
    )
    config = read_json(run / "vetting.json")
    configuration = metadata(binding["vetting_configuration"])
    overwritten = {
        "queue",
        "run_directory",
        "workers",
        "network",
        "quality_bitmask",
        "quality_selection",
        "discovery_contract",
        "disposition_policy",
        "input_manifest",
    }
    require(
        all(
            config.get(k) == v
            or (
                k == "pixel_cache_directory"
                and isinstance(config.get(k), str)
                and catalogue.root_path(config[k]).resolve()
                == catalogue.root_path(v).resolve()
            )
            for k, v in configuration.items()
            if k not in overwritten
        ),
        "Native runtime changed its bound vetting configuration",
    )
    policy = catalogue.root_path(config["disposition_policy"])
    require(
        config["triage_execution"] == "RECORD_EVERY_GATE"
        and same_reference(reference(policy), binding["policy"]),
        "Native control run changed record mode or policy",
    )
    execution = results["disposition_execution"]
    require(
        execution["policy_status"] == "DEVELOPMENT"
        and execution["boi_catalogue_enabled"] is False,
        "Native record runs cannot authorize BOI designations",
    )
    manifest = read_json(verified_reference(preparation["input_manifest"]))
    require(
        same_reference(manifest["native_control_binding"], binding_receipt)
        and Path(config["input_manifest"]).resolve()
        == Path(preparation["input_manifest"]["path"]).resolve(),
        "Native run used different bindings",
    )
    runtime_binding(manifest)
    originals = {r["source_id"]: r["identity"] for r in binding["trials"]}
    for row in preparation["sources"]:
        sid = row["source_id"]
        require(
            str(row["parent_tpf"]) == originals[sid]["parent_tpf"]
            and row["raw_fits_sha256"] == originals[sid]["raw_fits_sha256"],
            "Prepared native parent or raw identity changed",
        )
        runtime_source(manifest, sid)
    binary = detection["binary"]
    require(
        catalogue.digest(catalogue.root_path(binary["path"])) == binary["sha256"],
        "Native record detector binary changed",
    )
    inputs = {
        str(run / name): catalogue.digest(run / name)
        for name in (
            "preparation.json",
            "detection.json",
            "queue.json",
            "vetting.json",
            "vetting/results.json",
        )
    }
    inputs[binary["path"]] = binary["sha256"]
    inputs[preparation["input_manifest"]["path"]] = preparation["input_manifest"][
        "sha256"
    ]
    inputs.update(
        native_pixel_evidence(
            run, manifest, preparation["input_manifest"]["path"], sources
        )
    )
    return sources, execution, inputs


def build_report(binding_receipt, runs):
    binding = metadata(binding_receipt)
    verified_reference(binding["binding_implementation"])
    declaration, plan, protection, originals = load_binding_inputs(
        binding["declaration"], binding["materialization"]
    )
    require(
        binding["protocol"] == PROTOCOL
        and binding["role"] == "development"
        and binding["final_scoped_role_authorization"] is False,
        "Native reports require the opened binding contract",
    )
    prospective.exact_index(
        binding["trials"], [s["source_id"] for s in plan["sources"]]
    )
    rows = binding["trials"]
    inputs = {binding_receipt["path"]: binding_receipt["sha256"]}
    for receipt in (
        binding["declaration"],
        binding["materialization"],
        binding["policy"],
        binding["vetting_configuration"],
        binding["binding_implementation"],
        declaration["original_plan"],
        declaration["protected_inputs"],
    ):
        inputs[receipt["path"]] = receipt["sha256"]
    for source, case, original, row in zip(
        plan["sources"], declaration["cases"], originals, rows
    ):
        require(
            row["identity"] == identity(source)
            and row["case"] == case
            and row["original_input"] == original,
            "Native report binding identity changed",
        )
        if row["status"] == READY:
            runtime, _, _ = validate_case(source, case, original, plan, protection)
            require(
                row["runtime_input"] == runtime, "Native report runtime input changed"
            )
    require(set(runs) <= COHORTS, "Unknown native runtime cohort")
    evidence, executions, trials = {}, {}, []
    for cohort in sorted(COHORTS):
        selected = [
            r["source_id"]
            for r in rows
            if r["identity"]["cohort"] == cohort and r["status"] == READY
        ]
        run = Path(runs[cohort]) if cohort in runs else None
        sources = {}
        if run is not None:
            sources, execution, run_inputs = run_evidence(
                run, selected, binding_receipt, binding
            )
            inputs.update(run_inputs)
            executions[cohort] = execution
            evidence[cohort] = str(run.resolve())
        trials.extend(
            trial_result(row, run, sources)
            for row in rows
            if row["identity"]["cohort"] == cohort
        )
    index = {r["source_id"]: r for r in trials}
    trials = [index[r["source_id"]] for r in rows]
    for trial in trials:
        for key in ("proposal_evidence", "source_evidence"):
            receipt = trial[key]
            if receipt:
                inputs[receipt["path"]] = receipt["sha256"]
    complete = all(t["execution_complete"] for t in trials)
    return {
        "schema": 1,
        "protocol": PROTOCOL,
        "outcome_contract": OUTCOME_CONTRACT,
        "status": "COMPLETE_NATIVE_PROSPECTIVE_CONTROL_REPORT"
        if complete
        else "PARTIAL_NATIVE_PROSPECTIVE_CONTROL_REPORT",
        "role": "development",
        "binding": binding_receipt,
        "runs": evidence,
        "disposition_execution": executions,
        "trial_ledger": {"trials": trials, "sha256": catalogue.json_digest(trials)},
        "population": {
            "declared_sources": 200,
            "by_cohort": {
                c: {
                    "declared_sources": 100,
                    "completed_sources": sum(
                        t["execution_complete"] for t in trials if t["category"] == c
                    ),
                }
                for c in sorted(COHORTS)
            },
        },
        "false_boi_rate": {
            "status": "NOT_MEASURED_DEVELOPMENT_RECORD_ONLY",
            "confidence": 0.95,
            "fraction": None,
            "upper_bound": None,
        },
        "event_recovery": {"status": "NOT_MEASURED_NO_NATIVE_EVENT_MATCHING_CONTRACT"},
        "boi_catalogue_enabled": False,
        "final_scoped_role_authorization": False,
        "inputs": inputs,
    }


def verified_report(report):
    require(
        report.get("protocol") == PROTOCOL and report.get("status") in REPORT_STATUSES,
        "Unsupported native prospective report",
    )
    require(
        report == build_report(report["binding"], report["runs"]),
        "Native prospective report differs from its actual bound run evidence",
    )
    return report["trial_ledger"]["trials"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    binder = sub.add_parser("bind")
    for name in ("declaration", "materialization", "policy"):
        binder.add_argument(f"--{name}", type=Path, required=True)
        binder.add_argument(f"--{name}-sha256", required=True)
    binder.add_argument("--out", type=Path, required=True)
    reporter = sub.add_parser("report")
    reporter.add_argument("--bindings", type=Path, required=True)
    reporter.add_argument("--bindings-sha256", required=True)
    reporter.add_argument("--guide-run", type=Path)
    reporter.add_argument("--pair-run", type=Path)
    reporter.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "bind":
        bind(
            *(
                {
                    "path": str(getattr(args, name)),
                    "sha256": getattr(args, f"{name}_sha256"),
                }
                for name in ("declaration", "materialization", "policy")
            ),
            dump_path(args.out),
        )
    else:
        output = dump_path(args.out)
        require(not output.exists(), "Report output exists; preserve prior attempts")
        runs = {
            c: str(p)
            for c, p in (
                ("guide_star_family", args.guide_run),
                ("contamination_pair", args.pair_run),
            )
            if p
        }
        save(
            output,
            build_report(
                {"path": str(args.bindings), "sha256": args.bindings_sha256}, runs
            ),
        )


if __name__ == "__main__":
    main()
