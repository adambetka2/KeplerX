"""Native contracts preserve original scope and refuse before array decoding."""

import copy
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from pipeline.deterministic import cloud_public
from pipeline.deterministic import control_eligibility_archive as archive_headers
from pipeline.deterministic import control_mission_reference as mission
from pipeline.deterministic import control_pair_inputs as pair_inputs
from pipeline.deterministic import native_prospective_controls as native
from pipeline.deterministic import release_report as release
from pipeline.deterministic import run as runner
from pipeline.deterministic import test_control_mission_reference as mission_fixture
from pipeline.photometry import lc2
from pipeline.runtime import disposition_execution, implementation_hashes


def put(path, value):
    native.save(path, value)
    return native.reference(path)


def change(receipt, mutate):
    value = native.metadata(receipt)
    mutate(value)
    return put(Path(receipt["path"]), value)


def authorize_fixture_retirements(case, monkeypatch):
    guard = native.metadata(case.declaration["protected_inputs"])
    original = native.reference(Path(case.declaration["protected_inputs"]["path"]))
    retired = [case.plan["sources"][i]["source_id"] for i in (97, 98, 197, 198)]
    remaining = [case.plan["sources"][i]["source_id"] for i in (99, 199)]
    aliases = [str(800000 + i) for i in range(6)]
    aliases_key = "exact_historical_source_seeds_covered_by_whole_component_approval"
    application = put(
        case.root / "native-component-application.json",
        {
            "status": "NATIVE_DEVELOPMENT_ONLY_APPROVED_WHOLE_COMPONENT_APPLICATION",
            "approval": {"sha256": "a" * 64},
            aliases_key: aliases,
            "all_covered_seeds_absent_from_current_sealed_list": True,
            "native_protection_seed_application_only": True,
            "historical_exclusion_lists_or_null_guard_or_memberships_changed": False,
            "new_authorization_or_additional_retirements_claimed": False,
            "scientific_arrays_or_outcomes_read": False,
        },
    )
    specification = {
        "approval_sha256": "a" * 64,
        "exposure_overlay_sha256": "b" * 64,
        "complete_development_exposure_sha256": "c" * 64,
        "derivation_sha256": "d" * 64,
        "historical_protection_guard_sha256": original["sha256"],
        "native_component_application_sha256": application["sha256"],
        "historical_source_seeds_covered_by_whole_component_approval": aliases,
        "retired_trial_ids": [f"catalogue_fp:evaluation:{sid}" for sid in retired],
        "native_source_ids": retired,
        "remaining_protected_native_source_ids": remaining,
    }
    guard.update(
        schema=2,
        status="AUTHORIZED_RETIREMENT_FULL_COMPONENT_AND_INDEXED_PARENT_CLOSURE",
        authorization_scope="ORIGINAL_NATIVE_PROSPECTIVE_DEVELOPMENT_ONLY",
        source_ids=remaining,
        approval={"sha256": specification["approval_sha256"]},
        exposure_overlay={"sha256": specification["exposure_overlay_sha256"]},
        complete_development_exposure_before_access={"sha256": "c" * 64},
        derivation={"sha256": "d" * 64},
        historical_null_exclusion_guard=original,
        native_development_component_application=application,
        historical_source_seeds_covered_by_whole_component_approval=aliases,
        retired_trial_ids=specification["retired_trial_ids"],
        native_targets_reopened_by_authorization=retired,
        remaining_protected_native_source_ids=remaining,
        six_scene_link_witnesses_retained=True,
        historical_null_exclusions_not_released=True,
        original_native_trials_per_class=100,
        current_roles_launch_counts_and_caps_rewritten=False,
    )
    receipt = put(case.root / "authorized-guard.json", guard)
    specification["protection_guard_sha256"] = receipt["sha256"]
    path = case.root / "retirement-specification.json"
    put(path, specification)
    monkeypatch.setattr(pair_inputs, "RETIREMENT_SPECIFICATION", path)
    monkeypatch.setattr(
        pair_inputs, "RETIREMENT_SPECIFICATION_SHA256", native.reference(path)["sha256"]
    )
    case.declaration["protected_inputs"] = receipt
    return specification, path


@pytest.mark.parametrize("i", [97, 197])
def test_authorized_retirement_reaches_native_binding_with_original_denominators(
    case, monkeypatch, i
):
    authorize_fixture_retirements(case, monkeypatch)
    enable(case, i, actual_quarters=(1,))
    bound = bind(case)
    assert bound["counts"] == {native.READY: 1, "NATIVE_INPUT_REFUSED": 199}
    assert [r["source_id"] for r in bound["trials"]] == [
        s["source_id"] for s in case.plan["sources"]
    ]
    assert sum("Protected" in r.get("message", "") for r in bound["trials"]) == 2
    assert case.plan["sources"][i]["source_id"] in case.reads


@pytest.mark.parametrize(
    "kind",
    [
        "tamper",
        "approval",
        "extra-retirement",
        "extra-seed",
        "core-release",
        "null-release",
    ],
)
def test_retirement_receipt_refuses_unapproved_changes_before_payloads(
    case, monkeypatch, kind
):
    specification, path = authorize_fixture_retirements(case, monkeypatch)
    receipt = case.declaration["protected_inputs"]
    guard = native.metadata(receipt)
    if kind == "approval":
        guard["approval"]["sha256"] = "e" * 64
    elif kind == "extra-retirement":
        guard["retired_trial_ids"].append("ordinary:evaluation:999999")
    elif kind == "extra-seed":
        guard["historical_source_seeds_covered_by_whole_component_approval"].append(
            "999999"
        )
    elif kind == "core-release":
        guard["remaining_protected_native_source_ids"].pop()
    elif kind == "null-release":
        guard["historical_null_exclusions_not_released"] = False
    else:
        guard["source_ids"] = []
    changed = put(Path(receipt["path"]), guard)
    if kind != "tamper":
        specification["protection_guard_sha256"] = changed["sha256"]
        put(path, specification)
        monkeypatch.setattr(
            pair_inputs,
            "RETIREMENT_SPECIFICATION_SHA256",
            native.reference(path)["sha256"],
        )
    deny_decoding(case, monkeypatch)
    with pytest.raises(ValueError, match="Unapproved prospective retirement"):
        native.protection_sets(changed)


@pytest.mark.parametrize(
    "field",
    [
        "all_covered_seeds_absent_from_current_sealed_list",
        "native_protection_seed_application_only",
        "historical_exclusion_lists_or_null_guard_or_memberships_changed",
        "new_authorization_or_additional_retirements_claimed",
    ],
)
def test_component_application_refuses_current_seeds_or_null_release(
    case, monkeypatch, field
):
    specification, path = authorize_fixture_retirements(case, monkeypatch)
    receipt = case.declaration["protected_inputs"]
    guard = native.metadata(receipt)
    application = guard["native_development_component_application"]
    value = native.metadata(application)
    value[field] = not value[field]
    changed_application = put(Path(application["path"]), value)
    guard["native_development_component_application"] = changed_application
    changed = put(Path(receipt["path"]), guard)
    specification["native_component_application_sha256"] = changed_application["sha256"]
    specification["protection_guard_sha256"] = changed["sha256"]
    put(path, specification)
    monkeypatch.setattr(
        pair_inputs, "RETIREMENT_SPECIFICATION_SHA256", native.reference(path)["sha256"]
    )
    deny_decoding(case, monkeypatch)
    with pytest.raises(ValueError, match="Unapproved native component application"):
        native.protection_sets(changed)


def test_component_application_body_tamper_refuses_before_payloads(case, monkeypatch):
    authorize_fixture_retirements(case, monkeypatch)
    guard = native.metadata(case.declaration["protected_inputs"])
    application = guard["native_development_component_application"]
    value = native.metadata(application)
    value["native_protection_seed_application_only"] = False
    put(Path(application["path"]), value)
    deny_decoding(case, monkeypatch)
    with pytest.raises(ValueError, match="Input unavailable or changed"):
        native.protection_sets(case.declaration["protected_inputs"])


def test_component_application_uses_colocated_exact_bytes_on_another_host(
    case, monkeypatch
):
    specification, path = authorize_fixture_retirements(case, monkeypatch)
    receipt = case.declaration["protected_inputs"]
    guard = native.metadata(receipt)
    guard["native_development_component_application"]["path"] = (
        "/original-cloud-only/path/native-component-application.json"
    )
    changed = put(Path(receipt["path"]), guard)
    specification["protection_guard_sha256"] = changed["sha256"]
    put(path, specification)
    monkeypatch.setattr(
        pair_inputs, "RETIREMENT_SPECIFICATION_SHA256", native.reference(path)["sha256"]
    )
    deny_decoding(case, monkeypatch)
    protected = native.protection_sets(changed)
    assert protected["source_ids"] == set(
        specification["remaining_protected_native_source_ids"]
    )


def test_surviving_core_component_overrides_retirement_permission(case, monkeypatch):
    specification, path = authorize_fixture_retirements(case, monkeypatch)
    enable(case, 97, actual_quarters=(1,))
    receipt = case.declaration["protected_inputs"]
    guard = native.metadata(receipt)
    guard["protected_full_component_ids"] = sorted(
        guard["protected_full_component_ids"]
        + [native.identity(case.plan["sources"][97])["component_id"]]
    )
    changed = put(Path(receipt["path"]), guard)
    specification["protection_guard_sha256"] = changed["sha256"]
    put(path, specification)
    monkeypatch.setattr(
        pair_inputs, "RETIREMENT_SPECIFICATION_SHA256", native.reference(path)["sha256"]
    )
    case.declaration["protected_inputs"] = changed
    deny_decoding(case, monkeypatch)
    bound = bind(case)
    assert bound["counts"] == {"NATIVE_INPUT_REFUSED": 200}
    assert "Protected component" in bound["trials"][97]["message"]


def test_retirement_configuration_is_part_of_recorded_code_identity(case, monkeypatch):
    specification, path = authorize_fixture_retirements(case, monkeypatch)
    specification["approval_sha256"] = "e" * 64
    put(path, specification)
    deny_decoding(case, monkeypatch)
    with pytest.raises(ValueError, match="Input unavailable or changed"):
        native.protection_sets(case.declaration["protected_inputs"])


@pytest.fixture
def case(tmp_path, monkeypatch):
    sources = []
    originals = []
    records = []
    for i in range(200):
        sid = str(100000 + i)
        source = {
            "source_id": sid,
            "cohort": "guide_star_family" if i < 100 else "contamination_pair",
            "parent_tpf": str(10000 + i),
            "shard": "0",
            "uri": "fixture-native-input",
            "product": {"kic": 1},
            "pool": {
                "raw_fits_sha256": hashlib.sha256(sid.encode()).hexdigest(),
                "metadata_dependency_component_id": f"component-{sid}",
            },
        }
        sources.append(source)
        original = {
            "source_id": sid,
            "cohort": source["cohort"],
            "status": "INPUT_BINDING_FAILED",
        }
        originals.append(original)
        records.append(put(tmp_path / f"original-{sid}.json", original))
    plan = {
        "sources": sources,
        "donor_bindings": [
            {
                "source_id": s["source_id"],
                "parent_tpf": s["parent_tpf"],
                "public_donor_product_identities": [
                    {"gaia_designation": f"Gaia DR3 {300000 + i}"}
                ],
            }
            for i, s in enumerate(sources)
            if s["cohort"] == "contamination_pair"
        ],
    }
    plan_receipt = put(tmp_path / "plan.json", plan)
    monkeypatch.setattr(native, "ORIGINAL_PLAN_SHA256", plan_receipt["sha256"])
    protected = [sources[i]["source_id"] for i in (97, 98, 99, 197, 198, 199)]
    specification = native.read_json(
        Path(native.__file__).with_name("cloud-joint-scope.json")
    )
    guard = {
        "schema": 1,
        "status": "FULL_COMPONENT_AND_INDEXED_PARENT_CLOSURE_BEFORE_PIXEL_PROVISIONING",
        "inputs": {
            "current_graph": {
                "sha256": specification["artifact"]["canonical_json_sha256"]
            }
        },
        "six_scene_links_remain_protected": True,
        "five_retired_reservations_remain_historical_exclusions": True,
        "arrays_or_outcomes_read": False,
        "source_ids": protected,
        "raw_or_pixel_content_hashes": [],
        "parent_identities": [],
        "additional_original_stopped_components_protected": [],
        "protected_full_component_ids": ["protected-full-component"],
        "potential_evaluation_scene_links": [
            {"target_source_id": sid} for sid in protected
        ],
    }
    declaration = {
        "schema": 1,
        "protocol": native.PROTOCOL,
        "role": "development",
        "original_plan": plan_receipt,
        "protected_inputs": put(tmp_path / "guard.json", guard),
        "cases": [
            {"source_id": s["source_id"], "identity": native.identity(s)}
            for s in sources
        ],
    }
    materialization = {
        "inputs": {"plan": plan_receipt},
        "runtime_policy_required": "DEVELOPMENT",
        "final_scoped_role_authorization": False,
        "records": records,
    }
    state = SimpleNamespace(
        root=tmp_path,
        plan=plan,
        declaration=declaration,
        materialization=materialization,
        originals=originals,
        policy={"status": "DEVELOPMENT", "boi_catalogue_enabled": False},
        out=tmp_path / "bound",
        raw={},
        reads=[],
    )

    def check_fixture_plan(value, _locations):
        assert value == state.plan

    index = tmp_path / "parents.sqlite"
    with sqlite3.connect(index) as database:
        database.executescript(
            "CREATE TABLE metadata (key TEXT, value TEXT); "
            "CREATE TABLE sources (source_id TEXT, parent_tpf INTEGER);"
        )
        database.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [("schema", "1"), ("sources", "406547")],
        )
        database.executemany(
            "INSERT INTO sources VALUES (?,?)",
            [(s["source_id"], int(s["parent_tpf"])) for s in sources],
        )
    configuration = native.read_json(native.CONFIGURATION)
    configuration.update(
        parent_index=str(index), parent_index_sha256=native.reference(index)["sha256"]
    )
    put(tmp_path / "base-config.json", configuration)
    monkeypatch.setattr(native, "CONFIGURATION", tmp_path / "base-config.json")

    monkeypatch.setattr(native, "verify_original_plan", check_fixture_plan)

    def reader(path, sid, _):
        state.reads.append(sid)
        return state.raw[sid]

    monkeypatch.setattr(cloud_public, "fits_arrays", reader)
    return state


def receipts(case):
    case.declaration["original_plan"] = put(case.root / "plan.json", case.plan)
    case.materialization["inputs"]["plan"] = case.declaration["original_plan"]
    case.materialization["records"] = [
        put(case.root / f"original-{r['source_id']}.json", r) for r in case.originals
    ]
    return (
        put(case.root / "declaration.json", case.declaration),
        put(case.root / "materialization.json", case.materialization),
        put(case.root / "policy.json", case.policy),
    )


def enable(case, i=0, actual_quarters=(0, 1)):
    source = case.plan["sources"][i]
    sid = source["source_id"]
    identity = native.identity(source)
    original_file = case.root / f"{sid}.fits"
    original_file.write_bytes(sid.encode())
    cadences = 200 * len(actual_quarters)
    raw = {
        "source_id": np.full(cadences, int(sid), dtype=np.int64),
        "time_bkjd": np.arange(cadences, dtype=float) / 48 + 100,
        "quarter": np.repeat(actual_quarters, 200).astype(np.int32),
        "cadenceno": np.arange(cadences, dtype=np.int64),
        "psf_flat_kbonus_flux": np.ones(cadences),
        "psf_flat_kbonus_flux_err": np.full(cadences, 0.001),
        "sap_quality": np.zeros(cadences, dtype=np.int32),
        "flatten_mask": np.zeros(cadences, dtype=np.int32),
    }
    case.raw[sid] = raw
    folder = case.root / f"original-inputs-{sid}"
    (folder / "arrays").mkdir(parents=True)
    arrays = native.prospective.save_arrays(folder / "arrays" / f"{sid}.npz", raw)
    archive = {
        "source": source,
        "headers": [
            {
                "name": "PRIMARY",
                "cards": {
                    "GAIAID": sid,
                    "TPFORG": source["parent_tpf"],
                    "KEPLERID": 1,
                    "RA_OBJ": 280.0,
                    "DEC_OBJ": 40.0,
                    "GMAG": 15.0,
                },
            },
            *[
                {"name": f"LIGHTCURVE_Q{q}", "cards": {"QUARTER": q}}
                for q in actual_quarters
            ],
        ],
    }
    original = {
        "source_id": sid,
        "cohort": source["cohort"],
        "status": "ORIGINAL_RAW_READER_ARRAYS_AND_PREPARATION_BOUND",
        "component_id": identity["component_id"],
        "final_scoped_role_authorization": False,
        "original_payload": native.reference(original_file),
        "archive_record": put(case.root / f"{sid}-archive.json", archive),
        "metadata": native.prospective.source_metadata(source, archive),
        "preparation": native.prospective.mask_exposure(raw)[0],
        "arrays": arrays,
        "array_inventory": native.prospective.array_inventory(raw),
    }
    case.originals[i] = original
    clock = put(
        case.root / f"{sid}-clock.json",
        {
            "identity": identity,
            "time_scale": "BKJD",
            "signal_id": "explicit-fixture-native-clock",
            "independent_native_timing": True,
            "derived_from_detector_proposals": False,
            "events": [
                {"quarter": q, "centre_bkjd": 101.0 + q, "duration_days": 0.1}
                for q in actual_quarters
            ],
        },
    )
    measured = put(
        case.root / f"{sid}-measurements.json",
        {
            "identity": identity,
            "event_clock": clock,
            "status": "MEASURED",
            "native_inputs": [native.reference(original_file)],
            "measurements": {"explicit_fixture_measurement": 1.0},
            "pointing_pixel_coupling_measured": True,
            "target_donor_attribution_measured": True,
        },
    )
    witness = put(
        case.root / f"{sid}-mechanism.json",
        {
            "identity": identity,
            "native_false_class_verified": True,
            "mechanism": "explicit-fixture-mechanism",
            "event_clock": clock,
            "scientific_outcomes_used": False,
            "measurements": measured,
        },
    )
    truth = {
        "schema": 1,
        "identity": identity,
        "native_false_class_verified": True,
        "mechanism": "explicit-fixture-mechanism",
        "mechanism_evidence": [witness],
        "event_clock": clock,
        "required_inputs": sorted(native.REQUIREMENTS),
    }
    if source["cohort"] == "contamination_pair":
        donor_file = case.root / "donor.fits"
        primary = fits.PrimaryHDU()
        primary.header.update(
            KEPLERID=int(source["parent_tpf"]),
            GAIAID=f"Gaia DR3 {300000 + i}",
            TPFORG=int(source["parent_tpf"]),
        )
        tables = []
        for q in actual_quarters:
            table = fits.BinTableHDU.from_columns(
                [fits.Column(name="TIME", format="D", array=np.array([101.0 + q]))],
                name=f"LIGHTCURVE_Q{q}",
            )
            table.header.update(QUARTER=q, TPFORG=int(source["parent_tpf"]))
            tables.append(table)
        fits.HDUList([primary, *tables]).writeto(donor_file)
        headers = archive_headers.local_headers(donor_file)
        original_binding = next(
            b for b in case.plan["donor_bindings"] if b["source_id"] == sid
        )
        prior = put(
            case.root / "donor-original-headers.json",
            {
                "binding": original_binding,
                "status": "VERIFIED_DONOR_ARCHIVE_HEADERS",
                "archive_product_bytes": donor_file.stat().st_size,
                "headers": headers,
            },
        )
        truth["donor"] = {
            "identity": {
                **identity,
                "source_id": str(300000 + i),
                "raw_fits_sha256": native.reference(donor_file)["sha256"],
                "component_id": "fixture-donor",
            },
            "domain": "KBONUS_PSF_NATIVE_SIGNAL",
            "use": "DONOR_SIGNAL_REFERENCE_ONLY",
            "reference": native.reference(donor_file),
            "provisioning": put(
                case.root / "donor-provisioning.json",
                {
                    "binding": original_binding,
                    "status": "CANONICAL_DONOR_RAW_BYTES_AND_PREPARATION_BOUND",
                    "prior_archive_record": prior,
                    "headers": headers,
                    "payload": native.reference(donor_file),
                },
            ),
        }
    if source["cohort"] == "contamination_pair":
        clock = change(
            clock,
            lambda v: v.update(
                signal_source_id=truth["donor"]["identity"]["source_id"],
                native_signal_reference=truth["donor"]["reference"],
            ),
        )
        measured = change(measured, lambda v: v.update(event_clock=clock))
        witness = change(
            witness, lambda v: v.update(event_clock=clock, measurements=measured)
        )
        truth.update(event_clock=clock, mechanism_evidence=[witness])
    truth_receipt = put(case.root / f"{sid}-truth.json", truth)
    row = case.declaration["cases"][i]
    row["truth"] = truth_receipt
    row["recipe"] = put(
        case.root / f"{sid}-recipe.json",
        {
            "schema": 1,
            "identity": identity,
            "mode": "native_unchanged_replay",
            "operation": "identity",
            "truth": truth_receipt,
            "quarters": list(actual_quarters),
            "required_inputs": sorted(native.REQUIREMENTS),
        },
    )
    scene_path = case.root / f"{sid}-scene.jsonl"
    scene_path.write_text(json.dumps(original["metadata"]) + "\n")
    scene = native.reference(scene_path)
    quarters = []
    for q in actual_quarters:
        pixel = case.root / f"{sid}-q{q}.fits"
        primary = fits.PrimaryHDU()
        primary.header.update(KEPLERID=int(source["parent_tpf"]), QUARTER=q, CHANNEL=1)
        table = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="FLUX", format="4E", dim="(2,2)", array=np.zeros((1, 2, 2))
                )
            ]
        )
        table.header.update(TSTART=100.0, TSTOP=101.0, BJDREFI=2454833, BJDREFF=0.0)
        aperture = fits.ImageHDU(np.zeros((2, 2), dtype=np.int32))
        fits.HDUList([primary, table, aperture]).writeto(pixel)
        prf = case.root / "kplr02.1_2011265_prf.fits"
        if not prf.exists():
            prf_primary = fits.PrimaryHDU()
            prf_primary.header.update(CHANNEL=1, MODULE=2, OUTPUT=1)
            prf_primary.writeto(prf)
        quarter = {
            "quarter": q,
            "parent_tpf": source["parent_tpf"],
            "parent_candidates": [source["parent_tpf"]],
            "pixel": native.reference(pixel),
            "prf": native.reference(prf),
            "scene": scene,
            "siblings": {},
        }
        proof = {
            **quarter,
            "identity": identity,
            "extraction_parent_verified": True,
            "scene_complete": True,
            "siblings_complete": True,
        }
        quarter["witness"] = put(case.root / f"{sid}-q{q}-witness.json", proof)
        quarters.append(quarter)
    row["physical_bindings"] = put(
        case.root / f"{sid}-physical.json",
        {"identity": identity, "quarters": quarters, "scene_metadata": scene},
    )
    return row


def bind(case):
    return native.bind(*receipts(case), case.out)


def deny_decoding(case, monkeypatch):
    monkeypatch.setattr(
        cloud_public,
        "fits_arrays",
        lambda *_: pytest.fail("Payload decoded before refusal"),
    )


def test_missing_recipes_preserve_every_original_id_and_denominator(case, monkeypatch):
    deny_decoding(case, monkeypatch)
    bound = bind(case)
    assert [r["source_id"] for r in bound["trials"]] == [
        s["source_id"] for s in case.plan["sources"]
    ]
    assert bound["counts"] == {"NATIVE_INPUT_REFUSED": 200}
    jobs = native.read_json(case.out / "runtime.json")
    assert all(
        j["declared_sources"] == 100 and j["argv"] is None for j in jobs.values()
    )
    assert sum("Protected" in r["message"] for r in bound["trials"]) == 6
    report = native.build_report(native.reference(case.out / "bindings.json"), {})
    assert report["population"]["declared_sources"] == 200
    assert all(
        t["boi_catalogue"] is None and not t["execution_complete"]
        for t in report["trial_ledger"]["trials"]
    )
    assert report["false_boi_rate"]["upper_bound"] is None
    assert native.verified_report(report) == report["trial_ledger"]["trials"]


@pytest.mark.parametrize("i", [0, 100])
def test_explicit_native_replay_reaches_actual_runner_reader(case, i):
    enable(case, i)
    bound = bind(case)
    assert bound["counts"] == {native.READY: 1, "NATIVE_INPUT_REFUSED": 199}
    cohort = case.plan["sources"][i]["cohort"]
    jobs = native.read_json(case.out / "runtime.json")
    job = jobs[cohort]
    ids = runner.read(Path(job["source_ids"]["path"]))
    selected = runner.select_sources(
        [0], 1, ids, experimental_manifest=case.out / "input-manifest.json"
    )
    assert [s["source_id"] for s in selected] == ids
    metadata, raw = lc2.load_source(case.root, ids[0], case.out / "input-manifest.json")
    assert (
        metadata["raw_fits_sha256"]
        == case.plan["sources"][i]["pool"]["raw_fits_sha256"]
    )
    assert native.prospective.array_inventory(
        raw
    ) == native.prospective.array_inventory(case.raw[ids[0]])
    assert "--record-every-gate" in job["argv"]
    assert "--limit" in job["argv"] and job["selected_sources"] == 1


def test_complete_q1_guide_without_original_q0_binds_and_reads(case):
    row = enable(case, actual_quarters=(1,))
    bound = bind(case)
    assert bound["counts"] == {native.READY: 1, "NATIVE_INPUT_REFUSED": 199}
    assert [r["source_id"] for r in bound["trials"]] == [
        s["source_id"] for s in case.plan["sources"]
    ]
    jobs = native.read_json(case.out / "runtime.json")
    assert all(job["declared_sources"] == 100 for job in jobs.values())
    metadata, raw = lc2.load_source(
        case.root, row["source_id"], case.out / "input-manifest.json"
    )
    assert [q["quarter"] for q in metadata["quarter_parent_bindings"]] == [1]
    assert set(raw["quarter"]) == {1}
    assert (
        native.prospective.array_inventory(raw) == case.originals[0]["array_inventory"]
    )


def test_actual_original_quarter_cannot_be_discarded(case, monkeypatch):
    row = enable(case)
    recipe = native.metadata(row["recipe"])
    recipe["quarters"] = [1]
    physical = native.metadata(row["physical_bindings"])
    physical["quarters"] = [q for q in physical["quarters"] if q["quarter"] == 1]
    row["physical_bindings"] = put(Path(row["physical_bindings"]["path"]), physical)
    deny_decoding(case, monkeypatch)
    with pytest.raises(ValueError, match="discarded unchanged-replay quarter"):
        native.quarter_bindings(
            row,
            recipe,
            case.originals[0],
            native.metadata(case.declaration["protected_inputs"]),
        )


def test_q1_guide_cannot_bind_an_empty_quarter_recipe(case, monkeypatch):
    row = enable(case, actual_quarters=(1,))
    row["recipe"] = change(row["recipe"], lambda v: v.update(quarters=[]))
    deny_decoding(case, monkeypatch)
    bound = bind(case)
    assert "distinct ordered mission quarters" in bound["trials"][0]["message"]
    assert bound["counts"] == {"NATIVE_INPUT_REFUSED": 200}


@pytest.mark.parametrize(
    "kind",
    [
        "physical",
        "template",
        "truth",
        "mechanism",
        "clock",
        "missing-clock",
        "requirements",
        "coupling",
    ],
)
def test_absent_unsupported_or_nonphysical_truth_refuses_before_decoding(
    case, monkeypatch, kind
):
    row = enable(case, actual_quarters=(1,))
    if kind == "physical":
        row["recipe"] = change(
            row["recipe"], lambda v: v.update(mode="physical_perturbation")
        )
    elif kind == "template":
        row["recipe"] = change(
            row["recipe"], lambda v: v.update(operation="eclipsing_binary_template")
        )
    elif kind == "truth":
        row.pop("truth")
    elif kind == "mechanism":
        row["truth"] = change(
            row["truth"], lambda v: v.update(native_false_class_verified=False)
        )
        row["recipe"] = change(row["recipe"], lambda v: v.update(truth=row["truth"]))
    elif kind in {"clock", "missing-clock"}:
        truth = native.metadata(row["truth"])
        if kind == "missing-clock":
            truth.pop("event_clock")
        else:
            truth["event_clock"] = change(
                truth["event_clock"],
                lambda v: v.update(derived_from_detector_proposals=True),
            )
        row["truth"] = put(Path(row["truth"]["path"]), truth)
        row["recipe"] = change(row["recipe"], lambda v: v.update(truth=row["truth"]))
    elif kind == "coupling":
        witness = native.metadata(
            native.metadata(row["truth"])["mechanism_evidence"][0]
        )
        change(
            witness["measurements"],
            lambda v: v.update(pointing_pixel_coupling_measured=False),
        )
    else:
        row["recipe"] = change(row["recipe"], lambda v: v.update(required_inputs=[]))
    deny_decoding(case, monkeypatch)
    assert bind(case)["counts"] == {"NATIVE_INPUT_REFUSED": 200}


@pytest.mark.parametrize(
    "kind", ["source", "raw", "parent", "component", "full-component"]
)
def test_full_component_guard_precedes_native_truth_and_arrays(case, monkeypatch, kind):
    enable(case, actual_quarters=(1,))
    original = native.identity(case.plan["sources"][0])
    name, value = {
        "source": ("source_ids", original["source_id"]),
        "raw": ("raw_or_pixel_content_hashes", original["raw_fits_sha256"]),
        "parent": ("parent_identities", original["parent_tpf"]),
        "component": (
            "additional_original_stopped_components_protected",
            original["component_id"],
        ),
        "full-component": ("protected_full_component_ids", original["component_id"]),
    }[kind]
    case.declaration["protected_inputs"] = change(
        case.declaration["protected_inputs"],
        lambda v: v.update({name: sorted(set(v[name] + [value]))}),
    )
    deny_decoding(case, monkeypatch)
    bound = bind(case)
    assert "Protected" in bound["trials"][0]["message"]


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "parent",
        "ambiguous",
        "pixel",
        "prf",
        "scene",
        "siblings",
        "witness",
    ],
)
def test_physical_dependencies_are_exact_and_never_filled(case, monkeypatch, kind):
    row = enable(case, actual_quarters=(1,))
    physical = native.metadata(row["physical_bindings"])
    if kind == "missing":
        physical["quarters"].pop()
    elif kind == "extra":
        physical["quarters"].append(copy.deepcopy(physical["quarters"][0]))
    elif kind == "parent":
        physical["quarters"][0]["parent_tpf"] = "999"
    elif kind == "ambiguous":
        physical["quarters"][0]["parent_candidates"].append("999")
    elif kind in {"pixel", "prf"}:
        Path(physical["quarters"][0][kind]["path"]).write_bytes(b"changed-fixture")
    elif kind == "witness":
        physical["quarters"][0].pop("witness")
    else:
        quarter = physical["quarters"][0]
        quarter["witness"] = change(
            quarter["witness"], lambda v: v.update({f"{kind}_complete": False})
        )
    row["physical_bindings"] = put(Path(row["physical_bindings"]["path"]), physical)
    deny_decoding(case, monkeypatch)
    assert bind(case)["counts"] == {"NATIVE_INPUT_REFUSED": 200}


def test_ambiguous_original_donor_cannot_be_resolved_by_recipe(case, monkeypatch):
    enable(case, 100)
    case.plan["donor_bindings"][0]["public_donor_product_identities"].append(
        {"source_id": "400000"}
    )
    monkeypatch.setattr(
        native,
        "ORIGINAL_PLAN_SHA256",
        put(case.root / "plan.json", case.plan)["sha256"],
    )
    deny_decoding(case, monkeypatch)
    bound = bind(case)
    assert "ambiguous" in bound["trials"][100]["message"]


def test_runtime_rechecks_recipe_hash_before_npz_reader(case, monkeypatch):
    row = enable(case)
    bind(case)
    Path(row["recipe"]["path"]).write_text("{}")
    monkeypatch.setattr(
        lc2.np, "load", lambda *_a, **_k: pytest.fail("Decoded changed native contract")
    )
    with pytest.raises(ValueError, match="changed|differs"):
        lc2.load_source(case.root, row["source_id"], case.out / "input-manifest.json")


@pytest.mark.parametrize("mutation", ["drop", "reorder", "replace", "role"])
def test_original_scope_and_roles_are_immutable(case, monkeypatch, mutation):
    if mutation == "drop":
        case.declaration["cases"].pop()
    elif mutation == "reorder":
        case.declaration["cases"].reverse()
    elif mutation == "replace":
        case.declaration["cases"][0]["identity"]["raw_fits_sha256"] = "a" * 64
    else:
        case.declaration["role"] = "evaluation"
    deny_decoding(case, monkeypatch)
    with pytest.raises(ValueError):
        bind(case)
    assert not case.out.exists()


def test_previous_attempts_are_preserved(case):
    case.out.mkdir()
    (case.out / "retained").write_bytes(b"original-attempt")
    with pytest.raises(ValueError, match="Preserve prior attempts"):
        bind(case)
    assert (case.out / "retained").read_bytes() == b"original-attempt"


def fixture_run(case, source_index=0, proposed=False):
    enable(case, source_index)
    bind(case)
    run = case.root / "run"
    for name in ("inputs", "proposals", "vetting/sources"):
        (run / name).mkdir(parents=True)
    sid = case.plan["sources"][source_index]["source_id"]
    manifest = case.out / "input-manifest.json"
    policy_receipt = native.reference(case.root / "policy.json")
    fsc = run / "inputs" / "fixture.fsc"
    fsc.write_bytes(b"opaque-prepared-fixture")
    preparation = {
        "status": "PREPARED",
        "detector_contract": runner.detector_contract(),
        "implementation": implementation_hashes(),
        "requested_sources": 1,
        "selection": {"source_ids": [sid]},
        "cohort_sha256": runner.json_sha256([sid]),
        "input_manifest": native.reference(manifest),
        "sources": [
            {
                "source_id": sid,
                "parent_tpf": case.plan["sources"][source_index]["parent_tpf"],
                "raw_fits_sha256": case.plan["sources"][source_index]["pool"][
                    "raw_fits_sha256"
                ],
                "inputs": {
                    "psf_flat": {
                        "fsc_path": str(fsc),
                        "fsc_sha256": native.reference(fsc)["sha256"],
                    }
                },
            }
        ],
    }
    put(run / "preparation.json", preparation)
    binary = run / "detector-fixture"
    binary.write_bytes(b"explicit-detector-fixture")
    alerts = (
        [{"event_id": "native-event", "kind": "PERIODIC_WINDOW", "quarter": 1}]
        if proposed
        else []
    )
    put(
        run / "detection.json",
        {
            "status": "DETECTION_COMPLETE",
            "sources": 1,
            "alerted_sources": int(proposed),
            "alerts": len(alerts),
            "binary": native.reference(binary),
        },
    )
    put(run / "queue.json", {"rows": [{"source_id": sid}] if proposed else []})
    execution = disposition_execution(case.policy)
    configuration = native.read_json(case.out / "vetting-config.json")
    put(
        run / "vetting.json",
        {
            **configuration,
            "disposition_policy": policy_receipt["path"],
            "input_manifest": str(manifest),
            "triage_execution": "RECORD_EVERY_GATE",
        },
    )
    put(
        run / "vetting/results.json",
        {
            "processing_status": "EVIDENCE_STAGES_COMPLETE",
            "sources": int(proposed),
            "original_alerts_preserved": len(alerts),
            "disposition_execution": execution,
        },
    )
    put(run / "proposals" / f"{sid}.json", {"source_id": sid, "alerts": alerts})
    if proposed:
        specification = native.read_json(manifest)
        item = specification["sources"][sid]
        put(
            run / "vetting/pixel-retrieval.json",
            {
                "status": "EXPLICIT_EXPERIMENT_INPUTS",
                "measurement_scope": specification["measurement_scope"],
                "input_manifest": native.reference(manifest),
            },
        )
        put(
            run
            / "vetting/pixel-manifests"
            / f"{item['metadata']['origin_tpf_kic']}.json",
            {
                "quarters": {
                    q: dict(pixel, status="AVAILABLE")
                    for q, pixel in item["pixels"].items()
                }
            },
        )
        put(
            run / "vetting/sources" / f"{sid}.json",
            {
                "source_id": sid,
                "alerts": alerts,
                "events": [
                    {
                        "event_id": "native-event",
                        "original_alert": {"kind": "PERIODIC_WINDOW"},
                    }
                ],
                "signals": [
                    {
                        "signal_id": label,
                        "event_ids": ["native-event"],
                        "triage": {"status": "ACTIVE", "history": []},
                    }
                    for label in ("native-alias", "unmatched-alias")
                ],
                "processing_status": "EVIDENCE_STAGES_COMPLETE",
                "disposition_policy": case.policy,
                "disposition_execution": execution,
                "disposition": {
                    "status": "UNRESOLVED",
                    "boi_signal_count": 0,
                    "signals": [
                        {"signal_id": "native-alias", "disposition": "FALSE_ALARM"},
                        {"signal_id": "unmatched-alias", "disposition": "UNRESOLVED"},
                    ],
                },
            },
        )
    cohort = case.plan["sources"][source_index]["cohort"]
    return run, sid, {cohort: str(run)}


@pytest.mark.parametrize("proposed", [False, True])
def test_record_report_keeps_nonproposals_abstentions_and_every_alias(case, proposed):
    _, sid, runs = fixture_run(case, proposed=proposed)
    report = native.build_report(native.reference(case.out / "bindings.json"), runs)
    trial = next(r for r in report["trial_ledger"]["trials"] if r["source_id"] == sid)
    assert trial["execution_complete"] is True and trial["boi_catalogue"] is None
    assert trial["status"] == (
        "VETTING_COMPLETE" if proposed else "DETECTOR_NO_PROPOSAL"
    )
    assert trial["scientific_abstention"] is proposed
    assert [s["signal_id"] for s in trial["signals"]] == (
        ["native-alias", "unmatched-alias"] if proposed else []
    )
    assert report["status"] == "PARTIAL_NATIVE_PROSPECTIVE_CONTROL_REPORT"
    assert (
        report["population"]["by_cohort"]["guide_star_family"]["declared_sources"]
        == 100
    )
    assert report["false_boi_rate"]["upper_bound"] is None
    assert len(release.report_records(report)) == 200


@pytest.mark.parametrize(
    "kind",
    [
        "generic",
        "missing-receipt",
        "manifest",
        "missing-quarter",
        "hash",
        "path",
        "prf",
    ],
)
def test_native_report_requires_actual_bound_pixel_retrieval(case, kind):
    run, _, runs = fixture_run(case, proposed=True)
    retrieval = run / "vetting/pixel-retrieval.json"
    if kind == "missing-receipt":
        retrieval.unlink()
    elif kind in {"generic", "manifest"}:
        body = native.read_json(retrieval)
        if kind == "generic":
            body["status"] = "GENERIC_PARENT_INDEX_RETRIEVAL"
        else:
            body["input_manifest"]["sha256"] = "0" * 64
        put(retrieval, body)
    else:
        path = next((run / "vetting/pixel-manifests").glob("*.json"))
        body = native.read_json(path)
        if kind == "missing-quarter":
            body["quarters"].pop("1")
        elif kind == "prf":
            body["quarters"]["1"]["prf_calibration"]["sha256"] = "0" * 64
        else:
            body["quarters"]["1"]["sha256" if kind == "hash" else "path"] = "changed"
        put(path, body)
    with pytest.raises((ValueError, OSError)):
        native.build_report(native.reference(case.out / "bindings.json"), runs)


@pytest.mark.parametrize("proposed", [False, True])
@pytest.mark.parametrize("same_directory", [False, True])
def test_native_report_checks_resolved_pixel_cache_directory(
    case, proposed, same_directory
):
    run, _, runs = fixture_run(case, proposed=proposed)
    config = native.read_json(run / "vetting.json")
    original = native.catalogue.root_path(config["pixel_cache_directory"]).resolve()
    config["pixel_cache_directory"] = str(
        original if same_directory else original / "different-cache"
    )
    put(run / "vetting.json", config)
    if same_directory:
        native.build_report(native.reference(case.out / "bindings.json"), runs)
    else:
        with pytest.raises(ValueError, match="bound vetting configuration"):
            native.build_report(native.reference(case.out / "bindings.json"), runs)


@pytest.mark.parametrize("same_file", [False, True])
def test_native_report_checks_resolved_pixel_product_path(case, same_file):
    run, _, runs = fixture_run(case, proposed=True)
    path = next((run / "vetting/pixel-manifests").glob("*.json"))
    body = native.read_json(path)
    original = Path(body["quarters"]["1"]["path"])
    alternative = original.with_name("pixel-alias" if same_file else "different-pixel")
    if same_file:
        alternative.symlink_to(original)
    else:
        alternative.write_bytes(original.read_bytes())
    body["quarters"]["1"]["path"] = str(alternative)
    put(path, body)
    if same_file:
        report = native.build_report(native.reference(case.out / "bindings.json"), runs)
        assert (
            sum(row["execution_complete"] for row in report["trial_ledger"]["trials"])
            == 1
        )
    else:
        with pytest.raises(ValueError, match="differs from its physical binding"):
            native.build_report(native.reference(case.out / "bindings.json"), runs)


@pytest.mark.parametrize(
    "kind", ["record-mode", "parent", "raw", "proposal", "alias", "input", "binary"]
)
def test_report_refuses_changed_runtime_evidence(case, kind):
    run, sid, runs = fixture_run(case, proposed=True)
    if kind == "record-mode":
        change(
            native.reference(run / "vetting.json"),
            lambda v: v.update(triage_execution="SHORT_CIRCUIT"),
        )
    elif kind in {"parent", "raw"}:
        key = "parent_tpf" if kind == "parent" else "raw_fits_sha256"
        change(
            native.reference(run / "preparation.json"),
            lambda v: v["sources"][0].update({key: "changed"}),
        )
    elif kind == "proposal":
        change(
            native.reference(run / "proposals" / f"{sid}.json"),
            lambda v: v.update(source_id="999"),
        )
    elif kind == "alias":
        change(
            native.reference(run / "vetting/sources" / f"{sid}.json"),
            lambda v: v["disposition"]["signals"].pop(),
        )
    elif kind == "input":
        (run / "inputs" / "fixture.fsc").write_bytes(b"changed-input")
    else:
        (run / "detector-fixture").write_bytes(b"changed-binary")
    if kind == "alias":
        report = native.build_report(native.reference(case.out / "bindings.json"), runs)
        assert report["trial_ledger"]["trials"][0]["execution_complete"] is False
    else:
        with pytest.raises(ValueError):
            native.build_report(native.reference(case.out / "bindings.json"), runs)


def test_assembler_retains_all_200_without_adding_them_to_original_graph(
    case, monkeypatch
):
    bind(case)
    report = native.build_report(native.reference(case.out / "bindings.json"), {})
    receipt = put(case.root / "report.json", report)
    entry = {"report": {k: receipt[k] for k in ("path", "sha256")}}
    monkeypatch.setattr(
        release,
        "input_graph",
        lambda *_: ({"joint_independence_complete": True}, {}, {}),
    )
    assembled = release.build_report([entry], case.root / "graph", "development_replay")
    assert assembled["population"]["by_cohort"] == {
        "guide_star_family": 100,
        "contamination_pair": 100,
    }
    assert len(assembled["cases"]) == 200 and all(
        c["status"] == "FAILED" for c in assembled["cases"]
    )
    assert assembled["population"]["status"] == "INPUT_INDEPENDENCE_INCOMPLETE"
    assert assembled["boi_catalogue_enabled"] is False
    with pytest.raises(ValueError, match="repeat"):
        release.build_report([entry, entry], case.root / "graph", "development_replay")
    with pytest.raises(ValueError, match="development-only"):
        release.build_report([entry], case.root / "graph", "final_replay")


def test_tampered_native_report_cannot_hide_failed_rows(case):
    bind(case)
    report = native.build_report(native.reference(case.out / "bindings.json"), {})
    report["trial_ledger"]["trials"].pop()
    with pytest.raises(ValueError, match="actual bound run evidence"):
        release.report_records(report)


@pytest.mark.parametrize(
    "kind", ["quarter", "pixel-parent", "prf-channel", "q0", "sibling-membership"]
)
def test_native_header_and_original_inventory_mismatches_refuse(
    case, monkeypatch, kind
):
    row = enable(case)
    physical = native.metadata(row["physical_bindings"])
    quarter = physical["quarters"][0]
    if kind in {"quarter", "pixel-parent"}:
        with fits.open(quarter["pixel"]["path"], mode="update") as hdus:
            hdus[0].header["QUARTER" if kind == "quarter" else "KEPLERID"] = 17
        quarter["pixel"] = native.reference(quarter["pixel"]["path"])
    elif kind == "prf-channel":
        with fits.open(quarter["prf"]["path"], mode="update") as hdus:
            hdus[0].header["CHANNEL"] = 2
        quarter["prf"] = native.reference(quarter["prf"]["path"])
    elif kind == "q0":
        row["recipe"] = change(row["recipe"], lambda v: v.update(quarters=[1]))
    else:
        config = native.read_json(native.CONFIGURATION)
        with sqlite3.connect(config["parent_index"]) as database:
            database.execute("INSERT INTO sources VALUES (?,?)", ("555555", 10000))
        config["parent_index_sha256"] = native.reference(config["parent_index"])[
            "sha256"
        ]
        put(native.CONFIGURATION, config)
    if kind in {"quarter", "pixel-parent", "prf-channel"}:
        quarter["witness"] = change(
            quarter["witness"],
            lambda v: v.update({k: quarter[k] for k in ("pixel", "prf")}),
        )
        row["physical_bindings"] = put(Path(row["physical_bindings"]["path"]), physical)
    deny_decoding(case, monkeypatch)
    assert bind(case)["counts"] == {"NATIVE_INPUT_REFUSED": 200}


def test_protected_donor_cannot_be_read_for_a_permitted_target(case, monkeypatch):
    row = enable(case, 100)
    donor_id = native.metadata(row["truth"])["donor"]["identity"]["source_id"]
    case.declaration["protected_inputs"] = change(
        case.declaration["protected_inputs"],
        lambda v: v.update(source_ids=sorted(v["source_ids"] + [donor_id])),
    )
    deny_decoding(case, monkeypatch)
    bound = bind(case)
    assert "Protected" in bound["trials"][100]["message"]


@pytest.mark.parametrize("kind", ["receipt", "identity", "quarter", "hash"])
def test_kbonus_donor_requires_provisioned_native_product(case, monkeypatch, kind):
    row = enable(case, 100)
    truth = native.metadata(row["truth"])
    donor = truth["donor"]
    if kind == "receipt":
        donor.pop("provisioning")
    elif kind == "hash":
        Path(donor["reference"]["path"]).write_bytes(b"changed-donor")
    else:
        with fits.open(donor["reference"]["path"], mode="update") as hdus:
            if kind == "identity":
                hdus[0].header["GAIAID"] = "Gaia DR3 999"
            else:
                hdus[2].header["QUARTER"] = 17
        donor["reference"] = native.reference(donor["reference"]["path"])
        donor["identity"]["raw_fits_sha256"] = donor["reference"]["sha256"]
        donor["provisioning"] = change(
            donor["provisioning"], lambda v: v.update(payload=donor["reference"])
        )
        clock = change(
            truth["event_clock"],
            lambda v: v.update(native_signal_reference=donor["reference"]),
        )
        witness = truth["mechanism_evidence"][0]
        measured = change(
            native.metadata(witness)["measurements"],
            lambda v: v.update(event_clock=clock),
        )
        witness = change(
            witness, lambda v: v.update(event_clock=clock, measurements=measured)
        )
        truth.update(event_clock=clock, mechanism_evidence=[witness])
    row["truth"] = put(Path(row["truth"]["path"]), truth)
    row["recipe"] = change(row["recipe"], lambda v: v.update(truth=row["truth"]))
    deny_decoding(case, monkeypatch)
    bound = bind(case)
    assert bound["counts"] == {"NATIVE_INPUT_REFUSED": 200}
    expected = {
        "receipt": "provisioning",
        "identity": "Donor Gaia catalogue/header differs",
        "quarter": "Native KBonus donor header",
        "hash": "changed",
    }
    assert expected[kind] in bound["trials"][100]["message"]


@pytest.mark.parametrize(
    "changed", [None, "url", "source", "bytes", "headers", "status"]
)
def test_published_header_only_donor_receipt_reaches_native_runtime(
    case, monkeypatch, changed
):
    from pipeline.deterministic.control_donor_headers import donor_product

    row = enable(case, 100)
    truth = native.metadata(row["truth"])
    donor = truth["donor"]
    previous = native.metadata(donor["provisioning"])
    original = previous["binding"]
    prior = native.metadata(previous["prior_archive_record"])
    job = {
        "kind": "DONOR",
        "target_source_id": row["source_id"],
        "native_source_id": donor["identity"]["source_id"],
        "global_parent": donor["identity"]["parent_tpf"],
        "canonical_url": donor_product(original)["url"],
        "product_bytes": prior["archive_product_bytes"],
        "original_header_record": previous["prior_archive_record"],
    }
    published = {
        "schema": 1,
        "status": "CANONICAL_KBONUS_DONOR_BYTES_AND_IDENTITY_VERIFIED",
        "job": job,
        "native_headers": previous["headers"],
        "payload": donor["reference"],
        "flux_or_scientific_arrays_decoded": False,
        "native_donor_signal_reference_only": True,
        "original_donor_full_file_hash_witness_supplied": False,
    }
    if changed == "url":
        job["canonical_url"] = "https://example.invalid/substitute.fits"
    elif changed == "source":
        job["native_source_id"] = "999999"
    elif changed == "bytes":
        job["product_bytes"] += 1
    elif changed == "headers":
        published["native_headers"] = []
    elif changed == "status":
        published["status"] = "UNAVAILABLE"
    donor["provisioning"] = put(Path(donor["provisioning"]["path"]), published)
    row["truth"] = put(Path(row["truth"]["path"]), truth)
    row["recipe"] = change(row["recipe"], lambda v: v.update(truth=row["truth"]))
    if changed:
        deny_decoding(case, monkeypatch)
        assert bind(case)["counts"] == {"NATIVE_INPUT_REFUSED": 200}
    else:
        assert bind(case)["trials"][100]["status"] == native.READY
        _, raw = lc2.load_source(
            case.root, row["source_id"], case.out / "input-manifest.json"
        )
        assert (
            native.prospective.array_inventory(raw)
            == case.originals[100]["array_inventory"]
        )


def enable_mission(case, monkeypatch, kic=3230227):
    folder = case.root / "mission-fixture"
    folder.mkdir()
    binding, prior, receipts, header, payload = mission_fixture.inputs.__wrapped__(
        folder, kic=kic
    )
    source = case.plan["sources"][100]
    sid = source["source_id"]
    source["parent_tpf"] = binding["parent_tpf"]
    case.declaration["cases"][100]["identity"] = native.identity(source)
    configuration = native.read_json(native.CONFIGURATION)
    with sqlite3.connect(configuration["parent_index"]) as database:
        database.execute(
            "UPDATE sources SET parent_tpf=? WHERE source_id=?",
            (int(source["parent_tpf"]), sid),
        )
    configuration["parent_index_sha256"] = native.reference(
        configuration["parent_index"]
    )["sha256"]
    put(native.CONFIGURATION, configuration)
    binding.update(source_id=sid, public_donor_product_identities=[])
    case.plan["donor_bindings"][0] = binding
    case.declaration["original_plan"] = put(case.root / "plan.json", case.plan)
    case.materialization["inputs"]["plan"] = case.declaration["original_plan"]
    monkeypatch.setattr(
        native, "ORIGINAL_PLAN_SHA256", case.declaration["original_plan"]["sha256"]
    )
    prior = change(prior, lambda v: v.update(binding=binding))
    header = change(header, lambda v: v.update(source_target_id=sid))
    row = enable(case, 100)
    contracts, products = [], []
    for q in (0, 1):
        if q:
            with fits.open(payload["path"]) as hdus:
                hdus[0].header["QUARTER"] = q
                hdus.writeto(folder / "mission-q1.fits")
            payload = native.reference(folder / "mission-q1.fits")
            value = native.metadata(header)
            value.update(headers=archive_headers.local_headers(Path(payload["path"])))
            header = put(folder / "mission-q1-header.json", value)
        contract = mission.bind_reference(binding, prior, receipts, header, set())
        contracts.append(contract)
        products.append(
            {
                "original_contract_index": q,
                "identity": contract["identity"],
                "mission_product": contract["mission_product"],
                "domain": mission.DOMAIN,
                "use": "DONOR_SIGNAL_REFERENCE_ONLY",
                "status": "EXACT_NATIVE_SAP_REFERENCE_AVAILABLE",
                "payload": payload,
            }
        )
    ledger = put(
        folder / "contracts.json",
        {
            "status": "EXPLICIT_MISSION_SAP_REFERENCE_CONTRACTS_ONLY",
            "contracts": contracts,
        },
    )
    provisioning = put(
        folder / "provisioning.json",
        {
            "original_contracts": ledger,
            "contracts": products,
        },
    )
    truth = native.metadata(row["truth"])
    truth["donor"] = {
        "identity": {
            **row["identity"],
            "source_id": "3",
            "raw_fits_sha256": products[0]["payload"]["sha256"],
            "component_id": "fixture-mission-donor",
        },
        "domain": mission.DOMAIN,
        "use": "DONOR_SIGNAL_REFERENCE_ONLY",
        "reference": ledger,
        "provisioning": provisioning,
    }
    clock = change(
        truth["event_clock"],
        lambda v: v.update(
            signal_source_id="3",
            native_signal_reference=ledger,
        ),
    )
    witness = truth["mechanism_evidence"][0]
    measured = change(
        native.metadata(witness)["measurements"], lambda v: v.update(event_clock=clock)
    )
    witness = change(
        witness, lambda v: v.update(event_clock=clock, measurements=measured)
    )
    truth.update(event_clock=clock, mechanism_evidence=[witness])
    row["truth"] = put(Path(row["truth"]["path"]), truth)
    row["recipe"] = change(row["recipe"], lambda v: v.update(truth=row["truth"]))
    return row


def test_explicit_mission_references_do_not_enter_discovery_photometry(
    case, monkeypatch
):
    row = enable_mission(case, monkeypatch)
    bound = bind(case)
    assert bound["trials"][100]["status"] == native.READY
    manifest = native.read_json(case.out / "input-manifest.json")
    assert (
        manifest["sources"][row["source_id"]]["sha256"]
        == case.originals[100]["arrays"]["sha256"]
    )
    assert set(manifest["sources"]) == {row["source_id"]}
    _, raw = lc2.load_source(
        case.root, row["source_id"], case.out / "input-manifest.json"
    )
    assert (
        native.prospective.array_inventory(raw)
        == case.originals[100]["array_inventory"]
    )


def test_identifier_specific_kic_with_exact_distinct_donor_can_bind(case, monkeypatch):
    row = enable_mission(case, monkeypatch, kic=3757588)
    bound = bind(case)
    assert bound["counts"] == {native.READY: 1, "NATIVE_INPUT_REFUSED": 199}
    assert bound["trials"][100]["status"] == native.READY
    assert row["identity"]["parent_tpf"] == "3757588"
    _, raw = lc2.load_source(
        case.root, row["source_id"], case.out / "input-manifest.json"
    )
    assert (
        native.prospective.array_inventory(raw)
        == case.originals[100]["array_inventory"]
    )


@pytest.mark.parametrize("kind", ["ambiguous", "catalogue-target", "declared-target"])
def test_specific_kic_ambiguous_or_target_donor_refuses_before_payload(
    case, monkeypatch, kind
):
    row = enable_mission(case, monkeypatch, kic=3757588)
    truth = native.metadata(row["truth"])
    donor = truth["donor"]
    ledger = native.metadata(donor["reference"])
    if kind == "declared-target":
        donor["identity"]["source_id"] = row["source_id"]
    else:
        receipts = ledger["contracts"][0]["inputs"]["identity_receipts"]
        for direction in ("dr3_to_dr2", "dr2_to_dr3"):
            value = native.metadata(receipts[direction])
            if kind == "ambiguous":
                value["rows"].append({"dr3_source_id": 4, "dr2_source_id": 2})
            else:
                value["rows"][0]["dr3_source_id"] = int(row["source_id"])
            receipts[direction] = put(Path(receipts[direction]["path"]), value)
        for contract in ledger["contracts"]:
            contract["inputs"]["identity_receipts"] = receipts
        donor["reference"] = put(Path(donor["reference"]["path"]), ledger)
        donor["provisioning"] = change(
            donor["provisioning"],
            lambda v: v.update(original_contracts=donor["reference"]),
        )
    clock = change(
        truth["event_clock"],
        lambda v: v.update(
            signal_source_id=donor["identity"]["source_id"],
            native_signal_reference=donor["reference"],
        ),
    )
    witness = truth["mechanism_evidence"][0]
    measured = change(
        native.metadata(witness)["measurements"], lambda v: v.update(event_clock=clock)
    )
    witness = change(
        witness, lambda v: v.update(event_clock=clock, measurements=measured)
    )
    truth.update(event_clock=clock, mechanism_evidence=[witness])
    row["truth"] = put(Path(row["truth"]["path"]), truth)
    row["recipe"] = change(row["recipe"], lambda v: v.update(truth=row["truth"]))
    deny_decoding(case, monkeypatch)
    monkeypatch.setattr(
        native,
        "payload_path",
        lambda *_: pytest.fail("Payload accessed before identity refusal"),
    )
    bound = bind(case)
    message = "reciprocal" if kind == "ambiguous" else "same source|shared"
    assert bound["counts"] == {"NATIVE_INPUT_REFUSED": 200}
    assert re.search(message, bound["trials"][100]["message"])


@pytest.mark.parametrize(
    "kind", ["unavailable", "hash", "domain", "identity", "quarter", "protected"]
)
def test_mission_reference_product_gaps_refuse_before_target_decoding(
    case, monkeypatch, kind
):
    row = enable_mission(case, monkeypatch)
    truth = native.metadata(row["truth"])
    donor = truth["donor"]
    products = native.metadata(donor["provisioning"])
    if kind == "unavailable":
        products["contracts"][0]["status"] = "UNAVAILABLE"
    elif kind == "hash":
        Path(products["contracts"][0]["payload"]["path"]).write_bytes(
            b"changed-mission"
        )
    elif kind == "domain":
        donor["domain"] = "KBONUS_PSF_NATIVE_SIGNAL"
    elif kind == "identity":
        products["contracts"][0]["identity"]["gaia_dr3"] = "4"
    elif kind == "quarter":
        products["contracts"].pop()
    else:
        case.declaration["protected_inputs"] = change(
            case.declaration["protected_inputs"],
            lambda v: v.update(
                raw_or_pixel_content_hashes=[
                    products["contracts"][0]["payload"]["sha256"]
                ],
            ),
        )
    donor["provisioning"] = put(Path(donor["provisioning"]["path"]), products)
    row["truth"] = put(Path(row["truth"]["path"]), truth)
    row["recipe"] = change(row["recipe"], lambda v: v.update(truth=row["truth"]))
    deny_decoding(case, monkeypatch)
    assert bind(case)["counts"] == {"NATIVE_INPUT_REFUSED": 200}


def test_original_plan_relocations_preserve_content_and_historical_code(
    tmp_path, monkeypatch
):
    files = {
        key: put(tmp_path / f"{key}.json", {"original": key})
        for key in (
            "opened_packet",
            "public_catalogue",
            "stopped_identity_pool",
            "sealed_identity_exclusions",
        )
    }
    plan = {
        "inputs": files,
        "sources": [{"source_id": "original"}],
        "implementation": {"old-code.py": "a" * 64},
    }
    rebuilt = copy.deepcopy(plan)
    rebuilt["implementation"] = {"current-code.py": "b" * 64}
    rebuilt["inputs"]["sealed_identity_exclusions"]["path"] = "different-checkout"
    monkeypatch.setattr(native, "build_plan", lambda *_: rebuilt)
    native.verify_original_plan(plan, {})
    alternative = put(tmp_path / "alternate.json", {"replacement": True})
    with pytest.raises(ValueError, match="content changed"):
        native.verify_original_plan(plan, {"opened_packet": alternative})
    rebuilt["sources"][0]["source_id"] = "replacement"
    with pytest.raises(ValueError, match="recipe, identity"):
        native.verify_original_plan(plan, {})


def test_runtime_rejects_a_replaced_original_plan_before_decoding(case, monkeypatch):
    row = enable(case)
    bind(case)
    declaration_path = case.root / "declaration.json"
    declaration = native.read_json(declaration_path)
    declaration["original_plan"]["sha256"] = "a" * 64
    new_declaration = put(declaration_path, declaration)
    binding = native.read_json(case.out / "bindings.json")
    binding["declaration"] = new_declaration
    new_binding = put(case.out / "bindings.json", binding)
    manifest = native.read_json(case.out / "input-manifest.json")
    manifest["native_control_binding"] = new_binding
    put(case.out / "input-manifest.json", manifest)
    monkeypatch.setattr(
        lc2.np, "load", lambda *_a, **_kw: pytest.fail("Decoded replacement scope")
    )
    with pytest.raises(ValueError, match="original|Original"):
        lc2.load_source(case.root, row["source_id"], case.out / "input-manifest.json")


@pytest.mark.parametrize("protected", [False, True])
def test_original_sibling_content_proof_and_guard_reach_runtime(
    case, monkeypatch, protected
):
    source = case.plan["sources"][1]
    source["parent_tpf"] = case.plan["sources"][0]["parent_tpf"]
    case.declaration["cases"][1]["identity"] = native.identity(source)
    row = enable(case)
    enable(case, 1)
    case.declaration["cases"][1] = {
        "source_id": source["source_id"],
        "identity": native.identity(source),
    }
    index_config = native.read_json(native.CONFIGURATION)
    with sqlite3.connect(index_config["parent_index"]) as database:
        database.execute(
            "UPDATE sources SET parent_tpf=? WHERE source_id=?",
            (int(source["parent_tpf"]), source["source_id"]),
        )
    index_config["parent_index_sha256"] = native.reference(
        index_config["parent_index"]
    )["sha256"]
    put(native.CONFIGURATION, index_config)
    monkeypatch.setattr(
        native,
        "ORIGINAL_PLAN_SHA256",
        put(case.root / "plan.json", case.plan)["sha256"],
    )
    original = case.originals[1]
    sibling = {
        "identity": native.identity(source),
        "original_input": put(case.root / "original-100001.json", original),
        "input": {
            "arrays": original["arrays"]["path"],
            "sha256": original["arrays"]["sha256"],
            "metadata": original["metadata"],
        },
    }
    physical = native.metadata(row["physical_bindings"])
    scene_path = Path(physical["scene_metadata"]["path"])
    scene_path.write_text(
        json.dumps(case.originals[0]["metadata"])
        + "\n"
        + json.dumps(original["metadata"])
        + "\n"
    )
    physical["scene_metadata"] = native.reference(scene_path)
    for quarter in physical["quarters"]:
        quarter["scene"] = physical["scene_metadata"]
        quarter["siblings"] = {source["source_id"]: sibling}
        quarter["witness"] = change(
            quarter["witness"],
            lambda v: v.update({k: quarter[k] for k in ("scene", "siblings")}),
        )
    row["physical_bindings"] = put(Path(row["physical_bindings"]["path"]), physical)
    if protected:
        case.declaration["protected_inputs"] = change(
            case.declaration["protected_inputs"],
            lambda v: v.update(
                source_ids=sorted(v["source_ids"] + [source["source_id"]])
            ),
        )
        deny_decoding(case, monkeypatch)
    bound = bind(case)
    if protected:
        assert bound["counts"] == {"NATIVE_INPUT_REFUSED": 200}
        assert "Protected" in bound["trials"][0]["message"]
    else:
        assert bound["counts"][native.READY] == 1
        _, raw = lc2.load_source(
            case.root, source["source_id"], case.out / "input-manifest.json"
        )
        assert native.prospective.array_inventory(
            raw
        ) == native.prospective.array_inventory(case.raw[source["source_id"]])
