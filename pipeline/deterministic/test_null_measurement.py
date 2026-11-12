"""The null pair preserves inputs, proposal denominators and honest empty variants."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.vetting import calibrate

from pipeline.deterministic import null_exclusions as exclusions
from pipeline.deterministic import null_measurement as measurement
from pipeline.deterministic import (
    release_report as release,
    run,
    signal_contracts as signals,
)
from pipeline.deterministic import test_null_exclusions as exclusion_fixtures
from pipeline.deterministic import test_signal_contracts as signal_fixtures

case = exclusion_fixtures.case
raw = signal_fixtures.raw


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return exclusions.reference(path)


@pytest.fixture
def measured_inputs(case):
    exclusions.produce(
        case.base_path,
        case.base_ref["sha256"],
        case.directory,
        case.out,
        preparation=case.prep_path,
        preparation_sha=case.prep_ref["sha256"],
    )
    binding = exclusions.reference(case.out / "exclusions.json")
    return SimpleNamespace(
        base_manifest=case.base_path,
        base_sha256=case.base_ref["sha256"],
        original_preparation=case.prep_path,
        original_preparation_sha256=case.prep_ref["sha256"],
        exclusions=Path(binding["path"]),
        exclusions_sha256=binding["sha256"],
    )


def test_complete_accounting_does_not_require_original_vetting(measured_inputs):
    base, binding = measurement.measurement_inputs(measured_inputs)
    assert base["sha256"] == measured_inputs.base_sha256
    assert signals.exclusion_document(binding)["internal_adjudication_index"] is None


@pytest.mark.parametrize(
    "change", ["base", "preparation", "exclusions", "population", "evaluation"]
)
def test_measurement_refuses_stale_or_foreign_pins_before_construction(
    measured_inputs, change, monkeypatch
):
    args = measured_inputs
    args.constructed_manifests = args.constructed_sha256 = None
    args.out = args.base_manifest.parent / "measurement"
    if change in {"base", "preparation", "exclusions"}:
        setattr(
            args,
            {
                "base": "base_sha256",
                "preparation": "original_preparation_sha256",
                "exclusions": "exclusions_sha256",
            }[change],
            "0" * 64,
        )
    elif change == "population":
        index = run.read(args.exclusions)
        index["sources"].pop("100")
        args.exclusions_sha256 = write(args.exclusions, index)["sha256"]
    else:
        base = run.read(args.base_manifest)
        base["role"] = "evaluation"
        args.base_sha256 = write(args.base_manifest, base)["sha256"]
    monkeypatch.setattr(
        measurement,
        "execute_variant",
        lambda *_: pytest.fail("Invalid scope reached construction"),
    )
    with pytest.raises(ValueError):
        measurement.measure(args)
    assert not args.out.exists()


@pytest.mark.parametrize("invalid", ["archive", "reserve"])
def test_pixel_storage_refusal_precedes_construction(
    measured_inputs, tmp_path, monkeypatch, invalid
):
    args = measured_inputs
    args.constructed_manifests = args.constructed_sha256 = None
    policy = run.read(run.ROOT / run.DISPOSITION_POLICY)
    policy.update(status="DEVELOPMENT", boi_catalogue_enabled=False)
    args.disposition_policy = tmp_path / "development-policy.json"
    write(args.disposition_policy, policy)
    args.vetting_config = run.ROOT / "pipeline/vetting/config.json"
    args.threads, args.workers = 2, 3
    args.out = tmp_path / "measurement"
    external = tmp_path / "external/forgotten-stars-workspace/dump"
    args.pixel_cache_directory = external / (
        "pixel-archive" if invalid == "archive" else "controls-20260926/pixels"
    )
    args.pixel_cache_directory.mkdir(parents=True)
    marker = args.pixel_cache_directory / "original.fits"
    marker.write_bytes(b"original read-only input witness")
    args.pixel_minimum_free_bytes = 20 * 1024**3 if invalid == "archive" else 0
    contract = run.detector_contract()
    monkeypatch.setattr(run, "detector_contract", lambda: contract)
    monkeypatch.setattr(run, "ROOT", tmp_path / "repository")
    monkeypatch.setattr(run, "EXTERNAL_DUMPS", (external,))
    monkeypatch.setattr(run, "dump_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(
        measurement,
        "execute_variant",
        lambda *_: pytest.fail("Invalid storage reached null measurement"),
    )
    with pytest.raises(
        ValueError, match="declared external control storage|positive integer"
    ):
        measurement.measure(args)
    assert not args.out.exists()
    assert marker.read_bytes() == b"original read-only input witness"


def test_pixel_storage_cli_options_preserve_explicit_reserve(tmp_path, monkeypatch):
    calls = []

    def measured(args):
        calls.append(args)
        return {"status": "CLI_FIXTURE", "report": None, "counts": {}}

    monkeypatch.setattr(measurement, "measure", measured)
    cache = tmp_path / "controls-20260926/pixels"
    argv = [
        "null_measurement",
        "--pixel-cache-directory",
        str(cache),
        "--pixel-minimum-free-bytes",
        str(20 * 1024**3),
    ]
    for flag in (
        "base-manifest",
        "base-sha256",
        "original-preparation",
        "original-preparation-sha256",
        "exclusions",
        "exclusions-sha256",
        "input-graph",
        "out",
    ):
        argv.extend(["--" + flag, str(tmp_path / flag)])
    monkeypatch.setattr(sys, "argv", argv)
    measurement.main()
    assert calls[0].pixel_cache_directory == cache
    assert calls[0].pixel_minimum_free_bytes == 20 * 1024**3


def test_declared_membership_join_preserves_old_graph_without_independence_claim(case):
    old = {
        ("ordinary", "old"): {"origin_parent": "old"},
        ("null", "historical"): {"origin_parent": "history"},
    }
    joined, binding = release.declared_null_graph(
        old, "development_replay", case.base_ref
    )
    assert list(key[1] for key in joined if key[0] == "null") == case.ids
    assert old[("null", "historical")]["origin_parent"] == "history"
    assert joined[("ordinary", "old")] == old[("ordinary", "old")]
    assert binding["original_input_graph_retained_as_history"] is True
    assert "unresolved" in binding["component_basis"]
    with pytest.raises(ValueError, match="development-only"):
        release.declared_null_graph(old, "final_replay", case.base_ref)


@pytest.fixture
def standalone(tmp_path, raw):
    sources = {}
    for sid, quarters in (("300", 3), ("100", 2)):
        arrays = {name: values.copy() for name, values in raw.items()}
        if quarters == 2:
            arrays["quarter"][arrays["quarter"] == 3] = 2
            arrays["qdiag_quarter"][arrays["qdiag_quarter"] == 3] = 2
        path = tmp_path / f"{sid}.npz"
        np.savez_compressed(path, **arrays)
        sources[sid] = {
            "arrays": path.name,
            "sha256": exclusions.reference(path)["sha256"],
            "metadata": {
                "source_id": sid,
                "origin_tpf_kic": int(sid),
                "raw_fits_sha256": sid.zfill(64),
                "shard": 0,
                "shard_path": str(path),
            },
        }
    base = tmp_path / "base.json"
    write(
        base,
        {
            "schema": 1,
            "role": "development",
            "measurement_scope": "UNIT_OPENED_NULL_PAIR",
            "sources": sources,
        },
    )
    return base


def test_paired_construction_and_native_preparation_preserve_declared_cadences(
    standalone, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        run, "scan", lambda args: calls.append(("scan", args.run, args.threads))
    )
    pixel_configs = []

    def vetted(args):
        calls.append(("vet", args.run, args.record_every_gate))
        pixel_configs.append(run.pixel_cache_configuration(args, {}))

    monkeypatch.setattr(run, "vet", vetted)
    monkeypatch.setattr(run, "dump_path", lambda path: Path(path).resolve())
    monkeypatch.setattr(run, "EXTERNAL_DUMPS", (tmp_path / "external",))
    args = SimpleNamespace(
        base_manifest=standalone,
        exclusions=None,
        disposition_policy=run.DISPOSITION_POLICY,
        threads=2,
        workers=3,
        python=Path("python"),
        no_network=True,
        vetting_config=None,
        pixel_cache_directory=tmp_path / "external/controls-20260926/pixels",
        pixel_minimum_free_bytes=20 * 1024**3,
    )
    out = tmp_path / "pair"
    out.mkdir()
    for variant in ("inverted", "quarter_scrambled"):
        measured = measurement.execute_variant(args, out, variant, tmp_path / "binary")
        prepared = run.read(measured / "preparation.json")
        constructed = run.read(out / variant / "construction/input-manifest.json")
        assert prepared["selection"]["source_ids"] == list(constructed["sources"])
        assert list(constructed["sources"]) == (
            ["300", "100"] if variant == "inverted" else ["300"]
        )
        assert [row["source_id"] for row in constructed["construction_ledger"]] == [
            "300",
            "100",
        ]
        for row in prepared["sources"]:
            records = run.read_fsc(
                row["inputs"]["psf_flat"]["fsc_path"], row["source_id"], "psf_flat"
            )
            assert 10 not in records["cadence"] and 20 in records["cadence"]
    assert [name for name, _, _ in calls] == ["scan", "vet", "scan", "vet"]
    assert calls[1][2] is calls[3][2] is True
    assert (
        pixel_configs
        == [
            {
                "pixel_cache_directory": str(args.pixel_cache_directory),
                "pixel_minimum_free_bytes": 20 * 1024**3,
            }
        ]
        * 2
    )


def test_all_short_coverage_retains_empty_variant_without_fake_detector_pass(
    standalone, tmp_path, monkeypatch
):
    base = run.read(standalone)
    del base["sources"]["300"]
    write(standalone, base)
    args = SimpleNamespace(base_manifest=standalone, exclusions=None)
    out = tmp_path / "pair"
    out.mkdir()
    monkeypatch.setattr(
        run, "scan", lambda *_: pytest.fail("Empty variant reached detectors")
    )
    folder = measurement.execute_variant(
        args, out, "quarter_scrambled", tmp_path / "binary"
    )
    control, inputs, manifest, ids, recipe, sources = release.verified_null_run(
        folder, "development_replay"
    )
    assert control is None and ids == [] and sources == {}
    assert (
        manifest["construction_ledger"][0]["status"] == "INVERTED_ONLY_SHORT_COVERAGE"
    )
    assert recipe["variant"] == "quarter_scrambled" and len(inputs) == 2
    with pytest.raises(ValueError):
        release.verified_null_run(folder, "final_replay")
    receipt = run.read(folder / "empty-variant.json")
    receipt["detectors_or_vetting_run"] = True
    write(folder / "empty-variant.json", receipt)
    with pytest.raises(ValueError, match="inapplicable"):
        release.verified_null_run(folder, "development_replay")


@pytest.mark.parametrize("status", ["EVIDENCE_STAGES_COMPLETE", "PARTIAL", None])
def test_multiple_associations_retain_count_without_hiding_source_incompleteness(
    monkeypatch, status
):
    monkeypatch.setattr(release, "exclusion_references", lambda *_: [])
    alert = {
        "event_id": "event",
        "detector_signal_id": "unknown",
        "quarter": 1,
        "start_bkjd": 10,
        "end_bkjd": 10.1,
    }
    proposal = {
        "alerts": [alert],
        "ungated_alerts": [alert],
        "lane_decisions": [
            {
                "proposal_id": "unknown",
                "decision": "PASS",
                "null_exclusion": {"status": "ELIGIBLE", "matches": []},
            }
        ],
    }
    source = {
        "processing_status": status,
        "alerts": [alert],
        "events": [{"event_id": "event", "original_alert": alert}],
        "signals": [
            {
                "signal_id": name,
                "event_ids": ["event"],
                "triage": {
                    "record_every_gate": True,
                    "history": [
                        {
                            "stage": stage,
                            "decision": "DISMISS"
                            if dismissed and stage == "shape"
                            else "PASS",
                        }
                        for stage in signals.FLUX_GATES
                    ],
                },
            }
            for name, dismissed in (("dismissed", True), ("surviving", False))
        ],
    }
    row = release.null_variant_record(proposal, source, {}, "300", "inverted")
    measured = signals.null_variant_counts(row)
    assert measured["trials"] == measured["outcomes"] == 1
    case = {"usable_quarters": 2, "variants": [row], "scientific_abstention": False}
    population = signals.null_population([case])
    assert measured["complete"] is (status == "EVIDENCE_STAGES_COMPLETE")
    assert (
        population["sources"] == 1
        and population["trials"] == population["outcomes"] == 1
    )
    if status != "EVIDENCE_STAGES_COMPLETE":
        assert (
            measurement.measurement_status(population)
            == "INCOMPLETE_DEVELOPMENT_NULL_MEASUREMENT"
        )
        with pytest.raises(ValueError, match="complete null proposal execution"):
            calibrate.development_cases({"null_cases": [case]})
    else:
        assert (
            measurement.measurement_status(population)
            == "COMPLETE_DEVELOPMENT_NULL_MEASUREMENT"
        )
    row["adjudicated_alerts"] = 0
    assert (
        measurement.measurement_status(signals.null_population([case]))
        == "INCOMPLETE_DEVELOPMENT_NULL_MEASUREMENT"
    )
    assert measured["excluded_proposals"] == 0
    proposal["lane_decisions"][0]["decision"] = "REJECT"
    with pytest.raises(ValueError, match="emitted alerts differ"):
        release.null_variant_record(proposal, source, {}, "300", "inverted")


def test_assembly_rejects_foreign_declaration_before_reading_outcomes(
    case, monkeypatch
):
    monkeypatch.setattr(
        release,
        "input_graph",
        lambda *_: pytest.fail("Foreign declaration read outcomes"),
    )
    with pytest.raises(ValueError, match="opened paired runs"):
        release.build_report(
            [],
            case.root,
            "final_replay",
            null_runs=[case.root, case.root],
            null_base=case.base_ref,
        )


def test_null_validation_reuses_bound_base_and_refuses_changed_bytes(
    standalone, tmp_path, monkeypatch
):
    signals.construct(
        SimpleNamespace(
            input_manifest=standalone,
            exclusions=None,
            variant="inverted",
            seed=20261002,
            out=tmp_path / "null",
        )
    )
    manifest = tmp_path / "null/input-manifest.json"
    import evaluation_scope

    original = evaluation_scope.read_reference
    reads = []

    def observed(binding):
        reads.append(binding["path"])
        return original(binding)

    monkeypatch.setattr(evaluation_scope, "read_reference", observed)
    assert signals.null_manifest(run.ROOT, manifest)["seed"] == 20261002
    assert signals.null_manifest(run.ROOT, manifest)["seed"] == 20261002
    assert len(reads) == 1
    standalone.write_bytes(standalone.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        signals.null_manifest(run.ROOT, manifest)


def test_supplied_construction_is_hash_bound_and_avoids_second_array_copy(
    standalone, tmp_path, monkeypatch
):
    binding = write(
        tmp_path / "exclusions.json", {"scope": "ISOLATED_CONSTRUCTION_BINDING_FIXTURE"}
    )
    paths, checksums = [], []
    for variant in ("inverted", "quarter_scrambled"):
        folder = tmp_path / variant
        signals.construct(
            SimpleNamespace(
                input_manifest=standalone,
                exclusions=None,
                variant=variant,
                seed=20261002,
                out=folder,
            )
        )
        path = folder / "input-manifest.json"
        manifest = run.read(path)
        manifest["null_exclusions"] = binding
        checksums.append(write(path, manifest)["sha256"])
        paths.append(path)
    args = SimpleNamespace(
        base_manifest=standalone,
        base_sha256=exclusions.reference(standalone)["sha256"],
        exclusions=Path(binding["path"]),
        exclusions_sha256=binding["sha256"],
        constructed_manifests=paths,
        constructed_sha256=checksums,
    )
    monkeypatch.setattr(
        signals, "construct", lambda *_: pytest.fail("Reused inputs were reconstructed")
    )
    out = tmp_path / "new-run"
    for variant, expected in zip(("inverted", "quarter_scrambled"), paths):
        path, manifest = measurement.construction_input(args, out, variant)
        assert path == expected and manifest["null_construction"]["variant"] == variant
    assert not out.exists()
    args.constructed_sha256[0] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        measurement.construction_input(args, out, "inverted")
