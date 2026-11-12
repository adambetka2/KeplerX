import importlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "pipeline/photometry")]

from deterministic import (
    false_class_control_report,
    false_class_injections,
    false_class_overlay,
    injections,
    pixel_injections,
    pixel_parent_audit,
    pixel_support,
    pixel_transfer_audit,
)
from lc2 import manifest_source


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Unexpected generator contract failure"),
        KeyError("Missing mandatory array"),
        OSError("Disk unavailable"),
        RuntimeError("Broken transit model"),
    ],
)
def test_unexpected_eligibility_error_stops_without_excluding_input(
    tmp_path, monkeypatch, error
):
    item = {
        "category": "periodic_injection",
        "pool_rank": 1,
        "source_id": "123",
        "shard": 0,
    }

    def fail(*args):
        raise error

    monkeypatch.setattr(injections, "evaluate_eligibility", fail)
    results, progress = {}, {}
    with pytest.raises(type(error)) as caught:
        injections.evaluate_work(tmp_path, [item], {}, progress, results)
    assert caught.value is error
    assert results == {}
    assert progress == {}
    assert (tmp_path / "eligibility.jsonl").read_text() == ""
    assert not (tmp_path / "progress.json").exists()


@pytest.mark.parametrize(
    "error",
    [
        injections.InsufficientCadences("Insufficient observed cadences"),
        injections.InjectionIneligible(
            "Periodic injection has insufficient observed coverage"
        ),
    ],
)
def test_expected_eligibility_failure_keeps_explicit_exclusion(
    tmp_path, monkeypatch, error
):
    item = {
        "category": "periodic_injection",
        "pool_rank": 1,
        "source_id": "123",
        "shard": 0,
    }

    def fail(*args):
        raise error

    monkeypatch.setattr(injections, "evaluate_eligibility", fail)
    results, progress = {}, {}
    injections.evaluate_work(tmp_path, [item], {}, progress, results)
    recorded = json.loads((tmp_path / "eligibility.jsonl").read_text())
    assert recorded["status"] == "INELIGIBLE"
    assert recorded["reason"] == f"{type(error).__name__}: {error}"
    assert results[("periodic_injection", 1)] == recorded
    assert progress["stage"] == "ELIGIBILITY_COMPLETE"


def test_failed_eligibility_batch_preserves_prior_rows_and_stops_before_reserve(
    tmp_path, monkeypatch
):
    items = [
        {
            "category": "periodic_injection",
            "pool_rank": rank,
            "source_id": str(rank),
            "shard": 0,
        }
        for rank in range(1, 5)
    ]
    visited = []

    def evaluate(item, plan):
        rank = item["pool_rank"]
        visited.append(rank)
        if rank == 1:
            raise injections.InjectionIneligible(
                "Declared physical input is unavailable"
            )
        if rank == 3:
            raise ValueError("Unexpected raw identity mismatch")
        return {**item, "status": "ELIGIBLE", "truth": None}

    monkeypatch.setattr(injections, "evaluate_eligibility", evaluate)
    results, progress = {}, {}
    with pytest.raises(ValueError, match="Unexpected raw identity mismatch"):
        injections.evaluate_work(tmp_path, items, {}, progress, results)
    rows = [
        json.loads(line)
        for line in (tmp_path / "eligibility.jsonl").read_text().splitlines()
    ]
    assert visited == [1, 2, 3]
    assert [row["pool_rank"] for row in rows] == [1, 2]
    assert [row["status"] for row in rows] == ["INELIGIBLE", "ELIGIBLE"]
    assert set(results) == {("periodic_injection", 1), ("periodic_injection", 2)}
    assert progress.get("stage") != "ELIGIBILITY_COMPLETE"


def test_unmeasurable_noise_and_geometry_have_explicit_eligibility_errors():
    records = np.zeros(100, dtype=[("quarter", "i4"), ("flux", "f4")])
    with pytest.raises(injections.InjectionIneligible, match="point-noise"):
        injections.robust_point_sigma(records)
    with pytest.raises(
        injections.InjectionIneligible, match="circular transit geometry"
    ):
        injections.orbit(0.0001, 1, 0.2, 0.05)


def test_false_class_overlay_rebases_without_changing_prior_paths(
    tmp_path, monkeypatch
):
    local = tmp_path / "local"
    local.mkdir()
    prior = tmp_path / "prior"
    prior.mkdir()
    monkeypatch.setattr(false_class_overlay, "LOCAL_DUMP", local)
    assert false_class_overlay.local_output(local / "new") == local / "new"
    (local / "external-link").symlink_to(prior)
    with pytest.raises(ValueError, match="local repository dump"):
        false_class_overlay.local_output(local / "external-link")
    original = {
        "manifest": {
            "arrays": "arrays/1.npz",
            "pixels": {"1": {"path": "pixels/1/q01.fits.gz"}},
        }
    }
    rebased = false_class_overlay.rebase(original, prior)
    assert rebased["manifest"]["arrays"] == str(prior / "arrays/1.npz")
    assert rebased["manifest"]["pixels"]["1"]["path"] == str(
        prior / "pixels/1/q01.fits.gz"
    )
    assert original["manifest"]["arrays"] == "arrays/1.npz"
    source_cache, new_cache = prior / "cache", local / "cache"
    source_cache.mkdir()
    new_cache.mkdir()
    source = source_cache / "kplr000000001-2009131105131_lpd-targ.fits.gz"
    source.write_bytes(b"verified archive pixel")
    assert false_class_overlay.seed_cache(new_cache, source_cache) == 1
    assert (new_cache / source.name).is_symlink()
    assert false_class_overlay.seed_cache(new_cache, source_cache) == 0


def test_false_class_overlay_rejects_prior_gap(tmp_path):
    pool, prior, seed = (tmp_path / name for name in ("pool", "prior", "seed"))
    for folder in (pool, prior, seed, prior / "records", prior / "unavailable-records"):
        folder.mkdir(exist_ok=True)
    plan_path = tmp_path / "plan.json"
    plan = {
        "classes": ["instrumental_artifact"],
        "development_per_class": 0,
        "evaluation_per_class": 3,
    }
    plan_path.write_text(json.dumps(plan))
    (pool / "freeze.json").write_text("{}")
    binding = {"path": str(seed / "materialization.json")}
    identity = {
        "pool_sha256": false_class_overlay.digest(pool / "freeze.json"),
        "plan_sha256": false_class_overlay.digest(plan_path),
        "generator": {"generator": "frozen"},
        "category": None,
        "limit": None,
        "development_seed": binding,
    }
    false_class_overlay.write(
        prior / "progress.json",
        {"status": "RUNNING", "identity": identity, "complete": 2},
    )
    rows = [
        {
            "source_id": str(sid),
            "category": "instrumental_artifact",
            "parent_tpf": str(sid + 10),
            "raw_fits_sha256": f"raw-{sid}",
        }
        for sid in (1, 2, 3)
    ]
    for sid in (1, 3):
        array = prior / f"arrays/{sid}.npz"
        pixel = prior / f"pixels/{sid}/q01.fits.gz"
        array.parent.mkdir(parents=True, exist_ok=True)
        pixel.parent.mkdir(parents=True, exist_ok=True)
        array.write_bytes(b"array")
        pixel.write_bytes(b"pixel")
        false_class_overlay.write(
            prior / "records" / f"{sid}.json",
            {
                "truth": {
                    "source_id": str(sid),
                    "category": "instrumental_artifact",
                    "parent_tpf": str(sid + 10),
                    "raw_fits_sha256": f"raw-{sid}",
                    "injected_quarters": [1],
                },
                "manifest": {
                    "metadata": {"source_id": str(sid)},
                    "base_raw_fits_sha256": f"raw-{sid}",
                    "arrays": f"arrays/{sid}.npz",
                    "sha256": false_class_overlay.digest(array),
                    "pixels": {
                        "1": {
                            "path": f"pixels/{sid}/q01.fits.gz",
                            "base_parent_kic": sid + 10,
                        }
                    },
                },
            },
        )
    controls = SimpleNamespace(
        PLAN=plan_path,
        POOL=pool,
        pool_rows=lambda *args: rows,
        development_seed=lambda *args: {"binding": binding},
        generator_identity=lambda: identity["generator"],
    )
    pixels = SimpleNamespace(validate_record=lambda *args: True)
    with pytest.raises(ValueError, match="candidate-pool prefix"):
        false_class_overlay.audit_prior(prior, controls, pixels)


def test_false_class_overlay_allows_frozen_support_skip_after_partial_prefix(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(false_class_overlay, "ROOT", tmp_path)
    support = tmp_path / "dump/support"
    support.mkdir(parents=True)
    rule = support / "support-rule.json"
    rule.write_text("frozen support rule")
    row = {
        "source_id": "3",
        "category": "background_contaminant",
        "parent_tpf": "30",
        "raw_fits_sha256": "raw-3",
    }
    evidence = support / "support-3.json"
    false_class_overlay.write(
        evidence,
        {
            "source_id": "3",
            "category": row["category"],
            "frozen_parent_tpf": 30,
            "raw_fits_sha256": "raw-3",
            "rule_sha256": false_class_overlay.digest(rule),
            "unavailable_parent_quarters": [1],
        },
    )
    skipped = {
        "source_id": "3",
        "category": row["category"],
        "reason": "Frozen parent TPF unavailable in quarters [1]",
        "support_evidence": str(evidence),
        "support_evidence_sha256": false_class_overlay.digest(evidence),
        "support_rule_sha256": false_class_overlay.digest(rule),
    }
    assert false_class_overlay.verify_skip(row, skipped)
    rows = [{**row, "source_id": str(sid)} for sid in (1, 2, 3)]
    plan = {
        "classes": [row["category"]],
        "development_per_class": 0,
        "evaluation_per_class": 3,
    }
    counts, next_candidates = false_class_overlay.prior_prefix(
        rows, plan, {}, {"1": {"truth": {}, "manifest": {}}}, {"3": skipped}
    )
    assert counts == {row["category"]: 1}
    assert next_candidates == {row["category"]: "2"}
    with pytest.raises(ValueError, match="support reason changed"):
        false_class_overlay.verify_skip(
            row, {**skipped, "reason": "Frozen parent TPF unavailable in quarters [2]"}
        )
    rule.write_text("changed rule")
    with pytest.raises(ValueError, match="support rule changed"):
        false_class_overlay.verify_skip(row, skipped)


def test_false_class_overlay_omits_unselected_support_skips():
    count = false_class_overlay.unselected_support_count(
        {"1": {}},
        {},
        {"2": {"support_evidence": "frozen"}},
        {},
        {"1"},
        set(),
    )
    assert count == 1
    with pytest.raises(ValueError, match="unused generated outcomes"):
        false_class_overlay.unselected_support_count(
            {"1": {}}, {}, {}, {"2": {"reason": "new skip"}}, {"1"}, set()
        )


def test_false_class_waveforms_and_selected_quarters():
    truth = {
        "category": "eclipsing_binary",
        "period_days": 4.0,
        "epoch_bkjd": 10.0,
        "duration_days": 0.2,
        "depth_fraction": 0.04,
        "secondary_depth_ratio": 0.5,
    }
    depths = false_class_injections.waveform(np.array([10, 12, 11, np.nan]), truth)
    np.testing.assert_allclose(depths, [0.04, 0.02, 0, 0])
    truth["category"] = "instrumental_artifact"
    truth["artifact_kind"] = "stripe_exponential"
    depths = false_class_injections.waveform(np.array([10, 10.2, 11, np.nan]), truth)
    assert depths[0] == truth["depth_fraction"]
    assert 0 < depths[1] < depths[0]
    assert depths[2] == depths[3] == 0
    truth["artifact_kind"] = "column_pulse"
    np.testing.assert_allclose(
        false_class_injections.waveform(np.array([10, 10.2, np.nan]), truth),
        [0.04, 0, 0],
    )

    raw = {
        "time_bkjd": np.arange(4000, dtype=float),
        "quarter": np.repeat([1, 2, 3, 4], 1000),
        injections.PRIMARY_FLUX: np.ones(4000),
    }
    assert false_class_injections.select_quarters(raw) == [1, 2, 3, 4]


def test_false_class_skips_a_companion_without_required_quarter_flux():
    raw = {
        "quarter": np.array([1, 2]),
        injections.PRIMARY_FLUX: np.array([np.nan, 100.0]),
    }
    with pytest.raises(
        false_class_injections.ControlUnavailable,
        match="Contaminant has no usable flux in quarter 1",
    ):
        false_class_injections.required_source_flux(raw, 1, "Contaminant")
    assert false_class_injections.required_source_flux(raw, 2, "Contaminant") == 100


def test_false_class_resume_preserves_unavailable_selection(tmp_path, monkeypatch):
    category = "background_contaminant"
    rows = [
        {"source_id": str(source_id), "category": category} for source_id in (1, 2, 3)
    ]
    plan = {"development_per_class": 1}
    identity = {"candidate_pool": "frozen"}
    progress = false_class_injections.start_materialization(
        tmp_path / "run", identity, False
    )

    def interrupted(out, row, plan, mapping, existing):
        if row["source_id"] == "1":
            raise false_class_injections.ControlUnavailable("No target pixel support")
        raise RuntimeError("Interrupted after unavailable trial")

    monkeypatch.setattr(false_class_injections, "materialize_trial", interrupted)
    with pytest.raises(RuntimeError, match="Interrupted"):
        false_class_injections.collect_class(
            tmp_path / "run", rows, plan, category, 2, {}, {}, progress
        )
    assert (tmp_path / "run/unavailable-records/1.json").exists()

    visited = []

    def resumed(out, row, plan, mapping, existing):
        visited.append(row["source_id"])
        return {
            "truth": {"source_id": row["source_id"]},
            "manifest": {},
        }

    monkeypatch.setattr(false_class_injections, "materialize_trial", resumed)
    progress = false_class_injections.start_materialization(
        tmp_path / "run", identity, True
    )
    accepted, unavailable = false_class_injections.collect_class(
        tmp_path / "run", rows, plan, category, 2, {}, {}, progress
    )
    assert visited == ["2", "3"]
    assert [record["truth"]["source_id"] for record in accepted] == ["2", "3"]
    assert unavailable == [
        {"source_id": "1", "category": category, "reason": "No target pixel support"}
    ]


def test_false_class_seed_preserves_development_trial(tmp_path, monkeypatch):
    category = "background_contaminant"
    rows = [
        {"source_id": source_id, "category": category} for source_id in ("1", "2", "3")
    ]
    plan = {"development_per_class": 1}
    out = tmp_path / "run"
    progress = false_class_injections.start_materialization(
        out, {"seed": "frozen"}, False
    )
    skipped = {"source_id": "1", "category": category, "reason": "Missing pixel"}
    false_class_injections.write(out / "unavailable-records/1.json", skipped)
    seed = {
        "by_category": {category: ["2"]},
        "accepted": {"2": {"source_id": "2", "partition": "development"}},
        "manifest": {"2": {"sha256": "frozen"}},
    }

    def trial(out, row, plan, mapping, existing):
        return {
            "truth": {"source_id": row["source_id"]},
            "manifest": {"sha256": "frozen"},
        }

    monkeypatch.setattr(false_class_injections, "materialize_trial", trial)
    accepted, unavailable = false_class_injections.collect_class(
        out, rows, plan, category, 2, {}, {}, progress, seed
    )
    assert [record["truth"]["source_id"] for record in accepted] == ["2", "3"]
    assert unavailable == [skipped]

    seed["accepted"]["2"]["source_id"] = "changed"
    changed_out = tmp_path / "changed"
    changed_progress = false_class_injections.start_materialization(
        changed_out, {"seed": "frozen"}, False
    )
    false_class_injections.write(changed_out / "unavailable-records/1.json", skipped)
    with pytest.raises(ValueError, match="Frozen development trial changed"):
        false_class_injections.collect_class(
            changed_out, rows, plan, category, 2, {}, {}, changed_progress, seed
        )


def test_standalone_control_keeps_primary_and_qdiag_cadences(tmp_path):
    arrays = tmp_path / "source.npz"
    np.savez_compressed(
        arrays,
        source_id=np.array([42, 42, 42]),
        time_bkjd=np.array([1.0, 2.0, 3.0]),
        qdiag_time_bkjd=np.array([1.0, 3.0]),
        qdiag_psf_flux=np.array([100.0, 99.0]),
    )
    manifest = tmp_path / "input-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "measurement_scope": "Four-quarter controlled excerpt",
                "sources": {
                    "42": {
                        "metadata": {"source_id": "42"},
                        "arrays": arrays.name,
                        "sha256": pixel_injections.digest(arrays),
                    }
                },
            }
        )
    )
    metadata, raw = manifest_source(tmp_path, "42", manifest)
    assert metadata["source_id"] == "42"
    assert len(raw["time_bkjd"]) == 3
    assert len(raw["qdiag_time_bkjd"]) == 2


def test_false_class_excerpt_keeps_only_pixel_controlled_quarters():
    raw = {
        "quarter": np.array([1, 2, 2, 3]),
        "time_bkjd": np.array([1.0, 2.0, 2.5, 3.0]),
        "psf_flat_kbonus_flux": np.array([10.0, 10.0, 10.0, 10.0]),
        "qdiag_quarter": np.array([1, 2, 3]),
        "qdiag_time_bkjd": np.array([1.0, 2.0, 3.0]),
        "qdiag_psf_flux": np.array([20.0, 20.0, 20.0]),
    }
    deltas = {
        "psf_flat_kbonus_flux": np.array([0.0, -1.0, -2.0, 0.0]),
        "qdiag_psf_flux": np.array([0.0, -3.0, 0.0]),
    }
    excerpt = false_class_injections.excerpt_arrays(raw, deltas, [2])
    np.testing.assert_array_equal(excerpt["quarter"], [2, 2])
    np.testing.assert_array_equal(excerpt["psf_flat_kbonus_flux"], [9.0, 8.0])
    np.testing.assert_array_equal(excerpt["qdiag_psf_flux"], [17.0])


def test_false_class_report_requires_the_injected_signal_identity():
    truth = {
        "source_id": "42",
        "parent_tpf": "100",
        "category": "eclipsing_binary",
        "partition": "development",
        "artifact_kind": None,
        "period_days": 4.0,
        "epoch_bkjd": 10.0,
        "duration_days": 0.2,
        "expected_search_mes": 20.0,
    }
    row = {"classification": "EB_LIKE", "search_mes": "20"}
    hypothesis = {
        "signal_id": "signal-1",
        "period_days": 4.0,
        "epoch_bkjd": 10.0,
        "duration_days": 0.2,
        "followup_eligible": False,
        "rust_record": row,
    }
    proposal = {"rust_rows": [row], "supplied_hypotheses": [hypothesis]}
    matched = false_class_control_report.trial(truth, proposal, None, False)
    assert matched["status"] == "DETECTOR_REJECTED"
    assert matched["matches"][0]["harmonic"] == 1

    shallow = {
        **hypothesis,
        "signal_id": "shallow-1",
        "followup_eligible": True,
        "shallow_record": {"search_snr": "12"},
    }
    del shallow["rust_record"]
    shallow_match = false_class_control_report.periodic_matches(
        truth, {"supplied_hypotheses": [shallow]}
    )[0]
    assert shallow_match["detector_origin"] == "shallow_bls_v1"
    assert shallow_match["search_mes"] is None
    assert shallow_match["shallow_search_snr"] == 12

    hypothesis["epoch_bkjd"] = 11.0
    missed = false_class_control_report.trial(truth, proposal, None, False)
    assert missed["status"] == "IDENTITY_NOT_RECOVERED"
    assert missed["matches"] == []


@pytest.fixture
def frozen_false_generator_inputs(tmp_path, monkeypatch):
    report = false_class_control_report
    controls = tmp_path / "controls"
    archive = tmp_path / "generator-archive"
    active = tmp_path / "active"
    expected = {}
    for name in report.false_class_injections.generator_identity():
        original = archive / name
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_text(f"original frozen source: {name}")
        expected[name] = report.catalogue.digest(original)
        current = active / name
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text(f"subsequent implementation: {name}")
    monkeypatch.setattr(report, "ROOT", active)
    pool = tmp_path / "pool"
    monkeypatch.setattr(report.false_class_injections, "POOL", pool)

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    save(pool / "freeze.json", {"source": "frozen pool"})
    seed = tmp_path / "development-seed"
    save(seed / "materialization.json", {"generator": expected})
    save(seed / "parent-audit.json", {"parent_tpf": "100"})
    save(seed / "development-source-ids.json", ["42"])
    controls.mkdir()
    arrays = controls / "arrays.npz"
    np.savez(arrays, quarter=np.array([1]), qdiag_quarter=np.array([1]))
    pixel = controls / "q01.fits.gz"
    hdu = fits.PrimaryHDU()
    hdu.header["KEPLERID"] = 100
    hdu.header["QUARTER"] = 1
    hdu.writeto(pixel)
    manifest = {
        "schema": 1,
        "protocol": report.false_class_injections.MODEL,
        "measurement_scope": "Four observed-quarter excerpts",
        "sources": {
            "42": {
                "arrays": arrays.name,
                "sha256": report.catalogue.digest(arrays),
                "pixels": {
                    "1": {
                        "path": pixel.name,
                        "sha256": report.catalogue.digest(pixel),
                        "base_parent_kic": "100",
                    }
                },
            }
        },
    }
    save(controls / "input-manifest.json", manifest)
    save(controls / "source-ids.json", ["42"])
    save(controls / "development-source-ids.json", ["42"])
    truth = {
        "source_id": "42",
        "parent_tpf": "100",
        "injected_quarters": [1],
        "category": "eclipsing_binary",
        "partition": "development",
        "artifact_kind": None,
        "period_days": 4.0,
        "epoch_bkjd": 10.0,
        "duration_days": 0.2,
        "expected_search_mes": 20.0,
    }
    (controls / "truth.jsonl").write_text(json.dumps(truth) + "\n")
    frozen = {
        "protocol": report.false_class_injections.MODEL,
        "status": "FROZEN",
        "generator": expected,
        "candidate_pool_sha256": report.catalogue.digest(pool / "freeze.json"),
        "development_seed": {
            "path": str(seed / "materialization.json"),
            "sha256": report.catalogue.digest(seed / "materialization.json"),
            "parent_audit_sha256": report.catalogue.digest(seed / "parent-audit.json"),
            "generator": expected,
        },
        "artifacts": {
            name: report.catalogue.digest(controls / name)
            for name in [
                "input-manifest.json",
                "source-ids.json",
                "development-source-ids.json",
                "truth.jsonl",
            ]
        },
    }
    save(controls / "materialization.json", frozen)
    return SimpleNamespace(
        controls=controls, archive=archive, frozen=frozen, pool=pool, arrays=arrays
    )


def test_false_controls_accept_only_the_exact_explicit_frozen_generator(
    frozen_false_generator_inputs,
):
    fixture = frozen_false_generator_inputs
    report = false_class_control_report
    with pytest.raises(ValueError, match="generator-archive"):
        report.validate_controls(fixture.controls)
    frozen, ids, truth = report.validate_controls(fixture.controls, fixture.archive)
    assert frozen == fixture.frozen
    assert ids == ["42"]
    assert truth[0]["parent_tpf"] == "100"
    inputs = report.frozen_generator_inputs(frozen, fixture.archive)
    assert inputs == {
        fixture.archive / name: sha256 for name, sha256 in frozen["generator"].items()
    }


@pytest.mark.parametrize(
    "corruption", ["changed_source", "missing_source", "missing_binding"]
)
def test_frozen_generator_archive_rejects_incomplete_or_changed_identity(
    frozen_false_generator_inputs, corruption
):
    fixture = frozen_false_generator_inputs
    frozen = fixture.frozen
    name = next(iter(frozen["generator"]))
    if corruption == "changed_source":
        (fixture.archive / name).write_text("changed model")
    elif corruption == "missing_source":
        (fixture.archive / name).unlink()
    else:
        frozen = {
            **frozen,
            "generator": {
                key: value for key, value in frozen["generator"].items() if key != name
            },
        }
    with pytest.raises(ValueError, match="Frozen.*generator"):
        false_class_control_report.frozen_generator_inputs(frozen, fixture.archive)


@pytest.mark.parametrize("corruption", ["pool", "arrays"])
def test_generator_archive_preserves_frozen_pool_and_photometry_checks(
    frozen_false_generator_inputs, corruption
):
    fixture = frozen_false_generator_inputs
    path = fixture.pool / "freeze.json" if corruption == "pool" else fixture.arrays
    path.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="candidate pool|LC2 arrays"):
        false_class_control_report.validate_controls(fixture.controls, fixture.archive)


def test_generator_archive_cli_is_explicit_and_reaches_report(tmp_path, monkeypatch):
    calls = []

    def build(*args):
        calls.append(args)
        return {"population": {}, "policy": {}, "categories": {}}

    monkeypatch.setattr(false_class_control_report, "build_report", build)
    monkeypatch.setattr(
        false_class_control_report.pixel_parent_audit, "write_json", lambda *args: None
    )
    archive = tmp_path / "archive"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "false_class_control_report.py",
            "--controls",
            str(tmp_path / "controls"),
            "--run",
            str(tmp_path / "run"),
            "--out",
            str(tmp_path / "report.json"),
            "--partition",
            "development",
            "--generator-archive",
            str(archive),
        ],
    )
    false_class_control_report.main()
    assert calls == [(tmp_path / "controls", tmp_path / "run", "development", archive, False)]


def test_generator_archive_cannot_freeze_probe_controls(frozen_false_generator_inputs):
    fixture = frozen_false_generator_inputs
    probe = {**fixture.frozen, "status": "PROBE"}
    (fixture.controls / "materialization.json").write_text(json.dumps(probe))
    with pytest.raises(ValueError, match="require frozen"):
        false_class_control_report.validate_controls(fixture.controls, fixture.archive)


def test_false_report_binds_archived_generator_and_preserves_execution_boundary(
    frozen_false_generator_inputs, tmp_path
):
    fixture = frozen_false_generator_inputs
    report = false_class_control_report
    run = tmp_path / "run"
    policy_path = tmp_path / "policy.json"
    policy = {
        "policy_id": "fixture-development",
        "status": "DEVELOPMENT",
        "boi_catalogue_enabled": False,
    }
    files = {
        "preparation.json": {"sources": [{"source_id": "42"}]},
        "detection.json": {
            "status": "DETECTION_COMPLETE",
            "sources": 1,
            "alerted_sources": 0,
            "alerts": 0,
        },
        "queue.json": {"rows": []},
        "discovery-manifest.json": {},
        "vetting.json": {"disposition_policy": str(policy_path)},
        "vetting/manifest.json": {},
        "vetting/stage-timings.json": {},
        "vetting/results.json": {
            "processing_status": "EVIDENCE_STAGES_COMPLETE",
            "sources": 0,
            "original_alerts_preserved": 0,
            "disposition_execution": report.catalogue.disposition_execution(policy),
        },
        "proposals/42.json": {
            "source_id": "42",
            "alerts": [],
            "rust_rows": [],
            "supplied_hypotheses": [],
        },
    }
    policy_path.write_text(json.dumps(policy))
    for name, value in files.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    actual = report.build_report(fixture.controls, run, "development", fixture.archive)
    expected = {
        report.pixel_parent_audit.display_path(fixture.archive / name): sha256
        for name, sha256 in fixture.frozen["generator"].items()
    }
    assert actual["frozen_generator_inputs"] == expected
    assert all(actual["inputs"][path] == sha256 for path, sha256 in expected.items())
    assert actual["policy"]["boi_catalogue_enabled"] is False
    assert (
        actual["categories"]["eclipsing_binary"]["false_boi_rate"]["status"]
        == "NOT_MEASURED_BOI_CATALOGUE_DISABLED"
    )
    assert actual["trials"][0]["status"] == "IDENTITY_NOT_RECOVERED"
    del files["vetting/results.json"]["disposition_execution"]
    (run / "vetting/results.json").write_text(json.dumps(files["vetting/results.json"]))
    with pytest.raises(TypeError, match="no disposition execution record"):
        report.build_report(fixture.controls, run, "development", fixture.archive)


@pytest.mark.parametrize("use_archive", [False, True])
def test_false_class_evaluation_does_not_open_unfrozen_outcomes(
    tmp_path, monkeypatch, use_archive
):
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"status": "DEVELOPMENT", "boi_catalogue_enabled": False})
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "vetting.json").write_text(json.dumps({"disposition_policy": str(policy)}))
    monkeypatch.setattr(
        false_class_control_report,
        "validate_controls",
        lambda *args: ({"status": "FROZEN"}, ["42"], [{"source_id": "42"}]),
    )
    monkeypatch.setattr(
        false_class_control_report.catalogue,
        "validate_run",
        lambda *_: (_ for _ in ()).throw(AssertionError("Opened sealed run")),
    )
    with pytest.raises(ValueError, match="requires frozen controls and policy"):
        false_class_control_report.build_report(
            tmp_path,
            run,
            "evaluation",
            tmp_path / "frozen-generator" if use_archive else None,
        )


@pytest.mark.parametrize(
    "alias_disposition", ["UNRESOLVED", "BOI", "FALSE_ALARM"]
)
def test_false_class_parent_accounts_for_nonmatching_alias(alias_disposition):
    truth = {
        "source_id": "42",
        "parent_tpf": "100",
        "category": "eclipsing_binary",
        "partition": "development",
        "artifact_kind": None,
        "period_days": 4.0,
        "epoch_bkjd": 10.0,
        "duration_days": 0.2,
        "expected_search_mes": 20.0,
    }
    rust = {"classification": "TRANSIT_LIKE", "search_mes": "20"}
    proposal = {
        "rust_rows": [rust],
        "supplied_hypotheses": [
            {
                "signal_id": "injected",
                "period_days": 4.0,
                "epoch_bkjd": 10.0,
                "duration_days": 0.2,
                "followup_eligible": True,
                "rust_record": rust,
            }
        ],
    }
    source = {
        "processing_status": "EVIDENCE_STAGES_COMPLETE",
        "events": [
            {
                "event_id": name,
                "original_alert": {
                    "detector_signal_id": name,
                    "kind": "PERIODIC_WINDOW",
                },
            }
            for name in ("injected", "other-alias")
        ],
        "signals": [
            {
                "signal_id": name,
                "event_ids": [name],
                "triage": {
                    "status": "DISMISSED",
                    "dismissed_stage": "shape",
                    "reason": "NON_TRANSIT_MODEL_STRONGLY_PREFERRED",
                },
            }
            for name in ("injected", "other-alias")
        ],
        "disposition": {
            "status": "FALSE_ALARM"
            if alias_disposition == "FALSE_ALARM"
            else "UNRESOLVED",
            "boi_signal_count": None
            if alias_disposition == "BOI"
            else 0,
            "signals": [
                {"signal_id": "injected", "disposition": "FALSE_ALARM"},
                {"signal_id": "other-alias", "disposition": alias_disposition},
            ],
        },
    }
    result = false_class_control_report.trial(truth, proposal, source, True)
    expected = {
        "UNRESOLVED": "SCIENTIFIC_ABSTENTION",
        "BOI": "AUTOMATIC_PC",
        "FALSE_ALARM": "VETTING_DISMISSED",
    }
    assert result["status"] == expected[alias_disposition]
    assert len(result["matches"]) == 1
    assert len(result["signals"]) == 2
    assert result["scientific_abstention"] is (alias_disposition == "UNRESOLVED")
    assert result["disposition_incomplete"] is (alias_disposition == "BOI")
    assert result["boi_catalogue_on_parent"] is (alias_disposition == "BOI")


def write_pixel_materialization(path, sources, schema=2):
    path.mkdir()
    manifest = {"schema": schema, "sources": {}}
    for source_id, parents in sources.items():
        manifest["sources"][source_id] = {
            "pixels": {
                str(quarter): {"base_parent_kic": parent}
                for quarter, parent in enumerate(parents, 1)
            }
        }
    manifest_path = path / "input-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    report = {
        "status": "FROZEN",
        "sources": len(sources),
        "artifacts": {"input-manifest.json": pixel_parent_audit.digest(manifest_path)},
    }
    (path / "materialization.json").write_text(json.dumps(report))


def test_hash_draws_and_periodic_schedule_are_reproducible():
    plan = injections.read(injections.PLAN)
    first = injections.unit(plan["seed"], "source", "period")
    second = injections.unit(plan["seed"], "source", "period")
    assert first == second
    schedule = injections.periodic_schedule(
        plan["cohorts"]["periodic_injection"]["strata"], plan["seed"]
    )
    assert len(schedule) == plan["cohorts"]["periodic_injection"]["target"]
    assert schedule.count("ultra_short") == 500
    assert schedule.count("core") == 3500
    assert schedule.count("long_period") == 1000

    probe = injections.scaled_strata(
        plan["cohorts"]["periodic_injection"]["strata"], 10
    )
    assert sum(item["count"] for item in probe) == 10
    assert [item["count"] for item in probe] == [1, 7, 2]


def test_physical_transit_model_has_finite_supported_shape():
    plan = injections.read(injections.PLAN)
    radius_ratio = 0.03
    scaled_axis, inclination, duration = injections.orbit(12.0, 1.0, 0.4, radius_ratio)
    times = np.linspace(199.5, 200.5, 1000)
    loss = injections.transit_loss(
        times,
        [200.0],
        12.0,
        (radius_ratio, scaled_axis, inclination, duration),
        plan["signal"],
    )
    assert np.isfinite(loss).all()
    assert 0 < loss.max() < 0.01
    assert plan["signal"]["duration_hours"][0] < duration * 24 < 24
    assert math.isclose(loss[0], 0.0)


def test_full_materialization_stops_at_first_sufficient_pool_prefix(
    tmp_path, monkeypatch
):
    plan = {"cohorts": {"ordinary": {"target": 2}}}
    work = [
        {
            "category": "ordinary",
            "pool_rank": rank,
            "source_id": str(rank),
            "shard": rank,
            "stratum": "ordinary",
        }
        for rank in range(1, 151)
    ]
    calls = []

    def evaluate(out, scheduled, plan, progress, results):
        calls.append([item["pool_rank"] for item in scheduled])
        for item in scheduled:
            results[("ordinary", item["pool_rank"])] = {
                **item,
                "status": "INELIGIBLE" if item["pool_rank"] == 2 else "ELIGIBLE",
                "truth": None,
            }
        return results

    monkeypatch.setattr(injections, "evaluate_work", evaluate)
    _, accepted = injections.evaluate_until_selected(
        tmp_path, work, plan, ["ordinary"], None, {}
    )
    assert calls == [list(range(1, 101))]
    assert [row["pool_rank"] for row in accepted["ordinary"]] == [1, 3]


def test_pixel_injection_uses_quarter_source_flux_and_preserves_nan_pixels():
    raw = {
        "quarter": np.array([3, 3, 4]),
        injections.PRIMARY_FLUX: np.array([90.0, 110.0, 500.0]),
    }
    assert pixel_injections.source_flux(raw, 3) == 100.0
    flux = np.array([[[10.0, np.nan]], [[12.0, 8.0]]], dtype=np.float32)
    profile = np.array([[0.75, 0.25]])
    loss = np.array([0.0, 0.1])
    injected = pixel_injections.inject_flux(flux, profile, loss, 100.0)
    assert injected.dtype == flux.dtype
    assert np.array_equal(injected[0], flux[0], equal_nan=True)
    assert np.allclose(injected[1], [[[4.5, 5.5]]])


def test_pixel_fits_serialization_is_byte_reproducible(tmp_path):
    first = tmp_path / "first.fits.gz"
    second = tmp_path / "second.fits.gz"

    def image():
        return fits.HDUList([fits.PrimaryHDU(np.arange(6).reshape(2, 3))])

    pixel_injections.deterministic_fits(first, image())
    pixel_injections.deterministic_fits(second, image())
    assert pixel_injections.digest(first) == pixel_injections.digest(second)
    with fits.open(first, checksum=True) as hdus:
        hdus.verify("exception")
        assert hdus[0].header.comments["CHECKSUM"] == (
            "Forgotten Stars deterministic checksum"
        )


def test_gaia_pixel_query_has_stable_numeric_identity_order():
    query = pixel_injections.gaia_query(["20", "3", "11"])
    assert query.endswith("where source_id in (3,11,20)")


def test_pixel_partition_rows_filters_morphology(tmp_path):
    rows = [
        {"source_id": "1", "category": "sparse_injection"},
        {"source_id": "2", "category": "ttv_injection"},
        {"source_id": "3", "category": "ttv_injection"},
    ]
    path = tmp_path / "development-truth.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert pixel_injections.partition_rows(
        tmp_path, "development", category="ttv_injection", limit=1
    ) == [rows[1]]


@pytest.mark.parametrize(
    "category", ["periodic_injection", "sparse_injection", "ttv_injection"]
)
def test_pixel_transits_preserve_flux_injection_timing(category):
    period, epoch = 10.0, 100.0
    axis, inclination, duration = injections.orbit(period, 1.0, 0.3, 0.1)
    truth = {
        "category": category,
        "epoch_bkjd": epoch,
        "shape_period_days": period,
        "radius_ratio": 0.1,
        "scaled_semimajor_axis": axis,
        "inclination_degrees": inclination,
        "duration_hours": duration * 24,
        "ttv": {
            "amplitude_days": duration * 2,
            "superperiod_cycles": 5,
            "phase_radians": math.pi / 2,
        },
    }
    times = np.linspace(75, 125, 10001)
    flux = np.full(len(times), 10000.0)
    raw = {
        "time_bkjd": times,
        "qdiag_time_bkjd": times,
        "quarter": np.ones(len(times)),
        "qdiag_quarter": np.ones(len(times)),
        **{field: flux for _, _, fields in injections.DELTA_GROUPS for field in fields},
    }
    plan = injections.read(injections.PLAN)
    deltas = injections.delta_arrays(raw, truth, plan)
    loss = pixel_injections.transit_loss(times, truth, plan["signal"])
    for field, delta in deltas.items():
        np.testing.assert_allclose(
            (-flux * loss).astype(np.float32), delta, rtol=0, atol=0, err_msg=field
        )
    central = np.argmin(abs(times - epoch))
    previous = np.argmin(abs(times - (epoch - period)))
    if category == "sparse_injection":
        assert loss[central] > 0 and loss[previous] == 0
    elif category == "ttv_injection":
        shifted = np.argmin(abs(times - (epoch + truth["ttv"]["amplitude_days"])))
        assert loss[central] == 0 and loss[shifted] > 0
    else:
        assert loss[central] > 0 and loss[previous] > 0


@pytest.mark.parametrize("category", ["sparse_injection", "ttv_injection"])
@pytest.mark.parametrize("wrong_timing", [False, True])
@pytest.mark.parametrize(
    "position,supported",
    [
        ([0.5, 0.5], True),
        ([-2.0, 0.0], True),
        ([-1.5, -1.5], False),
        ([-3.65, -1.29], False),
        ([3.0, 1.0], True),
        ([3.01, 1.0], False),
    ],
)
def test_transfer_audit_detects_periodic_pixels_for_morphology(
    tmp_path, monkeypatch, category, wrong_timing, position, supported
):
    period, epoch = 10.0, 100.0
    axis, inclination, duration = injections.orbit(period, 1.0, 0.3, 0.1)
    truth = {
        "category": category,
        "epoch_bkjd": epoch,
        "shape_period_days": period,
        "radius_ratio": 0.1,
        "scaled_semimajor_axis": axis,
        "inclination_degrees": inclination,
        "duration_hours": duration * 24,
        "ttv": {
            "amplitude_days": duration * 2,
            "superperiod_cycles": 5,
            "phase_radians": math.pi / 2,
        },
    }
    signal = injections.read(injections.PLAN)["signal"]
    times = np.array([epoch - period, epoch, epoch + duration * 2, epoch + period])
    profile = np.full((2, 2), 0.25)
    flux = np.full((len(times), 2, 2), 1000.0, dtype=np.float32)
    header = fits.Header({"QUARTER": 0, "KEPLERID": 42, "CHANNEL": 1})
    columns = [
        fits.Column(name="TIME", format="D", array=times),
        fits.Column(name="FLUX", format="4E", dim="(2,2)", array=flux),
        fits.Column(
            name="FLUX_ERR", format="4E", dim="(2,2)", array=np.ones_like(flux)
        ),
    ]
    hdus = fits.HDUList(
        [fits.PrimaryHDU(header=header), fits.BinTableHDU.from_columns(columns)]
    )
    base = tmp_path / "base.fits.gz"
    pixel_injections.deterministic_fits(base, hdus)
    injected_truth = (
        {**truth, "category": "periodic_injection"} if wrong_timing else truth
    )
    loss = pixel_injections.transit_loss(times, injected_truth, signal)
    hdus[1].data["FLUX"] = pixel_injections.inject_flux(flux, profile, loss, 10000)
    hdus[0].header["FSCSRC"] = "123"
    hdus[0].header["FSCMODEL"] = "physical-prf-v1"
    controlled = tmp_path / "controlled.fits.gz"
    pixel_injections.deterministic_fits(controlled, hdus)
    calibration_path = pixel_transfer_audit.kepler_prf_path(1, tmp_path)
    calibration_path.write_bytes(b"test calibration")
    calibration = {
        "filename": calibration_path.name,
        "sha256": pixel_transfer_audit.digest(calibration_path),
    }
    monkeypatch.setattr(
        pixel_injections,
        "pixel_profile",
        lambda *_: (profile, calibration, position, [1.0, 2.0], "PROPAGATED", True),
    )
    pixel = {
        "path": controlled.name,
        "sha256": pixel_transfer_audit.digest(controlled),
        "base_path": str(base),
        "base_sha256": pixel_transfer_audit.digest(base),
        "base_parent_kic": 42,
        "prf_calibration": calibration,
        "target_astrometry": "PROPAGATED",
        "target_inside_stamp": True,
        "target_pixel": position,
        "target_sky_deg": [1.0, 2.0],
        "prf_stamp_fraction": 1.0,
        "source_flux_electrons_per_second": 10000,
        "maximum_relative_loss": float(loss.max()),
    }
    raw = {
        injections.PRIMARY_FLUX: np.full(len(times), 10000.0),
        "quarter": np.zeros(len(times)),
    }
    result = pixel_transfer_audit.verify_pixel(
        tmp_path, "123", truth, ("0", pixel), raw, {}, signal, tmp_path, {}
    )
    assert result["transfer_matches_declared_truth"] is not wrong_timing
    assert bool(result["mismatched_pixel_values"]) is wrong_timing
    assert result["structural_support"] is supported


def test_transfer_audit_rejects_changed_uncertainties():
    column = fits.Column(name="FLUX_ERR", format="E", array=np.ones(4))
    base = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([column])])
    controlled = fits.HDUList([hdu.copy() for hdu in base])
    controlled[1].data["FLUX_ERR"][0] = 2
    with pytest.raises(ValueError, match="Non-flux FITS column changed: FLUX_ERR"):
        pixel_transfer_audit.verify_unchanged(base, controlled)


@pytest.mark.parametrize("incomplete", [False, True])
@pytest.mark.parametrize("unsupported", [False, True])
def test_transfer_audit_requires_complete_category_scope(
    tmp_path, monkeypatch, incomplete, unsupported
):
    partitions = tmp_path / "partitions"
    controls = tmp_path / "controls"
    partitions.mkdir()
    controls.mkdir()
    frozen_truth = [
        {"source_id": "1", "category": "sparse_injection"},
        {"source_id": "2", "category": "sparse_injection"},
        {"source_id": "3", "category": "ttv_injection"},
    ]
    (partitions / "freeze.json").write_text("{}")
    (partitions / "development-truth.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in frozen_truth)
    )
    selected = frozen_truth[: 1 if incomplete else 2]
    (controls / "truth.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in selected)
    )
    (controls / "input-manifest.json").write_text(
        json.dumps({"sources": {row["source_id"]: {} for row in selected}})
    )
    (controls / "gaia-astrometry.json").write_text(json.dumps({"rows": selected}))
    report = {
        "status": "FROZEN",
        "sources": len(selected),
        "partition": "development",
        "category": "sparse_injection",
        "source_ids": [row["source_id"] for row in selected],
        "partition_freeze_sha256": pixel_transfer_audit.digest(
            partitions / "freeze.json"
        ),
        "artifacts": {
            path.name: pixel_transfer_audit.digest(path) for path in controls.iterdir()
        },
    }
    (controls / "materialization.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        pixel_transfer_audit,
        "verify_source",
        lambda folder, sid, truth, *args: {
            "source_id": sid,
            "category": truth["category"],
            "status": "PHYSICAL_INPUT_FAILED" if unsupported else "VERIFIED",
            "quarters": [
                {
                    "transfer_matches_declared_truth": True,
                    "structural_support": not unsupported,
                }
            ],
        },
    )
    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--materialization",
            str(controls),
            "--partitions",
            str(partitions),
            "--prf-cache",
            str(tmp_path),
            "--out",
            str(output),
        ],
    )
    if incomplete:
        with pytest.raises(
            ValueError, match="Physical truth or ordered input scope differs"
        ):
            pixel_transfer_audit.main()
        assert not (output / "audit.json").exists()
    else:
        if unsupported:
            with pytest.raises(SystemExit) as error:
                pixel_transfer_audit.main()
            assert error.value.code == 1
        else:
            pixel_transfer_audit.main()
        summary = pixel_transfer_audit.read(output / "audit.json")
        assert summary["sources"] == 2
        assert summary["unsupported_pixel_products"] == 2 * unsupported
        assert len(pixel_transfer_audit.rows(output / "trials.jsonl")) == 2


def test_pixel_support_absence_is_a_structural_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pixel_injections,
        "mast_spatial_tpf_query",
        lambda *_: {"products": []},
    )
    try:
        pixel_injections.spatial_tpfs(tmp_path / "query.json", {}, {}, {5}, {})
    except pixel_injections.PixelUnavailable as error:
        assert str(error) == "MAST spatial query lacks quarters [5]"
    else:
        raise AssertionError("Missing target pixels did not fail structurally")


def test_pixel_support_unavailable_record_is_resumable(tmp_path):
    row = {
        "source_id": "1",
        "category": "sparse_injection",
        "observed_quarters": [5],
    }
    record = {
        **row,
        "status": "PIXELS_UNAVAILABLE",
        "reason": "No spatial TPF supports source in quarters [5]",
        "spatial_tpf_query": None,
    }
    assert pixel_support.validate_record(record, tmp_path, row)


def test_resolved_pixel_parent_audit_reports_every_overlap(tmp_path, monkeypatch):
    current = tmp_path / "current"
    reference = tmp_path / "reference"
    write_pixel_materialization(
        current,
        {"1": [10, 20], "2": [20], "3": [30]},
    )
    write_pixel_materialization(reference, {"4": [10]})
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(
        pixel_parent_audit.planet_control_partitions,
        "excluded_parents",
        lambda value: ({"30"}, [], []),
    )
    report = pixel_parent_audit.audit(current, plan, [reference])
    assert report["status"] == "PARENT_OVERLAP"
    assert report["checks"]["within_materialization"] == {"20": ["1", "2"]}
    assert report["checks"]["excluded_control_parents"] == {"30": ["3"]}
    assert report["checks"]["reference_materializations"][0]["parent_overlap"] == {
        "10": ["1"]
    }


def test_resolved_pixel_parent_audit_accepts_independent_sources(tmp_path, monkeypatch):
    current = tmp_path / "current"
    write_pixel_materialization(current, {"1": [10], "2": [20]}, schema=1)
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(
        pixel_parent_audit.planet_control_partitions,
        "excluded_parents",
        lambda value: (set(), [], []),
    )
    report = pixel_parent_audit.audit(current, plan)
    assert report["status"] == "PARENT_INDEPENDENT"
    assert report["distinct_resolved_parents"] == 2


def test_gaia_query_retries_transient_http_failure(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            if len(calls) == 1:
                raise pixel_injections.requests.HTTPError("transient")

        def json(self):
            return {"metadata": [], "data": []}

    def get(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(pixel_injections.requests, "get", get)
    assert pixel_injections.query_gaia("select 1") == {"metadata": [], "data": []}
    assert len(calls) == 2


def test_gaia_astrometry_canonicalizes_remote_row_order(tmp_path, monkeypatch):
    payload = {
        "metadata": [{"name": name} for name in pixel_injections.GAIA_COLUMNS],
        "data": [
            [20, 2.0, 3.0, None, None],
            [3, 4.0, 5.0, None, None],
        ],
    }
    monkeypatch.setattr(pixel_injections, "query_gaia", lambda _: payload)
    path = tmp_path / "astrometry.json"
    rows = pixel_injections.gaia_astrometry(path, ["20", "3"])
    assert list(rows) == ["3", "20"]
    assert [row["source_id"] for row in pixel_injections.read(path)["rows"]] == [
        3,
        20,
    ]


def test_archive_download_plan_is_quarter_ordered_and_unique():
    filenames = [
        "kplr000000001-300_lpd-targ.fits.gz",
        "kplr000000001-100_lpd-targ.fits.gz",
        "kplr000000001-200_lpd-targ.fits.gz",
    ]
    plan = pixel_injections.archive_download_plan(
        filenames, {3, 7}, {"100": 3, "200": 7, "300": 7}
    )
    assert plan == [
        (3, "kplr000000001-100_lpd-targ.fits.gz"),
        (7, "kplr000000001-300_lpd-targ.fits.gz"),
    ]


def test_mast_spatial_query_reuses_matching_cache(tmp_path):
    source = {"ra_deg": 10.0, "dec_deg": 20.0}
    result = {
        "query": {
            "coordinates_deg": [10.0, 20.0],
            "radius_deg": 1 / 60,
            "obs_collection": "Kepler",
            "dataproduct_type": "timeseries",
        },
        "products": [{"filename": "pixel.fits.gz"}],
    }
    path = tmp_path / "query.json"
    pixel_injections.write_json(path, result)
    assert pixel_injections.mast_spatial_tpf_query(path, source, {}) == result


def test_mast_spatial_query_empty_observations_are_unavailable(tmp_path, monkeypatch):
    from astroquery.mast import Observations

    monkeypatch.setattr(Observations, "query_criteria", lambda **_: [])
    monkeypatch.setattr(
        Observations,
        "get_product_list",
        lambda _: pytest.fail("No product query should follow empty observations"),
    )
    path = tmp_path / "query.json"
    with pytest.raises(pixel_injections.PixelUnavailable, match="no Kepler"):
        pixel_injections.mast_spatial_tpf_query(
            path, {"ra_deg": 10.0, "dec_deg": 20.0}, {}
        )
    assert not path.exists()


@pytest.mark.parametrize("timeout", [None, 2, (1, 2)])
def test_mast_transport_bounds_missing_timeout_and_preserves_explicit_timeout(
    monkeypatch, timeout
):
    received = []

    def send(adapter, request, **kwargs):
        received.append(kwargs["timeout"])
        return object()

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    adapter = pixel_injections.MastHTTPAdapter()
    adapter.send(object(), timeout=timeout)
    assert received == [(10, 60) if timeout is None else timeout]


def test_mast_spatial_query_bounds_http_reads_and_retains_failed_queries(
    tmp_path, monkeypatch
):
    from astroquery.mast import Observations

    connection = Observations._portal_api_connection
    monkeypatch.setattr(connection, "_session", requests.Session())
    timeouts = []

    def send(adapter, request, **kwargs):
        timeouts.append(kwargs["timeout"])
        raise requests.ReadTimeout("The MAST socket read exceeded its timeout")

    def query(**kwargs):
        return connection._request("POST", connection.MAST_REQUEST_URL, data="{}")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    monkeypatch.setattr(Observations, "query_criteria", query)
    path = tmp_path / "query.json"
    with pytest.raises(OSError, match="socket read exceeded"):
        pixel_injections.mast_spatial_tpf_query(
            path, {"ra_deg": 10.0, "dec_deg": 20.0}, {}, attempts=2
        )
    assert timeouts == [(10, 60), (10, 60)]
    assert not path.exists()


def test_spatial_tpf_selection_rejects_unsupported_nearest_stamp(tmp_path, monkeypatch):
    products = [
        {
            "quarter": 5,
            "parent_kic": parent,
            "filename": f"kplr{parent:09d}-100_lpd-targ.fits.gz",
        }
        for parent in (1, 2)
    ]
    for product in products:
        path = tmp_path / product["filename"]
        fits.HDUList([fits.PrimaryHDU(header=fits.Header({"QUARTER": 5}))]).writeto(
            path
        )

    monkeypatch.setattr(
        pixel_injections,
        "mast_spatial_tpf_query",
        lambda path, source, mapping: {"products": products},
    )
    monkeypatch.setattr(
        pixel_injections,
        "download_pixel",
        lambda base, filename, parent: tmp_path / filename,
    )
    monkeypatch.setattr(
        pixel_injections,
        "stamp_distance",
        lambda path, astrometry: 3.0 if "000000001" in path.name else 0.0,
    )
    selected = pixel_injections.spatial_tpfs(tmp_path / "query.json", {}, {}, {5}, {})
    assert selected == {5: tmp_path / products[1]["filename"]}


def test_available_tpfs_replaces_unsupported_origin_stamp(tmp_path, monkeypatch):
    origin = tmp_path / "origin.fits.gz"
    spatial = tmp_path / "spatial.fits.gz"
    monkeypatch.setattr(pixel_injections, "stamp_distance", lambda *_: 3.0)
    monkeypatch.setattr(pixel_injections, "archive_filenames", lambda *_: [])
    monkeypatch.setattr(
        pixel_injections,
        "spatial_tpfs",
        lambda *args: {5: spatial},
    )
    selected = pixel_injections.available_tpfs(
        {"origin_tpf_kic": 1},
        {},
        {5},
        {},
        {(1, 5): origin},
        tmp_path / "query.json",
    )
    assert selected == {5: spatial}


def test_localization_accepts_declared_spatial_parent(tmp_path, monkeypatch):
    pixels = importlib.import_module("vetting_pixels")
    monkeypatch.setitem(
        sys.modules, "incinerator.localize", SimpleNamespace(Localize=object)
    )
    path = tmp_path / "pixel.fits"
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TIME", format="D", array=np.array([1.0, 2.0])),
            fits.Column(name="QUALITY", format="J", array=np.array([0, 0])),
        ]
    )
    fits.HDUList(
        [fits.PrimaryHDU(header=fits.Header({"KEPLERID": 2, "QUARTER": 5})), table]
    ).writeto(path)
    monkeypatch.setattr(
        pixels, "query_evidence", lambda source_id, kind: {"matches": []}
    )
    source = {"source_id": "3", "origin_tpf_kic": 1}
    event = {"quarter": 5, "centre_bkjd": 10.0, "duration_days": 0.1}
    assert pixels.localize_event(path, source, event, 1)["status"] == (
        "PIXEL_IDENTITY_OR_QUARTER_MISMATCH"
    )
    assert pixels.localize_event(path, source, event, 2)["status"] == (
        "INSUFFICIENT_PIXEL_TIME_COVERAGE"
    )


def test_prf_prefetch_deduplicates_channels(tmp_path, monkeypatch):
    calls = []

    def ensure(channel, cache):
        calls.append((channel, cache))

    monkeypatch.setattr(pixel_injections, "ensure_kepler_prf", ensure)
    pixel_injections.prefetch_prfs([9, 5, 9], tmp_path)
    assert sorted(calls) == [(5, tmp_path), (9, tmp_path)]


def test_pixel_profile_preserves_integer_detector_origin(monkeypatch):
    class Wcs:
        def world_to_pixel_values(self, ra, dec):
            return -0.2, 2.5

    class Prf:
        def evaluate(self, targets, origin, shape):
            assert all(isinstance(value, int) for value in origin)
            return np.ones((1, *shape))

    monkeypatch.setattr(pixel_injections, "WCS", lambda header: Wcs())
    monkeypatch.setattr(
        pixel_injections,
        "propagated_position",
        lambda astrometry, epoch: (10.0, 20.0, "PROPAGATED_GAIA_POSITION"),
    )
    monkeypatch.setattr(
        pixel_injections,
        "load_kepler_prf",
        lambda channel, cache: (Prf(), {"sha256": "calibration"}),
    )
    hdus = [
        SimpleNamespace(header={"CHANNEL": 1}),
        SimpleNamespace(
            header={"1CRV4P": 1000, "2CRV4P": 200},
            data={
                "TIME": np.array([1.0, 2.0]),
                "FLUX": np.ones((2, 5, 4)),
            },
        ),
        SimpleNamespace(header={}),
    ]
    profile, _, pixel, _, _, inside = pixel_injections.pixel_profile(hdus, {})
    assert profile.shape == (5, 4)
    assert pixel == [-0.2, 2.5]
    assert inside is False
