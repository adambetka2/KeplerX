"""Parent-audit identities remain valid across reporting checkouts."""

import json

import pytest

from pipeline.deterministic import pixel_parent_audit, planet_control_report


@pytest.fixture
def audited_controls(tmp_path, monkeypatch):
    producer = tmp_path / "producer"
    controls = producer / "dump/pixels"
    controls.mkdir(parents=True)
    manifest = {
        "schema": 2,
        "sources": {
            "1001": {"pixels": {"1": {"base_parent_kic": 41}}},
            "1002": {"pixels": {"1": {"base_parent_kic": 42}}},
        },
    }
    (controls / "input-manifest.json").write_text(json.dumps(manifest))
    (controls / "source-ids.json").write_text(json.dumps(["1001", "1002"]))
    (controls / "truth.jsonl").write_text(
        "".join(
            json.dumps({"source_id": sid, "category": "periodic_injection"}) + "\n"
            for sid in manifest["sources"]
        )
    )
    report = {
        "status": "FROZEN",
        "protocol": "fsc-kepler-pixel-planet-controls-v1",
        "sources": 2,
        "artifacts": {
            "input-manifest.json": pixel_parent_audit.digest(
                controls / "input-manifest.json"
            )
        },
    }
    (controls / "materialization.json").write_text(json.dumps(report))
    plan = producer / "plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(pixel_parent_audit, "ROOT", producer)
    monkeypatch.setattr(
        pixel_parent_audit.planet_control_partitions,
        "excluded_parents",
        lambda _: (set(), set(), []),
    )
    audit = pixel_parent_audit.audit(controls, plan)
    path = controls / "parent-audit.json"
    path.write_text(json.dumps(audit))
    consumer = tmp_path / "consumer"
    consumer.mkdir()
    monkeypatch.setattr(pixel_parent_audit, "ROOT", consumer)
    monkeypatch.setattr(planet_control_report, "ROOT", consumer)
    return controls, path, audit


def test_parent_audit_from_another_checkout_preserves_identity(audited_controls):
    controls, path, expected = audited_controls
    _, ids, _, _, actual = planet_control_report.validate_controls(controls, path)
    assert ids == ["1001", "1002"]
    assert actual == expected
    assert actual["schema"] == 2
    assert actual["repository_root"] != str(planet_control_report.ROOT)
    assert actual["materialization"] == "dump/pixels"


@pytest.mark.parametrize("schema,root", [(1, None), (2, None), (2, "relative/root")])
def test_parent_audit_requires_an_explicit_absolute_root(
    audited_controls, schema, root
):
    controls, path, audit = audited_controls
    audit["schema"] = schema
    audit.pop("repository_root")
    if root is not None:
        audit["repository_root"] = root
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="schema 2.*absolute repository root"):
        planet_control_report.validate_controls(controls, path)


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "parents",
        "repository",
        "materialization",
        "report_hash",
        "manifest_hash",
    ],
)
def test_cross_checkout_audit_does_not_relax_identity_checks(audited_controls, change):
    controls, path, audit = audited_controls
    if change == "status":
        audit["status"] = "PARENT_OVERLAP"
    elif change == "parents":
        audit["source_parents"]["1001"] = ["99"]
    elif change == "repository":
        audit["repository_root"] = str(planet_control_report.ROOT)
    elif change == "materialization":
        audit["materialization"] = "dump/other-controls"
    elif change == "report_hash":
        audit["inputs"]["dump/pixels/materialization.json"] = "0" * 64
    elif change == "manifest_hash":
        audit["inputs"]["dump/pixels/input-manifest.json"] = "0" * 64
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="absent, stale, or failed"):
        planet_control_report.validate_controls(controls, path)


def test_duplicate_resolved_parent_audit_paths_are_rejected(audited_controls):
    controls, path, audit = audited_controls
    audit["inputs"][str(controls / "materialization.json")] = audit["inputs"][
        "dump/pixels/materialization.json"
    ]
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="duplicate resolved input paths"):
        planet_control_report.validate_controls(controls, path)
