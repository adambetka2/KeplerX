"""Outcome-blind selection protects whole components and preserves exact quotas."""

import copy
import csv
import hashlib
import io
import json
import tarfile

import pytest

from pipeline.deterministic import cloud_null as pool
from pipeline.deterministic import null_membership as nulls


def sha(value):
    return hashlib.sha256(value).hexdigest()


def row(source, component=None):
    component = component or source
    return {
        "source_id": str(source),
        "shard": "0",
        "origin_parent": str(1000 + component),
        "raw_fits_sha256": sha(str(source).encode()),
        "metadata_dependency_component_id": sha(str(component).encode()),
        "linked_existing_scopes": "",
        "linked_full_witness_component_ids": "",
        **pool.EXPOSURE,
    }


@pytest.fixture
def protected():
    return {key: set() for key in (*nulls.SET_FIELDS, "alias_parents")}


@pytest.fixture
def rows():
    return [row(1, 2), row(2), *[row(source) for source in range(3, 12)]]


def test_fixed_seed_algorithm_preserves_rows_and_alternates_roles(rows, protected):
    original = copy.deepcopy(rows)
    selected, decisions = nulls.select(rows, protected, quota=3)

    def independent_hash(item):
        data = json.dumps(
            [20261002, item["metadata_dependency_component_id"], item["source_id"]], separators=(",", ":")
        ).encode()
        return sha(data)

    representatives = [min(rows[:2], key=independent_hash), *rows[2:]]
    expected = sorted(representatives, key=independent_hash)[:6]
    assert [r["source_id"] for r in selected] == [r["source_id"] for r in expected]
    assert [r["role"] for r in selected] == ["development", "evaluation"] * 3
    assert [r["role_position"] for r in selected] == [0, 0, 1, 1, 2, 2]
    assert len({r["metadata_dependency_component_id"] for r in selected}) == 6
    assert rows == original
    assert nulls.SELECTION_SEED == nulls.TRANSFORM_SEED == 20261002
    assert nulls.select(list(reversed(rows)), protected, quota=3) == (selected, decisions)


@pytest.mark.parametrize(
    "scope",
    [
        "CURRENT_OPENED_DEVELOPMENT",
        "CURRENT_CORE_FINAL",
        "DEFERRED_FINAL",
        "HOLDOUT_VETTING_DISPOSITION_ONLY",
        "DROPPED_CLASS_IDENTITY_PRESERVED",
    ],
)
def test_every_existing_role_excludes_whole_linked_component(rows, protected, scope):
    rows[1]["linked_existing_scopes"] = scope
    selected, decisions = nulls.select(rows, protected, quota=2)
    assert not {"1", "2"} & {r["source_id"] for r in selected}
    decision = next(r for r in decisions if "1" in r["source_ids"])
    assert decision["representative"] is None
    assert decision["exclusions"] == [
        {"reason": "linked_existing_scopes", "source_id": "2", "identity": scope}
    ]


def test_linked_full_component_excluded_even_without_a_scope(rows, protected):
    rows[1]["linked_full_witness_component_ids"] = "a" * 64
    selected, decisions = nulls.select(rows, protected, quota=2)
    assert not {"1", "2"} & {r["source_id"] for r in selected}
    assert any(e["reason"] == "linked_full_witness_component_ids" for d in decisions for e in d["exclusions"])


@pytest.mark.parametrize(
    "field,key",
    [
        ("source_id", "source_ids"),
        ("origin_parent", "parent_identities"),
        ("raw_fits_sha256", "raw_or_pixel_content_hashes"),
        ("metadata_dependency_component_id", "additional_original_stopped_components_protected"),
    ],
)
def test_protected_identity_excludes_unprotected_sibling(rows, protected, field, key):
    protected[key].add(rows[1][field])
    selected, decisions = nulls.select(rows, protected, quota=2)
    assert not {"1", "2"} & {r["source_id"] for r in selected}
    decision = next(r for r in decisions if "1" in r["source_ids"])
    assert any(e["reason"] == key for e in decision["exclusions"])
    ledger = list(nulls.pool_ledger(rows, selected, decisions))
    assert ledger[0]["status"] == ledger[1]["status"] == "EXCLUDED_COMPONENT"


def test_conservative_alias_remains_excluded_without_quarter_certification(rows, protected):
    protected["alias_parents"].add(rows[1]["origin_parent"])
    selected, decisions = nulls.select(rows, protected, quota=2)
    assert not {"1", "2"} & {r["source_id"] for r in selected}
    assert any(e["reason"] == "conservative_parent_alias" for d in decisions for e in d["exclusions"])


@pytest.mark.parametrize("kind", ["origin_parent", "raw_fits_sha256"])
def test_cross_component_identity_cannot_be_split_between_roles(rows, protected, kind):
    rows[2][kind] = rows[1][kind]
    with pytest.raises(ValueError, match="identity split"):
        nulls.select(rows, protected, quota=2)


def test_insufficient_quota_and_duplicate_source_refuse(rows, protected):
    with pytest.raises(ValueError, match="Exact development/evaluation quotas refused"):
        nulls.select(rows, protected, quota=6)
    with pytest.raises(ValueError, match="Repeated pool source"):
        nulls.select(rows + [rows[0]], protected, quota=2)


def test_full_ledger_and_ordered_csv_keep_original_exposures(tmp_path, rows, protected):
    protected["source_ids"].add("11")
    selected, decisions = nulls.select(rows, protected, quota=2)
    ledger = list(nulls.pool_ledger(rows, selected, decisions))
    assert len(ledger) == len(rows)
    assert [{k: r[k] for k in pool.FIELDS} for r in ledger] == rows
    assert {r["status"] for r in ledger} == {
        "SELECTED",
        "EXCLUDED_COMPONENT",
        "NOT_COMPONENT_REPRESENTATIVE",
        "ELIGIBLE_COMPONENT_BEYOND_QUOTA",
    }
    for role in ("development", "evaluation"):
        path = tmp_path / f"{role}.csv"
        nulls.write_membership(path, [r for r in selected if r["role"] == role])
        with path.open() as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == list(pool.FIELDS)
            assert list(reader) == [{k: r[k] for k in pool.FIELDS} for r in selected if r["role"] == role]
    assert nulls.RECIPES["execution_authorized"] is False
    assert ">2 usable discovery quarters" in nulls.RECIPES["quarter_scrambled"]


def packet(tmp_path, monkeypatch, names=("reports/test.json",), declared=None):
    path = tmp_path / "reports.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        for name in names:
            item = tarfile.TarInfo(name)
            item.size = 2
            stream.addfile(item, io.BytesIO(b"{}"))
    members = {name: {"bytes": 2, "sha256": sha(b"{}")} for name in (declared or names)}
    manifest = {"archive": nulls.reference(path), "members": members}
    target = tmp_path / "manifest.json"
    target.write_bytes(pool.canonical(manifest))
    monkeypatch.setattr(nulls, "MANIFEST_SHA", pool.digest(target))
    return target, path


def test_closed_packet_hashes_verified(tmp_path, monkeypatch):
    packet(tmp_path, monkeypatch)
    manifest, files, refs = nulls.verified_packet(tmp_path)
    assert files == {"reports/test.json": b"{}"}
    assert len(manifest["members"]) == 1 and len(refs) == 2


@pytest.mark.parametrize(
    "names,declared,error",
    [
        (("../escape.json",), None, "Unsafe archive"),
        (("/escape.json",), None, "Unsafe archive"),
        (("reports/a.json",) * 2, None, "closed and unique"),
        (("reports/a.json",), ("reports/b.json",), "closed and unique"),
    ],
)
def test_packet_refuses_unsafe_or_unclosed_members(tmp_path, monkeypatch, names, declared, error):
    packet(tmp_path, monkeypatch, names, declared)
    with pytest.raises(ValueError, match=error):
        nulls.verified_packet(tmp_path)


def test_packet_refuses_changed_bytes_and_manifest(tmp_path, monkeypatch):
    target, archive = packet(tmp_path, monkeypatch)
    archive.write_bytes(archive.read_bytes() + b"x")
    with pytest.raises(ValueError, match="Metadata bytes changed"):
        nulls.verified_packet(tmp_path)
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="packet manifest"):
        nulls.verified_packet(tmp_path)


def test_quota_refusal_publishes_nothing(tmp_path, monkeypatch, rows, protected):
    monkeypatch.setattr(nulls, "implementation", lambda: {"files": []})
    monkeypatch.setattr(nulls, "load_inputs", lambda _: (rows, protected, {}, []))
    output = tmp_path / "membership"
    with pytest.raises(ValueError, match="quotas refused"):
        nulls.declare(tmp_path, output)
    assert not output.exists()
