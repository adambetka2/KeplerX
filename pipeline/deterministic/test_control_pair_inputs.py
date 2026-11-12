import json
from types import SimpleNamespace

import pytest

from pipeline.deterministic import control_pair_inputs as pairs
from pipeline.deterministic.control_bindings import digest, reference


def test_modified_original_target_bytes_are_rejected(tmp_path):
    path = tmp_path / "raw.fits"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Original target raw hash differs"):
        pairs.verify_product_bytes(path, "a" * 64, set())


def test_protected_raw_bytes_are_rejected_before_header_or_flux_read(tmp_path):
    path = tmp_path / "raw.fits"
    path.write_bytes(b"protected")
    with pytest.raises(ValueError, match="Protected sealed raw payload"):
        pairs.verify_product_bytes(path, None, {digest(path)})


def test_new_donor_hash_is_explicitly_first_bound_and_not_original_proof(tmp_path):
    path = tmp_path / "raw.fits"
    path.write_bytes(b"public")
    assert pairs.verify_product_bytes(path, None, set()) == reference(path)


def test_capacity_accounts_for_existing_use_on_the_whole_filesystem(
    tmp_path, monkeypatch
):
    (tmp_path / "reports.json").write_bytes(b"1")
    monkeypatch.setattr(
        pairs.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=pairs.MINIMUM_FREE_BYTES + 9),
    )
    with pytest.raises(ValueError, match="reserve would be crossed"):
        pairs.capacity(tmp_path, 10)


def test_capacity_keeps_revised_retained_limit(tmp_path, monkeypatch):
    (tmp_path / "reports.json").write_bytes(b"1234")
    monkeypatch.setattr(pairs, "MAXIMUM_RETAINED_BYTES", 5)
    with pytest.raises(ValueError, match="retained-data limit"):
        pairs.capacity(tmp_path, 2)


def test_current_historical_and_holdout_raw_identities_are_protected():
    scopes = [
        (
            None,
            {
                "role": "development",
                "trials": [{"source_id": "1", "raw_fits_sha256": "a" * 64}],
            },
        ),
        (
            None,
            {
                "role": "evaluation",
                "trials": [{"source_id": "2", "raw_fits_sha256": "b" * 64}],
            },
        ),
        (
            None,
            {
                "role": "holdout",
                "trials": [{"source_id": "3", "raw_fits_sha256": "c" * 64}],
            },
        ),
    ]
    assert pairs.sealed_raw_hashes(scopes, {"1"}) == {"a" * 64, "b" * 64, "c" * 64}


def write(path, value):
    path.write_text(json.dumps(value))
    return reference(path)


def test_missing_donor_does_not_trigger_reselection_or_fresh_flux_request(tmp_path):
    binding = {"source_id": "1"}
    ref = write(
        tmp_path / "previous.json", {"binding": binding, "status": "UNAVAILABLE"}
    )
    result = pairs.donor_record(None, binding, {"record": ref}, tmp_path, set(), set())
    assert result["status"] == "CANONICAL_DONOR_UNAVAILABLE_IN_ORIGINAL_HEADER_AUDIT"
    assert result["prior_archive_record"] == ref
    assert result["attempts"] == [] and "payload" not in result


def test_protected_donor_rejected_before_new_request(tmp_path):
    binding = {"source_id": "1"}
    ref = write(
        tmp_path / "previous.json",
        {"binding": binding, "status": "VERIFIED_DONOR_ARCHIVE_HEADERS"},
    )
    previous = {"record": ref, "donor": {"archive_gaia_source_id": "2"}}
    result = pairs.donor_record(None, binding, previous, tmp_path, {"2"}, set())
    assert result["status"] == "UNAVAILABLE"
    assert result["message"] == "Protected sealed donor identity"
    assert result["attempts"] == [] and "payload" not in result


def test_protected_target_rejected_before_any_prior_or_flux_access(tmp_path):
    result = pairs.target_record(
        None, {"source_id": "1"}, {"archive_record": {}}, tmp_path, {"1"}, set()
    )
    assert result["message"] == "Protected sealed target identity"
    assert result["attempts"] == []


def test_scene_protected_pair_target_blocks_its_donor_before_any_prior_access(tmp_path):
    result = pairs.donor_record(
        None, {"source_id": "1"}, {"record": {}}, tmp_path, {"1"}, set()
    )
    assert result["message"] == "Protected sealed pair target"
    assert result["attempts"] == [] and "payload" not in result


def test_failed_archive_attempt_is_retained_in_source_record(tmp_path, monkeypatch):
    source = {"source_id": "1", "pool": {"raw_fits_sha256": "a" * 64}}
    ref = write(
        tmp_path / "previous.json", {"source": source, "archive_product_bytes": 10}
    )

    def fail(session, supplied, total, path, expected, root, protected, attempts):
        attempts.append({"status": 404, "url": "canonical"})
        raise ValueError("Both canonical requests failed")

    monkeypatch.setattr(pairs, "receive_product", fail)
    result = pairs.target_record(
        None, source, {"archive_record": ref}, tmp_path, set(), set()
    )
    assert result["status"] == "UNAVAILABLE"
    assert result["attempts"] == [{"status": 404, "url": "canonical"}]


def test_protected_download_is_not_parsed(tmp_path, monkeypatch):
    def receive(session, source, start, end, attempts):
        attempts.append({"status": 206})
        return b"protected", len(b"protected")

    monkeypatch.setattr(pairs, "range_request", receive)
    monkeypatch.setattr(pairs, "capacity", lambda *args: {})
    path = tmp_path / "raw.fits"
    expected = write(tmp_path / "identity.json", {})
    import hashlib

    protected = {hashlib.sha256(b"protected").hexdigest()}
    attempts = []
    with pytest.raises(ValueError, match="Protected sealed raw"):
        pairs.receive_product(
            None,
            {},
            len(b"protected"),
            path,
            expected["sha256"],
            tmp_path,
            protected,
            attempts,
        )
    assert attempts == [{"status": 206}]
