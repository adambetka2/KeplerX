"""Frozen retrieval preserves identities, reserves and interruption evidence."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "pipeline"),
    str(ROOT / "pipeline/deterministic"),
    str(ROOT / "pipeline/vetting"),
]
from deterministic import frozen_pixel_inputs as frozen


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        frozen.shutil, "disk_usage", lambda path: SimpleNamespace(free=64 * 1024**3)
    )
    payload = tmp_path / "reference.fits.gz"
    primary = fits.PrimaryHDU()
    primary.header.update(KEPLERID=123, QUARTER=0, CHANNEL=1)
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="FLUX", format="4E", dim="(2,2)", array=np.zeros((2, 4))),
        ]
    )
    table.header.update(TSTART=1.0, TSTOP=2.0, BJDREFI=2454833, BJDREFF=0.0)
    fits.HDUList([primary, table, fits.ImageHDU(np.zeros((2, 2)))]).writeto(payload)
    source = {
        "source_id": "7",
        "role": "development",
        "cohort": "ordinary",
        "origin_parent_kic": 123,
        "raw_fits_sha256": "a" * 64,
    }
    source_inventory = tmp_path / "sources.jsonl"
    source_inventory.write_text(json.dumps(source) + "\n")
    request = dict(source)
    request.update(
        parent_kic=123,
        quarter=0,
        status="PUBLIC_PRODUCT_REQUEST",
        transport_uri="https://stpubdata.s3.us-east-1.amazonaws.com/kepler/public/target_pixel_files/0000/000000123/kplr000000123-2009166043257_lpd-targ.fits.gz",
        size_bytes=payload.stat().st_size,
        expected_sha256=frozen.digest(payload),
        astrometry=None,
        etag=None,
    )
    unresolved = {**request, "quarter": 1, "status": "UNRESOLVED_ORIGIN_PRODUCT"}
    requests_path = tmp_path / "requests.json"
    write(
        requests_path,
        {
            "schema": 1,
            "status": "FROZEN_PUBLIC_PIXEL_INPUT_REQUESTS",
            "requests": [request, unresolved],
            "inputs": {str(source_inventory): frozen.digest(source_inventory)},
        },
    )
    folder = tmp_path / "inventory"
    folder.mkdir()
    (folder / "spool").mkdir()
    (folder / "products").mkdir()
    record = {
        "request_index": 0,
        "request": request,
        "payload_retained": False,
        "status": frozen.VERIFIED,
        "transfer": {
            "sha256": frozen.digest(payload),
            "bytes_received": payload.stat().st_size,
        },
        "header": frozen.header_record(payload, request),
    }
    records = [
        record,
        {
            "request_index": 1,
            "request": unresolved,
            "payload_retained": False,
            "status": "ORIGIN_PRODUCT_UNRESOLVED",
        },
    ]
    indices = []
    for index, row in enumerate(records):
        path = folder / "products" / f"product-{index:06d}.json"
        write(path, row)
        indices.append({"request_index": index, "sha256": frozen.digest(path)})
    write(folder / "results.json", indices)
    write(
        folder / "experiment.json",
        {
            "inputs": {
                str(requests_path): frozen.digest(requests_path),
                str(source_inventory): frozen.digest(source_inventory),
            }
        },
    )
    write(
        folder / "audit.json",
        {
            "status": "COMPLETE_BOUNDED_PUBLIC_PIXEL_INPUT_RECORDS",
            "temporary_payloads_remaining": 0,
            "scientific_outcomes_opened": False,
            "boi_catalogue_enabled": False,
            "requests": 2,
            "experiment_sha256": frozen.digest(folder / "experiment.json"),
            "results_sha256": frozen.digest(folder / "results.json"),
        },
    )
    ledger_path = tmp_path / "ledger.json"
    frozen.freeze_inventory(folder, requests_path, source_inventory, ledger_path)
    return SimpleNamespace(
        payload=payload,
        folder=folder,
        source_inventory=source_inventory,
        requests=requests_path,
        ledger_path=ledger_path,
        source=source,
        records=records,
        cache=tmp_path / "cache",
    )


def ledger(item):
    return frozen.FrozenPixelInputs(item.ledger_path, frozen.digest(item.ledger_path))


def test_freeze_preserves_input_bindings_after_symlink_relocation(inventory):
    external = inventory.folder.parent / "external"
    external.mkdir()
    for path in (inventory.folder, inventory.requests, inventory.source_inventory):
        destination = external / path.name
        path.rename(destination)
        path.symlink_to(destination, target_is_directory=destination.is_dir())
    output = external / "relocated-ledger.json"
    frozen.freeze_inventory(
        inventory.folder, inventory.requests, inventory.source_inventory, output
    )
    assert frozen.read(output)["source_quarters"] == 2
    inventory.source_inventory.write_text("changed input bytes")
    with pytest.raises(ValueError, match="input changed"):
        frozen.freeze_inventory(
            inventory.folder,
            inventory.requests,
            inventory.source_inventory,
            external / "changed-ledger.json",
        )


def copier(item, calls):
    def receive(request, target, *, reserve=frozen.RESERVE):
        calls.append(request)
        shutil.copyfile(item.payload, target)

    return receive


def test_freeze_retains_unresolved_quarters_and_exact_role(inventory):
    value = ledger(inventory)
    source = value.ledger["sources"]["7"]
    assert value.ledger["source_quarters"] == 2
    assert source["role"] == "development"
    assert value.record(source, 1)["status"] == "ORIGIN_PRODUCT_UNRESOLVED"
    assert not value.ledger["joint_independence_complete"]
    assert not value.ledger["scientific_outcomes_opened"]


def test_changed_ledger_and_content_record_are_rejected(inventory):
    checksum = frozen.digest(inventory.ledger_path)
    inventory.ledger_path.write_text(inventory.ledger_path.read_text() + " ")
    with pytest.raises(ValueError, match="ledger SHA256"):
        frozen.FrozenPixelInputs(inventory.ledger_path, checksum)
    value = ledger(inventory)
    path = inventory.folder / "products/product-000000.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="content record changed"):
        value.record(value.ledger["sources"]["7"], 0)


@pytest.mark.parametrize(
    "field,value",
    [("role", "evaluation"), ("origin_tpf_kic", 124), ("raw_fits_sha256", "b" * 64)],
)
def test_source_role_parent_and_raw_identity_are_enforced(inventory, field, value):
    metadata = {"origin_tpf_kic": 123, "raw_fits_sha256": "a" * 64}
    role = value if field == "role" else "development"
    if field != "role":
        metadata[field] = value
    with pytest.raises(ValueError, match="source, role, parent or raw identity"):
        ledger(inventory).source("7", metadata, role)


def test_exact_content_is_reused_and_missing_inputs_stay_unresolved(
    inventory, monkeypatch
):
    calls = []
    monkeypatch.setattr(frozen, "receive_product", copier(inventory, calls))
    first = frozen.retrieve_records(inventory.records, inventory.cache)
    second = frozen.retrieve_records(inventory.records, inventory.cache, network=False)
    assert first[0]["path"] == second[0]["path"]
    assert not first[0]["cache_hit"] and second[0]["cache_hit"]
    assert first[0]["downloaded_bytes"] == inventory.payload.stat().st_size
    assert second[0]["downloaded_bytes"] == 0
    assert first[1] == second[1]
    assert len(calls) == 1
    assert calls[0]["expected_sha256"] == frozen.digest(inventory.payload)
    assert first[0]["status"] == "AVAILABLE"
    assert first[0]["base_parent_kic"] == 123
    assert first[1]["status"] == "FROZEN_PIXEL_INPUT_UNRESOLVED"
    assert not list(inventory.cache.glob("*.tmp"))


def test_corrupt_cache_is_preserved_and_rejected_without_redownload(
    inventory, monkeypatch
):
    monkeypatch.setattr(frozen, "receive_product", copier(inventory, []))
    row = frozen.retrieve_records([inventory.records[0]], inventory.cache)[0]
    path = Path(row["path"])
    path.write_bytes(b"corrupt")
    monkeypatch.setattr(
        frozen,
        "receive_product",
        lambda *args: pytest.fail("Corruption must not silently redownload"),
    )
    with pytest.raises(ValueError, match="cached frozen pixel payload"):
        frozen.retrieve_records([inventory.records[0]], inventory.cache)
    assert path.read_bytes() == b"corrupt"


def test_disk_reserve_and_cache_bound_prevent_transfer(inventory, monkeypatch):
    monkeypatch.delenv("SPACELAB_EXECUTION_PROFILE", raising=False)
    monkeypatch.setattr(
        frozen,
        "receive_product",
        lambda *args: pytest.fail("A bounded failure must not start a transfer"),
    )
    rows = [inventory.records[0]]
    with pytest.raises(OSError, match="byte allowance is exhausted"):
        frozen.retrieve_records(rows, inventory.cache, maximum_bytes=1)
    monkeypatch.setattr(
        frozen.shutil, "disk_usage", lambda path: SimpleNamespace(free=frozen.RESERVE)
    )
    with pytest.raises(OSError, match="disk reserve"):
        frozen.retrieve_records(rows, inventory.cache)
    with pytest.raises(OSError, match="absent from the offline cache"):
        frozen.retrieve_records(rows, inventory.cache, network=False)
    with pytest.raises(ValueError, match="original reserve"):
        frozen.retrieve_records(rows, inventory.cache, reserve=1)


def test_cloud_development_uses_only_an_explicit_smaller_reserve(inventory, monkeypatch):
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    monkeypatch.setattr(frozen, "EXTERNAL_DUMPS", ())
    available = 3 * 1024**3
    monkeypatch.setattr(frozen.shutil, "disk_usage", lambda path: SimpleNamespace(free=available))
    calls = []

    def receive(request, target, *, reserve):
        calls.append(reserve)
        shutil.copyfile(inventory.payload, target)

    monkeypatch.setattr(frozen, "receive_product", receive)
    with pytest.raises(OSError, match="disk reserve"):
        frozen.retrieve_records([inventory.records[0]], inventory.cache)
    result = frozen.retrieve_records([inventory.records[0]], inventory.cache, reserve=2 * 1024**3)
    assert result[0]["status"] == "AVAILABLE"
    assert calls == [2 * 1024**3]
    assert frozen.digest(Path(result[0]["path"])) == frozen.digest(inventory.payload)
    Path(result[0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="cached frozen pixel payload"):
        frozen.retrieve_records([inventory.records[0]], inventory.cache, reserve=2 * 1024**3)


@pytest.mark.parametrize("protected", ["external_cache", "evaluation_inputs"])
def test_cloud_profile_cannot_lower_external_or_evaluation_reserve(inventory, monkeypatch, protected):
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    rows = [json.loads(json.dumps(inventory.records[0]))]
    roots = (inventory.cache.parent,) if protected == "external_cache" else ()
    monkeypatch.setattr(frozen, "EXTERNAL_DUMPS", roots)
    if protected == "evaluation_inputs":
        rows[0]["request"]["role"] = "evaluation"
    with pytest.raises(ValueError, match="original reserve"):
        frozen.retrieve_records(rows, inventory.cache, reserve=1)
    assert not inventory.cache.exists()


@pytest.mark.parametrize("reserve", [0, -1, True, 1.5])
def test_cloud_profile_rejects_invalid_reserve(inventory, monkeypatch, reserve):
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    with pytest.raises(ValueError, match="positive integer"):
        frozen.retrieve_records([inventory.records[0]], inventory.cache, reserve=reserve)


def test_failed_transfer_cleans_only_its_own_bytes_and_can_resume(
    inventory, monkeypatch
):
    def interrupted(request, target, **kwargs):
        target.write_bytes(b"partial")
        raise OSError("Interrupted transfer")

    monkeypatch.setattr(frozen, "receive_product", interrupted)
    with pytest.raises(OSError, match="Interrupted transfer"):
        frozen.retrieve_records([inventory.records[0]], inventory.cache)
    assert not list(inventory.cache.glob("*.tmp"))
    assert not list(inventory.cache.glob("*.fits.gz"))
    monkeypatch.setattr(frozen, "receive_product", copier(inventory, []))
    assert (
        frozen.retrieve_records([inventory.records[0]], inventory.cache)[0]["status"]
        == "AVAILABLE"
    )


def test_occupied_temporary_payload_is_never_removed(inventory, monkeypatch):
    inventory.cache.mkdir()
    temporary = inventory.cache / f"{frozen.digest(inventory.payload)}.fits.tmp"
    temporary.write_bytes(b"prior interrupted evidence")
    monkeypatch.setattr(
        frozen,
        "receive_product",
        lambda *args: pytest.fail("An occupied temporary path must not be reused"),
    )
    with pytest.raises(FileExistsError, match="temporary path is already occupied"):
        frozen.retrieve_records([inventory.records[0]], inventory.cache)
    assert temporary.read_bytes() == b"prior interrupted evidence"


def test_changed_transport_content_is_not_published(inventory, monkeypatch):
    monkeypatch.setattr(
        frozen,
        "receive_product",
        lambda request, target, **kwargs: target.write_bytes(b"changed"),
    )
    with pytest.raises(ValueError, match="content identity"):
        frozen.retrieve_records([inventory.records[0]], inventory.cache)
    assert not list(inventory.cache.glob("*.tmp"))
    assert not list(inventory.cache.glob("*.fits.gz"))


def test_existing_cache_cannot_exceed_a_smaller_declared_bound(inventory, monkeypatch):
    monkeypatch.setattr(frozen, "receive_product", copier(inventory, []))
    row = frozen.retrieve_records([inventory.records[0]], inventory.cache)[0]
    checksum = frozen.digest(row["path"])
    with pytest.raises(OSError, match="exceeds its byte allowance"):
        frozen.retrieve_records(
            [inventory.records[0]], inventory.cache, maximum_bytes=1
        )
    assert frozen.digest(row["path"]) == checksum


def test_incomplete_content_inventory_cannot_freeze(inventory):
    records = frozen.read(inventory.folder / "results.json")[:-1]
    source_inventory = inventory.source_inventory
    with pytest.raises(ValueError, match="retained exactly once"):
        frozen.inventory_sources(
            inventory.folder,
            frozen.read(inventory.requests)["requests"],
            records,
            source_inventory,
        )


def test_cli_configuration_requires_an_explicit_frozen_role(inventory):
    from deterministic.run import frozen_pixel_configuration

    args = SimpleNamespace(frozen_pixel_inputs=inventory.ledger_path)
    with pytest.raises(ValueError, match="explicit input role"):
        frozen_pixel_configuration(args)
    args.pixel_input_role = "development"
    binding = frozen_pixel_configuration(args)["frozen_pixel_inputs"]
    assert binding["sha256"] == frozen.digest(inventory.ledger_path)
    assert binding["maximum_cache_bytes"] == frozen.DEFAULT_CACHE_BYTES
    args.frozen_pixel_inputs = None
    with pytest.raises(ValueError, match="require a content ledger"):
        frozen_pixel_configuration(args)


def test_vetting_uses_frozen_bytes_and_rejects_wrong_scope(
    inventory, monkeypatch, tmp_path
):
    import vetting_pixels as pixels

    calls = []
    monkeypatch.setattr(frozen, "receive_product", copier(inventory, calls))
    run = tmp_path / "run"
    monkeypatch.setattr(pixels, "RUN", run)
    monkeypatch.setattr(pixels, "PIXELS", inventory.cache)
    config = tmp_path / "config.json"
    write(
        config,
        {
            "network": True,
            "evaluation_scope": {"evaluation_role": "development_replay"},
        },
    )
    monkeypatch.setattr(pixels, "CONFIG", config)
    pixels.save(
        run / "sources/7.json",
        {
            "signals": [{"event_ids": ["event"], "triage": {"status": "ACTIVE"}}],
            "pixel_jobs": [{"event_id": "event", "quarter": 0}],
        },
    )
    binding = {
        "path": str(inventory.ledger_path),
        "sha256": frozen.digest(inventory.ledger_path),
        "role": "development",
        "maximum_cache_bytes": frozen.DEFAULT_CACHE_BYTES,
    }
    metadata = {"7": {"origin_tpf_kic": 123, "raw_fits_sha256": "a" * 64}}
    pixels.retrieve_frozen_pixels(binding, metadata)
    manifest = frozen.read(run / "pixel-manifests/123.json")
    assert manifest["quarters"]["0"]["sha256"] == frozen.digest(inventory.payload)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="authorized evaluation"):
        pixels.retrieve_frozen_pixels({**binding, "role": "evaluation"}, metadata)


def reconciled_fixture(inventory, source, quarter, item):
    path = inventory.folder.parent / "reconciled-ledger.json"
    write(
        path,
        {
            "schema": 2,
            "status": "FROZEN_CONTROL_PIXEL_CONTENT_LEDGER",
            "inputs": {},
            "sources": {
                source["source_id"]: {**source, "quarters": {str(quarter): item}}
            },
        },
    )
    return frozen.FrozenPixelInputs(path, frozen.digest(path))


def test_reconciled_reserve_receipt_requires_role_transition_and_exact_bytes(
    inventory, monkeypatch
):
    source = {**inventory.source, "role": "evaluation", "cohort": "native_eb"}
    record = {
        **inventory.records[0],
        "header": {
            **inventory.records[0]["header"],
            "target_geometry": {"target_point_inside_pixel_footprint": True},
        },
        "request": {
            **inventory.records[0]["request"],
            "role": "UNASSIGNED_RESERVE",
            "cohort": "catalogue_eclipsing_binary_unassigned_reserve",
            "status": "UNASSIGNED_RESERVE_PIXEL_INPUT_UNRESOLVED",
        },
    }
    path = inventory.folder / "products/reserve.json"
    write(path, record)
    item = {
        **source,
        "quarter": 0,
        "kind": "REPLACEMENT_RESERVE_ROLE_BOUND",
        "path": str(path),
        "sha256": frozen.digest(path),
        "actual_parent_kic": 123,
        "content_sha256": frozen.digest(inventory.payload),
    }
    ledger = reconciled_fixture(inventory, source, 0, item)
    bound = ledger.source(
        source["source_id"],
        {"origin_tpf_kic": 123, "raw_fits_sha256": source["raw_fits_sha256"]},
        "evaluation",
    )
    normalized = ledger.record(bound, 0)
    assert normalized["request"]["role"] == "evaluation"
    assert normalized["request"]["cohort"] == "native_eb"
    assert normalized["request"]["status"] == "PUBLIC_PRODUCT_REQUEST"
    monkeypatch.setattr(frozen, "receive_product", copier(inventory, []))
    assert (
        frozen.retrieve_records([normalized], inventory.cache)[0]["status"]
        == "AVAILABLE"
    )
    bad = {**record, "request": {**record["request"], "role": "development"}}
    write(path, bad)
    item["sha256"] = frozen.digest(path)
    with pytest.raises(ValueError, match="original reserve role"):
        frozen.reconciled_record(item, source, 0)


def test_reconciled_unavailable_quarter_keeps_reason_and_rejects_known_support(
    inventory,
):
    source = inventory.source
    path = inventory.folder / "products/product-000001.json"
    item = {
        **source,
        "quarter": 1,
        "kind": "UNAVAILABLE",
        "reason": "ORIGIN_PRODUCT_UNRESOLVED",
        "nominal_path": str(path),
        "nominal_sha256": frozen.digest(path),
    }
    ledger = reconciled_fixture(inventory, source, 1, item)
    bound = ledger.source(
        source["source_id"],
        {"origin_tpf_kic": 123, "raw_fits_sha256": source["raw_fits_sha256"]},
        "development",
    )
    result = frozen.retrieve_records(
        [ledger.record(bound, 1)], inventory.cache, network=False
    )[0]
    assert result["status"] == "FROZEN_PIXEL_INPUT_UNRESOLVED"
    assert result["input_reason"] == "ORIGIN_PRODUCT_UNRESOLVED"
    supported = inventory.folder / "products/product-000000.json"
    supported_record = frozen.read(supported)
    supported_record["header"]["target_geometry"] = {
        "target_point_inside_pixel_footprint": True
    }
    write(supported, supported_record)
    item.update(
        quarter=0, nominal_path=str(supported), nominal_sha256=frozen.digest(supported)
    )
    with pytest.raises(ValueError, match="has public support"):
        frozen.reconciled_record(item, source, 0)
