"""Exact opened180 transport, bounded ownership and schema2 handoff checks."""

import copy
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from deterministic import cloud_opened_pixels as transport
from deterministic.test_cloud_injected_pixels import (
    Response,
    inventory,
    npz,
    reference_injection,
    tpf,
)

SID = "2052214955493700224"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def record(payload):
    return {"bytes": len(payload), "sha256": sha(payload)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    path.write_bytes(transport.serialized(value))


def preservation(payload):
    with fits.open(io.BytesIO(gzip.decompress(payload)), memmap=False) as hdus:
        ignore = {"CHECKSUM", "DATASUM", "FSCSRC", "FSCMODEL"}
        headers = [{"hdu": i, "original_noninjection_cards_sha256": transport.canonical_hash(
            [card.image for card in hdu.header.cards if card.keyword not in ignore])} for i, hdu in enumerate(hdus)]
        arrays = {}
        for i, hdu in enumerate(hdus):
            if hdu.data is not None:
                for name in hdu.columns.names:
                    item = inventory({name: np.asarray(hdu.data[name])})[name]
                    arrays[f"{i}:{name}"] = {"base": item, "injected": item, "unchanged": name != "FLUX"}
        return {"headers": headers, "arrays": arrays}


def archive(case, members=None):
    with tarfile.open(case.archive, "w:gz", format=tarfile.USTAR_FORMAT) as stream:
        for name, payload, kind in members or [(name, value, tarfile.REGTYPE) for name, value in case.contents.items()]:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(payload) if kind == tarfile.REGTYPE else 0
            stream.addfile(info, io.BytesIO(payload) if info.size else None)
    case.data["archive"] = {"path": case.archive.name, **transport.identity(case.archive)}


@pytest.fixture
def case(tmp_path, monkeypatch):
    root, corpus, cache, originals = (tmp_path / name for name in ("pixels", "corpus", "cache", "originals"))
    originals.mkdir()
    corpus.mkdir()
    cache.mkdir()
    base = tpf(123, 1, 1)
    factors = {"profile": np.array([[0.1, 0.2], [0.3, 0.4]]), "loss": np.array([0.0, 0.05, 0.1]), "amplitude": np.array(10.0)}
    recipe = f"{SID}-q01.npz"
    injected = reference_injection(base, SID, factors)
    filename = "kplr000000123-2009000000001_lpd-targ.fits.gz"
    prf = "kplr02.1_2011265_prf.fits"
    buffer = io.BytesIO()
    fits.PrimaryHDU(np.ones(2)).writeto(buffer)
    prf_payload = buffer.getvalue()
    pixel = {"path": f"pixels/{SID}/q01.fits.gz", "sha256": sha(injected), "base_path": f"original/{filename}",
             "base_parent_kic": 123, "base_sha256": sha(base), "prf_calibration": {"filename": prf, "sha256": sha(prf_payload)},
             "target_pixel": [3.0, 3.0], "target_inside_stamp": False, "target_sky_deg": [294.0, 39.0],
             "target_astrometry": "ORIGINAL_OPENED_ASTROMETRY", "source_flux_electrons_per_second": 10.0,
             "maximum_relative_loss": 0.1, "prf_stamp_fraction": 1.0}
    with fits.open(io.BytesIO(gzip.decompress(base)), memmap=False) as hdus:
        flux = inventory({"flux": hdus[1].data["FLUX"]})["flux"]
    base_row = {"filename": filename, "url": f"{transport.pixels.S3}/0000/000000123/{filename}",
                **record(base), "parent_kic": 123, "quarter": 1, "channel": 1, "flux": flux}
    quarter = {"recipe": recipe, "base": base_row, "injected": record(injected),
               "original_pixel_record": pixel, "preservation": preservation(injected)}
    source = {"source_id": SID, "metadata": {"source_id": SID, "origin_tpf_kic": 999, "shard_path": "original/shard-0000"},
              "base_raw_fits_sha256": "a" * 64, "deltas": f"original/{SID}.npz", "sha256": "b" * 64,
              "truth_index": 0, "pixels": {"1": pixel}, "spatial_tpf_query": {"status": "ORIGINAL"}}
    gaia_row = {"source_id": SID, "ra": 294.0, "dec": 39.0}
    gaia = transport.serialized({"status": "COMPLETE", "rows": [gaia_row]})
    ancillary = "lineage/periodic_planet/gaia-astrometry.json"
    contents = {recipe: npz(factors), ancillary: gaia}
    data = {"source_order": [SID], "sources": {SID: {"source_id": SID, "opened_cohort": "periodic_planet",
        "opened_scope": "CURRENT_OPENED_DEVELOPMENT", "raw_fits_sha256": "a" * 64, "origin_parent_kic": 999,
        "original_source": source, "original_source_sha256": transport.canonical_hash(source),
        "gaia_row_sha256": transport.canonical_hash(gaia_row), "quarters": {"1": quarter}}},
        "files": {recipe: {**record(contents[recipe]), "arrays": inventory(factors)}, ancillary: record(gaia)},
        "lineage": {"periodic_planet": {"ancillary": {"gaia-astrometry.json": {"member": ancillary, **record(gaia)}},
            "original_files": {"input-manifest.json": {"sha256": "c" * 64}}}},
        "prf_calibrations": {prf: {**record(prf_payload), "url": f"{transport.pixels.PRF}/{prf}"}},
        "counts": {"sources": 180, "quarters": 1540}, "provenance": {"original_scope_counts": {"original": 876}}}
    template = {"schema": 2, "transform": "additive_flux_delta", "measurement_scope": "PHYSICAL_PIXEL_INJECTION",
                "protocol": "ORIGINAL", "generator": {"original.py": "d" * 64},
                "truth": {"path": "original.jsonl", "sha256": "e" * 64},
                "gaia_astrometry": {"path": "original-gaia.json", "sha256": sha(gaia)}, "sources": {SID: copy.deepcopy(source)}}
    template["sources"][SID]["metadata"]["shard_path"] = str(corpus / "discSSD/data/kbonus/lc2/shard-0000")
    manifest = corpus / "transport/periodic_planet/input-manifest.json"
    write(manifest, template)
    prep = tmp_path / "preparation.json"
    write(prep, {"verified": "UNIT_FIXTURE_ONLY"})
    packet = {"budgets": {"max_cloud_storage_bytes": 32_000_000_000, "cloud_development_minimum_free_bytes": 2 * 1024**3,
              "external_minimum_free_bytes": 20 * 1024**3, "paired_run_headroom_bytes": 4 * 1024**3}}
    original = {"selection": {"selected_source_ids": [SID]}, "manifests": {"periodic_planet": str(manifest)}}
    mapping = {"schema": 1, "source_ids": [SID], "base": {}, "prf": {}}
    for kind, name, value in (("base", filename, base), ("prf", prf, prf_payload)):
        path = originals / name
        path.write_bytes(value)
        path.chmod(0o444)
        mapping[kind][name] = {"path": str(path), **record(value)}
    result = SimpleNamespace(root=root, corpus=corpus, cache=cache, prep=prep, data=data, contents=contents,
        mapping=mapping, base=base, injected=injected, filename=filename, prf=prf, factors=factors,
        spec=tmp_path / "cloud-opened-pixels.json", archive=tmp_path / "cloud-opened-pixels.tar.gz", calls=[])
    archive(result)
    result.spec.write_bytes(b"unit-specification")
    monkeypatch.setattr(transport, "metadata", lambda spec: (data, {spec: transport.opened.digest(spec)}))
    monkeypatch.setattr(transport, "context", lambda *_: (original, {}, packet,
                        {manifest: transport.opened.digest(manifest), prep: transport.opened.digest(prep)}))
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: SimpleNamespace(total=32_000_000_000, free=31_000_000_000))
    def get(url, **kwargs):
        assert kwargs == {"stream": True, "timeout": (15, 120), "allow_redirects": False}
        result.calls.append(url)
        payload = base if url == base_row["url"] else prf_payload
        return Response(url, payload)
    monkeypatch.setattr(transport.pixels.requests, "get", get)
    return result


def run(case, **kwargs):
    return transport.provision(case.prep, corpus_root=case.corpus, opened_cache=case.cache,
        destination=case.root, specification=case.spec, mode="retained", retained_inputs=case.mapping, **kwargs)


def withdrawn(case):
    assert not any((case.root / name).exists() for name in transport.SUCCESS)
    assert not (case.root / "transport/periodic_planet/input-manifest.json").exists()


def test_exact_retained_replay_original_parent_and_schema2_handoff(case, monkeypatch):
    import vetting_pixels

    result = run(case)
    assert result["metrics"]["network_bytes"] == 0 and case.calls == []
    assert result["selection"]["full180_transport_complete"] is True  # Tiny unit scope, never a real denominator.
    assert result["counts"] == {"sources": 180, "quarters": 1540}
    assert all(result[flag] is False for flag in transport.FLAGS)
    assert result["full_frozen_materialization_created"] is False
    output = case.root / f"pixels/{SID}/q01.fits.gz"
    assert output.read_bytes() == case.injected and output.stat().st_mode & 0o222 == 0
    assert Path(case.mapping["base"][case.filename]["path"]).read_bytes() == case.base
    manifest = Path(result["manifests"]["periodic_planet"])
    data = json.loads(manifest.read_bytes())
    row = data["sources"][SID]
    assert data["schema"] == 2 and row["pixels"]["1"]["base_parent_kic"] == 123
    assert row["metadata"]["origin_tpf_kic"] == 999 and row["pixels"]["1"]["target_inside_stamp"] is False
    assert data["generator"] == {"original.py": "d" * 64}
    run_folder = case.root / "test-vetting"
    write(run_folder / "sources" / f"{SID}.json", {"pixel_jobs": [{"quarter": 1}]})
    monkeypatch.setattr(vetting_pixels, "RUN", run_folder)
    vetting_pixels.retrieve_experiment(manifest, {SID: row["metadata"]})
    fetched = json.loads((run_folder / "pixel-manifests/999.json").read_bytes())
    assert fetched["quarters"]["1"]["status"] == "AVAILABLE"
    assert fetched["quarters"]["1"]["target_inside_stamp"] is False


def test_verified_cache_rechecks_every_dependency_without_network(case):
    first = run(case)
    second = transport.verified_pixels(case.prep, corpus_root=case.corpus, opened_cache=case.cache,
        destination=case.root, specification=case.spec)
    assert second["metrics"]["network_bytes"] == 0 and case.calls == []
    for name in ("config.json", "source-ids.json"):
        assert first["dependencies"][str(case.root / name)] == second["dependencies"][str(case.root / name)]
    assert first["relocation_witnesses"] == second["relocation_witnesses"]


@pytest.mark.parametrize("kind", ["base", "prf", "pixel", "recipe", "preparation"])
def test_changed_input_or_output_withdraws_previous_success(case, kind):
    run(case)
    paths = {"base": Path(case.mapping["base"][case.filename]["path"]), "prf": case.root / "prf" / case.prf,
             "pixel": case.root / f"pixels/{SID}/q01.fits.gz", "recipe": case.root / "recipes" / f"{SID}-q01.npz",
             "preparation": case.prep}
    paths[kind].chmod(0o644)
    paths[kind].write_bytes(b"changed")
    with pytest.raises(ValueError):
        transport.verified_pixels(case.prep, corpus_root=case.corpus, opened_cache=case.cache,
            destination=case.root, specification=case.spec)
    withdrawn(case)


@pytest.mark.parametrize("name", ["receipt.json", "config.json", "source-ids.json", "base/evil", "prf/evil", "recipes/evil"])
def test_owned_symlink_failure_removes_only_safe_success_outputs(case, name):
    run(case)
    outside = case.root.parent / "outside"
    outside.write_bytes(b"untouched")
    path = case.root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink|redirect"):
        run(case)
    assert path.is_symlink() and outside.read_bytes() == b"untouched"
    assert all(not (case.root / entry).exists() or (case.root / entry).is_symlink() for entry in transport.SUCCESS)


def test_unrelated_occupied_destination_is_unchanged(case):
    write(case.root / "receipt.json", {"unrelated": True})
    before = (case.root / "receipt.json").read_bytes()
    with pytest.raises(ValueError, match="unambiguously owned"):
        run(case)
    assert (case.root / "receipt.json").read_bytes() == before
    assert not (case.root / transport.OWNER).exists()


@pytest.mark.parametrize("target", ["manifest", "source-ids", "config", "receipt"])
def test_publication_drift_withdraws_all_usable_outputs(case, monkeypatch, target):
    publish = transport.pixels.publish
    def drift(path, payload, root):
        publish(path, payload, root)
        hit = path.name == f"{target}.json" or target == "manifest" and path.name == "input-manifest.json"
        if hit:
            path.chmod(0o644)
            path.write_bytes(payload + b" ")
    monkeypatch.setattr(transport.pixels, "publish", drift)
    with pytest.raises(ValueError, match="Published pixel"):
        run(case)
    withdrawn(case)


def test_late_preparation_dependency_drift_withdraws_success(case, monkeypatch):
    finish = transport.finish
    def drift(*args):
        result = finish(*args)
        case.prep.write_bytes(b"changed")
        return result
    monkeypatch.setattr(transport, "finish", drift)
    with pytest.raises(ValueError):
        run(case)
    withdrawn(case)


@pytest.mark.parametrize("change", ["extra", "missing", "wrong-hash", "wrong-source"])
def test_retained_mapping_is_closed_and_never_downloads(case, change):
    if change == "extra":
        case.mapping["base"]["unknown"] = next(iter(case.mapping["base"].values()))
    elif change == "missing":
        case.mapping["prf"].clear()
    elif change == "wrong-hash":
        case.mapping["base"][case.filename]["sha256"] = "f" * 64
    else:
        case.mapping["source_ids"] = ["9" * 19]
    with pytest.raises(ValueError):
        run(case)
    assert case.calls == []
    withdrawn(case)


@pytest.mark.parametrize("change", ["duplicate", "unknown", "traversal", "directory", "wrong-array"])
def test_recipe_archive_is_closed_regular_and_exact(case, change):
    items = [(name, value, tarfile.REGTYPE) for name, value in case.contents.items()]
    if change == "duplicate":
        items.append(items[0])
    elif change in {"unknown", "traversal"}:
        items.append(("unknown" if change == "unknown" else "../escape", b"bad", tarfile.REGTYPE))
    elif change == "directory":
        items[0] = (items[0][0], b"", tarfile.DIRTYPE)
    else:
        factors = dict(case.factors, loss=np.zeros(3, dtype=np.float32))
        value = npz(factors)
        items[0] = (items[0][0], value, tarfile.REGTYPE)
        case.data["files"][items[0][0]].update(record(value))
    archive(case, items)
    with pytest.raises(ValueError):
        run(case)
    withdrawn(case)


def test_capacity_uses_decimal32gb_and_paired_headroom(case, monkeypatch):
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    monkeypatch.setattr(transport.corpus, "cloud_destination", lambda _: True)
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: SimpleNamespace(total=40_000_000_000, free=13_000_000_000))
    with pytest.raises(ValueError, match="exact32GB"):
        run(case)
    assert case.calls == []
    withdrawn(case)


def test_external_reserve_cannot_be_reduced_by_cloud_profile(case, monkeypatch):
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    monkeypatch.setattr(transport.corpus, "cloud_destination", lambda _: False)
    monkeypatch.setattr(transport.shutil, "disk_usage", lambda _: SimpleNamespace(total=32_000_000_000, free=25_000_000_000))
    with pytest.raises(ValueError, match="reserve and paired headroom"):
        run(case)
    withdrawn(case)


def test_checked_in_full_original_recipe_and_provenance_inventory():
    descriptor = transport.opened.parse(transport.opened.SPECIFICATION.read_bytes())
    metadata, _ = transport.opened.decoded_metadata(transport.opened.SPECIFICATION, descriptor)
    if any(transport.opened.digest(transport.opened.ROOT / item["path"]) != item["sha256"] for item in metadata["reader_programs"]):
        with pytest.raises(ValueError, match="wrong-sized opened input|Opened input bytes changed"):
            transport.metadata(transport.SPECIFICATION)
        return
    data, dependencies = transport.metadata(transport.SPECIFICATION)
    contents = transport.bundle(transport.SPECIFICATION, data)
    assert len(data["source_order"]) == 180 and len(contents) == 1545
    assert data["counts"]["quarters"] == 1540 and data["counts"]["nonorigin_parent_quarters"] == 73
    assert data["counts"]["outside_stamp_quarters"] == 207 and len(data["prf_calibrations"]) == 67
    assert all(data[flag] is False for flag in transport.FLAGS)
    transport.opened.recheck(dependencies)


@pytest.mark.parametrize("change", ["boolean-quarter", "missing-stamp", "large-flux", "wrong-profile", "wrong-channel", "unknown-parent"])
def test_original_quarter_binding_preflight_rejects_corruption(change):
    data = json.loads(gzip.decompress(transport.SPECIFICATION.with_suffix(".json.gz").read_bytes()))
    sid = data["source_order"][0]
    quarter, original = next(iter(data["sources"][sid]["quarters"].items()))
    row = copy.deepcopy(original)
    if change == "boolean-quarter":
        row["base"]["quarter"] = True
    elif change == "missing-stamp":
        row["original_pixel_record"].pop("target_inside_stamp")
    elif change == "large-flux":
        row["base"]["flux"]["shape"] = [9999, 9999, 9999]
    elif change == "wrong-profile":
        data["files"][row["recipe"]]["arrays"]["profile"]["dtype"] = "float32"
    elif change == "wrong-channel":
        row["base"]["channel"] = 84
    else:
        row["base"]["parent_kic"] += 1
    with pytest.raises(ValueError):
        transport.check_quarter(sid, quarter, row, data)


@pytest.mark.parametrize("key", ["scientific_acceptance", "joint_independence_complete", "pixel_readiness_complete", "automatic_pc_enabled"])
def test_forged_readiness_receipt_is_rejected_and_withdrawn(case, key):
    run(case)
    receipt = json.loads((case.root / "receipt.json").read_bytes())
    receipt[key] = True
    write(case.root / "receipt.json", receipt)
    with pytest.raises(ValueError, match="scientific boundary"):
        transport.verified_pixels(case.prep, corpus_root=case.corpus, opened_cache=case.cache,
            destination=case.root, specification=case.spec)
    withdrawn(case)


def test_cached_handoff_cannot_silently_change_preparation_path(case):
    run(case)
    other = case.prep.with_name("other-preparation.json")
    other.write_bytes(case.prep.read_bytes())
    with pytest.raises(ValueError, match="upstream paths"):
        transport.verified_pixels(other, corpus_root=case.corpus, opened_cache=case.cache,
            destination=case.root, specification=case.spec)
    withdrawn(case)


def test_retained_alias_drift_after_publication_withdraws_success(case, monkeypatch):
    original = Path(case.mapping["base"][case.filename]["path"])
    alias = original.with_name("alias.fits.gz")
    alias.symlink_to(original)
    case.mapping["base"][case.filename]["path"] = str(alias)
    other = original.with_name("other.fits.gz")
    other.write_bytes(original.read_bytes())
    finish = transport.finish
    def drift(*args):
        result = finish(*args)
        alias.unlink()
        alias.symlink_to(other)
        return result
    monkeypatch.setattr(transport, "finish", drift)
    with pytest.raises(ValueError, match="redirected"):
        run(case)
    withdrawn(case)
    assert original.read_bytes() == case.base


def test_download_mode_uses_closed_canonical_products_then_warm_cache(case):
    result = transport.provision(case.prep, corpus_root=case.corpus, opened_cache=case.cache,
        destination=case.root, specification=case.spec)
    assert len(case.calls) == 2 and result["metrics"]["network_bytes"] > 0
    second = transport.verified_pixels(case.prep, corpus_root=case.corpus, opened_cache=case.cache,
        destination=case.root, specification=case.spec)
    assert len(case.calls) == 2 and second["metrics"]["network_bytes"] == 0


def test_network_policy_failure_does_not_publish_success(case, monkeypatch):
    def blocked(*_, **__):
        raise OSError("configured network policy rejected canonical host")
    monkeypatch.setattr(transport.pixels.requests, "get", blocked)
    with pytest.raises(OSError, match="network policy"):
        transport.provision(case.prep, corpus_root=case.corpus, opened_cache=case.cache,
            destination=case.root, specification=case.spec)
    withdrawn(case)


def test_resume_does_not_blindly_trust_source_checkpoint(case):
    run(case)
    path = case.root / "checkpoints" / f"{SID}.json"
    write(path, {"forged": "not an input oracle"})
    result = run(case)
    checkpoint = json.loads(path.read_bytes())
    assert checkpoint["publication_is_not_complete_transport"] is True
    assert checkpoint["status"] == "SOURCE_PIXEL_INPUT_BYTES_VERIFIED"
    assert checkpoint["raw_fits_sha256"] == "a" * 64
    assert all(checkpoint[flag] is False for flag in transport.FLAGS)
    assert result["dependencies"][str(path)] == sha(path.read_bytes())


@pytest.mark.parametrize("change", ["unknown-dependency", "removed-dependency", "rebound-manifest"])
def test_rebound_cached_receipt_cannot_replace_closed_handoff(case, change):
    run(case)
    receipt_path = case.root / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    if change == "unknown-dependency":
        receipt["dependencies"][str(case.root.parent / "do-not-open")] = "a" * 64
    elif change == "removed-dependency":
        receipt["dependencies"].pop(str(case.prep))
    else:
        path = Path(receipt["manifests"]["periodic_planet"])
        manifest = json.loads(path.read_bytes())
        manifest["sources"][SID]["pixels"]["1"]["target_inside_stamp"] = True
        write(path, manifest)
        receipt["dependencies"][str(path)] = sha(path.read_bytes())
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="dependency inventory|bound dependencies"):
        transport.verified_pixels(case.prep, corpus_root=case.corpus, opened_cache=case.cache,
            destination=case.root, specification=case.spec)
    withdrawn(case)


def test_recipe_archive_orphan_bytes_are_rejected(case):
    decoded = gzip.decompress(case.archive.read_bytes())
    case.archive.write_bytes(gzip.compress(decoded + b"orphan" + bytes(506), mtime=0))
    case.data["archive"].update(transport.identity(case.archive))
    with pytest.raises(ValueError, match="orphan bytes"):
        run(case)
    withdrawn(case)


@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b""])
def test_truth_reference_keeps_original_eol_and_clock_contracts(tmp_path, ending):
    row = {"source_id": SID, "clock": "UNIT_FIXTURE_ONLY"}
    line = json.dumps(row).encode()
    payload = line + ending
    path = tmp_path / "original-truth.jsonl"
    path.write_bytes(payload)
    source = {"opened_cohort": "periodic_planet", "original_source": {"truth_index": 0},
              "truth": {"original_truth_line_index": 0, "original_truth_line_sha256": sha(line),
                        "original_truth_clock_payload_sha256": transport.canonical_hash(row)}}
    reference = {"truth_index": 0, "cohort": "periodic_planet", "path": str(path),
                 "file_sha256": sha(payload), "line_sha256": sha(payload)}
    transport.truth_reference(reference, source, {"truth": {"sha256": sha(payload)}})
    assert path.read_bytes() == payload
    if ending:
        reference["line_sha256"] = sha(line)
        with pytest.raises(ValueError, match="line bytes, line ending or clock"):
            transport.truth_reference(reference, source, {"truth": {"sha256": sha(payload)}})


def test_truth_reference_does_not_accept_rehashed_clock_change(tmp_path):
    row = {"source_id": SID, "clock": "ORIGINAL_UNIT_FIXTURE"}
    original = json.dumps(row).encode()
    changed = json.dumps(dict(row, clock="changed")).encode()
    path = tmp_path / "original-truth.jsonl"
    path.write_bytes(changed + b"\n")
    source = {"opened_cohort": "periodic_planet", "original_source": {"truth_index": 0},
              "truth": {"original_truth_line_index": 0, "original_truth_line_sha256": sha(original),
                        "original_truth_clock_payload_sha256": transport.canonical_hash(row)}}
    reference = {"truth_index": 0, "cohort": "periodic_planet", "path": str(path),
                 "file_sha256": sha(changed + b"\n"), "line_sha256": sha(changed + b"\n")}
    with pytest.raises(ValueError, match="line bytes, line ending or clock"):
        transport.truth_reference(reference, source, {"truth": {"sha256": reference["file_sha256"]}})
