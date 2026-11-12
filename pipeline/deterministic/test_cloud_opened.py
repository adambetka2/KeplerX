"""Bounded opened replay preserves exact readers, recipes and failure boundaries."""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deterministic import cloud_opened as opened
from deterministic.catalogue_controls import product_url


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def inventory(arrays):
    return {name: {"dtype": str(array.dtype), "shape": list(array.shape),
                   "bytes": array.nbytes, "sha256": sha(array.tobytes())}
            for name, array in arrays.items()}


def make_fits(path, sid, kic, parent):
    primary = fits.PrimaryHDU()
    primary.header.update({"GAIAID": f"Gaia DR3 {sid}", "KEPLERID": kic, "TPFORG": parent,
                           "HLSPID": "KBONUS-BKG", "HLSPVER": "V1.0", "RA_OBJ": 12.3,
                           "DEC_OBJ": -4.5, "GMAG": 13.0})

    def table(columns, name):
        result = []
        for column in columns.values():
            integer = column in {"CADENCENO", "QUARTER", "SAP_QUALITY", "FLATTEN_MASK"}
            values = np.full(200, 1 if column == "QUARTER" else 0, dtype=np.int32) if integer else np.full(200, 1.0 if "ERR" in column else 1000.0)
            if column == "TIME":
                values = np.arange(200, dtype=float) * 0.02
            if column == "CADENCENO":
                values = np.arange(200, dtype=np.int32)
            result.append(fits.Column(name=column, format="J" if integer else "D", array=values))
        hdu = fits.BinTableHDU.from_columns(result, name=name)
        hdu.header.update({"QUARTER": 1, "MODULE": 2, "OUTPUT": 1, "CHANNEL": 1})
        return hdu

    fits.HDUList([primary, table(opened.STITCHED_COLUMNS, "LIGHTCURVE_STITCHED"),
                  table(opened.public.DIAGNOSTIC_COLUMNS, "LIGHTCURVE_Q1")]).writeto(path)


def checks(raw):
    return {"raw_view_sha256": opened.public.curve_digest(opened.public.photometry(raw, "psf_flat")),
            "prepared_records_sha256": sha(opened.public.prepare_records(raw).tobytes())}


def write_packet(case):
    data = case.data
    payload = opened.serialized(data)
    zipped = gzip.compress(payload, mtime=0)
    case.metadata.write_bytes(zipped)
    shared = ("schema", "scope", "role", *opened.FLAGS, "original_opened876_selection_unchanged",
              "counts", "budgets", "reader_programs", "preparation_profiles", "scope_metadata", "joint_scope_decoded_sha256", "boundary")
    descriptor = {key: data[key] for key in shared}
    descriptor.update({"status": "CLOUD_OPENED876_ARRAY_REPLAY_PACKET",
                       "artifact": {"path": case.metadata.name, "bytes": len(zipped), "sha256": sha(zipped),
                                    "decoded_bytes": len(payload), "decoded_sha256": sha(payload)},
                       "deltas": data["deltas"]["archive"]})
    case.spec.write_bytes(opened.serialized(descriptor))


def make_source(tmp_path, sid, cohort_name, kic, parent):
    path = tmp_path / f"original-{sid}.fits"
    make_fits(path, sid, kic, parent)
    raw_payloads = {sid: path.read_bytes()}
    delta_payloads = {}
    metadata = {"source_id": sid, "kic": kic, "origin_tpf_kic": parent, "shard": 0,
                "ra_deg": 12.3, "dec_deg": -4.5, "phot_g_mean_mag": 13.0,
                "raw_fits_sha256": sha(raw_payloads[sid]), "shard_path": opened.public.product_uri(kic)}
    row = {"metadata": metadata, "product_kic": kic, "url": opened.public.checked_url({"product_kic": kic, "metadata": metadata,
            "url": product_url(kic)}),
           "bytes": len(raw_payloads[sid]), "sha256": sha(raw_payloads[sid]),
           "reader": "catalogue" if cohort_name == "native_eb" else "lc2", "opened_cohort": cohort_name,
           "opened_scope": "CURRENT_OPENED_DEVELOPMENT", "role": "development", "original_rank": 1,
           "parents": {"origin": str(parent), "resolved": [], "additional": []},
           "origin": {"source_domain": "KBonus_LC1" if cohort_name == "native_eb" else "KBonus_LC2"},
           "local_canonical_proof_state": "LOCAL_EXACT", "raw_header": {key: fits.getheader(path)[key] for key in ("GAIAID", "KEPLERID", "TPFORG", "HLSPID", "HLSPVER")}}
    raw = opened.read_product(path, kic)[1] if cohort_name == "native_eb" else opened.public.fits_arrays(path, sid, row)
    row.update({"base_arrays": inventory(raw), "base_checks": checks(raw)})
    if cohort_name == "periodic_planet":
        values = {name: np.zeros(200, dtype=np.float32) for name in opened.DELTA_FLUX.values()}
        for value in values.values():
            value[50:60] = -0.1
        stream = io.BytesIO()
        np.savez_compressed(stream, **values)
        delta_payloads[f"{sid}.npz"] = stream.getvalue()
        row["delta"] = {"path": f"{sid}.npz", "bytes": len(stream.getvalue()), "sha256": sha(stream.getvalue()),
                        "transform": "additive_flux_delta", "arrays": inventory(values)}
        for name, value in values.items():
            raw[name] = np.asarray(raw[name], dtype=np.float64) + value
    row.update({"arrays": inventory(raw), **checks(raw), "prepared_cadences": 200, "original_prepared_array_bytes": 1000})
    return row, path.read_bytes(), delta_payloads


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "repo"
    root.mkdir()
    monkeypatch.setattr(opened, "ROOT", root)
    monkeypatch.setattr(opened, "EXTERNAL_DUMPS", ())
    monkeypatch.setattr(opened, "dump_path", lambda path: Path(path).resolve())
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    monkeypatch.setattr(opened.shutil, "disk_usage", lambda path: SimpleNamespace(free=24_000_000_000, used=0, total=32_000_000_000))
    profiles = [{"name": "original-main", "files": [{"path": "pipeline/deterministic/run.py", "bytes": 4, "sha256": sha(b"main")}]},
                {"name": "coordinated-default-preparation", "files": [{"path": "pipeline/deterministic/run.py", "bytes": 5, "sha256": sha(b"coord")},
                    {"path": "pipeline/deterministic/signal_contracts.py", "bytes": 8, "sha256": sha(b"relative")}]}]
    monkeypatch.setattr(opened, "PREPARATION_PROFILES", profiles)
    (root / "pipeline/deterministic").mkdir(parents=True)
    (root / "pipeline/deterministic/run.py").write_bytes(b"main")
    programs = []
    for name in ("pipeline/deterministic/cloud_public.py", "pipeline/deterministic/catalogue_controls.py", "pipeline/photometry/lc2.py"):
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        programs.append({"path": name, **opened.identity(path)})
    scope_metadata = []
    for name in ("pipeline/deterministic/cloud-joint-scope.json", "pipeline/deterministic/cloud-joint-scope.json.gz"):
        path = root / name
        path.write_bytes(name.encode())
        scope_metadata.append({"path": name, **opened.identity(path)})
    monkeypatch.setattr(opened, "SCOPE_METADATA", scope_metadata)
    sources, raw_payloads, delta_payloads = {}, {}, {}
    for sid, cohort_name, kic, parent in (("123", "ordinary", 100, 99), ("124", "native_eb", 101, 101), ("125", "periodic_planet", 102, 98)):
        row, raw_payload, deltas = make_source(tmp_path, sid, cohort_name, kic, parent)
        sources[sid] = row
        raw_payloads[sid] = raw_payload
        delta_payloads.update(deltas)
    archive = tmp_path / "deltas.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, payload in delta_payloads.items():
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            bundle.addfile(item, io.BytesIO(payload))
    unavailable = {sid: {"source_id": sid, "state": state, "cohort": c, "role": "development",
                        "opened_scope": "CURRENT_OPENED_DEVELOPMENT", "raw": {"sha256": "a" * 64}}
                   for sid, state, c in (("126", "PROTECTED_LINK_METADATA_ONLY", "ordinary"),
                        ("127", "UNAVAILABLE_ORIGINAL_BACKGROUND_RECIPE", "background"))}
    data = {"schema": 1, "status": "FIXED_OPENED876_PUBLIC_ARRAY_REPLAY_SPECIFICATION",
            "scope": opened.public.SCOPE, "role": "development", **{flag: False for flag in opened.FLAGS},
            "original_opened876_selection_unchanged": True, "sources": sources, "unavailable": unavailable,
            "original_order": ["123", "124", "125", "126", "127"], "supported_order": ["123", "124", "125"],
            "deltas": {"archive": {"path": archive.name, **opened.identity(archive)},
                       "files": {name: {"bytes": len(payload), "sha256": sha(payload)} for name, payload in delta_payloads.items()}},
            "reader_programs": programs, "preparation_profiles": profiles,
            "scope_metadata": scope_metadata, "joint_scope_decoded_sha256": opened.JOINT_SCOPE_DECODED_SHA256,
            "actual_header_parent_edges": [], "boundary": "Fixed opened development only"}
    data["counts"] = opened.metadata_counts(data)
    monkeypatch.setattr(opened, "COUNTS", copy.deepcopy(data["counts"]))
    data["budgets"] = {"raw_fits_bytes": sum(row["bytes"] for row in sources.values()),
                       "original_compressed_array_bytes": 3000,
                       "array_uncompressed_bytes": sum(item["bytes"] for row in sources.values() for item in row["arrays"].values()),
                       "fsc_forecast_bytes": sum(opened.preparation_bytes(row) for row in sources.values()), "paired_run_headroom_bytes": 4 * 1024**3,
                       "max_archive_bytes": opened.LIMIT, "max_cloud_storage_bytes": 32_000_000_000,
                       "external_minimum_free_bytes": 20 * 1024**3, "cloud_development_minimum_free_bytes": 2 * 1024**3}
    case = SimpleNamespace(data=data, spec=tmp_path / "packet.json", metadata=tmp_path / "metadata.json.gz",
                           out=tmp_path / "cloud-opened", raw_payloads=raw_payloads, archive=archive)
    write_packet(case)

    class Response:
        status_code = 200

        def __init__(self, row):
            self.url = row["url"]
            self.payload = raw_payloads[row["metadata"]["source_id"]]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, size):
            yield self.payload

    monkeypatch.setattr(opened.public.requests, "get", lambda url, **kwargs: Response(next(row for row in sources.values() if row["url"] == url)))
    return case


def run(case, **kwargs):
    return opened.provision("1st", case.spec, case.out, **kwargs)


def test_exact_original21_29_and_float32_recipe_replay(cohort):
    receipt = run(cohort)
    assert receipt["status"] == "OPENED_PUBLIC_INPUTS_VERIFIED"
    assert receipt["provisioned_sources"] == 3
    assert receipt["selected_preparation_profile"]["name"] == "original-main"
    manifest = json.loads((cohort.out / "input-manifest.json").read_bytes())
    assert len(opened.npz_values(cohort.out / "124.npz", cohort.data["sources"]["124"]["arrays"])) == 21
    assert len(opened.npz_values(cohort.out / "125.npz", cohort.data["sources"]["125"]["arrays"])) == 29
    assert manifest["joint_independence_complete"] is False
    assert receipt["metrics"]["network_bytes"] == sum(len(value) for value in cohort.raw_payloads.values())


def test_verified_cache_replay_has_no_downloads(cohort, monkeypatch):
    first = run(cohort)
    monkeypatch.setattr(opened.public.requests, "get", lambda *args, **kwargs: pytest.fail("Unexpected cache network request"))
    second = run(cohort)
    assert second["metrics"]["network_bytes"] == 0
    assert second["metrics"]["array_cache_hits"] == second["metrics"]["raw_cache_hits"] == 3
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_missing_local_base_oracle_still_requires_full_final_oracle(cohort):
    row = cohort.data["sources"]["125"]
    row["base_arrays"] = row["base_checks"] = None
    cohort.data["counts"] = opened.metadata_counts(cohort.data)
    opened.COUNTS.update(cohort.data["counts"])
    write_packet(cohort)
    assert run(cohort)["provisioning_complete"] is True


def test_source_limit_retains_full_denominator_and_partial_status(cohort):
    receipt = run(cohort, source_limit=1)
    assert receipt["status"] == "OPENED_PUBLIC_INPUTS_PARTIAL_VERIFIED"
    assert receipt["provisioning_complete"] is False
    assert receipt["original_counts"]["original"] == 5
    assert json.loads((cohort.out / "source-ids.json").read_bytes()) == ["123"]


def test_dry_run_writes_nothing(cohort, monkeypatch):
    monkeypatch.setattr(opened.public.requests, "get", lambda *args, **kwargs: pytest.fail("Dry-run download"))
    receipt = run(cohort, dry_run=True)
    assert receipt["status"] == "OPENED_METADATA_DRY_RUN"
    assert not cohort.out.exists()


@pytest.mark.parametrize("field", opened.FLAGS)
def test_scientific_flags_cannot_be_enabled(cohort, field):
    run(cohort)
    cohort.data[field] = True
    write_packet(cohort)
    with pytest.raises(ValueError, match="boundaries"):
        run(cohort)
    assert not (cohort.out / "receipt.json").exists()
    assert not (cohort.out / "input-manifest.json").exists()


@pytest.mark.parametrize("field", ["source_order", "denominator", "dtype", "shape", "oracle", "role", "header_parent"])
def test_changed_source_contract_rejects_and_withdraws_metadata(cohort, field):
    run(cohort)
    row = cohort.data["sources"]["123"]
    if field == "source_order":
        cohort.data["supported_order"].reverse()
    elif field == "denominator":
        cohort.data["counts"]["supported"] = True
    elif field == "dtype":
        row["arrays"]["quarter"]["dtype"] = "object"
    elif field == "shape":
        row["arrays"]["time_bkjd"]["shape"] = [2_000_001]
    elif field == "oracle":
        row["arrays"]["quarter"]["sha256"] = "0" * 64
    elif field == "role":
        row["role"] = "evaluation"
    else:
        row["raw_header"]["TPFORG"] += 1
    write_packet(cohort)
    with pytest.raises(ValueError):
        run(cohort)
    assert all(not (cohort.out / name).exists() for name in opened.OUTPUT_NAMES)


@pytest.mark.parametrize("name", ["receipt.json", "input-manifest.json", "source-ids.json", "123.fits", "deltas/125.npz"])
def test_previous_success_symlink_failure_removes_usable_outputs(cohort, name):
    run(cohort)
    path = cohort.out / name
    outside = cohort.out.parent / "outside"
    outside.write_bytes(b"untouched")
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="redirect|symlink"):
        run(cohort)
    assert outside.read_bytes() == b"untouched"
    assert not (cohort.out / "input-manifest.json").exists() or (cohort.out / "input-manifest.json").is_symlink()
    assert not (cohort.out / "receipt.json").exists() or (cohort.out / "receipt.json").is_symlink()


def test_stale_partial_requires_explicit_removal(cohort):
    run(cohort)
    (cohort.out / "123.fits.part").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="Stale"):
        run(cohort)
    assert not (cohort.out / "receipt.json").exists()


@pytest.mark.parametrize("version", ["main", "coord", "unknown"])
def test_closed_preparation_versions(cohort, version):
    path = opened.ROOT / "pipeline/deterministic/run.py"
    path.write_bytes(version.encode())
    signal = opened.ROOT / "pipeline/deterministic/signal_contracts.py"
    signal.write_bytes(b"relative")
    if version == "unknown":
        with pytest.raises(ValueError, match="exactly one"):
            run(cohort)
    else:
        expected = "original-main" if version == "main" else "coordinated-default-preparation"
        assert run(cohort)["selected_preparation_profile"]["name"] == expected


@pytest.mark.parametrize("name", opened.OUTPUT_NAMES)
def test_output_drift_cannot_be_adopted(cohort, monkeypatch, name):
    original = opened.publish

    def mutate(root, path, payload):
        result = original(root, path, payload)
        if path.name == name:
            path.chmod(0o644)
            path.write_bytes(payload + b" ")
        return result

    monkeypatch.setattr(opened, "publish", mutate)
    with pytest.raises(ValueError, match="changed|drifted"):
        run(cohort)
    assert all(not (cohort.out / output).exists() for output in opened.OUTPUT_NAMES)


def test_preparation_drift_during_verification_withdraws_outputs(cohort, monkeypatch):
    original = opened.reconstruct

    def mutate(*args):
        raw = original(*args)
        (opened.ROOT / "pipeline/deterministic/run.py").write_bytes(b"changed")
        return raw

    monkeypatch.setattr(opened, "reconstruct", mutate)
    with pytest.raises(ValueError, match="dependency changed"):
        run(cohort)
    assert not (cohort.out / "receipt.json").exists()


def test_cloud32gb_forecast_includes_existing_filesystem_use(cohort, monkeypatch):
    monkeypatch.setattr(opened.shutil, "disk_usage", lambda path: SimpleNamespace(free=24_000_000_000, used=30_000_000_000, total=64_000_000_000))
    with pytest.raises(ValueError, match="declared32GB"):
        run(cohort)


def test_local_external_reserve_stays20gib(cohort, monkeypatch):
    monkeypatch.setattr(opened, "EXTERNAL_DUMPS", (cohort.out.parent,))
    monkeypatch.setattr(opened.shutil, "disk_usage", lambda path: SimpleNamespace(free=22 * 1024**3, used=0, total=64_000_000_000))
    with pytest.raises(ValueError, match="capacity"):
        run(cohort)


def test_output_traversal_is_rejected_without_outside_writes(cohort):
    cohort.out.mkdir()
    outside = cohort.out.parent / "outside"
    with pytest.raises(ValueError, match="escapes"):
        opened.publish(cohort.out, cohort.out / ".." / "outside", b"bad")
    assert not outside.exists()


def test_verified_preparation_handoff_keeps_original_oracles_and_scope(cohort):
    run(cohort)
    manifest, rows, dependencies = opened.verified_inputs(cohort.spec, cohort.out)
    assert list(rows) == list(manifest["sources"]) == ["123", "124", "125"]
    assert len(dependencies) == 20
    assert manifest["pixel_readiness_complete"] is False


def rebind_handoff(case, manifest, receipt):
    path = case.out / "input-manifest.json"
    path.chmod(0o644)
    path.write_bytes(opened.serialized(manifest))
    receipt["manifest_sha256"] = opened.digest(path)
    receipt["dependencies"][str(path)] = opened.digest(path)
    path = case.out / "receipt.json"
    path.chmod(0o644)
    path.write_bytes(opened.serialized(receipt))


@pytest.mark.parametrize("change", ["qdiag", "mask", "dtype"])
def test_rehashed_corrupted_array_container_rejects_at_handoff(cohort, change):
    receipt = run(cohort)
    path = cohort.out / "123.npz"
    values = opened.npz_values(path, cohort.data["sources"]["123"]["arrays"])
    key = "qdiag_quarter" if change == "qdiag" else "flatten_mask"
    values[key] = values[key].astype(np.float32) if change == "dtype" else values[key] + 1
    path.chmod(0o644)
    np.savez_compressed(path, **values)
    manifest = json.loads((cohort.out / "input-manifest.json").read_bytes())
    manifest["sources"]["123"]["sha256"] = opened.digest(path)
    receipt["dependencies"][str(path)] = opened.digest(path)
    rebind_handoff(cohort, manifest, receipt)
    with pytest.raises(ValueError, match="dtype|inventory|hash|dimensions"):
        opened.verified_inputs(cohort.spec, cohort.out)
    assert all(not (cohort.out / name).exists() for name in opened.OUTPUT_NAMES)


@pytest.mark.parametrize("change", ["scope", "dependency", "source", "config", "cohort"])
def test_rebound_handoff_metadata_cannot_weaken_bindings(cohort, change):
    receipt = run(cohort)
    manifest = json.loads((cohort.out / "input-manifest.json").read_bytes())
    if change == "scope":
        receipt["scientific_acceptance"] = True
    elif change == "dependency":
        receipt["dependencies"].pop(str(cohort.out / "123.fits"))
    elif change == "source":
        manifest["sources"]["123"]["arrays"] = "../outside.npz"
    elif change == "config":
        receipt["config"]["frozen_pixel_inputs"] = "public-ledger.json"
    else:
        manifest["sources"]["123"]["opened_cohort"] = "native_eb"
    rebind_handoff(cohort, manifest, receipt)
    with pytest.raises(ValueError):
        opened.verified_inputs(cohort.spec, cohort.out)
    assert not (cohort.out / "receipt.json").exists()


@pytest.mark.parametrize("name", ["cloud-joint-scope.json", "cloud-joint-scope.json.gz"])
def test_stale_parent_overlay_stops_before_download(cohort, monkeypatch, name):
    run(cohort)
    (opened.ROOT / "pipeline/deterministic" / name).write_bytes(b"stale")
    monkeypatch.setattr(opened.public.requests, "get", lambda *args, **kwargs: pytest.fail("Stale overlay download"))
    with pytest.raises(ValueError, match="wrong-sized|changed"):
        run(cohort)
    assert not (cohort.out / "receipt.json").exists()


@pytest.mark.parametrize("name", ["cloud-joint-scope.json", "cloud-joint-scope.json.gz"])
def test_parent_overlay_drift_withdraws_success(cohort, monkeypatch, name):
    original = opened.reconstruct

    def mutate(*args):
        raw = original(*args)
        (opened.ROOT / "pipeline/deterministic" / name).write_bytes(b"drifted")
        return raw

    monkeypatch.setattr(opened, "reconstruct", mutate)
    with pytest.raises(ValueError, match="dependency changed"):
        run(cohort)
    assert not (cohort.out / "input-manifest.json").exists()


def test_intermediate_destination_symlink_cannot_escape(cohort):
    folder = cohort.out.parent / "redirect"
    folder.symlink_to(cohort.out.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        opened.provision("1st", cohort.spec, folder / "elsewhere")
    assert not (cohort.out.parent / "elsewhere").exists()


def test_unknown_cache_file_rejects_and_withdraws_metadata(cohort):
    run(cohort)
    (cohort.out / "unknown.npz").write_bytes(b"kept")
    with pytest.raises(ValueError, match="unknown"):
        run(cohort)
    assert not (cohort.out / "receipt.json").exists()
    assert (cohort.out / "unknown.npz").read_bytes() == b"kept"


@pytest.mark.parametrize("change", ["unknown", "doubled", "traversal", "directory", "symlink", "missing", "content"])
def test_recipe_archive_inventory_is_closed(cohort, change):
    with tarfile.open(cohort.archive) as archive:
        payload = archive.extractfile("125.npz").read()
    with tarfile.open(cohort.archive, "w:gz") as archive:
        names = ["125.npz", "125.npz"] if change == "doubled" else ["../125.npz" if change == "traversal" else "unknown.npz" if change == "unknown" else "125.npz"]
        for name in [] if change == "missing" else names:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.type = tarfile.DIRTYPE if change == "directory" else tarfile.SYMTYPE if change == "symlink" else tarfile.REGTYPE
            archive.addfile(member, io.BytesIO(payload[:-1] + b"x" if change == "content" else payload))
    cohort.data["deltas"]["archive"].update(opened.identity(cohort.archive))
    write_packet(cohort)
    with pytest.raises(ValueError, match="recipe"):
        run(cohort)
    assert not (cohort.out / "receipt.json").exists()


def test_npy_header_is_checked_before_array_allocation(cohort, monkeypatch):
    row = cohort.data["sources"]["125"]
    path = cohort.out.parent / "bad.npz"
    values = {name: np.zeros(200, dtype=np.float32) for name in opened.DELTA_FLUX.values()}
    values["qdiag_psf_flux"] = np.zeros(201, dtype=np.float32)
    np.savez_compressed(path, **values)
    monkeypatch.setattr(opened.np, "load", lambda *args, **kwargs: pytest.fail("Malformed NPY allocated"))
    with pytest.raises(ValueError, match="dimensions"):
        opened.npz_values(path, row["delta"]["arrays"])


def test_raw_header_change_rejects_before_flux_reader(cohort, monkeypatch):
    row = cohort.data["sources"]["123"]
    path = cohort.out.parent / "original-123.fits"
    with fits.open(path, mode="update") as hdus:
        hdus[0].header["TPFORG"] = 111
    cohort.raw_payloads["123"] = path.read_bytes()
    row["sha256"] = row["metadata"]["raw_fits_sha256"] = sha(path.read_bytes())
    write_packet(cohort)
    monkeypatch.setattr(opened.public, "fits_arrays", lambda *args, **kwargs: pytest.fail("Unexpected raw parent flux access"))
    with pytest.raises(ValueError, match="raw header"):
        run(cohort)
    assert not (cohort.out / "receipt.json").exists()


def test_metadata_decompression_is_exact_and_bounded(cohort):
    descriptor = json.loads(cohort.spec.read_bytes())
    descriptor["artifact"]["decoded_bytes"] = 100
    cohort.spec.write_bytes(opened.serialized(descriptor))
    with pytest.raises(ValueError, match="decompression"):
        run(cohort)


@pytest.mark.parametrize("name", ["packet.json", "metadata.json.gz", "deltas.tar.gz"])
def test_packet_input_drift_rejects_success(cohort, monkeypatch, name):
    original = opened.reconstruct

    def mutate(*args):
        raw = original(*args)
        path = cohort.spec.parent / name
        path.write_bytes(path.read_bytes() + b" ")
        return raw

    monkeypatch.setattr(opened, "reconstruct", mutate)
    with pytest.raises(ValueError, match="dependency changed"):
        run(cohort)
    assert not (cohort.out / "receipt.json").exists()


def qualify_native_parent(case):
    row = case.data["sources"]["124"]
    path = case.spec.parent / "original-124.fits"
    with fits.open(path, mode="update") as hdus:
        hdus[0].header["TPFORG"] = 999
    case.raw_payloads["124"] = path.read_bytes()
    row["sha256"] = row["metadata"]["raw_fits_sha256"] = sha(path.read_bytes())
    row["raw_header"]["TPFORG"] = 999
    row["local_canonical_proof_state"] = "LOCAL_EXACT_PARENT_QUALIFIED"
    case.data["actual_header_parent_edges"] = [{"source_id": "124", "original_reader_parent": "101",
        "raw_fits_sha256": row["sha256"], "actual_raw_header_TPFORG": "999",
        "known_protected_link_found": False, "joint_independence_complete": False,
        "must_enter_future_joint_null_connectivity_before_selection_or_freezing": True}]
    case.data["counts"] = opened.metadata_counts(case.data)
    opened.COUNTS.update(case.data["counts"])
    write_packet(case)


def test_qualified_native_header_never_replaces_original_reader_parent(cohort):
    qualify_native_parent(cohort)
    receipt = run(cohort)
    manifest, rows, _ = opened.verified_inputs(cohort.spec, cohort.out)
    assert rows["124"]["raw_header"]["TPFORG"] == 999
    assert manifest["sources"]["124"]["metadata"]["origin_tpf_kic"] == 101
    assert receipt["joint_independence_complete"] is False


@pytest.mark.parametrize("field", ["original_reader_parent", "raw_fits_sha256", "actual_raw_header_TPFORG", "known_protected_link_found", "joint_independence_complete", "must_enter_future_joint_null_connectivity_before_selection_or_freezing"])
def test_qualified_native_parent_witness_cannot_change(cohort, field):
    qualify_native_parent(cohort)
    edge = cohort.data["actual_header_parent_edges"][0]
    edge[field] = not edge[field] if type(edge[field]) is bool else "wrong"
    write_packet(cohort)
    with pytest.raises(ValueError, match="parent provenance"):
        run(cohort)
    assert not (cohort.out / "receipt.json").exists()


@pytest.fixture
def preparation_case(cohort, monkeypatch):
    import runtime
    from deterministic import cloud

    run(cohort)
    verified = opened.verified_inputs
    packet = opened.packet
    monkeypatch.setattr(opened, "verified_inputs", lambda: verified(cohort.spec, cohort.out))
    monkeypatch.setattr(opened, "packet", lambda specification=cohort.spec: packet(specification))
    monkeypatch.setattr(cloud, "DUMP", cohort.out.parent)
    monkeypatch.setattr(runtime, "dump_path", lambda path: Path(path).resolve())
    return cohort, cloud, cohort.out.parent / "prepared"


def test_preparation_adapter_verifies_actual_original_fsc(preparation_case):
    cohort, cloud, out = preparation_case
    report = cloud.prepare_opened(out)
    assert report["sources"] == 3
    assert report["provisioning_complete"] is True
    assert report["scientific_acceptance"] is False
    assert json.loads((out / "preparation.json").read_bytes())["selection"]["source_ids"] == cohort.data["supported_order"]


@pytest.mark.parametrize("change", ["cohort", "parent", "raw_hash", "cadences", "records", "dependency", "manifest"])
def test_preparation_adapter_withdraws_success_on_drift(preparation_case, monkeypatch, change):
    from deterministic import run as runner

    cohort, cloud, out = preparation_case
    prepare = runner.prepare

    def mutate(args):
        prepare(args)
        path = out / "preparation.json"
        saved = json.loads(path.read_bytes())
        row = saved["sources"][0]
        if change == "cohort":
            saved["selection"]["source_ids"].reverse()
        elif change == "parent":
            row["parent_tpf"] = "999"
        elif change == "raw_hash":
            row["raw_fits_sha256"] = "a" * 64
        elif change == "cadences":
            row["inputs"]["psf_flat"]["retained_cadences"] -= 1
        elif change == "records":
            fsc = Path(row["inputs"]["psf_flat"]["fsc_path"])
            payload = bytearray(fsc.read_bytes())
            payload[-1] ^= 1
            fsc.write_bytes(payload)
            row["inputs"]["psf_flat"]["fsc_sha256"] = opened.digest(fsc)
        elif change == "manifest":
            saved["input_manifest"] = {"path": "/unrelated/manifest.json", "sha256": "0" * 64}
        else:
            (cohort.out / "123.fits").unlink()
        path.write_bytes(opened.serialized(saved))

    monkeypatch.setattr(runner, "prepare", mutate)
    with pytest.raises((ValueError, FileNotFoundError)):
        cloud.prepare_opened(out)
    assert json.loads((out / "preparation.json").read_bytes())["status"] == "PREPARATION_FAILED"


def test_preparation_adapter_requires_fresh_output(preparation_case):
    _, cloud, out = preparation_case
    out.mkdir()
    sentinel = out / "retained"
    sentinel.write_text("history")
    with pytest.raises(ValueError, match="fresh"):
        cloud.prepare_opened(out)
    assert sentinel.read_text() == "history"


@pytest.mark.parametrize("name", ["preparation.json", "inputs/123.psf_flat.fsc", "inputs/123.sap_qdiag.fsc"])
def test_preparation_output_drift_after_oracle_check_rejects(preparation_case, monkeypatch, name):
    _, cloud, out = preparation_case
    checked = cloud.check_opened_preparation

    def mutate(*args):
        dependencies = checked(*args)
        path = out / name
        payload = path.read_bytes()
        path.write_bytes(payload.replace(b'"PREPARED"', b'"REBOUND"') if name.endswith(".json") else payload[:-1] + bytes([payload[-1] ^ 1]))
        return dependencies

    monkeypatch.setattr(cloud, "check_opened_preparation", mutate)
    with pytest.raises(ValueError, match="dependency changed"):
        cloud.prepare_opened(out)
    assert json.loads((out / "preparation.json").read_bytes())["status"] == "PREPARATION_FAILED"


@pytest.mark.parametrize("change", ["reserve", "vm_budget"])
def test_preparation_checks_all_view_capacity_before_writes(preparation_case, monkeypatch, change):
    _, cloud, out = preparation_case
    disk = SimpleNamespace(free=2 * 1024**3 + 4 * 1024**3 + 20_000, used=0, total=32_000_000_000)
    if change == "vm_budget":
        disk.free, disk.used = 24_000_000_000, 30_000_000_000
    monkeypatch.setattr(opened.shutil, "disk_usage", lambda path: disk)
    with pytest.raises(ValueError, match="All-view"):
        cloud.prepare_opened(out)
    assert not out.exists()


def write_truth_packet(case, monkeypatch, members=None, header=None):
    if members is None:
        members = case.truth_payloads.items()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as bundle:
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(payload), 0o444
            if header is not None:
                header(member)
            bundle.addfile(member, io.BytesIO(payload))
    encoded = gzip.compress(raw.getvalue(), mtime=0)
    case.truth_archive.write_bytes(encoded)
    case.truth_data["archive"].update({"path": case.truth_archive.name, "bytes": len(encoded), "sha256": sha(encoded),
                                      "decoded_bytes": len(raw.getvalue()), "decoded_sha256": sha(raw.getvalue())})
    case.truth_spec.write_bytes(opened.serialized(case.truth_data))
    monkeypatch.setattr(opened, "TRUTH_SPECIFICATION_SHA256", sha(case.truth_spec.read_bytes()))


@pytest.fixture
def truth_case(cohort, monkeypatch):
    data = cohort.data
    for sid, name, kic, parent in (("128", "sparse_injection", 103, 97), ("129", "ttv_injection", 104, 96)):
        row, payload, _ = make_source(cohort.spec.parent, sid, "periodic_planet", kic, parent)
        row["opened_cohort"] = name
        data["sources"][sid] = row
        cohort.raw_payloads[sid] = payload
    data["supported_order"] = ["123", "124", "125", "128", "129"]
    data["original_order"] = [*data["supported_order"], "126", "127"]
    recipe_names = [f"{sid}.npz" for sid, row in data["sources"].items() if "delta" in row]
    with tarfile.open(cohort.archive, "r:gz") as archive:
        base = archive.extractfile("125.npz").read()
    with tarfile.open(cohort.archive, "w:gz") as archive:
        for name in recipe_names:
            member = tarfile.TarInfo(name)
            member.size = len(base)
            archive.addfile(member, io.BytesIO(base))
    data["deltas"] = {"archive": {"path": cohort.archive.name, **opened.identity(cohort.archive)},
                      "files": {name: {"bytes": len(base), "sha256": sha(base)} for name in recipe_names}}
    for row in data["sources"].values():
        if "delta" in row:
            row["delta"].update({"bytes": len(base), "sha256": sha(base)})
    data["counts"] = opened.metadata_counts(data)
    monkeypatch.setattr(opened, "COUNTS", copy.deepcopy(data["counts"]))
    data["budgets"].update({"raw_fits_bytes": sum(row["bytes"] for row in data["sources"].values()),
                            "original_compressed_array_bytes": 5000,
                            "array_uncompressed_bytes": sum(item["bytes"] for row in data["sources"].values() for item in row["arrays"].values()),
                            "fsc_forecast_bytes": sum(opened.preparation_bytes(row) for row in data["sources"].values())})
    template = json.loads(opened.TRUTH_SPECIFICATION.read_bytes())
    tiny_cohorts = {name: (value[0], value[1], 1) for name, value in opened.TRUTH_COHORTS.items()}
    monkeypatch.setattr(opened, "TRUTH_COHORTS", tiny_cohorts)
    truth_payloads = {}
    for group, sid in zip(template["cohorts"], ("125", "128", "129"), strict=True):
        row = data["sources"][sid]
        manifest = {"path": str(cohort.spec.parent / group["opened_cohort"] / "input-manifest.json"), "sha256": "a" * 64}
        row["origin"].update({"original_input_manifest": manifest,
                              "original_array": {"path": f"original/{sid}.npz", "bytes": 1000, "sha256": "b" * 64}})
        row["original_cohort"] = group["original_cohort"]
        truth = dict.fromkeys(opened.TRUTH_FIELDS, 1.0)
        truth.update({"source_id": sid, "category": group["category"], "ttv": None,
                      "observed_quarters": [1], "observed_events": 1, "stratum": "tiny-fixture", "pool_rank": 1})
        if sid == "128":
            truth["period_days"] = None
        if sid == "129":
            truth["ttv"] = {"amplitude_days": 0.01, "amplitude_durations": 0.24,
                            "superperiod_cycles": 10.0, "phase_radians": 0.0}
        payload = (json.dumps(truth) + "\n").encode()
        item = {"path": f"{group['opened_cohort']}/truth.jsonl", "bytes": len(payload), "sha256": sha(payload)}
        truth_payloads[item["path"]] = payload
        original_files = [{"path": manifest["path"], "bytes": 100, "sha256": manifest["sha256"]},
                          {"path": str(Path(manifest["path"]).parent / "materialization.json"), "bytes": 100, "sha256": "c" * 64},
                          {"path": str(Path(manifest["path"]).parent / "source-ids.json"), "bytes": 100, "sha256": "d" * 64},
                          {"path": str(Path(manifest["path"]).parent / "truth.jsonl"), "bytes": len(payload), "sha256": sha(payload)}]
        binding = {"source_id": sid, "truth_index": 0, "line_bytes": len(payload), "line_sha256": sha(payload),
                   "opened_cohort": row["opened_cohort"], "role": "development", "reader": "lc2",
                   "reader_parent": str(row["metadata"]["origin_tpf_kic"]),
                   "base_raw": {"bytes": row["bytes"], "sha256": row["sha256"]}, "original_delta": row["delta"],
                   "original_delta_reference": f"deltas/{sid}.npz", "original_injected_array": row["origin"]["original_array"],
                   "raw_view_sha256": row["raw_view_sha256"], "prepared_records_sha256": row["prepared_records_sha256"]}
        group.update({"sources": 1, "source_order": [sid], "source_truth_bindings": [binding],
                      "truth": item, "original_inputs": original_files})
    write_packet(cohort)
    current = json.loads(cohort.spec.read_bytes())
    template["fixed_array_packet"] = {"descriptor": {"path": str(cohort.spec), **opened.identity(cohort.spec)},
                                       **{key: current[key] for key in ("artifact", "deltas", "counts", "reader_programs", "preparation_profiles", "joint_scope_decoded_sha256")}}
    template["protected_metadata_only_sources_excluded"] = ["126"]
    template["members"] = [group["truth"] for group in template["cohorts"]]
    template["counts"] = {"sources": 3, "periodic_planet": 1, "sparse_injection": 1, "ttv_injection": 1,
                           "members": 3, "truth_bytes": sum(map(len, truth_payloads.values())),
                           "bound_original_float32_delta_arrays": 12, "protected_truths_read": 0, "mixed_sealed_truth_files_read": 0}
    cohort.truth_data, cohort.truth_payloads = template, truth_payloads
    cohort.truth_spec = cohort.spec.parent / "truths.json"
    cohort.truth_archive = cohort.spec.parent / "truths.tar.gz"
    write_truth_packet(cohort, monkeypatch)
    run(cohort)
    return cohort


def provision_truth(case, **kwargs):
    return opened.provision_truths("1st", case.spec, case.truth_spec, case.out, **kwargs)


def prepare_truth_case(case):
    import runtime
    from deterministic import run as runner

    out = case.out.parent / "truth-prepared"
    previous = runtime.dump_path
    runtime.dump_path = lambda path: Path(path).resolve()
    try:
        runner.prepare(SimpleNamespace(shards=[0], limit=len(json.loads((case.out / "source-ids.json").read_bytes())),
                       source_ids=case.out / "source-ids.json", input_manifest=case.out / "input-manifest.json", out=out))
    finally:
        runtime.dump_path = previous
    return out


def test_truths_preserve_three_original_files_and_prepared_all_views(truth_case):
    case = truth_case
    receipt = provision_truth(case)
    assert receipt["selected_truth_source_ids"] == ["125", "128", "129"]
    assert receipt["prepared_fsc_verified"] is False and receipt["scientific_acceptance"] is False
    for name, payload in case.truth_payloads.items():
        path = case.out / "truths" / name
        assert path.read_bytes() == payload and path.stat().st_mode & 0o777 == 0o444
    prepared = prepare_truth_case(case)
    refs, dependencies = opened.truth_refs(prepared, case.spec, case.truth_spec, case.out)
    assert list(refs) == ["125", "128", "129"]
    assert len([path for path in dependencies if path.suffix == ".fsc"]) == 20
    assert refs["129"]["truth_index"] == 0
    assert refs["129"]["line_sha256"] == sha(case.truth_payloads["ttv_injection/truth.jsonl"])


@pytest.mark.parametrize("field", ["source_id", "truth_index", "reader_parent", "base_raw", "original_delta", "prepared_records_sha256", "raw_view_sha256", "role", "reader", "opened_cohort"])
def test_truth_binding_mismatch_rejects_even_with_rebound_descriptor(truth_case, monkeypatch, field):
    provision_truth(truth_case)
    binding = truth_case.truth_data["cohorts"][0]["source_truth_bindings"][0]
    binding[field] = 1 if field == "truth_index" else "wrong"
    write_truth_packet(truth_case, monkeypatch)
    with pytest.raises((ValueError, KeyError, TypeError)):
        provision_truth(truth_case)
    assert not (truth_case.out / "truths/receipt.json").exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "extra", "traversal", "reverse"])
def test_truth_archive_requires_closed_exact_three_regular_members(truth_case, monkeypatch, change):
    provision_truth(truth_case)
    members = list(truth_case.truth_payloads.items())
    if change == "missing":
        members.pop()
    elif change == "duplicate":
        members.append(members[0])
    elif change == "extra":
        members.append(("other/truth.jsonl", b"{}\n"))
    elif change == "traversal":
        members[0] = ("../outside", members[0][1])
    else:
        members.reverse()
    write_truth_packet(truth_case, monkeypatch, members)
    with pytest.raises(ValueError, match="archive"):
        provision_truth(truth_case)
    assert not (truth_case.out / "truths/receipt.json").exists()


@pytest.mark.parametrize("target", ["truths", "truths/periodic_planet", "truths/periodic_planet/truth.jsonl"])
def test_truth_symlink_never_changes_external_bytes(truth_case, target):
    case = truth_case
    outside = case.out.parent / "outside"
    outside.mkdir()
    sentinel = outside / "truth.jsonl"
    sentinel.write_bytes(b"retained")
    path = case.out / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(sentinel if path.name == "truth.jsonl" else outside)
    with pytest.raises(ValueError, match="symlink"):
        provision_truth(case)
    assert sentinel.read_bytes() == b"retained"


def test_truth_stale_partial_withdraws_old_receipt(truth_case):
    provision_truth(truth_case)
    (truth_case.out / "truths/periodic_planet/truth.jsonl.part").write_bytes(b"partial")
    with pytest.raises(ValueError, match="stale partial"):
        provision_truth(truth_case)
    assert not (truth_case.out / "truths/receipt.json").exists()


@pytest.mark.parametrize("view", ["psf_flat", "sap_qdiag", "psf_qdiag", "psf_nova"])
def test_truth_refs_reject_rehashed_altered_fsc(truth_case, view):
    case = truth_case
    provision_truth(case)
    prepared = prepare_truth_case(case)
    manifest = prepared / "preparation.json"
    report = json.loads(manifest.read_bytes())
    item = report["sources"][2]["inputs"][view]
    path = Path(item["fsc_path"])
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    item["fsc_sha256"] = opened.digest(path)
    manifest.write_bytes(opened.serialized(report))
    with pytest.raises(ValueError, match="exact prepared records"):
        opened.truth_refs(prepared, case.spec, case.truth_spec, case.out)
    assert not (case.out / "truths/receipt.json").exists()


def test_truth_refs_reject_changed_truth_and_rebound_receipt(truth_case):
    case = truth_case
    provision_truth(case)
    prepared = prepare_truth_case(case)
    path = case.out / "truths/sparse_injection/truth.jsonl"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(b'"duration_hours": 1.0', b'"duration_hours": 2.0'))
    receipt_path = case.out / "truths/receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["dependencies"][str(path)] = opened.digest(path)
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(opened.serialized(receipt))
    with pytest.raises(ValueError, match="bytes changed"):
        opened.truth_refs(prepared, case.spec, case.truth_spec, case.out)
    assert not receipt_path.exists()


def test_truth_prefix_keeps_full_metadata_and_explicit_omitted_counts(truth_case):
    case = truth_case
    run(case, source_limit=3)
    receipt = provision_truth(case)
    assert receipt["array_source_ids"] == ["123", "124", "125"]
    assert receipt["selected_truth_source_ids"] == ["125"]
    assert receipt["omitted_truth_sources"] == 2 and receipt["full180_metadata_verified"] is True
    assert all((case.out / "truths" / name).read_bytes() == payload for name, payload in case.truth_payloads.items())
    refs, _ = opened.truth_refs(prepare_truth_case(case), case.spec, case.truth_spec, case.out)
    assert list(refs) == ["125"]


@pytest.mark.parametrize("change", ["empty", "missing_source", "protected_source", "category", "fields", "generator", "function", "current_packet", "count_type", "original_manifest"])
def test_truth_metadata_cannot_rebind_missing_or_mixed_scientific_identities(truth_case, monkeypatch, change):
    case = truth_case
    provision_truth(case)
    spec = case.truth_data
    group = spec["cohorts"][0]
    if change == "empty":
        spec["cohorts"] = []
    elif change == "missing_source":
        group["source_truth_bindings"] = []
    elif change == "protected_source":
        group["source_order"] = ["126"]
        group["source_truth_bindings"][0]["source_id"] = "126"
    elif change == "category":
        group["category"] = "background"
    elif change == "fields":
        group["original_truth_fields"].remove("epoch_bkjd")
    elif change == "generator":
        group["original_manifest_generator"]["pipeline/deterministic/injections.py"] = "e" * 64
    elif change == "function":
        spec["original_clock_generation"][0]["functions"]["truth_centres"]["source"] = "raise RuntimeError('never execute')"
    elif change == "current_packet":
        spec["fixed_array_packet"]["artifact"]["decoded_sha256"] = "e" * 64
    elif change == "count_type":
        spec["counts"]["protected_truths_read"] = False
    else:
        group["original_inputs"][0]["sha256"] = "e" * 64
    write_truth_packet(case, monkeypatch)
    with pytest.raises((ValueError, KeyError, TypeError)):
        provision_truth(case)
    assert not (case.out / "truths/receipt.json").exists()


@pytest.mark.parametrize("change", ["scope", "counts", "prefix", "implementation", "contract", "parent", "raw", "missing_view", "view_identity", "view_counts"])
def test_truth_refs_require_exact_preparation_and_source_accounting(truth_case, change):
    case = truth_case
    provision_truth(case)
    prepared = prepare_truth_case(case)
    path = prepared / "preparation.json"
    report = json.loads(path.read_bytes())
    if change == "scope":
        report["evaluation_scope"] = "sealed"
    elif change == "counts":
        report["preparation_counts"]["accepted_sources"] -= 1
    elif change == "prefix":
        report["selection"]["source_ids"].reverse()
    elif change == "implementation":
        report["implementation"] = {}
    elif change == "contract":
        report["detector_contract"]["discovery_view"] = "sap_qdiag"
    elif change == "parent":
        report["sources"][2]["parent_tpf"] = "999"
    elif change == "raw":
        report["sources"][2]["raw_fits_sha256"] = "e" * 64
    elif change == "missing_view":
        del report["sources"][2]["inputs"]["sap_qdiag"]
    elif change == "view_counts":
        report["preparation_counts"]["prepared_views"] += 1
    else:
        report["sources"][2]["inputs"]["sap_qdiag"]["view"] = "psf_qdiag"
    path.write_bytes(opened.serialized(report))
    with pytest.raises(ValueError):
        opened.truth_refs(prepared, case.spec, case.truth_spec, case.out)
    assert not (case.out / "truths/receipt.json").exists()


@pytest.mark.parametrize("change", ["scope", "whitespace", "dependency", "section"])
def test_truth_refs_require_exact_closed_receipt(truth_case, change):
    case = truth_case
    provision_truth(case)
    prepared = prepare_truth_case(case)
    path = case.out / "truths/receipt.json"
    payload = path.read_bytes()
    report = json.loads(payload)
    if change == "scope":
        report["scientific_acceptance"] = True
    elif change == "dependency":
        report["dependencies"].pop(next(iter(report["dependencies"])))
    elif change == "section":
        report["section"] = "third"
    path.chmod(0o644)
    path.write_bytes(payload + b" " if change == "whitespace" else opened.serialized(report))
    with pytest.raises(ValueError, match="receipt"):
        opened.truth_refs(prepared, case.spec, case.truth_spec, case.out)
    assert not path.exists()


@pytest.mark.parametrize("change", ["descriptor", "archive", "truth", "receipt"])
def test_truth_late_dependency_drift_withdraws_published_success(truth_case, monkeypatch, change):
    case = truth_case
    original = opened.publish

    def mutate(root, path, payload):
        expected = original(root, path, payload)
        if path == case.out / "truths/receipt.json":
            target = {"descriptor": case.truth_spec, "archive": case.truth_archive,
                      "truth": case.out / "truths/periodic_planet/truth.jsonl", "receipt": path}[change]
            target.chmod(0o644)
            target.write_bytes(target.read_bytes() + b" ")
        return expected

    monkeypatch.setattr(opened, "publish", mutate)
    with pytest.raises(ValueError, match="dependency changed"):
        provision_truth(case)
    assert not (case.out / "truths/receipt.json").exists()


def test_truth_descriptor_is_closed(truth_case, monkeypatch):
    truth_case.truth_data["unknown_fallback"] = True
    write_truth_packet(truth_case, monkeypatch)
    with pytest.raises(ValueError, match="schema"):
        provision_truth(truth_case)


@pytest.mark.parametrize("field,value", [("type", tarfile.SYMTYPE), ("type", tarfile.LNKTYPE),
                                         ("mode", 0o644), ("gid", 1), ("uid", 1), ("mtime", 1),
                                         ("uname", "user"), ("gname", "group"), ("linkname", "outside")])
def test_truth_archive_rejects_noncanonical_header(truth_case, monkeypatch, field, value):
    provision_truth(truth_case)
    write_truth_packet(truth_case, monkeypatch, header=lambda member: setattr(member, field, value))
    with pytest.raises(ValueError, match="header"):
        provision_truth(truth_case)
    assert not (truth_case.out / "truths/receipt.json").exists()


@pytest.mark.parametrize("field", ["decoded_bytes", "decoded_sha256", "sha256"])
def test_truth_archive_rejects_wrong_bounded_decode_or_hash(truth_case, monkeypatch, field):
    provision_truth(truth_case)
    archive = truth_case.truth_data["archive"]
    archive[field] = 10 if field == "decoded_bytes" else "e" * 64
    truth_case.truth_spec.write_bytes(opened.serialized(truth_case.truth_data))
    monkeypatch.setattr(opened, "TRUTH_SPECIFICATION_SHA256", sha(truth_case.truth_spec.read_bytes()))
    with pytest.raises(ValueError):
        provision_truth(truth_case)
    assert not (truth_case.out / "truths/receipt.json").exists()


def test_truth_provision_keeps_external_twenty_gib_reserve(truth_case, monkeypatch):
    provision_truth(truth_case)
    monkeypatch.setattr(opened, "EXTERNAL_DUMPS", (truth_case.out,))
    monkeypatch.setattr(opened.shutil, "disk_usage", lambda path: SimpleNamespace(free=19 * 1024**3))
    with pytest.raises(ValueError, match="capacity"):
        provision_truth(truth_case)
    assert not (truth_case.out / "truths/receipt.json").exists()


@pytest.mark.parametrize("target", ["fsc", "preparation", "array", "truth"])
def test_truth_refs_late_drift_never_returns_usable_refs(truth_case, monkeypatch, target):
    case = truth_case
    provision_truth(case)
    prepared = prepare_truth_case(case)
    validate = opened.truth_preparation

    def mutate(*args):
        dependencies = validate(*args)
        path = {"fsc": prepared / "inputs/125.psf_flat.fsc",
                "preparation": prepared / "preparation.json", "array": case.out / "125.npz",
                "truth": case.out / "truths/periodic_planet/truth.jsonl"}[target]
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b" ")
        return dependencies

    monkeypatch.setattr(opened, "truth_preparation", mutate)
    with pytest.raises(ValueError, match="dependency changed"):
        opened.truth_refs(prepared, case.spec, case.truth_spec, case.out)
    assert not (case.out / "truths/receipt.json").exists()


def test_truth_refs_rechecks_canonical_array_inventory_after_rebound_cache(truth_case):
    case = truth_case
    provision_truth(case)
    prepared = prepare_truth_case(case)
    path = case.out / "125.npz"
    with np.load(path) as data:
        raw = {name: data[name] for name in data.files}
    raw["psf_flat_kbonus_flux"][50] += 0.01
    path.chmod(0o644)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **raw)
    manifest_path = case.out / "input-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["sources"]["125"]["sha256"] = opened.digest(path)
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(opened.serialized(manifest))
    receipt_path = case.out / "receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["manifest_sha256"] = opened.digest(manifest_path)
    receipt["dependencies"].update({str(path): opened.digest(path), str(manifest_path): opened.digest(manifest_path)})
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(opened.serialized(receipt))
    with pytest.raises(ValueError, match="content differs"):
        opened.truth_refs(prepared, case.spec, case.truth_spec, case.out)
    assert not (case.out / "truths/receipt.json").exists()


def test_truth_refs_accept_original_repository_dump_symlink(truth_case):
    case = truth_case
    provision_truth(case)
    prepared = prepare_truth_case(case)
    (opened.ROOT / "dump").symlink_to(case.out.parent, target_is_directory=True)
    refs, _ = opened.truth_refs(opened.ROOT / "dump" / prepared.name, case.spec, case.truth_spec, case.out)
    assert list(refs) == ["125", "128", "129"]


@pytest.mark.parametrize("failure", [False, True])
def test_array_refresh_withdraws_truth_success_without_deleting_originals(truth_case, failure):
    case = truth_case
    provision_truth(case)
    files = {case.out / "truths" / name: payload for name, payload in case.truth_payloads.items()}
    if failure:
        case.data["sources"]["123"]["role"] = "evaluation"
        write_packet(case)
        with pytest.raises(ValueError, match="role"):
            run(case)
    else:
        run(case)
    assert not (case.out / "truths/receipt.json").exists()
    assert all(path.read_bytes() == payload for path, payload in files.items())


def test_actual_fixed_split_exact_order_union_and_cold_capacity(tmp_path, monkeypatch):
    descriptor = opened.parse(opened.SPECIFICATION.read_bytes())
    metadata, _ = opened.decoded_metadata(opened.SPECIFICATION, descriptor)
    if any(opened.digest(opened.ROOT / item["path"]) != item["sha256"] for item in metadata["reader_programs"]):
        with pytest.raises(ValueError, match="wrong-sized opened input|Opened input bytes changed"):
            opened.packet()
        return
    data, _, _ = opened.packet()
    original = opened.serialized(data)
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    monkeypatch.setattr(opened, "EXTERNAL_DUMPS", ())
    monkeypatch.setattr(opened.shutil, "disk_usage", lambda path: SimpleNamespace(free=32_000_000_000, used=0, total=32_000_000_000))
    groups = [(["periodic_planet", "sparse_injection", "ttv_injection"], 180, 8_323_844_038, 751_782_268),
              (["ordinary", "native_eb"], 591, 15_458_373_296, 2_070_892_484)]
    sets = []
    for names, count, growth, fsc in groups:
        selection, ids = opened.computational_selection(data, names)
        assert ids == [sid for sid in data["supported_order"] if data["sources"][sid]["opened_cohort"] in names]
        assert selection["cohort_sources"] == selection["requested_sources"] == len(ids) == count
        assert all(data["sources"][sid]["role"] == "development" for sid in ids)
        resources = opened.forecast(tmp_path.resolve(), data, ids)
        assert resources["forecast_growth_bytes"] == growth
        assert resources["fsc_forecast_bytes"] == fsc
        assert resources["capacity_sufficient"] is True
        sets.append(set(ids))
    assert not sets[0] & sets[1] and sets[0] | sets[1] == set(data["supported_order"])
    assert len(data["original_order"]) == 876 and len(data["unavailable"]) == 105
    assert opened.serialized(data) == original


def test_explicit_batch_completion_is_distinct_from_full_supported_completion(cohort):
    receipt = run(cohort, cohorts=["native_eb", "ordinary"])
    assert receipt["status"] == "OPENED_PUBLIC_INPUTS_COHORTS_VERIFIED"
    assert receipt["selected_cohorts_complete"] is True and receipt["provisioning_complete"] is False
    selection = receipt["computational_selection"]
    assert selection == {"schema": 1, "cohorts": ["ordinary", "native_eb"], "source_limit": None,
                         "cohort_sources": 2, "requested_sources": 2}
    manifest, rows, _ = opened.verified_inputs(cohort.spec, cohort.out)
    assert list(rows) == list(manifest["sources"]) == ["123", "124"]
    assert manifest["computational_selection"] == selection
    assert receipt["original_counts"] == cohort.data["counts"]
    assert receipt["unavailable"] == cohort.data["unavailable"]
    full = run(cohort)
    assert full["status"] == "OPENED_PUBLIC_INPUTS_VERIFIED"
    assert full["provisioning_complete"] is full["selected_cohorts_complete"] is True
    assert full["computational_selection"]["cohorts"] == list(opened.COHORTS)


def test_source_limit_is_applied_after_original_cohort_filter(truth_case):
    case = truth_case
    receipt = run(case, cohorts=["ttv_injection", "sparse_injection"], source_limit=1)
    assert receipt["status"] == "OPENED_PUBLIC_INPUTS_PARTIAL_VERIFIED"
    assert receipt["selected_cohorts_complete"] is False and receipt["provisioning_complete"] is False
    manifest, rows, _ = opened.verified_inputs(case.spec, case.out)
    assert list(rows) == ["128"]
    assert manifest["computational_selection"] == {"schema": 1, "cohorts": ["sparse_injection", "ttv_injection"],
                                                   "source_limit": 1, "cohort_sources": 2, "requested_sources": 1}


@pytest.mark.parametrize("names", [[], "ordinary", True, {"ordinary"}, ["ordinary", "ordinary"], ["background"], ["protected"], ["missing"], [1], [{}]])
def test_invalid_computational_cohorts_withdraw_old_success(cohort, names):
    run(cohort)
    with pytest.raises(ValueError, match="cohorts"):
        run(cohort, cohorts=names)
    assert not (cohort.out / "receipt.json").exists()
    assert not (cohort.out / "input-manifest.json").exists()


@pytest.mark.parametrize("limit", [True, 0, -1, 1.0, "1", 3])
def test_cohort_limit_must_be_typed_positive_within_filtered_count(cohort, limit):
    run(cohort)
    with pytest.raises(ValueError, match="integer"):
        run(cohort, cohorts=["ordinary", "native_eb"], source_limit=limit)
    assert not (cohort.out / "receipt.json").exists()


def rebind_array_handoff(case, manifest, receipt, ids=None):
    values = {"input-manifest.json": manifest}
    if ids is not None:
        values["source-ids.json"] = ids
    for name, value in values.items():
        path = case.out / name
        path.chmod(0o644)
        path.write_bytes(opened.serialized(value))
        receipt["dependencies"][str(path)] = opened.digest(path)
    receipt["manifest_sha256"] = opened.digest(case.out / "input-manifest.json")
    receipt["source_ids_sha256"] = opened.digest(case.out / "source-ids.json")
    path = case.out / "receipt.json"
    path.chmod(0o644)
    path.write_bytes(opened.serialized(receipt))


@pytest.mark.parametrize("field,value", [("schema", True), ("schema", 2), ("cohorts", ["native_eb", "ordinary"]),
                                         ("cohorts", ["ordinary", "ordinary"]), ("cohorts", ["periodic_planet"]),
                                         ("cohorts", "ordinary"), ("cohort_sources", 3), ("requested_sources", 1),
                                         ("requested_sources", 2.0), ("source_limit", True), ("source_limit", 1),
                                         ("source_limit", 3), ("extra", True)])
def test_rehashed_selection_cannot_forge_order_count_or_limit(cohort, field, value):
    receipt = run(cohort, cohorts=["ordinary", "native_eb"])
    manifest = json.loads((cohort.out / "input-manifest.json").read_bytes())
    manifest["computational_selection"][field] = value
    receipt["computational_selection"][field] = value
    rebind_array_handoff(cohort, manifest, receipt)
    with pytest.raises(ValueError):
        opened.verified_inputs(cohort.spec, cohort.out)
    assert not (cohort.out / "receipt.json").exists()


@pytest.mark.parametrize("change", ["old", "full_complete", "cohort_complete", "status", "receipt_selection", "identity_order", "original_denominator", "unavailable"])
def test_batch_receipt_rejects_legacy_or_forged_completion_and_identities(cohort, change):
    receipt = run(cohort, cohorts=["ordinary", "native_eb"])
    manifest = json.loads((cohort.out / "input-manifest.json").read_bytes())
    ids = None
    if change == "old":
        del manifest["computational_selection"]
        del receipt["computational_selection"]
    elif change == "full_complete":
        manifest["provisioning_complete"] = receipt["provisioning_complete"] = True
    elif change == "cohort_complete":
        manifest["selected_cohorts_complete"] = receipt["selected_cohorts_complete"] = False
    elif change == "status":
        receipt["status"] = "OPENED_PUBLIC_INPUTS_VERIFIED"
    elif change == "receipt_selection":
        receipt["computational_selection"]["cohorts"] = ["ordinary"]
    elif change == "identity_order":
        ids = ["124", "123"]
        manifest["sources"] = {sid: manifest["sources"][sid] for sid in ids}
    elif change == "original_denominator":
        receipt["original_counts"]["original"] = 2
    else:
        receipt["unavailable"] = {}
    rebind_array_handoff(cohort, manifest, receipt, ids)
    with pytest.raises(ValueError):
        opened.verified_inputs(cohort.spec, cohort.out)
    assert not (cohort.out / "receipt.json").exists()


def test_requested_forecast_counts_only_batch_growth_and_all_existing_use(cohort, monkeypatch):
    _, ids = opened.computational_selection(cohort.data, ["ordinary"])
    scope = opened.forecast(cohort.out.resolve(), cohort.data, ids)
    full = opened.forecast(cohort.out.resolve(), cohort.data)
    assert scope["forecast_growth_bytes"] < full["forecast_growth_bytes"]
    assert scope["fsc_forecast_bytes"] == opened.preparation_bytes(cohort.data["sources"]["123"])
    reserve = scope["reserve_bytes"]
    growth = scope["forecast_growth_bytes"]
    for free, used, expected in ((reserve + growth, 0, True), (reserve + growth - 1, 0, False),
                                  (32_000_000_000, 32_000_000_000 - reserve - growth + 1, False)):
        monkeypatch.setattr(opened.shutil, "disk_usage", lambda path: SimpleNamespace(free=free, used=used, total=32_000_000_000))
        assert opened.forecast(cohort.out.resolve(), cohort.data, ids)["capacity_sufficient"] is expected


@pytest.mark.parametrize("names,expected", [(["periodic_planet", "sparse_injection", "ttv_injection"], ["125", "128", "129"]),
                                          (["ordinary", "native_eb"], [])])
def test_truths_preserve_all_bindings_and_return_only_verified_batch_refs(truth_case, names, expected):
    case = truth_case
    array_receipt = run(case, cohorts=names)
    truth = provision_truth(case)
    assert truth["computational_selection"] == array_receipt["computational_selection"]
    assert truth["selected_truth_source_ids"] == expected
    assert truth["omitted_truth_sources"] == 3 - len(expected)
    assert truth["counts"]["sources"] == 3 and truth["full180_metadata_verified"] is True
    assert all((case.out / "truths" / name).read_bytes() == payload for name, payload in case.truth_payloads.items())
    refs, _ = opened.truth_refs(prepare_truth_case(case), case.spec, case.truth_spec, case.out)
    assert list(refs) == expected


def test_switching_batch_or_failed_filter_withdraws_truth_success(truth_case):
    case = truth_case
    run(case, cohorts=["periodic_planet", "sparse_injection", "ttv_injection"])
    provision_truth(case)
    run(case, cohorts=["ordinary", "native_eb"])
    assert not (case.out / "truths/receipt.json").exists()
    provision_truth(case)
    with pytest.raises(ValueError, match="cohorts"):
        run(case, cohorts=[])
    assert not (case.out / "truths/receipt.json").exists()
    assert all((case.out / "truths" / name).read_bytes() == payload for name, payload in case.truth_payloads.items())
