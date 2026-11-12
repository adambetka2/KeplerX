"""Exact opened pixel replay must retain original inputs and separate experiments."""

import copy
import gzip
import hashlib
import io
import json
import stat
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from deterministic import cloud, cloud_public
from deterministic import cloud_injected_pixels as replay

IDS = ("2052214955493700224", "2052761344047469312", "2052697327552095744",
       "2053084085069214464", "2052146785773639040", "2086750237755948288")
QUARTERS = (14, 13, 15, 17, 15, 14)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def inventory(arrays):
    return {name: {"dtype": str(value.dtype), "shape": list(value.shape), "bytes": value.nbytes,
                   "sha256": sha(value.tobytes(order="C"))} for name, value in arrays.items()}


def npz(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def tpf(parent, quarter, channel):
    primary = fits.PrimaryHDU()
    for key, value in {"KEPLERID": parent, "QUARTER": quarter, "CHANNEL": channel}.items():
        primary.header[key] = value
    flux = np.arange(12, dtype=np.float32).reshape(3, 2, 2) + 100
    table = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.arange(3, dtype=float)),
        fits.Column(name="FLUX", format="4E", dim="(2,2)", array=flux),
        fits.Column(name="FLUX_ERR", format="4E", dim="(2,2)", array=np.ones_like(flux)),
    ])
    buffer = io.BytesIO()
    fits.HDUList([primary, table]).writeto(buffer)
    return gzip.compress(buffer.getvalue(), mtime=0)


def reference_injection(base, sid, arrays):
    with fits.open(io.BytesIO(gzip.decompress(base)), memmap=False) as hdus:
        original = hdus[1].data["FLUX"].copy()
        result = original.copy()
        for cadence in range(3):
            for row in range(2):
                for column in range(2):
                    result[cadence, row, column] = (float(original[cadence, row, column])
                        - arrays["amplitude"].item() * arrays["loss"][cadence] * arrays["profile"][row, column])
        hdus[1].data["FLUX"] = result
        hdus[0].header["FSCSRC"] = (sid, "Injected Gaia DR3 source")
        hdus[0].header["FSCMODEL"] = ("physical-prf-v1", "Forgotten Stars control")
        for hdu in hdus:
            hdu.add_datasum(when="Forgotten Stars deterministic checksum")
            hdu.add_checksum(when="Forgotten Stars deterministic checksum", override_datasum=True)
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as compressed:
            hdus.writeto(compressed)
    return buffer.getvalue()


def write_archive(fixture, members=None):
    members = list(fixture.recipes.items()) if members is None else members
    with tarfile.open(fixture.archive, "w:gz") as bundle:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    fixture.specification["archive"] = {"path": fixture.archive.name,
        "bytes": fixture.archive.stat().st_size, "sha256": replay.digest(fixture.archive)}
    write_json(fixture.spec, fixture.specification)


def public_inputs(folder):
    raw = {"time_bkjd": np.arange(200, dtype=float) * 0.02, "cadenceno": np.arange(200, dtype=np.int32),
           "quarter": np.ones(200, dtype=np.int16), "psf_flat_kbonus_flux": np.full(200, 1000.0),
           "psf_flat_kbonus_flux_err": np.ones(200), "sap_quality": np.zeros(200, dtype=np.int32),
           "flatten_mask": np.zeros(200, dtype=np.int32)}
    public, sources = {}, {}
    pointer = {"path": "opened-metadata.json", "bytes": 1, "sha256": "a" * 64}
    for index, sid in enumerate(IDS):
        cohort = "periodic_planet" if index < 4 else "shallow_planet_extra"
        declaration = {"metadata": {"source_id": sid, "shard": index, "origin_tpf_kic": 100 + index,
                                    "raw_fits_sha256": str(index + 1) * 64},
            "opened_cohort": cohort, "opened_scope": cloud_public.CONTROL_COHORTS[cohort][1],
            "origin": {"original_manifest": dict(pointer), "truth_reference": dict(pointer), "generator_reference": {"original.py": "a" * 64}},
            "arrays": inventory(raw), "raw_view_sha256": cloud_public.curve_digest(cloud_public.photometry(raw, "psf_flat")),
            "prepared_records_sha256": sha(cloud_public.prepare_records(raw).tobytes())}
        path = folder / f"{sid}.npz"
        path.write_bytes(npz(raw))
        source = {key: declaration[key] for key in ("metadata", "opened_cohort", "opened_scope", "origin")}
        sources[sid] = dict(source, arrays=path.name, sha256=replay.digest(path))
        public[sid] = declaration
    return public, sources


def recipe_specification(public):
    sources = {sid: {"source_id": sid, "opened_cohort": row["opened_cohort"], "opened_scope": row["opened_scope"],
                    "origin_parent_kic": row["metadata"]["origin_tpf_kic"], "raw_fits_sha256": row["metadata"]["raw_fits_sha256"],
                    **row["origin"], "quarters": {}} for sid, row in public.items()}
    return {"schema": 1, "status": "EXACT_ORIGINAL_OPENED_INJECTED_PIXEL_REPLAY", "scope": "OPENED_DEVELOPMENT_ONLY",
            "files": {}, "sources": sources, "prf_calibrations": {}, "origin": {"opened_only": True},
            "boundary": "Six original opened controls only; no release or independent scope.", **{key: False for key in replay.FLAGS}}


def add_quarter(fixture, sid, quarter, channel, prf_name):
    source = fixture.specification["sources"][sid]
    parent = source["origin_parent_kic"]
    filename = f"kplr{parent:09d}-{2009000000000 + quarter}_lpd-targ.fits.gz"
    base = tpf(parent, quarter, channel)
    padded = f"{parent:09d}"
    base_url = f"{replay.S3}/{padded[:4]}/{padded}/{filename}"
    arrays = {"profile": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
              "loss": np.array([0.0, 0.05, 0.1]), "amplitude": np.array(10.0)}
    name = f"{sid}/q{quarter:02d}.npz"
    payload = npz(arrays)
    fixture.recipes[name] = payload
    fixture.specification["files"][name] = {"bytes": len(payload), "sha256": sha(payload), "arrays": inventory(arrays)}
    injected = reference_injection(base, sid, arrays)
    with fits.open(io.BytesIO(gzip.decompress(base)), memmap=False) as hdus:
        flux_identity = inventory({"flux": hdus[1].data["FLUX"]})["flux"]
    pixel = {"path": f"pixels/{sid}/q{quarter:02d}.fits.gz", "sha256": sha(injected),
             "base_path": f"original/{filename}", "base_parent_kic": parent, "base_sha256": sha(base),
             "prf_calibration": {"filename": prf_name, "sha256": fixture.specification["prf_calibrations"][prf_name]["sha256"]},
             "target_inside_stamp": False, "target_pixel": [3.0, 3.0], "source_flux_electrons_per_second": 10.0,
             "target_sky_deg": [294.0, 39.0], "target_astrometry": "ORIGINAL_OPENED_ASTROMETRY",
             "maximum_relative_loss": 0.1, "prf_stamp_fraction": 1.0}
    source["quarters"][str(quarter)] = {
        "recipe": name, "base": {"filename": filename, "url": base_url, "bytes": len(base), "sha256": sha(base),
                                  "parent_kic": parent, "quarter": quarter, "channel": channel, "flux": flux_identity},
        "injected": {"bytes": len(injected), "sha256": sha(injected)}, "original_pixel_record": pixel,
    }
    fixture.responses[base_url] = base
    fixture.expected_pixels[pixel["path"]] = injected


class Response:
    def __init__(self, url, payload):
        self.url, self.payload = url, payload
        self.status_code = 200
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, _):
        yield self.payload


@pytest.fixture
def pixel_replay(tmp_path, monkeypatch):
    from lkprf.utils import channel_to_module_output

    controls = tmp_path / "dump/cloud-controls"
    controls.mkdir(parents=True)
    public, sources = public_inputs(controls)
    fixture = SimpleNamespace(controls=controls, folder=controls / "injected-pixels", public=public,
        public_path=tmp_path / "public-controls.json", spec=tmp_path / "cloud-injected-pixels.json",
        archive=tmp_path / "cloud-injected-pixels.tar.gz", recipes={}, responses={}, expected_pixels={}, calls=[])
    fixture.specification = recipe_specification(public)
    channels = list(range(1, 13))
    for channel in channels:
        module, output = channel_to_module_output(channel)
        name = f"kplr{module:02}.{output}_2011265_prf.fits"
        buffer = io.BytesIO()
        fits.PrimaryHDU(np.array([channel], dtype=float)).writeto(buffer)
        payload = buffer.getvalue()
        url = f"{replay.PRF}/{name}"
        fixture.specification["prf_calibrations"][name] = {"filename": name, "bytes": len(payload), "sha256": sha(payload), "url": url}
        fixture.responses[url] = payload
    for sid, count in zip(IDS, QUARTERS):
        for quarter in range(1, count + 1):
            channel = channels[(quarter - 1) % len(channels)]
            module, output = channel_to_module_output(channel)
            add_quarter(fixture, sid, quarter, channel, f"kplr{module:02}.{output}_2011265_prf.fits")
    quarters = [row for source in fixture.specification["sources"].values() for row in source["quarters"].values()]
    fixture.specification["counts"] = {
        "sources": 6, "quarters": 88, "recipe_bytes": sum(len(value) for value in fixture.recipes.values()),
        "base_pixel_bytes": sum(row["base"]["bytes"] for row in quarters),
        "injected_pixel_bytes": sum(row["injected"]["bytes"] for row in quarters),
        "prf_files": 12, "prf_bytes": sum(row["bytes"] for row in fixture.specification["prf_calibrations"].values()),
    }
    write_json(fixture.public_path, {"sources": public})
    fixture.specification["public_controls_sha256"] = replay.digest(fixture.public_path)
    boundaries = {"role": "development", "survey_context_complete": False, "automatic_pc_enabled": False, "joint_independence_complete": False}
    write_json(controls / "input-manifest.json", {"schema": 1, "measurement_scope": "OPENED_DEVELOPMENT_ONLY", "sources": sources, **boundaries})
    write_json(controls / "source-ids.json", list(sources))
    write_json(controls / "receipt.json", {"status": "PUBLIC_INPUTS_VERIFIED", "scope": "OPENED_DEVELOPMENT_ONLY", **boundaries,
               "specification_sha256": replay.digest(fixture.public_path), "manifest_sha256": replay.digest(controls / "input-manifest.json"),
               "source_ids_sha256": replay.digest(controls / "source-ids.json")})
    write_archive(fixture)

    def get(url, **kwargs):
        assert kwargs == {"stream": True, "timeout": (15, 120), "allow_redirects": False}
        fixture.calls.append(url)
        return Response(url, fixture.responses[url])

    monkeypatch.setattr(replay.requests, "get", get)
    for key, value in {"DUMP": controls.parent, "PUBLIC_CONTROL_SPECIFICATION": fixture.public_path,
                       "INJECTED_PIXEL_SPECIFICATION": fixture.spec, "INJECTED_PIXEL_BUNDLE": fixture.archive}.items():
        monkeypatch.setattr(cloud, key, value)
    return fixture


def test_replay_is_exact_separate_and_ready_for_experimental_retrieval(pixel_replay, monkeypatch):
    import vetting_pixels

    original = {name: (pixel_replay.controls / name).read_bytes() for name in replay.NAMES}
    result = cloud.injected_pixels()
    assert result["counts"]["quarters"] == 88
    assert result["config"]["frozen_pixel_inputs"] is None
    assert result["config"]["prf_cache_directory"] == str(pixel_replay.folder / "prf")
    assert all(result[key] is False for key in replay.FLAGS)
    manifest = cloud.read(pixel_replay.folder / "input-manifest.json")
    assert list(manifest["sources"]) == list(IDS)
    assert all(source["pixels"] for source in manifest["sources"].values())
    for name, expected in pixel_replay.expected_pixels.items():
        path = pixel_replay.folder / name
        assert path.read_bytes() == expected
        assert path.stat().st_mode & 0o222 == 0
    assert all((pixel_replay.controls / name).read_bytes() == payload for name, payload in original.items())
    assert len(pixel_replay.calls) == 100
    second = cloud.injected_pixels()
    assert second["manifest_sha256"] == result["manifest_sha256"]
    assert second["metrics"]["network_bytes"] == 0
    assert len(pixel_replay.calls) == 100
    for name in replay.NAMES[:2]:
        path = pixel_replay.folder / name
        assert second["dependencies"][str(path)] == replay.digest(path)
    run = pixel_replay.controls.parent / "vetting"
    (run / "sources").mkdir(parents=True)
    metadata = {sid: row["metadata"] for sid, row in manifest["sources"].items()}
    for sid in metadata:
        write_json(run / "sources" / f"{sid}.json", {"pixel_jobs": [{"quarter": 1}]})
    write_json(run / "metadata.json", metadata)
    config = run / "config.json"
    write_json(config, result["config"])
    monkeypatch.setattr(vetting_pixels, "RUN", run)
    monkeypatch.setattr(vetting_pixels, "CONFIG", config)
    monkeypatch.setattr(vetting_pixels, "PIXELS", pixel_replay.folder / "base")
    vetting_pixels.retrieve()
    assert cloud.read(run / "pixel-retrieval.json")["status"] == "EXPLICIT_EXPERIMENT_INPUTS"
    for parent in {row["origin_tpf_kic"] for row in metadata.values()}:
        quarters = cloud.read(run / "pixel-manifests" / f"{parent}.json")["quarters"]
        assert all(row["status"] == "AVAILABLE" and row["target_inside_stamp"] is False for row in quarters.values())


def test_checked_in_opened_recipe_package_has_exact_closed_inventory():
    specification = cloud.read(cloud.INJECTED_PIXEL_SPECIFICATION)
    public = cloud.read(cloud.PUBLIC_CONTROL_SPECIFICATION)["sources"]
    replay.check_specification(specification, cloud.INJECTED_PIXEL_BUNDLE, public, cloud.PUBLIC_CONTROL_SPECIFICATION)
    recipes = replay.recipe_archive(cloud.INJECTED_PIXEL_BUNDLE, specification)
    assert list(specification["sources"]) == list(IDS)
    assert len(recipes) == 88
    assert sum(len(value) for value in recipes.values()) == 432073
    assert sum(not row["original_pixel_record"]["target_inside_stamp"]
               for source in specification["sources"].values() for row in source["quarters"].values()) == 13


@pytest.mark.parametrize("flag", replay.FLAGS)
def test_replay_rejects_scientific_claims(pixel_replay, flag):
    pixel_replay.specification[flag] = True
    write_json(pixel_replay.spec, pixel_replay.specification)
    with pytest.raises(ValueError, match="scientific boundary"):
        cloud.injected_pixels()
    assert not list(pixel_replay.folder.glob("*.json"))


@pytest.mark.parametrize("field,value", [("opened_cohort", "native_eb"), ("opened_scope", "SEALED"), ("raw_fits_sha256", "0" * 64), ("origin_parent_kic", 999)])
def test_replay_preserves_original_source_roles_and_raw_identity(pixel_replay, field, value):
    pixel_replay.specification["sources"][IDS[0]][field] = value
    write_json(pixel_replay.spec, pixel_replay.specification)
    with pytest.raises(ValueError, match="cohort|binding"):
        cloud.injected_pixels()


@pytest.mark.parametrize("name", ["unknown.npz", "../escaped.npz", "/escaped.npz", "duplicate"])
def test_recipe_archive_is_closed_and_has_no_duplicate_or_escaping_members(pixel_replay, name):
    members = list(pixel_replay.recipes.items())
    members.append((members[0][0] if name == "duplicate" else name, members[0][1]))
    write_archive(pixel_replay, members)
    with pytest.raises(ValueError, match="unknown, doubled"):
        cloud.injected_pixels()
    assert not pixel_replay.calls


@pytest.mark.parametrize("defect", ["dtype", "shape", "content", "duplicate", "unknown"])
def test_recipe_npz_rejects_changed_arrays_and_archive_members(pixel_replay, defect):
    name, payload = next(iter(pixel_replay.recipes.items()))
    identity = pixel_replay.specification["files"][name]
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if defect == "dtype":
        arrays["loss"] = arrays["loss"].astype(np.float32)
    elif defect == "shape":
        arrays["profile"] = arrays["profile"].reshape(4)
    elif defect == "content":
        arrays["loss"][0] = 0.2
    else:
        buffer = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(payload)) as old, zipfile.ZipFile(buffer, "w") as new:
            for member in old.infolist():
                new.writestr(member.filename, old.read(member))
            new.writestr("loss.npy" if defect == "duplicate" else "unknown.npy", old.read("loss.npy"))
        payload = buffer.getvalue()
    if defect in {"dtype", "shape", "content"}:
        payload = npz(arrays)
    identity.update(bytes=len(payload), sha256=sha(payload))
    with pytest.raises(ValueError, match="dtype|inventory"):
        replay.factors(payload, identity)


@pytest.mark.parametrize("defect", ["symlink", "huge_header"])
def test_recipe_npz_rejects_unsafe_member_headers_before_array_allocation(pixel_replay, monkeypatch, defect):
    name, payload = next(iter(pixel_replay.recipes.items()))
    identity = dict(pixel_replay.specification["files"][name])
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as old, zipfile.ZipFile(buffer, "w") as new:
        for member in old.infolist():
            array = old.read(member)
            if member.filename == "loss.npy":
                if defect == "symlink":
                    member.external_attr = (stat.S_IFLNK | 0o777) << 16
                else:
                    unsafe = io.BytesIO()
                    np.lib.format.write_array_header_1_0(unsafe, {"descr": "<f8", "fortran_order": False, "shape": (1000000000,)})
                    array = unsafe.getvalue() + b"\0" * 24
            new.writestr(member, array)
    payload = buffer.getvalue()
    identity.update(bytes=len(payload), sha256=sha(payload))
    monkeypatch.setattr(replay.np, "load", lambda *args, **kwargs: pytest.fail("Unsafe header must not allocate arrays"))
    with pytest.raises(ValueError, match="nonregular|header dtype"):
        replay.factors(payload, identity)


@pytest.mark.parametrize("name", ["input-manifest.json", "receipt.json", "source-ids.json", "base", "recipes", "prf", f"pixels/{IDS[0]}", "receipt.json.part"])
def test_replay_rejects_output_and_parent_symlinks_before_writes(pixel_replay, tmp_path, name):
    pixel_replay.folder.mkdir()
    path = pixel_replay.folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "original").write_bytes(b"unchanged")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="redirect"):
        cloud.injected_pixels()
    assert (outside / "original").read_bytes() == b"unchanged"
    assert path.is_symlink()
    assert not pixel_replay.calls


@pytest.mark.parametrize("relative", ["../outside.bin", "base/../../outside.bin"])
def test_pixel_publish_rejects_lexical_parent_traversal(tmp_path, relative):
    folder = tmp_path / "experiment"
    folder.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="escapes"):
        replay.publish(folder / relative, b"new bytes", folder)
    assert outside.read_bytes() == b"unchanged"
    assert not list(folder.rglob("*"))


def test_replay_rejects_stale_partials_and_invalidates_a_previous_manifest(pixel_replay):
    pixel_replay.folder.mkdir()
    for name in replay.NAMES:
        (pixel_replay.folder / name).write_bytes(b"previous usable metadata")
    (pixel_replay.folder / "receipt.json.part").write_bytes(b"inspect")
    with pytest.raises(ValueError, match="Stale"):
        cloud.injected_pixels()
    assert not list(pixel_replay.folder.glob("*.json"))
    assert (pixel_replay.folder / "receipt.json.part").read_bytes() == b"inspect"


def test_replay_withdraws_previous_success_when_a_nested_cache_becomes_a_symlink(pixel_replay, tmp_path):
    cloud.injected_pixels()
    row = pixel_replay.specification["sources"][IDS[0]]["quarters"]["1"]
    path = pixel_replay.folder / "base" / row["base"]["filename"]
    path.unlink()
    outside = tmp_path / "outside-pixel"
    outside.write_bytes(b"outside unchanged")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="redirect"):
        cloud.injected_pixels()
    assert path.is_symlink() and outside.read_bytes() == b"outside unchanged"
    assert not list(pixel_replay.folder.glob("*.json"))


@pytest.mark.parametrize("defect", ["quarter_bool", "missing_pixel_field", "pixel_bool", "flux_dtype", "flux_shape", "flux_bytes", "huge_flux"])
def test_replay_rejects_malformed_original_metadata_before_downloads(pixel_replay, defect):
    row = pixel_replay.specification["sources"][IDS[0]]["quarters"]["1"]
    if defect == "quarter_bool":
        row["base"]["quarter"] = True
    elif defect == "missing_pixel_field":
        del row["original_pixel_record"]["target_inside_stamp"]
    elif defect == "pixel_bool":
        row["original_pixel_record"]["target_inside_stamp"] = 0
    elif defect == "huge_flux":
        row["base"]["flux"].update(shape=[10000, 1000, 400], bytes=16000000000)
    else:
        row["base"]["flux"][defect.removeprefix("flux_")] = {"flux_dtype": "float64", "flux_shape": [3, True, 2], "flux_bytes": 47}[defect]
    write_json(pixel_replay.spec, pixel_replay.specification)
    with pytest.raises(ValueError, match="quarter|record|descriptor|byte count"):
        cloud.injected_pixels()
    assert not pixel_replay.calls
    assert not list(pixel_replay.folder.glob("*.json"))


@pytest.mark.parametrize("field,value", [("NAXIS2", 1000000000), ("NAXIS1", 1000000000),
                                        ("TDIM2", "(1000,1000)"), ("TFORM2", "1000000000E"), ("PCOUNT", 1000000000)])
def test_actual_tpf_header_is_bounded_before_flux_allocation(pixel_replay, field, value):
    row = pixel_replay.specification["sources"][IDS[0]]["quarters"]["1"]
    payload = pixel_replay.responses[row["base"]["url"]]
    with replay.tpf_hdus(io.BytesIO(payload)) as hdus:
        hdus[1].header[field] = value
        assert "data" not in hdus[1].__dict__
        with pytest.raises(ValueError, match="bound|dimensions|row width"):
            replay.tpf_dimensions(hdus, row["base"]["flux"])
        assert "data" not in hdus[1].__dict__


def test_uncompressed_tpf_is_bounded_before_fits_parser(tmp_path, monkeypatch):
    path = tmp_path / "oversized.fits.gz"
    path.write_bytes(gzip.compress(b" " * 4097))
    monkeypatch.setattr(replay, "MAX_TPF_BYTES", 4096)
    monkeypatch.setattr(replay.fits, "open", lambda *args, **kwargs: pytest.fail("FITS parser must not run"))
    with pytest.raises(ValueError, match="uncompressed pixel memory"):
        replay.tpf_hdus(path)


def test_original_empty_kepler_column_has_zero_bounded_storage():
    assert replay.column_width("0E", 868) == 0
    assert replay.column_width("D", 868) == 8
    assert replay.column_width("35E", 868) == 140


@pytest.mark.parametrize("kind", ["base", "prf", "injected", "copied_array"])
def test_replay_rejects_changed_existing_cache_bytes(pixel_replay, kind):
    cloud.injected_pixels()
    row = pixel_replay.specification["sources"][IDS[0]]["quarters"]["1"]
    relative = {"base": "base/" + row["base"]["filename"],
                "prf": "prf/" + row["original_pixel_record"]["prf_calibration"]["filename"],
                "injected": row["original_pixel_record"]["path"], "copied_array": f"{IDS[0]}.npz"}[kind]
    path = pixel_replay.folder / relative
    path.chmod(0o600)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Pinned pixel input changed"):
        cloud.injected_pixels()
    assert path.read_bytes() == b"changed"
    assert not list(pixel_replay.folder.glob("*.json"))


@pytest.mark.parametrize("header", ["QUARTER", "KEPLERID", "CHANNEL"])
def test_replay_rechecks_tpf_headers_even_when_base_hash_is_rebound(pixel_replay, header):
    row = pixel_replay.specification["sources"][IDS[0]]["quarters"]["1"]
    payload = pixel_replay.responses[row["base"]["url"]]
    with fits.open(io.BytesIO(gzip.decompress(payload)), memmap=False) as hdus:
        hdus[0].header[header] += 1
        buffer = io.BytesIO()
        hdus.writeto(buffer)
    payload = gzip.compress(buffer.getvalue(), mtime=0)
    base = pixel_replay.controls / row["base"]["filename"]
    base.write_bytes(payload)
    row = copy.deepcopy(row)
    row["base"].update(bytes=len(payload), sha256=sha(payload))
    recipe = replay.factors(pixel_replay.recipes[row["recipe"]], pixel_replay.specification["files"][row["recipe"]])
    with pytest.raises(ValueError, match="header changed"):
        replay.replay(base, pixel_replay.controls / "out.fits.gz", IDS[0], row, recipe, pixel_replay.controls)


def test_replay_requires_original_prf_for_the_actual_channel(pixel_replay):
    row = pixel_replay.specification["sources"][IDS[0]]["quarters"]["1"]
    calibration = next(value for value in pixel_replay.specification["prf_calibrations"].values()
                       if value["filename"] != row["original_pixel_record"]["prf_calibration"]["filename"])
    row["original_pixel_record"]["prf_calibration"] = {key: calibration[key] for key in ("filename", "sha256")}
    write_json(pixel_replay.spec, pixel_replay.specification)
    with pytest.raises(ValueError, match="PRF calibration"):
        cloud.injected_pixels()


def test_replay_cannot_rehash_corrupted_control_arrays_to_bypass_frozen_inventory(pixel_replay):
    path = pixel_replay.controls / f"{IDS[0]}.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["psf_flat_kbonus_flux"][0] += 1
    path.write_bytes(npz(arrays))
    manifest_path = pixel_replay.controls / "input-manifest.json"
    manifest = cloud.read(manifest_path)
    manifest["sources"][IDS[0]]["sha256"] = replay.digest(path)
    write_json(manifest_path, manifest)
    receipt_path = pixel_replay.controls / "receipt.json"
    receipt = cloud.read(receipt_path)
    receipt["manifest_sha256"] = replay.digest(manifest_path)
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="dtype, shape or content"):
        cloud.injected_pixels()
    assert not pixel_replay.calls


@pytest.mark.parametrize("output", ["input-manifest.json", "receipt.json"])
def test_replay_cannot_publish_success_after_dependency_drift(pixel_replay, monkeypatch, output):
    original = replay.publish

    def change_specification(path, payload, root):
        original(path, payload, root)
        if path.name == output:
            pixel_replay.spec.write_bytes(b"changed during provisioning")

    monkeypatch.setattr(replay, "publish", change_specification)
    with pytest.raises(ValueError, match="dependency changed"):
        cloud.injected_pixels()
    assert not list(pixel_replay.folder.glob("*.json"))


@pytest.mark.parametrize("output", ["input-manifest.json", "receipt.json"])
def test_replay_rejects_changed_scope_in_published_output(pixel_replay, monkeypatch, output):
    original = replay.publish

    def change_manifest(path, payload, root):
        if path.name == output:
            manifest = json.loads(payload)
            manifest["scientific_acceptance"] = True
            payload = replay.canonical(manifest)
        original(path, payload, root)

    monkeypatch.setattr(replay, "publish", change_manifest)
    with pytest.raises(ValueError, match="Pinned pixel input changed"):
        cloud.injected_pixels()
    assert not list(pixel_replay.folder.glob("*.json"))


@pytest.mark.parametrize("output", replay.NAMES)
def test_replay_binds_generated_output_bytes_before_publication(pixel_replay, monkeypatch, output):
    original = replay.publish

    def rewrite_output(path, payload, root):
        original(path, payload, root)
        if path.name == output:
            path.chmod(0o600)
            path.write_bytes(payload + b" ")

    monkeypatch.setattr(replay, "publish", rewrite_output)
    with pytest.raises(ValueError, match="Pinned pixel input changed"):
        cloud.injected_pixels()
    assert not list(pixel_replay.folder.glob("*.json"))


def test_blocked_download_stops_without_usable_experiment_or_network_bypass(pixel_replay, monkeypatch):
    def blocked(*args, **kwargs):
        raise replay.requests.HTTPError("403: host blocked by configured policy")

    monkeypatch.setattr(replay.requests, "get", blocked)
    with pytest.raises(replay.requests.HTTPError, match="403"):
        cloud.injected_pixels()
    assert not list(pixel_replay.folder.glob("*.json"))
    assert not list(pixel_replay.folder.rglob("*.part"))
