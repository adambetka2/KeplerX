"""Public cohort downloads fail closed before publishing usable inputs."""

import hashlib
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pytest
import requests
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deterministic import cloud_public as public
from deterministic.catalogue_controls import product_url


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "DUMP", tmp_path)
    monkeypatch.setattr(public, "SPECIFICATION", tmp_path / "specification.json")
    monkeypatch.setattr(public, "EXTERNAL_DUMPS", ())
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    monkeypatch.setattr(
        public.shutil, "disk_usage", lambda path: SimpleNamespace(free=4 * public.LIMIT)
    )
    raw = {
        "source_id": np.full(200, 123, dtype=np.int64),
        "time_bkjd": np.arange(200, dtype=np.float64) * 0.02,
        "cadenceno": np.arange(200, dtype=np.int32),
        "quarter": np.ones(200, dtype=np.int16),
        "psf_flat_kbonus_flux": np.full(200, 1000.0),
        "psf_flat_kbonus_flux_err": np.ones(200),
        "sap_quality": np.zeros(200, dtype=np.int32),
        "flatten_mask": np.zeros(200, dtype=np.int32),
        "qdiag_time_bkjd": np.arange(240, dtype=np.float64) * 0.02,
        "qdiag_cadenceno": np.arange(240, dtype=np.int32),
    }
    payload = b"original-FITS"
    row = {
        "url": product_url(100),
        "product_kic": 100,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "raw_view_sha256": public.curve_digest(
            public.photometry(raw, public.DISCOVERY_VIEW)
        ),
        "prepared_records_sha256": hashlib.sha256(
            public.prepare_records(raw).tobytes()
        ).hexdigest(),
        "metadata": {
            "source_id": "123",
            "raw_fits_sha256": hashlib.sha256(payload).hexdigest(),
            "origin_tpf_kic": 99,
            "kic": 100,
            "shard": 7,
            "ra_deg": 12.3,
            "dec_deg": -4.5,
        },
    }
    spec = {
        "schema": 1,
        "scope": public.SCOPE,
        "max_download_bytes": public.LIMIT,
        "sources": {"123": row},
    }
    public.SPECIFICATION.write_text(json.dumps(spec))
    monkeypatch.setattr(
        public, "fits_arrays", lambda path, sid, value: raw, raising=False
    )
    return SimpleNamespace(
        row=row, raw=raw, payload=payload, spec=spec, out=tmp_path / "cloud-public"
    )


class Response:
    status_code = 200

    def __init__(self, row, blocks, length=None):
        self.url = row["url"]
        self.blocks = blocks
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"MAST HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for block in self.blocks:
            if isinstance(block, Exception):
                raise block
            yield block


def response(monkeypatch, cohort, blocks=None, length=None):
    value = Response(cohort.row, [cohort.payload] if blocks is None else blocks, length)
    monkeypatch.setattr(public.requests, "get", lambda *args, **kwargs: value)
    return value


def test_fetch_preserves_metadata_all_arrays_and_scope(cohort, monkeypatch):
    response(monkeypatch, cohort, length=len(cohort.payload))
    receipt = public.fetch_public("1st")
    assert receipt["status"] == "PUBLIC_INPUTS_VERIFIED"
    assert receipt["scope"] == public.SCOPE
    assert receipt["survey_context_complete"] is False
    manifest = json.loads((cohort.out / "input-manifest.json").read_text())
    assert manifest["sources"]["123"]["metadata"] == cohort.row["metadata"]
    with np.load(cohort.out / "123.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(cohort.raw)
        assert len(saved["qdiag_time_bkjd"]) == 240
        for name, value in cohort.raw.items():
            assert saved[name].dtype == value.dtype
            np.testing.assert_array_equal(saved[name], value)
    assert json.loads((cohort.out / "source-ids.json").read_text()) == ["123"]
    assert receipt["manifest_sha256"] == public.digest(
        cohort.out / "input-manifest.json"
    )


def test_valid_existing_fits_is_rehashed_without_network(cohort, monkeypatch):
    cohort.out.mkdir()
    (cohort.out / "123.fits").write_bytes(cohort.payload)
    monkeypatch.setattr(
        public.requests, "get", lambda *args, **kwargs: pytest.fail("network reuse")
    )
    assert public.fetch_public("2nd")["section"] == "2nd"
    (cohort.out / "123.fits").write_bytes(b"X" * len(cohort.payload))
    with pytest.raises(ValueError, match="hash differs"):
        public.fetch_public("2nd")
    assert not (cohort.out / "receipt.json").exists()


def test_declared_gaia_product_url_is_bound_to_source_identity(cohort):
    cohort.row["product_kic"] = 0
    cohort.row["metadata"]["kic"] = 0
    cohort.row["url"] = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(
        public.declared_uri(cohort.row), safe=""
    )
    public.checked_source("123", cohort.row)
    cohort.row["url"] = cohort.row["url"].replace("gaia-dr3-123", "gaia-dr3-124")
    with pytest.raises(ValueError, match="Noncanonical"):
        public.checked_source("123", cohort.row)


def test_changed_specification_during_materialization_is_not_published(
    cohort, monkeypatch
):
    response(monkeypatch, cohort)

    def changed(path, sid, row):
        public.SPECIFICATION.write_text(public.SPECIFICATION.read_text() + "\n")
        return cohort.raw

    monkeypatch.setattr(public, "fits_arrays", changed)
    with pytest.raises(ValueError, match="specification changed"):
        public.fetch_public("1st")
    assert not (cohort.out / "receipt.json").exists()
    assert not (cohort.out / "input-manifest.json").exists()


@pytest.mark.parametrize(
    "url",
    [
        "http://mast.stsci.edu/api/v0.1/Download/file?uri=x",
        "https://mast.stsci.edu.evil.test/api/v0.1/Download/file?uri=x",
        "https://user@mast.stsci.edu/api/v0.1/Download/file?uri=x",
        "https://mast.stsci.edu:443/api/v0.1/Download/file?uri=x",
        "https://mast.stsci.edu/api/v0.1/Download/file?uri=x",
        product_url(101),
        product_url(100) + "&uri=x",
        product_url(100) + "&other=x",
        product_url(100) + "#fragment",
    ],
)
def test_noncanonical_urls_reject_before_network(cohort, monkeypatch, url):
    cohort.spec["sources"]["123"]["url"] = url
    public.SPECIFICATION.write_text(json.dumps(cohort.spec))
    monkeypatch.setattr(
        public.requests, "get", lambda *args, **kwargs: pytest.fail("unsafe URL")
    )
    with pytest.raises(ValueError, match="Noncanonical"):
        public.fetch_public("1st")


@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        ([b"changed-FITS!"], "hash differs"),
        ([b"short"], "wrong-sized"),
        ([b"original-FITS", b"extra"], "exceeds"),
    ],
)
def test_stream_hash_truncation_overflow_keep_partial(
    cohort, monkeypatch, blocks, message
):
    response(monkeypatch, cohort, blocks=blocks)
    with pytest.raises(ValueError, match=message):
        public.fetch_public("1st")
    assert not (cohort.out / "123.fits").exists()
    assert (cohort.out / "123.fits.part").exists()
    assert not (cohort.out / "receipt.json").exists()
    assert not (cohort.out / "input-manifest.json").exists()


def test_network_failure_keeps_partial_and_original_exception(cohort, monkeypatch):
    response(
        monkeypatch,
        cohort,
        blocks=[b"first", requests.ConnectionError("network stopped")],
    )
    with pytest.raises(requests.ConnectionError, match="network stopped"):
        public.fetch_public("1st")
    assert (cohort.out / "123.fits.part").read_bytes() == b"first"
    assert not (cohort.out / "receipt.json").exists()


def test_content_length_and_redirect_are_not_accepted(cohort, monkeypatch):
    value = response(monkeypatch, cohort, length=len(cohort.payload) + 1)
    with pytest.raises(ValueError, match="Content-Length"):
        public.fetch_public("1st")
    value.headers = {}
    value.status_code = 302
    with pytest.raises(ValueError, match="response status"):
        public.fetch_public("1st")


def test_disk_checkpoint_covers_growth_and_preserves_local_reserve(cohort, monkeypatch):
    monkeypatch.setattr(
        public.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=public.LIMIT + len(cohort.payload) - 1),
    )
    with pytest.raises(ValueError, match="reserve"):
        public.fetch_public("1st")
    monkeypatch.setattr(
        public.shutil, "disk_usage", lambda path: SimpleNamespace(free=4 * public.LIMIT)
    )
    monkeypatch.setattr(public, "EXTERNAL_DUMPS", (cohort.out.parent.resolve(),))
    with pytest.raises(ValueError, match=str(20 * 1024**3)):
        public.fetch_public("1st")
    monkeypatch.setattr(public, "EXTERNAL_DUMPS", ())
    monkeypatch.delenv("SPACELAB_EXECUTION_PROFILE")
    with pytest.raises(ValueError, match=str(20 * 1024**3)):
        public.fetch_public("1st")


def test_disk_exhaustion_during_stream_never_publishes(cohort, monkeypatch):
    response(monkeypatch, cohort, blocks=[b"first", b"rest"])
    free = iter(
        [4 * public.LIMIT, 4 * public.LIMIT, 4 * public.LIMIT, public.LIMIT - 1]
    )
    monkeypatch.setattr(
        public.shutil, "disk_usage", lambda path: SimpleNamespace(free=next(free))
    )
    with pytest.raises(ValueError, match="reserve"):
        public.fetch_public("1st")
    assert (cohort.out / "123.fits.part").read_bytes() == b"first"
    assert not (cohort.out / "receipt.json").exists()


@pytest.mark.parametrize("key", ["raw_view_sha256", "prepared_records_sha256"])
def test_changed_canonical_array_digest_rejects(cohort, monkeypatch, key):
    response(monkeypatch, cohort)
    cohort.spec["sources"]["123"][key] = "0" * 64
    public.SPECIFICATION.write_text(json.dumps(cohort.spec))
    with pytest.raises(ValueError, match="frozen ordinary-control"):
        public.fetch_public("1st")
    assert not (cohort.out / "receipt.json").exists()
    assert not (cohort.out / "input-manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", 2),
        ("scope", "SEALED"),
        ("sources", {}),
        ("max_download_bytes", public.LIMIT + 1),
        ("max_download_bytes", 1),
    ],
)
def test_invalid_scope_or_budget_rejects(cohort, field, value):
    cohort.spec[field] = value
    public.SPECIFICATION.write_text(json.dumps(cohort.spec))
    with pytest.raises(ValueError, match="Invalid opened-development|byte budget"):
        public.fetch_public("1st")


def test_checked_in_opened_cohort_supports_both_product_identities():
    spec = public.specification()
    rows = list(spec["sources"].values())
    assert rows
    assert any(row["product_kic"] == 0 for row in rows)
    assert any(row["product_kic"] > 0 for row in rows)
    parents = [row["metadata"]["origin_tpf_kic"] for row in rows]
    assert len(set(parents)) == len(parents)
    assert all(row["url"] == row["metadata"]["product_url"] for row in rows)
    assert sum(row["bytes"] for row in rows) <= public.LIMIT


@pytest.fixture
def fits_product(tmp_path):
    primary = fits.PrimaryHDU()
    primary.header["KEPLERID"] = 100
    primary.header["GAIAID"] = "Gaia DR3 123"

    def table(columns, cadences, name):
        data = []
        for column in columns.values():
            integer = column in {"CADENCENO", "QUARTER", "SAP_QUALITY", "FLATTEN_MASK"}
            values = (
                np.array(cadences)
                if column == "CADENCENO"
                else np.arange(len(cadences)) + 10
            )
            data.append(
                fits.Column(name=column, format="J" if integer else "E", array=values)
            )
        return fits.BinTableHDU.from_columns(data, name=name)

    stitched = table(public.STITCHED_COLUMNS, [0, 2, 4], "LIGHTCURVE_STITCHED")
    quarters = []
    for quarter, cadences in [(2, [4, 3]), (1, [0, 2, 1])]:
        hdu = table(public.DIAGNOSTIC_COLUMNS, cadences, f"LIGHTCURVE_Q{quarter}")
        for key, value in {
            "QUARTER": quarter,
            "MODULE": 13,
            "OUTPUT": 2,
            "CHANNEL": 46,
        }.items():
            hdu.header[key] = value
        quarters.append(hdu)
    path = tmp_path / "product.fits"
    fits.HDUList([primary, stitched, *quarters]).writeto(path)
    return path


def test_fits_reader_retains_full_quarters_order_and_canonical_dtypes(fits_product):
    raw = public.fits_arrays(fits_product, "123", {"product_kic": 100})
    assert len(raw["time_bkjd"]) == 3
    np.testing.assert_array_equal(raw["qdiag_cadenceno"], [0, 2, 1, 4, 3])
    np.testing.assert_array_equal(raw["qdiag_quarter"], [1, 1, 1, 2, 2])
    np.testing.assert_array_equal(raw["qdiag_module"], np.full(5, 13))
    np.testing.assert_array_equal(raw["source_id"], np.full(3, 123))
    expected = (
        set(public.STITCHED_COLUMNS)
        | set(public.DIAGNOSTIC_COLUMNS)
        | {
            "source_id",
            "qdiag_quarter",
            "qdiag_module",
            "qdiag_output",
            "qdiag_channel",
        }
    )
    assert set(raw) == expected
    for name, array in raw.items():
        assert array.dtype == np.dtype(public.INTEGER_TYPES.get(name, np.float64))


@pytest.mark.parametrize(("sid", "kic"), [("124", 100), ("123", 101)])
def test_fits_reader_rejects_wrong_source_identity(fits_product, sid, kic):
    with pytest.raises(ValueError, match="source identity"):
        public.fits_arrays(fits_product, sid, {"product_kic": kic})


def test_fits_reader_gaia_product_requires_explicit_empty_kepler_id(fits_product):
    with fits.open(fits_product, mode="update") as hdus:
        hdus[0].header["KEPLERID"] = ""
    assert (
        public.fits_arrays(fits_product, "123", {"product_kic": 0})["source_id"][0]
        == 123
    )
    with fits.open(fits_product, mode="update") as hdus:
        hdus[0].header["KEPLERID"] = 100
    with pytest.raises(ValueError, match="source identity"):
        public.fits_arrays(fits_product, "123", {"product_kic": 0})


def test_fits_reader_rejects_inconsistent_quarter_header(fits_product):
    with fits.open(fits_product, mode="update") as hdus:
        hdus["LIGHTCURVE_Q1"].header["QUARTER"] = 2
    with pytest.raises(ValueError, match="quarter identity"):
        public.fits_arrays(fits_product, "123", {"product_kic": 100})


def array_identities(arrays):
    return {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "bytes": array.nbytes,
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
        for name, array in arrays.items()
    }


@pytest.fixture
def control(cohort, monkeypatch):
    monkeypatch.setattr(public, "CONTROL_SPECIFICATION", public.SPECIFICATION)
    monkeypatch.setattr(public, "CONTROL_DELTAS", public.DUMP / "deltas.tar.gz")
    delta = {"psf_flat_kbonus_flux": np.linspace(-0.1, 0, 200, dtype=np.float32)}
    payload = io.BytesIO()
    np.savez_compressed(payload, **delta)
    content = payload.getvalue()
    with tarfile.open(public.CONTROL_DELTAS, "w:gz") as bundle:
        member = tarfile.TarInfo("123.npz")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    transformed = dict(cohort.raw)
    transformed["psf_flat_kbonus_flux"] = (
        np.asarray(cohort.raw["psf_flat_kbonus_flux"], dtype=np.float64)
        + delta["psf_flat_kbonus_flux"]
    )
    identity = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    base_checks = {
        key: cohort.row[key] for key in ("raw_view_sha256", "prepared_records_sha256")
    }
    cohort.row.update(
        {
            "opened_cohort": "shallow_planet_extra",
            "opened_scope": public.CONTROL_COHORTS["shallow_planet_extra"][1],
            "reader": "lc2",
            "origin": {"original_rank": 1},
            "arrays": array_identities(transformed),
            "base_arrays": array_identities(cohort.raw),
            "base_checks": base_checks,
            "delta": {"path": "123.npz", **identity, "arrays": array_identities(delta)},
            "raw_view_sha256": public.curve_digest(
                public.photometry(transformed, "psf_flat")
            ),
            "prepared_records_sha256": hashlib.sha256(
                public.prepare_records(transformed).tobytes()
            ).hexdigest(),
        }
    )
    cohort.spec.update(
        {
            "status": "BOUNDED_OPENED_CONTROL_INPUTS",
            "automatic_pc_enabled": False,
            "joint_independence_complete": False,
            "boundary": "Development extra only",
            "deltas": {
                "archive": {
                    "bytes": public.CONTROL_DELTAS.stat().st_size,
                    "sha256": public.digest(public.CONTROL_DELTAS),
                },
                "files": {"123.npz": identity},
            },
        }
    )
    public.CONTROL_SPECIFICATION.write_text(json.dumps(cohort.spec))
    monkeypatch.setattr(public, "fits_arrays", lambda *args: dict(cohort.raw))
    cohort.transformed = transformed
    cohort.control_out = public.DUMP / "cloud-controls"
    response(monkeypatch, cohort)
    return cohort


def test_controls_preserve_original_delta_arrays_and_development_boundary(control):
    receipt = public.fetch_controls("2nd")
    assert receipt["automatic_pc_enabled"] is False
    assert receipt["joint_independence_complete"] is False
    assert receipt["cohorts"]["shallow_planet_extra"]["sources"] == 1
    assert not control.out.exists()
    with np.load(control.control_out / "123.npz", allow_pickle=False) as saved:
        public.verify_inventory(dict(saved), array_identities(control.transformed))
    manifest = json.loads((control.control_out / "input-manifest.json").read_text())
    assert manifest["sources"]["123"]["opened_scope"].startswith("DEVELOPMENT_EXTRA")
    assert manifest["sources"]["123"]["origin"] == control.row["origin"]
    delta_path = control.control_out / "deltas/123.npz"
    assert delta_path.stat().st_mode & 0o222 == 0
    assert public.digest(delta_path) == control.row["delta"]["sha256"]


@pytest.mark.parametrize(
    "mutation", ["diagnostic", "dtype", "missing", "base", "delta"]
)
def test_controls_reject_changes_outside_discovery_view(control, mutation):
    if mutation == "diagnostic":
        control.raw["qdiag_time_bkjd"][0] += 1
    elif mutation == "dtype":
        control.raw["qdiag_cadenceno"] = control.raw["qdiag_cadenceno"].astype(np.int64)
    elif mutation == "missing":
        del control.raw["qdiag_cadenceno"]
    elif mutation == "base":
        control.row["base_checks"]["prepared_records_sha256"] = "0" * 64
    else:
        control.row["delta"]["arrays"]["psf_flat_kbonus_flux"]["sha256"] = "0" * 64
    public.CONTROL_SPECIFICATION.write_text(json.dumps(control.spec))
    with pytest.raises(ValueError, match="Control array|Public photometry"):
        public.fetch_controls("1st")
    assert not (control.control_out / "receipt.json").exists()
    assert not (control.control_out / "input-manifest.json").exists()


@pytest.mark.parametrize(
    "field", ["automatic_pc_enabled", "joint_independence_complete"]
)
def test_controls_reject_scientific_boundary_changes(control, field):
    control.spec[field] = True
    public.CONTROL_SPECIFICATION.write_text(json.dumps(control.spec))
    with pytest.raises(ValueError, match="development boundary"):
        public.fetch_controls("1st")
    assert not control.control_out.exists()


@pytest.mark.parametrize(
    "mutation", ["cohort", "reader", "delta_path", "delta_inventory"]
)
def test_controls_reject_identity_or_reader_substitution(control, mutation):
    if mutation == "cohort":
        control.row["opened_cohort"] = "sealed_planet"
    elif mutation == "reader":
        control.row["reader"] = "catalogue"
    elif mutation == "delta_path":
        control.row["delta"]["path"] = "../123.npz"
    else:
        control.spec["deltas"]["files"]["124.npz"] = control.spec["deltas"]["files"][
            "123.npz"
        ]
    public.CONTROL_SPECIFICATION.write_text(json.dumps(control.spec))
    with pytest.raises(ValueError, match="Control cohort|delta path|delta inventory"):
        public.fetch_controls("1st")


@pytest.mark.parametrize("name", ["receipt.json", "input-manifest.json", "deltas"])
def test_control_output_symlink_rejects_without_writing_target(control, name):
    control.control_out.mkdir()
    target = public.DUMP / "protected"
    target.write_text("preserve")
    (control.control_out / name).symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        public.fetch_controls("1st")
    assert target.read_text() == "preserve"


@pytest.mark.parametrize("name", ["../123.npz", "unexpected.npz", "duplicate"])
def test_delta_archive_rejects_paths_extra_members_and_duplicates(control, name):
    with tarfile.open(public.CONTROL_DELTAS, "r:gz") as bundle:
        content = bundle.extractfile("123.npz").read()
    names = ["123.npz", "123.npz"] if name == "duplicate" else [name]
    with tarfile.open(public.CONTROL_DELTAS, "w:gz") as bundle:
        for entry in names:
            member = tarfile.TarInfo(entry)
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))
    control.spec["deltas"]["archive"] = {
        "bytes": public.CONTROL_DELTAS.stat().st_size,
        "sha256": public.digest(public.CONTROL_DELTAS),
    }
    public.CONTROL_SPECIFICATION.write_text(json.dumps(control.spec))
    with pytest.raises(ValueError, match="archive member"):
        public.fetch_controls("1st")
    assert not (control.control_out / "receipt.json").exists()
    assert not (public.DUMP / "123.npz").exists()


@pytest.mark.parametrize("mutation", ["unknown", "shape", "nonfinite"])
def test_original_delta_contract_rejects_invalid_flux_transform(control, mutation):
    values = {"psf_flat_kbonus_flux": np.zeros(200, dtype=np.float32)}
    if mutation == "unknown":
        values = {"time_bkjd": values["psf_flat_kbonus_flux"]}
    elif mutation == "shape":
        values["psf_flat_kbonus_flux"] = np.zeros(201, dtype=np.float32)
    else:
        values["psf_flat_kbonus_flux"][0] = np.nan
    path = public.DUMP / "123.npz"
    np.savez_compressed(path, **values)
    control.row["delta"].update(
        {
            "bytes": path.stat().st_size,
            "sha256": public.digest(path),
            "arrays": array_identities(values),
        }
    )
    with pytest.raises(ValueError, match="unknown flux|shape or values"):
        public.control_arrays(path, "123", control.row, public.DUMP)


def test_native_reader_preserves_original_endian_and_stitched_quarter_alignment(
    fits_product,
):
    with fits.open(fits_product, mode="update") as hdus:
        hdus[0].header["RA_OBJ"] = 1.0
        hdus[0].header["DEC_OBJ"] = 2.0
    metadata, arrays = public.read_product(fits_product, 100)
    row = {
        "reader": "catalogue",
        "product_kic": 100,
        "metadata": metadata,
        "arrays": array_identities(arrays),
    }
    actual = public.control_arrays(fits_product, "123", row, fits_product.parent)
    assert len(actual) == 21
    assert actual["cadenceno"].dtype.str == ">i4"
    assert actual["time_bkjd"].dtype.str == ">f4"
    np.testing.assert_array_equal(actual["qdiag_cadenceno"], [0, 2, 4])
    public.verify_inventory(actual, row["arrays"])
    row["metadata"] = dict(metadata, source_id="124")
    with pytest.raises(ValueError, match="metadata differs"):
        public.control_arrays(fits_product, "123", row, fits_product.parent)


def test_checked_in_controls_are_bounded_opened_inputs_with_original_injections():
    specification = public.control_specification()
    assert len(specification["sources"]) == 14
    assert Counter(
        row["opened_cohort"] for row in specification["sources"].values()
    ) == {
        "native_eb": 4,
        "native_confirmed_extra": 4,
        "periodic_planet": 4,
        "shallow_planet_extra": 2,
    }
    public.checked_file(public.CONTROL_DELTAS, specification["deltas"]["archive"])
    with tarfile.open(public.CONTROL_DELTAS, "r:gz") as bundle:
        for member in bundle:
            payload = bundle.extractfile(member).read()
            expected = specification["deltas"]["files"][member.name]
            assert len(payload) == expected["bytes"]
            assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
    for row in specification["sources"].values():
        assert len(row["arrays"]) == (21 if row["reader"] == "catalogue" else 29)
        if "delta" in row:
            assert len(row["delta"]["arrays"]) == 4
