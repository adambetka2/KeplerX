"""Check bounded pixel byte/header verification and temporary-file ownership."""

import hashlib
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from pipeline.deterministic import control_pixel_inputs as pixels


def fixture_payload(tmp_path):
    primary = fits.PrimaryHDU()
    primary.header.update(KEPLERID=42, QUARTER=3, CHANNEL=1)
    flux = fits.Column(
        name="FLUX",
        format="4E",
        dim="(2,2)",
        array=np.zeros((3, 2, 2), dtype=np.float32),
    )
    table = fits.BinTableHDU.from_columns([flux])
    table.header.update(TSTART=1.0, TSTOP=2.0, BJDREFI=2454833, BJDREFF=0.0)
    aperture = fits.ImageHDU(np.zeros((2, 2), dtype=np.int32))
    path = tmp_path / "reference.fits.gz"
    fits.HDUList([primary, table, aperture]).writeto(path)
    return path, path.read_bytes()


def request(payload):
    return {
        "source_id": "123",
        "cohort": "ordinary",
        "role": "evaluation",
        "quarter": 3,
        "parent_kic": 42,
        "origin_parent_kic": 42,
        "status": "PUBLIC_PRODUCT_REQUEST",
        "size_bytes": len(payload),
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
        "etag": '"fixture"',
        "transport_uri": "https://stpubdata.s3.us-east-1.amazonaws.com/kepler/public/"
        "target_pixel_files/0000/000000042/kplr000000042-2009166043257_lpd-targ.fits.gz",
        "astrometry": None,
    }


class Response:
    status_code = 200

    def __init__(self, payload, length=None):
        self.payload = payload
        self.headers = {
            "Content-Length": str(len(payload) if length is None else length),
            "ETag": '"fixture"',
        }
        self.body_reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def raise_for_status(self):
        return None

    def iter_content(self, size):
        self.body_reads += 1
        yield self.payload


def output(tmp_path, monkeypatch, response):
    (tmp_path / "spool").mkdir()
    space = namedtuple("Space", "total used free")
    monkeypatch.setattr(
        pixels.shutil, "disk_usage", lambda path: space(100 * 1024**3, 0, 100 * 1024**3)
    )
    monkeypatch.setattr(pixels.requests, "get", lambda *args, **kwargs: response)
    return tmp_path


def test_complete_bytes_headers_without_reading_pixel_values(tmp_path, monkeypatch):
    original, payload = fixture_payload(tmp_path)
    folder = output(tmp_path, monkeypatch, Response(payload))

    def forbidden_data(self):
        raise AssertionError("Pixel science arrays must not be read")

    monkeypatch.setattr(fits.BinTableHDU, "data", property(forbidden_data))
    monkeypatch.setattr(fits.ImageHDU, "data", property(forbidden_data))
    row = request(payload)
    pixels.validate_request(row)
    result = pixels.verify_product(0, row, folder)
    assert result["status"] == "PUBLIC_BYTE_AND_HEADER_IDENTITIES_VERIFIED"
    assert result["transfer"]["sha256"] == row["expected_sha256"]
    assert result["header"]["stamp_shape_yx"] == [2, 2]
    assert result["header"]["target_geometry"]["status"] == "ASTROMETRY_UNAVAILABLE"
    assert result["header"]["pixel_flux_values_read"] is False
    assert list((folder / "spool").iterdir()) == []
    assert original.read_bytes() == payload


def test_frozen_sha_mismatch_is_unresolved_and_cleans_payload(tmp_path, monkeypatch):
    _, payload = fixture_payload(tmp_path)
    folder = output(tmp_path, monkeypatch, Response(payload))
    row = {**request(payload), "expected_sha256": "0" * 64}
    result = pixels.verify_product(0, row, folder)
    assert result["status"] == "PUBLIC_PRODUCT_VERIFICATION_UNRESOLVED"
    assert "frozen SHA256" in result["error"]
    assert "header" not in result
    assert not list((folder / "spool").iterdir())


def test_wrong_quarter_header_is_unresolved(tmp_path, monkeypatch):
    _, payload = fixture_payload(tmp_path)
    folder = output(tmp_path, monkeypatch, Response(payload))
    result = pixels.verify_product(0, {**request(payload), "quarter": 4}, folder)
    assert result["status"] == "PUBLIC_PRODUCT_VERIFICATION_UNRESOLVED"
    assert "another quarter" in result["error"]
    assert not list((folder / "spool").iterdir())


def test_changed_transport_length_rejected_before_payload_read(tmp_path, monkeypatch):
    _, payload = fixture_payload(tmp_path)
    response = Response(payload, length=len(payload) + 1)
    folder = output(tmp_path, monkeypatch, response)
    result = pixels.verify_product(0, request(payload), folder)
    assert result["status"] == "PUBLIC_PRODUCT_VERIFICATION_UNRESOLVED"
    assert response.body_reads == 0
    assert not list((folder / "spool").iterdir())


def test_existing_payload_is_never_deleted(tmp_path, monkeypatch):
    _, payload = fixture_payload(tmp_path)
    folder = output(tmp_path, monkeypatch, Response(payload))
    existing = folder / "spool/product-000000.fits.gz"
    existing.write_bytes(b"pre-existing bytes")
    with pytest.raises(FileExistsError, match="already occupied"):
        pixels.verify_product(0, request(payload), folder)
    assert existing.read_bytes() == b"pre-existing bytes"


def test_parent_uri_and_oversized_payload_are_rejected():
    row = request(b"a")
    with pytest.raises(ValueError, match="declared parent"):
        pixels.validate_request({**row, "parent_kic": 43})
    with pytest.raises(ValueError, match="byte allowance"):
        pixels.validate_request({**row, "size_bytes": pixels.MAXIMUM_PRODUCT + 1})


def test_cloud_stream_preserves_the_explicit_reserve(tmp_path, monkeypatch):
    _, payload = fixture_payload(tmp_path)
    output(tmp_path, monkeypatch, Response(payload))
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    monkeypatch.setattr(pixels, "EXTERNAL_DUMPS", ())
    monkeypatch.setattr(
        pixels.shutil, "disk_usage", lambda path: SimpleNamespace(free=3 * 1024**3)
    )
    row = {**request(payload), "role": "development"}
    target = tmp_path / "download.fits.gz"
    transfer = pixels.receive_product(row, target, reserve=2 * 1024**3)
    assert transfer["sha256"] == row["expected_sha256"]
    assert target.read_bytes() == payload
    target.unlink()
    monkeypatch.setattr(
        pixels.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024**3)
    )
    with pytest.raises(OSError, match="disk reserve"):
        pixels.receive_product(row, target, reserve=2 * 1024**3)


@pytest.mark.parametrize("protected", ["local", "external", "evaluation"])
def test_cloud_stream_cannot_lower_a_protected_reserve(tmp_path, monkeypatch, protected):
    row = {**request(b"bytes"), "role": "development"}
    monkeypatch.setenv("SPACELAB_EXECUTION_PROFILE", "cloud-development")
    monkeypatch.setattr(pixels, "EXTERNAL_DUMPS", ())
    if protected == "local":
        monkeypatch.delenv("SPACELAB_EXECUTION_PROFILE")
    elif protected == "external":
        monkeypatch.setattr(pixels, "EXTERNAL_DUMPS", (tmp_path,))
    else:
        row["role"] = "evaluation"
    with pytest.raises(ValueError, match="original reserve"):
        pixels.receive_product(row, tmp_path / "download.fits.gz", reserve=2 * 1024**3)
