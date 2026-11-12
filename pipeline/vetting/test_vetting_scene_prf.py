"""Fixed-source pixel comparisons preserve model identity and fail clearly."""

import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
scene_prf = importlib.import_module("vetting_scene_prf")


def test_fixed_source_fit_prefers_the_true_pixel_profile():
    image = np.array([3.0, 0.0, 0.0])
    error = np.ones(3)
    target = scene_prf.fixed_source_fit([1.0, 0.0, 0.0], image, error)
    neighbour = scene_prf.fixed_source_fit([0.0, 1.0, 0.0], image, error)
    assert target["chi2"] == 0
    assert neighbour["chi2"] == 9
    assert scene_prf.fixed_source_fit([1, 0, 0], -image, error)["amplitude"] == 0


def test_fixed_source_fit_rejects_invalid_pixel_errors():
    with pytest.raises(ValueError, match="invalid"):
        scene_prf.fixed_source_fit([1, 0], [1, 0], [1, 0])


def test_scene_comparison_requires_a_complete_scene_and_good_quarters():
    row = {"source_id": "123", "gaia_scene": {"status": "PARTIAL_QUERY_COVERAGE"}}
    assert (
        scene_prf.compare_signal(row, {}, "/missing", "/missing")["status"]
        == "UNAVAILABLE_GAIA_SCENE"
    )
    row["gaia_scene"]["status"] = "GAIA_SCENE_QUERIED"
    assert (
        scene_prf.compare_signal(row, {}, "/missing", "/missing")["status"]
        == "UNAVAILABLE_GOOD_PIXEL_QUARTERS"
    )
    assert scene_prf.supported_pixel(-7.4, 1, (5, 5))
    assert not scene_prf.supported_pixel(-8, 1, (5, 5))


def pixel_file(path, quarter=4):
    from astropy.io import fits
    from astropy.wcs import WCS

    primary = fits.PrimaryHDU()
    primary.header["QUARTER"] = quarter
    primary.header["CHANNEL"] = 11
    table = fits.BinTableHDU.from_columns([])
    table.header["1CRV4P"] = 101
    table.header["2CRV4P"] = 202
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [3, 4]
    wcs.wcs.cdelt = [-0.001, 0.001]
    wcs.wcs.crval = [285, 45]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    aperture = fits.ImageHDU(np.ones((5, 6)), header=wcs.to_header())
    fits.HDUList([primary, table, aperture]).writeto(path, overwrite=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_header_cache_preserves_wcs_and_returns_independent_headers(
    tmp_path, monkeypatch
):
    from astropy.wcs import WCS

    path = tmp_path / "pixels.fits.gz"
    checksum = pixel_file(path)
    original_open = scene_prf.fits.open
    with original_open(path, memmap=False) as hdus:
        expected = [hdu.header.copy() for hdu in hdus]
    opened = []

    def tracked_open(*args, **kwargs):
        opened.append(args[0])
        return original_open(*args, **kwargs)

    monkeypatch.setattr(scene_prf.fits, "open", tracked_open)
    scene_prf._pixel_headers.cache_clear()
    for _ in range(3):
        headers = scene_prf.pixel_headers(path, checksum)
        assert [h.tostring() for h in headers] == [h.tostring() for h in expected]
        actual = WCS(headers[2]).world_to_pixel_values([285, 285.01], [45, 45.01])
        want = WCS(expected[2]).world_to_pixel_values([285, 285.01], [45, 45.01])
        for a, b in zip(actual, want):
            assert a.tobytes() == b.tobytes()
        headers[0]["QUARTER"] = 99
        headers[2]["CRPIX1"] = 99
    assert len(opened) == 1


def test_header_cache_rejects_changed_and_incorrect_file_identities(tmp_path):
    import os

    path = tmp_path / "pixels.fits.gz"
    checksum = pixel_file(path)
    assert scene_prf.pixel_headers(path, checksum)[0]["QUARTER"] == 4
    with pytest.raises(ValueError, match="hash changed"):
        scene_prf.pixel_headers(path, "0" * 64)
    before = path.stat().st_mtime_ns
    changed = pixel_file(path, quarter=5)
    os.utime(path, ns=(before + 1000000, before + 1000000))
    with pytest.raises(ValueError, match="hash changed"):
        scene_prf.pixel_headers(path, checksum)
    assert scene_prf.pixel_headers(path, changed)[0]["QUARTER"] == 5
