"""Keep the PSF_FLAT route distinct from the deliberately different SAP channel."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from post_boi.prepare import prepare
from post_boi.psf_flat import ingest, photons
from post_boi.records import digest, read, write

SOURCE = "2079714428492352768"


@pytest.fixture
def product(tmp_path):
    time = np.arange(256) * 0.02043361111111111 + 130
    in_transit = abs((time - 130.4 + 0.35) % 0.7 - 0.35) < 0.025
    flat = 1000 * (1 - 0.005 * in_transit)
    error = np.full(len(time), 0.1)
    error[4] = 0
    quality = np.zeros(len(time), dtype=np.int32)
    quality[2] = 1
    flatten = np.zeros(len(time), dtype=np.int32)
    flatten[3] = 1
    columns = [
        fits.Column(name="TIME", format="D", unit="BJD - 2454833", array=time),
        fits.Column(name="CADENCENO", format="J", array=np.arange(len(time))),
        fits.Column(name="QUARTER", format="J", array=np.ones(len(time))),
        fits.Column(name="PSF_FLAT_FLUX", format="D", unit="e-/s", array=flat),
        fits.Column(name="PSF_FLAT_FLUX_ERR", format="D", unit="e-/s", array=error),
        fits.Column(
            name="SAP_FLUX",
            format="D",
            unit="e-/s",
            array=4000 * (1 - 0.2 * in_transit),
        ),
        fits.Column(name="SAP_QUALITY", format="J", array=quality),
        fits.Column(name="FLATTEN_MASK", format="J", array=flatten),
    ]
    primary = fits.PrimaryHDU()
    primary.header.update(GAIAID="Gaia DR3 " + SOURCE, TIMESYS="TDB", XPOSURE=1765.464)
    stitched = fits.BinTableHDU.from_columns(columns, name="LIGHTCURVE_STITCHED")
    path = tmp_path / "product.fits"
    fits.HDUList([primary, stitched]).writeto(path, checksum=True)
    return path, flat, in_transit


def test_real_preparation_path_retains_the_psf_transit(product, tmp_path):
    original, flat, transit = product
    path = ingest(original, SOURCE, digest(original), tmp_path / "input")
    with np.load(path) as arrays:
        assert np.array_equal(arrays["flux"], flat)
        assert np.flatnonzero(~arrays["valid"]).tolist() == [2, 3, 4]
        assert arrays["source_id"].item() == SOURCE
    config = read(Path(__file__).with_name("candidate-template.json"))
    config.update(
        source_id=SOURCE,
        signal_id="psf-routing-control",
        input_basis="Synthetic routing control with distinct PSF_FLAT and SAP dips",
        photometry=str(path),
        flux_convention="source_corrected",
        exposure_days=1765.464 / 86400,
        recovered={"period_days": 0.7, "epoch_bkjd": 130.4, "duration_days": 0.05},
        other_signals=[],
    )
    config["detrending"].update(window_days=0.5, polyorder=2)
    write(tmp_path / "candidate.json", config)
    prepare(tmp_path / "candidate.json", tmp_path / "prepared")
    with np.load(tmp_path / "prepared/detrended.npz") as arrays:
        selected = transit[arrays["original_index"]]
        assert np.allclose(arrays["flux"][selected], 0.995)
        assert np.allclose(arrays["flux"][~selected], 1)
    review = read(tmp_path / "input/input-review.json")
    assert review["dilution_correction_applied"] is False
    assert review["probability_ready"] is False
    assert digest(tmp_path / "input/reader.py") == review["reader"]["sha256"]


def test_rejects_wrong_source_changed_product_and_existing_generation(
    product, tmp_path
):
    original = product[0]
    with pytest.raises(ValueError, match="Gaia identity"):
        photons(original, "1", digest(original))
    with pytest.raises(ValueError, match="hash differs"):
        photons(original, SOURCE, "0" * 64)
    output = tmp_path / "input"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        ingest(original, SOURCE, digest(original), output)
    assert list(output.iterdir()) == []
