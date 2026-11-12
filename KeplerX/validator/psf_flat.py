"""Authenticate delivered PSF_FLAT photons for the standard native input path."""

import argparse
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits

from validator.records import digest, receipt, require, write

COLUMNS = {
    "time_bkjd": "TIME",
    "cadenceno": "CADENCENO",
    "quarter": "QUARTER",
    "flux": "PSF_FLAT_FLUX",
    "flux_err": "PSF_FLAT_FLUX_ERR",
    "sap_quality": "SAP_QUALITY",
    "flatten_mask": "FLATTEN_MASK",
}


def photons(original, source_id, sha256):
    """Retain the delivered arrays and exact discovery-view quality selection."""
    require(
        isinstance(source_id, str) and source_id.isdecimal(), "Exact Gaia ID required"
    )
    require(digest(original) == sha256, "Original PSF_FLAT product hash differs")
    with fits.open(original, memmap=False) as hdus:
        require(
            hdus[0].header["GAIAID"] == "Gaia DR3 " + source_id,
            "PSF_FLAT product Gaia identity differs",
        )
        require(
            all(hdu.verify_checksum() == 1 for hdu in hdus),
            "PSF_FLAT product checksum failure",
        )
        require(hdus[0].header["TIMESYS"] == "TDB", "Require native TDB times")
        exposure = float(hdus[0].header["XPOSURE"]) / 86400
        require(np.isfinite(exposure) and exposure > 0, "Invalid measured exposure")
        stitched = hdus["LIGHTCURVE_STITCHED"]
        require(
            stitched.columns["TIME"].unit == "BJD - 2454833",
            "Unrecognized native PSF_FLAT time convention",
        )
        require(
            all(
                stitched.columns[name].unit == "e-/s"
                for name in ("PSF_FLAT_FLUX", "PSF_FLAT_FLUX_ERR")
            ),
            "Unrecognized delivered PSF_FLAT flux units",
        )
        arrays = {
            name: np.asarray(stitched.data[column]).copy()
            for name, column in COLUMNS.items()
        }
        release = {
            key: hdus[0].header.get(key)
            for key in ("ORIGIN", "VERSION", "HLSPVER", "PROCVER", "TPFORG")
        }
    arrays["source_id"] = np.asarray(source_id)
    arrays["valid"] = (
        np.isfinite(arrays["time_bkjd"])
        & np.isfinite(arrays["flux"])
        & np.isfinite(arrays["flux_err"])
        & (arrays["flux_err"] > 0)
        & (arrays["sap_quality"] == 0)
        & (arrays["flatten_mask"] == 0)
    )
    require(arrays["valid"].any(), "No usable delivered PSF_FLAT photons")
    return arrays, {
        "exposure_days": exposure,
        "time_system": "BKJD_TDB",
        "release": release,
    }


def ingest(original, source_id, sha256, output):
    """Create a separate preserved PSF_FLAT input, without a dilution correction."""
    original, output = Path(original), Path(output)
    require(not output.exists(), "PSF_FLAT input generation already exists")
    arrays, metadata = photons(original, source_id, sha256)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "native-photons.npz", **arrays)
    shutil.copyfile(__file__, output / "reader.py")
    write(
        output / "input-review.json",
        dict(
            source_id=source_id,
            view="psf_flat",
            original=receipt(original),
            photons=receipt(output / "native-photons.npz"),
            reader=receipt(output / "reader.py"),
            columns=COLUMNS,
            retained_valid=int(arrays["valid"].sum()),
            delivered_rows=len(arrays["time_bkjd"]),
            dilution_correction_applied=False,
            flux_basis="Delivered deblended PSF source-amplitude product with archived flattening and quarter rescaling retained; no SAP, PSF_NOVA or CROWDSAP substitution.",
            probability_ready=False,
            response_basis=None,
            unresolved="Authenticate and qualify the extraction and flattening response before probability allocation; SAP aperture fractions are not a PSF response.",
            **metadata,
        ),
    )
    return output / "native-photons.npz"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(ingest(args.original, args.source_id, args.sha256, args.output))
