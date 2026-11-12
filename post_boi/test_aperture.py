"""Gold regressions from native Kepler pixels and delivered KBonus SAP."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from post_boi.aperture import physical_aperture
from post_boi.records import read, receipt


def inputs(quarter):
    root = Path(__file__).with_name("fixtures")
    provenance = read(root / "kepler-aperture-replay-sources.json")
    fixture = root / "kepler-aperture-replay.npz"
    assert receipt(fixture)["sha256"] == provenance["fixture"]["sha256"]
    with np.load(fixture, allow_pickle=False) as a:
        q, channel, col, row, count, parent_col, parent_row = a[f"header_q{quarter}"]
        source = fits.BinTableHDU(data=a[f"source_q{quarter}"])
        source.header.update(
            QUARTER=q, CHANNEL=channel, TPF_COL=col, TPF_ROW=row, NPIXSAP=count
        )
        aperture = fits.ImageHDU(data=a[f"mask_q{quarter}"])
        parent = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.BinTableHDU(data=a[f"frame_q{quarter}"]),
                fits.ImageHDU(data=np.zeros(aperture.data.shape, dtype=np.uint8)),
            ]
        )
        parent[0].header.update(QUARTER=q, CHANNEL=channel)
        parent[2].header.update(CRVAL1P=parent_col, CRVAL2P=parent_row)
        return source, aperture, parent, a[f"expected_pixels_q{quarter}"].copy()


@pytest.mark.parametrize("quarter", [1, 4, 12])
def test_original_pixels_authenticate_aperture_despite_conflicting_header(quarter):
    source, aperture, parent, expected = inputs(quarter)
    result, review = physical_aperture(source, aperture, parent)
    np.testing.assert_array_equal(result, expected)
    assert review["accepted_source_cadences"] == 8
    assert review["metadata_conflict"] == (quarter != 1)
    accepted = [r["convention"] for r in review["checks"] if r["reproduces"]]
    assert accepted == ["FLUX" if quarter == 12 else "FLUX_PLUS_FLUX_BKG_MINUS_SAP_BKG"]


@pytest.mark.parametrize("quarter", [1, 4, 12])
def test_matching_stamp_shape_cannot_hide_wrong_source_flux(quarter):
    source, aperture, parent, _ = inputs(quarter)
    source.data["SAP_FLUX"][0] += 100
    with pytest.raises(ValueError, match="do not reproduce source SAP"):
        physical_aperture(source, aperture, parent)
