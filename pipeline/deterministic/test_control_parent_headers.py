import copy
import gzip
import io

import numpy as np
import pytest
from astropy.io import fits

from pipeline.deterministic import control_parent_headers as parents


def original_headers():
    primary = fits.PrimaryHDU()
    primary.header.update(
        {"KEPLERID": 7, "QUARTER": 1, "MODULE": 2, "OUTPUT": 3, "CHANNEL": 7}
    )
    columns = [fits.Column(name=f"C{i}", format="D", array=[1.0]) for i in range(3)]
    columns.append(
        fits.Column(name="FLUX", format="4E", dim="(2,2)", array=np.ones((1, 2, 2)))
    )
    table = fits.BinTableHDU.from_columns(columns, name="TARGETTABLE")
    table.header.update({"1CRV4P": 11, "2CRV4P": 22})
    stream = io.BytesIO()
    fits.HDUList([primary, table]).writeto(stream)
    return parents.decode_headers(gzip.compress(stream.getvalue()))


def test_bounded_decoder_returns_two_headers_without_pixel_arrays():
    headers = original_headers()
    assert [r["name"] for r in headers] == ["PRIMARY", "TARGETTABLE"]
    geometry = parents.verify_parent_headers(headers, "7", 1)
    assert geometry["column_origin"] == 11 and geometry["row_origin"] == 22
    assert geometry["stamp_columns"] == geometry["stamp_rows"] == 2
    assert geometry["pixel_flux_read"] is False
    assert geometry["whole_product_hash_verified"] is False


def test_insufficient_compressed_headers_fail_explicitly():
    header = fits.PrimaryHDU().header.tostring().encode("ascii")
    with pytest.raises(ValueError, match="both TPF headers"):
        parents.decode_headers(gzip.compress(header))


@pytest.mark.parametrize("parent,quarter", [("9", 1), ("7", 0)])
def test_parent_or_quarter_mismatch_rejected(parent, quarter):
    with pytest.raises(ValueError):
        parents.verify_parent_headers(original_headers(), parent, quarter)


def test_post_receipt_parent_header_card_drift_is_rejected():
    headers = original_headers()
    headers[0]["cards"]["KEPLERID"] = 9
    with pytest.raises(ValueError, match="content or identity changed"):
        parents.verify_parent_headers(headers, "9", 1)


def test_stamp_match_is_not_extraction_uniqueness_or_attribution():
    geometry = parents.verify_parent_headers(original_headers(), "7", 1)
    quarter = {
        "quarter": 1,
        "archive_header": {
            "sha256": "a" * 64,
            "cards": {
                "MODULE": 2,
                "OUTPUT": 3,
                "CHANNEL": 7,
                "TPF_ROW": 22,
                "TPF_COL": 11,
                "ROW": 22.5,
                "COLUMN": 11.5,
            },
        },
    }
    result = parents.stamp_witness(quarter, {"geometry": geometry})
    assert result["status"] == "CANONICAL_PARENT_STAMP_MATCH"
    assert result["target_header_position_inside_candidate_stamp"] is True
    assert result["original_extraction_parent_uniqueness_verified"] is False
    assert result["physical_attribution_measured"] is False
    outside = copy.deepcopy(quarter)
    outside["archive_header"]["cards"]["COLUMN"] = 15.0
    assert (
        parents.stamp_witness(outside, {"geometry": geometry})[
            "target_header_position_inside_candidate_stamp"
        ]
        is False
    )
    wrong = copy.deepcopy(quarter)
    wrong["archive_header"]["cards"]["TPF_ROW"] += 1
    result = parents.stamp_witness(wrong, {"geometry": geometry})
    assert result["status"] == "NOMINAL_PARENT_STAMP_MISMATCH"
    assert result["target_header_position_inside_candidate_stamp"] is None


def test_noncanonical_timestamp_rejected():
    with pytest.raises(ValueError, match="timestamp"):
        parents.parent_url("7", "../other")
