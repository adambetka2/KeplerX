import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from pipeline.deterministic import control_mission_reference as mission
from pipeline.deterministic.control_bindings import reference
from pipeline.deterministic.control_eligibility_archive import local_headers


def write(path, value):
    path.write_text(json.dumps(value))
    return reference(path)


@pytest.fixture
def inputs(tmp_path, kic=3230227):
    binding = {
        "source_id": "1",
        "parent_tpf": str(kic),
        "catalogues": [{"parent_rows": [{"kepid": kic}]}],
    }
    prior = write(
        tmp_path / "original-donor.json",
        {"binding": binding, "status": "UNAVAILABLE"},
    )
    tic_row = {
        "KIC": kic,
        "ID": 122224576,
        "GAIA": "2",
        "version": "20190415",
        "duplicate_id": None,
        "disposition": None,
    }
    paging = {"pagesFiltered": 1, "rowsFiltered": 1}
    response = write(
        tmp_path / "tic-response.json",
        {"status": "COMPLETE", "data": [tic_row], "paging": paging, "fields": []},
    )
    receipts = {
        "tic": write(
            tmp_path / "tic.json",
            {
                "canonical_url": "https://mast.stsci.edu/api/v0/invoke",
                "response": response,
                "service_status": "COMPLETE",
                "http_status": 200,
                "request": {"service": "Mast.Catalogs.Filtered.Tic"},
                "paging": paging,
                "rows": [tic_row],
                "fields": [],
            },
        ),
    }
    for key in ("dr3_to_dr2", "dr2_to_dr3"):
        receipts[key] = write(
            tmp_path / f"{key}.json",
            {
                "endpoint": mission.GAIA_ENDPOINT,
                "status": "COMPLETE",
                "rows": [{"dr3_source_id": 3, "dr2_source_id": 2}],
            },
        )
    primary = fits.PrimaryHDU()
    for key, value in {
        "MISSION": "Kepler",
        "OBSMODE": "long cadence",
        "KEPLERID": kic,
        "QUARTER": 0,
    }.items():
        primary.header[key] = value
    columns = [
        fits.Column(name=name, format=format_, array=np.array(values))
        for name, format_, values in (
            ("TIME", "D", [120.0, 121.0]),
            ("CADENCENO", "J", [100, 101]),
            ("SAP_FLUX", "E", [500.0, 510.0]),
            ("SAP_FLUX_ERR", "E", [1.0, 1.0]),
            ("SAP_QUALITY", "J", [0, 1]),
        )
    ]
    table = fits.BinTableHDU.from_columns(columns, name="LIGHTCURVE")
    for key, value in {
        "BJDREFI": 2454833,
        "BJDREFF": 0.0,
        "TIMESYS": "TDB",
        "TIMEREF": "SOLARSYSTEM",
        "TIMEUNIT": "d",
        "TIMEDEL": 0.02043359821692,
    }.items():
        table.header[key] = value
    path = tmp_path / "mission.fits"
    fits.HDUList([primary, table]).writeto(path)
    identifier = f"{kic:09d}"
    record = {
        "status": "CANONICAL_KEPLER_KIC_QUARTER_AND_SAP_HEADER_VERIFIED",
        "reference_domain": mission.DOMAIN,
        "catalogued_parent_kic": str(kic),
        "source_target_id": "1",
        "headers": local_headers(path),
        "archive_product_bytes": path.stat().st_size,
        "url": f"https://archive.stsci.edu/pub/kepler/lightcurves/{identifier[:4]}/{identifier}/"
        f"kplr{identifier}-2009131105131_llc.fits",
    }
    header = write(tmp_path / "mission-header.json", record)
    return binding, prior, receipts, header, reference(path)


def test_native_sap_reference_retains_raw_flux_clock_and_quality(inputs):
    binding, prior, receipts, header, payload = inputs
    contract = mission.bind_reference(binding, prior, receipts, header, set())
    result = mission.read_reference(contract, payload, set(), set())
    assert result["domain"] == mission.DOMAIN
    assert result["view"] == "MISSION_SAP_REFERENCE"
    assert result["arrays"]["time_bkjd"].tolist() == [120.0, 121.0]
    assert result["arrays"]["sap_flux"].tolist() == [500.0, 510.0]
    assert result["arrays"]["sap_quality"].tolist() == [0, 1]
    assert result["usable_reference"].tolist() == [True, False]
    assert not any("psf" in key or "flatten" in key for key in result["arrays"])
    assert contract["original_kbonus_donor_status"] == "UNAVAILABLE"
    assert contract["full_product_hash_provisioned"] is False


@pytest.mark.parametrize("direction", ["dr3_to_dr2", "dr2_to_dr3"])
def test_ambiguous_gaia_mapping_refused(inputs, direction):
    binding, prior, receipts, header, _ = inputs
    path = receipts[direction]["path"]
    value = mission.read_json(path)
    value["rows"].append(
        {
            "dr3_source_id": 3 if direction == "dr3_to_dr2" else 4,
            "dr2_source_id": 4 if direction == "dr3_to_dr2" else 2,
        }
    )
    receipts[direction] = write(Path(path), value)
    with pytest.raises(ValueError, match="strictly reciprocal"):
        mission.bind_reference(binding, prior, receipts, header, set())


def test_tic_parsed_identity_must_match_original_response(inputs):
    binding, prior, receipts, header, _ = inputs
    path = receipts["tic"]["path"]
    value = mission.read_json(path)
    value["rows"][0]["GAIA"] = "4"
    receipts["tic"] = write(Path(path), value)
    with pytest.raises(ValueError, match="Incomplete exact KIC/TIC"):
        mission.bind_reference(binding, prior, receipts, header, set())


@pytest.mark.parametrize("protection", [{"1"}, {"3"}])
def test_protected_target_or_donor_refused_before_product_access(inputs, protection):
    binding, prior, receipts, header, _ = inputs
    with pytest.raises(ValueError, match="shared or protected"):
        mission.bind_reference(binding, prior, receipts, {"path": "absent"}, protection)


def test_changed_reference_domain_cannot_be_read_as_kbonus(inputs):
    binding, prior, receipts, header, payload = inputs
    contract = mission.bind_reference(binding, prior, receipts, header, set())
    contract["reader_contract"]["view"] = "PSF_FLAT"
    with pytest.raises(ValueError, match="reference contract changed"):
        mission.read_reference(contract, payload, set(), set())


def test_protected_raw_hash_refused_before_fits_parser(inputs, monkeypatch):
    binding, prior, receipts, header, payload = inputs
    contract = mission.bind_reference(binding, prior, receipts, header, set())
    calls = []
    monkeypatch.setattr(fits, "open", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="Protected mission raw input"):
        mission.read_reference(contract, payload, set(), {payload["sha256"]})
    assert calls == []


@pytest.mark.parametrize("change", ["clock", "kic", "column", "header_hash", "url"])
def test_changed_mission_header_or_product_refused(inputs, change):
    _, _, _, header, _ = inputs
    record = mission.read_json(header["path"])
    if change == "clock":
        header = record["headers"][1]
        raw = fits.Header.fromstring(header["raw_header_ascii"], sep="")
        raw["BJDREFI"] = 2400000
        record["headers"][1] = mission.header_record(
            raw.tostring().encode("ascii"), header["offset"]
        )
    elif change == "kic":
        record["catalogued_parent_kic"] = "3230228"
    elif change == "column":
        header = record["headers"][1]
        raw = fits.Header.fromstring(header["raw_header_ascii"], sep="")
        raw["TTYPE4"] = "FLUX_ERR"
        record["headers"][1] = mission.header_record(
            raw.tostring().encode("ascii"), header["offset"]
        )
    elif change == "header_hash":
        record["headers"][0]["sha256"] = "a" * 64
    else:
        record["url"] = record["url"].replace("003230227", "003230228")
    with pytest.raises(ValueError):
        mission.mission_header(record, "3230227")
