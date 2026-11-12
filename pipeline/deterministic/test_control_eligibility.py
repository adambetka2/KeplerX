import copy
import io

import numpy as np
import pytest
from astropy.io import fits

from pipeline.deterministic import control_eligibility as audit
from pipeline.deterministic import control_eligibility_archive as archive


def proposal(source_id="1", multiple=1):
    return {
        "source_id": source_id,
        "classification": "TRANSIT_LIKE",
        "period_days": str(1.6958 * multiple),
        "t0_bkjd": str(1.6958 * 71.4),
        "duration_hours": "4.0",
    }


@pytest.fixture
def packet():
    return {
        "selections": {
            "guide_star_family": [
                {"source_id": str(i), "parent_tpf": str(i + 300), "shard": "1"}
                for i in range(1, 101)
            ],
            "contamination_pair": [
                {"source_id": str(i), "parent_tpf": str(i + 300), "shard": "1"}
                for i in range(101, 201)
            ],
        },
        "original_family_rule": {"rule": audit.FAMILY_RULE},
        "original_family_source_ids": [str(i) for i in range(1, 240)],
        "original_family_proposals": [proposal() for _ in range(984)],
        "scientific_acceptance": False,
        "frozen": False,
    }


@pytest.mark.parametrize(
    "change", ["duplicate", "overlap", "missing", "count", "scope", "recipe", "frozen"]
)
def test_reject_changed_selection_or_original_rule(packet, change):
    selected = packet["selections"]["guide_star_family"]
    if change == "duplicate":
        selected[1] = copy.deepcopy(selected[0])
    elif change == "overlap":
        packet["selections"]["contamination_pair"][0] = copy.deepcopy(selected[0])
    elif change == "missing":
        del selected[0]["source_id"]
    elif change == "count":
        selected.pop()
    elif change == "scope":
        packet["selections"]["sealed"] = []
    elif change == "recipe":
        packet["original_family_rule"]["rule"] += " changed"
    else:
        packet["frozen"] = True
    with pytest.raises(ValueError):
        audit.validate_packet(packet)


@pytest.mark.parametrize("multiple", [1, 2, 3])
def test_family_phase_uses_common_base_clock(multiple):
    result = audit.family_ephemeris(proposal(multiple=multiple))
    assert result["period_multiple"] == multiple
    assert result["base_clock_phase"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "field,value",
    [("classification", "OTHER"), ("period_days", "9"), ("t0_bkjd", "1.6958")],
)
def test_reject_proposal_outside_original_family(field, value):
    row = proposal()
    row[field] = value
    with pytest.raises(ValueError, match="family ephemeris"):
        audit.family_ephemeris(row)


def test_catalogue_donor_identity_is_not_scene_attribution(packet):
    selected = packet["selections"]["contamination_pair"][0]
    packet["catalogues"] = [
        {
            "path": "original.json",
            "bytes": 1,
            "sha256": "a" * 64,
            "parent_rows": [{"kepid": int(selected["parent_tpf"]), "koi_period": 10.0}],
        }
    ]
    donor = {"kic": int(selected["parent_tpf"]), "gaia_designation": "Gaia DR3 900"}
    results = audit.donor_bindings(packet, [donor])
    assert len(results) == 100
    assert results[0]["catalogue_parent_binding"] is True
    assert results[0]["public_donor_product_identities"] == [donor]
    assert results[0]["scene_acceptance"] is False
    assert results[0]["target_donor_attribution"] == "NOT_MEASURED"
    assert sum(r["catalogue_parent_binding"] for r in results) == 1


def test_selected_opened_candidate_overlapping_sealed_raw_is_rejected(packet):
    exclusions = {"current": {"source_ids": ["1"]}}
    with pytest.raises(ValueError, match="sealed raw exclusions"):
        audit.reject_protected_sources(packet["selections"], exclusions)


@pytest.mark.parametrize("rows", [[], [{"gaia_designation": "Gaia DR3 1"}] * 2])
def test_missing_or_ambiguous_public_product_rejected(rows):
    with pytest.raises(ValueError):
        audit.unique_sources(rows, {"1"})


def source():
    product = {"kic": 7, "fname": "kic-000000007", "gaia_designation": "Gaia DR3 1"}
    return {
        "source_id": "1",
        "parent_tpf": "8",
        "product": product,
        **audit.product_identity(product),
    }


def headers():
    return [
        {
            "name": "PRIMARY",
            "cards": {"GAIAID": "Gaia DR3 1", "KEPLERID": 7, "TPFORG": 8},
        },
        {"name": "LIGHTCURVE_Q0", "cards": {"QUARTER": 0}},
    ]


def test_global_parent_does_not_resolve_quarter_parent():
    result = audit.header_identity(source(), headers())
    assert result[0]["quarter_parent"] is None
    assert result[0]["global_origin_parent"] == "8"
    assert result[0]["quarter_parent_status"] == "NOT_DECLARED_IN_LIGHTCURVE_HEADER"


@pytest.mark.parametrize(
    "field,value", [("GAIAID", "Gaia DR3 9"), ("KEPLERID", 9), ("TPFORG", 9)]
)
def test_archive_header_rejects_wrong_identity(field, value):
    changed = headers()
    changed[0]["cards"][field] = value
    with pytest.raises(ValueError):
        audit.header_identity(source(), changed)


def test_duplicate_or_mismatched_quarter_rejected():
    changed = headers()
    changed.append(copy.deepcopy(changed[1]))
    with pytest.raises(ValueError, match="Duplicate quarter"):
        audit.header_identity(source(), changed)
    changed = headers()
    changed[1]["cards"]["QUARTER"] = 1
    with pytest.raises(ValueError, match="Quarter extension"):
        audit.header_identity(source(), changed)


def raw_curve(count=120):
    return {
        "time_bkjd": np.arange(count) * 0.02 + 120,
        "cadenceno": np.arange(count),
        "quarter": np.zeros(count, dtype=np.int16),
        "psf_flat_kbonus_flux": np.ones(count) * 10,
        "psf_flat_kbonus_flux_err": np.ones(count),
        "sap_quality": np.zeros(count, dtype=np.int32),
        "flatten_mask": np.zeros(count, dtype=np.int32),
    }


def test_production_masks_measured_separately_and_no_q1_fabricated():
    raw = raw_curve()
    raw["sap_quality"][0] = 1
    raw["flatten_mask"][1] = 1
    raw["psf_flat_kbonus_flux_err"][2] = 0
    result, curve, kept = audit.mask_exposure(raw)
    quarter = result["quarters"][0]
    assert result["status"] == "PREPARATION_ELIGIBLE"
    assert quarter["raw_cadences"] == 120
    assert quarter["finite_positive_error"] == 119
    assert quarter["after_sap_quality_zero"] == 118
    assert quarter["after_flatten_mask_zero"] == 117
    assert result["prepared_cadences"] == 117
    exposure = audit.proposal_exposure(proposal(), curve, kept)
    assert exposure["q0_q1_mask_exposure"][1]["raw_window_cadences"] == 0
    assert exposure["physical_recovery_measured"] is False


def test_insufficient_preparation_retained_without_reselection():
    result, _, _ = audit.mask_exposure(raw_curve(99))
    assert result["status"] == "INSUFFICIENT_PREPARED_CADENCES"
    assert result["prepared_cadences"] == 99
    assert result["prepared_records_sha256"] is None


def test_changed_original_raw_hash_rejected_before_flux_read(tmp_path):
    path = tmp_path / "changed.fits"
    path.write_bytes(b"changed")
    row = {**source(), "pool": {"raw_fits_sha256": "a" * 64}}
    with pytest.raises(ValueError, match="raw FITS hash changed"):
        audit.guide_witness(row, path, [], [])


def test_failed_sources_remain_in_denominator():
    rows = [
        {"cohort": "guide_star_family", "status": "UNAVAILABLE"} for _ in range(100)
    ]
    result = audit.summarize(rows, [])
    assert result["guide_selected"] == 100
    assert result["source_statuses"] == {"UNAVAILABLE": 100}
    assert result["guide_full_raw_hash_verified"] == 0


class Response:
    def __init__(self, data, status=206, content_range=None):
        self.raw = io.BytesIO(data)
        self.status_code = status
        self.headers = {
            "Content-Range": content_range or f"bytes 0-{len(data) - 1}/{len(data)}"
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class Session:
    def __init__(self, body, status=206, wrong_range=False):
        self.body, self.status, self.wrong_range = body, status, wrong_range

    def get(self, url, **kwargs):
        assert url.startswith("https://mast.stsci.edu/api/v0.1/Download/file?")
        assert kwargs["allow_redirects"] is False
        start, end = map(int, kwargs["headers"]["Range"].split("=")[1].split("-"))
        content_range = (
            "bytes 1-3/4"
            if self.wrong_range
            else f"bytes {start}-{end}/{len(self.body)}"
        )
        return Response(self.body[start : end + 1], self.status, content_range)


@pytest.mark.parametrize(
    "status,wrong_range", [(404, False), (200, False), (302, False), (206, True)]
)
def test_range_failures_have_two_closed_attempts(status, wrong_range):
    attempts = []
    with pytest.raises(ValueError, match="Both bounded"):
        archive.range_request(
            Session(b"FITS", status, wrong_range), source(), 0, 3, attempts
        )
    assert len(attempts) == 2
    assert all(r["status"] == status and "error" in r for r in attempts)


def test_noncanonical_product_url_rejected_before_request():
    changed = source()
    changed["url"] = "https://other.example/source.fits"
    with pytest.raises(ValueError, match="Archive URL changed"):
        archive.range_request(Session(b"FITS"), changed, 0, 3, [])


def test_remote_complete_header_inventory_matches_local(tmp_path):
    path = tmp_path / "headers.fits"
    primary = fits.PrimaryHDU()
    primary.header.update({"GAIAID": "Gaia DR3 1", "KEPLERID": 7, "TPFORG": 8})
    quarter = fits.BinTableHDU.from_columns(
        [fits.Column(name="TIME", format="D", array=[1.0, 2.0])], name="LIGHTCURVE_Q0"
    )
    quarter.header["QUARTER"] = 0
    aperture = fits.ImageHDU(np.ones((2, 2)), name="APERTURE_Q0")
    fits.HDUList([primary, quarter, aperture]).writeto(path)
    body = path.read_bytes()
    remote = archive.remote_headers(Session(body), source(), body[:5760], len(body), [])
    assert remote == archive.local_headers(path)
    archive.verify_headers(remote, len(body))
    changed = copy.deepcopy(remote)
    changed[0]["cards"]["TPFORG"] = 9
    with pytest.raises(ValueError, match="header content"):
        archive.verify_headers(changed, len(body))
    with pytest.raises(ValueError, match="incomplete"):
        archive.verify_headers(remote[:-1], len(body))


def test_budget_and_reserve_are_enforced(tmp_path, monkeypatch):
    (tmp_path / "retained").write_bytes(b"1234")
    monkeypatch.setattr(archive, "MAXIMUM_BYTES", 5)
    with pytest.raises(ValueError, match="retained-data budget"):
        archive.capacity(tmp_path, 2)
    monkeypatch.setattr(archive, "MINIMUM_FREE", 2**64)
    with pytest.raises(ValueError, match="free-space reserve"):
        archive.capacity(tmp_path)


def test_null_numeric_catalogue_field_is_explicit_missing():
    assert audit.scalar(np.float64(np.nan)) is None
    assert audit.scalar(b" Gaia DR3 1 ") == "Gaia DR3 1"
