import json
from pathlib import Path

import pytest

from pipeline.deterministic.catalogue_control_report import accepted_harmonic, phase_delta
from pipeline.runtime import EVENT_MATCHING_CONTRACT

FIXTURE = json.loads(Path(__file__).with_name("published-signal-fixture.json").read_text())
RECORDED = [case for case in FIXTURE["cases"] if case["native_proposal"] is not None]


def test_published_fixture_preserves_partial_run_and_measurement_boundaries():
    assert FIXTURE["purpose"] == "PUBLISHED_PERIOD_EPOCH_REGRESSION_ONLY"
    assert FIXTURE["physical_event_recovery"] == "NOT_MEASURED"
    assert FIXTURE["vetting_and_disposition"] == "NOT_MEASURED"
    assert FIXTURE["phase_clock"] == "PUBLISHED_PERIOD"
    assert FIXTURE["stopped_run_checkpoint"]["status"] == "RUNNING"
    assert FIXTURE["artifact_row_counts"]["cohort.csv"] == 122000
    assert FIXTURE["stopped_run_checkpoint"]["accepted_sources"] == 122500
    assert len(FIXTURE["cases"]) == 8
    assert len(RECORDED) == 3
    assert len({case["source_id"] for case in FIXTURE["cases"]}) == 8
    for case in FIXTURE["cases"]:
        assert case["source_id"] == case["published_row"]["source_id"]
        if case["native_proposal"] is None:
            assert case["stopped_run_status"] == "NOT_REACHED"
            assert case["input_binding"] is None
            assert case["expected_ephemeris"] is None


@pytest.mark.parametrize("case", RECORDED, ids=lambda case: case["source_id"])
def test_published_signal_period_and_signed_epoch_regression(case):
    published = case["published_row"]
    proposal = case["native_proposal"]
    expected = case["expected_ephemeris"]
    period = float(published["published_period_days"])
    epoch = float(published["published_epoch_bkjd"])
    native_period = float(proposal["period_days"])
    native_epoch = float(proposal["t0_bkjd"])
    assert proposal["source_id"] == case["source_id"]
    assert case["input_binding"]["source_id"] == case["source_id"]
    assert case["input_binding"]["parent_tpf"] == published["parent_tpf"]
    assert proposal["photometry"] == "psf_flat"
    assert proposal["signal_rank"] == "1"
    assert proposal["classification"] == "TRANSIT_LIKE"
    assert accepted_harmonic(native_period, period) == expected["harmonic"] == 1
    assert abs(native_period / period - 1) == pytest.approx(
        expected["period_error_fraction"], rel=0, abs=1e-14
    )
    offset = phase_delta(native_epoch, epoch, period)
    assert offset == pytest.approx(expected["signed_phase_offset_days"], rel=0, abs=1e-11)
    assert (native_epoch - epoch - offset) / period == pytest.approx(
        expected["published_cycle_at_native_epoch"], rel=0, abs=1e-11
    )
    assert round(offset * 1440, 1) == float(published["stopped_run_epoch_offset_minutes"])
    assert abs(offset) < EVENT_MATCHING_CONTRACT["exposure_days"]


@pytest.mark.parametrize("case", RECORDED, ids=lambda case: case["source_id"])
def test_published_phase_check_distinguishes_a_half_cycle_error(case):
    published = case["published_row"]
    period = float(published["published_period_days"])
    epoch = float(published["published_epoch_bkjd"])
    native_epoch = float(case["native_proposal"]["t0_bkjd"])
    shifted = abs(phase_delta(native_epoch + period / 2, epoch, period))
    assert shifted > EVENT_MATCHING_CONTRACT["exposure_days"]


def test_published_fixture_preserves_extraction_and_aperture_identities():
    cases = {case["source_id"]: case for case in RECORDED}
    source = cases["2078315162505745920"]
    assert source["input_binding"]["parent_tpf"] == "8175105"
    assert source["published_row"]["published_aperture"] == "KIC 8175131"
    assert source["aperture_clarification"]["lc2_object_kic"] == 8175127
    exba = cases["2077240046296834304"]
    assert exba["input_binding"]["parent_tpf"] == "7214804"
    assert exba["aperture_clarification"]["published_aperture"] == (
        "Combined EXBA and KIC 7214804 TPF pixels"
    )
    assert exba["aperture_clarification"]["kic_tpf_quarters"] == [7, 8, 11, 12, 15, 16]
    assert exba["expected_ephemeris"]["published_cycle_at_native_epoch"] == -56
