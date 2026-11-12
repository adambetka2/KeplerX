"""Protect domain, real measurement errors, source identity and complete evidence."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from post_boi.campaign import unfavorable
from post_boi.native import duration_samples, stellar_density
from post_boi.prepare import detrend, fold, load_protection
from post_boi.protocol import declared_clocks, statistics
from post_boi.records import write
from post_boi.scene import KeplerScene


def test_revised_protection_binds_source_and_original_nan_gap_times(tmp_path):
    path = tmp_path / "protection.npz"
    time = np.array([10.0, 10.02, np.nan, 10.06])
    protected = np.array([False, True, False, False])
    np.savez(path, source_id="123", time_bkjd=time, protected=protected)
    c = dict(source_id="123", protection_mask=dict(path=path, basis="Observed contact"))
    np.testing.assert_array_equal(load_protection(c, dict(time_bkjd=time)), protected)
    c["source_id"] = "456"
    with pytest.raises(ValueError, match="Protection source differs"):
        load_protection(c, dict(time_bkjd=time))
    c["source_id"] = "123"
    with pytest.raises(ValueError, match="Protection photon times differ"):
        load_protection(c, dict(time_bkjd=time + 1))


def test_long_duration_and_dense_small_host_are_supported():
    priors = {"ecc_p1": {"hyperparameters": 0.0}, "omega_p1": {"hyperparameters": 90.0}}
    long = dict(
        P_p1=np.array([40.0]),
        p_p1=np.array([0.02]),
        b_p1=np.array([0.2]),
        rho=np.array([0.3 * 1409.779824307526]),
    )
    dense = dict(
        P_p1=np.array([3.0]),
        p_p1=np.array([0.4]),
        b_p1=np.array([0.2]),
        rho=np.array([58.0 * 1409.779824307526]),
    )
    assert duration_samples(long, priors)[0] * 24 > 6
    assert np.isfinite(duration_samples(dense, priors)).all()


def test_density_uses_reported_errors_and_does_not_invent_missing_errors():
    mass = dict(value=0.1, error=0.003)
    radius = dict(value=0.12, error=0.002)
    result = stellar_density(mass, radius)
    centre, sigma = result["hyperparameters"]
    assert centre / 1409.779824307526 > 50
    assert sigma == pytest.approx(centre * np.hypot(0.003 / 0.1, 3 * 0.002 / 0.12))
    with pytest.raises(ValueError, match="reported stellar errors"):
        stellar_density(dict(value=0.1, error=None), radius)


def test_duration_scaled_preparation_preserves_long_transit_and_both_flux_conventions():
    time = np.arange(-3, 3, 0.020433598216918748)
    signal = dict(period_days=40.0, epoch_bkjd=0.0, duration_days=0.4)
    source = 1 - 0.02 * (abs(time) < 0.2)
    baseline = 1000 * (1 + 0.01 * time)
    outputs = []
    for fraction in [1.0, 0.6]:
        a = dict(
            time_bkjd=time,
            flux=(1 + fraction * (source - 1)) * baseline,
            flux_err=np.full(len(time), 0.1),
            quarter=np.full(len(time), 3),
            valid=np.ones(len(time), bool),
        )
        d = detrend(a, [signal], 0.020433598216918748, 2.0, 2)
        np.testing.assert_allclose(d["flux"], 1 + fraction * (source - 1), atol=1e-10)
        assert len(d["flux"]) == len(time)
        outputs.append(d)
    np.testing.assert_allclose(
        1 + (outputs[1]["flux"] - 1) / 0.6, outputs[0]["flux"], atol=1e-10
    )
    clock = dict(signal, observed_depth_fraction=0.02)
    packet = fold(outputs[0], clock, [], 0.020433598216918748)
    assert len(packet["flux"]) <= 200
    assert packet["original_index"].size > 0


def scene_files(tmp_path, missing=False, measured=False):
    stars = pd.DataFrame(
        dict(
            ID=["2102521666898377600", "2102521666898377601"],
            Tmag=[15.0, 15.0 + 2.5 * np.log10(4)],
            mass=[0.1, np.nan if missing else 0.5],
            rad=[0.12, 0.5],
            Teff=[3000.0, 3800.0],
            plx=[4.0, 1.0],
            ra=[284.0, 284.0],
            dec=[40.0, 40.0],
            Jmag=[14.0, 15.0],
            Hmag=[13.0, 14.0],
            Kmag=[12.0, 13.0],
        )
    )
    csv = tmp_path / "stars.csv"
    stars.to_csv(csv, index=False)
    ap = np.array([[x, y] for x in range(5) for y in range(5)], float)
    geometry = tmp_path / "geometry.npz"
    values = dict(
        source_id=stars.ID.to_numpy(dtype="U"),
        quarter=np.array([3]),
        xy_q3=np.array([[2.0, 2.0], [2.0, 2.0]]),
        aperture_q3=ap,
    )
    if measured:
        values["flux_fraction_q3"] = np.array([0.8, 0.2])
    np.savez(geometry, **values)
    background = tmp_path / "background.csv"
    background.write_text(
        "Provided upstream population; this test exercises depth/scene only.\n"
    )
    return csv, geometry, background


@pytest.mark.parametrize(
    "response", ["upstream_gaussian", "measured_kepler", "calibrated_kepler"]
)
def test_gaia_only_scene_retains_neighbor_and_sap_dilution(tmp_path, response):
    paths = scene_files(tmp_path, measured=response != "upstream_gaussian")
    target = KeplerScene(*paths, "2102521666898377600", response)
    target.prepare_depths(0.008, False)
    np.testing.assert_allclose(target.stars.fluxratio, [0.8, 0.2], atol=1e-12)
    np.testing.assert_allclose(target.stars.tdepth, [0.01, 0.04], atol=1e-12)
    assert target.mission == "Kepler" and len(target.stars) == 2


def test_missing_possible_contaminant_is_not_silently_dropped(tmp_path):
    paths = scene_files(tmp_path, missing=True)
    target = KeplerScene(*paths, "2102521666898377600", "upstream_gaussian")
    with pytest.raises(ValueError, match="Missing mass"):
        target.prepare_depths(0.008, False)
    assert len(target.stars) == 2


def test_wrong_source_is_not_replaced_by_parent_target(tmp_path):
    paths = scene_files(tmp_path)
    with pytest.raises(ValueError, match="authenticated source"):
        KeplerScene(*paths, "123", "upstream_gaussian")


def test_rejected_infinite_values_are_explicit_and_results_immutable(tmp_path):
    p = tmp_path / "result.json"
    write(p, dict(attempts=[dict(logp=-np.inf), dict(logp=-127.98)]))
    assert json.loads(p.read_text())["attempts"][0]["logp"] == {"nonfinite": "-inf"}
    with pytest.raises(FileExistsError):
        write(p, dict(replaced=True))


def test_screen_and_complete_statistics_keep_unfavorable_repeats():
    assert unfavorable([dict(FPP=0.1, NFPP=0), dict(FPP=0.2, NFPP=0)])
    assert not unfavorable([dict(FPP=0.1, NFPP=0), dict(FPP=0.01, NFPP=0)])
    results = [dict(FPP=0.001, NFPP=0.0) for _ in range(20)]
    results[3]["FPP"] = 0.02
    report = statistics(results)
    assert report["FPP"]["unfavorable_indices"].tolist() == [3]
    assert len(report["FPP"]["repeats"]) == 20


def test_recovery_changes_clock_only():
    c = dict(recovery_diagnostics=3, recovered=dict(period_days=3.01, epoch_bkjd=1.0))
    final = dict(
        period_days=3.0,
        epoch_bkjd=1.01,
        duration_days=0.2,
        observed_depth_fraction=0.01,
    )
    before = copy.deepcopy(final)
    result = declared_clocks(c, final)
    assert final == before
    assert result["recovered"]["duration_days"] == final["duration_days"]
    assert (
        result["recovered"]["observed_depth_fraction"]
        == final["observed_depth_fraction"]
    )
