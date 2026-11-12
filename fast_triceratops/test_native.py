"""Pinned orbit and full native-curve parity, including the repaired domain tails."""

import contextlib
import hashlib
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import pytest
from batman import _rsky

from fast_triceratops.native import RSKY_SHA256, accelerated_native, sky

SUPPORTED = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.metadata.version("batman-package") == "2.5.3+spacelab.2"
    and hashlib.sha256(Path(_rsky.__file__).read_bytes()).hexdigest() == RSKY_SHA256
)
pytestmark = pytest.mark.skipif(
    not SUPPORTED, reason="requires the reviewed Mac batman wheel"
)


def same(left, right):
    assert np.array_equal(
        np.asarray(left).view(np.uint64), np.asarray(right).view(np.uint64)
    )


@pytest.mark.parametrize("period", [0.837490, 51.5278, 3000.0])
@pytest.mark.parametrize(
    "eccentricity", [0.0, np.nextafter(1e-5, 0), 1e-5, 0.3, 0.95, np.nextafter(1.0, 0)]
)
@pytest.mark.parametrize("transit", [1, 2])
def test_sky_and_true_anomaly_match_pinned_binary(period, eccentricity, transit):
    rng = np.random.default_rng([transit, int(period * 1e6)])
    time = np.ascontiguousarray(
        808.3
        + period * np.r_[rng.uniform(-100, 100, 311), np.linspace(-0.01, 0.01, 1001)]
    )
    for omega in (0.0, np.pi / 2, np.pi, 1.2, 2 * np.pi):
        args = (808.3, period, 60.0, np.deg2rad(89.0), eccentricity, omega, transit, 1)
        same(sky(time, *args), _rsky._rsky(time, *args))
        same(sky(time, *args, true_anomaly=True), _rsky._getf(time, *args))


@pytest.mark.parametrize(
    "period,k,density,eccentricity",
    [
        (40.0, 0.02, 0.3, 0.0),
        (3.0, 0.4, 58.0, 0.0),
        (10.0, 0.08, 1.0, 0.2),
        (0.837490, 0.0138, 0.79, 0.05),
    ],
)
def test_complete_batman_curve_over_known_photons_and_domain_tails(
    period, k, density, eccentricity
):
    import batman

    from fast_triceratops.gold import EXPTIME, FIXTURE

    raw = np.loadtxt(FIXTURE / "lightcurve.csv", delimiter=",")
    params = batman.TransitParams()
    params.t0, params.per, params.rp = 0.0, period, k
    params.a = (
        6.67408e-11 * density * 1409.779824307526 * (period * 86400) ** 2 / (3 * np.pi)
    ) ** (1 / 3)
    params.inc = np.rad2deg(np.arccos(0.2 / params.a))
    params.ecc, params.w = eccentricity, 90.0
    params.u, params.limb_dark = [0.3, 0.2], "quadratic"

    def curve():
        model = batman.TransitModel(
            params, raw[:, 0].copy(), supersample_factor=20, exp_time=EXPTIME
        )
        return model.light_curve(params)

    expected = curve()
    with accelerated_native():
        same(curve(), expected)
    assert np.min(expected) < 1


def test_native_context_restores_after_failure():
    original = _rsky._rsky
    with pytest.raises(RuntimeError, match="control failure"):
        with accelerated_native():
            assert _rsky._rsky is sky
            raise RuntimeError("control failure")
    assert _rsky._rsky is original


@pytest.mark.parametrize(
    "period,k,density,eccentricity",
    [(40.0, 0.02, 0.3, 0.0), (3.0, 0.4, 58.0, 0.0), (10.0, 0.08, 1.0, 0.2)],
)
def test_full_native_control_retains_seeded_posterior(
    tmp_path, period, k, density, eccentricity
):
    from post_boi.test_native_controls import (
        test_real_native_preparation_and_juliet_domain as control,
    )

    outputs = []
    for mode, context in (
        ("stock", contextlib.nullcontext()),
        ("native", accelerated_native()),
    ):
        directory = tmp_path / mode
        directory.mkdir()
        with context:
            control(directory, period, k, density, eccentricity)
        outputs.append(directory)
    for name in ("photons.npz", "prepared/detrended.npz", "native/posterior.npz"):
        with np.load(outputs[0] / name) as a, np.load(outputs[1] / name) as b:
            assert a.files == b.files
            for key in a.files:
                assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                assert a[key].tobytes() == b[key].tobytes(), (name, key)


@pytest.mark.parametrize(
    "parameter,value", [(5, -0.1), (5, 1.0), (5, np.nan), (2, 0.0), (8, 2)]
)
def test_invalid_domain_or_execution_options_fail(parameter, value):
    args = [np.linspace(-0.1, 0.1, 20), 0.0, 2.0, 8.0, 1.5, 0.2, 1.2, 1, 1]
    args[parameter] = value
    with pytest.raises(ValueError):
        sky(*args)
