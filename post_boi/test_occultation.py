import numpy as np

from post_boi.occultation import measure


def light_curve(secondary_ppm, seed=1):
    rng = np.random.default_rng(seed)
    time = np.arange(0, 400, 0.0204)
    quarter = (time // 90).astype(int)
    P, T, D = 3.0, 1.0, 0.1
    phase = (time - T) / P % 1
    flux = 1000 * (1 + quarter * 0.01) * (1 + rng.normal(0, 1e-4, len(time)))
    flux[np.minimum(phase, 1 - phase) < D / P / 2] *= 0.99
    flux[np.abs(phase - 0.5) < D / P / 2] *= 1 - secondary_ppm * 1e-6
    return (
        time,
        flux,
        np.full(len(time), 1e-4),
        np.ones(len(time), bool),
        quarter,
        P,
        T,
        D,
    )


def test_recovers_an_injected_secondary_beside_a_deep_primary():
    r = measure(*light_curve(300))
    assert abs(r["phase_half"]["depth_ppm"] - 300) < 4 * r["phase_half"]["error_ppm"]
    assert r["phase_half"]["snr"] > 10 and r["unresolved"]


def test_no_secondary_stays_resolved():
    r = measure(*light_curve(0))
    assert abs(r["phase_half"]["snr"]) < 4 and not r["unresolved"]
