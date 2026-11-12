"""Occultation (secondary-eclipse) depth from native photons, at phase 0.5 and over a phase scan."""

import sys
from pathlib import Path

import numpy as np

from validator.records import receipt, require, write


def event_depth(x, f, s, D):
    """Depth fraction and error of one event against a weighted linear baseline 1-3 durations either side."""
    inside, side = np.abs(x) < D / 2, (np.abs(x) > D) & (np.abs(x) < 3 * D)
    if inside.sum() < 2 or (x[side] < 0).sum() < 3 or (x[side] > 0).sum() < 3:
        return None
    A, w = np.column_stack([np.ones(side.sum()), x[side]]), 1 / s[side] ** 2
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    beta = cov @ (A.T @ (w * f[side]))
    base = beta[0] + beta[1] * x[inside]
    wi = (base / s[inside]) ** 2
    depth = np.sum(wi * (1 - f[inside] / base)) / np.sum(wi)
    xm = np.average(x[inside], weights=wi)
    base_error = np.sqrt(cov[0, 0] + 2 * xm * cov[0, 1] + xm**2 * cov[1, 1]) / np.mean(
        base
    )
    return depth, np.hypot(1 / np.sqrt(np.sum(wi)), base_error)


def combined(t, f, s, P, T, D, phase):
    k = np.arange(np.floor((t[0] - T) / P) - 1, np.ceil((t[-1] - T) / P) + 1)
    events = []
    for c in T + (k + phase) * P:
        lo, hi = np.searchsorted(t, [c - 3 * D, c + 3 * D])
        e = event_depth(t[lo:hi] - c, f[lo:hi], s[lo:hi], D) if hi - lo > 8 else None
        if e is not None:
            events.append(e)
    if not events:
        return None
    d, e = np.array(events).T
    w = 1 / e**2
    depth, error = np.sum(w * d) / np.sum(w), 1 / np.sqrt(np.sum(w))
    return {
        "phase": float(phase),
        "depth_ppm": float(depth * 1e6),
        "error_ppm": float(error * 1e6),
        "snr": float(depth / error),
        "events": len(events),
        "reduced_chi2": float(np.sum(w * (d - depth) ** 2) / max(1, len(events) - 1)),
    }


def measure(time, flux, sigma, valid, quarter, P, T, D):
    ok = valid & np.isfinite(sigma) & np.isfinite(flux)
    f = np.array(flux, dtype=float)
    for q in np.unique(quarter[ok]):
        m = ok & (quarter == q)
        f[m] /= np.median(f[m])
    keep = ok & ~(np.abs((time - T + P / 2) % P - P / 2) < D)
    order = np.argsort(time[keep])
    t, f, s = time[keep][order], f[keep][order], sigma[keep][order]
    half = combined(t, f, s, P, T, D, 0.5)
    require(half is not None, "No measurable phase-0.5 events")
    scan = [
        r
        for r in (
            combined(t, f, s, P, T, D, x) for x in np.arange(0.1, 0.9 + 1e-9, D / P / 4)
        )
        if r is not None
    ]
    return {
        "phase_half": half,
        "scan_maximum": max(scan, key=lambda r: r["snr"]),
        "independent_trial_phases": int(0.8 * P / D),
        "unresolved": half["snr"] >= 3,
    }


def main(photons, noise, P, T, D, out):
    with np.load(photons) as z, np.load(noise) as n:
        require(
            np.array_equal(z["time_bkjd"], n["time_bkjd"]),
            "Noise is not aligned with the photons",
        )
        result = measure(
            z["time_bkjd"], z["flux"], n["sigma"], z["valid"], z["quarter"], P, T, D
        )
    write(
        out,
        dict(
            clock={"period_days": P, "epoch_bkjd": T, "duration_days": D},
            photons=receipt(photons),
            noise=receipt(noise),
            helper=receipt(__file__),
            method=__doc__.strip(),
            **result,
        ),
    )
    return result


if __name__ == "__main__":
    print(
        main(
            Path(sys.argv[1]),
            Path(sys.argv[2]),
            *map(float, sys.argv[3:6]),
            Path(sys.argv[6]),
        )
    )
