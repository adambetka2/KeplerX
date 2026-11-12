"""KBonus preparation for the joint quarter likelihood: flux closure and reviewed noise."""

import numpy as np

from validator.records import require

CLOSURE_TOLERANCE = 0.2


def closure(throughput, relative_flux, target_psf_flux, sap_flux):
    """Predicted over observed SAP flux; source fluxes scale the target's PSF flux."""
    finite = np.isfinite(target_psf_flux) and np.isfinite(sap_flux)
    require(
        not finite or (target_psf_flux > 0 and sap_flux > 0),
        "Closure needs positive fluxes",
    )
    return float(
        target_psf_flux * np.sum(np.asarray(throughput) * relative_flux) / sap_flux
    )


def passing_quarters(closures, tolerance=CLOSURE_TOLERANCE):
    return sorted(
        q for q, c in closures.items() if np.isfinite(c) and abs(c - 1) <= tolerance
    )


def red_noise_beta(residual, time, sizes):
    """Median binned-scatter ratio over bins without gaps longer than two time steps; at least one."""
    sigma, step, ratios = np.std(residual), np.median(np.diff(time)), []
    for n in sizes:
        means = [
            residual[s : s + n].mean()
            for s in range(0, len(residual) - n + 1, n)
            if time[s + n - 1] - time[s] <= (n + 1) * step
        ]
        if len(means) >= 10:
            expected = sigma / np.sqrt(n) * np.sqrt(len(means) / (len(means) - 1))
            ratios.append(float(np.std(means) / expected))
    return max(1.0, float(np.median(ratios))) if ratios else 1.0, ratios


def quarter_noise(flux, flux_err, time, quarter, transits, cadence, duration):
    """Per-photon reviewed sigma and the per-quarter scales that produced it."""
    sizes = sorted(
        {max(2, round(k)) for k in np.geomspace(1, max(2, duration / cadence), 6)}
    )
    scale = np.full(len(flux), np.nan)
    rows = []
    for q in np.unique(quarter):
        ix = np.flatnonzero((quarter == q) & ~transits)
        require(len(ix) > 100, f"Too few out-of-transit cadences in quarter {q}")
        z = (flux[ix] - 1) / flux_err[ix]
        robust = 1.4826 * np.median(np.abs(z - np.median(z)))
        beta, ratios = red_noise_beta(flux[ix] - 1, time[ix], sizes)
        scale[quarter == q] = max(1.0, robust) * beta
        rows.append(
            {
                "quarter": int(q),
                "robust_normalized_scatter": float(robust),
                "beta": beta,
                "beta_ratios": ratios,
                "bin_sizes": sizes,
                "applied_scale": float(max(1.0, robust) * beta),
            }
        )
    return flux_err * scale, rows
