"""Exposure-integrated, signed box depths with a local linear baseline."""

import numpy as np


def phase(time, clock):
    p, t0 = clock["period_days"], clock["epoch_bjd_tdb"]
    cycle = np.floor((np.where(np.isfinite(time), time, t0) - t0) / p + 0.5).astype(int)
    return time - (t0 + cycle * p), cycle


def overlap(dt, duration, exposure):
    """Fraction of each exposure inside the transit, including partial cadences."""
    return (
        np.maximum(
            0,
            np.minimum(dt + exposure / 2, duration / 2)
            - np.maximum(dt - exposure / 2, -duration / 2),
        )
        / exposure
    )


def fit_event(dt, flux, error, duration, exposure):
    h = overlap(dt, duration, exposure)
    counts = {
        "cadences": len(dt),
        "in_transit": int((h > 0).sum()),
        "before": int((dt + exposure / 2 < -duration / 2).sum()),
        "after": int((dt - exposure / 2 > duration / 2).sum()),
        "transit_exposure_days": float(np.sum(h * exposure)),
    }
    if len(dt) <= 3 or not np.any(h):
        return {**counts, "status": "NO_TRANSIT_OR_BASELINE_SUPPORT"}, None
    design = np.column_stack((np.ones(len(dt)), dt / duration, -h))
    a, y = design / error[:, None], flux / error
    beta, _, rank, _ = np.linalg.lstsq(a, y, rcond=None)
    if rank != 3:
        return {**counts, "status": "SINGULAR_DESIGN"}, None
    cov = np.linalg.inv(a.T @ a)
    residual = flux - design @ beta
    chi2 = np.sum((residual / error) ** 2)
    report = dict(
        **counts,
        status="MEASURED",
        baseline=float(beta[0]),
        amplitude=float(beta[2]),
        amplitude_error=float(np.sqrt(cov[2, 2])),
        reduced_chi2=float(chi2 / (len(dt) - 3)),
    )
    if beta[0] > 0:
        gradient = np.array([-beta[2] / beta[0] ** 2, 0, 1 / beta[0]])
        sigma = np.sqrt(gradient @ cov @ gradient)
        report.update(
            depth_fraction=float(beta[2] / beta[0]),
            formal_error=float(sigma),
            residual_scaled_error=float(sigma * np.sqrt(report["reduced_chi2"])),
        )
    else:
        report.update(
            depth_fraction=None, formal_error=None, residual_scaled_error=None
        )
    # This linear estimator is also applied to every pixel for difference imaging.
    kernel = (cov @ (design.T / error**2))[2]
    return report, (kernel, beta, residual, h)


def combine(events):
    rows = [
        r
        for r in events
        if r["status"] == "MEASURED"
        and r["depth_fraction"] is not None
        and r["formal_error"] > 0
    ]
    if not rows:
        return {"status": "NO_MEASURABLE_DEPTH", "events": 0}
    d = np.array([r["depth_fraction"] for r in rows])
    w = np.array([r["formal_error"] ** -2 for r in rows])
    w /= w.sum()
    mean = float(w @ d)
    formal = float(
        np.sqrt(np.sum(w**2 * np.array([r["formal_error"] for r in rows]) ** 2))
    )
    scaled = float(
        np.sqrt(
            np.sum(w**2 * np.array([r["residual_scaled_error"] for r in rows]) ** 2)
        )
    )
    # The same transit can straddle sectors or appear in two cameras.
    cycles = {r["cycle"] for r in rows}
    scores = [
        sum(w[i] * (d[i] - mean) for i, r in enumerate(rows) if r["cycle"] == cycle)
        for cycle in cycles
    ]
    # Event-cluster sandwich SE retains within-event correlations; N=1 is undefined.
    cluster = (
        float(np.sqrt(len(cycles) / (len(cycles) - 1) * np.sum(np.square(scores))))
        if len(cycles) > 1
        else None
    )
    return {
        "status": "MEASURED",
        "events": len(cycles),
        "windows": len(rows),
        "depth_fraction": mean,
        "formal_error": formal,
        "residual_scaled_error": scaled,
        "event_scatter_error": cluster,
        "formal_snr": mean / formal,
        "depth_chi2": float(
            sum(((r["depth_fraction"] - mean) / r["formal_error"]) ** 2 for r in rows)
        ),
        "depth_chi2_dof": len(rows) - 1,
    }


def summarize(events):
    primary = [r for r in events if r["phase"] == 0]
    secondary = [r for r in events if r["phase"] == 0.5]
    odd = combine([r for r in primary if r["cycle"] % 2])
    even = combine([r for r in primary if not r["cycle"] % 2])
    difference = None
    scatter_difference = None
    if odd["status"] == even["status"] == "MEASURED":
        error = np.hypot(odd["formal_error"], even["formal_error"])
        difference = (odd["depth_fraction"] - even["depth_fraction"]) / error
        if (
            odd["event_scatter_error"] is not None
            and even["event_scatter_error"] is not None
        ):
            scatter_error = np.hypot(
                odd["event_scatter_error"], even["event_scatter_error"]
            )
            if scatter_error > 0:
                scatter_difference = (
                    odd["depth_fraction"] - even["depth_fraction"]
                ) / scatter_error
    return {
        "primary": combine(primary),
        "secondary_phase_0_5": combine(secondary),
        "odd": odd,
        "even": even,
        "odd_minus_even_formal_sigma": difference,
        "odd_minus_even_event_scatter_sigma": scatter_difference,
    }


def exclusion(time, exposure, clocks):
    mask = np.zeros(len(time), bool)
    for clock in clocks:
        dt, _ = phase(time, clock)
        mask |= overlap(dt, clock["duration_days"], exposure) > 0
        secondary = {
            **clock,
            "epoch_bjd_tdb": clock["epoch_bjd_tdb"] + clock["period_days"] / 2,
        }
        dt, _ = phase(time, secondary)
        mask |= overlap(dt, clock["duration_days"], exposure) > 0
    return mask


def clock_error(clock, cycle):
    covariance = clock.get("covariance_t0_period_days2")
    if covariance is None:
        return None
    vector = np.array([1, cycle])
    return float(np.sqrt(max(0, vector @ np.array(covariance) @ vector)))


def event_windows(data, clock, other_clocks, window_durations):
    time, exposure = data["time"], data["exposure"]
    other = exclusion(time, exposure, other_clocks)
    p, duration = clock["period_days"], clock["duration_days"]
    # Midpoints to the adjacent primary/secondary give disjoint windows.
    half = min(window_durations * duration, p / 4)
    for offset in (0, 0.5):
        shifted = {**clock, "epoch_bjd_tdb": clock["epoch_bjd_tdb"] + offset * p}
        dt, cycles = phase(time, shifted)
        in_event = overlap(dt, duration, exposure) > 0
        good = data["good"] & (~other | in_event)
        for cycle in np.unique(cycles):
            rows = np.flatnonzero(good & (cycles == cycle) & (np.abs(dt) < half))
            meta = {
                "phase": offset,
                "cycle": int(cycle),
                "centre_bjd_tdb": float(shifted["epoch_bjd_tdb"] + cycle * p),
                "clock_sigma_days": clock_error(clock, cycle + offset),
                "half_window_days": half,
                "other_signal_overlap_cadences": int(
                    np.sum(other[rows] & in_event[rows])
                ),
            }
            yield rows, dt[rows], meta


def measure(data, clock, other_clocks, window_durations):
    events, samples = [], []
    shape = data["pixels"].shape[1:]
    diff_num, diff_den = np.zeros(shape), np.zeros(shape)
    for rows, dt, meta in event_windows(data, clock, other_clocks, window_durations):
        event, fitted = fit_event(
            dt,
            data["flux"][rows],
            data["error"][rows],
            clock["duration_days"],
            data["exposure"][rows],
        )
        events.append({**meta, **event})
        if fitted is None or meta["phase"] != 0:
            continue
        kernel, beta, _residual, h = fitted
        amplitude = np.einsum("i,ijk->jk", kernel, data["pixels"][rows])
        variance = np.einsum("i,ijk->jk", kernel**2, data["pixel_error"][rows] ** 2)
        valid = np.isfinite(amplitude) & np.isfinite(variance) & (variance > 0)
        diff_num[valid] += amplitude[valid] / variance[valid]
        diff_den[valid] += 1 / variance[valid]
        baseline = beta[0] + beta[1] * dt / clock["duration_days"]
        valid_flux = baseline > 0
        samples.extend(
            zip(
                rows[valid_flux],
                dt[valid_flux],
                100 * (data["flux"][rows][valid_flux] / baseline[valid_flux] - 1),
                100 * data["error"][rows][valid_flux] / baseline[valid_flux],
                h[valid_flux],
                strict=True,
            )
        )
    difference = np.full(shape, np.nan)
    difference_error = np.full(shape, np.nan)
    np.divide(diff_num, diff_den, out=difference, where=diff_den > 0)
    np.sqrt(
        np.divide(1, diff_den, out=np.full(shape, np.nan), where=diff_den > 0),
        out=difference_error,
    )
    return {
        "events": events,
        "summary": summarize(events),
        "difference_centroid": difference_centroid(difference, difference_error, data),
    }, {
        "samples": np.array(samples).reshape(-1, 5),
        "difference": difference,
        "difference_error": difference_error,
    }


def difference_centroid(difference, error, data):
    """Signed first moment inside the fixed aperture; no positive-pixel clipping."""
    yy, xx = np.indices(difference.shape)
    mask = data["aperture"] & np.isfinite(difference) & np.isfinite(error)
    total = float(difference[mask].sum())
    if not mask.any() or total <= 0:
        return {"status": "NONPOSITIVE_OR_MISSING_DIFFERENCE", "summed_flux": total}
    xy = np.array([np.sum(grid[mask] * difference[mask]) / total for grid in (xx, yy)])
    gradient = np.array(
        [
            (grid[mask] - centre) / total
            for grid, centre in zip((xx, yy), xy, strict=True)
        ]
    )
    cov = (gradient * error[mask] ** 2) @ gradient.T
    return {
        "status": "MEASURED_SIGNED_APERTURE_MOMENT",
        "summed_flux": total,
        "pixel_xy": xy.tolist(),
        "covariance_pixels2": cov.tolist(),
        "offset_pixels": (xy - data["meta"]["target_pixel_xy"]).tolist(),
    }
