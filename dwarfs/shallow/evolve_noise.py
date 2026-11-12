"""Blind diagonal noise estimates for paired detector development.

Flux is unchanged. Weighted BLS must use these same errors for its data and
templates, including refinement and final statistics. Duration block scatter
approximates correlated noise; it is neither covariance whitening nor TPS MES.
"""

import numpy as np
from numba import njit

POINT_DEFAULTS = {
    "window_days": 5.0,
    "gap_days": 0.5,
    "point_prior_samples": 64,
    "minimum_scale_fraction": 0.5,
    "maximum_scale_fraction": 2.0,
}
BLOCK_DEFAULTS = {
    "block_prior_samples": 12,
    "block_maximum_scale_fraction": 4.0,
}
REFERENCES = {
    "duration_dependent_noise": "https://arxiv.org/abs/1604.06140",
    "weighted_boxes": "https://docs.astropy.org/en/stable/timeseries/bls.html",
}


def settings_values(settings):
    method = settings["method"]
    if method not in ("local_point", "duration_block"):
        raise ValueError(f"Unknown experimental noise method: {method}")
    defaults = POINT_DEFAULTS | (BLOCK_DEFAULTS if method == "duration_block" else {})
    unknown = set(settings) - set(defaults) - {"method"}
    if unknown:
        raise ValueError(f"Unknown experimental noise settings: {sorted(unknown)}")
    selected = defaults | settings
    for name in defaults:
        value = selected[name]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Experimental noise requires finite positive {name}")
        if name.endswith("prior_samples") and type(value) is not int:
            raise ValueError(f"Experimental noise requires integer {name}")
    if (
        not selected["minimum_scale_fraction"]
        <= 1
        <= selected["maximum_scale_fraction"]
    ):
        raise ValueError("Point scale bounds must contain one")
    if method == "duration_block" and selected["block_maximum_scale_fraction"] < 1:
        raise ValueError("Duration block scale maximum must be at least one")
    return selected


def checked_series(time, flux, quarter, base_noise, duration_days):
    arrays = [np.asarray(value) for value in (time, flux, quarter, base_noise)]
    if any(value.ndim != 1 for value in arrays) or not len(arrays[0]):
        raise ValueError("Noise estimates require nonempty one-dimensional inputs")
    if any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("Noise input arrays must have identical shapes")
    if not np.issubdtype(arrays[2].dtype, np.integer):
        raise ValueError("Noise quarter labels must be integers")
    if any(arrays[index].dtype.kind not in "iuf" for index in (0, 1, 3)):
        raise ValueError("Noise clocks, flux and errors must be real numeric arrays")
    time, flux, quarter, base_noise = arrays
    time, flux, base_noise = (
        np.asarray(value, dtype=np.float64) for value in (time, flux, base_noise)
    )
    if not all(np.isfinite(value).all() for value in (time, flux, base_noise)):
        raise ValueError("Noise clocks, flux and base errors must be finite")
    if np.any(np.diff(time) < 0) or np.any(base_noise <= 0):
        raise ValueError("Noise requires sorted clocks and positive base errors")
    if (
        isinstance(duration_days, bool)
        or not np.isfinite(duration_days)
        or duration_days <= 0
    ):
        raise ValueError("Noise duration must be finite and positive")
    return time, flux, quarter, base_noise


@njit(cache=True)
def window_mad(time, values, segment, window_days):
    scatter = np.empty(len(time), dtype=np.float64)
    counts = np.empty(len(time), dtype=np.int64)
    lower = upper = 0
    for index in range(len(time)):
        while (
            segment[lower] != segment[index]
            or time[lower] < time[index] - window_days / 2
        ):
            lower += 1
        upper = max(upper, index)
        while (
            upper < len(time)
            and segment[upper] == segment[index]
            and time[upper] <= time[index] + window_days / 2
        ):
            upper += 1
        sample = values[lower:upper]
        counts[index] = len(sample)
        scatter[index] = 1.4826 * np.median(np.abs(sample - np.median(sample)))
    return scatter, counts


def shrink_scale(scatter, counts, prior, lower, upper):
    variance = (counts * scatter**2 + prior) / (counts + prior)
    scale = np.sqrt(variance)
    if not np.isfinite(scale).all():
        raise ValueError("Experimental noise produced nonfinite local variance")
    return np.clip(scale, lower, upper)


def point_estimate(time, flux, quarter, base_noise, selected, cache):
    keys = tuple((name, selected[name]) for name in POINT_DEFAULTS)
    inputs = (time, flux, quarter, base_noise)
    saved = cache.get("point") if cache is not None else None
    if saved is not None:
        if saved["settings"] != keys or any(
            not np.array_equal(value, original)
            for value, original in zip(inputs, saved["inputs"])
        ):
            raise ValueError(
                "Caller noise cache belongs to different inputs or settings"
            )
        return saved["estimate"]
    boundaries = np.r_[
        True,
        (quarter[1:] != quarter[:-1]) | (np.diff(time) > selected["gap_days"]),
    ]
    segment = np.cumsum(boundaries, dtype=np.int64) - 1
    standardized = flux / base_noise
    if not np.isfinite(standardized).all():
        raise ValueError("Experimental noise produced nonfinite standardized flux")
    scatter, counts = window_mad(time, standardized, segment, selected["window_days"])
    scale = shrink_scale(
        scatter,
        counts,
        selected["point_prior_samples"],
        selected["minimum_scale_fraction"],
        selected["maximum_scale_fraction"],
    )
    estimate = {
        "segment": segment,
        "point_raw_mad": scatter,
        "point_count": counts,
        "point_scale": scale,
    }
    if cache is not None:
        cache["point"] = {
            "settings": keys,
            "inputs": tuple(value.copy() for value in inputs),
            "estimate": estimate,
        }
    return estimate


def duration_blocks(time, flux, point_noise, segment, duration_days):
    starts = np.flatnonzero(np.r_[True, np.diff(segment) != 0])
    start_for_row = starts[segment]
    phase = (time - time[start_for_row]) / duration_days
    if not np.isfinite(phase).all() or np.any(phase >= np.iinfo(np.int64).max):
        raise ValueError("Experimental noise duration block index exceeds capacity")
    labels = np.floor(phase).astype(np.int64)
    new_block = np.r_[True, (np.diff(segment) != 0) | (np.diff(labels) != 0)]
    first = np.flatnonzero(new_block)
    membership = np.cumsum(new_block, dtype=np.int64) - 1
    sizes = np.diff(np.r_[first, len(time)])
    standardized = flux / point_noise
    for lower, upper in zip(starts, np.r_[starts[1:], len(time)]):
        standardized[lower:upper] -= np.median(standardized[lower:upper])
    values = np.add.reduceat(standardized, first) / np.sqrt(sizes)
    block_time = np.add.reduceat(time, first) / sizes
    if not np.isfinite(values).all() or not np.isfinite(block_time).all():
        raise ValueError("Experimental noise produced nonfinite duration block sums")
    return {
        "block_time_days": block_time,
        "block_segment": segment[first],
        "block_size": sizes,
        "block_standardized_sum": values,
        "block_membership": membership,
    }


def block_estimate(time, flux, base_noise, duration_days, selected, point):
    blocks = duration_blocks(
        time, flux, base_noise * point["point_scale"], point["segment"], duration_days
    )
    scatter, counts = window_mad(
        blocks["block_time_days"],
        blocks["block_standardized_sum"],
        blocks["block_segment"],
        selected["window_days"],
    )
    scale = shrink_scale(
        scatter,
        counts,
        selected["block_prior_samples"],
        1.0,
        selected["block_maximum_scale_fraction"],
    )
    return blocks | {
        "block_raw_mad": scatter,
        "block_count": counts,
        "block_scale": scale,
    }


def describe(values):
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def noise_weights(
    time,
    flux,
    quarter,
    base_noise,
    duration_days,
    settings,
    diagnostics=None,
    cache=None,
):
    selected = settings_values(settings)
    time, flux, quarter, base_noise = checked_series(
        time, flux, quarter, base_noise, duration_days
    )
    point = point_estimate(time, flux, quarter, base_noise, selected, cache)
    arrays = dict(point)
    scale = point["point_scale"].copy()
    if selected["method"] == "duration_block":
        blocks = block_estimate(time, flux, base_noise, duration_days, selected, point)
        arrays.update(blocks)
        scale *= blocks["block_scale"][blocks["block_membership"]]
    noise = base_noise * scale
    if not np.isfinite(noise).all() or np.any(noise <= 0):
        raise ValueError("Experimental noise produced nonpositive or nonfinite errors")
    if diagnostics is not None:
        diagnostics.update(
            {
                "settings": selected,
                "metric": "Diagonal weighted boxes; flux waveform unchanged",
                "variance_shrinkage": "(n * local_MAD_squared + prior) / (n + prior)",
                "references": REFERENCES,
                "duration_days": float(duration_days),
                "cadences": len(time),
                "segments": int(point["segment"][-1] + 1),
                "base_noise": describe(base_noise),
                "noise": describe(noise),
                "point_scale": describe(point["point_scale"]),
                "point_bound_counts": {
                    "minimum": int(
                        np.sum(
                            point["point_scale"] == selected["minimum_scale_fraction"]
                        )
                    ),
                    "maximum": int(
                        np.sum(
                            point["point_scale"] == selected["maximum_scale_fraction"]
                        )
                    ),
                },
                "arrays": arrays | {"dy": noise.copy()},
            }
        )
        if selected["method"] == "duration_block":
            diagnostics["block_scale"] = describe(blocks["block_scale"])
            diagnostics["blocks"] = len(blocks["block_size"])
            diagnostics["block_bound_counts"] = {
                "minimum": int(np.sum(blocks["block_scale"] == 1.0)),
                "maximum": int(
                    np.sum(
                        blocks["block_scale"]
                        == selected["block_maximum_scale_fraction"]
                    )
                ),
            }
    return noise
