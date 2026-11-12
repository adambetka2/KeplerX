"""Adapt prepared arrays to the unchanged upstream LightCurve and Dump APIs."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from common import require
from detector.gap_padding import Segments
from detector.repeat_rule import Repeats

REFERENCES = ("native", "compiled")
# Template bank from two cadences (0.98 h) in upstream's 0.25-1.5 duration window (docs/keplerX.md).
MIN_TEMPLATE_CADENCES = 2
DURATION_WINDOW = (0.25, 1.5)
# Upstream's period step (d_leastsq 0.075) times this.
PERIOD_STEP_FACTOR = 2.0
# upstream's repeat tolerance, which the references keep.
UPSTREAM_REPEAT_TOLERANCE = 1e-4


def template_bank(time, exptime, density, min_transits):
    """upstream's _calc_best_tdurs with its two limits set."""
    baseline = np.ptp(time)
    max_period = baseline / min_transits
    max_dur = min(
        13 / 24 * (max_period / 365) ** (1 / 3) * density ** (-1 / 3)
        * DURATION_WINDOW[1],
        1.5,
    )  # fmt: skip
    tdurs = np.unique(exptime * 1.1 ** np.arange(2, 100) // exptime) * exptime
    shortest = tdurs > (MIN_TEMPLATE_CADENCES - 0.5) * exptime
    return tdurs[(tdurs < max_dur) & shortest]


def make_dump(arrays, metadata, settings, stage="detector", numerical_threads=1,
              reference=None):  # fmt: skip
    """The Detector's Dump of a prepared curve for `stage` (its own upstream copy)."""
    require(
        isinstance(numerical_threads, int) and numerical_threads >= 1,
        "Numerical threads must be a positive integer",
    )
    require(
        reference is None or reference in REFERENCES, f"Unknown reference {reference}"
    )
    upstream = importlib.import_module(f"{stage}.upstream.dump")
    if reference == "native":
        module = SimpleNamespace(
            Dump=type(
                "NativeDump",
                (Repeats, Segments, upstream.Dump),
                {"_upstream": upstream},
            )
        )
        construction = nullcontext()
        sampling = upstream.kep_period_sampling
    else:
        from detector.compiled_search import (
            deferred_period_sampling,
            dump_class,
            kep_period_sampling,
        )

        module = SimpleNamespace(Dump=dump_class(stage))
        construction = deferred_period_sampling(upstream)
        sampling = kep_period_sampling
    lightcurves = importlib.import_module(f"{stage}.upstream.lightcurves")
    stellar = metadata["stellar"]
    for name in ("teff", "logg", "density_solar"):
        require(np.isfinite(stellar[name]), f"Declared stellar {name} required")
    require(stellar["density_solar"] > 0, "Stellar density must be positive")
    good = arrays["mask"].astype(bool)
    time, flux, error = (
        np.asarray(arrays[key][good], dtype=float) for key in ("time", "flux", "error")
    )
    # Upstream uses numeric display IDs; the adapter retains Gaia as a string.
    lightcurve = lightcurves.LightCurve(
        time,
        flux,
        error,
        ID=0,
        flags=np.zeros(len(time), dtype=np.int64),
        mask=np.ones(len(time), dtype=bool),
        trend=np.ones(len(time)),
        mission="KEPLER",
    )
    lightcurve.exptime = metadata["exposure_days"]
    bank = template_bank(
        time, lightcurve.exptime, stellar["density_solar"],
        settings["min_observed_transits"],
    )  # fmt: skip
    with construction:
        dump = module.Dump(
            lightcurve,
            tdurs=bank,
            star_logg=stellar["logg"],
            star_teff=stellar["teff"],
            star_density=stellar["density_solar"],
            min_transits=settings["min_observed_transits"],
            P_range=[settings["period_min_days"], settings["period_max_days"]],
        )
    dump.whitening_threads = numerical_threads
    # Upstream's sampling with its step scaled by PERIOD_STEP_FACTOR, compiled (bitwise).
    periods = sampling(
        time=dump.lc.time[dump.lc.mask], P_min=None, n_tr=dump.min_transits,
        rho_star=dump.star_density, d_leastsq=0.075 * PERIOD_STEP_FACTOR,
    )  # fmt: skip
    low, high = settings["period_min_days"], settings["period_max_days"]
    dump.search_periods = periods[(periods < high) & (periods > low)]
    dump.reference = reference
    if reference is not None:
        dump.repeat_tolerance = UPSTREAM_REPEAT_TOLERANCE
    return dump
