"""post_boi.prepare's duration-aware preparation, without importing Lightkurve."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from validator.records import candidate, receipt, require, write


def phase(time, period, epoch):
    return (time - epoch + period / 2) % period - period / 2


def native_arrays(path, source_id):
    with np.load(path, allow_pickle=False) as archive:
        a = dict(archive)
    require(str(a["source_id"].item()) == source_id, "Native source identity differs")
    names = ("time_bkjd", "flux", "flux_err", "quarter", "valid")
    require(all(a[n].ndim == 1 for n in names), "Native arrays must be one dimensional")
    require(len({a[n].shape for n in names}) == 1, "Native array shapes differ")
    require(a["valid"].dtype == bool, "Supply the authenticated boolean quality mask")
    finite = np.isfinite(a["time_bkjd"]) & np.isfinite(a["flux"])
    finite &= np.isfinite(a["flux_err"]) & (a["flux_err"] > 0)
    finite &= np.isfinite(a["quarter"])
    require(np.all(finite[a["valid"]]), "Accepted native photons have invalid values")
    return a


def transit_mask(time, signals, exposure):
    mask = np.zeros(len(time), dtype=bool)
    for signal in signals:
        p, t, d = (signal[k] for k in ("period_days", "epoch_bkjd", "duration_days"))
        require(np.isfinite([p, t, d]).all() and 0 < d < p, "Invalid supplied clock")
        mask |= abs(phase(time, p, t)) <= d + exposure / 2
    return mask


def flatten_trend(time, flux, window_length, polyorder, protect, break_tolerance=5,
                  niters=3, sigma=3):  # fmt: skip
    """Lightkurve 2.6.0 LightCurve.flatten(..., mask=protect, return_trend=True)."""
    mask = ~protect
    extra_mask = np.isfinite(flux)
    extra_mask &= np.nan_to_num(np.abs(flux - np.nanmedian(flux))) <= (
        np.nanstd(flux) * sigma
    )
    mask &= extra_mask
    for _ in np.arange(0, niters):
        if polyorder >= window_length:
            polyorder = window_length - 1
        dt = time[mask][1:] - time[mask][0:-1]
        cut = np.where(dt > break_tolerance * np.nanmedian(dt))[0] + 1
        low = np.append([0], cut)
        high = np.append(cut, len(time[mask]))
        trend_signal = np.zeros(len(time[mask]))
        for lo, hi in zip(low, high):
            if np.any([window_length > (hi - lo), (hi - lo) < break_tolerance]):
                trend_signal[lo:hi] = np.nanmedian(flux[mask][lo:hi])
            else:
                trend_signal[lo:hi] = savgol_filter(
                    x=flux[mask][lo:hi],
                    window_length=window_length,
                    polyorder=polyorder,
                )
        mask1 = np.nan_to_num(np.abs(flux[mask] - trend_signal)) < (
            np.nanstd(flux[mask] - trend_signal) * sigma + 1e-14
        )
        f = interp1d(time[mask][mask1], trend_signal[mask1], fill_value="extrapolate")
        trend_signal = f(time)
        mask[mask] &= mask1
    return trend_signal


def detrend(
    arrays, signals, exposure, window_days, polyorder, additional_protection=None
):
    require(
        window_days is not None and window_days > 0,
        "Decide detrending window per candidate",
    )
    require(type(polyorder) is int and polyorder >= 0, "Invalid polynomial order")
    selected = np.flatnonzero(arrays["valid"])
    a = {k: arrays[k][selected] for k in ("time_bkjd", "flux", "flux_err", "quarter")}
    a["original_index"] = selected
    trend = np.empty(len(selected))
    protect = transit_mask(a["time_bkjd"], signals, exposure)
    if additional_protection is not None:
        require(
            additional_protection.dtype == bool
            and additional_protection.shape == arrays["valid"].shape,
            "Additional protection must align with original photons",
        )
        protect |= additional_protection[selected]
    for quarter in np.unique(a["quarter"]):
        ix = np.flatnonzero(a["quarter"] == quarter)
        require(
            np.all(np.diff(a["time_bkjd"][ix]) > 0),
            "Native times must increase within quarter",
        )
        require(
            np.sum(~protect[ix]) > polyorder,
            "Insufficient unmasked baseline for detrending",
        )
        cadence = np.median(np.diff(a["time_bkjd"][ix]))
        window = int(np.ceil(window_days / cadence)) | 1
        require(window > polyorder, "Detrending window does not support its polynomial")
        trend[ix] = flatten_trend(
            a["time_bkjd"][ix], a["flux"][ix], window, polyorder, protect[ix].copy()
        )
    require(np.isfinite(trend).all() and np.all(trend > 0), "Invalid fitted baseline")
    a["flux"] = a["flux"] / trend
    a["flux_err"] = a["flux_err"] / trend
    a["trend"], a["protected"] = trend, protect
    return a


def load_protection(c, arrays):
    protection = c.get("protection_mask")
    if protection is None:
        return None
    require(bool(protection["basis"]), "Justify the revised transit protection")
    with np.load(protection["path"], allow_pickle=False) as archive:
        require(
            str(archive["source_id"].item()) == c["source_id"],
            "Protection source differs",
        )
        require(
            np.array_equal(archive["time_bkjd"], arrays["time_bkjd"], equal_nan=True),
            "Protection photon times differ",
        )
        return archive["protected"].copy()


def prepare(config_path, output):
    c, out = candidate(config_path), Path(output)
    require(
        c["flux_convention"] in ("raw_aperture", "source_corrected", "psf_flat"),
        "Declare dilution convention",
    )
    require(
        bool(c["signal_id"]) and bool(c["input_basis"]),
        "Bind exact signal and photometry basis",
    )
    exposure = c["exposure_days"]
    require(exposure is not None and exposure > 0, "Bind measured cadence exposure")
    a = native_arrays(c["photometry"], c["source_id"])
    signals = [c["recovered"], *c["other_signals"]]
    protection = c.get("protection_mask")
    extra = load_protection(c, a)
    d = detrend(a, signals, exposure, **c["detrending"], additional_protection=extra)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "detrended.npz", **d)
    write(
        out / "preparation.json",
        dict(
            config=receipt(config_path),
            photons=receipt(c["photometry"]),
            output=receipt(out / "detrended.npz"),
            source_id=c["source_id"],
            signal_id=c["signal_id"],
            flux_convention=c["flux_convention"],
            supplied_valid_points=int(a["valid"].sum()),
            retained_points=len(d["flux"]),
            additional_protection=receipt(protection["path"]) if protection else None,
            model=(
                "Transit-protected Lightkurve Savitzky-Golay; conditional errors "
                "divided by trend; no added clipping of delivered photons or "
                "covariance/particle-bank likelihood."
            ),
            implementation="KeplerX validator.preparation (Lightkurve 2.6.0 flatten without astropy objects)",
        ),
    )


def fold(arrays, clock, other_signals, exposure):
    p, t, d = (clock[k] for k in ("period_days", "epoch_bkjd", "duration_days"))
    x = phase(arrays["time_bkjd"], p, t)
    selected = abs(x) <= 2 * d
    selected &= ~transit_mask(arrays["time_bkjd"], other_signals, exposure)
    require(selected.any(), "No photons in adopted probability window")
    x, y, e = x[selected], arrays["flux"][selected], arrays["flux_err"][selected]
    if len(x) <= 200:
        return dict(
            time=x,
            flux=y,
            flux_err=e,
            original_index=arrays["original_index"][selected],
        )
    bins = np.minimum(((x + 2 * d) / (4 * d) * 200).astype(int), 199)
    rows = []
    for b in np.unique(bins):
        ix = bins == b
        weight = 1 / e[ix] ** 2
        rows.append(
            [
                np.average(x[ix], weights=weight),
                np.average(y[ix], weights=weight),
                1 / np.sqrt(weight.sum()),
            ]
        )
    values = np.asarray(rows).T
    return dict(
        time=values[0],
        flux=values[1],
        flux_err=values[2],
        original_index=arrays["original_index"][selected],
    )


def arguments(parser):
    parser.add_argument("candidate", type=Path)
    parser.add_argument("out", type=Path)


def run(args, config):
    """run.sh stage: the prepared arrays and their record in OUT/prepared."""
    prepare(args.candidate, args.out / "prepared")
    record = json.loads((args.out / "prepared" / "preparation.json").read_text())
    return {k: record[k] for k in ("source_id", "signal_id", "retained_points")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    prepare(args.candidate, args.output)


if __name__ == "__main__":
    main()
