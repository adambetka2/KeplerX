"""Measure detrending transfer on observed photons, without a tail-coverage veto."""

import argparse
from pathlib import Path

import batman
import numpy as np

from post_boi.prepare import detrend, load_protection, native_arrays, phase
from post_boi.records import candidate, read, receipt, require, write


def joint_model(samples, priors, index, time, exposure, supersample):
    def value(name):
        return (
            float(samples[name][index])
            if name in samples
            else priors[name]["hyperparameters"]
        )

    p = batman.TransitParams()
    p.t0, p.per, p.rp = (value(k) for k in ("t0_p1", "P_p1", "p_p1"))
    p.a = (6.67408e-11 * value("rho") * (p.per * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)
    p.ecc, p.w = value("ecc_p1"), value("omega_p1")
    q = (1 - p.ecc**2) / (1 + p.ecc * np.sin(np.deg2rad(p.w)))
    p.inc = np.rad2deg(np.arccos(value("b_p1") / (p.a * q)))
    q1, q2 = value("q1_Kepler"), value("q2_Kepler")
    p.u, p.limb_dark = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)], "quadratic"
    require(np.isfinite(p.inc), "Invalid joint orbital geometry")
    return batman.TransitModel(
        p, time, supersample_factor=supersample, exp_time=exposure
    ).light_curve(p)


def assess(config, posterior, indices, output):
    c, out = candidate(config), Path(output)
    raw = native_arrays(c["photometry"], c["source_id"])
    priors = read(c["native"]["priors"])
    with np.load(posterior, allow_pickle=False) as archive:
        samples = dict(archive)
    signals = [c["recovered"], *c["other_signals"]]
    options = dict(
        signals=signals,
        exposure=c["exposure_days"],
        additional_protection=load_protection(c, raw),
        **c["detrending"],
    )
    original = detrend(raw, **options)
    valid = raw["valid"]
    clock = c["recovered"]
    fitted = (
        abs(phase(raw["time_bkjd"][valid], clock["period_days"], clock["epoch_bkjd"]))
        <= 3 * clock["duration_days"]
    )
    extra = np.zeros(len(valid), dtype=bool)
    rows = []
    for index in indices:
        model = joint_model(
            samples, priors, index, raw["time_bkjd"][valid], c["exposure_days"], 40
        )
        require(np.all(model > 0), "Cannot remove an exactly total eclipse by division")
        extra[valid] |= model < 1
        without = {k: v.copy() for k, v in raw.items()}
        without["flux"][valid] /= model
        without["flux_err"][valid] /= model
        flattened = detrend(without, **options)
        # Transit removal and a full detrending refit measure actual signal transfer.
        change = original["flux"] - flattened["flux"] * model
        deficit = 1 - model
        error = original["flux_err"]
        norm = np.sum((deficit / error) ** 2)
        projection = float(np.sum(deficit * change / error**2) / norm)
        unprotected = (~original["protected"]) & (deficit > 0)
        rows.append(
            dict(
                joint_sample_index=int(index),
                unprotected_observed_cadences=int(unprotected.sum()),
                maximum_unprotected_model_loss_ppm=float(
                    deficit[unprotected].max(initial=0) * 1e6
                ),
                maximum_all_observed_flux_change_ppm=float(abs(change).max() * 1e6),
                maximum_native_window_flux_change_ppm=float(
                    abs(change[fitted]).max() * 1e6
                ),
                native_window_change_formal_norm=float(
                    np.linalg.norm(change[fitted] / error[fitted])
                ),
                projected_transit_amplitude_change_fraction=projection,
                projected_change_in_formal_sigma=float(projection * np.sqrt(norm)),
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "protection.npz",
        source_id=c["source_id"],
        time_bkjd=raw["time_bkjd"],
        protected=extra,
    )
    write(
        out / "review.json",
        dict(
            source_id=c["source_id"],
            signal_id=c["signal_id"],
            inputs=[
                receipt(config),
                receipt(posterior),
                receipt(c["photometry"]),
                receipt(c["native"]["priors"]),
            ],
            trials=rows,
            protection=receipt(out / "protection.npz"),
            scope="Paired removal/refit of specified valid joint transit models on original observed photons. No hypothetical gap events or automatic all-tail coverage gate. Model-dependent diagnostic, not FPP or a physical admission. The new mask requires a separately reviewed preparation/fit if adopted.",
        ),
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("posterior", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--indices", required=True, type=int, nargs="+")
    args = parser.parse_args()
    print(assess(args.config, args.posterior, args.indices, args.output))
