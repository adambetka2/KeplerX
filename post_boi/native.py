"""Delegate the native transit fit to juliet with explicit candidate priors."""

import argparse
import traceback
from pathlib import Path

import numpy as np

from post_boi.prepare import phase, transit_mask
from post_boi.records import candidate, digest, read, receipt, require, write


def stellar_density(mass, radius):
    """First-order SI density prior from declared measurements; no error defaults."""
    m, r = mass["value"], radius["value"]
    require(m > 0 and r > 0, "Nonpositive stellar mass/radius")
    errors = [mass["error"], radius["error"]]
    require(
        all(e is not None and e > 0 for e in errors),
        "Supply reported stellar errors or an explicit justified uncertainty model",
    )
    rho = 1409.779824307526 * m / r**3
    sigma = rho * np.hypot(errors[0] / m, 3 * errors[1] / r)
    return dict(distribution="normal", hyperparameters=[rho, sigma])


def duration_samples(samples, priors):
    def value(name):
        return samples[name] if name in samples else priors[name]["hyperparameters"]

    p, k, b, rho = (value(n) for n in ("P_p1", "p_p1", "b_p1", "rho"))
    axis = (6.67408e-11 * rho * (p * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)
    ecc, omega = value("ecc_p1"), np.deg2rad(value("omega_p1"))
    q = (1 - ecc**2) / (1 + ecc * np.sin(omega))
    inclination_cosine = b / (axis * q)
    ratio = np.sqrt((1 + k) ** 2 - b**2) / (axis * np.sqrt(1 - inclination_cosine**2))
    return (
        p / np.pi * np.arcsin(ratio) * np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(omega))
    )


def fit(config_path, prepared, output):
    import juliet

    from post_boi.protocol import environment, scientific_files

    c, out = candidate(config_path), Path(output)
    n = c["native"]
    priors = read(n["priors"])
    for name in ("P_p1", "t0_p1", "p_p1", "b_p1", "rho"):
        require(
            priors[name]["distribution"] != "fixed",
            "Native clock/geometry must be fitted",
        )
    require(
        bool(n["prior_basis"]),
        "Justify candidate priors, including measured stellar uncertainties",
    )
    require(n["supersample"] is not None, "Choose and check exposure supersampling")
    options = dict(n["fit_options"])
    if options.get("sampler") == "dynamic_dynesty":
        options["rstate"] = np.random.default_rng(n["seed"])
    require(
        bool(options.get("sampler")), "Declare upstream sampler and convergence budget"
    )
    require(
        options["sampler"] not in ("emcee", "zeus") or n["starting_point"] is not None,
        "Supply juliet starting_point for MCMC",
    )
    with np.load(Path(prepared) / "detrended.npz") as archive:
        a = dict(archive)
    preparation = read(Path(prepared) / "preparation.json")
    require(
        preparation["config"]["sha256"] == digest(config_path),
        "Candidate changed after preparation",
    )
    require(
        preparation["output"]["sha256"] == digest(Path(prepared) / "detrended.npz"),
        "Prepared photons changed",
    )
    clock = c["recovered"]
    use = (
        abs(phase(a["time_bkjd"], clock["period_days"], clock["epoch_bkjd"]))
        <= 3 * clock["duration_days"]
    )
    use &= ~transit_mask(a["time_bkjd"], c["other_signals"], c["exposure_days"])
    require(use.any(), "No native fitting photons")
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "inputs.json",
        dict(
            candidate=receipt(config_path),
            priors=receipt(n["priors"]),
            photons=receipt(Path(prepared) / "detrended.npz"),
            native_points=int(use.sum()),
            environment=environment(),
            sources=[receipt(p) for p in scientific_files()],
            scope="Duration-scaled +/-3T14, no duration/radius/density ceiling in wrapper; parameters and sampler supplied to juliet unchanged.",
        ),
    )
    np.random.seed(n["seed"])
    try:
        data = juliet.load(
            priors=priors,
            starting_point=n["starting_point"],
            t_lc={"Kepler": a["time_bkjd"][use]},
            y_lc={"Kepler": a["flux"][use]},
            yerr_lc={"Kepler": a["flux_err"][use]},
            out_folder=str(out / "juliet"),
            lc_instrument_supersamp=["Kepler"],
            lc_n_supersamp=[n["supersample"]],
            lc_exptime_supersamp=[c["exposure_days"]],
        )
        result = data.fit(**options)
        summarize(result, priors, out, c)
    except Exception:
        write(out / "failure.json", dict(traceback=traceback.format_exc()))
        raise


def summarize(result, priors, out, config):
    samples = result.posteriors["posterior_samples"]
    np.savez_compressed(out / "posterior.npz", **samples)
    summary = {
        name: np.quantile(v, [0.16, 0.5, 0.84])
        for name, v in samples.items()
        if name != "unnamed"
    }
    d = duration_samples(samples, priors)
    require(
        np.isfinite(d).all() and np.all(d > 0),
        "Inspect unsupported fitted transit geometry",
    )
    summary["duration_days"] = np.quantile(d, [0.16, 0.5, 0.84])
    k = summary["p_p1"]
    snr = k[1] / ((k[2] - k[0]) / 2)
    write(
        out / "summary.json",
        dict(
            quantiles_16_50_84=summary,
            radius_ratio_snr_quantile=snr,
            convergence="Review upstream sampler output, mixing/effective samples, residuals, timing, boundaries and exposure accuracy before adopting clock. A summary is not convergence certification.",
        ),
    )
    write(
        out / "review.json",
        dict(
            source_id=config["source_id"],
            signal_id=config["signal_id"],
            accepted=None,
            evidence=None,
            period_days=float(summary["P_p1"][1]),
            epoch_bkjd=float(summary["t0_p1"][1]),
            duration_days=float(summary["duration_days"][1]),
            observed_depth_fraction=None,
            probability_noise_sigma=None,
            probability_noise_basis=None,
            note="Fill after inspecting native fit. Duration uses the usual eccentric-duration approximation; inspect long/grazing geometry. Depth is the observed-convention fitted depth, not intrinsic depth after a second dilution correction.",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate")
    parser.add_argument("prepared")
    parser.add_argument("output")
    args = parser.parse_args()
    fit(args.candidate, args.prepared, args.output)
