"""Native transit fit: post_boi.native's inputs, sampler and outputs, faster."""

import argparse
import os
import pickle
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from validator import kernel_bindings
from validator.native_likelihood import NativeLikelihood, PriorTransform
from validator.records import candidate, digest, read, receipt, require, write
from validator.upstream import dynesty
from validator.upstream.dynesty import pool as dynesty_pool
from validator.upstream.dynesty.utils import resample_equal

ROOT = Path(__file__).resolve().parent
# juliet.fit's own named arguments; the rest of fit_options reach dynesty.
JULIET_ARGUMENTS = {"sampler", "n_live_points", "nthreads", "dynesty_save_states"}


def phase(time, period, epoch):
    return (time - epoch + period / 2) % period - period / 2


def transit_mask(time, signals, exposure):
    mask = np.zeros(len(time), dtype=bool)
    for signal in signals:
        p, t, d = (signal[k] for k in ("period_days", "epoch_bkjd", "duration_days"))
        require(np.isfinite([p, t, d]).all() and 0 < d < p, "Invalid supplied clock")
        mask |= abs(phase(time, p, t)) <= d + exposure / 2
    return mask


def native_photons(c, prepared):
    with np.load(Path(prepared) / "detrended.npz") as archive:
        a = dict(archive)
    clock = c["recovered"]
    use = (
        abs(phase(a["time_bkjd"], clock["period_days"], clock["epoch_bkjd"]))
        <= 3 * clock["duration_days"]
    )
    use &= ~transit_mask(a["time_bkjd"], c["other_signals"], c["exposure_days"])
    require(use.any(), "No native fitting photons")
    return a["time_bkjd"][use], a["flux"][use], a["flux_err"][use]


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


def summarize(samples, priors, out, config):
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


def sources():
    paths = [p for p in ROOT.rglob("*") if p.is_file() and p.suffix in (".py", ".c")]
    return [receipt(p) for p in sorted(paths) if "__pycache__" not in p.parts]


def environment():
    import scipy

    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        dynesty=dynesty.__version__,
        machine=platform.machine(),
    )


def sampler_arguments(fit_options, rstate):
    """juliet's split of fit_options between DynamicNestedSampler and run_nested."""
    kwargs = {k: v for k, v in fit_options.items() if k not in JULIET_ARGUMENTS}
    kwargs["rstate"] = rstate
    init = dict(bound="multi", sample="rwalk", nlive=fit_options["n_live_points"])
    names = dynesty.DynamicNestedSampler.__init__.__code__.co_varnames
    init.update({k: v for k, v in kwargs.items() if k in names})
    run_names = dynesty.dynamicsampler.DynamicSampler.run_nested.__code__.co_varnames
    run = {k: v for k, v in kwargs.items() if k in run_names}
    return init, run


def check_candidate(c, n, priors):
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
    options = n["fit_options"]
    require(
        options.get("sampler") == "dynamic_dynesty",
        "Validator reproduces juliet's dynamic_dynesty fits only",
    )
    require(options.get("nthreads"), "Declare the proposal queue (nthreads)")


def _rows(chunk):
    return dynesty_pool.FunctionCache.loglike.rows(chunk)[0]


def run_sampler(like, prior, n, out, workers, rstate):
    """Sample, resample to equal weights and recompute each sample's likelihood."""
    options = n["fit_options"]
    init, run = sampler_arguments(options, rstate)
    if options.get("dynesty_save_states"):
        run["checkpoint_file"] = str(out / "sampler" / "checkpoint.pkl")
    queue = int(options["nthreads"])
    timings = {}
    with dynesty_pool.Pool(workers or queue, like, prior) as pool:
        start = time.perf_counter()
        sampler = dynesty.DynamicNestedSampler(
            pool.loglike, pool.prior_transform, len(prior.names), pool=pool,
            queue_size=queue, **init,
        )  # fmt: skip
        sampler.run_nested(**run)
        results = sampler.results
        timings["sampling_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        weights = np.exp(results["logwt"] - results["logz"][-1])
        posterior = resample_equal(results.samples, weights, rstate=rstate)
        chunks = np.array_split(posterior, max(1, min(len(posterior), 4 * pool.njobs)))
        loglike = np.concatenate(pool.pool.map(_rows, chunks))
        timings["posterior_seconds"] = time.perf_counter() - start
    return results, posterior, loglike, timings


def fit(config_path, prepared, output, workers=None):
    wall, cpu = time.perf_counter(), time.process_time()
    kernel = dict(
        source_sha256=kernel_bindings.source_digest(),
        flags=list(kernel_bindings.FLAGS + kernel_bindings.LINK),
    )
    c, out = candidate(config_path), Path(output)
    n = c["native"]
    priors = read(n["priors"])
    check_candidate(c, n, priors)
    preparation = read(Path(prepared) / "preparation.json")
    require(
        preparation["config"]["sha256"] == digest(config_path),
        "Candidate changed after preparation",
    )
    require(
        preparation["output"]["sha256"] == digest(Path(prepared) / "detrended.npz"),
        "Prepared photons changed",
    )
    time_bkjd, flux, flux_err = native_photons(c, prepared)
    out.mkdir(parents=True, exist_ok=False)
    (out / "sampler").mkdir()
    write(
        out / "inputs.json",
        dict(
            candidate=receipt(config_path),
            priors=receipt(n["priors"]),
            photons=receipt(Path(prepared) / "detrended.npz"),
            native_points=int(time_bkjd.size),
            environment=environment(),
            kernel=kernel,
            sources=sources(),
            scope="post_boi.native's photons, priors, seed and dynesty settings; KeplerX Validator likelihood and dynesty's caching pool.",
        ),
    )
    np.random.seed(n["seed"])
    rstate = np.random.default_rng(n["seed"])
    like = NativeLikelihood(
        priors, time_bkjd, flux, flux_err, n["supersample"], c["exposure_days"]
    )
    prior = PriorTransform(priors)
    try:
        setup = time.perf_counter() - wall
        results, posterior, loglike, timings = run_sampler(
            like, prior, n, out, workers, rstate
        )
        samples = {"unnamed": posterior, "loglike": loglike}
        for i, name in enumerate(prior.names):
            samples[name] = posterior[:, i]
        summarize(samples, priors, out, c)
        with (out / "sampler" / "results.pkl").open("xb") as stream:
            pickle.dump(results, stream)
        usage = [
            resource.getrusage(r)
            for r in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
        ]
        write(
            out / "sampler.json",
            dict(
                lnZ=float(results.logz[-1]),
                lnZerr=float(results.logzerr[-1]),
                calls=int(np.sum(results.ncall)),
                iterations=int(results.niter),
                posterior_samples=int(len(posterior)),
                native_points=int(time_bkjd.size),
                workers=int(workers or n["fit_options"]["nthreads"]),
                queue_size=int(n["fit_options"]["nthreads"]),
                setup_seconds=setup,
                **timings,
                wall_seconds=time.perf_counter() - wall,
                main_cpu_seconds=time.process_time() - cpu,
                worker_cpu_seconds=usage[1].ru_utime + usage[1].ru_stime,
                main_peak_rss_bytes=usage[0].ru_maxrss,
                worker_peak_rss_bytes=usage[1].ru_maxrss,
                kernel=kernel,
            ),
        )
    except Exception:
        write(out / "failure.json", dict(traceback=traceback.format_exc()))
        raise


def arguments(parser):
    parser.add_argument("candidate", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=None)


def run(args, config):
    """run.sh stage: the fit's records in OUT/native."""
    fit(args.candidate, args.prepared, args.out / "native", args.workers)
    record = read(args.out / "native" / "sampler.json")
    return {k: record[k] for k in ("lnZ", "lnZerr", "calls", "posterior_samples")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate")
    parser.add_argument("prepared")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    fit(args.candidate, args.prepared, args.output, args.workers)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT.parent))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
