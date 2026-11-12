"""Integrate shared TRICERATOPS draws over quarter responses; preserve repeats."""

import argparse
import contextlib
import datetime
import fcntl
import os
import pickle
import traceback
from pathlib import Path

import numpy as np

from post_boi.protocol import (
    DRAWS,
    REPEATS,
    declared_clocks,
    freeze,
    statistics,
    verify,
)
from post_boi.records import digest, read, receipt, require, write


def calculate(c, clock, packet, draws=DRAWS, joint=None):
    """The no-joint route is the stock reference used by the frozen gold control."""
    from post_boi.scene import KeplerScene

    corrected = c["flux_convention"] == "source_corrected"
    if c["flux_convention"] == "psf_flat":
        from post_boi.psf_scene import prepared_scene

        require(
            joint is None, "Load PSF response only through its authenticated inputs"
        )
        target, joint = prepared_scene(c, clock, packet)
    else:
        target = KeplerScene(
            c["scene"],
            c["geometry"],
            c["background"],
            c["source_id"],
            c["aperture_response"],
        )
        target.prepare_depths(clock["observed_depth_fraction"], corrected, joint)
    options = c["probability"]
    extra = {} if joint is None else {"joint": joint}
    target.calc_probs(
        time=packet["time"],
        flux_0=packet["flux"] if joint is None else packet["binned_flux"],
        flux_err_0=clock["probability_noise_sigma"],
        P_orb=clock["period_days"],
        N=draws,
        parallel=True,
        verbose=1,
        flatpriors=False,
        exptime=c["exposure_days"],
        nsamples=options["supersample"],
        contrast_curve_file=options["contrast_curve"],
        filt=options["contrast_filter"],
        molusc_file=options["molusc"],
        dilution_corrected=corrected,
        **extra,
    )
    require(
        not target.FPP_degenerate, "Degenerate upstream evidence; no planet admission"
    )
    require(np.isfinite([target.FPP, target.NFPP]).all(), "Nonfinite probability")
    expected = 15 + 3 * (int((target.stars["tdepth"] > 0).sum()) - 1)
    require(len(target.probs) == expected, "Incomplete upstream scenario census")
    return target


def calculate_quarters(c, clock, packet, draws=DRAWS):
    from post_boi.quarters import QuarterLikelihood

    if c["flux_convention"] == "psf_flat":
        return calculate(c, clock, packet, draws)
    with np.load(c["geometry"], allow_pickle=False) as geometry:
        quarters = geometry["quarter"]
        joint = QuarterLikelihood(
            packet,
            geometry["source_id"],
            quarters,
            np.stack([geometry[f"flux_fraction_q{q}"] for q in quarters]),
        )
    return calculate(c, clock, packet, draws, joint)


def attempt(plan_path, stage, index):
    from numba import set_num_threads

    plan, frozen = verify(plan_path)
    c, root = plan["candidate"], Path(plan_path).resolve().parent
    set_num_threads(plan["cpu_budget"])
    seed = plan["seeds"][index]
    directory = root / "runs" / stage / f"seed-{index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    result = directory / "result.json"
    if result.exists():
        saved = read(result)
        require(
            saved["freeze_sha256"] == frozen
            and saved["seed"] == seed
            and saved["stage"] == stage
            and saved["index"] == index
            and saved["draws_per_scenario"] == DRAWS,
            "Saved repeat identity differs",
        )
        for item in saved["files"]:
            require(
                digest(item["path"]) == item["sha256"], "Saved result evidence changed"
            )
        verify_evidence(saved)
        return saved
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    trial = directory / f"attempt-{stamp}"
    trial.mkdir()
    write(
        trial / "identity.json",
        dict(
            seed=seed,
            stage=stage,
            index=index,
            freeze_sha256=frozen,
            draws_per_scenario=DRAWS,
        ),
    )
    with (
        (trial / "worker.log").open("x") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            np.random.seed(seed)
            clock = declared_clocks(c, plan["clock"])[stage]
            with np.load(root / f"{stage}.npz") as packet:
                target = calculate_quarters(c, clock, packet)
            target.probs.to_csv(trial / "scenarios.csv", index=False)
            target.stars.to_csv(trial / "stars.csv", index=False)
            np.savez_compressed(trial / "evidence.npz", log_evidence=target.lnZ)
            with (trial / "upstream-output.pkl").open("xb") as stream:
                pickle.dump(target, stream)
            files = [
                receipt(trial / name)
                for name in (
                    "scenarios.csv",
                    "stars.csv",
                    "evidence.npz",
                    "upstream-output.pkl",
                )
            ]
            saved = dict(
                status="COMPLETE_QUARTER_TRICERATOPS_REPEAT",
                seed=seed,
                stage=stage,
                index=index,
                freeze_sha256=frozen,
                FPP=float(target.FPP),
                NFPP=float(target.NFPP),
                files=files,
                draws_per_scenario=DRAWS,
            )
            write(result, saved)
            return saved
        except Exception:
            write(
                trial / "failure.json",
                dict(traceback=traceback.format_exc(), seed=seed, freeze_sha256=frozen),
            )
            raise


def unfavorable(results):
    require(
        len(results) == 2, "Allocation screen requires exactly two complete repeats"
    )
    return all(r["FPP"] >= 0.10 for r in results) or all(
        r["NFPP"] >= 0.01 for r in results
    )


def resume(plan_path):
    plan, _ = verify(plan_path)
    root = Path(plan_path).resolve().parent
    with (root / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first = [attempt(plan_path, "final", i) for i in range(2)]
        if unfavorable(first):
            finish(root, "STOPPED_EARLY_FUTILITY_REMAINS_PC", first, [])
            return
        results = first + [attempt(plan_path, "final", i) for i in range(2, REPEATS)]
        diagnostics = [
            attempt(plan_path, "recovered", i)
            for i in range(plan["candidate"]["recovery_diagnostics"])
        ]
        finish(
            root,
            "COMPLETE_NUMERICAL_PRIMARY_SET_PHYSICAL_REVIEW_REQUIRED",
            results,
            diagnostics,
        )


def finish(root, status, results, diagnostics):
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    primary = statistics(results)
    passed = len(results) == REPEATS and all(r["mean_passes"] for r in primary.values())
    conflict = bool(diagnostics) and not all(
        r["mean_passes"] for r in statistics(diagnostics).values()
    )
    if len(results) == REPEATS:
        if not passed:
            status = "COMPLETE_NUMERICAL_THRESHOLDS_NOT_MET_REMAINS_PC"
        elif conflict:
            status = "TIMING_DIAGNOSTIC_CONFLICT_REVIEW_REQUIRED"
    report = dict(
        status=status,
        numerical_primary_pass=passed,
        recovery_requires_review=conflict,
        primary=primary,
        diagnostics=diagnostics,
        scope=(
            "Shared upstream draws and physical curves with a joint response "
            "Gaussian likelihood. No exported full per-draw weights/ESS/omission "
            "audit. Numerical pass is separate from physical validation and RV "
            "confirmation."
        ),
    )
    write(root / f"audit-{stamp}.json", report)
    print(report["status"], flush=True)


def verify_evidence(saved):
    import pandas as pd
    from triceratops._numerics import _normalize_probabilities

    paths = {Path(r["path"]).name: r["path"] for r in saved["files"]}
    table = pd.read_csv(paths["scenarios.csv"])
    with np.load(paths["evidence.npz"]) as archive:
        probability, status = _normalize_probabilities(archive["log_evidence"])
    require(
        status == "ok" and np.allclose(probability, table.prob, rtol=1e-12, atol=1e-15),
        "Saved scenario normalization differs",
    )
    fpp = 1 - probability[[0, 3, 9]].sum()
    nfpp = probability[15:].sum()
    require(
        np.allclose([saved["FPP"], saved["NFPP"]], [fpp, nfpp], rtol=1e-12, atol=1e-15),
        "Saved probabilities differ from upstream evidence",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["freeze", "verify", "resume"])
    parser.add_argument("path")
    parser.add_argument("output", nargs="?")
    args = parser.parse_args()
    if args.operation == "freeze":
        require(args.output is not None, "Supply a new output directory")
        print(freeze(args.path, args.output))
    elif args.operation == "verify":
        verify(args.path)
        print("VERIFIED")
    else:
        resume(args.path)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
