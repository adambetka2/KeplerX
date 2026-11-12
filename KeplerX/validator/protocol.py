"""Freeze twenty TRICERATOPS calculations with joint quarter responses."""

import hashlib
import importlib.metadata
import json
import os
import platform
import sysconfig
from pathlib import Path

import numpy as np

from validator.quarters import fold_quarters
from validator.records import candidate as load_candidate
from validator.records import digest, read, receipt, require, write

PROTOCOL_ID = "TRICERATOPS_KEPLER_V3"
# A likelihood revision does not select a new Monte Carlo seed sequence.
SEED_NAMESPACE = "TRICERATOPS_KEPLER_V2"
UPSTREAM_COMMIT = "d20875c27d6761349cf032728bb4a59ec732a84d"
REPEATS, DRAWS = 20, 1_000_000
THRESHOLDS = {"FPP": 0.015, "NFPP": 0.001}


def environment():
    packages = sorted(
        [d.metadata["Name"].lower(), d.version]
        for d in importlib.metadata.distributions(path=[sysconfig.get_path("purelib")])
    )
    return dict(python=platform.python_version(), packages=packages)


def scientific_files():
    import batman
    import dynesty
    import emcee
    import juliet
    import triceratops

    paths = list(Path(__file__).parent.glob("*.py"))
    paths.extend(Path(__file__).with_name("patches").glob("*"))
    for module in (batman, dynesty, emcee, juliet, triceratops):
        root = Path(module.__file__).parent
        paths.extend(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix in (".py", ".csv", ".so", ".pyd", ".dylib")
        )
    return sorted({p.resolve() for p in paths})


def clock_review(c):
    review = read(c["native_review"])
    require(
        review["accepted"] is True and bool(review["evidence"]),
        "Native fit still needs scientific review",
    )
    require(
        (review["source_id"], review["signal_id"]) == (c["source_id"], c["signal_id"]),
        "Reviewed source/signal differs",
    )
    p, t, d, depth = (
        review[k]
        for k in (
            "period_days",
            "epoch_bkjd",
            "duration_days",
            "observed_depth_fraction",
        )
    )
    require(
        all(v is not None for v in [p, t, d, depth])
        and np.isfinite([p, t, d, depth]).all()
        and 0 < d < p
        and 0 < depth < 1,
        "Incomplete adopted clock/depth",
    )
    require(
        review["probability_noise_sigma"] is not None
        and np.isfinite(review["probability_noise_sigma"])
        and review["probability_noise_sigma"] > 0
        and bool(review["probability_noise_basis"]),
        "Review scalar folded noise; formal errors are not automatically total noise",
    )
    return review


def declarations(c):
    require(c["protocol"] == PROTOCOL_ID, "Wrong generation; no legacy adapter/imports")
    require(
        c["flux_convention"] in ("raw_aperture", "source_corrected", "psf_flat"),
        "Declare flux convention",
    )
    require(
        type(c["probability"]["supersample"]) is int
        and c["probability"]["supersample"] > 0,
        "Decide exposure integration per candidate",
    )
    if c["flux_convention"] == "psf_flat":
        require(
            c["aperture_response"] == "signed_psf_flat" and bool(c.get("psf_response")),
            "Supply a separately reviewed signed PSF response",
        )
    else:
        require(
            c["aperture_response"] in ("measured_kepler", "calibrated_kepler")
            and c.get("psf_response") is None,
            "Supply authenticated quarter response fractions",
        )
    require(
        bool(c["probability"]["noise"]) and bool(c["probability"]["noise_basis"]),
        "Supply reviewed native noise and applied dilution; no folded substitution",
    )
    require(c["recovery_diagnostics"] in (0, 3), "Recovery budget is zero or three")
    require(
        type(c["cpu_budget"]) is int and 0 < c["cpu_budget"] <= os.cpu_count(),
        "Invalid CPU budget",
    )
    require(
        c["probability"]["contrast_curve"] is None
        or bool(c["probability"]["contrast_filter"]),
        "Bind measured contrast-curve filter",
    )
    require(bool(c["timing_review"]), "Declare timing review; do not choose by FPP")
    readiness = read(c["readiness"])
    require(
        readiness["accepted"] is True and bool(readiness["evidence"]),
        "Readiness unresolved",
    )
    require(
        (readiness["source_id"], readiness["signal_id"])
        == (c["source_id"], c["signal_id"]),
        "Readiness identity differs",
    )


def freeze(candidate, output):
    import shutil

    from validator.dependencies import quarter_patch_state

    c, out = load_candidate(candidate), Path(output).resolve()
    direct = json.loads(
        importlib.metadata.distribution("triceratops").read_text("direct_url.json")
    )
    require(
        direct.get("vcs_info", {}).get("commit_id") == UPSTREAM_COMMIT,
        "Install the pinned official TRICERATOPS commit",
    )
    quarter_patch_state(patched=True)
    declarations(c)
    clock = clock_review(c)
    psf_files = []
    if c["flux_convention"] == "psf_flat":
        from validator.psf_response import reviewed_files

        psf_files = reviewed_files(c, clock)
    prepared = Path(c["prepared"])
    prep = read(prepared / "preparation.json")
    require(
        prep["config"]["sha256"] == digest(candidate),
        "Candidate changed after preparation",
    )
    require(
        prep["output"]["sha256"] == digest(prepared / "detrended.npz"),
        "Prepared photons changed",
    )
    require(
        prep["photons"]["sha256"] == digest(c["photometry"]),
        "Native photons changed after preparation",
    )
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source/validator"
    source.mkdir(parents=True)
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(p, source / p.name)
    shutil.copytree(Path(__file__).with_name("patches"), source / "patches")
    shutil.copytree(
        Path(__file__).with_name("probability"),
        source / "probability",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    files = [Path(candidate), prepared / "preparation.json", prepared / "detrended.npz"]
    files += [
        Path(c[k])
        for k in (
            "photometry",
            "scene",
            "geometry",
            "background",
            "native_review",
            "readiness",
        )
    ]
    files.append(Path(c["native"]["priors"]))
    files.extend(psf_files)
    if c.get("protection_mask") is not None:
        files.append(Path(c["protection_mask"]["path"]))
    for review in (clock, read(c["readiness"])):
        for r in review["evidence"]:
            require(digest(r["path"]) == r["sha256"], "Review evidence changed")
            files.append(Path(r["path"]))
    files.extend(
        Path(c["probability"][k])
        for k in ("contrast_curve", "molusc")
        if c["probability"][k] is not None
    )
    noise_path = Path(c["probability"]["noise"])
    files.append(noise_path)
    with (
        np.load(prepared / "detrended.npz", allow_pickle=False) as a,
        np.load(noise_path, allow_pickle=False) as noise,
    ):
        require(
            str(noise["source_id"].item()) == c["source_id"], "Noise source differs"
        )
        for stage, current in declared_clocks(c, clock).items():
            packet = out / f"{stage}.npz"
            folded = fold_quarters(
                a,
                current,
                c["other_signals"],
                c["exposure_days"],
                noise,
                c["flux_convention"],
            )
            if c["flux_convention"] == "psf_flat":
                from validator.psf_scene import prepared_scene

                prepared_scene(c, current, folded)
            np.savez_compressed(packet, **folded)
            files.append(packet)
    seeds = [
        int.from_bytes(
            hashlib.sha256(
                f"{SEED_NAMESPACE}|{c['source_id']}|{c['signal_id']}|{i}".encode()
            ).digest()[:4],
            "big",
        )
        for i in range(REPEATS)
    ]
    require(len(set(seeds)) == REPEATS, "Seed collision")
    plan = dict(
        protocol=PROTOCOL_ID,
        candidate=c,
        clock=clock,
        seeds=seeds,
        seed_namespace=SEED_NAMESPACE,
        draws_per_scenario=DRAWS,
        cpu_budget=c["cpu_budget"],
        upstream_commit=UPSTREAM_COMMIT,
    )
    write(out / "plan.json", plan)
    files += [out / "plan.json", *source.glob("*.py"), *source.glob("patches/*")]
    files += [
        p for p in scientific_files() if not p.is_relative_to(Path(__file__).parent)
    ]
    write(
        out / "freeze.json",
        dict(
            files=[receipt(p) for p in sorted(set(files))],
            environment=environment(),
            scope=(
                "Pinned upstream priors, sampling and physical curves with an "
                "explicit joint response Gaussian likelihood. Immutable "
                "candidate, original photon indices, reviewed noise/dilution, "
                "native/readiness and instrument-response evidence, patched "
                "sources and environment. No inherited results."
            ),
        ),
    )
    return out / "plan.json"


def declared_clocks(c, clock):
    clocks = {"final": clock}
    if c["recovery_diagnostics"]:
        clocks["recovered"] = dict(
            clock, **{k: c["recovered"][k] for k in ("period_days", "epoch_bkjd")}
        )
    return clocks


def verify(plan):
    path = Path(plan).resolve()
    frozen = read(path.parent / "freeze.json")
    require(frozen["environment"] == environment(), "Pinned environment changed")
    for item in frozen["files"]:
        require(
            digest(item["path"]) == item["sha256"],
            "Frozen file changed: " + item["path"],
        )
    bound = {r["path"] for r in frozen["files"]}
    require(
        all(str(p.resolve()) in bound for p in Path(__file__).parent.glob("*.py")),
        "Execute from the frozen source copy",
    )
    return read(path), digest(path.parent / "freeze.json")


def statistics(results):
    report = {}
    for name, threshold in THRESHOLDS.items():
        a = np.asarray([r[name] for r in results])
        report[name] = dict(
            repeats=a,
            mean=float(a.mean()),
            sample_standard_deviation=float(a.std(ddof=1)),
            minimum=float(a.min()),
            maximum=float(a.max()),
            unfavorable_indices=np.flatnonzero(a >= threshold),
            mean_passes=bool(a.mean() < threshold),
        )
    return report
