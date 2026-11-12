"""Paired delivered-flux transfer and restricted blind-search experiment."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.update(
    OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1"
)

import batman
import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.deterministic.signal_contracts import relative_curve

SPEC = importlib.util.spec_from_file_location(
    "experiment_shallow", ROOT / "pipeline/shallow/run.py"
)
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)
METHODS = ("biweight_0.5d", "biweight_1.0d", "joint_spline_1.5d")
DESIGN = {
    "schema": 1,
    "seed": 20261014,
    "span_days": 45.0,
    "period_min_days": 9.75,
    "period_max_days": 10.25,
    "period_count": 81,
    "phase_count": 256,
    "duration_hours": [2.0, 4.0, 6.0],
    "depth_ppm": [40.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0],
    "calibration_nulls": 128,
    "held_nulls": 128,
    "false_alarm_alpha": 0.05,
    "null_block_days": [1.0, 2.0],
    "exposure_days": 0.02043361111111111,
    "supersample": 11,
    "phases_per_depth_duration": 2,
    "nuisance_knot_days": 1.5,
    "methods": METHODS,
    "search": "positive weighted box score; global maximum over fixed bank",
    "mask": "production relative_curve separately after each raw injection",
    "null": "independent day-block Rademacher signs on spline residual, with trend restored",
    "limitations": [
        "restricted 0.5-day period band",
        "conditional surrogate nulls",
        "imposed transit durations are transfer probes, not a stellar population",
        "not complete production detector",
        "source-level dependence across injections",
    ],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def physical_loss(time, period, epoch, duration_hours, depth_ppm):
    params = batman.TransitParams()
    params.t0, params.per = epoch, period
    params.rp = np.sqrt(depth_ppm * 1e-6)
    params.a = (1 + params.rp) / np.sin(np.pi * duration_hours / 24 / period)
    params.inc, params.ecc, params.w = 90.0, 0.0, 90.0
    params.u, params.limb_dark = [0.5, 0.2], "quadratic"
    model = batman.TransitModel(
        params,
        np.asarray(time),
        supersample_factor=11,
        exp_time=DESIGN["exposure_days"],
    )
    return 1 - model.light_curve(params)


def spline_basis(time, error):
    knots = np.r_[
        np.repeat(time[0], 4),
        np.arange(time[0] + 1.5, time[-1], 1.5),
        np.repeat(time[-1], 4),
    ]
    basis = BSpline.design_matrix(time, knots, 3).toarray() / error[:, None]
    q, r = np.linalg.qr(basis, mode="reduced")
    if np.min(np.abs(np.diag(r))) < 1e-10:
        raise ValueError("Rank-deficient nuisance basis")
    return q


class Search:
    def __init__(self, time, error, parameters=None, origin=None):
        self.time, self.error = time, error
        self.q = spline_basis(time, error)
        periods = np.linspace(9.75, 10.25, 81)
        origin = time[0] if origin is None else origin
        self.parameters = np.array(
            [
                (p, phase * p / 256 + origin, hours / 24)
                for p in periods
                for phase in range(256)
                for hours in (2.0, 4.0, 6.0)
            ]
        )
        if parameters is not None:
            self.parameters = parameters
        self.bank = np.empty((len(self.parameters), len(time)), dtype=np.float32)
        for index, (period, epoch, duration) in enumerate(self.parameters):
            phase = (time - epoch + period / 2) % period - period / 2
            self.bank[index] = (np.abs(phase) < duration / 2) / -error
        norm = np.einsum("ij,ij->i", self.bank, self.bank, dtype=np.float64)
        self.norms = {}
        for name, q in (
            ("box", (1 / error)[:, None] / np.linalg.norm(1 / error)),
            ("joint", self.q),
        ):
            projection = self.bank @ q.astype(np.float32)
            self.norms[name] = np.sqrt(
                np.maximum(norm - np.sum(projection**2, axis=1), 1e-20)
            )

    def series(self, flux, method):
        if method.startswith("biweight"):
            window = 0.5 if method == METHODS[0] else 1.0
            flux = flux - COMMON.detrend(
                self.time,
                flux,
                97,
                {"method": "biweight", "window_days": window, "cval": 5.0},
            )
            weighted = flux / self.error
            constant = 1 / self.error
            return weighted - constant * (constant @ weighted) / (constant @ constant)
        weighted = flux / self.error
        return weighted - self.q @ (self.q.T @ weighted)

    def search(self, flux, method):
        series = self.series(flux, method)
        norm = self.norms["joint" if method == METHODS[2] else "box"]
        scores = (self.bank @ series.astype(np.float32)) / norm
        index = int(np.argmax(scores))
        return {
            "score": float(scores[index]),
            "bank_index": index,
            "period_days": float(self.parameters[index, 0]),
            "epoch_bkjd": float(self.parameters[index, 1]),
            "duration_hours": float(self.parameters[index, 2] * 24),
        }

    def known_template(self, loss, method):
        template = -loss / self.error
        if method == METHODS[2]:
            template = template - self.q @ (self.q.T @ template)
        else:
            constant = 1 / self.error
            template = template - constant * (constant @ template) / (
                constant @ constant
            )
        return template

    def known_depth(self, flux, loss, method):
        template = self.known_template(loss, method)
        return float(template @ self.series(flux, method) / (template @ template))


def prepared(curve):
    indices, arrays = relative_curve(curve)
    time, flux, error = arrays[:3]
    if len(time) < 100 or np.any(np.diff(time) <= 0):
        raise ValueError("Insufficient or nonmonotonic native support")
    return indices, time, flux, error


def normalized_search(search, time, flux, error, method):
    ratio = error / search.error
    if not np.array_equal(time, search.time) or not np.allclose(
        ratio, ratio[0], rtol=1e-12, atol=0
    ):
        raise ValueError(
            "Search reuse requires identical times and scalar error normalization"
        )
    proposal = search.search(flux, method)
    proposal["score"] /= float(ratio[0])
    return proposal


def raw_injection(curve, loss):
    result = dict(curve)
    valid = np.isfinite(curve["flux"]) & (curve["quality"] == 0)
    source_flux = curve.get("source_flux", np.median(curve["flux"][valid]))
    result["flux"] = curve["flux"] - source_flux * loss
    return result


def null_curves(curve, search, indices, flux, rng, count, block_days):
    weighted = flux / search.error
    trend = search.q @ (search.q.T @ weighted) * search.error
    residual = flux - trend
    blocks = np.floor((search.time - search.time[0]) / block_days).astype(int)
    scale = np.median(curve["flux"][indices])
    for _ in range(count):
        signs = rng.choice([-1, 1], size=int(blocks.max()) + 1)
        simulated = trend + residual * signs[blocks]
        null = dict(curve, flux=curve["flux"].copy())
        null["flux"][indices] = (1 + simulated) * scale
        yield null


def calibrated_nulls(curve, search, indices, flux, rng, folder, block_days):
    counts = DESIGN["calibration_nulls"], DESIGN["held_nulls"]
    maxima = {method: [] for method in METHODS}
    masks = []
    realizations = []
    for number, null in enumerate(
        null_curves(curve, search, indices, flux, rng, sum(counts), block_days)
    ):
        selected, time, values, error = prepared(null)
        engine = (
            search
            if np.array_equal(selected, indices)
            else Search(time, error, search.parameters)
        )
        masks.append(selected)
        realizations.append(null["flux"])
        for method in METHODS:
            maxima[method].append(
                normalized_search(engine, time, values, error, method)["score"]
            )
        if (number + 1) % 64 == 0:
            print(folder.name, block_days, "day nulls", number + 1, flush=True)
    np.savez_compressed(
        folder / f"null-masks-{block_days}d.npz",
        **{str(i): x for i, x in enumerate(masks)},
    )
    np.savez_compressed(
        folder / f"null-raw-flux-{block_days}d.npz", flux=np.asarray(realizations)
    )
    result = {}
    for method, values in maxima.items():
        calibration, held = np.array(values[: counts[0]]), np.array(values[counts[0] :])
        # Plus-one conformal rank: a detection requires p <= alpha.
        pvalues = (1 + (calibration[:, None] >= held).sum(axis=0)) / (
            len(calibration) + 1
        )
        failures, total = int((pvalues <= 0.05).sum()), len(held)
        interval = [
            float(beta.ppf(0.025, failures, total - failures + 1)) if failures else 0.0,
            float(beta.ppf(0.975, failures + 1, total - failures))
            if failures < total
            else 1.0,
        ]
        result[method] = {
            "calibration_maxima": calibration.tolist(),
            "held_maxima": held.tolist(),
            "held_pvalues": pvalues.tolist(),
            "held_false_alarms": failures,
            "held_false_alarm_rate": failures / total,
            "conditional_binomial_95_interval": interval,
        }
    write(folder / f"nulls-{block_days}d.json", result)
    return result


def truth_rows(curve, rng):
    start = float(np.nanmin(curve["time"]))
    for hours in DESIGN["duration_hours"]:
        for depth in DESIGN["depth_ppm"]:
            for repeat in range(DESIGN["phases_per_depth_duration"]):
                period = float(rng.uniform(9.8, 10.2))
                yield {
                    "period_days": period,
                    "epoch_bkjd": start + float(rng.uniform(0, period)),
                    "duration_hours": hours,
                    "depth_ppm": depth,
                    "phase_repeat": repeat,
                }


def injected_trial(curve, search, indices, flux, truth, nulls, folder, number):
    loss = physical_loss(
        curve["time"],
        *[
            truth[k]
            for k in ("period_days", "epoch_bkjd", "duration_hours", "depth_ppm")
        ],
    )
    injected = raw_injection(curve, loss)
    selected, time, values, error = prepared(injected)
    engine = (
        search
        if np.array_equal(selected, indices)
        else Search(time, error, search.parameters)
    )
    np.savez_compressed(
        folder / f"injection-{number:02d}.npz",
        loss=loss,
        injected_flux=injected["flux"],
        selected_indices=selected,
        prepared_flux=values,
        time=time,
        error=error,
    )
    original_selected = dict(curve)
    original_selected["quality"] = np.ones(len(curve["time"]), dtype=int)
    original_selected["quality"][selected] = 0
    base_indices, _, base, _ = prepared(original_selected)
    if not np.array_equal(base_indices, selected):
        raise ValueError(
            "Base and injected support differ after matched-mask normalization"
        )
    source_flux = np.broadcast_to(
        curve.get("source_flux", np.median(curve["flux"][selected])), loss.shape
    )
    delivered_loss = (
        loss[selected] * source_flux[selected] / np.median(curve["flux"][selected])
    )
    rows = []
    for method in METHODS:
        proposal = normalized_search(engine, time, values, error, method)
        calibration = np.asarray(nulls[method]["calibration_maxima"])
        pvalue = float(
            (1 + (calibration >= proposal["score"]).sum()) / (len(calibration) + 1)
        )
        retention = engine.known_depth(
            values, delivered_loss, method
        ) - engine.known_depth(base, delivered_loss, method)
        period_match = abs(proposal["period_days"] / truth["period_days"] - 1) < 0.01
        phase = abs(
            (proposal["epoch_bkjd"] - truth["epoch_bkjd"] + truth["period_days"] / 2)
            % truth["period_days"]
            - truth["period_days"] / 2
        )
        proposed_phase = (
            time - proposal["epoch_bkjd"] + proposal["period_days"] / 2
        ) % proposal["period_days"] - proposal["period_days"] / 2
        proposed_window = np.abs(proposed_phase) < proposal["duration_hours"] / 48
        energy_covered = float(
            np.sum(delivered_loss[proposed_window] ** 2) / np.sum(delivered_loss**2)
        )
        rows.append(
            {
                "injection": number,
                "method": method,
                "truth": truth,
                "intrinsic_peak_loss_ppm": float(np.max(loss) * 1e6),
                "delivered_peak_loss_ppm": float(np.max(delivered_loss) * 1e6),
                "proposal": proposal,
                "empirical_p": pvalue,
                "known_clock_retained_amplitude": retention,
                "known_clock_formal_score_delta": retention
                * float(np.linalg.norm(engine.known_template(delivered_loss, method)))
                / float(error[0] / engine.error[0]),
                "injected_signal_energy_covered": energy_covered,
                "recovered": bool(
                    pvalue <= 0.05
                    and period_match
                    and phase < truth["duration_hours"] / 24
                    and energy_covered >= 0.5
                ),
                "mask_changed": not np.array_equal(indices, selected),
            }
        )
    return rows


def run_source(entry, out):
    folder = out / entry["name"]
    folder.mkdir()
    if digest(entry["path"]) != entry["input_sha256"]:
        raise ValueError("Input changed after design freeze")
    with np.load(entry["path"], allow_pickle=False) as data:
        curve = {
            name: data[name].copy()
            for name in ("time", "flux", "error", "quality", "quarter", "cadence")
        }
        if "source_flux" in data:
            curve["source_flux"] = np.broadcast_to(
                data["source_flux"], curve["time"].shape
            ).copy()
    if len(np.unique(curve["quarter"])) != 1:
        raise ValueError("Pilot input must contain one declared quarter")
    use = np.isfinite(curve["time"]) & (curve["time"] <= np.nanmin(curve["time"]) + 45)
    curve = {key: value[use] for key, value in curve.items()}
    np.savez_compressed(folder / "raw.npz", **curve)
    indices, time, flux, error = prepared(curve)
    search = Search(time, error, origin=float(np.nanmin(curve["time"])))
    np.savez_compressed(
        folder / "baseline.npz",
        indices=indices,
        time=time,
        flux=flux,
        error=error,
        bank_parameters=search.parameters,
    )
    identity = f"{entry['source_id']}:{int(curve['quarter'][0])}"
    source_seed = int.from_bytes(
        hashlib.sha256(identity.encode()).digest()[:4], "little"
    )
    rng = np.random.default_rng(np.random.SeedSequence([DESIGN["seed"], source_seed]))
    truths = list(truth_rows(curve, rng))
    write(folder / "truth.json", truths)
    write(
        folder / "uninjected-search.json",
        {method: search.search(flux, method) for method in METHODS},
    )
    nulls = {}
    for block_days in DESIGN["null_block_days"]:
        null_rng = np.random.default_rng(
            np.random.SeedSequence([DESIGN["seed"], source_seed, int(block_days)])
        )
        nulls[str(block_days)] = calibrated_nulls(
            curve, search, indices, flux, null_rng, folder, block_days
        )
    rows = []
    for number, truth in enumerate(truths):
        trial = injected_trial(
            curve, search, indices, flux, truth, nulls["1.0"], folder, number
        )
        for row in trial:
            calibration = np.asarray(nulls["2.0"][row["method"]]["calibration_maxima"])
            row["empirical_p_2day_null"] = float(
                (1 + (calibration >= row["proposal"]["score"]).sum())
                / (len(calibration) + 1)
            )
        rows.extend(trial)
        write(folder / "results.json", rows)
    if digest(entry["path"]) != entry["input_sha256"]:
        raise ValueError("Input changed during source experiment")
    return {
        "source": entry,
        "input_sha256": digest(entry["path"]),
        "cadences": len(time),
        "span_days": float(np.ptp(time)),
        "results": rows,
        "nulls": nulls,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source_sha256 = digest(__file__)
    (args.out / "preparation.py").write_bytes(Path(__file__).read_bytes())
    entries = json.loads(args.manifest.read_text())
    entries = [entry | {"input_sha256": digest(entry["path"])} for entry in entries]
    write(
        args.out / "design.json",
        DESIGN
        | {
            "inputs": entries,
            "source_sha256": source_sha256,
            "manifest_sha256": digest(args.manifest),
            "dependency_sources": {
                name: digest(ROOT / name)
                for name in (
                    "pipeline/shallow/run.py",
                    "pipeline/deterministic/signal_contracts.py",
                )
            },
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "batman-package", "wotan", "numba")
            },
        },
    )
    results = []
    for entry in entries:
        try:
            results.append(run_source(entry, args.out))
        except Exception as error:
            write(
                args.out / "first-failure.json", {"source": entry, "error": repr(error)}
            )
            raise
        write(args.out / "results.json", results)
        print(entry["name"], "complete", flush=True)
    if digest(__file__) != source_sha256:
        raise ValueError("Experiment source changed during execution")


if __name__ == "__main__":
    main()
