"""Fuse native interpolation and affine detrending with the frozen CSR operators."""

import importlib

import numpy as np
from numba import njit

from pipeline.vetting.scenario_bounds import physical_intensity


@njit(cache=True, fastmath=False)
def bound_radii(
    fraction,
    quarter,
    baseline,
    radius,
    photons,
    unity,
    bin_weight,
    bin_ptr,
    bin_index,
    quarter_low,
    quarter_radius,
):
    result = np.empty((len(bin_ptr) - 1, len(fraction)))
    safe = np.ones(len(fraction), np.bool_)
    for draw in range(len(fraction)):
        certificate = True
        for q in range(fraction.shape[1]):
            if fraction[draw, q] * quarter_radius[q] >= quarter_low[q] - 1e-12:
                certificate = False
        if not certificate:
            for point in range(len(baseline)):
                if (
                    baseline[point] - fraction[draw, quarter[point]] * radius[point]
                    <= 0
                ):
                    safe[draw] = False
        for phase_bin in range(len(bin_ptr) - 1):
            total = 0.0
            for item in range(bin_ptr[phase_bin], bin_ptr[phase_bin + 1]):
                point = bin_index[item]
                f = fraction[draw, quarter[point]]
                r = f * radius[point]
                low = baseline[point] - r if safe[draw] else 1.0
                lower_model = photons[point] * (1 - f) / (baseline[point] + r)
                upper_model = photons[point] * (1 + f) / low
                deviation = max(
                    abs(upper_model - unity[point]), abs(lower_model - unity[point])
                )
                total += bin_weight[item] * deviation
            result[phase_bin, draw] = total
    return result.T, safe


def binned_bound(bound, fraction):
    like = bound.likelihood
    if like.noise != "FIXED_EXPORTED_MAP_SCALE":
        raise ValueError("Runtime requires the frozen fixed MAP noise scale")
    if not hasattr(bound, "runtime_safety"):
        bound.runtime_safety = (
            np.array(
                [
                    bound.baseline[bound.quarter_index == q].min()
                    for q in range(len(bound.quarters))
                ]
            ),
            np.array(
                [
                    bound.baseline_radius[bound.quarter_index == q].max()
                    for q in range(len(bound.quarters))
                ]
            ),
        )
    radii, safe = bound_radii(
        fraction,
        bound.quarter_index,
        bound.baseline,
        bound.baseline_radius,
        like.baseline,
        bound.unity_ratio,
        like.binning.data,
        like.binning.indptr,
        like.binning.indices,
        *bound.runtime_safety,
    )
    chi2 = np.zeros(len(fraction))
    for (index, _), (residual, whiten) in zip(
        like.blocks, bound.residuals, strict=True
    ):
        whitened = radii[:, index] @ whiten
        chi2 += (np.maximum(residual[None] - whitened, 0) ** 2).sum(axis=1)
    upper = -0.5 * chi2
    upper[~safe] = np.inf
    return upper + 1e-6


@njit(cache=True, fastmath=False)
def native_bins(
    first,
    second,
    left,
    fraction,
    flux1,
    flux2,
    quarter,
    baseline,
    constant,
    group,
    x,
    a0,
    a1,
    group_ptr,
    group_index,
    bin_weight,
    bin_ptr,
    bin_index,
):
    result = np.empty((len(bin_ptr) - 1, len(first)))
    photons = np.empty(len(left))
    c0, c1 = np.empty(len(group_ptr) - 1), np.empty(len(group_ptr) - 1)
    for draw in range(len(first)):
        for point in range(len(left)):
            lo, mix = left[point], fraction[point]
            primary = first[draw, lo] * (1 - mix) + first[draw, lo + 1] * mix
            secondary = second[draw, lo] * (1 - mix) + second[draw, lo + 1] * mix
            model = 1 - primary * flux1[draw, quarter[point]]
            model -= secondary * flux2[draw, quarter[point]]
            if not np.isfinite(model):
                raise ValueError("Nonfinite native model")
            photons[point] = model * baseline[point]
        for event in range(len(c0)):
            c0[event], c1[event] = 0.0, 0.0
            for item in range(group_ptr[event], group_ptr[event + 1]):
                c0[event] += a0[item] * photons[group_index[item]]
                c1[event] += a1[item] * photons[group_index[item]]
        for point in range(len(left)):
            affine = constant[point] + c0[group[point]] + c1[group[point]] * x[point]
            if affine <= 0 or not np.isfinite(affine):
                raise ValueError("Unsupported affine MAP baseline")
            photons[point] /= affine
        for phase_bin in range(len(bin_ptr) - 1):
            total = 0.0
            for item in range(bin_ptr[phase_bin], bin_ptr[phase_bin + 1]):
                total += bin_weight[item] * photons[bin_index[item]]
            result[phase_bin, draw] = total
    return result.T


def coordinates(likelihood, config, twin):
    if not hasattr(likelihood, "runtime_coordinates"):
        likelihood.runtime_coordinates = {}
    if twin not in likelihood.runtime_coordinates:
        period = config["period_days"]
        time = likelihood.time - config["epoch_bkjd"]
        phase = np.mod(time + period / 2, period) - period / 2
        grid = np.linspace(
            phase.min(), phase.max(), config["phase_grid_nodes_per_cycle"]
        )
        if twin:
            odd = (np.floor((time + period / 2) / period).astype(int) % 2) == 1
            phase[odd] += period
            grid = np.concatenate([grid, grid + period])
        right = np.clip(np.searchsorted(grid, phase, side="right"), 1, len(grid) - 1)
        left = right - 1
        fraction = (phase - grid[left]) / (grid[right] - grid[left])
        likelihood.runtime_coordinates[twin] = grid, left, fraction
    return likelihood.runtime_coordinates[twin]


def fractions(draws, scene, geometry, config, quarters):
    kind = draws["name"].split(":")[0]
    primary, secondary, eclipse = [], [], []
    for quarter in quarters:
        q = str(int(quarter))
        tau = {sid: value[q] for sid, value in geometry["resolved"].items()}
        if any(
            not np.isfinite(value).all() or np.any((value <= 0) | (value > 1))
            for value in tau.values()
        ):
            raise ValueError("Unsupported resolved photon response")
        total = sum(
            source["flux_relative_target"] * tau[sid] for sid, source in scene.items()
        )
        unresolved = geometry["unresolved"][q]
        if kind.startswith(("P", "S", "D", "B")) and (
            not np.isfinite(unresolved).all()
            or np.any((unresolved <= 0) | (unresolved > 1))
        ):
            raise ValueError("Unsupported unresolved photon response")
        host = unresolved if kind.startswith(("S", "B")) else tau[draws["host_source"]]
        if kind.startswith("S"):
            total += (
                draws["extra_flux"] - scene[config["source_id"]]["flux_relative_target"]
            ) * tau[config["source_id"]]
            total += draws["host_flux"] * unresolved
        elif kind.startswith("B"):
            total += draws["host_flux"] * unresolved
        else:
            total += (
                draws["host_flux"] - scene[draws["host_source"]]["flux_relative_target"]
            ) * host
            if kind.startswith(("P", "D")):
                total += draws["extra_flux"] * unresolved
        total += draws["secondary_flux"] * host
        if np.any(total <= 0) or not np.isfinite(total).all():
            raise ValueError("Nonpositive or nonfinite aperture photon denominator")
        primary.append(draws["host_flux"] * host / total)
        secondary.append(draws["secondary_flux"] * host / total)
        eclipse.append((draws["host_flux"] + draws["secondary_flux"]) * host / total)
    return np.asarray(primary).T, np.asarray(secondary).T, np.asarray(eclipse).T


def evaluate(draws, likelihood, flux1, flux2, config):
    if config["renderer"] != "PIECEWISE_LINEAR_PHASE_GRID":
        raise ValueError("The runtime requires the frozen phase-grid renderer")
    if not draws["valid"].all():
        raise ValueError("Runtime renderer requires explicitly selected physical draws")
    if not np.array_equal(
        likelihood.a0.indices, likelihood.a1.indices
    ) or not np.array_equal(likelihood.a0.indptr, likelihood.a1.indptr):
        raise ValueError("Affine CSR operators have different native point ordering")
    keys = (
        "inc",
        "ecc",
        "argp",
        "a_cm",
        "host_radius",
        "occultor_radius",
        "u1",
        "u2",
        "host_flux",
        "extra_flux",
        "secondary_flux",
        "secondary_u1",
        "secondary_u2",
    )
    values = {key: np.asarray(draws[key]) for key in keys}
    grid, left, fraction = coordinates(likelihood, config, draws["twin"])
    core = importlib.import_module("kepler_scenarios")
    first, second = core.eclipse_losses(
        grid,
        values,
        draws["period"],
        config["exposure_days"],
        config["integration_samples"],
        draws["binary"],
    )
    _, quarter = np.unique(likelihood.quarter, return_inverse=True)
    bins = native_bins(
        first,
        second,
        left,
        fraction,
        flux1,
        flux2,
        quarter,
        likelihood.baseline,
        likelihood.constant,
        likelihood.group,
        likelihood.x,
        likelihood.a0.data,
        likelihood.a1.data,
        likelihood.a0.indptr,
        likelihood.a0.indices,
        likelihood.binning.data,
        likelihood.binning.indptr,
        likelihood.binning.indices,
    )
    return likelihood.from_bins(bins)


def evaluate_sample(
    state, physical, ratio, scene, geometry, bound, cutoff, render_batch
):
    from pipeline.vetting.statistical_validation import subset

    likelihood, config = state[3], state[5]
    size = physical["size"]
    logs, bounded = np.full(size, -np.inf), np.zeros(size, dtype=bool)
    selected = np.flatnonzero(physical["valid"])
    if not len(selected):
        return logs, bounded
    if set(scene) != set(geometry["resolved"]):
        raise ValueError("Resolved geometry differs from the complete source census")
    current = subset(physical, selected, size)
    current["size"] = len(selected)
    scene = {
        sid: {
            "flux_relative_target": np.asarray(source["flux_relative_target"])[selected]
        }
        for sid, source in scene.items()
    }
    geometry = subset(geometry, selected, size)
    quarters = np.unique(likelihood.quarter)
    flux1, flux2, eclipse = fractions(current, scene, geometry, config, quarters)
    for begin in range(0, len(selected), render_batch):
        choose = np.arange(begin, min(begin + render_batch, len(selected)))
        index = selected[choose]
        omitted = np.zeros(len(index), dtype=bool)
        if bound is not None and physical["name"].startswith("N"):
            if bound.native:
                raise ValueError("Runtime requires the frozen binned likelihood bound")
            fraction = eclipse[choose]
            if not np.isfinite(fraction).all() or np.any(
                (fraction <= 0) | (fraction > 1 + 1e-12)
            ):
                raise ValueError(
                    "Eclipse fraction is outside the physical aperture-flux domain"
                )
            upper = binned_bound(bound, np.minimum(fraction + 1e-12, 1.0))
            proper = physical_intensity(current["u1"][choose], current["u2"][choose])
            if physical["binary"]:
                proper &= physical_intensity(
                    current["secondary_u1"][choose], current["secondary_u2"][choose]
                )
            upper[~proper] = np.inf
            upper = upper + current["logprior"][choose] + ratio[index]
            omitted = upper < cutoff
            bounded[index[omitted]] = True
        actual = choose[~omitted]
        if len(actual):
            part = subset(current, actual, len(selected))
            part["size"] = len(actual)
            values = (
                evaluate(part, likelihood, flux1[actual], flux2[actual], config)
                + part["logprior"]
                + ratio[selected[actual]]
            )
            if np.isnan(values).any() or np.isposinf(values).any():
                raise ValueError("Invalid importance weight")
            logs[selected[actual]] = values
    return logs, bounded
