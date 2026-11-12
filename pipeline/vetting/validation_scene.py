"""Cache fixed stellar-particle CDFs without changing the random draw stream."""

import importlib
import types

import numpy as np
from scipy.special import logsumexp


def prepare(prior, target, radius):
    prior.runtime_cdf = {}
    for sid in prior.ids:
        weights = prior.banks[sid]["log_weights"]
        probability = np.exp(weights - logsumexp(weights))
        cumulative = probability.cumsum()
        cumulative /= cumulative[-1]
        cumulative.setflags(write=False)
        prior.runtime_cdf[sid] = cumulative
    prior.runtime_radius = radius
    prior.runtime_target = target
    prior.runtime_pointing_count = prior.tau.shape[1]
    prior.runtime_tau = np.ascontiguousarray(prior.tau.transpose(2, 0, 1))
    prior.sample = types.MethodType(sample, prior)


def select(prior, size, rng):
    selected, indices = {}, {}
    supported = np.ones(size, dtype=bool)
    for sid in prior.ids:
        pick = prior.runtime_cdf[sid].searchsorted(rng.random(size), side="right")
        values = prior.banks[sid]["values"]
        selected[sid] = {
            key: np.asarray(values[key])[pick]
            for key in (
                "mass",
                "radius",
                "teff",
                "feh",
                "av",
                "distance_pc",
                "g_residual_mag",
                "anchored_kp",
                "radiometry_supported",
            )
        }
        supported &= selected[sid]["radiometry_supported"] & np.isfinite(
            selected[sid]["anchored_kp"]
        )
        indices[sid] = pick
    return selected, supported, indices


def scene_values(selected, target):
    core = importlib.import_module("kepler_scenarios")
    scene = {}
    for sid, values in selected.items():
        star = core.star(
            *(values[key] for key in ("mass", "radius", "teff", "feh", "av"))
        )
        star["distance_pc"] = values["distance_pc"]
        star["g_residual_mag"] = values["g_residual_mag"]
        scene[sid] = {
            "star": star,
            "flux_relative_target": 10
            ** (-0.4 * (values["anchored_kp"] - selected[target]["anchored_kp"])),
        }
    return scene


def unresolved_geometry(prior, states, position):
    left = np.searchsorted(prior.axis, position, side="right") - 1
    left = np.clip(left, 0, len(prior.axis) - 2)
    fraction = (position - prior.axis[left]) / (prior.axis[left + 1] - prior.axis[left])
    output = {}
    for qi, quarter in enumerate(prior.quarters):
        value = np.zeros(len(position))
        for di in (0, 1):
            for ri in (0, 1):
                weight = (fraction[:, 0] if di else 1 - fraction[:, 0]) * (
                    fraction[:, 1] if ri else 1 - fraction[:, 1]
                )
                value += (
                    weight
                    * prior.unresolved_tau[
                        qi, states[:, qi], left[:, 0] + di, left[:, 1] + ri
                    ]
                )
        output[str(quarter)] = value
    return output


def sample(prior, size, rng, unresolved_radius_arcsec=None):
    radius = prior.runtime_radius
    if unresolved_radius_arcsec is not None and unresolved_radius_arcsec != radius:
        raise ValueError("Runtime unresolved radius differs from the frozen prior")
    if not isinstance(rng, np.random.Generator) or size <= 0:
        raise ValueError("An explicit Generator and positive sample size are required")
    selected, supported, indices = select(prior, size, rng)
    scene = scene_values(selected, prior.runtime_target)
    shape = (size, len(prior.quarters))
    states = (
        np.zeros(shape, dtype=int)
        if prior.pointing == "NOMINAL"
        else rng.integers(0, prior.runtime_pointing_count, shape)
    )
    resolved = np.take_along_axis(prior.runtime_tau, states.T[None], axis=2)
    geometry = {
        "resolved": {
            sid: {str(q): resolved[si, qi] for qi, q in enumerate(prior.quarters)}
            for si, sid in enumerate(prior.ids)
        }
    }
    radial = radius * np.sqrt(rng.random(size))
    angle = 2 * np.pi * rng.random(size)
    position = np.column_stack([radial * np.sin(angle), radial * np.cos(angle)])
    geometry["unresolved"] = unresolved_geometry(prior, states, position)
    return (
        scene,
        geometry,
        supported,
        indices,
        states,
        position,
        selected[prior.runtime_target]["anchored_kp"],
    )
