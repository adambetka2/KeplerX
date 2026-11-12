"""Conservative likelihood bounds for diluted eclipses on frozen observations."""

import numpy as np


def eclipse_fraction(draws, scene, geometry, config, quarters):
    kind = draws["name"].split(":")[0]
    host_unresolved = kind.startswith(("S", "B"))
    extra_unresolved = kind.startswith(("P", "D"))
    fractions = []
    for quarter in quarters:
        q = str(int(quarter))
        throughputs = {sid: values[q] for sid, values in geometry["resolved"].items()}
        total = sum(
            source["flux_relative_target"] * throughputs[sid]
            for sid, source in scene.items()
        )
        unresolved = geometry["unresolved"][q]
        host = unresolved if host_unresolved else throughputs[draws["host_source"]]
        if kind.startswith("S"):
            total += (
                draws["extra_flux"] - scene[config["source_id"]]["flux_relative_target"]
            ) * throughputs[config["source_id"]]
            total += draws["host_flux"] * unresolved
        elif kind.startswith("B"):
            total += draws["host_flux"] * unresolved
        else:
            total += (
                draws["host_flux"] - scene[draws["host_source"]]["flux_relative_target"]
            ) * host
            if extra_unresolved:
                total += draws["extra_flux"] * unresolved
        total += draws["secondary_flux"] * host
        fractions.append((draws["host_flux"] + draws["secondary_flux"]) * host / total)
    fraction = np.asarray(fractions).T
    if not np.isfinite(fraction).all() or np.any(
        (fraction <= 0) | (fraction > 1 + 1e-12)
    ):
        raise ValueError(
            "Eclipse fraction is outside the physical aperture-flux domain"
        )
    return np.minimum(fraction + 1e-12, 1.0)


def physical_intensity(u1, u2):
    vertex = np.divide(-u1, 2 * u2, out=np.zeros_like(u1), where=u2 < 0)
    vertex = np.clip(vertex, 0, 1)
    return (np.minimum(1 - u1 - u2, 1 - u1 * vertex - u2 * vertex**2) >= 0) & (
        1 - u1 / 3 - u2 / 6 > 0
    )


class LikelihoodBound:
    def __init__(self, likelihood, native):
        self.likelihood = likelihood
        self.native = native
        self.quarters, self.quarter_index = np.unique(
            likelihood.quarter, return_inverse=True
        )
        group = likelihood.group
        for value in np.unique(group):
            if np.unique(likelihood.quarter[group == value]).size != 1:
                raise ValueError(
                    "A bound requires the frozen event baseline to stay within its quarter"
                )
        if native:
            self.prepare_native()
        else:
            self.prepare_binned()

    def prepare_binned(self):
        like = self.likelihood
        self.reference = like.bin_models(np.ones((1, len(like.time))))[0]
        baseline = like.constant + np.asarray(like.a0 @ like.baseline)[like.group]
        baseline += np.asarray(like.a1 @ like.baseline)[like.group] * like.x
        self.baseline = baseline
        self.unity_ratio = like.baseline / baseline
        self.baseline_radius = np.asarray(abs(like.a0) @ like.baseline)[like.group]
        self.baseline_radius += (
            abs(like.x) * np.asarray(abs(like.a1) @ like.baseline)[like.group]
        )
        if np.any(like.baseline <= 0) or np.any(baseline <= 0):
            raise ValueError(
                "Positive prepared photon and unity baseline values required"
            )
        self.residuals = [
            (abs((like.data[index] - self.reference[index]) @ whiten), abs(whiten))
            for index, whiten in like.blocks
        ]

    def prepare_native(self):
        like = self.likelihood
        self.unity_log_likelihood = float(like(np.ones((1, len(like.time))))[0])
        w, x = like.weight, like.x
        h00 = np.bincount(like.group, weights=w)
        h01 = np.bincount(like.group, weights=w * x)
        h11 = np.bincount(like.group, weights=w * x * x)
        matrix = np.array([[h00, h01], [h01, h11]]).transpose(2, 0, 1)
        eigenvalue = np.linalg.eigvalsh(matrix)
        if np.any(eigenvalue[:, 0] <= 0):
            raise ValueError("Positive unity baseline design singular values required")
        self.minimum_singular = np.sqrt(eigenvalue[:, 0])
        self.maximum_singular = np.sqrt(eigenvalue[:, 1])
        self.weight_norm = np.sqrt(h00)
        residual = like.flux - 1
        u0 = np.bincount(like.group, weights=w * residual)
        u1 = np.bincount(like.group, weights=w * residual * x)
        coefficient = np.linalg.solve(matrix, np.array([u0, u1]).T)
        remaining = (
            residual - coefficient[like.group, 0] - coefficient[like.group, 1] * x
        )
        chi2 = np.bincount(like.group, weights=w * remaining**2)
        precision = 1 / np.asarray(like.baseline_sigma) ** 2
        self.residual_norm = np.sqrt(
            np.exp(2) * chi2 + (coefficient**2 * precision).sum(axis=1)
        )
        denominator = np.exp(-2) * eigenvalue[:, 0] + precision.min()
        self.coefficient_norm = np.linalg.norm(coefficient, axis=1)
        self.coefficient_norm += (
            np.linalg.norm(coefficient * precision, axis=1) / denominator
        )
        self.group_quarter = np.array(
            [
                self.quarter_index[np.flatnonzero(like.group == value)[0]]
                for value in np.unique(like.group)
            ]
        )

    def __call__(self, draws, scene, geometry, config):
        fraction = eclipse_fraction(draws, scene, geometry, config, self.quarters)
        upper = (
            self.native_bound(fraction) if self.native else self.binned_bound(fraction)
        )
        proper = physical_intensity(draws["u1"], draws["u2"])
        if draws["binary"]:
            proper &= physical_intensity(draws["secondary_u1"], draws["secondary_u2"])
        upper[~proper] = np.inf
        return upper

    def binned_bound(self, fraction):
        like = self.likelihood
        f = fraction[:, self.quarter_index]
        radius = f * self.baseline_radius[None]
        low = self.baseline[None] - radius
        safe = np.all(low > 0, axis=1)
        low[~safe] = 1.0
        high = self.baseline[None] + radius
        lower_model = like.baseline[None] * (1 - f) / high
        upper_model = like.baseline[None] * (1 + f) / low
        deviation = np.maximum(
            abs(upper_model - self.unity_ratio[None]),
            abs(lower_model - self.unity_ratio[None]),
        )
        binned_radius = (like.binning @ deviation.T).T
        chi2 = np.zeros(len(fraction))
        for (index, _), (residual, whiten) in zip(
            like.blocks, self.residuals, strict=True
        ):
            whitened_radius = binned_radius[:, index] @ whiten
            chi2 += (np.maximum(residual[None] - whitened_radius, 0) ** 2).sum(axis=1)
        if like.noise == "FIXED_EXPORTED_MAP_SCALE":
            upper = -0.5 * chi2
        elif like.noise == "COMMON_UNIFORM_LOG_SCALE_MINUS1_PLUS1":
            from binned_scenario_likelihood import scale_integral

            upper = scale_integral(chi2, like.rank)
        else:
            raise ValueError("Unsupported frozen noise measure")
        upper[~safe] = np.inf
        return upper + 1e-6

    def native_bound(self, fraction):
        f = fraction[:, self.group_quarter]
        perturbation = f * self.maximum_singular[None]
        denominator = self.minimum_singular[None] - perturbation
        safe = np.all((denominator > 0) & (f < 1), axis=1)
        denominator[~safe] = 1.0
        coefficient = perturbation / denominator
        residual_change = coefficient * self.residual_norm[None]
        residual_change += (
            np.exp(1)
            * f
            * (self.weight_norm + self.maximum_singular * self.coefficient_norm)[None]
        )
        gain = (self.residual_norm[None] * residual_change).sum(axis=1)
        gain -= 2 * np.log1p(-np.minimum(f, 1 - 1e-15)).sum(axis=1)
        upper = self.unity_log_likelihood + gain + 1e-5
        upper[~safe] = np.inf
        return upper
