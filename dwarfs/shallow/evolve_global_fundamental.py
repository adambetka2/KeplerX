"""Final conditional transit likelihood with one coherent harmonic background."""

import numpy as np
from scipy.sparse import csr_matrix

SETTINGS = {
    "method": "profiled_fixed_diagonal_global_fundamental",
    "stage": "after_final_distinct_proposals",
}
EPS = np.finfo(float).eps


def check_settings(config):
    if config["global_fundamental_conditioning"] != SETTINGS:
        raise ValueError("Explicit fixed diagonal global fundamental settings required")


class Metric:
    def __init__(self, band):
        self.data = band["preprocessing_arrays"]
        data = self.data
        basis = csr_matrix(
            (data["basis_data"], data["basis_indices"], data["basis_indptr"]),
            shape=band["basis_shape"],
        )
        self.segments, self.arrays = [], {}
        for number, label in enumerate(np.unique(data["segment"])):
            index = np.flatnonzero(data["segment"] == label)
            local = basis[index]
            columns = np.unique(local.indices)
            white_basis = local[:, columns].toarray() / data["noise"][index, None]
            q, r = np.linalg.qr(white_basis, mode="reduced")
            if r.shape[0] != r.shape[1] or np.any(
                np.abs(np.diag(r))
                <= 128 * EPS * max(white_basis.shape) * np.linalg.norm(r)
            ):
                raise ValueError("Full rank fixed nuisance basis required")
            flux = data["prepared_flux"][index] / data["noise"][index]
            residual = flux - q @ (q.T @ flux)
            self.segments.append((index, q, residual))
            self.arrays.update(
                {
                    f"segment_{number}_" + k: v
                    for k, v in {
                        "cadence_index": index,
                        "basis_columns": columns,
                        "whitened_basis": white_basis,
                        "nuisance_q": q,
                        "nuisance_r": r,
                        "whitened_flux": flux,
                        "projected_flux": residual,
                    }.items()
                }
            )

    def project(self, design):
        z, y, diagonal, arrays = [], [], np.zeros(design.shape[1]), {}
        for number, (index, q, residual) in enumerate(self.segments):
            white = design[index] / self.data["noise"][index, None]
            projected = white - q @ (q.T @ white)
            z.append(projected)
            y.append(residual)
            diagonal += np.einsum("ij,ij->j", white, white)
            arrays.update(
                {
                    f"segment_{number}_" + k: v
                    for k, v in {
                        "template_design": design[index],
                        "whitened_templates": white,
                        "projected_templates": projected,
                    }.items()
                }
            )
        return np.concatenate(z), np.concatenate(y), diagonal, arrays


def reference(proposal, band):
    matches = []
    for number, refinement in enumerate(band["physical_refinements"]):
        trials = [r for r in refinement["trials"] if "snr" in r]
        if not trials:
            continue
        best = max(trials, key=lambda r: r["snr"])
        if (
            best["period_days"],
            best["t0_bkjd"],
            24 * best["duration_days"],
            best["profile_index"],
        ) == (
            proposal["period_days"],
            proposal["t0_bkjd"],
            proposal["duration_hours"],
            proposal["profile_index"],
        ):
            matches.append((number, refinement, best))
    if not matches:
        raise ValueError("Final proposal requires exact stored best physical trial")
    return matches[0], [m[0] for m in matches]


def conditioned(z, flux, diagonal):
    scales = np.sqrt(diagonal[1:])
    use = scales > 0
    u, singular, vh = np.linalg.svd(z[:, 1:][:, use] / scales[use], full_matrices=False)
    tolerance = 128 * EPS * max(1, int(use.sum()))
    identified = singular**2 > tolerance
    q = u[:, identified]
    residual = flux - q @ (q.T @ flux)
    original = z[:, 0] - q @ (q.T @ z[:, 0])
    c, e = float(-original @ residual), float(original @ original)
    measurable = e > 128 * EPS * diagonal[0]
    result = {
        "background_rank": int(identified.sum()),
        "rank_tolerance": tolerance,
        "measurable": bool(measurable),
        "C": c,
        "E": e,
        "signed_snr": c / np.sqrt(e) if measurable else None,
        "depth": c / e if measurable else None,
        "variance": 1 / e if measurable else None,
    }
    arrays = {
        "background_scales": scales,
        "background_nonzero_mask": use,
        "background_singular_values": singular,
        "background_right_vectors": vh,
        "background_identified_mask": identified,
        "background_span_q": q,
        "conditioned_flux": residual,
        "conditioned_template": original,
        "unprojected_diagonal": diagonal,
    }
    return result, arrays


def measure(proposal, band, metric, physical, exposure):
    (number, refinement, best), matches = reference(proposal, band)
    table = tuple(
        refinement["profile_arrays"][f"table_{best['table_index']}_{name}"]
        for name in ["phase_days", "integrated_loss"]
    )
    time = metric.data["time"]
    index, values = physical.sampled_template(
        time, best["period_days"], best["t0_bkjd"], table
    )
    original = np.zeros(len(time))
    original[index] = values
    phase = np.remainder((time - time.min()) / best["period_days"], 1.0)
    response = np.sinc(exposure / best["period_days"])
    design = np.c_[
        original,
        response * np.sin(2 * np.pi * phase),
        response * np.cos(2 * np.pi * phase),
    ]
    z, flux, diagonal, arrays = metric.project(design)
    native_c, native_e = float(-z[:, 0] @ flux), float(z[:, 0] @ z[:, 0])
    np.testing.assert_allclose(native_c, best["C"], rtol=3e-10, atol=1e-7)
    np.testing.assert_allclose(native_e, best["E"], rtol=3e-10, atol=1e-5)
    np.testing.assert_allclose(
        native_c / np.sqrt(native_e), best["snr"], rtol=3e-10, atol=1e-9
    )
    result, saved = conditioned(z, flux, diagonal)
    arrays.update(saved)
    arrays.update(
        template_index=index,
        template_values=values,
        physical_table_phase_days=table[0],
        physical_table_integrated_loss=table[1],
        projected_gram=z.T @ z,
        correlation=-z.T @ flux,
    )
    return {
        "saved_refinement_indices": matches,
        "saved_best_trial": best,
        "common_native_C": native_c,
        "common_native_E": native_e,
        "statistic": result,
        "arrays": arrays,
    }


def rejection(stat, config):
    if not stat["measurable"]:
        return ["GLOBAL_FUNDAMENTAL_TRANSIT_UNIDENTIFIABLE"]
    reasons = []
    if stat["signed_snr"] < config["search_snr_min"]:
        reasons.append("GLOBAL_FUNDAMENTAL_CONDITIONAL_SNR")
    if not config["depth_ppm_min"] <= stat["depth"] * 1e6 <= config["depth_ppm_max"]:
        reasons.append("GLOBAL_FUNDAMENTAL_CONDITIONAL_DEPTH_DOMAIN")
    return reasons


def apply(proposals, evidence, config, physical):
    check_settings(config)
    trace = {
        "method": SETTINGS,
        "source_proposals": proposals,
        "contexts": {},
        "decisions": [],
    }
    evidence["global_fundamental_conditioning"] = trace
    contexts, retained = {}, []
    for number, proposal in enumerate(proposals):
        name = proposal["search_band"]
        bands = [
            b
            for b in evidence["bands"]
            if b.get("band") == name + "_physical" and b["status"] == "SEARCHED"
        ]
        if len(bands) != 1:
            raise ValueError(
                "Exactly one native physical context per proposal required"
            )
        if name not in contexts:
            metric = Metric(bands[0])
            contexts[name] = metric
            trace["contexts"][name] = {
                "basis_shape": bands[0]["basis_shape"],
                "arrays": metric.arrays,
            }
        measurement = measure(
            proposal,
            bands[0],
            contexts[name],
            physical,
            config["physical_template"]["exposure_days"],
        )
        reasons = rejection(measurement["statistic"], config)
        trace["decisions"].append(
            {
                "proposal_index": number,
                "retained": not reasons,
                "reasons": reasons,
                **measurement,
            }
        )
        if not reasons:
            retained.append(proposal)
    return retained
