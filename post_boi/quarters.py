"""Joint diagonal-Gaussian likelihood in the observed flux convention."""

from dataclasses import dataclass

import numpy as np

from post_boi.records import require


def fold_quarters(arrays, clock, other_signals, exposure, noise, flux_convention):
    """Retain every selected photon behind the existing common phase grid."""
    from post_boi.prepare import fold, phase

    original = arrays["original_index"]
    require(
        np.array_equal(noise["time_bkjd"][original], arrays["time_bkjd"]),
        "Reviewed noise does not match original photon times",
    )
    folded = fold(arrays, clock, other_signals, exposure)
    selected = np.isin(original, folded["original_index"])
    sigma = noise["sigma"][original][selected]
    applied = noise["applied_fraction"][original][selected]
    require(
        flux_convention in ("raw_aperture", "source_corrected", "psf_flat"),
        "Declare observed flux convention",
    )
    require(
        flux_convention == "source_corrected" or np.all(applied == 1),
        "Raw SAP/PSF_FLAT cannot already have a dilution correction",
    )
    x = phase(arrays["time_bkjd"][selected], clock["period_days"], clock["epoch_bkjd"])
    if len(x) <= 200:
        bins = np.arange(len(x))
    else:
        duration = clock["duration_days"]
        labels = np.minimum(
            ((x + 2 * duration) / (4 * duration) * 200).astype(int), 199
        )
        _, bins = np.unique(labels, return_inverse=True)
    require(len(np.unique(bins)) == len(folded["time"]), "Phase grid differs")
    return {
        "time": folded["time"],
        "binned_flux": folded["flux"],
        "flux": arrays["flux"][selected],
        "sigma": sigma,
        "bin_index": bins,
        "quarter": arrays["quarter"][selected],
        "time_bkjd": arrays["time_bkjd"][selected],
        "applied_fraction": applied,
        "original_index": original[selected],
    }


@dataclass
class SourceLikelihood:
    """Sufficient statistics for one resolved source and one shared transit."""

    time: np.ndarray
    centre: np.ndarray
    precision: np.ndarray
    residual: float
    log_normalization: float

    def cost(self, model):
        model = np.asarray(model, dtype=float)
        require(model.shape[-1] == len(self.time), "Model phase grid differs")
        difference = model - 1.0 - self.centre
        return 0.5 * (self.residual + np.sum(self.precision * difference**2, axis=-1))

    def __call__(self, function, time, flux, sigma, *args, **kwargs):
        """Retain upstream physical curves and its existing secondary screen."""
        from triceratops import likelihoods

        require(np.array_equal(time, self.time), "Likelihood phase grid differs")
        name = function.__name__
        require(
            name
            in (
                "lnL_TP",
                "lnL_TP_p",
                "lnL_EB",
                "lnL_EB_p",
                "lnL_EB_twin",
                "lnL_EB_twin_p",
            ),
            "Unsupported upstream likelihood",
        )
        binary = "EB" in name
        simulator = "simulate_EB_transit" if binary else "simulate_TP_transit"
        simulator += "_p" if name.endswith("_p") else ""
        result = getattr(likelihoods, simulator)(time, *args, **kwargs)
        model = result[0] if binary else result
        cost = self.cost(model)
        if binary and "twin" not in name:
            secondary = np.asarray(result[1]).reshape(np.shape(cost))
            cost = np.where(secondary < 1.5 * sigma, cost, np.inf)
        return cost


class QuarterLikelihood:
    """Keep quarter response before integrating the shared orbital parameters.

    ``sigma`` must be reviewed noise in the supplied observed flux units.
    ``applied_fraction`` records dilution already removed from each photon;
    it is one for raw SAP. It is never inferred from the current PRF model.
    The common phase grid has at most 200 points. Within-bin timing remains
    a model approximation to check against the original cadences.
    """

    def __init__(self, packet, source_ids, quarters, fractions):
        self.source_ids = np.asarray(source_ids)
        require(
            self.source_ids.ndim == 1
            and self.source_ids.dtype.kind in "US"
            and len(self.source_ids) > 0
            and len(np.unique(self.source_ids)) == len(self.source_ids),
            "Supply unique exact source identities as strings",
        )
        self.source_ids = self.source_ids.astype(str)
        self.quarters = np.asarray(quarters)
        fractions = np.asarray(fractions, dtype=float)
        require(
            self.quarters.ndim == 1
            and len(self.quarters) > 0
            and len(np.unique(self.quarters)) == len(self.quarters),
            "Duplicate or absent response quarters",
        )
        require(
            fractions.shape == (len(self.quarters), len(self.source_ids))
            and np.isfinite(fractions).all()
            and (fractions >= 0).all()
            and np.allclose(fractions.sum(axis=1), 1, rtol=0, atol=1e-10),
            "Supply complete nonnegative normalized quarter fractions",
        )
        self.time = np.asarray(packet["time"], dtype=float)
        self.flux = np.asarray(packet["flux"], dtype=float)
        self.sigma = np.asarray(packet["sigma"], dtype=float)
        self.bins = np.asarray(packet["bin_index"])
        observed_quarter = np.asarray(packet["quarter"])
        applied = np.asarray(packet["applied_fraction"], dtype=float)
        require(
            self.time.ndim == 1
            and 0 < len(self.time) <= 200
            and np.isfinite(self.time).all(),
            "Invalid common phase grid",
        )
        arrays = (self.flux, self.sigma, self.bins, observed_quarter, applied)
        require(
            all(a.ndim == 1 and a.shape == self.flux.shape for a in arrays)
            and len(self.flux) > 0,
            "Photon arrays differ",
        )
        require(
            self.bins.dtype.kind in "iu"
            and (self.bins >= 0).all()
            and (self.bins < len(self.time)).all(),
            "Invalid photon phase bins",
        )
        require(
            np.isfinite(self.flux).all()
            and np.isfinite(self.sigma).all()
            and (self.sigma > 0).all()
            and np.isfinite(applied).all()
            and (applied > 0).all()
            and (applied <= 1).all(),
            "Invalid observed flux, reviewed noise or applied dilution",
        )
        require(
            np.isin(observed_quarter, self.quarters).all(),
            "Observed quarter has no authenticated response",
        )
        index = {q: i for i, q in enumerate(self.quarters)}
        rows = np.array([index[q] for q in observed_quarter])
        self.response = fractions[rows] / applied[:, None]
        self.log_normalization = float(
            -0.5 * len(self.flux) * np.log(2 * np.pi) - np.log(self.sigma).sum()
        )

    def for_source(self, source_id):
        matches = np.flatnonzero(self.source_ids == str(source_id))
        require(len(matches) == 1, "Scenario source absent from quarter scene")
        response = self.response[:, matches[0]]
        weight = 1 / self.sigma**2
        precision = np.bincount(
            self.bins, weights=weight * response**2, minlength=len(self.time)
        )
        numerator = np.bincount(
            self.bins,
            weights=weight * response * (self.flux - 1),
            minlength=len(self.time),
        )
        centre = np.divide(
            numerator, precision, out=np.zeros_like(precision), where=precision > 0
        )
        residual = float(
            np.sum(weight * (self.flux - 1 - response * centre[self.bins]) ** 2)
        )
        return SourceLikelihood(
            self.time, centre, precision, residual, self.log_normalization
        )
