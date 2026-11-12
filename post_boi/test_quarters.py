"""Compare the compressed joint likelihood with explicit observed-flux sums."""

import unittest
from pathlib import Path

import numpy as np

from post_boi.quarters import QuarterLikelihood, fold_quarters


class QuarterLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.time = np.linspace(-0.12, 0.12, 31)
        self.shape = np.maximum(0, 1 - abs(self.time) / 0.08)
        self.fractions = np.array([[0.8, 0.2], [0.2, 0.8], [1.0, 0.0]])
        self.bins = np.tile(np.arange(len(self.time)), 3)
        self.quarter = np.repeat([3, 4, 5], len(self.time))
        self.sigma = np.linspace(0.00008, 0.0002, len(self.bins))
        self.response = np.repeat(self.fractions, len(self.time), axis=0)
        self.flux = 1 - self.response[:, 0] * 0.001 * self.shape[self.bins]
        self.flux += self.sigma * np.sin(np.arange(len(self.flux)))

    def likelihood(self, flux=None, applied=None):
        applied = np.ones(len(self.flux)) if applied is None else applied
        flux = self.flux if flux is None else flux
        return QuarterLikelihood(
            {
                "time": self.time,
                "flux": 1 + (flux - 1) / applied,
                "sigma": self.sigma / applied,
                "bin_index": self.bins,
                "quarter": self.quarter,
                "applied_fraction": applied,
            },
            np.array(["100", "200"]),
            [3, 4, 5],
            self.fractions,
        )

    def test_shared_draws_equal_direct_photon_likelihood_for_both_hosts(self):
        joint = self.likelihood()
        models = 1 - np.array([0, 0.0007, 0.001, 0.003])[:, None] * self.shape
        for i, source_id in enumerate(joint.source_ids):
            observed = 1 + self.response[:, i] * (models[:, self.bins] - 1)
            direct = -0.5 * np.sum(
                ((self.flux - observed) / self.sigma) ** 2
                + np.log(2 * np.pi * self.sigma**2),
                axis=1,
            )
            source = joint.for_source(source_id)
            np.testing.assert_allclose(
                source.log_normalization - source.cost(models),
                direct,
                rtol=1e-12,
                atol=1e-9,
            )

    def test_raw_and_corrected_flux_give_same_bayes_factors(self):
        raw = self.likelihood()
        applied = self.response[:, 0]
        corrected = self.likelihood(applied=applied)
        model = 1 - 0.001 * self.shape
        raw_cost = [raw.for_source(s).cost(model) for s in raw.source_ids]
        corrected_cost = [corrected.for_source(s).cost(model) for s in raw.source_ids]
        np.testing.assert_allclose(raw_cost, corrected_cost, rtol=1e-11, atol=1e-9)
        self.assertAlmostEqual(
            corrected.log_normalization - raw.log_normalization,
            np.log(applied).sum(),
            places=10,
        )

    def test_kepler10_curve_with_declared_injected_host_is_localized(self):
        # The real example supplies the shape; dilution/host are synthetic controls.
        path = Path(__file__).parent / "fixtures/kepler10/lightcurve.csv"
        curve = np.loadtxt(path, delimiter=",")
        labels = np.minimum(np.arange(len(curve)) * 200 // len(curve), 199)
        weight = 1 / curve[:, 2] ** 2
        total = np.bincount(labels, weights=weight)
        curve = np.column_stack(
            [
                np.bincount(labels, weights=weight * curve[:, 0]) / total,
                np.bincount(labels, weights=weight * curve[:, 1]) / total,
                total**-0.5,
            ]
        )
        self.time = curve[:, 0]
        self.assertLessEqual(len(self.time), 200)
        self.bins = np.tile(np.arange(len(self.time)), 3)
        self.quarter = np.repeat([3, 4, 5], len(self.time))
        self.response = np.repeat(self.fractions, len(self.time), axis=0)
        self.sigma = np.tile(curve[:, 2], 3)
        for host in range(2):
            self.flux = 1 + self.response[:, host] * (curve[self.bins, 1] - 1)
            joint = self.likelihood()
            costs = [joint.for_source(s).cost(curve[:, 1]) for s in joint.source_ids]
            self.assertAlmostEqual(float(costs[host]), 0, places=10)
            self.assertGreater(float(costs[1 - host]), float(costs[host]))

    def test_quarter_average_loses_observed_host_information(self):
        self.fractions = np.array([[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
        self.response = np.repeat(self.fractions, len(self.time), axis=0)
        self.flux = 1 - self.response[:, 0] * 0.001 * self.shape[self.bins]
        joint = self.likelihood()
        model = 1 - 0.001 * self.shape
        self.assertAlmostEqual(float(joint.for_source("100").cost(model)), 0, places=10)
        self.assertGreater(float(joint.for_source("200").cost(model)), 1)
        np.testing.assert_array_equal(self.fractions.mean(axis=0), [0.5, 0.5])

    def test_missing_or_negative_response_fails_explicitly(self):
        self.fractions[0] = [-0.1, 1.1]
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.likelihood()
        self.fractions[0] = [0.8, 0.2]
        self.quarter[0] = 6
        with self.assertRaisesRegex(ValueError, "no authenticated response"):
            self.likelihood()

    def test_preparation_preserves_photons_and_existing_phase_grid(self):
        from post_boi.prepare import fold

        time = np.linspace(-2, 2, 1601)
        arrays = {
            "time_bkjd": time,
            "flux": 1 - 0.001 * (abs(time) < 0.05),
            "flux_err": np.full(len(time), 0.0001),
            "quarter": np.where(time < 0, 3, 4),
            "original_index": np.arange(len(time)),
        }
        clock = {"period_days": 4, "epoch_bkjd": 0, "duration_days": 0.15}
        noise = {
            "time_bkjd": time,
            "sigma": arrays["flux_err"],
            "applied_fraction": np.ones(len(time)),
        }
        old = fold(arrays, clock, [], 0.02)
        packet = fold_quarters(arrays, clock, [], 0.02, noise, "raw_aperture")
        np.testing.assert_array_equal(packet["time"], old["time"])
        np.testing.assert_array_equal(packet["original_index"], old["original_index"])
        np.testing.assert_array_equal(packet["binned_flux"], old["flux"])
        self.assertEqual(len(packet["flux"]), len(old["original_index"]))
        noise["time_bkjd"] = time + 1e-4
        with self.assertRaisesRegex(ValueError, "original photon times"):
            fold_quarters(arrays, clock, [], 0.02, noise, "raw_aperture")


if __name__ == "__main__":
    unittest.main()
