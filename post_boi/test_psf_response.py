"""Known-example shape and direct photon likelihood controls for signed response."""

from pathlib import Path

import numpy as np
import pytest

from post_boi.psf_response import PSFFlatLikelihood
from post_boi.quarters import QuarterLikelihood

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def observed():
    # Real upstream example shape; response, host and added noise are declared controls.
    curve = np.loadtxt(
        REPO / "post_boi/fixtures/kepler10/lightcurve.csv", delimiter=","
    )
    labels = np.minimum(np.arange(len(curve)) * 100 // len(curve), 99)
    weight = 1 / curve[:, 2] ** 2
    total = np.bincount(labels, weights=weight)
    time = np.bincount(labels, weights=weight * curve[:, 0]) / total
    model = np.bincount(labels, weights=weight * curve[:, 1]) / total
    bins = np.tile(np.arange(len(time)), 3)
    response = np.column_stack(
        (
            0.99 + 0.01 * np.sin(np.arange(len(bins))),
            0.95 + 0.01 * np.cos(np.arange(len(bins))),
            -0.03 * np.ones(len(bins)),
        )
    )
    sigma = np.tile(total**-0.5, 3)
    flux = 1 + response[:, 0] * (model[bins] - 1) + sigma * np.sin(np.arange(len(bins)))
    packet = dict(
        time=time,
        flux=flux,
        sigma=sigma,
        bin_index=bins,
        quarter=np.repeat([3, 4, 5], len(time)),
        applied_fraction=np.ones(len(bins)),
        original_index=np.arange(len(bins)) * 2,
    )
    return packet, response, model


def test_signed_nonnormalized_response_matches_direct_known_shape_likelihood(observed):
    packet, response, model = observed
    joint = PSFFlatLikelihood(
        packet, ["1", "2", "3"], packet["original_index"], response
    )
    models = 1 + np.array([0, 0.5, 1, 2])[:, None] * (model - 1)
    for i, sid in enumerate(joint.source_ids):
        expected = 1 + response[:, i] * (models[:, packet["bin_index"]] - 1)
        direct = -0.5 * np.sum(
            ((packet["flux"] - expected) / packet["sigma"]) ** 2
            + np.log(2 * np.pi * packet["sigma"] ** 2),
            axis=1,
        )
        source = joint.for_source(sid)
        np.testing.assert_allclose(
            source.log_normalization - source.cost(models),
            direct,
            rtol=1e-11,
            atol=1e-8,
        )
    np.testing.assert_array_equal(joint.response, response)


def test_negative_response_reproduces_a_transit_induced_brightening(observed):
    packet, response, model = observed
    packet = dict(packet, flux=1 + response[:, 2] * (model[packet["bin_index"]] - 1))
    joint = PSFFlatLikelihood(
        packet, ["1", "2", "3"], packet["original_index"], response
    )
    assert float(joint.for_source("3").cost(model)) < 1e-16
    assert float(joint.for_source("1").cost(model)) > 1


def test_aperture_limit_matches_existing_source_kernel(observed):
    packet, _, model = observed
    fractions = np.array([[0.8, 0.2], [0.4, 0.6], [1, 0]])
    response = np.repeat(fractions, len(packet["time"]), axis=0)
    psf = PSFFlatLikelihood(packet, ["1", "2"], packet["original_index"], response)
    sap = QuarterLikelihood(packet, np.array(["1", "2"]), [3, 4, 5], fractions)
    for sid in psf.source_ids:
        np.testing.assert_allclose(
            psf.for_source(sid).cost(model),
            sap.for_source(sid).cost(model),
            rtol=0,
            atol=0,
        )
        assert (
            psf.for_source(sid).log_normalization
            == sap.for_source(sid).log_normalization
        )


def test_photon_binding_and_duplicate_dilution_fail(observed):
    packet, response, _ = observed
    with pytest.raises(ValueError, match="original photon indices"):
        PSFFlatLikelihood(
            packet, ["1", "2", "3"], packet["original_index"][::-1], response
        )
    with pytest.raises(ValueError, match="second dilution"):
        PSFFlatLikelihood(
            dict(packet, applied_fraction=packet["applied_fraction"] * 0.8),
            ["1", "2", "3"],
            packet["original_index"],
            response,
        )
