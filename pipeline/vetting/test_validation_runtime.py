from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.vetting import validation_models, validation_scene


@pytest.mark.parametrize("size", (1, 256, 8192))
def test_cached_particle_sampling_preserves_indices_and_generator_state(size):
    logs = np.log(
        [0.1, 0.0, 0.3, 0.6],
        where=np.array([True, False, True, True]),
        out=np.full(4, -np.inf),
    )
    fields = (
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
    bank = {
        "log_weights": logs,
        "values": {
            key: np.ones(4, dtype=bool)
            if key == "radiometry_supported"
            else np.arange(4.0)
            for key in fields
        },
    }
    prior = SimpleNamespace(
        ids=["a", "b"], banks={"a": bank, "b": bank}, tau=np.ones((2, 5, 2))
    )
    validation_scene.prepare(prior, "a", 2.2)
    expected, actual = np.random.default_rng(17492), np.random.default_rng(17492)
    probability = np.exp(logs - np.log(np.exp(logs).sum()))
    picks = {sid: expected.choice(4, size, p=probability) for sid in prior.ids}
    _, supported, indices = validation_scene.select(prior, size, actual)
    assert supported.all()
    for sid in prior.ids:
        np.testing.assert_array_equal(indices[sid], picks[sid])
    assert actual.bit_generator.state == expected.bit_generator.state


def native_case():
    return {
        "first": np.array([[0.0, 0.2, 0.4]]),
        "second": np.zeros((1, 3)),
        "left": np.array([0, 0, 1]),
        "fraction": np.array([0.0, 0.5, 0.5]),
        "flux1": np.array([[0.5]]),
        "flux2": np.zeros((1, 1)),
        "quarter": np.zeros(3, int),
        "baseline": np.ones(3),
        "constant": np.ones(3),
        "group": np.zeros(3, int),
        "x": np.zeros(3),
        "a0": np.zeros(3),
        "a1": np.zeros(3),
        "group_ptr": np.array([0, 3]),
        "group_index": np.arange(3),
        "bin_weight": np.array([0.5, 0.5, 1.0]),
        "bin_ptr": np.array([0, 2, 3]),
        "bin_index": np.arange(3),
    }


def test_native_bins_preserve_photon_dilution_and_weighted_phase_bins():
    np.testing.assert_allclose(
        validation_models.native_bins(**native_case()),
        [[0.975, 0.85]],
        rtol=0,
        atol=1e-15,
    )


def test_native_bins_reject_nonpositive_affine_baseline():
    inputs = native_case()
    inputs["constant"] = np.zeros(3)
    with pytest.raises(ValueError, match="affine MAP baseline"):
        validation_models.native_bins(**inputs)


def test_native_bins_reject_nonfinite_photon_models():
    inputs = native_case()
    inputs["first"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite native model"):
        validation_models.native_bins(**inputs)
