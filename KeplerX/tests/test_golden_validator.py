"""Golden: the Validator's preparation, prior transform and native likelihood on a production case."""

import hashlib
import json
import pickle
import shutil

import numpy as np
import pytest
from conftest import FIXTURES, GOLDEN

from validator.native_fit import native_photons
from validator.native_likelihood import NativeLikelihood, PriorTransform
from validator.preparation import prepare
from validator.records import candidate

CASE = FIXTURES / "validator-m-dwarf-2126055064030395392-1"


def reference():
    return json.loads((GOLDEN / "validator.json").read_text())


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@pytest.fixture
def case(tmp_path):
    for name in ("candidate.json", "native-photons.npz", "native-priors.json"):
        shutil.copy(CASE / name, tmp_path / name)
    prepare(tmp_path / "candidate.json", tmp_path / "prepared")
    return tmp_path


def test_preparation_reproduces_production_bit_for_bit(case):
    with np.load(case / "prepared" / "detrended.npz") as arrays:
        prepared = {name: digest(arrays[name]) for name in arrays.files}
    assert prepared == reference()["prepared_sha256"]


def test_prior_transform_and_likelihood_match_juliet(case):
    ref = reference()
    c = candidate(case / "candidate.json")
    priors = json.loads((case / "native-priors.json").read_text())
    prior = pickle.loads(pickle.dumps(PriorTransform(priors)))
    for cube, value in zip(ref["prior_cubes"], ref["prior_values"]):
        assert np.array_equal(prior(np.array(cube)), np.array(value))
    like = NativeLikelihood(
        priors, *native_photons(c, case / "prepared"), c["native"]["supersample"],
        c["exposure_days"],
    )  # fmt: skip
    like = pickle.loads(pickle.dumps(like))
    vectors = np.array(ref["likelihood_vectors"])
    values, modelled = like.rows(vectors)
    expected = np.array(ref["juliet_loglike"])
    rejected = expected <= -1e100
    assert np.array_equal(values <= -1e100, rejected)
    k = ref["likelihood_posterior_rows"]
    assert np.max(np.abs(values[:k] - expected[:k])) < 1e-8
    special = slice(
        k + len(ref["prior_values"]), None
    )  # circular, very eccentric, central
    assert np.max(np.abs(values[special] - expected[special])) < 1e-8
    relative = np.abs(values[~rejected] - expected[~rejected]) / np.abs(
        expected[~rejected]
    )
    assert relative.max() < 1e-9
    assert np.array_equal(values, [like(v) for v in vectors])
    # Photons outside the transit windows stay 1; the rest agree to rounding.
    accepted = vectors[~rejected]
    full = np.array([like.model(v, everywhere=True)[0] for v in accepted])
    windowed = np.array([like.model(v)[0] for v in accepted])
    assert np.all(full[windowed == 1.0] == 1.0)
    assert np.max(np.abs(full - windowed)) < 1e-10
    assert modelled[:k].max() < like.time.size
