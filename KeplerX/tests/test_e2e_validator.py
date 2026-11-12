"""End-to-end: the Validator's preparation and native fit of a production M-dwarf signal."""

import hashlib
import json
import shutil

import numpy as np
from conftest import FIXTURES, GOLDEN

from validator.native_fit import fit
from validator.preparation import prepare

CASE = FIXTURES / "validator-m-dwarf-2126055064030395392-1"


def test_native_fit_reproduces_production_posterior(tmp_path):
    ref = json.loads((GOLDEN / "validator.json").read_text())
    for name in ("candidate.json", "native-photons.npz", "native-priors.json"):
        shutil.copy(CASE / name, tmp_path / name)
    prepare(tmp_path / "candidate.json", tmp_path / "prepared")
    fit(tmp_path / "candidate.json", tmp_path / "prepared", tmp_path / "native")
    with np.load(tmp_path / "native" / "posterior.npz") as posterior:
        samples = np.ascontiguousarray(posterior["unnamed"])
    assert list(samples.shape) == ref["posterior_shape"]
    assert hashlib.sha256(samples.tobytes()).hexdigest() == ref["posterior_sha256"]
    sampler = json.loads((tmp_path / "native" / "sampler.json").read_text())
    assert sampler["calls"] == ref["calls"]
    assert abs(sampler["lnZ"] - ref["lnZ"]) < 1e-8
    review = json.loads((tmp_path / "native" / "review.json").read_text())
    assert {k: review[k] for k in ref["review_clock"]} == ref["review_clock"]
