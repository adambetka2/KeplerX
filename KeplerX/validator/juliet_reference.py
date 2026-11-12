"""Write tests/golden/validator.json from a production case and juliet itself."""

import hashlib
import json
import pickle
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from validator.native_fit import native_photons
from validator.records import candidate


class _Captured(Exception):
    pass


def juliet_functions(case):
    """juliet's loglike and prior_transform_r, as post_boi.native builds them."""
    import dynesty
    import juliet

    c = candidate(case / "candidate.json")
    n = c["native"]
    priors = json.loads(Path(n["priors"]).read_text())
    t, y, e = native_photons(c, case / "prepared")
    box = {}

    def stub(loglike, prior_transform, ndim, **kwargs):
        box.update(loglike=loglike, prior_transform=prior_transform)
        raise _Captured()

    saved, dynesty.DynamicNestedSampler = dynesty.DynamicNestedSampler, stub
    try:
        data = juliet.load(
            priors=priors, starting_point=None, t_lc={"Kepler": t}, y_lc={"Kepler": y},
            yerr_lc={"Kepler": e}, out_folder=tempfile.mkdtemp() + "/",
            lc_instrument_supersamp=["Kepler"], lc_n_supersamp=[n["supersample"]],
            lc_exptime_supersamp=[c["exposure_days"]],
        )  # fmt: skip
        options = {k: v for k, v in n["fit_options"].items() if k != "nthreads"}
        try:
            data.fit(**options, rstate=np.random.default_rng(n["seed"]))
        except _Captured:
            pass
    finally:
        dynesty.DynamicNestedSampler = saved
    return box


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main(case, output):
    case = Path(case)
    box = juliet_functions(case)
    prepared = np.load(case / "prepared" / "detrended.npz")
    posterior = np.load(case / "native" / "posterior.npz")
    stored = pickle.loads(
        (case / "native" / "juliet" / "_dynesty_DNS_posteriors.pkl").read_bytes()
    )
    log = (case / "native.log").read_bytes().decode(errors="ignore")
    review = json.loads((case / "native" / "review.json").read_text())
    rng = np.random.default_rng(20261102)
    cubes = rng.uniform(size=(150, posterior["unnamed"].shape[1]))
    prior_values = [box["prior_transform"](u.copy()) for u in cubes]
    samples = posterior["unnamed"][
        np.linspace(0, len(posterior["unnamed"]) - 1, 100).astype(int)
    ]
    # Branches random draws rarely reach: circular timing, very eccentric orbits, central transits.
    priors = json.loads((case / "native-priors.json").read_text())
    names = [n for n, prior in priors.items() if prior["distribution"] != "fixed"]
    ecc, omega, impact = (names.index(n) for n in ("ecc_p1", "omega_p1", "b_p1"))
    special = []
    for e, w, b in [(0.0, 90.0, None), (5e-6, 30.0, None), (0.9, 90.0, None), (0.95, 270.0, None),
                    (0.97, 10.0, None), (0.3, 180.0, 0.0), (0.0, 0.0, 0.0)]:  # fmt: skip
        v = samples[len(special)].copy()
        v[ecc], v[omega] = e, w
        if b is not None:
            v[impact] = b
        special.append(v)
    vectors = np.r_[samples, prior_values, special]
    record = dict(
        case=str(case),
        prepared_sha256={k: array_digest(prepared[k]) for k in prepared.files},
        posterior_sha256=array_digest(posterior["unnamed"]),
        posterior_shape=list(posterior["unnamed"].shape),
        calls=int(re.findall(r"ncall: (\d+)", log[-4000:])[-1]),
        lnZ=float(stored["lnZ"]),
        review_clock={
            k: review[k] for k in ("period_days", "epoch_bkjd", "duration_days")
        },
        prior_cubes=cubes.tolist(),
        prior_values=np.asarray(prior_values).tolist(),
        likelihood_vectors=vectors.tolist(),
        likelihood_posterior_rows=len(samples),
        juliet_loglike=[float(box["loglike"](v.copy())) for v in vectors],
    )
    Path(output).write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
