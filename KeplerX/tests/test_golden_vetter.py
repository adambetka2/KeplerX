"""Golden: the compiled Vetter against upstream native Vetter on the nine stored signals of a curve."""

import json

import numpy as np

from common import read
from detector.upstream_adapter import make_dump
from vetter.accelerated_vetter import FastVetter
from vetter.metrics import measure
from vetter.upstream.vetting import Vetter


def failing_curve(fixtures):
    record = read(fixtures / "vetting-failure.json")
    with np.load(fixtures / "vetting-failure.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    candidates = record["detections"]["candidates"]
    tces = np.array(
        [[0, c["period"], c["mes"], c["t0"], c["tdur"]] for c in candidates],
        dtype=float,
    )
    return arrays, record["packet"], tces


def dump(fixtures, config, reference):
    arrays, metadata, _ = failing_curve(fixtures)
    return make_dump(
        arrays, metadata, config["detection"], "vetter", reference=reference
    )


def vet_both(fixtures, config):
    """Upstream's Vetter and the compiled one, both on upstream's whitening."""
    tces = failing_curve(fixtures)[2]
    native = Vetter(dump(fixtures, config, "native"), tces)
    # Native comparisons keep upstream's sigma**2 refinement residual.
    compiled = FastVetter(
        dump(fixtures, config, "compiled"), tces, refinement_weight="sigma_squared"
    )
    results = [
        (measure(native, i, True), measure(compiled, i, True)) for i in range(len(tces))
    ]
    assert len(results) == 9
    return results


def components(errors):
    return [error["component"] for error in errors]


def test_compiled_vetter_is_native_bit_for_bit_with_batmans_rule(
    fixtures, config, monkeypatch
):
    from vetter import transit_model

    monkeypatch.setattr(transit_model, "occulted", transit_model.batman_rule)
    for (expected, expected_errors), (got, got_errors) in vet_both(fixtures, config):
        assert json.dumps(got, sort_keys=True, default=str) == json.dumps(
            expected, sort_keys=True, default=str
        )
        assert components(got_errors) == components(expected_errors)


def test_the_limb_darkening_integral_keeps_every_outcome(fixtures, config):
    """These weak short-period fits are multimodal: values move, outcomes stay."""
    for (expected, expected_errors), (got, got_errors) in vet_both(fixtures, config):
        assert set(got) == set(expected)
        assert components(got_errors) == components(expected_errors)
        for name in ("max_mes", "num_transits"):
            assert got[name] == expected[name]


def test_fft_whitening_keeps_every_outcome(fixtures, config):
    """The compiled Vetter on upstream's whitening."""
    tces = failing_curve(fixtures)[2]
    vetters = [
        FastVetter(dump(fixtures, config, reference), tces)
        for reference in ("compiled", None)
    ]
    for i in range(len(tces)):
        (expected, expected_errors), (got, got_errors) = (
            measure(vetter, i, True) for vetter in vetters
        )
        assert set(got) == set(expected)
        assert components(got_errors) == components(expected_errors)
        assert got["num_transits"] == expected["num_transits"]
        assert np.isclose(got["max_mes"], expected["max_mes"], rtol=1e-9, atol=0)
