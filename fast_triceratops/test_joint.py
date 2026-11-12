"""The fused joint likelihood must equal post_boi.quarters.SourceLikelihood bit for bit."""

import numpy as np
import pytest
from triceratops import likelihoods as upstream

from fast_triceratops import joint
from fast_triceratops.test_equivalence import rows, same
from post_boi.quarters import QuarterLikelihood, SourceLikelihood

EXPTIME = 0.020433598216918748
NAMES = ["lnL_TP_p", "lnL_EB_p", "lnL_EB_twin_p"]
stock = SourceLikelihood.__call__


def source(rng, npt):
    """Sufficient statistics as for_source would give them, with some empty bins."""
    time = np.sort(rng.uniform(-0.4, 0.4, npt))
    precision = np.where(rng.random(npt) < 0.1, 0.0, 10 ** rng.uniform(5, 8, npt))
    centre = np.where(precision > 0, rng.normal(0, 1e-3, npt), 0.0)
    return SourceLikelihood(time, centre, precision, float(rng.uniform(100, 1000)),
                            float(rng.normal()))


def call(function, source, name, time, sigma, row, host, nsamples):
    r = {key: np.copy(value) for key, value in row.items()}
    head = [r["size"]] + ([r["EB_fluxratio"]] if "EB_fluxratio" in r else [])
    result = function(source, getattr(upstream, name), time, np.ones_like(time), sigma, *head,
                      r["P_orb"], r["inc"], r["a"], r["R_s"], r["u1"], r["u2"], r["ecc"],
                      r["argp"], companion_fluxratio=r["companion_fluxratio"],
                      companion_is_host=host, exptime=EXPTIME, nsamples=nsamples)
    return result, r["inc"]  # upstream converts inc to radians in place


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("npt,nsamples", [(37, 1), (200, 20), (150, 80)])
@pytest.mark.parametrize("host", [False, True])
def test_random_rows_bitwise(name, npt, nsamples, host):
    rng = np.random.default_rng([npt, nsamples, host, len(name)])
    s = source(rng, npt)
    row = rows(rng, 300, binary="EB" in name)
    expected, inc_expected = call(stock, s, name, s.time, 1e-3, row, host, nsamples)
    result, inc_result = call(joint.call, s, name, s.time, 1e-3, row, host, nsamples)
    assert same(result, expected)
    assert same(inc_result, inc_expected)
    assert np.isfinite(expected).sum() > (10 if name == "lnL_EB_p" else 150)  # rows transit
    if name == "lnL_EB_p":
        assert np.isinf(expected).sum() > 1  # the secondary veto fires beyond the NaN-ldc row


@pytest.mark.parametrize("name", NAMES)
def test_quarter_likelihood_sources_bitwise(name):
    """Real for_source statistics, including bins where one source has no response."""
    rng = np.random.default_rng(len(name))
    time = np.linspace(-0.12, 0.12, 31)
    fractions = np.array([[0.8, 0.2], [0.2, 0.8], [1.0, 0.0]])
    bins = np.concatenate([np.arange(5, 31), np.arange(5, 31), np.arange(31)])
    quarter = np.concatenate([np.full(26, 3), np.full(26, 4), np.full(31, 5)])
    response = fractions[quarter - 3]
    sigma = rng.uniform(0.00008, 0.0002, len(bins))
    shape = np.maximum(0, 1 - abs(time) / 0.08)
    flux = 1 - response[:, 0] * 0.001 * shape[bins] + sigma * rng.normal(size=len(bins))
    packet = {"time": time, "flux": flux, "sigma": sigma, "bin_index": bins, "quarter": quarter,
              "applied_fraction": np.ones(len(bins))}
    quarters = QuarterLikelihood(packet, np.array(["100", "200"]), [3, 4, 5], fractions)
    row = rows(rng, 120, binary="EB" in name)
    for source_id in quarters.source_ids:
        s = quarters.for_source(source_id)
        assert (s.precision == 0).any() == (source_id == "200")
        expected = call(stock, s, name, time, 1.2e-4, row, False, 20)[0]
        result = call(joint.call, s, name, time, 1.2e-4, row, False, 20)[0]
        assert same(result, expected)
        assert np.isfinite(expected).sum() > (3 if name == "lnL_EB_p" else 12)


@pytest.mark.parametrize("name", NAMES)
def test_degenerate_shapes_delegate_to_upstream(name):
    rng = np.random.default_rng(11)
    s = source(rng, 50)
    row = {key: value[4:5] for key, value in rows(rng, 20, "EB" in name).items()}
    if name == "lnL_TP_p":  # one TP row broadcasts back to (1, npt) through the dilution
        expected = call(stock, s, name, s.time, 1e-3, row, False, 5)[0]
        assert same(call(joint.call, s, name, s.time, 1e-3, row, False, 5)[0], expected)
    else:  # upstream's squeeze breaks one-row EBs: np.min(axis=1) of a 1-D secondary
        for function in (stock, joint.call):
            with pytest.raises(np.exceptions.AxisError):
                call(function, s, name, s.time, 1e-3, row, False, 5)
    one = source(rng, 1)
    row = {key: value[:5] for key, value in rows(rng, 20, "EB" in name).items()}
    for function in (stock, joint.call):
        with pytest.raises(ValueError, match="Model phase grid differs"):
            call(function, one, name, one.time, 1e-3, row, False, 5)


def test_other_functions_and_checks_delegate_to_upstream():
    rng = np.random.default_rng(5)
    s = source(rng, 40)
    row = {key: value[4:5] for key, value in rows(rng, 20, False).items()}
    scalar = {key: float(value[0]) for key, value in row.items()}
    args = (scalar["size"], scalar["P_orb"], scalar["inc"], scalar["a"], scalar["R_s"],
            scalar["u1"], scalar["u2"], scalar["ecc"], scalar["argp"])
    expected = stock(s, upstream.lnL_TP, s.time, np.ones(40), 1e-3, *args,
                     companion_fluxratio=scalar["companion_fluxratio"], exptime=EXPTIME, nsamples=4)
    result = joint.call(s, upstream.lnL_TP, s.time, np.ones(40), 1e-3, *args,
                        companion_fluxratio=scalar["companion_fluxratio"], exptime=EXPTIME, nsamples=4)
    assert same(result, expected)

    def lnL_other(*args, **kwargs):
        return 0.0

    with pytest.raises(ValueError, match="Unsupported upstream likelihood"):
        joint.call(s, lnL_other, s.time, np.ones(40), 1e-3)
    row = rows(rng, 20, False)
    for function in (stock, joint.call):
        with pytest.raises(ValueError, match="Likelihood phase grid differs"):
            call(function, s, "lnL_TP_p", s.time + 1e-9, 1e-3, row, False, 4)


def test_accelerated_joint_swaps_and_restores():
    rng = np.random.default_rng(3)
    s = source(rng, 60)
    row = rows(rng, 40, True)
    expected = call(stock, s, "lnL_EB_p", s.time, 1e-3, row, True, 3)[0]
    with joint.accelerated_joint():
        assert SourceLikelihood.__call__ is joint.call
        assert same(call(SourceLikelihood.__call__, s, "lnL_EB_p", s.time, 1e-3, row, True, 3)[0],
                    expected)
    assert SourceLikelihood.__call__ is stock


def transit_source(rng, npt):
    """A 2,000 ppm transit at 30-100 ppm per bin, so deep rows are hopeless."""
    time = np.sort(rng.uniform(-0.3, 0.3, npt))
    shape = np.clip((0.08 - np.abs(time)) / 0.015, 0, 1)
    precision = np.where(rng.random(npt) < 0.05, 0.0, 10 ** rng.uniform(8, 9, npt))
    centre = np.where(precision > 0, -2e-3 * shape + rng.normal(0, 1e-4, npt), 0.0)
    return SourceLikelihood(time, centre, precision, float(rng.uniform(100, 1000)),
                            float(rng.uniform(1e3, 1e5)))


def priors(rng, n, best):
    """None, flat, random with -inf entries, and one that sinks the best rows so that
    rows deferred against them must be modelled after all."""
    noisy = np.where(rng.random(n) < 0.05, -np.inf, rng.uniform(-30, 0, n))
    sunk = np.zeros(n)
    sunk[best[:5]] = -3000.0
    return {"none": None, "flat": np.zeros(n), "noisy": noisy, "sunk": sunk}


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("limit", [joint.EXACT, joint.CERTIFIED])
def test_pruned_evidence_and_best_draw(name, limit):
    from triceratops._numerics import _log_mean_exp

    from fast_triceratops.scenarios import top

    rng = np.random.default_rng([len(name), int(limit)])
    s = transit_source(rng, 200)
    row = rows(rng, 3000, binary="EB" in name)
    sigma = 1e-2 if name == "lnL_EB_p" else 1e-4  # a loose veto leaves primaries to prune
    expected = call(stock, s, name, s.time, sigma, row, False, 20)[0]
    lnL_stock = s.log_normalization - expected
    deferred = 0
    for kind, prior in priors(rng, expected.size, np.argsort(expected)).items():
        with joint.pruning(limit):
            lnL = s.log_normalization - call(joint.call, s, name, s.time, sigma, row, False, 20)[0]
            deferred += np.count_nonzero(np.isfinite(lnL) & (lnL <= -joint.SENTINEL))
            got = joint.fast_log_mean_exp(lnL, prior, N_total=lnL.size)
            assert not joint.DEFERRED
        want = _log_mean_exp(lnL_stock if prior is None else lnL_stock + prior,
                             N_total=lnL.size)
        if limit == joint.EXACT:
            assert same(got, want), kind
        else:
            assert abs(got - want) <= 1e-13 * abs(want), kind
        assert top(-lnL, 100)[0] == (-lnL_stock).argsort()[0], kind
    assert deferred > (200 if name == "lnL_EB_p" else 1000)  # most rows were never modelled


@pytest.mark.parametrize("name", NAMES)
def test_tied_best_rows_are_all_modelled(name):
    rng = np.random.default_rng(len(name) + 1)
    s = transit_source(rng, 120)
    row = {key: np.tile(value, 2) for key, value in rows(rng, 800, "EB" in name).items()}
    expected = call(stock, s, name, s.time, 1e-4, row, False, 20)[0]
    with joint.pruning():
        assert same(call(joint.call, s, name, s.time, 1e-4, row, False, 20)[0], expected)


def test_top_equals_argsort_head():
    from fast_triceratops.scenarios import top

    rng = np.random.default_rng(9)
    for trial in range(200):
        n = int(rng.integers(1, 3000))
        keys = rng.normal(size=n)
        if trial % 4 == 1:
            keys = np.round(keys, 1)  # ties among the best
        if trial % 4 == 2:
            keys[rng.random(n) < 0.97] = np.inf  # few finite draws
        if trial % 4 == 3:
            keys[rng.random(n) < 0.1] = np.nan
        assert np.array_equal(top(keys, 100), keys.argsort()[:100])


def test_finite_proof_holds_at_pytransit_edges():
    """Wherever kernels.unsafe declares a point model safe, pytransit's own compiled
    eval_quad_z_s is finite, including exactly at z = k, the 1e-6 shift edge, the
    edge-at-origin boundary and z = 0.5 beside the snap; non-finite values occur only
    inside the declared bands."""
    from pytransit.models.numba.ma_quadratic_nb import eval_quad_z_s

    from fast_triceratops.kernels import unsafe

    rng = np.random.default_rng(4)
    n = 4000
    offsets = np.concatenate([[0.0], np.outer([1, -1], 10.0 ** -np.arange(3, 18)).ravel(),
                              np.outer([1, -1], [1.5e-7, 3e-7, 1e-6]).ravel()])
    nonfinite = 0
    for edge in ("1+k", "1-k", "k", "shift", "origin", "half", "0"):
        k = np.concatenate([10 ** rng.uniform(-3, -0.31, n // 2), rng.uniform(0.5, 1 - 1e-6, n // 2)])
        if edge == "half":
            k = 0.5 + rng.choice([-1, 1], n) * rng.uniform(2e-4, 3e-4, n)
        k = k[np.abs(k - 0.5) >= 2e-4]
        base = {"1+k": 1 + k, "1-k": 1 - k, "k": k, "shift": k - 1e-6,
                "origin": k + np.maximum(1e-6, 2e-4 * k), "half": np.full_like(k, 0.5),
                "0": np.zeros_like(k)}[edge]
        z = np.abs(base + rng.choice(offsets, k.size) * np.maximum(base, 1e-3))
        u = np.stack([rng.uniform(0.0, 0.9, k.size), rng.uniform(-0.2, 0.6, k.size)], axis=1)
        for zi, ki, ui in zip(z, k, u):
            if 1 - ui[0] / 3 - ui[1] / 6 < 0.05:
                continue
            try:
                finite = np.isfinite(eval_quad_z_s(zi, ki, ui))
            except ZeroDivisionError:  # numba's python error model
                finite = False
            assert finite or unsafe(zi, zi, ki), (edge, zi, ki)
            nonfinite += not finite
    assert nonfinite > 100  # the bands are real


def test_finite_rows_never_wrong_on_mixed_rows():
    from fast_triceratops import kernels

    rng = np.random.default_rng(12)
    proven = 0
    for binary, npt, ns in [(False, 200, 20), (True, 200, 80), (False, 60, 1), (True, 137, 40)]:
        time = np.sort(rng.uniform(-0.4, 0.4, npt))
        r = rows(rng, 3000, binary)
        args = [r["size"]] + ([r["EB_fluxratio"]] if binary else []) + [
            r[key] for key in ("P_orb", "inc", "a", "R_s", "u1", "u2", "ecc", "argp")]
        prep = (kernels.prepare_eb if binary else kernels.prepare_tp)(
            *args, r["companion_fluxratio"], exptime=EXPTIME, nsamples=ns)
        every = np.arange(prep["k"].size)
        proof = kernels.finite_rows(prep, every, time)
        pvp = np.array([prep["k"], np.zeros_like(prep["k"]), prep["P_orb"], prep["a"],
                        prep["inc"], prep["ecc"], prep["w"]]).T
        kernels.tm.set_data(time, exptimes=EXPTIME, nsamples=ns)
        model = kernels.tm.evaluate_pv(pvp=pvp, ldc=np.array([prep["u1"], prep["u2"]]).T)
        assert np.isfinite(model[proof]).all()
        proven += proof.sum()
    assert proven > 6000


def v3_runtime():
    from triceratops import marginal_likelihoods

    from fast_triceratops.kernels import require_pinned
    from fast_triceratops.scenarios import PINNED_V3

    try:
        require_pinned(marginal_likelihoods, PINNED_V3)
    except RuntimeError:
        return False
    return True


def kepler10_v3():
    """The gold Kepler-10 photons as a V3 joint likelihood over three quarters, with a
    declared response for one neighbour so its scenarios run too."""
    from fast_triceratops.gold import FIXTURE, kepler10_target

    target = kepler10_target()
    raw = np.loadtxt(FIXTURE / "lightcurve.csv", delimiter=",")
    labels = np.minimum(np.arange(len(raw)) * 200 // len(raw), 199)
    weight = 1 / raw[:, 2] ** 2
    total = np.bincount(labels, weights=weight)
    time = np.bincount(labels, weights=weight * raw[:, 0]) / total
    binned = np.bincount(labels, weights=weight * raw[:, 1]) / total
    ids = np.array(target.stars["ID"].values[:2], dtype=str)
    fractions = np.array([[0.9, 0.1], [0.7, 0.3], [0.95, 0.05]])
    packet = {"time": time, "flux": raw[:, 1], "sigma": raw[:, 2], "bin_index": labels,
              "quarter": 3 + np.arange(len(raw)) % 3, "applied_fraction": np.ones(len(raw))}
    quarters = QuarterLikelihood(packet, ids, [3, 4, 5], fractions)
    target.stars["fluxratio"] = 0.0
    target.stars.loc[:1, "fluxratio"] = fractions.mean(axis=0)
    target.stars["tdepth"] = 0.0
    target.stars.loc[:1, "tdepth"] = 0.00019 / fractions.max(axis=0)
    return target, quarters, time, binned, float(np.median(total**-0.5))


@pytest.mark.skipif(not v3_runtime(), reason="needs the V3-patched TRICERATOPS runtime")
@pytest.mark.parametrize("certified", [False, True])
def test_v3_calc_probs_bitwise(certified):
    from fast_triceratops import accelerated_v3
    from fast_triceratops.gold import EXPTIME, P_ORB

    def run(draws):
        target, quarters, time, binned, sigma = kepler10_v3()
        np.random.seed(20261025)
        target.calc_probs(time=time, flux_0=binned, flux_err_0=sigma, P_orb=P_ORB, N=draws,
                          parallel=True, verbose=0, flatpriors=False, exptime=EXPTIME,
                          nsamples=20, contrast_curve_file=None, filt=None, molusc_file=None,
                          dilution_corrected=False, joint=quarters)
        return target

    stock = run(8192)
    with accelerated_v3(certified=certified):
        fast = run(8192)
    assert len(stock.probs) == 18  # both stars' scenarios ran
    assert same(fast.lnZ, stock.lnZ)
    assert fast.probs.equals(stock.probs)
    assert (fast.FPP, fast.NFPP) == (stock.FPP, stock.NFPP)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("npt,nsamples", [(200, 20), (150, 80), (61, 1)])
def test_cheap_bound_never_exceeds_the_cost(name, npt, nsamples):
    """Certified mode defers on kernels.cheap_row; it must stay below upstream's cost on
    mixed rows (special radius ratios, grazing, eccentric, diluted, invalid)."""
    from fast_triceratops import kernels

    rng = np.random.default_rng([npt, nsamples, len(name)])
    s = transit_source(rng, npt)
    row = rows(rng, 4000, binary="EB" in name)
    host = bool(rng.random() < 0.5)
    expected = call(stock, s, name, s.time, 1e-2, row, host, nsamples)[0]
    r = {key: np.copy(value) for key, value in row.items()}
    head = [r["size"]] + ([r["EB_fluxratio"]] if "EB_fluxratio" in r else [])
    prepared = kernels.PREPARE[name](*head, r["P_orb"], r["inc"], r["a"], r["R_s"], r["u1"],
                                     r["u2"], r["ecc"], r["argp"],
                                     companion_fluxratio=r["companion_fluxratio"],
                                     companion_is_host=host, exptime=EXPTIME, nsamples=nsamples)
    every = np.arange(expected.size)
    bound, guess = kernels.cheap_bounds(prepared, every, s.time,
                                        kernels.cost_inputs(np.ones(npt), 1e-2, s))
    finite = np.isfinite(expected) & np.isfinite(bound)
    assert finite.sum() > (100 if name == "lnL_EB_p" else 1000)  # the veto removes most EBs
    assert (bound[finite] <= expected[finite]).all()
    assert (guess[finite] >= bound[finite]).all()
