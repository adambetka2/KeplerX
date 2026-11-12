"""The fused likelihoods must equal pinned upstream bit for bit."""

import numpy as np
import pytest
from triceratops import likelihoods as upstream

from fast_triceratops import accelerated, kernels
from fast_triceratops.gold import FIXTURE, case_probs

G, MSUN, RSUN = upstream.G, upstream.Msun, upstream.Rsun


def same(a, b):
    """Bitwise equality, with any NaN equal to any NaN."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.all((a.view(np.int64) == b.view(np.int64))
                                              | (np.isnan(a) & np.isnan(b))))


def rows(rng, n, binary):
    """Transiting rows across period, size, impact up to grazing, eccentricity and dilution,
    with special radius ratios (0.5 and pytransit's snap edges, 1, beyond 1) and an
    invalid row."""
    period = 10 ** rng.uniform(-0.5, 1.5, n)
    R_s = rng.uniform(0.3, 2.5, n)
    a = (G * rng.uniform(0.2, 2.0, n) * MSUN / (4 * np.pi**2) * (period * 86400) ** 2) ** (1 / 3)
    ecc = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0, 0.9, n))
    ratio = rng.uniform(0.05, 1.6, n) if binary else 10 ** rng.uniform(-2.3, -0.5, n)
    ratio[4:12] = [0.5, 0.50005, 0.49995, 0.500099, 0.9, 1.0, 1.2, 2.0]
    impact = rng.uniform(0, 1 + ratio)
    impact[12:16] = [0.0, 1.0, 1 + ratio[14], 1 - ratio[15]]
    inc = np.degrees(np.arccos(np.clip(impact * R_s * RSUN / a, 0, 1)))
    size = ratio * R_s * (1 if binary else RSUN / upstream.Rearth)
    cfr = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0, 0.9, n))
    u1, u2 = rng.uniform(0.1, 0.7, n), rng.uniform(0.0, 0.4, n)
    row = {"size": size, "P_orb": period, "inc": inc, "a": a, "R_s": R_s, "u1": u1, "u2": u2,
           "ecc": ecc, "argp": rng.uniform(0, 360, n), "companion_fluxratio": cfr}
    if binary:
        row["EB_fluxratio"] = rng.uniform(0, 0.6, n)
    row["size"][0] = np.nan  # an invalid row
    row["u1"][3] = np.nan
    return row


def call(function, time, flux, sigma, row, host, exptime, nsamples):
    r = {name: np.copy(value) for name, value in row.items()}
    head = [r["size"]] + ([r["EB_fluxratio"]] if "EB_fluxratio" in r else [])
    result = function(time, flux, sigma, *head, r["P_orb"], r["inc"], r["a"], r["R_s"],
                      r["u1"], r["u2"], r["ecc"], r["argp"],
                      companion_fluxratio=r["companion_fluxratio"], companion_is_host=host,
                      exptime=exptime, nsamples=nsamples)
    return result, r["inc"]  # upstream converts inc to radians in place


@pytest.mark.parametrize("name", ["lnL_TP_p", "lnL_EB_p", "lnL_EB_twin_p"])
@pytest.mark.parametrize("npt,nsamples,exptime", [(37, 1, 0.0), (200, 20, 0.020433598216918748),
                                                  (478, 80, 0.2), (9000, 3, 1e-6)])
@pytest.mark.parametrize("host", [False, True])
def test_random_rows_bitwise(name, npt, nsamples, exptime, host):
    rng = np.random.default_rng([npt, nsamples, host, len(name)])
    n = 300 if npt < 9000 else 40
    time = np.sort(rng.uniform(-0.4, 0.4, npt))
    flux = 1 - 3e-3 * (np.abs(time) < 0.05) + rng.normal(0, 1e-3, npt)
    row = rows(rng, n, binary="EB" in name)
    expected, inc_expected = call(getattr(upstream, name), time, flux, 1e-3, row, host,
                                  exptime, nsamples)
    result, inc_result = call(getattr(kernels, name), time, flux, 1e-3, row, host,
                              exptime, nsamples)
    assert same(result, expected)
    assert same(inc_result, inc_expected)
    flat = 0.5 * np.sum((flux - 1) ** 2 / 1e-3**2)
    transiting = np.isfinite(expected) & (expected != flat)
    assert transiting.sum() > (0 if name == "lnL_EB_p" else n // 2)  # transits touch the data


def test_single_row_delegates_to_upstream():
    rng = np.random.default_rng(7)
    time, flux = np.linspace(-0.2, 0.2, 50), np.ones(50)
    tp = {key: value[4:5] for key, value in rows(rng, 20, False).items()}
    expected = call(upstream.lnL_TP_p, time, flux, 1e-3, tp, False, 0.02, 5)[0]
    assert same(call(kernels.lnL_TP_p, time, flux, 1e-3, tp, False, 0.02, 5)[0], expected)
    eb = {key: value[4:5] for key, value in rows(rng, 20, True).items()}
    for function in (upstream.lnL_EB_twin_p, kernels.lnL_EB_twin_p):  # upstream's squeeze fails
        with pytest.raises(np.exceptions.AxisError):
            call(function, time, flux, 1e-3, eb, False, 0.02, 5)


def test_nonstandard_inputs_go_to_upstream():
    rng = np.random.default_rng(11)
    time = np.linspace(-0.2, 0.2, 60)
    row = rows(rng, 20, False)
    for change in ("short flux", "scalar flux", "float32 inc", "scalar dilution", "array sigma"):
        r = {key: np.copy(value) for key, value in row.items()}
        flux, sigma = np.ones(60), 1e-3
        if change == "short flux":
            flux = np.ones(30)
        elif change == "scalar flux":
            flux = 1.0
        elif change == "float32 inc":
            r["inc"] = r["inc"].astype(np.float32)
        elif change == "scalar dilution":
            r["companion_fluxratio"] = 0.0
        else:
            sigma = np.full(60, 1e-3)
        outcomes = []
        for function in (upstream.lnL_TP_p, kernels.lnL_TP_p):
            try:
                outcomes.append(call(function, time, flux, sigma, r, False, 0.02, 5)[0])
            except Exception as error:  # noqa: BLE001 (both must fail the same way)
                outcomes.append(type(error))
        if isinstance(outcomes[0], type):
            assert outcomes[1] is outcomes[0], change
        else:
            assert same(outcomes[1], outcomes[0]), change


def test_accelerated_restores_upstream():
    from triceratops import likelihoods, marginal_likelihoods, triceratops

    with accelerated():
        assert marginal_likelihoods.lnL_TP_p is kernels.lnL_TP_p
        assert triceratops.lnZ_STP is not marginal_likelihoods.lnZ_STP
    assert marginal_likelihoods.lnL_TP_p is likelihoods.lnL_TP_p
    assert triceratops.lnZ_STP is marginal_likelihoods.lnZ_STP
    assert "_fast_ldc_lookup" not in vars(marginal_likelihoods)


def test_kepler10_gold_bitwise():
    with accelerated():
        fast = case_probs(32768)
    with np.load(FIXTURE / "expected.npz") as gold:
        assert same(fast.lnZ, gold["reference_lnZ"])


@pytest.mark.parametrize("case", ["kepler10", "typical"])
def test_calc_probs_bitwise(case):
    stock = case_probs(4096, seed=7, case=case)
    with accelerated():
        fast = case_probs(4096, seed=7, case=case)
    assert same(fast.lnZ, stock.lnZ)
    assert fast.probs.equals(stock.probs)
    assert (fast.FPP, fast.NFPP) == (stock.FPP, stock.NFPP)
    assert same(fast.u1, stock.u1) and same(fast.fluxratio_comp, stock.fluxratio_comp)


def test_ldc_lookup_equals_upstream_loop():
    from triceratops import marginal_likelihoods as ml

    from fast_triceratops.scenarios import ldc_lookup, repaired

    table = ml.ldc_K[ml.ldc_K_Zs == ml.ldc_K_Zs[np.abs(ml.ldc_K_Zs - 0.0).argmin()]]
    columns = (np.array(table.Teff, dtype=int), np.array(table.logg, dtype=float),
               np.array(table.a, dtype=float), np.array(table.b, dtype=float))
    rng = np.random.default_rng(3)
    teffs = rng.choice(np.arange(3500, 10001, 250), 5000).astype(float)
    loggs = rng.choice(np.arange(3.5, 5.01, 0.5), 5000)
    u1, u2 = np.zeros(5000), np.zeros(5000)
    for i, (teff, logg) in enumerate(zip(teffs, loggs)):  # upstream's loop
        mask = (columns[0] == teff) & (columns[1] == logg)
        u1[i], u2[i] = columns[2][mask].item(), columns[3][mask].item()
    fast = ldc_lookup(teffs, loggs, *columns)
    assert same(fast[0], u1) and same(fast[1], u2)
    with pytest.raises(ValueError):
        ldc_lookup(np.array([3600.0]), np.array([4.5]), *columns)
    assert set(repaired()) == {"lnZ_STP", "lnZ_SEB"}


def test_background_ldc_equals_upstream_loop():
    """lnZ_BTP/lnZ_BEB's per-star loop against its vectorized form, with exact ties
    halfway between table values and NaN stars."""
    from triceratops import marginal_likelihoods as ml

    from fast_triceratops.scenarios import background_ldc

    table = (ml.ldc_K_Teffs, ml.ldc_K_loggs, ml.ldc_K_Zs, ml.ldc_K_u1s, ml.ldc_K_u2s)
    rng = np.random.default_rng(8)
    n = 3000
    teffs = rng.uniform(3600, 9800, n)
    teffs[:200] = rng.choice(np.unique(table[0])[:-1], 200) + 125.0  # halfway on a 250 K grid
    loggs = rng.uniform(3.6, 4.9, n)
    loggs[200:400] = rng.choice(np.unique(table[1])[:-1], 200) + 0.25
    zs = rng.uniform(-1.0, 0.4, n)
    zs[400:600] = np.round(zs[400:600] * 4) / 4 + 0.125
    teffs[600], loggs[601], zs[602] = np.nan, np.nan, np.nan
    u1, u2 = np.zeros(n), np.zeros(n)
    for i in range(n):  # upstream's loop (marginal_likelihoods.py, lnZ_BTP)
        this_Teff = table[0][np.argmin(np.abs(table[0] - teffs[i]))]
        this_logg = table[1][np.argmin(np.abs(table[1] - loggs[i]))]
        these_Zs = table[2][(table[0] == this_Teff) & (table[1] == this_logg)]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs - zs[i]))]
        mask = (table[2] == this_Z) & (table[0] == this_Teff) & (table[1] == this_logg)
        u1[i], u2[i] = table[3][mask].item(), table[4][mask].item()
    fast = background_ldc(teffs, loggs, zs, *table)
    assert same(fast[0], u1) and same(fast[1], u2)
