"""Exact serial-side repairs of pinned upstream scenario functions.

lnZ_STP and lnZ_SEB look up companion limb darkening with a Python loop over
all N draws (marginal_likelihoods.py:994-999 and :1222-1227). The repair
rebuilds each function from its hash-pinned source with only that loop
replaced by a lookup per unique (Teff, logg) pair. The function keeps
upstream's globals, so its draws, priors and likelihood calls are unchanged.

Under the V3 patch (post_boi/patches), repaired_v3() rebuilds all 14 scenario
functions the same way and routes each evidence through
joint.fast_log_mean_exp, which resolves the rows joint.pruned deferred. Each
scenario's lnL is otherwise only assigned and argsorted, so nothing else sees
a deferred row. The full argsort that keeps the best 100 draws becomes top(), and the
per-background-star limb-darkening loops of lnZ_BTP and lnZ_BEB become background_ldc().
"""

import functools
import inspect
import textwrap

import numpy as np

from fast_triceratops.kernels import require_pinned

PINNED = {
    "lnZ_STP": "52d14d58f2cbbda35face2bd597d7e502032dd6c17841eb526c4f3e9bead7083",
    "lnZ_SEB": "5b188ffe018fc572df8da5525a745097a03462bac3b7e7106ab457e7adb1080a",
}
PINNED_V3 = {  # TRICERATOPS_KEPLER_V3: d20875c with post_boi/patches/triceratops-quarters.patch
    "lnZ_BEB": "7c20e213520de2b6cd01e508f9278effe558c160702889380b74627f0f3b3f22",
    "lnZ_BTP": "192ad0bf3791bb41952e9015231b945129c503233326393adc224a8311151b0b",
    "lnZ_DEB": "3b1e9b6efb2617d74f71c70bea7671320d4d929c58cecb8c7eb623b1ce52df4e",
    "lnZ_DTP": "01012ebaaed85d01dc65997bfbd9430225d6a78b1498227d7c021f5ee917a71b",
    "lnZ_NEB_evolved": "cb3af01116d8b6b551e290d0f033920a8cd35cad1ea4d532349a22aac6906ee3",
    "lnZ_NEB_unknown": "186df56f5fe1d2a34acf0379da9531904df90fb7ec952943d351cd84fde0746d",
    "lnZ_NTP_evolved": "8550381db08deba7ea3fa82968d22ce3f2b89d111d2c942845edd395c71d3640",
    "lnZ_NTP_unknown": "88a385796cb1bb27435ea42eadee36132f7de51a20173beda786f7557aad952a",
    "lnZ_PEB": "fc7640ff0f97aca08610f4a612d95aa22b874c95dfad1552909262a21e5eebb7",
    "lnZ_PTP": "88c2dd2940e2146f6a5b6ad57a4bb436e168bd626b49ad8a50deca0b1386e270",
    "lnZ_SEB": "a1f9e42e62fe4eaf08deae0cbf184deade2ba78aa750e0a8ede4ec6c8b4c05fa",
    "lnZ_STP": "01266d0e66ddce394703e2500c12d6754976d411ef62005b4b3622916d566771",
    "lnZ_TEB": "9843abeaea7ad3f93c5c647fa1b01fc3e7894f7f8011cff067d0d81b11996612",
    "lnZ_TTP": "de4ab94be729be31c30e36bf74bf577ea65faec7b287cfc67d10447dc8478b47",
}
EVIDENCE = {
    "_log_mean_exp(lnL, N_total=N)": "_fast_log_mean_exp(lnL, None, N_total=N)",
    "_log_mean_exp(lnL_twin, N_total=N)": "_fast_log_mean_exp(lnL_twin, None, N_total=N)",
    "_log_mean_exp(lnL + lnprior_companion, N_total=N)":
        "_fast_log_mean_exp(lnL, lnprior_companion, N_total=N)",
    "_log_mean_exp(lnL_twin + lnprior_companion, N_total=N)":
        "_fast_log_mean_exp(lnL_twin, lnprior_companion, N_total=N)",
}
BACKGROUND = {  # the per-background-star limb-darkening loop, as written in each function
    "lnZ_BTP": """    u1s_comp, u2s_comp = np.zeros(N_comp), np.zeros(N_comp)
    for i in range(N_comp):
        this_Teff = ldc_Teffs[np.argmin(np.abs(ldc_Teffs-Teffs_comp[i]))]
        this_logg = ldc_loggs[np.argmin(np.abs(ldc_loggs-loggs_comp[i]))]
""",
    "lnZ_BEB": """    u1s_comp, u2s_comp = np.zeros(N_comp), np.zeros(N_comp)
    for i in range(N_comp):
        this_Teff = ldc_Teffs[np.argmin(
            np.abs(ldc_Teffs-Teffs_comp[i])
            )]
        this_logg = ldc_loggs[np.argmin(
            np.abs(ldc_loggs-loggs_comp[i])
            )]
""",
}
BACKGROUND_TAIL = """        mask1 = (ldc_Teffs == this_Teff) & (ldc_loggs == this_logg)
        these_Zs = ldc_Zs[mask1]
        this_Z = these_Zs[np.argmin(np.abs(these_Zs-Zs_comp[i]))]
        mask = (
            (ldc_Zs == this_Z)
            & (ldc_Teffs == this_Teff)
            & (ldc_loggs == this_logg)
            )
        u1s_comp[i], u2s_comp[i] = ldc_u1s[mask].item(), ldc_u2s[mask].item()
"""
BACKGROUND_LOOKUP = """    u1s_comp, u2s_comp = _fast_background_ldc(
        Teffs_comp, loggs_comp, Zs_comp, ldc_Teffs, ldc_loggs, ldc_Zs, ldc_u1s, ldc_u2s)
"""
RANKING = {
    "(-lnL).argsort()[:N_samples]": "_fast_top(-lnL, N_samples)",
    "(-lnL_twin).argsort()[:N_samples]": "_fast_top(-lnL_twin, N_samples)",
}
LOOP = """    u1s_comp, u2s_comp = np.zeros(N), np.zeros(N)
    for i, (comp_Teff, comp_logg) in enumerate(
            zip(rounded_Teffs_comp, rounded_loggs_comp)
            ):
        mask = (Teffs_at_Z == comp_Teff) & (loggs_at_Z == comp_logg)
        u1s_comp[i], u2s_comp[i] = u1s_at_Z[mask].item(), u2s_at_Z[mask].item()
"""
LOOKUP = """    u1s_comp, u2s_comp = _fast_ldc_lookup(
        rounded_Teffs_comp, rounded_loggs_comp,
        Teffs_at_Z, loggs_at_Z, u1s_at_Z, u2s_at_Z)
"""


def ldc_lookup(teffs, loggs, table_teffs, table_loggs, table_u1, table_u2):
    """The upstream loop, once per unique (Teff, logg) pair; .item() still demands exactly
    one table row per pair."""
    unique_teffs, teff_codes = np.unique(teffs, return_inverse=True)
    unique_loggs, logg_codes = np.unique(loggs, return_inverse=True)
    pairs, inverse = np.unique(teff_codes * unique_loggs.size + logg_codes, return_inverse=True)
    u1, u2 = np.empty(pairs.size), np.empty(pairs.size)
    for j, code in enumerate(pairs):
        teff, logg = unique_teffs[code // unique_loggs.size], unique_loggs[code % unique_loggs.size]
        mask = (table_teffs == teff) & (table_loggs == logg)
        u1[j], u2[j] = table_u1[mask].item(), table_u2[mask].item()
    return u1[inverse], u2[inverse]


def nearest(table, values):
    """table[np.argmin(np.abs(table - v))] for each v: the nearest table value, a tie going to
    the one that occurs first in table and a NaN v to table[0], as np.argmin does."""
    unique, first = np.unique(table, return_index=True)
    distance = np.abs(unique[None, :] - values[:, None])
    tied = distance == distance.min(axis=1, keepdims=True)
    out = unique[np.where(tied, first[None, :], table.size).argmin(axis=1)]
    out[np.isnan(values)] = table[0]
    return out


def background_ldc(teffs, loggs, zs, table_teffs, table_loggs, table_zs, table_u1, table_u2):
    """The per-background-star loop of lnZ_BTP and lnZ_BEB, once per unique table entry;
    .item() still demands exactly one table row."""
    u1, u2 = np.zeros(teffs.size), np.zeros(teffs.size)
    teffs, loggs = nearest(table_teffs, teffs), nearest(table_loggs, loggs)
    for teff, logg in set(zip(teffs.tolist(), loggs.tolist())):
        stars = (teffs == teff) & (loggs == logg)
        nearest_z = nearest(table_zs[(table_teffs == teff) & (table_loggs == logg)], zs[stars])
        for z in np.unique(nearest_z):
            mask = (table_zs == z) & (table_teffs == teff) & (table_loggs == logg)
            pick = np.flatnonzero(stars)[nearest_z == z]
            u1[pick], u2[pick] = table_u1[mask].item(), table_u2[mask].item()
    return u1, u2


def top(keys, n):
    """keys.argsort()[:n] without sorting everything: when the n + 1 smallest keys are
    finite and distinct, their order fixes the first n indices for any sort algorithm."""
    if keys.size <= n + 1:
        return keys.argsort()[:n]
    smallest = np.argpartition(keys, n)[:n + 1]
    order = smallest[np.argsort(keys[smallest], kind="stable")]
    ranked = keys[order]
    if not (np.isfinite(ranked).all() and (np.diff(ranked) > 0).all()):
        return keys.argsort()[:n]
    return order[:n]


def rebuild(module, name, replacements, bound):
    """module.name compiled from its source with each (old, new) replaced exactly as often
    as declared; a closure binds the names in bound, so the module namespace is untouched."""
    source = inspect.getsource(getattr(module, name))
    for old, new, count in replacements:
        if source.count(old) != count:
            raise RuntimeError(f"{name}: {old!r} not found {count} times")
        source = source.replace(old, new)
    factory = (f"def bind({', '.join(bound)}):\n" + textwrap.indent(source, "    ")
               + f"    return {name}\n")
    scope = {}
    code = compile(factory, f"<fast_triceratops {name}>", "exec")
    exec(code, module.__dict__, scope)  # noqa: S102 (pinned upstream source)
    return scope["bind"](*bound.values())


def repaired():
    """Rebuilt lnZ_STP and lnZ_SEB; fails if the installed source is not the pinned one."""
    from triceratops import marginal_likelihoods

    require_pinned(marginal_likelihoods, PINNED)
    return {name: rebuild(marginal_likelihoods, name, [(LOOP, LOOKUP, 1)],
                          {"_fast_ldc_lookup": ldc_lookup}) for name in PINNED}


def repaired_v3():
    """All 14 V3 scenario functions with the lookup repair and pruning-aware evidence;
    fails unless the installed source is the pinned V3 patch."""
    from triceratops import marginal_likelihoods

    from fast_triceratops.joint import fast_log_mean_exp, resolved

    require_pinned(marginal_likelihoods, PINNED_V3)
    functions = {}
    for name in PINNED_V3:
        source = inspect.getsource(getattr(marginal_likelihoods, name))
        replacements = [(old, new, source.count(old)) for old, new in EVIDENCE.items()]
        ranking = [(old, new, source.count(old)) for old, new in RANKING.items()]
        if (sum(count for _, _, count in replacements) != source.count("_log_mean_exp(")
                or sum(count for _, _, count in ranking) != source.count("argsort()")):
            raise RuntimeError(f"{name}: unexpected evidence or ranking call")
        replacements += ranking
        if name in PINNED:
            replacements.append((LOOP, LOOKUP, 1))
        if name in BACKGROUND:
            replacements.append((BACKGROUND[name] + BACKGROUND_TAIL, BACKGROUND_LOOKUP, 1))
        function = rebuild(marginal_likelihoods, name, replacements,
                           {"_fast_ldc_lookup": ldc_lookup, "_fast_log_mean_exp": fast_log_mean_exp,
                            "_fast_top": top, "_fast_background_ldc": background_ldc})
        functions[name] = functools.wraps(function)(resolved(function))
    return functions
