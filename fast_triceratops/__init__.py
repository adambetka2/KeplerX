"""Bitwise-equal fast likelihoods for pinned TRICERATOPS d20875c."""

import contextlib


@contextlib.contextmanager
def accelerated():
    """Swap in the fast likelihoods and repaired scenarios, then restore upstream's."""
    from triceratops import marginal_likelihoods, triceratops

    from fast_triceratops.kernels import REPLACEMENTS
    from fast_triceratops.scenarios import repaired

    swaps = [(marginal_likelihoods, name, f) for name, f in REPLACEMENTS.items()]
    swaps += [(triceratops, name, f) for name, f in repaired().items()]
    saved = [(module, name, getattr(module, name)) for module, name, _ in swaps]
    try:
        for module, name, function in swaps:
            setattr(module, name, function)
        yield
    finally:
        for module, name, function in saved:
            setattr(module, name, function)


@contextlib.contextmanager
def engine():
    """Run calc_probs from fast_triceratops.engine, the performance-engineered copy of
    TRICERATOPS (stock route and V3 joint route), then restore upstream's."""
    from triceratops import triceratops as upstream

    from fast_triceratops.engine import require_unmodified
    from fast_triceratops.engine import triceratops as copy

    require_unmodified()
    saved = upstream.target.calc_probs
    upstream.target.calc_probs = copy.target.calc_probs
    try:
        yield
    finally:
        upstream.target.calc_probs = saved


@contextlib.contextmanager
def accelerated_v3(certified=False):
    """TRICERATOPS_KEPLER_V3 (the patched runtime with post_boi's joint likelihood): swap in
    the fused, pruned SourceLikelihood call and the rebuilt scenarios, then restore.
    Bitwise equal to upstream by default; certified=True drops weights below e^-50 of the
    maximum instead (joint.py)."""
    from triceratops import triceratops

    from fast_triceratops import joint
    from fast_triceratops.scenarios import repaired_v3

    functions = repaired_v3()
    saved = {name: getattr(triceratops, name) for name in functions}
    limit = joint.CERTIFIED if certified else joint.EXACT
    with joint.accelerated_joint(), joint.pruning(limit):
        try:
            for name, function in functions.items():
                setattr(triceratops, name, function)
            yield
        finally:
            for name, function in saved.items():
                setattr(triceratops, name, function)
