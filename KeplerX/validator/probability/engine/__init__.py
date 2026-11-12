"""Adapted from TRICERATOPS (Giacalone et al. 2021), performance-
engineered with bitwise-equal outputs.

marginal_likelihoods.py and triceratops.py start from TRICERATOPS_KEPLER_V3:
upstream d20875c with validator/patches/triceratops-quarters.patch (sha256 41f53352...
and 1e3a3d65..., the patch's recorded results), importing the unmodified modules below
from the installed package. Every change since is listed in docs/fast-triceratops.md.
"""

import functools
import hashlib
from pathlib import Path

UNMODIFIED = {  # installed upstream modules the copy imports, as pinned at d20875c
    "likelihoods": "29935ef86963fa7beefac9d2c528daad0a1cc93019eafe192ae9c0da62a6b2d6",
    "priors": "173b8c6a90c9bccaf02747d0cee2d7c65e8387e11613cf799394af5ee2d59722",
    "funcs": "234f18fa7fbd7ae122046dfbd50603e47a4a1eaba773a524bd42b04a0e7bdabf",
    "_numerics": "ebb567d0bf2e411605eb6462c7466a3a7e7dedad47616762a4fffc3cbe3cee0e",
}
FITPACK_SHA256 = "e4d4e6fb57f87999e06c9da69d607676e8fc51cce196860e81f3a5f51a8fa261"


def require_unmodified():
    """Fail unless the installed modules the copy imports are the pinned ones."""
    import importlib

    for name, checksum in UNMODIFIED.items():
        source = Path(
            importlib.import_module(f"triceratops.{name}").__file__
        ).read_bytes()
        if hashlib.sha256(source).hexdigest() != checksum:
            raise RuntimeError(
                f"triceratops.{name} differs from pinned TRICERATOPS d20875c"
            )
    from scipy.interpolate import dfitpack

    if (
        hashlib.sha256(Path(dfitpack.__file__).read_bytes()).hexdigest()
        != FITPACK_SHA256
    ):
        raise RuntimeError("Requalify the spline arithmetic for this FITPACK binary")


def pruned_calculation(function):
    """Resolve the copy's deferred rows inside one complete probability call."""

    @functools.wraps(function)
    def run(*args, **kwargs):
        from validator.probability.joint import pruning

        require_unmodified()
        with pruning():
            return function(*args, **kwargs)

    return run
