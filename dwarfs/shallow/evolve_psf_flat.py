"""Search delivered PSF Flat with constant offsets instead of another time filter."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

PATH = Path(__file__).with_name("evolve_native_coherence.py")
SPEC = importlib.util.spec_from_file_location(__name__ + "_native", PATH)
NATIVE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NATIVE
SPEC.loader.exec_module(NATIVE)
ShallowIneligible = NATIVE.ShallowIneligible
bls_library = NATIVE.bls_library
PREPARATION = {
    "view": "psf_flat",
    "flatten_mask": True,
    "seed_trend": "quarter_median",
    "physical_nuisance": "segment_intercept",
}


def constant_trend(time, flux, window, settings):
    if settings != {"method": "quarter_median"}:
        raise ValueError("PSF Flat requires explicit quarter-median centering")
    return np.full(len(flux), np.median(flux))


def intercept_basis(time, quarter, selected):
    segment = (
        np.cumsum(
            np.r_[
                True, (np.diff(quarter) != 0) | (np.diff(time) > selected["gap_days"])
            ]
        )
        - 1
    )
    matrix = csr_matrix(
        (np.ones(len(time)), (np.arange(len(time)), segment)),
        shape=(len(time), int(segment[-1]) + 1),
    )
    descriptions = [
        {
            "segment": int(label),
            "rows": int(np.count_nonzero(segment == label)),
            "degree": 0,
            "model": "fixed_intercept",
        }
        for label in np.unique(segment)
    ]
    return matrix, segment, descriptions


# NATIVE and its dependencies are private module instances, leaving the incumbent intact.
NATIVE.COMMON.detrend = constant_trend
NATIVE.PHYSICAL.spline_basis = intercept_basis


def scan(records, config, diagnostics=None):
    if config.get("psf_flat_preparation") != PREPARATION:
        raise ValueError("Explicit delivered PSF Flat preparation contract required")
    if config["input_view"] != "psf_flat":
        raise ValueError("PSF Flat detector refuses another declared input view")
    chosen = {
        key: value for key, value in config.items() if key != "psf_flat_preparation"
    }
    chosen["detrending"] = {"method": "quarter_median"}
    chosen["long_period_search"] = chosen["long_period_search"] | {
        "detrending": {"method": "quarter_median"}
    }
    evidence = {} if diagnostics is None else diagnostics
    proposals = NATIVE.scan(records, chosen, diagnostics=evidence)
    evidence["psf_flat_preparation"] = PREPARATION | {
        "source_sha256": NATIVE.COMMON.digest(__file__),
        "scope": "Conditional on archived upstream PSF flattening; its transit transfer is unmeasured",
    }
    return proposals
