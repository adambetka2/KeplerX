"""Keep numerical metric failures explicit without discarding other evidence."""

import traceback

import numpy as np

from common import json_value
from vetter.binary_evidence import binary_evidence
from vetter.odd_even_box import odd_even_box
from vetter.policy import assess
from vetter.upstream.vetting import get_p_tdur_t0

NUMERICAL_FAILURES = (RuntimeError, ValueError, ArithmeticError, np.linalg.LinAlgError)


def failure(component, error):
    return {
        "component": component,
        "state": "UNRESOLVED",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


# The components a policy reads, and those a dismissed candidate may skip.
POLICY_COMPONENTS = (
    "mes",
    "chi2",
    "transit_cadences",
    "secondary",
    "cosine_vs_transit",
)
COSTLY_COMPONENTS = ("odd_even", "local_morphology")


def transit_cadences(time, period, epoch, exposure):
    """Cadences within one exposure of a transit centre."""
    fold = (time - epoch + period / 2.0) % period - period / 2.0
    return {"temporal_chi2_cadences": int(np.sum(np.abs(fold) <= exposure))}


def evaluate(components, metrics):
    """Run components into metrics; their numerical failures, in order."""
    errors = []
    for name, compute in components:
        try:
            metrics.update(compute())
        except NUMERICAL_FAILURES as error:
            errors.append(failure(name, error))
    return errors


def measure(vetter, index, local_test, policy=None, skipped=None):
    """Every metric of one signal, and its numerical failures."""
    try:
        bestfit = vetter._update_tce(index, return_fit=True)
        depth = vetter.refined_tces[index, 2]
        period, duration, epoch = get_p_tdur_t0(vetter.refined_tces[index])
        previous = vetter._get_previous_tce_masks(index)
    except NUMERICAL_FAILURES as error:
        return {}, [failure("transit_refinement", error)]
    metrics = {
        "tce_id": float(vetter.tces[index, 0] + (index + 1) * 1e-2),
        "period": period,
        "depth_ppt": depth * 1e3,
        "tdur": duration,
        "t0": epoch,
        "b": bestfit["b"],
    }
    components = [
        ("mes", lambda: vetter._get_mes_metrics(index, use_mask=previous)),
        ("chi2", lambda: vetter.chi2_tests(index, use_mask=None)),
        (
            "transit_cadences",
            lambda: transit_cadences(
                vetter.dump.lc.time[vetter.dump.lc.mask], period, epoch, vetter.exptime
            ),
        ),
        ("secondary", lambda: vetter.weak_secondary_test(index, mask=previous)),
        ("odd_even", lambda: vetter.odd_even_depth_test(index, True)),
        (
            "cosine_vs_transit",
            lambda: vetter.cosine_vs_transit_global(index, depth=depth),
        ),
        (
            "odd_even_box",
            lambda: odd_even_box(
                vetter.time,
                vetter.flux,
                vetter.flux_err,
                period,
                epoch,
                duration,
                vetter.exptime,
            ),
        ),  # fmt: skip
    ]
    if local_test:
        components.append(
            ("local_morphology", lambda: vetter.local_morphology_test(index))
        )
    # Last: it whitens again with the other signals masked.
    components.append(
        ("binary_evidence", lambda: binary_evidence(vetter, index, metrics))
    )
    if not policy:
        errors = evaluate(components, metrics)
        return json_value(metrics), errors
    first = [c for c in components if c[0] in POLICY_COMPONENTS]
    rest = [c for c in components if c[0] not in POLICY_COMPONENTS]
    errors = evaluate(first, metrics)
    if not errors and assess(json_value(metrics), policy)["state"] == "DISMISSED":
        if skipped is not None:
            skipped.extend(name for name, _ in rest if name in COSTLY_COMPONENTS)
        rest = [c for c in rest if c[0] not in COSTLY_COMPONENTS]
    errors += evaluate(rest, metrics)
    return json_value(metrics), errors
