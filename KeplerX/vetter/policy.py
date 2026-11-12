"""Vetting policies: named rules that may dismiss a candidate from its metrics."""

import math
import numbers

# Excess scatter per unit signal is clipped to [FLOOR, CEILING] before its log.
FLOOR, CEILING = 0.01, 10.0

FEATURES = (
    "log_depth_scatter", "log_channel_scatter", "secondary_to_primary_mes",
    "negative_to_primary_mes", "log_model_misfit", "log_period", "log_noise_mad",
    "log_max_mes", "cross_lane_ratio",
)  # fmt: skip

POLICIES = {
    "linear-score": {
        "intercept": 1.2775,
        "weights": {
            "log_depth_scatter": -0.8446,
            "log_channel_scatter": -0.4080,
            "secondary_to_primary_mes": -3.5684,
            "negative_to_primary_mes": -0.9271,
            "log_model_misfit": -0.6132,
            "log_period": -1.2650,
            "log_noise_mad": -4.2752,
            "log_max_mes": 0.0,
        },  # fmt: skip
        "nonnegative": ("log_max_mes",),
        "cut": -2.2385,
    },
}

# Odd/even at 5 sigma alone, or two pieces, never a secondary alone (docs/keplerX-2026-11-03-binary-gate-strengthened.md).
BINARY_GATE = {
    "secondary_significance": 5.0,  # over the fold's noise (MAD of the MES)
    "secondary_uniqueness": 2.0,  # over the next dip and the brightest bump, in noise
    "secondary_event_sigma": 3.0,  # its depth event by event, over their scatter
    "secondary_halves_sigma": 2.0,  # the same in either half of the baseline
    "occultation_albedo": 1.0,
    "odd_even_sigma": 5.0,
    "odd_even_corroborating_sigma": 3.0,
    "odd_even_transits": 3,  # each of odd and even
    "ellipsoidal_sigma": 7.0,
    "ellipsoidal_mass_ratio": 0.05,  # 50 Jupiter masses for a solar host
    "deep_primary_ppt": 30.0,
    "grazing_depth_ppt": 10.0,
    "grazing_impact": 0.8,
}
POLICIES["linear-score-binary-gate"] = POLICIES["linear-score"] | {
    "binary_gate": BINARY_GATE
}
# Each lane's own score where it is not PSF Flat's (docs/keplerX-2026-11-03-sap-dilution-and-calibration.md).
LANE_SCORES = {
    "sap_flat": {
        "intercept": 2.9522,
        "weights": {
            "log_depth_scatter": -0.6108,
            "log_channel_scatter": -0.5010,
            "secondary_to_primary_mes": -4.4477,
            "negative_to_primary_mes": -1.2775,
            "log_model_misfit": 0.3005,
            "log_period": -1.5441,
            "log_noise_mad": -4.7176,
            "log_max_mes": 0.0,
        },
        "cut": -2.7250,
    },
    "sap_flat_with_psf_flat": {
        "intercept": 3.5219,
        "weights": {
            "log_depth_scatter": -0.4223,
            "log_channel_scatter": -0.3397,
            "secondary_to_primary_mes": -3.6446,
            "negative_to_primary_mes": -1.0994,
            "log_model_misfit": 0.2536,
            "log_period": -1.5513,
            "log_noise_mad": -3.8095,
            "log_max_mes": 0.0,
            "cross_lane_ratio": 2.8294,
        },
        "cut": -2.2645,
    },
}  # fmt: skip
WITH_OTHER_LANE = {"sap_flat": "sap_flat_with_psf_flat"}

REASONS = {
    "score": "The vetting score is below its cut",
    "secondary": "A unique secondary eclipse deeper than a planet's occultation",
    "odd_even": "Odd and even transits differ in depth",
    "ellipsoidal": "Ellipsoidal variation of a stellar companion",
    "deep_primary": "A primary eclipse deeper or more grazing than a planet's usually is",
}


def features(metrics):
    """Every policy feature from a signal's metrics; None where unavailable."""
    m = {k: v if _finite(v) else None for k, v in metrics.items()}
    primary = m.get("max_mes") if (m.get("max_mes") or 0) > 0 else None
    period = m.get("period") if (m.get("period") or 0) > 0 else None
    cadences = m.get("temporal_chi2_cadences")
    sec, neg, noise = m.get("max_sec_mes"), m.get("min_sec_mes"), m.get("mad_sec_mes")
    misfit = m.get("global_tran_red_chi2")
    return {
        "log_depth_scatter": _scatter(
            m.get("temporal_chi2_stat"), m.get("temporal_red_chi2"),
            cadences - 1 if cadences else None,
        ),
        "log_channel_scatter": _scatter(
            m.get("channel_chi2_stat"), m.get("channel_red_chi2"), cadences
        ),
        "secondary_to_primary_mes": sec / primary if primary and sec is not None else None,
        "negative_to_primary_mes": -neg / primary if primary and neg is not None else None,
        "log_model_misfit": _clipped_log(math.sqrt(max(misfit - 1.0, 0.0)) / primary)
        if primary and misfit is not None else None,
        "log_period": math.log10(period) if period else None,
        "log_noise_mad": math.log10(noise) if noise and noise > 0 else None,
        "log_max_mes": math.log10(max(primary, 1.0)) if primary else None,
        "cross_lane_ratio": _cross_lane_ratio(m, primary),
    }  # fmt: skip


def _cross_lane_ratio(m, primary):
    """The other lane's MES at the candidate's ephemeris."""
    cross, seen = m.get("cross_lane_mes"), m.get("cross_lane_events")
    if cross is None or not seen or not primary:
        return None
    expected = primary * math.sqrt(seen / max(m.get("num_transits") or 1, 1))
    return math.log10(min(max(cross / expected, 0.05), 3.0))


def score(values, policy):
    """A linear policy's score, or None when a feature is unavailable."""
    total = policy["intercept"]
    for name, weight in policy["weights"].items():
        if values[name] is None:
            return None
        total += weight * values[name]
    return total


def lane_policy(name, lane):
    """The policy `name` with `lane`'s own score."""
    return POLICIES[name] | LANE_SCORES.get(lane, {})


def has_score(lane):
    """Whether `lane` has its own fitted score (PSF Flat's is the policies')."""
    return lane == "psf_flat" or lane in LANE_SCORES


def score_lane(lane, values):
    """The lane whose score judges these feature values."""
    other = WITH_OTHER_LANE.get(lane)
    if other in LANE_SCORES and values.get("cross_lane_ratio") is not None:
        return other
    return lane


def assess(metrics, name, lane="psf_flat"):
    """{"state": "KEPT" or "DISMISSED", "failed": [what fired]}."""
    values = features(metrics)
    policy = lane_policy(name, score_lane(lane, values))
    value = score(values, policy)
    failed, evidence = [], []
    if value is not None and value < policy["cut"]:
        failed = [{"feature": "score", "value": value, "cut": f"< {policy['cut']}",
                   "reason": REASONS["score"]}]  # fmt: skip
    if policy.get("binary_gate"):
        pieces, dismissed = binary_pieces(metrics, policy["binary_gate"])
        if dismissed:
            failed += pieces
        else:
            evidence = pieces
    verdict = {
        "policy": name,
        "state": "DISMISSED" if failed else "KEPT",
        "failed": failed,
    }
    if evidence and not failed:
        verdict["binary_evidence"] = evidence
    return verdict


def gaussian_sigma(t, dof):
    """The Gaussian significance with the chance of Student's t at `dof` degrees of freedom."""
    from scipy import stats

    return float(stats.norm.isf(stats.t.sf(abs(t), dof)))


def binary_pieces(metrics, gate):
    """The pieces of binary evidence a signal's metrics show, and whether they dismiss it."""
    m = {
        k: v if _finite(v) or isinstance(v, bool) else None for k, v in metrics.items()
    }
    odd_even, decisive = _odd_even_piece(m, gate)
    secondary, ellipsoidal = _secondary_piece(m, gate), _ellipsoidal_piece(m, gate)
    primary = _primary_piece(m, gate)
    pieces = [
        piece
        for piece in (secondary, odd_even, ellipsoidal, primary)
        if piece is not None
    ]
    measured = [piece for piece in (secondary, odd_even, ellipsoidal) if piece]
    corroborated = len(measured) >= 2 or bool(primary and (secondary or ellipsoidal))
    return pieces, decisive or corroborated


def _at_least(value, cut):
    return value is not None and value >= cut


def _secondary_piece(m, gate):
    sec, noise = m.get("binary_secondary_mes"), m.get("binary_secondary_noise")
    rivals = [m.get("binary_next_dip_mes"), m.get("binary_brightening_mes")]
    albedo = m.get("binary_occultation_albedo")
    if sec is None or noise is None or albedo is None or None in rivals:
        return None
    noise = max(noise, 1.0)
    if (
        sec / noise >= gate["secondary_significance"]
        and (sec - max(rivals)) / noise >= gate["secondary_uniqueness"]
        and not m.get("binary_secondary_is_half_period")
        and not m.get("binary_secondary_is_own_transit")
        and _at_least(
            m.get("binary_secondary_event_sigma"), gate["secondary_event_sigma"]
        )
        and _at_least(
            m.get("binary_secondary_halves_sigma"), gate["secondary_halves_sigma"]
        )
        and albedo > gate["occultation_albedo"]
    ):
        return {"feature": "binary_occultation_albedo", "value": albedo,
                "cut": f"> {gate['occultation_albedo']}", "reason": REASONS["secondary"]}  # fmt: skip
    return None


def _odd_even_piece(m, gate):
    sigma = m.get("binary_odd_even_sigma")
    transits = min(
        m.get("binary_odd_transits") or 0, m.get("binary_even_transits") or 0
    )
    if sigma is None or transits < gate["odd_even_transits"]:
        return None, False
    # Welch's t, judged at its fewest degrees of freedom.
    sigma = gaussian_sigma(sigma, transits - 1)
    decisive = sigma >= gate["odd_even_sigma"]
    cut = gate["odd_even_sigma" if decisive else "odd_even_corroborating_sigma"]
    if sigma < cut:
        return None, False
    return {"feature": "binary_odd_even_sigma", "value": sigma, "cut": f">= {cut}",
            "reason": REASONS["odd_even"]}, decisive  # fmt: skip


def _ellipsoidal_piece(m, gate):
    ratio = m.get("binary_ellipsoidal_mass_ratio")
    strength = m.get("binary_ellipsoidal_sigma")
    if (
        ratio is not None
        and strength is not None
        and m.get("binary_ellipsoidal_halves_agree")
        and strength >= gate["ellipsoidal_sigma"]
        and ratio > gate["ellipsoidal_mass_ratio"]
    ):
        return {"feature": "binary_ellipsoidal_mass_ratio", "value": ratio,
                "cut": f"> {gate['ellipsoidal_mass_ratio']}", "reason": REASONS["ellipsoidal"]}  # fmt: skip
    return None


def _primary_piece(m, gate):
    """A primary eclipse deep or grazing enough to corroborate other binary evidence."""
    depth, impact = m.get("depth_ppt"), m.get("b")
    if depth is None:
        return None
    grazing = impact is not None and impact >= gate["grazing_impact"]
    if depth >= gate["deep_primary_ppt"] or (
        grazing and depth >= gate["grazing_depth_ppt"]
    ):
        return {"feature": "depth_ppt", "value": depth,
                "cut": f">= {gate['deep_primary_ppt']}, or >= {gate['grazing_depth_ppt']} at b >= {gate['grazing_impact']}",
                "reason": REASONS["deep_primary"]}  # fmt: skip
    return None


def _scatter(stat, red_chi2, dof):
    """log10 of sqrt((red_chi2 - 1) dof) / Z, with Z = stat sqrt(red_chi2) the statistic's signal."""
    if stat is None or red_chi2 is None or not dof or red_chi2 <= 0 or dof < 1:
        return None
    z = stat * math.sqrt(red_chi2)
    if z <= 0:
        return _clipped_log(CEILING)
    return _clipped_log(math.sqrt(max(red_chi2 - 1.0, 0.0) * dof) / z)


def _clipped_log(value):
    return math.log10(min(value, CEILING) + FLOOR)


def _finite(value):
    """A real number (numpy's too) that is finite; never a boolean."""
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )
