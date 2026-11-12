"""Frozen catalogue ephemeris associations shared by both detector lanes."""

import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

CADENCE_DAYS = 0.02043365
RULE = {
    "id": "complete-catalogue-common-recurrence-v1",
    "spatial_screen_arcsec": 1800,
    "position_compatible_arcsec": 2,
    "maximum_harmonic_order": 15,
    "maximum_period_error_fraction": 0.001,
    "minimum_common_centres": 3,
    "maximum_nearby_phase_coverage": 0.2,
    "scope": "Exact Gaia ID or positional compatibility plus a common recurrence of the supplied proposal centres. Aperture KIC equality never establishes source ownership. Flags are catalogue associations, not physical validation, false-positive denial or novelty.",
}
REQUIRED = {
    "NASA_cumulative_KOI",
    "NASA_Kepler_DR25_TCE",
    "NASA_confirmed_planets_PS",
    "Villanova_Kepler_EB_live",
    "Bienias_Szabo_2025",
}
RATIOS = [(m, n) for m in range(1, 16) for n in range(1, 16) if math.gcd(m, n) == 1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xyz(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).T


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class Snapshot:
    def __init__(self, manifest_path, manifest_sha256):
        path = Path(manifest_path)
        if digest(path) != manifest_sha256:
            raise ValueError("Catalogue manifest hash changed")
        self.manifest = json.loads(path.read_text())
        if (
            self.manifest["schema"] != 1
            or set(self.manifest["required_catalogues"]) != REQUIRED
        ):
            raise ValueError("Unsupported or incomplete catalogue manifest")
        rows_path = path.parent / self.manifest["signals"]["path"]
        if digest(rows_path) != self.manifest["signals"]["sha256"]:
            raise ValueError("Catalogue rows hash changed")
        self.rows = json.loads(rows_path.read_text())
        if len(self.rows) != self.manifest["counts"]["normalized_signal_rows"]:
            raise ValueError("Catalogue row count changed")
        if {row["catalogue"] for row in self.rows} != REQUIRED:
            raise ValueError("Required catalogue absent from rows")
        self.manifest_sha256 = manifest_sha256
        self.gaia = defaultdict(list)
        positioned = []
        for index, row in enumerate(self.rows):
            if row["gaia_dr3"] is not None:
                self.gaia[str(row["gaia_dr3"])].append(index)
            if finite(row["ra"]) and finite(row["dec"]):
                positioned.append(index)
            elif row["gaia_dr3"] is None:
                raise ValueError(f"No searchable identity or position: {row['name']}")
        self.indices = np.array(positioned)
        self.tree = cKDTree(
            xyz(
                [self.rows[i]["ra"] for i in positioned],
                [self.rows[i]["dec"] for i in positioned],
            )
        )

    def candidates(self, source):
        if not str(source.get("source_id", "")).isdigit():
            raise ValueError("Gaia source identity is unavailable or invalid")
        if (
            not all(finite(source.get(key)) for key in ("ra_deg", "dec_deg"))
            or not -90 <= source["dec_deg"] <= 90
        ):
            raise ValueError("Source coordinates are unavailable or invalid")
        point = xyz(source["ra_deg"], source["dec_deg"])
        radius = 2 * np.sin(np.radians(RULE["spatial_screen_arcsec"] / 3600) / 2)
        nearby = self.indices[self.tree.query_ball_point(point, radius)].tolist()
        return sorted(set(nearby) | set(self.gaia.get(str(source["source_id"]), [])))


@lru_cache(maxsize=2)
def snapshot(path, sha256):
    return Snapshot(path, sha256)


def snapshot_paths(config, root):
    spec = config.get("catalogue_search")
    if not spec or spec.get("rule") != RULE["id"]:
        raise ValueError("Required catalogue search configuration is missing")
    path = (Path(root) / spec["manifest"]).resolve()
    return path, path.parent / json.loads(path.read_text())["signals"]["path"]


def recurrence(centres, period, epoch, duration, known, numerator, denominator):
    tolerance = max(duration * 24, known["duration_hours"] or 0) / 48 + CADENCE_DAYS
    cycles = np.rint((centres - epoch) / period).astype(np.int64)
    shifts = (
        [0, known["period"] / 2] if known["classification"] == "EB_CATALOGUE" else [0]
    )
    groups = []
    for shift in shifts:
        residual = np.abs(
            (centres - known["epoch"] - shift + known["period"] / 2) % known["period"]
            - known["period"] / 2
        )
        for residue in range(denominator):
            values = residual[cycles % denominator == residue]
            if not len(values):
                continue
            span = float(np.ptp(centres[cycles % denominator == residue]))
            independent_cycles = len(np.unique(cycles[cycles % denominator == residue]))
            matched = values <= tolerance
            groups.append(
                {
                    "boi_cycle_residue": residue,
                    "boi_cycle_modulus": denominator,
                    "known_phase": shift / known["period"],
                    "event_count": len(values),
                    "independent_common_cycles": independent_cycles,
                    "matched_count": int(matched.sum()),
                    "match_fraction": float(matched.mean()),
                    "time_span_days": span,
                    "max_residual_days": float(values.max()),
                    "median_residual_days": float(np.median(values)),
                    "timing_tolerance_days": tolerance,
                    "accepted": independent_cycles >= RULE["minimum_common_centres"]
                    and span + 2 * tolerance >= 2 * numerator * known["period"]
                    and bool(np.all(matched)),
                }
            )
    return (
        max(
            groups,
            key=lambda g: (g["accepted"], g["match_fraction"], -g["max_residual_days"]),
        )
        if groups
        else None
    )


def compare(source, hypothesis, centres, known):
    period, epoch, duration = (
        hypothesis.get(key) for key in ("period_days", "epoch_bkjd", "duration_days")
    )
    if (
        not all(finite(value) for value in (period, epoch, duration))
        or period <= 0
        or duration <= 0
    ):
        return None
    exact = known["gaia_dr3"] is not None and str(known["gaia_dr3"]) == str(
        source["source_id"]
    )
    distance = None
    if known["ra"] is not None and known["dec"] is not None:
        chord = np.linalg.norm(
            xyz(source["ra_deg"], source["dec_deg"]) - xyz(known["ra"], known["dec"])
        )
        distance = float(np.degrees(2 * np.arcsin(min(1, chord / 2))) * 3600)
    proxy = "flux-weighted centroid" in known.get("position_source", "")
    same = (
        exact
        or distance is not None
        and distance <= RULE["position_compatible_arcsec"]
        and not proxy
    )
    if not finite(known["period"]) or known["period"] <= 0:
        return (
            {
                "catalogue": known["catalogue"],
                "name": known["name"],
                "flag": False,
                "availability": "NO_COMPARABLE_CATALOGUE_PERIOD",
                "host_relation": "SAME_SOURCE_EPHEMERIS_UNAVAILABLE",
            }
            if same
            else None
        )
    numerator, denominator = min(
        RATIOS, key=lambda r: abs(period / known["period"] / (r[0] / r[1]) - 1)
    )
    mismatch = abs(period / known["period"] / (numerator / denominator) - 1)
    if mismatch > RULE["maximum_period_error_fraction"]:
        return None
    result = {key: value for key, value in known.items() if key != "raw"}
    result.update(
        separation_arcsec=distance,
        exact_gaia_dr3_identity=exact,
        catalogued_kic_equals_input=known["kic"] == str(source.get("kic")),
        host_relation="SAME_GAIA_ID"
        if exact
        else "POSITION_COMPATIBLE_WITHIN_2_ARCSEC"
        if same
        else "NEARBY_CATALOGUED_SOURCE",
        rational_numerator=numerator,
        rational_denominator=denominator,
        rational_relative_mismatch=mismatch,
        proposal_hypothesis_id=hypothesis.get("hypothesis_id"),
        position_is_centroid_proxy=proxy,
    )
    coverage = min(
        1,
        2
        * (max(duration * 24, known["duration_hours"] or 0) / 48 + CADENCE_DAYS)
        / known["period"]
        * denominator,
    )
    if not same and (
        numerator > 1
        and denominator > 1
        or coverage > RULE["maximum_nearby_phase_coverage"]
    ):
        return {
            **result,
            "flag": False,
            "availability": "OUTSIDE_NEARBY_ASSOCIATION_RULE",
            "common_recurrence_phase_coverage": coverage,
        }
    if not finite(known["epoch"]) or not len(centres):
        return {
            **result,
            "flag": False,
            "availability": "NO_COMPARABLE_EPOCH_OR_CENTRES",
        }
    best = recurrence(centres, period, epoch, duration, known, numerator, denominator)
    coverage = (
        min(1, 2 * best["timing_tolerance_days"] / known["period"] * denominator)
        if best
        else None
    )
    flag = best is not None and best["accepted"]
    if not same:
        flag = (
            flag
            and coverage <= RULE["maximum_nearby_phase_coverage"]
            and (numerator == 1 or denominator == 1)
        )
    return {
        **result,
        "flag": bool(flag),
        "availability": "COMPARABLE",
        "best_phase_class": best,
        "common_recurrence_phase_coverage": coverage,
    }


def hypothesis_centres(events, hypothesis, count):
    ids = hypothesis.get("unit_ids")
    if ids is None:
        if count != 1:
            raise ValueError(
                "Multiple catalogue hypotheses require their own supplied event membership"
            )
        selected = events
    else:
        selected = [event for event in events if event.get("unit_id") in set(ids)]
        if {event.get("unit_id") for event in selected} != set(ids):
            raise ValueError("Catalogue hypothesis lost supplied event membership")
    return np.array(sorted({event["centre_bkjd"] for event in selected}))


def same_catalogue_object(left, right):
    if (left.get("planet_name") or left["name"]) != (
        right.get("planet_name") or right["name"]
    ):
        return False
    if left.get("proposal_hypothesis_id") != right.get("proposal_hypothesis_id"):
        return False
    if not finite(left.get("period")) or not finite(right.get("period")):
        return False
    if (
        abs(left["period"] / right["period"] - 1)
        > RULE["maximum_period_error_fraction"]
    ):
        return False
    if left.get("gaia_dr3") is not None and left["gaia_dr3"] == right.get("gaia_dr3"):
        return True
    if not all(finite(row.get(key)) for row in (left, right) for key in ("ra", "dec")):
        return False
    chord = np.linalg.norm(
        xyz(left["ra"], left["dec"]) - xyz(right["ra"], right["dec"])
    )
    return (
        np.degrees(2 * np.arcsin(min(1, chord / 2))) * 3600
        <= RULE["position_compatible_arcsec"]
    )


def resolve_coverage(reviewed):
    complete = [match for match in reviewed if match["availability"] == "COMPARABLE"]
    unavailable = [
        match
        for match in reviewed
        if match["availability"]
        not in {"COMPARABLE", "OUTSIDE_NEARBY_ASSOCIATION_RULE"}
    ]
    unresolved, covered = [], []
    for match in unavailable:
        donors = [
            {
                key: row[key]
                for key in (
                    "catalogue",
                    "name",
                    "row_index",
                    "period",
                    "epoch",
                    "time_reference",
                )
            }
            for row in complete
            if same_catalogue_object(match, row)
        ]
        if donors:
            covered.append(
                {
                    **match,
                    "object_coverage_closed_by": donors,
                    "interpretation": "This literature row retains its unavailable epoch; the same named, host-compatible and period-compatible object was compared using these qualified rows. No epoch is substituted.",
                }
            )
        else:
            unresolved.append(match)
    return unresolved, covered


def search(source, signal, catalogue):
    events = signal.get("events", [])
    if any(not finite(event.get("centre_bkjd")) for event in events):
        raise ValueError("Supplied event centres are unavailable or nonfinite")
    centres = np.array(sorted({event["centre_bkjd"] for event in events}))
    hypotheses = [
        item
        for item in signal.get("timing_hypotheses", [])
        if item.get("timing_basis") == "ORIGINAL_PROPOSAL"
    ]
    candidates = catalogue.candidates(source)
    selected = [
        (hypothesis, hypothesis_centres(events, hypothesis, len(hypotheses)))
        for hypothesis in hypotheses
    ]
    periodic = finite(signal.get("period_days")) and signal["period_days"] > 0
    periodic |= any(
        finite(item.get("period_days")) and item["period_days"] > 0
        for item in signal.get("timing_hypotheses", [])
    )
    invalid = periodic and (
        not hypotheses or any(not len(values) for _, values in selected)
    )
    invalid |= signal.get("period_days") is not None and (
        not finite(signal["period_days"]) or signal["period_days"] <= 0
    )
    invalid |= periodic and any(
        not all(
            finite(item.get(key))
            for key in ("period_days", "epoch_bkjd", "duration_days")
        )
        or item["period_days"] <= 0
        or item["duration_days"] <= 0
        for item in hypotheses
    )
    reviewed = [
        match
        for index in candidates
        for hypothesis, values in selected
        if (match := compare(source, hypothesis, values, catalogue.rows[index]))
        is not None
    ]
    matches = [match for match in reviewed if match["flag"]]
    incomplete, covered = resolve_coverage(reviewed)
    return {
        "status": "COMPLETE"
        if not incomplete and not invalid
        else "EPHEMERIS_COVERAGE_INCOMPLETE",
        "rule": RULE,
        "manifest_sha256": catalogue.manifest_sha256,
        "catalogue_counts": catalogue.manifest["counts"],
        "position_qualification": catalogue.manifest["position_qualification"],
        "matches": matches,
        "incomplete_pairs": incomplete,
        "covered_unavailable_literature_pairs": covered,
        "reviewed_period_compatible_pairs": reviewed,
        "spatial_candidates": len(candidates),
        "supplied_centres": centres.tolist(),
        "hypotheses_compared": len(hypotheses),
        "scope": "Unknown-period signals have no periodic identity test; unknown period alone remains eligible. No accepted association is not a novelty claim.",
    }


def measure_row(row, source, config, root):
    try:
        path, _ = snapshot_paths(config, root)
        catalogue = snapshot(str(path), config["catalogue_search"]["manifest_sha256"])
    except (OSError, ValueError, KeyError) as exc:
        unavailable = {
            "status": "SNAPSHOT_UNAVAILABLE",
            "rule": RULE,
            "error": f"{type(exc).__name__}: {exc}",
            "matches": [],
        }
        for signal in row.get("signals", []):
            signal["catalogue_search"] = unavailable
        return
    for signal in row.get("signals", []):
        try:
            signal["catalogue_search"] = search(source, signal, catalogue)
        except (KeyError, TypeError, ValueError) as exc:
            signal["catalogue_search"] = {
                "status": "SIGNAL_IDENTITY_UNAVAILABLE",
                "rule": RULE,
                "error": f"{type(exc).__name__}: {exc}",
                "matches": [],
            }


def admission(signal):
    report = signal.get("catalogue_search", {})
    matches = report.get("matches", [])
    compatible = [
        match
        for match in matches
        if match["host_relation"]
        in {"SAME_GAIA_ID", "POSITION_COMPATIBLE_WITHIN_2_ARCSEC"}
    ]
    if compatible:
        return "RECOVERED_CATALOGUE_ASSOCIATION", "KNOWN_SIGNAL_ASSOCIATION"
    if matches:
        return "NEARBY_SOURCE_UNCONFIRMED", "CATALOGUE_SOURCE_ATTRIBUTION_REQUIRED"
    if report.get("status") != "COMPLETE":
        return "CATALOGUE_SEARCH_INCOMPLETE", "CATALOGUE_EVIDENCE_INCOMPLETE"
    return "NO_MATCH_IN_QUERIED_SOURCES", None
