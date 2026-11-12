"""Independent Vetter metrics with explicit, separately calibrated policy."""

from pathlib import Path

import numpy as np

from common import digest, load_packet, read, receipt, require, write
from detector.upstream_adapter import make_dump
from vetter.metrics import measure
from vetter.policy import assess, has_score

DISMISSED_METRICS = ("all", "policy")


def vet_one(vetter, index, candidate, config, lane, other_lane=None):
    policy = config.get("policy")
    mode = config.get("dismissed_metrics", "all")
    require(mode in DISMISSED_METRICS, f"Unknown vetting.dismissed_metrics {mode}")
    skipped = []
    metrics, errors = measure(
        vetter, index, config["local_morphology"],
        policy=policy if mode == "policy" else None, skipped=skipped,
    )  # fmt: skip
    metrics |= other_lane or {}
    unavailable = [key for key, value in metrics.items() if value is None]
    state = {
        "state": "UNRESOLVED" if errors else "UNCLASSIFIED",
        "basis": "Numerical metric failure"
        if errors
        else "Metrics only; no calibrated acceptance policy",
        "failed_components": [error["component"] for error in errors],
    }
    # A policy may dismiss, never accept; a metric failure stays UNRESOLVED.
    if config.get("policy") and not errors:
        verdict = assess(metrics, config["policy"], lane)
        state["state"] = verdict["state"]
        state["basis"] = (
            f"Vetting policy {config['policy']}, {lane} score (not a validation)"
        )
        state["policy_rules_fired"] = verdict["failed"]
        if verdict.get("binary_evidence"):
            state["binary_evidence"] = verdict["binary_evidence"]
        if skipped:
            state["skipped_components"] = skipped
    return {
        "signal_id": candidate["signal_id"],
        "detector_clock": candidate,
        "metrics": metrics,
        "unavailable_metrics": unavailable,
        "metric_errors": errors,
        "disposition": state,
        "view": lane,
        "raw_flux_reconstruction": "NOT_AVAILABLE; upstream trend=1 represents the prepared lane as supplied",
    }


def run(args, config):
    arrays, metadata = load_packet(args.prepared)
    detections_path = Path(args.detections) / "detections.json"
    detections = read(detections_path)
    require(
        detections["source_id"] == metadata["source_id"],
        "Detection and photometry source differ",
    )
    require(
        detections["prepared"]["sha256"] == digest(Path(args.prepared) / "packet.json"),
        "Detection photometry generation differs",
    )
    candidates = detections["candidates"]
    # The lane is the packet's view (older packets carry PSF Flat without one).
    lane = metadata.get("view", "psf_flat")
    require(
        not config["vetting"].get("policy") or has_score(lane),
        f"No vetting score is fitted for lane {lane}",
    )
    results = []
    weight = config["vetting"].get("refinement_weight", "sigma")
    if candidates:
        dump = make_dump(
            arrays, metadata, config["detection"], stage="vetter",
            numerical_threads=config["execution"]["numerical_threads"],
        )  # fmt: skip
        # Repeats the search masked stay in the TCE list, as upstream's vetting masks them.
        rows = detections.get("search_detections") or candidates
        tces = np.array(
            [[0, row["period"], row["mes"], row["t0"], row["tdur"]] for row in rows],
            dtype=float,
        )
        position = {row["signal_id"]: i for i, row in enumerate(rows)}
        from vetter.accelerated_vetter import FastVetter

        vetter = FastVetter(dump, tces, refinement_weight=weight)
        vetter.morphology_groups = config["vetting"].get("morphology_groups")
        other = other_lane(args, config, candidates)
        for number, candidate in enumerate(candidates, 1):
            index = position[candidate["signal_id"]]
            result = vet_one(
                vetter, index, candidate, config["vetting"], lane,
                other.get(candidate["signal_id"]),
            )  # fmt: skip
            results.append(result)
            write(Path(args.out) / f"signal-{number:02d}.json", result)
    result = {
        "source_id": metadata["source_id"],
        "detections": receipt(detections_path),
        "results": results,
        "detection_engine": "compiled",
        "refinement_weight": weight,
        "decision_policy_applied": bool(config["vetting"].get("policy")),
        "policy": config["vetting"].get("policy"),
        "lane": lane,
        "other_lane": receipt(Path(args.cross_lane) / "packet.json")
        if getattr(args, "cross_lane", None)
        else None,
        "validation": "NOT_PERFORMED",
    }
    write(Path(args.out) / "vetting.json", result)
    return result


def other_lane(args, config, candidates):
    """The source's other lane."""
    if not getattr(args, "cross_lane", None):
        return {}
    from vetter.cross_lane import measure as cross_lane

    arrays, metadata = load_packet(args.cross_lane)
    source = read(Path(args.prepared) / "packet.json")["source_id"]
    require(metadata["source_id"] == source, "The other lane's source differs")
    return cross_lane(
        arrays, metadata, config["detection"], candidates,
        numerical_threads=config["execution"]["numerical_threads"],
    )  # fmt: skip


def arguments(parser):
    parser.add_argument("prepared", type=Path)
    parser.add_argument("detections", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--cross-lane",
        type=Path,
        help="Prepared packet of the source's other lane (PSF Flat for SAP)",
    )
