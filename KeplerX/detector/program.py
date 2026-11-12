"""Run the Detector's search on a prepared curve and save its candidates."""

from pathlib import Path

import numpy as np
import pandas as pd

from common import json_value, load_packet, receipt, write
from detector.gap_padding import MIN_SEARCH_CADENCES, DegenerateCurve
from detector.upstream_adapter import DURATION_WINDOW, make_dump


def kernels(compiled):
    if not compiled:
        return None
    from detector import compiled_search, gpu_screening, kernel_bindings

    lib = kernel_bindings.library()
    return {
        "source_sha256": lib.digest,
        "openblas_core": kernel_bindings.openblas_core(),
        "compiled_direct_convolution": lib.direct_convolution,
        "gpu_screening": gpu_screening.describe()
        if compiled_search.USAGE["gpu_chunks"]
        else None,
        "search_passes": dict(compiled_search.USAGE),
    }


def unsearchable_reason(dump):
    """Why upstream's grid admits no search, or None."""
    if len(dump.lc.time) < MIN_SEARCH_CADENCES:
        return f"Fewer than {MIN_SEARCH_CADENCES} usable cadences"
    if len(dump.tdurs) == 0:
        return "No template duration is admitted for this baseline and density"
    if len(dump.search_periods) == 0:
        return "No trial period is admitted for this baseline"
    return None


def search_detections(dump, candidates):
    """Every detection the passes masked, in order, with its candidate's signal ID."""
    rows = getattr(dump, "searched_TCEs", None)
    if rows is None or not len(rows) or dump.searched_is_candidate.all():
        return None
    ids = iter(candidate["signal_id"] for candidate in candidates)
    return [
        {
            "period": float(row[1]), "mes": float(row[2]), "t0": float(row[3]),
            "tdur": float(row[4]), "signal_id": next(ids) if flag else None,
        }
        for row, flag in zip(rows, dump.searched_is_candidate, strict=True)
    ]  # fmt: skip


def run(args, config, reference=None):
    """reference: one of the golden tests' upstream references (upstream_adapter.make_dump)."""
    arrays, metadata = load_packet(args.prepared)
    settings = config["detection"]
    compiled = reference != "native"
    if compiled:
        from detector import compiled_search

        for key in compiled_search.USAGE:
            compiled_search.USAGE[key] = 0.0 if key == "gpu_seconds" else 0
    dump = make_dump(
        arrays, metadata, settings,
        numerical_threads=config["execution"]["numerical_threads"],
        reference=reference,
    )  # fmt: skip
    if compiled:
        dump.search_only = True
    unsearchable = unsearchable_reason(dump)
    frame = None
    if not unsearchable:
        try:
            frame = dump.Iterative_TCE_Search(
                niter=settings["max_passes"],
                threshold=settings["mes_threshold"],
                remove_hises=False,
                check_vetoes=False,
                progbar=False,
                print_updates=False,
                dur_range=DURATION_WINDOW,
            )
        except DegenerateCurve as error:
            unsearchable = f"Whitening: {error}"
    if unsearchable:
        # Upstream fails on such curves downstream; they have no candidates.
        dump.tce_mask = dump.lc.mask.copy()
        frame = pd.DataFrame(columns=["star_id", "period", "mes", "t0", "tdur"])
    records = json_value(frame.to_dict("records"))
    write(Path(args.out) / "upstream-returned-candidates.json", records)
    candidates = []
    for row in records:
        if row["period"] is None:
            continue
        candidates.append(
            row
            | {
                "signal_id": f"{metadata['source_id']}:{len(candidates) + 1:02d}",
                "source_id": metadata["source_id"],
            }
        )
    searched = search_detections(dump, candidates)
    np.savez_compressed(
        Path(args.out) / "search-masks.npz",
        packet_indices=np.flatnonzero(arrays["mask"]),
        initial_mask=dump.lc.mask,
        final_mask=dump.tce_mask,
    )
    result = {
        "schema": 1,
        "source_id": metadata["source_id"],
        "engine": "upstream.Iterative_TCE_Search",
        "detection_engine": f"{reference} reference" if reference else "compiled",
        "compiled_kernels": kernels(compiled),
        "prepared": receipt(Path(args.prepared) / "packet.json"),
        "duration_days": dump.tdurs,
        "period_trial_count": len(dump.search_periods),
        "unsearchable": unsearchable,
        "candidates": candidates,
        "search_detections": searched,
        "internal_proposals": "NOT_EXPOSED_BY_UPSTREAM",
        "scope": "Native upstream baseline; population recovery and vetting acceptance are not calibrated",
    }
    write(Path(args.out) / "detections.json", result)
    return result


def arguments(parser):
    parser.add_argument("prepared", type=Path)
    parser.add_argument("out", type=Path)
