"""The source's other lane at a candidate's ephemeris: PSF Flat's MES where SAP detected a signal."""

import math

import numpy as np

from detector import candidate_ephemerides as ce
from detector.gap_padding import DegenerateCurve
from detector.upstream_adapter import make_dump


def measure(arrays, metadata, settings, candidates, numerical_threads=1):
    """{signal_id: {"cross_lane_mes", "cross_lane_events"}} of `candidates`."""
    empty = {"cross_lane_mes": None, "cross_lane_events": 0}
    if not candidates:
        return {}
    try:
        dump = make_dump(
            arrays, metadata, settings, numerical_threads=numerical_threads
        )
        tdurs = np.asarray(dump.tdurs)
        rows = {
            c["signal_id"]: int(np.argmin(np.abs(tdurs - c["tdur"])))
            for c in candidates
        }
        active = np.zeros(len(tdurs), dtype=bool)
        active[sorted(set(rows.values()))] = True
        mask = dump.lc.mask.copy()
        _, num, den = dump.Calculate_SES_by_Segment(
            mask=mask, fill_mode="reflect", _active_rows=active
        )
    except DegenerateCurve:
        return {c["signal_id"]: dict(empty) for c in candidates}
    time = dump.lc.time[mask]
    t = np.ascontiguousarray(time - time[0])
    num, den = np.ascontiguousarray(num), np.ascontiguousarray(den)
    use = np.ones(len(t), dtype=bool)
    out = {}
    for c in candidates:
        nums, dens, _ = ce.event_terms(
            t, num, den, use, c["period"], c["t0"] - time[0],
            float(dump.lc.exptime), rows[c["signal_id"]],
        )  # fmt: skip
        total = float(dens.sum())
        out[c["signal_id"]] = {
            "cross_lane_mes": float(nums.sum() / math.sqrt(total))
            if len(nums) and total > 0
            else None,
            "cross_lane_events": int(len(nums)),
        }
    return out
