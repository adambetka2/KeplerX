"""End-to-end: the calibration chain."""

import numpy as np
import pandas as pd
from conftest import inject, load_lane_host, planet

from calibration import sample_builder as builder
from calibration import study
from common import read, write


def mini_sample(root):
    lane = load_lane_host()
    host, out = lane.host, root / "sample"
    rows, signals = [], []
    rows.append(
        builder.save_case(
            out,
            "control_original",
            host,
            lane.prepared,
            lane.metadata,
            {"scope": "Uninjected"},
        )
    )
    for index, body in enumerate(
        (planet(lane, 4.8, 1.5, 1.3), planet(lane, 9.7, 0.8, 2.1))
    ):
        arrays, metadata, truth = inject(lane, [body])
        rows.append(builder.save_case(out, "synthetic_planet", host, arrays, metadata,
                                      {"signals": truth, "system_size": 1}, index))  # fmt: skip
        signals += [
            {"case_id": rows[-1]["case_id"], "case_kind": "synthetic_planet", **s}
            for s in truth
        ]
    inverted = builder.transformed_control(
        lane.prepared, lane.metadata, "inverted", 7, 480
    )
    rows.append(builder.save_case(out, "control_inverted", host, inverted, lane.metadata,
                                  {"transformation": "inverted", "seed": 7}))  # fmt: skip
    cases = pd.DataFrame(rows).assign(
        role="lane", artifact_pair=None, lane_weight=[1.0, np.nan, np.nan, np.nan]
    )
    builder.write_batches(out, cases, builder.PLAN)
    pd.DataFrame([builder.flat_signal(s) for s in signals]).to_parquet(
        out / "signals.parquet"
    )
    write(
        out / "summary.json",
        {"cases": len(rows), "scope": "Miniature end-to-end sample"},
    )
    registry = root / "registered-sample.json"
    builder.register(out, registry)
    return registry, signals


def test_a_calibration_study_end_to_end(tmp_path):
    registry, truth = mini_sample(tmp_path)
    strong, weak = truth
    assert strong["detectable"] and not weak["detectable"]
    studies = tmp_path / "studies"
    settings = dict(
        registry=registry, studies=studies, bootstrap=5, require_clean=False
    )
    study.run("miniature", 4, ["development"], **settings)
    folder = studies / "miniature"
    record = read(folder / "study.json")
    assert record["state"] == "SCORED" and len(record["attempts"]) == 1
    assert record["attempts"][0]["exit_code"] == 0
    signals = pd.read_parquet(folder / "score/signals.parquet").set_index("case_id")
    assert signals.loc[strong["case_id"], "outcome"] == "exact"
    assert signals.loc[weak["case_id"], "outcome"] == "miss"
    candidates = pd.read_parquet(folder / "score/candidates.parquet")
    nulls = candidates[candidates.kind.isin(["control_original", "control_inverted"])]
    assert not np.isclose(nulls.period, 4.8, rtol=0.01).any()
    summary = read(folder / "score/summary.json")["partitions"]["development"]
    assert summary["detection_on_detectable_planets"]["exact"] == 1.0
    study.run("miniature", 4, ["development"], **settings)
    again = read(folder / "study.json")
    assert len(again["attempts"]) == 1 and (folder / "score-2/summary.json").exists()
