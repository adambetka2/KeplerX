"""Real survey curves whose parent pixels are on the mounted discs, with localization requests."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from calibration import survey_benchmark as survey
from common import write
from localizator import pixel_requests

PIXELS = (
    Path("/Volumes/discSSD/data/pixels/kepler"),
    Path("/Volumes/discPCIe/data/pixels/kepler"),
)
MIN_QUARTERS = 8
PER_PARENT = 3


def stored_tpfs():
    """{parent KIC: its indexed TPFs} for parents with MIN_QUARTERS or more."""
    table = pixel_requests.index(PIXELS)
    return {
        kic: group for kic, group in table.groupby("kic") if len(group) >= MIN_QUARTERS
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    frame, _ = survey.eligible()
    tpfs = stored_tpfs()
    rng = np.random.default_rng(survey.SEED)
    rows = []
    for kic, group in frame[frame.origin_tpf_kic.isin(tpfs)].groupby("origin_tpf_kic"):
        take = group.sample(n=min(PER_PARENT, len(group)), random_state=rng)
        rows.append(take.assign(set="pixels", weight=1.0, quick=False))
    cases = pd.concat(rows, ignore_index=True)
    cases["case_id"] = [survey.case_id(r) for r in cases.to_dict("records")]
    jobs = [(r, str(args.out)) for r in cases.to_dict("records")]
    with ProcessPoolExecutor(args.workers) as pool:
        done = {c: (f, e) for c, f, e in pool.map(survey.build_one, jobs)}
    cases["prepared"] = [done[c][0] for c in cases.case_id]
    cases["preparation_error"] = [done[c][1] for c in cases.case_id]
    cases["quarters"] = [len(tpfs[k]) for k in cases.origin_tpf_kic]
    columns = [
        "case_id", "source_id", "kic", "origin_tpf_kic", "family", "population",
        "phot_g_mean_mag", "ra_deg", "dec_deg", "quarters", "prepared",
        "preparation_error",
    ]  # fmt: skip
    cases = cases.reindex(columns=columns)
    cases.to_parquet(args.out / "cases.parquet")
    ok = cases[cases.prepared.notna()]
    with (args.out / "manifest.jsonl").open("w") as stream:
        for row in ok.itertuples():
            request = pixel_requests.request(
                row.source_id, row.origin_tpf_kic, row.ra_deg, row.dec_deg,
                tpfs[row.origin_tpf_kic],
            )  # fmt: skip
            line = {"case_id": row.case_id, "prepared": row.prepared}
            stream.write(json.dumps(line | {"localization": request}) + "\n")
    write(args.out / "build.json", {
        "seed": survey.SEED, "min_quarters": MIN_QUARTERS, "per_parent": PER_PARENT,
        "parents": int(ok.origin_tpf_kic.nunique()), "cases": len(cases),
        "prepared": len(ok), "tpf_quarters": int(ok.quarters.sum()),
    })  # fmt: skip
    print((args.out / "build.json").read_text())


if __name__ == "__main__":
    main()
