"""Raw pixel to production-record endpoint contract."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import native_reprocess as endpoint
from test_reprocess import gold


def test_adapter_refits_before_production_preparation(tmp_path):
    data = gold()
    data.update(cbv=np.zeros((len(data["time"]), 4)), source_flux=np.array(1000.0))
    data["cadence"] = np.arange(len(data["time"]))
    path = tmp_path / "input.npz"
    np.savez_compressed(path, **data)
    entry = {
        "name": "synthetic-q1",
        "source_id": "gold",
        "path": str(path),
        "sha256": endpoint.reprocess.digest(path),
    }
    jobs, truth = endpoint.inputs(entry, tmp_path / "prepared")
    assert len(jobs) == 9
    assert truth["period_days"] == 9.9731
    common_errors = []
    for job in jobs:
        with np.load(job["path"]) as saved:
            records = saved["records"]
            assert len(saved["flux"]) == len(data["time"])
            assert np.all(records["quarter"] == 1)
            assert np.all(np.diff(records["time"]) > 0)
            recovered_error = records["error"] * np.median(saved["flux"])
            np.testing.assert_allclose(recovered_error, np.sqrt(data["aperture"].sum()))
            common_errors.append(recovered_error)
    assert len(common_errors) == 9
