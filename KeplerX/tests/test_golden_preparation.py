"""Golden: production preparation of real photometry keeps its cadences and mask reasons."""

import numpy as np
from conftest import array_digests, golden, load_lane_host, tres2_packet

from common import load_packet
from runner.program import verify_upstream


def record(arrays, scales):
    return {
        "usable_cadences": int(np.asarray(arrays["mask"]).sum()),
        "exclusion_bits": {str(bit): int(((arrays["exclusion_bits"] & bit) > 0).sum()) for bit in (1, 2, 4, 8, 16, 32, 64)},
        "normalization_by_quarter": scales,
        "arrays": array_digests(arrays),
    }  # fmt: skip


def test_preparation_is_golden(tmp_path, config):
    prepared, _ = tres2_packet(tmp_path / "tres2", config)
    arrays, metadata = load_packet(prepared)
    lane = load_lane_host()
    golden("preparation", {
        "tres2_q0": record(arrays, metadata["normalization_by_quarter"]),
        "lane_k_dwarf_four_quarters": record(lane.prepared, lane.metadata["normalization_by_quarter"]),
    })  # fmt: skip


def test_vendored_upstream_files_match_their_checksums():
    assert set(verify_upstream()) == {"detector", "vetter", "localizator", "validator"}
