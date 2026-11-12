from pathlib import Path

import pytest

from pipeline import runtime


def test_workers_use_linux_cpu_affinity(monkeypatch):
    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 64)
    assert runtime.worker_count() == 32


def test_workers_use_mac_cpu_count_when_affinity_is_unavailable(monkeypatch):
    monkeypatch.delattr(runtime.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 12)
    assert runtime.worker_count() == 12
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: None)
    assert runtime.worker_count() == 1


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "32"])
def test_invalid_worker_limits_fail(value):
    with pytest.raises(ValueError, match="positive integers"):
        runtime.worker_count(value)


def test_full_cpu_profile_inherits_and_preserves_explicit_limits(monkeypatch):
    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    automatic = {"workers": None, "network_workers": None, "alias": {"workers": None}}
    assert runtime.execution_workers(automatic) == {
        "workers": 32, "network_workers": 32, "alias": {"workers": 32}
    }
    assert automatic["workers"] is None and automatic["alias"]["workers"] is None
    assert runtime.execution_workers({"workers": 8, "alias": {"workers": None}}) == {
        "workers": 8, "network_workers": 8, "alias": {"workers": 8}
    }
    explicit = {"workers": 8, "network_workers": 16, "alias": {"workers": 4}}
    assert runtime.execution_workers(explicit) == explicit


def test_dump_path_accepts_dedicated_external_workspace(tmp_path, monkeypatch):
    local = tmp_path / "dump"
    external = tmp_path / "external/dump"
    local.mkdir()
    external.mkdir(parents=True)
    (external / "archive").mkdir()
    (local / "archive").symlink_to(external / "archive")
    monkeypatch.setattr(runtime, "DUMP", local)
    monkeypatch.setattr(runtime, "EXTERNAL_DUMPS", (external,))
    monkeypatch.setattr(Path, "is_mount", lambda path: path == external.parents[1])

    assert runtime.dump_path(local / "archive") == external / "archive"
    assert runtime.dump_path(external / "archive") == external / "archive"

    (local / "outside").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Pipeline outputs must stay"):
        runtime.dump_path(local / "outside")


def test_output_path_rejects_unmounted_external_volume(monkeypatch):
    monkeypatch.setattr(Path, "is_mount", lambda path: False)
    with pytest.raises(ValueError, match="volume is not mounted"):
        runtime.dump_path(runtime.EXTERNAL_DUMPS[0] / "unmounted-output")


@pytest.mark.parametrize(
    ("status", "authorized", "role", "enabled", "production"),
    [
        ("DEVELOPMENT", False, None, False, False),
        ("BOI_CATALOGUE", False, None, True, True),
        ("FROZEN_EVALUATION", False, None, False, False),
        ("FROZEN_EVALUATION", True, "development_replay", True, False),
        ("FROZEN_EVALUATION", True, "final_replay", True, False),
        ("RELEASED", False, None, True, True),
    ],
)
def test_disposition_execution_records_effective_authorization(
    status, authorized, role, enabled, production
):
    policy = {
        "policy_id": "deterministic-boi-v1",
        "status": status,
        "boi_catalogue_enabled": True,
        "_evaluation_authorized": authorized,
        "_evaluation_role": role,
    }
    execution = runtime.disposition_execution(policy)
    assert execution["boi_catalogue_enabled"] is enabled
    assert execution["production_boi_enabled"] is production
    assert execution["evaluation_role"] == role
    policy["boi_catalogue_enabled"] = False
    assert runtime.disposition_execution(policy)["boi_catalogue_enabled"] is False


def test_authorized_evaluation_requires_a_recorded_role():
    with pytest.raises(ValueError, match="valid role"):
        runtime.disposition_execution(
            {"status": "FROZEN_EVALUATION", "_evaluation_authorized": True}
        )


def test_peak_memory_samples_this_process_tree():
    assert runtime.process_tree_rss_bytes() > 0
    memory = runtime.PeakMemory(interval=0.01)
    while memory.peak == 0:
        memory.halt.wait(0.01)
    assert memory.stop() >= memory.peak > 0
    assert runtime.cpu_seconds() > 0
