"""Shared runtime constraints for pipeline programs."""

import hashlib
import json
import os
import resource
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUMP = (ROOT / "dump").resolve()
EXTERNAL_DUMPS = (
    Path("/Volumes/discSSD/forgotten-stars-workspace/dump"),
    Path("/Volumes/discPCIe/forgotten-stars-workspace/dump"),
)

# Set BLAS threads before importing NumPy to avoid oversubscription and keep sums reproducible.
SERIAL_BLAS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def worker_count(value=None):
    if value is None:
        affinity = getattr(os, "sched_getaffinity", None)
        value = len(affinity(0)) if affinity else os.cpu_count() or 1
    if type(value) is not int or value < 1:
        raise ValueError("Worker counts must be positive integers or null for all CPUs")
    return value


def execution_workers(config):
    config = dict(config)
    config["workers"] = worker_count(config.get("workers"))
    config["network_workers"] = worker_count(
        config.get("network_workers") if config.get("network_workers") is not None
        else config["workers"]
    )
    alias = dict(config.get("alias", {}))
    alias["workers"] = worker_count(
        alias.get("workers") if alias.get("workers") is not None else config["workers"]
    )
    config["alias"] = alias
    return config


OUTCOME_CONTRACT = "boi-and-triage-v1"
DISPOSITION_LABELS = {"BOI", "HELD", "DISMISSED", "FALSE_ALARM", "FALSE_POSITIVE", "UNRESOLVED"}
EVENT_MATCHING_CONTRACT = {
    "id": "physical-event-membership-v1",
    "time_scale": "BKJD",
    "exposure_days": 1765.464 / 86400,
    "timing_tolerance": "half_truth_duration_plus_one_exposure",
    "required_supplied_event_match_fraction": 1.0,
    "minimum_recovered_truth_events": 1,
    "assignment": "unique_quarter_compatible_truth_event_per_supplied_unit",
    "truth_support": "prepared_cadences_inside_physical_truth_duration",
    "disposition_timing": "selected_canonical_physical_fit",
}

DISPOSITION_CERTIFICATE_FIELDS = {
    "_evaluation_authorized",
    "_evaluation_role",
    "boi_catalogue_enabled",
    "current_state",
    "definition_sha256",
    "release_binding",
    "release_checks",
    "status",
}


def disposition_execution(policy):
    policy = policy or {}
    status = policy.get("status", "UNCONFIGURED")
    evaluation = (
        status == "FROZEN_EVALUATION" and policy.get("_evaluation_authorized") is True
    )
    role = policy.get("_evaluation_role") if evaluation else None
    if evaluation and role not in {"development_replay", "final_replay"}:
        raise ValueError("Authorized disposition evaluation has no valid role")
    enabled = policy.get("boi_catalogue_enabled") is True and (
        status in {"BOI_CATALOGUE", "RELEASED"} or evaluation
    )
    return {
        "policy_id": policy.get("policy_id"),
        "policy_status": status,
        "definition_sha256": policy.get("definition_sha256"),
        "evaluation_role": role,
        "boi_catalogue_enabled": enabled,
        "production_boi_enabled": enabled and status in {"BOI_CATALOGUE", "RELEASED"},
    }


def dump_path(path):
    path = Path(path).resolve()
    for root in EXTERNAL_DUMPS:
        if (path == root or root in path.parents) and not root.parents[1].is_mount():
            raise ValueError(
                f"External output volume is not mounted: {root.parents[1]}"
            )
    roots = (DUMP, *(root.resolve() for root in EXTERNAL_DUMPS if root.is_dir()))
    if any(path == root or root in path.parents for root in roots):
        return path
    raise ValueError(f"Pipeline outputs must stay under {roots}: {path}")


def artifact_path(path, root):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def reference_catalogue_paths(config):
    directory = ROOT / "discPCIe/data/reference-catalogues"
    primary = [
        directory / f"{name}-rows.json"
        for name in ("cumulative", "q1_q17_dr25_koi", "ps", "toi")
    ]
    extra = [
        directory / "kepler-eb-2016-0694ada5b64a.csv",
        directory / "exofop-ctoi-84a829ccae3d.txt",
    ]
    return (
        tuple(ROOT / path for path in config.get("catalogues", primary)),
        tuple(ROOT / path for path in config.get("extra_catalogues", extra)),
    )


def catalogue_search_paths(config):
    spec = config.get("catalogue_search")
    if spec is None:
        return ()
    path = (ROOT / spec["manifest"]).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["manifest_sha256"]:
        raise ValueError("Configured complete catalogue manifest changed")
    manifest = json.loads(path.read_text())
    rows = path.parent / manifest["signals"]["path"]
    if hashlib.sha256(rows.read_bytes()).hexdigest() != manifest["signals"]["sha256"]:
        raise ValueError("Configured complete catalogue rows changed")
    return path, rows


def implementation_hashes():
    paths = sorted((ROOT / "pipeline").rglob("*.py"))
    paths += sorted((ROOT / "pipeline").rglob("*.c"))
    paths += sorted((ROOT / "pipeline").rglob("requirements.txt"))
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def cpu_seconds():
    usage = [
        resource.getrusage(resource.RUSAGE_SELF),
        resource.getrusage(resource.RUSAGE_CHILDREN),
    ]
    return sum(row.ru_utime + row.ru_stime for row in usage)


def process_tree_rss_bytes():
    fields = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, check=False
    ).stdout.split()
    children, rss = {}, {}
    for pid, parent, kib in zip(fields[::3], fields[1::3], fields[2::3]):
        children.setdefault(int(parent), []).append(int(pid))
        rss[int(pid)] = int(kib) * 1024
    total, queue = 0, [os.getpid()]
    while queue:
        pid = queue.pop()
        total += rss.get(pid, 0)
        queue.extend(children.get(pid, []))
    return total


class PeakMemory:
    """Samples the summed resident size of this process tree until stopped."""

    def __init__(self, interval=1.0):
        self.interval, self.peak = interval, 0
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.halt.is_set():
            self.peak = max(self.peak, process_tree_rss_bytes())
            self.halt.wait(self.interval)

    def stop(self):
        self.halt.set()
        self.thread.join()
        return self.peak


def disposition_definition(policy):
    return {
        key: value
        for key, value in policy.items()
        if key not in DISPOSITION_CERTIFICATE_FIELDS
    }


def disposition_definition_sha256(policy):
    return json_sha256(disposition_definition(policy))


def disposition_calibration_parameters(policy):
    return {
        name: policy[name]
        for name in (
            "assessment_rules",
            "pipeline_contract",
            "implementation",
            "outcome_contract",
            "event_matching_contract",
            "lane_gates",
        )
    }


def disposition_evaluation_token(policy, role):
    binding = policy.get("evaluation_binding", {})
    return json_sha256(
        {
            "definition_sha256": disposition_definition_sha256(policy),
            "role": role,
            "binding": binding.get(role),
        }
    )
