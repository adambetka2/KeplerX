"""Compare the vetting records of two runs of the same cohort field by field.

Timing fields and timestamps are ignored, and so are evidence reference hashes:
they cover timing fields of an artifact in the same record, which is compared
itself. Every other difference is reported by its key path, so an optimization
can be checked for output preservation against its paired baseline run.
"""

import argparse
import json
import re
import sys
from pathlib import Path

VOLATILE = {
    "wall_seconds",
    "cpu_seconds",
    "started_utc",
    "retrieved_utc",
    "created_utc",
    "completed_utc",
}
MISSING = "<missing>"


def read(path):
    return json.loads(path.read_text())


def run_text(path, run):
    """Return the file text with every spelling of the run directory replaced."""
    text = path.read_text()
    roots = {str(run), str(run.absolute()), str(run.resolve())}
    for root in sorted(roots, key=len, reverse=True):
        text = text.replace(root, "<run>")
    return text


def evidence_hash(path, key):
    return key == "sha256" and path[-2:-1] == ("evidence_refs",)


def stripped(text, ignored):
    """Return the text with the scalar values of ignored keys blanked."""
    names = "|".join(re.escape(key) for key in sorted(ignored))
    pattern = rf'"(?:{names})"\s*:\s*(?:"[^"]*"|[-+0-9.eE]+|null|true|false)'
    return re.sub(pattern, '""', text)


class Differences:
    def __init__(self, ignored, rtol, atol):
        self.ignored, self.rtol, self.atol = ignored, rtol, atol
        self.material, self.tolerated = {}, {}

    def add(self, source, path, kind, baseline, candidate):
        key = "/".join("[]" if part.startswith("[") else part for part in path)
        relative = None
        bucket = self.material
        if kind == "numeric":
            scale = max(abs(baseline), abs(candidate))
            relative = abs(baseline - candidate) / scale if scale else 0.0
            if abs(baseline - candidate) <= self.atol + self.rtol * scale:
                bucket = self.tolerated
        entry = bucket.setdefault(
            key,
            {
                "kind": kind,
                "count": 0,
                "records": set(),
                "max_relative_difference": 0.0,
                "example": {
                    "record": source,
                    "path": "/".join(path),
                    "baseline": baseline,
                    "candidate": candidate,
                },
            },
        )
        entry["count"] += 1
        entry["records"].add(source)
        if relative is not None:
            entry["max_relative_difference"] = max(
                entry["max_relative_difference"], relative
            )

    def compare(self, source, baseline, candidate, path=()):
        if isinstance(baseline, dict) and isinstance(candidate, dict):
            self.compare_dicts(source, baseline, candidate, path)
        elif isinstance(baseline, list) and isinstance(candidate, list):
            self.compare_lists(source, baseline, candidate, path)
        elif numeric(baseline) and numeric(candidate):
            if baseline != candidate:
                self.add(source, path, "numeric", baseline, candidate)
        elif type(baseline) is not type(candidate):
            self.add(source, path, "type", baseline, candidate)
        elif baseline != candidate:
            self.add(source, path, "value", baseline, candidate)

    def compare_dicts(self, source, baseline, candidate, path):
        for key in sorted(set(baseline) | set(candidate)):
            if key in self.ignored or evidence_hash(path, key):
                continue
            if key in baseline and key in candidate:
                self.compare(source, baseline[key], candidate[key], path + (key,))
            else:
                left = baseline.get(key, MISSING)
                right = candidate.get(key, MISSING)
                self.add(source, path + (key,), "missing", left, right)

    def compare_lists(self, source, baseline, candidate, path):
        if len(baseline) != len(candidate):
            self.add(source, path, "length", len(baseline), len(candidate))
            return
        for index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
            self.compare(source, left, right, path + (f"[{index}]",))

    def report(self, bucket):
        return {
            key: {**entry, "records": len(entry["records"])}
            for key, entry in sorted(bucket.items(), key=lambda item: -item[1]["count"])
        }


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def record_paths(run):
    vetting = run / "vetting"
    paths = sorted((vetting / "sources").glob("*.json"))
    paths += sorted((vetting / "triage").glob("*.json"))
    if (vetting / "results.json").exists():
        paths.append(vetting / "results.json")
    return {str(path.relative_to(run)): path for path in paths}


def stage_walls(run):
    path = run / "vetting" / "stage-timings.json"
    if not path.exists():
        return {}
    return {
        stage["stage"]: stage.get("wall_seconds")
        for stage in read(path)["stages"]
        if stage.get("status") == "COMPLETE"
    }


def compare_runs(baseline, candidate, ignored, rtol, atol):
    left, right = record_paths(baseline), record_paths(candidate)
    ignored = VOLATILE | set(ignored)
    differences = Differences(ignored, rtol, atol)
    identical, compared = 0, 0
    for name in sorted(set(left) & set(right)):
        compared += 1
        left_text = run_text(left[name], baseline)
        right_text = run_text(right[name], candidate)
        if stripped(left_text, ignored) == stripped(right_text, ignored):
            identical += 1
            continue
        before = len(differences.material), len(differences.tolerated)
        differences.compare(name, json.loads(left_text), json.loads(right_text))
        counts = (len(differences.material), len(differences.tolerated))
        if counts == before and not any(
            name in entry["records"]
            for bucket in (differences.material, differences.tolerated)
            for entry in bucket.values()
        ):
            identical += 1
    differing = sorted(
        {
            record
            for bucket in (differences.material, differences.tolerated)
            for entry in bucket.values()
            for record in entry["records"]
        }
    )
    material = sorted(
        {
            record
            for entry in differences.material.values()
            for record in entry["records"]
        }
    )
    return {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "ignored_keys": sorted(VOLATILE | set(ignored)),
        "relative_tolerance": rtol,
        "absolute_tolerance": atol,
        "records": {
            "compared": compared,
            "identical": identical,
            "differing": len(differing),
            "materially_differing": len(material),
            "missing_in_candidate": sorted(set(left) - set(right)),
            "missing_in_baseline": sorted(set(right) - set(left)),
        },
        "material_differences": differences.report(differences.material),
        "tolerated_differences": differences.report(differences.tolerated),
        "stage_wall_seconds": {
            "baseline": stage_walls(baseline),
            "candidate": stage_walls(candidate),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, help="Write the full JSON report here")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--ignore", nargs="*", default=[], help="Extra ignored keys")
    args = parser.parse_args()
    report = compare_runs(
        args.baseline, args.candidate, args.ignore, args.rtol, args.atol
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    records = report["records"]
    print(json.dumps(records))
    for key, entry in list(report["material_differences"].items())[:20]:
        count, records = entry["count"], entry["records"]
        print(f"{entry['kind']:8} {count:8d} in {records:4d} records  {key}")
    tolerated = report["tolerated_differences"]
    if tolerated:
        print(f"{len(tolerated)} key paths differ only within tolerance")
    failed = bool(
        report["material_differences"]
        or records["missing_in_candidate"]
        or records["missing_in_baseline"]
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
