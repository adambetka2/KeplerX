"""Bind a passed frozen experiment to its production disposition policy."""

import argparse
import json
import sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pipeline.deterministic.release_report import bind_release, catalogue

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    names = (
        "run_spec",
        "detector",
        "disposition_calibration",
        "development_replay",
        "final_replay",
    )
    for name in names:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    args = parser.parse_args()
    references = {
        name: {"path": str(path.resolve()), "sha256": catalogue.digest(path)}
        for name in names
        for path in (getattr(args, name),)
    }
    path = bind_release(args.policy, references, args.out)
    print(json.dumps({"status": "RELEASED", "policy": str(path)}))


if __name__ == "__main__":
    main()
