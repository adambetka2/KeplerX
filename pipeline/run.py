"""Run a pipeline stage with its local and shared modules available."""

import argparse
import os
import runpy
import sys
from pathlib import Path

from runtime import SERIAL_BLAS

ROOT = Path(__file__).resolve().parent
STAGES = {
    "deterministic": "deterministic/run.py",
    "vetting": "vetting/vetting.py",
    "evidence": "vetting/vetting_evidence.py",
    "pixels": "vetting/vetting_pixels.py",
    "flux": "vetting/vetting_flux.py",
    "signals": "vetting/vetting_signals.py",
    "orbital": "vetting/vetting_periodic.py",
    "instrument": "vetting/vetting_systematics.py",
    "host": "vetting/vetting_host.py",
    "disposition": "vetting/vetting_disposition.py",
    "calibrate": "vetting/calibrate.py",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ.update(SERIAL_BLAS)
    program = ROOT / STAGES[args.stage]
    sys.path[:0] = [
        str(program.parent),
        str(ROOT / "photometry"),
    ]
    sys.argv = [str(program), *args.arguments]
    runpy.run_path(str(program), run_name="__main__")
