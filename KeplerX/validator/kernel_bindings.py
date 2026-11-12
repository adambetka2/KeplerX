"""Compile transit_likelihood.c once per source digest and expose typed ctypes calls."""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).with_name("transit_likelihood.c")
# IEEE arithmetic only; Accelerate's vForce supplies the batched logarithms and arc cosines.
FLAGS = ("-O3", "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC")
LINK = ("-framework", "Accelerate")
_LIBRARY = None


def _pointer(dtype):
    return np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")


def source_digest():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def library_path(cache=None):
    key = hashlib.sha256(
        SOURCE.read_bytes() + " ".join(FLAGS + LINK).encode()
    ).hexdigest()
    default = Path(__file__).resolve().parents[1] / "runtime" / "cache"
    root = Path(cache or os.environ.get("KEPLERX_KERNEL_CACHE", default))
    return root / "compiled-kernels" / f"validator-{key[:20]}.dylib"


def library(cache=None):
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    path = library_path(cache)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        subprocess.run(
            ["cc", *FLAGS, "-o", str(temporary), str(SOURCE), *LINK], check=True
        )
        os.replace(temporary, path)
    lib = ctypes.CDLL(str(path))
    i64, i32, f64 = ctypes.c_int64, ctypes.c_int32, ctypes.c_double
    f64p, i64p = _pointer(np.float64), _pointer(np.int64)
    lib.validator_loglike.restype = f64
    lib.validator_loglike.argtypes = [
        i64, f64p, f64p, f64p, i32, f64p, f64, f64p, f64p, f64p, i64p,
    ]  # fmt: skip
    # The same function with raw addresses, for the per-proposal call.
    lib.validator_loglike_address = lib.validator_loglike
    address = ctypes.CFUNCTYPE(
        f64, i64, *([ctypes.c_void_p] * 3), i32, ctypes.c_void_p, f64,
        *([ctypes.c_void_p] * 4),
    )  # fmt: skip
    lib.validator_loglike_address = address(("validator_loglike", lib))
    lib.validator_model.restype = ctypes.c_int
    lib.validator_model.argtypes = [i64, f64p, i32, f64p, f64, f64p, f64p, i64p, i32]
    lib.validator_loglike_rows.restype = None
    lib.validator_loglike_rows.argtypes = [
        i64, f64p, i64, f64p, f64p, f64p, i32, f64p, f64, f64p, f64p, f64p, i64p,
    ]  # fmt: skip
    _LIBRARY = lib
    return lib
