"""Compile vetter/residual_kernels.c once per source digest and expose typed calls."""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).with_name("residual_kernels.c")
# IEEE arithmetic only: no fast-math, no implicit multiply-add contraction.
FLAGS = ("-O3", "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC")
TRAP, BOX, SPSD, SINE = 0, 1, 2, 3
_LIBRARY = None


def library(cache=None):
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    key = hashlib.sha256(SOURCE.read_bytes() + " ".join(FLAGS).encode()).hexdigest()
    default = Path(__file__).resolve().parents[1] / "runtime" / "cache"
    root = Path(cache or os.environ.get("KEPLERX_KERNEL_CACHE", default))
    path = root / "compiled-kernels" / f"vetter-{key[:20]}.dylib"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        subprocess.run(["cc", *FLAGS, "-o", str(temporary), str(SOURCE)], check=True)
        os.replace(temporary, path)
    lib = ctypes.CDLL(str(path))
    lib.v3_residual.restype = ctypes.c_int
    lib.v3_residual.argtypes = [
        ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]  # fmt: skip
    lib.v3_residual_sums.restype = ctypes.c_int
    lib.v3_residual_sums.argtypes = [
        ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
    ]  # fmt: skip
    lib.v3_direct_gradient.restype = ctypes.c_int
    lib.v3_direct_gradient.argtypes = [
        ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
    ]  # fmt: skip
    lib.v3_value_gradient.restype = ctypes.c_int
    lib.v3_value_gradient.argtypes = [
        ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]  # fmt: skip
    lib.v3_nonlinear_ld.restype = ctypes.c_int
    lib.v3_nonlinear_ld.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]  # fmt: skip
    lib.v3_nonlinear_ld_fast.restype = ctypes.c_int
    lib.v3_nonlinear_ld_fast.argtypes = lib.v3_nonlinear_ld.argtypes
    lib.digest = key
    _LIBRARY = lib
    return lib


def _gauss_tables(largest=64):
    """cos(phi_j) and sin(phi_j) w_j pi/2 of every Legendre rule in phi up to `largest` nodes."""
    offsets = np.zeros(largest + 1, dtype=np.int32)
    cosines, weights = [], []
    for m in range(1, largest + 1):
        offsets[m] = sum(len(c) for c in cosines)
        z, w = np.polynomial.legendre.leggauss(m)
        phi = np.pi * (z + 1.0) / 2.0
        cosines.append(np.cos(phi))
        weights.append(np.sin(phi) * w * np.pi / 2.0)
    return np.concatenate(cosines), np.concatenate(weights), offsets


_GAUSS = _gauss_tables()


def nonlinear_ld(ds, rp, c1, c2, c3, c4, interpolate=False):
    """batman._nonlinear_ld's flux, as the accurate integral (residual_kernels.c)."""
    ds = np.ascontiguousarray(ds, dtype=np.float64)
    flux = np.empty(len(ds))
    lib = library()
    (lib.v3_nonlinear_ld_fast if interpolate else lib.v3_nonlinear_ld)(
        len(ds), ds.ctypes.data, float(abs(rp)), float(c1), float(c2), float(c3),
        float(c4), *(table.ctypes.data for table in _GAUSS), flux.ctypes.data,
    )  # fmt: skip
    return flux


# np.sum reduces float64 arrays in blocks of this many elements.
ONE_BLOCK = 8192


class Window:
    """One fit's (x, data, err) arrays, with their addresses taken once."""

    __slots__ = (
        "n", "x", "data", "err", "pointers", "parameters", "call", "scratch", "sums",
        "call_sums", "addresses",
    )  # fmt: skip

    def __init__(self, x, data, err):
        self.x = np.ascontiguousarray(x, dtype=np.float64)
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.err = np.ascontiguousarray(err, dtype=np.float64)
        self.n = len(self.x)
        if len(self.data) != self.n or len(self.err) != self.n:
            raise ValueError("x, data and err differ in length")
        self.pointers = (
            self.x.ctypes.data,
            self.data.ctypes.data,
            self.err.ctypes.data,
        )
        self.parameters = np.empty(7)
        self.call = library().v3_residual
        self.call_sums = library().v3_residual_sums
        self.scratch = (np.empty(self.n), np.empty(self.n))
        self.sums = np.empty(2)
        # Addresses of the parameter vector, scratch arrays and sums, taken once for every call.
        self.addresses = (
            self.parameters.ctypes.data, self.scratch[0].ctypes.data,
            self.scratch[1].ctypes.data, self.sums.ctypes.data,
        )  # fmt: skip

    def residual(self, kind, values):
        """(residual array, model has NaN) for the parameter sequence `values`."""
        p = self.parameters
        p[: len(values)] = values
        out = np.empty(self.n)
        nan = self.call(kind, self.n, *self.pointers, p.ctypes.data, out.ctypes.data)
        return out, bool(nan)

    def residual_sums(self, kind, values):
        """(np.sum(residual), (residual*residual).sum(), model has NaN)."""
        self.parameters[: len(values)] = values
        nan = self.call_sums(kind, self.n, *self.pointers, *self.addresses)
        return self.sums[0], self.sums[1], bool(nan)


class Gradient:
    """One direct fit's fixed parameter mapping and reusable native workspace."""

    def __init__(self, window, kind, names, variables, lower, upper, template):
        self.window = window
        self.arrays = (
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
            np.array(
                [variables.index(n) if n in variables else -1 for n in names], np.int32
            ),
            np.array([template[n] if n not in variables else 0.0 for n in names]),
        )
        self.prefix = (kind, window.n, *window.pointers, len(variables))
        self.arguments = (
            self.arrays[0].ctypes.data, self.arrays[1].ctypes.data, len(names),
            self.arrays[2].ctypes.data, self.arrays[3].ctypes.data,
            window.scratch[0].ctypes.data, window.scratch[1].ctypes.data,
        )  # fmt: skip
        self.last = np.empty(len(variables))
        self.call = library().v3_direct_gradient
        self.joint = library().v3_value_gradient

    def __call__(self, x, h, f0):
        out = np.empty(len(x))
        status = self.call(
            *self.prefix, x.ctypes.data, h.ctypes.data, float(f0), *self.arguments,
            out.ctypes.data, self.last.ctypes.data,
        )  # fmt: skip
        return None if status else out

    def value_gradient(self, x):
        out = np.empty(len(x) + 2)
        status = self.joint(
            *self.prefix, x.ctypes.data, *self.arguments,
            out.ctypes.data, self.last.ctypes.data,
        )  # fmt: skip
        return None if status else (out[0], out[1:-1], out[-1])
