"""Experimental batman orbit kernel; the repaired Kepler solver stays unchanged.

Compile C to LLVM on stdout, then JIT in memory. No build directory, wheel,
installed-runtime mutation or persistent machine-code cache is needed.
"""

import contextlib
import ctypes
import functools
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import numpy as np

RSKY_SHA256 = "0fb9b316dc8b72b4556157a96aa761cf93554691e1ca5f10d3453caf89501bf4"
SOURCE = Path(__file__).with_name("native_orbits.c")


@functools.lru_cache(maxsize=1)
def compiled():
    from llvmlite import binding as llvm

    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise RuntimeError("Native orbit experiment is tested on the arm64 Mac only")
    version = subprocess.run(
        ["/usr/bin/clang", "--version"], text=True, capture_output=True, check=True
    ).stdout
    if (
        llvm.llvm_version_info != (22, 1, 0)
        or version.splitlines()[0] != "Apple clang version 17.0.0 (clang-1700.6.4.2)"
    ):
        raise RuntimeError("Requalify the native kernel for this Clang/LLVM toolchain")
    command = [
        "/usr/bin/clang",
        "-O3",
        "-S",
        "-emit-llvm",
        "-x",
        "c",
        "-",
        "-o",
        "-",
        "-target",
        "arm64-apple-macosx11.0.0",
        "-mcpu=generic",
        "-mno-stack-arg-probe",
    ]
    ir = subprocess.run(
        command, input=SOURCE.read_text(), text=True, capture_output=True, check=True
    ).stdout
    # Apple Clang emits this even with -mno-stack-arg-probe. LLVM 22 cannot
    # lower it; this numerical kernel has no dynamic or large stack allocations.
    ir = ir.replace(' "probe-stack"="__chkstk_darwin"', "")
    if '"probe-stack"' in ir:
        raise RuntimeError("Unexpected Clang stack-probe ABI")
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    library = ctypes.CDLL(None)
    for name in ("sin", "cos", "atan", "tan", "sqrt", "remainder", "fabs"):
        llvm.add_symbol(
            name, ctypes.cast(getattr(library, name), ctypes.c_void_p).value
        )
    target = llvm.Target.from_default_triple().create_target_machine(
        cpu="generic", opt=3
    )
    module = llvm.parse_assembly(ir)
    module.triple = llvm.get_default_triple()
    module.data_layout = str(target.target_data)
    module.verify()
    engine = llvm.create_mcjit_compiler(module, target)
    engine.finalize_object()
    signature = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        *([ctypes.c_double] * 6),
        ctypes.c_int,
        ctypes.c_int,
    )
    function = signature(engine.get_function_address("sky"))
    info = dict(
        clang_version=version,
        llvm_version=list(llvm.llvm_version_info),
        command=command,
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        ir_sha256=hashlib.sha256(ir.encode()).hexdigest(),
        target_layout=module.data_layout,
    )
    return function, engine, info


def provenance():
    return dict(compiled()[2])


def sky(
    time,
    tc,
    period,
    axis,
    inclination,
    eccentricity,
    omega,
    transit,
    threads=1,
    *,
    true_anomaly=False,
):
    """Same arguments/units as batman._rsky; float64 contiguous arrays, one thread."""
    time = np.asarray(time)
    if time.ndim != 1 or time.dtype != np.float64 or not time.flags.c_contiguous:
        raise ValueError("Supply a contiguous float64 time vector")
    if threads != 1 or time.size > np.iinfo(np.int32).max:
        raise ValueError(
            "Native orbit experiment uses one thread and int32 cadence counts"
        )
    if (
        not np.isfinite([tc, period, axis, inclination, eccentricity, omega]).all()
        or period <= 0
    ):
        raise ValueError("Supply finite orbital parameters and a positive period")
    if not 0 <= eccentricity < 1:
        raise ValueError("Elliptic eccentricity must be finite and in [0, 1)")
    result = np.empty_like(time)
    function, _, _ = compiled()
    function(
        time.ctypes.data,
        result.ctypes.data,
        time.size,
        tc,
        period,
        axis,
        inclination,
        eccentricity,
        omega,
        transit,
        int(true_anomaly),
    )
    return result


@contextlib.contextmanager
def accelerated_native():
    """Use the experimental orbit kernel in a dedicated process, then restore batman."""
    from batman import _rsky

    if importlib.metadata.version("batman-package") != "2.5.3+spacelab.2":
        raise RuntimeError("Require the reviewed batman 2.5.3+spacelab.2 runtime")
    if hashlib.sha256(Path(_rsky.__file__).read_bytes()).hexdigest() != RSKY_SHA256:
        raise RuntimeError(
            "Installed batman orbit binary differs from the reviewed Mac wheel"
        )
    compiled()
    saved = _rsky._rsky
    _rsky._rsky = sky
    try:
        yield
    finally:
        _rsky._rsky = saved
