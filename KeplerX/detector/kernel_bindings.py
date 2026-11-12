"""Compile search_kernels.c once per source digest and expose typed ctypes calls."""

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).with_name("search_kernels.c")
SORT_SOURCE = Path(__file__).with_name("stable_ranking.cpp")
# IEEE arithmetic only: no fast-math, no implicit multiply-add contraction.
FLAGS = ("-O3", "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC")
_LIBRARY = None


def _pointer(dtype):
    return np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")


def library(cache=None):
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    key = hashlib.sha256(
        SOURCE.read_bytes()
        + SORT_SOURCE.read_bytes()
        + " ".join((*FLAGS, "-std=c++17")).encode()
    ).hexdigest()
    default = Path(__file__).resolve().parents[1] / "runtime" / "cache"
    root = Path(cache or os.environ.get("KEPLERX_KERNEL_CACHE", default))
    path = root / "compiled-kernels" / f"kernels-{key[:20]}.dylib"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        subprocess.run(["cc", *FLAGS, "-o", str(temporary), str(SOURCE)], check=True)
        os.replace(temporary, path)
    sort_path = path.with_name(f"ranking-{key[:20]}.dylib")
    if not sort_path.exists():
        temporary = sort_path.with_name(f"{sort_path.name}.{os.getpid()}.tmp")
        subprocess.run(
            ["c++", *FLAGS, "-std=c++17", "-o", str(temporary), str(SORT_SOURCE)],
            check=True,
        )
        os.replace(temporary, sort_path)
    lib = ctypes.CDLL(str(path))
    i64, i32, f64 = ctypes.c_int64, ctypes.c_int32, ctypes.c_double
    lib.rank_library = ctypes.CDLL(str(sort_path))
    lib.rank_pairs = lib.rank_library.rank_pairs
    lib.rank_pairs.restype = ctypes.c_int
    lib.rank_pairs.argtypes = [
        i64, _pointer(np.float64), _pointer(np.int32), _pointer(np.float64),
    ]  # fmt: skip
    lib.v3_fold_search.restype = ctypes.c_int
    lib.v3_fold_search.argtypes = [
        i64, _pointer(np.float64), i32, _pointer(np.float64), _pointer(np.float64),
        i64, _pointer(np.float64), _pointer(np.int32), _pointer(np.int32),
        _pointer(np.int32), f64, f64, i64, _pointer(np.float64),
        _pointer(np.int32), _pointer(np.int32),
    ]  # fmt: skip
    lib.v3_running_kth.restype = ctypes.c_int
    lib.v3_running_kth.argtypes = [
        i64, _pointer(np.int32), i64, _pointer(np.float64), i64, _pointer(np.float64),
    ]  # fmt: skip
    lib.v3_convolve_same.restype = ctypes.c_int
    lib.v3_convolve_same.argtypes = [
        i64, _pointer(np.float64), _pointer(np.float64), i64, i64, i64,
        _pointer(np.int64), _pointer(np.float64),
    ]  # fmt: skip
    lib.v3_ses_sums.restype = ctypes.c_int
    lib.v3_ses_sums.argtypes = [
        i64, i64, _pointer(np.float64), _pointer(np.float64), _pointer(np.float64),
        i64, _pointer(np.int64), _pointer(np.float64), _pointer(np.float64),
    ]  # fmt: skip
    lib.v3_mad_pairs.restype = ctypes.c_int
    lib.v3_mad_pairs.argtypes = [
        i64, _pointer(np.int64), _pointer(np.int64), i64, _pointer(np.float64),
        _pointer(np.int32), _pointer(np.float64), ctypes.c_void_p,
        _pointer(np.float64),
    ]  # fmt: skip
    lib.v3_single_events.restype = None
    lib.v3_single_events.argtypes = [
        i64, _pointer(np.float64), _pointer(np.float64), i64, _pointer(np.float64),
        _pointer(np.int32), _pointer(np.int32), _pointer(np.int32),
        _pointer(np.float64), _pointer(np.float64), f64, f64, _pointer(np.int8),
        _pointer(np.int64), _pointer(np.int64), _pointer(np.float64), _pointer(np.int8),
    ]  # fmt: skip
    lib.v3_bins_mes.restype = None
    lib.v3_bins_mes.argtypes = [
        i64, _pointer(np.float64), _pointer(np.float64), _pointer(np.float64), i64,
        _pointer(np.float64), _pointer(np.int32), _pointer(np.int32),
        _pointer(np.int32), f64, _pointer(np.int32), i64, f64, _pointer(np.float64),
    ]  # fmt: skip
    lib.v3_swt_level.restype = ctypes.c_int
    lib.v3_swt_level.argtypes = [
        i64, _pointer(np.float64), i64, _pointer(np.float64), i64, _pointer(np.float64),
    ]  # fmt: skip
    lib.v3_period_sampling.restype = ctypes.c_int64
    lib.v3_period_sampling.argtypes = [
        f64,
        f64,
        f64,
        f64,
        f64,
        _pointer(np.float64),
        i64,
    ]
    lib.digest = key
    lib.direct_convolution = _direct_convolution_matches(lib)
    _LIBRARY = lib
    return lib


def openblas_core():
    path = Path(np.__file__).parent / ".dylibs" / "libopenblas64_.0.dylib"
    try:
        blas = ctypes.CDLL(str(path))
        blas.openblas_get_corename64_.restype = ctypes.c_char_p
        return blas.openblas_get_corename64_().decode()
    except (OSError, AttributeError):
        return None


def _direct_convolution_matches(lib):
    """Use the compiled direct convolution only where it reproduces NumPy."""
    if openblas_core() != "armv8":
        return False
    rng = np.random.default_rng(20261027)
    for n in (12, 13, 16, 64, 255, 256, 1000, 2048):
        a = rng.normal(size=n) * 10.0 ** rng.integers(-3, 4, n)
        kernel = np.zeros(n)
        lo = int(rng.integers(0, n // 2))
        hi = int(rng.integers(lo + 1, n + 1))
        kernel[lo:hi] = rng.normal(size=hi - lo)
        need = np.arange(n, dtype=np.int64)
        out = np.empty(n)
        lib.v3_convolve_same(n, a, kernel, lo, hi, n, need, out)
        if not np.array_equal(out, np.convolve(a, kernel[::-1].copy(), "same")):
            return False
    return True


def convolve_same(a, kernel, support, need):
    """np.convolve(a, kernel[::-1], "same")[need]; kernel is NumPy's ap2."""
    lib = library()
    out = np.empty(len(need))
    lib.v3_convolve_same(
        len(a), np.ascontiguousarray(a, dtype=np.float64), kernel, support[0],
        support[1], len(need), need, out,
    )  # fmt: skip
    return out


def ses_sums(flux, sig2, templates, need):
    """Direct calculate_SES convolutions and ordered level sums in one call."""
    num, den = np.empty(len(need)), np.empty(len(need))
    result = library().v3_ses_sums(
        len(sig2), sig2.shape[1], np.ascontiguousarray(flux), sig2,
        np.ascontiguousarray(templates), len(need), need, num, den,
    )  # fmt: skip
    if result:
        raise MemoryError("Could not allocate direct whitening workspace")
    return num, den


def mad_pairs(flux, levels, windows):
    """calc_var_stat's weights (1 / MAD**2) of one segment for (level, window points) pairs."""
    lib = library()
    flux = np.ascontiguousarray(flux, dtype=np.float64)
    n = flux.shape[1]
    levels = np.ascontiguousarray(levels, dtype=np.int64)
    windows = np.ascontiguousarray(windows, dtype=np.int64)
    rank = np.empty(flux.shape, dtype=np.int32)
    ordered = np.empty(flux.shape)
    for i in np.unique(levels[windows < 15001]):
        rank[i], ordered[i] = ranking(np.abs(flux[i]))
    sig2 = np.empty((len(levels), n))
    status = lib.v3_mad_pairs(
        len(levels), levels, windows, n, flux, rank, ordered,
        ctypes.cast(lib.rank_pairs, ctypes.c_void_p), sig2,
    )  # fmt: skip
    if status < 0:
        raise MemoryError("Could not allocate the whitening weights workspace")
    return None if status else sig2


def decisive_cadences(ses, threshold, frac):
    """Per duration, the cadences that can decide a single-event veto."""
    usable = np.zeros(len(ses), dtype=np.int8)
    lists = []
    for d, row in enumerate(ses):
        if threshold >= 0.0 and frac >= 0.0 and not np.isnan(row).any():
            usable[d] = 1
            lists.append(np.flatnonzero(row > threshold))
            lists.append(np.flatnonzero(row > frac * threshold * (1.0 - 1e-12)))
        else:
            lists.extend([np.empty(0, dtype=np.int64)] * 2)
    sizes = np.array([len(x) for x in lists], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    cadences = (
        np.concatenate(lists).astype(np.int64) if lists else np.empty(0, np.int64)
    )
    return usable, starts, np.ascontiguousarray(cadences)


def single_events(t0_time, ses, periods, nbins, bins, durations, tdurs, mes,
                  P_lim=50.0, frac=0.7, threshold=None):  # fmt: skip
    """Peak locations and single-event vetoes for many winning periods."""
    lib = library()
    count = len(periods)
    ses = np.ascontiguousarray(ses, dtype=np.float64)
    peaks = np.empty(count)
    vetoed = np.empty(count, dtype=np.int8)
    if threshold is None:
        threshold = -np.inf
    usable, starts, cadences = decisive_cadences(ses, float(threshold), float(frac))
    lib.v3_single_events(
        len(t0_time), t0_time, ses, count,
        np.ascontiguousarray(periods, dtype=np.float64),
        np.ascontiguousarray(nbins, dtype=np.int32),
        np.ascontiguousarray(bins, dtype=np.int32),
        np.ascontiguousarray(durations, dtype=np.int32),
        np.ascontiguousarray(tdurs, dtype=np.float64),
        np.ascontiguousarray(mes, dtype=np.float64), float(P_lim), float(frac),
        usable, starts, cadences, peaks, vetoed,
    )  # fmt: skip
    return peaks, vetoed


def fold_search(t, num, den, periods, nbins, lo, hi, n_trans, threshold, budget):
    lib = library()
    t = np.ascontiguousarray(t, dtype=np.float64)
    num = np.ascontiguousarray(num, dtype=np.float64)
    den = np.ascontiguousarray(den, dtype=np.float64)
    periods = np.ascontiguousarray(periods, dtype=np.float64)
    count = len(periods)
    mes = np.empty(count)
    duration = np.empty(count, dtype=np.int32)
    position = np.empty(count, dtype=np.int32)
    status = lib.v3_fold_search(
        len(t), t, num.shape[0], num, den, count, periods,
        np.ascontiguousarray(nbins, dtype=np.int32),
        np.ascontiguousarray(lo, dtype=np.int32),
        np.ascontiguousarray(hi, dtype=np.int32),
        float(n_trans), float(threshold), int(budget), mes, duration, position,
    )  # fmt: skip
    if status:
        raise MemoryError("fold search allocation failed")
    return mes, duration, position


def ranking(values):
    """Stable sort order of values: positions' ranks and the sorted values."""
    if (
        type(values) is np.ndarray
        and values.dtype == np.float64
        and values.ndim == 1
        and values.flags.c_contiguous
        and values.flags.aligned
    ):
        rank = np.empty(len(values), dtype=np.int32)
        ordered = np.empty(len(values))
        if library().rank_pairs(len(values), values, rank, ordered) == 0:
            return rank, ordered
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), dtype=np.int32)
    rank[order] = np.arange(len(values), dtype=np.int32)
    return rank, np.ascontiguousarray(values[order])


def running_median(values, window, ranks=None):
    """Exact utils.running_median_insort(values, window) for NaN-free values."""
    lib = library()
    values = np.ascontiguousarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        np.pad(values, (window // 2, 0), mode="reflect")  # same empty-input error
    rank, ordered = ranks if ranks is not None else ranking(values)
    out = np.empty(n)
    if lib.v3_running_kth(n + window // 2, rank, n, ordered, window, out):
        raise MemoryError("running median work space")
    return out


_DB2 = {}


def _db2_filters():
    """The rescaled db2 filters pywt.swt(..., norm=True) passes to its C code."""
    if not _DB2:
        import pywt
        from pywt._swt import _rescale_wavelet_filterbank

        wavelet = _rescale_wavelet_filterbank(pywt.Wavelet("db2"), 1 / np.sqrt(2))
        _DB2["lo"] = np.ascontiguousarray(wavelet.dec_lo, dtype=np.float64)
        _DB2["hi"] = np.ascontiguousarray(wavelet.dec_hi, dtype=np.float64)
    return _DB2["lo"], _DB2["hi"]


def ocwt(x, max_level=14, upstream=None):
    """ses.ocwt's stationary db2 wavelet transform, level by level; other cases go to upstream's ocwt."""
    x = np.asarray(x)
    level = min(max_level, np.log2(len(x)) - 1) if len(x) else -1
    if (
        x.dtype != np.float64
        or x.ndim != 1
        or len(x) % 2
        or not float(level).is_integer()
        or level < 1
        or (1 << int(level)) * 2 > len(x)
        or len(x) % (1 << int(level))
        or not np.isfinite(x).all()
    ):
        return upstream(x, max_level=max_level)
    lib = library()
    low, high = _db2_filters()
    data = np.ascontiguousarray(x)
    details = []
    for step in range(1, int(level) + 1):
        detail, approximation = np.empty(len(x)), np.empty(len(x))
        fstep = 1 << (step - 1)
        if lib.v3_swt_level(
            len(x), data, len(high), high, fstep, detail
        ) or lib.v3_swt_level(len(x), data, len(low), low, fstep, approximation):
            return upstream(x, max_level=max_level)
        details.append(detail)
        data = approximation
    return np.array([data] + details[::-1])[::-1]


def period_sampling(p_min, limit, baseline, d_leastsq, rho_star):
    """kep_period_sampling's periods from p_min while below limit (search_kernels.c)."""
    args = (
        float(p_min),
        float(limit),
        float(baseline),
        float(d_leastsq),
        float(rho_star),
    )
    count = library().v3_period_sampling(*args, np.empty(1), 0)
    out = np.empty(count)
    library().v3_period_sampling(*args, out, count)
    return out
