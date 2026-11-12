"""GPU screening of the period search, resolved exactly on the CPU."""

import contextlib
import ctypes
import fcntl
import hashlib
import itertools
import os
import subprocess
from pathlib import Path

import numpy as np

from detector import kernel_bindings

HERE = Path(__file__).parent
SHADER = HERE / "gpu_screening.metal"
HOST = HERE / "gpu_screening_host.m"
CHUNK = 16  # gpu_screening_host.m MAX_DURATIONS: one window_fold pipeline per duration count
GRID_DAYS = 0.004
# Bin edges in double-float arithmetic: error below 2**-46 * max(t) days.
EDGE_DAYS = 2e-10
TILE = np.dtype(
    [
        ("period", "<i4"), ("lanes", "<i4"), ("bins", "<i4"), ("b0", "<i4"),
        ("d0", "<i4"), ("m", "<i4"), ("partial", "<i4"), ("tiles", "<i4"),
    ]
)  # fmt: skip
PERIOD = np.dtype(
    [
        ("nb", "<i4"), ("epochs", "<i4"), ("ph", "<f4"), ("pl", "<f4"),
        ("wh", "<f4"), ("wl", "<f4"),
    ]
)  # fmt: skip
GROUP = np.dtype(
    [
        ("period", "<i4"), ("count", "<i4"), ("rows", "<i4"), ("d0", "<i4"),
        ("cell", "<i4"), ("m", "<i4"), ("pad0", "<i4"), ("pad1", "<i4"),
    ]
)  # fmt: skip
# Window tiles: lanes x bins x (1 + 2 m) histogram integers within WINDOW_INTS, or one lane.
WINDOW_INTS = 3840
WINDOW_LANES = 16
WINDOW_THREADS = 256
WINDOW_BINS = 32
WINDOW_BINS_LONG = 128
WINDOW_LONG_BINS = 3072
# Short-duration runs go to the fold kernel, in groups whose histograms fit FOLD_INTS.
FOLD_DURATIONS = 16
FOLD_INTS = 7680
FOLD_GROUP = 32
FOLD_THREADS = 512
# Runs whose fold groups would hold fewer periods go to window tiles.
FOLD_MIN_PERIODS = 2
# (numerator, denominator) are stored as rint(x / step), a power-of-two step, magnitudes up to INT_LIMIT.
INT_LIMIT = 2**19
MAX_EPOCHS = 4096
_STATE = {}


def _library():
    if "lib" in _STATE:
        return _STATE["lib"]
    key = hashlib.sha256(HOST.read_bytes()).hexdigest()
    root = Path(
        os.environ.get("KEPLERX_KERNEL_CACHE", HERE.parent / "runtime" / "cache")
    )
    path = root / "compiled-kernels" / f"gpu-{key[:20]}.dylib"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        subprocess.run(
            ["clang", "-O2", "-fobjc-arc", "-dynamiclib", "-framework", "Metal",
             "-framework", "Foundation", "-o", str(temporary), str(HOST)],
            check=True,
        )  # fmt: skip
        os.replace(temporary, path)
    lib = ctypes.CDLL(str(path))
    lib.v3_gpu_open.restype = ctypes.c_int
    lib.v3_gpu_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.v3_gpu_max_threads.restype = ctypes.c_int
    lib.v3_gpu_max_threads.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.v3_gpu_screen.restype = ctypes.c_int
    i32, i64 = ctypes.c_int32, ctypes.c_int64
    f32, ptr = ctypes.c_float, ctypes.c_void_p
    lib.v3_gpu_screen.argtypes = [
        i64, i32, ptr, i32, ptr, f32, i32, ptr, i64, ptr, i32, ptr, i32, i32, ptr, i32,
        i32, ptr, ptr, ptr, ptr, i32, ptr, i32, i32, ptr, ptr, i32, i32, i32, ptr, ptr,
        ptr, ptr, f32, f32, f32, ctypes.POINTER(ctypes.c_double),
    ]  # fmt: skip
    message = ctypes.create_string_buffer(4096)
    status = lib.v3_gpu_open(SHADER.read_bytes(), message, len(message))
    _STATE["lib"] = lib if status == 0 else None
    _STATE["device"] = message.value.decode()
    _STATE["digest"] = hashlib.sha256(
        SHADER.read_bytes() + HOST.read_bytes()
    ).hexdigest()
    return _STATE["lib"]


@contextlib.contextmanager
def claim():
    """Queue one screening chunk, keeping CPU cores available for other stages."""
    root = Path(
        os.environ.get("KEPLERX_KERNEL_CACHE", HERE.parent / "runtime" / "cache")
    )
    root.mkdir(parents=True, exist_ok=True)
    slots = int(os.environ.get("KEPLERX_GPU_SLOTS", "3"))
    names = ["gpu.lock"] + [f"gpu.lock.{i}" for i in range(1, slots)]
    handles = [open(root / name, "w") for name in names]
    try:
        held = None
        for handle in handles:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = handle
                break
            except BlockingIOError:
                continue
        if held is None:
            held = handles[os.getpid() % len(handles)]
            fcntl.flock(held, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)
    finally:
        for handle in handles:
            handle.close()


def available():
    return os.environ.get("KEPLERX_GPU", "1") != "0" and _library() is not None


def describe():
    _library()
    return {"device": _STATE.get("device"), "sources_sha256": _STATE.get("digest")}


def _split(values):
    high = values.astype(np.float32)
    return high, (values - high.astype(np.float64)).astype(np.float32)


def _plain(num, den, threshold):
    """Inputs for which every float32 bound is valid."""
    if not threshold >= 0.0:
        return False
    if not (np.isfinite(num).all() and np.isfinite(den).all() and (den > 0).all()):
        return False
    magnitude = np.abs(num[num != 0])
    return (
        (magnitude.size == 0 or (magnitude.min() > 1e-30 and magnitude.max() < 1e30))
        and den.min() > 1e-30
        and den.max() < 1e30
    )


def _runs(lo, hi):
    """Runs of consecutive periods with equal admitted durations (lo, hi)."""
    count = len(lo)
    change = np.ones(count, dtype=bool)
    change[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    start = np.flatnonzero(change)
    return start, np.diff(np.append(start, count))


def _groups(run_start, run_length, size_run):
    """Split each run into consecutive groups of at most size_run periods."""
    groups_run = (run_length + size_run - 1) // size_run
    run_of = np.repeat(np.arange(len(run_start)), groups_run)
    index = np.arange(len(run_of)) - np.repeat(
        np.cumsum(groups_run) - groups_run, groups_run
    )
    first = run_start[run_of] + index * size_run[run_of]
    size = np.minimum(size_run[run_of], run_start[run_of] + run_length[run_of] - first)
    return run_of, first, size


def _classes(keys):
    """Start of each run of equal keys in a sorted key array, and its key."""
    change = np.ones(len(keys), dtype=bool)
    change[1:] = keys[1:] != keys[:-1]
    begin = np.flatnonzero(change)
    return np.append(begin, len(keys)).astype(np.int32), begin


def _fold_layout(run_start, run_length, nbins, lo, widths, cell_start):
    """Fold groups over runs (cells written by the fold kernel), by width."""
    rows = nbins[run_start].astype(np.int64)
    width = widths[run_start]
    size_run = np.minimum(FOLD_GROUP, FOLD_INTS // (rows * (1 + 2 * width)))
    run_of, first, size = _groups(run_start, run_length, size_run)
    groups = np.zeros(len(first), dtype=GROUP)
    groups["period"] = first
    groups["count"] = size
    # Periods descend, so each group's first period has the most bins.
    groups["rows"] = nbins[first]
    groups["d0"] = lo[first]
    groups["cell"] = cell_start[first]
    groups["m"] = width[run_of]
    groups = groups[np.argsort(groups["m"], kind="stable")]
    starts, begin = _classes(groups["m"])
    return groups, (starts, groups["m"][begin].astype(np.int32))


def _layout(order, nbins, lo, hi, caps):
    """Fold groups for runs with at most FOLD_DURATIONS durations."""
    widths = (hi - lo).astype(np.int64)
    cell_start = np.cumsum(widths) - widths
    run_start, run_length = _runs(lo, hi)
    run_width = widths[run_start]
    fold = (
        (run_width >= 1)
        & (run_width <= FOLD_DURATIONS)
        & (
            FOLD_INTS // (nbins[run_start].astype(np.int64) * (1 + 2 * run_width))
            >= FOLD_MIN_PERIODS
        )
    )
    groups, group_classes = _fold_layout(
        run_start[fold], run_length[fold], nbins, lo, widths, cell_start
    )
    keep = (run_width > 0) & ~fold
    run_start, run_length = run_start[keep], run_length[keep]
    # A run's chunks share the tile shape of its first (longest) period and first chunk.
    rows = nbins[run_start].astype(np.int64)
    bins_run = np.where(rows >= WINDOW_LONG_BINS, WINDOW_BINS_LONG, WINDOW_BINS)
    bins_run = np.minimum(bins_run, (rows + 31) // 32 * 32)
    m_run = np.minimum(widths[run_start], CHUNK)
    lanes_run = np.clip(WINDOW_INTS // (bins_run * (1 + 2 * m_run)), 1, WINDOW_LANES)
    run_of, first_period, lanes = _groups(run_start, run_length, lanes_run)
    bins = bins_run[run_of]
    spans = (nbins[first_period].astype(np.int64) + bins - 1) // bins
    width = widths[first_period]
    chunks = (width + CHUNK - 1) // CHUNK
    group_of = np.repeat(np.arange(len(first_period)), chunks)
    chunk = np.arange(len(group_of)) - np.repeat(np.cumsum(chunks) - chunks, chunks)
    m = np.minimum(CHUNK, width[group_of] - CHUNK * chunk)
    block_size = lanes[group_of] * m * spans[group_of]
    block_start = np.cumsum(block_size) - block_size
    tile_block = np.repeat(np.arange(len(group_of)), spans[group_of])
    span = np.arange(len(tile_block)) - np.repeat(
        np.cumsum(spans[group_of]) - spans[group_of], spans[group_of]
    )
    g = group_of[tile_block]
    tiles = np.zeros(len(tile_block), dtype=TILE)
    tiles["period"] = first_period[g]
    tiles["lanes"] = lanes[g]
    tiles["bins"] = bins[g]
    tiles["b0"] = span * bins[g]
    tiles["d0"] = lo[first_period[g]] + CHUNK * chunk[tile_block]
    tiles["m"] = m[tile_block]
    tiles["partial"] = block_start[tile_block] + span
    tiles["tiles"] = spans[g]
    # Cells: (group, chunk, lane q, duration j) -> its partials; fold cells have none.
    entries = lanes[group_of] * m
    cell_block = np.repeat(np.arange(len(group_of)), entries)
    k = np.arange(len(cell_block)) - np.repeat(np.cumsum(entries) - entries, entries)
    q, j = np.divmod(k, m[cell_block])
    gb = group_of[cell_block]
    cells = cell_start[first_period[gb] + q] + CHUNK * chunk[cell_block] + j
    first = np.zeros(int(widths.sum()), dtype=np.int32)
    length = np.zeros(int(widths.sum()), dtype=np.int32)
    first[cells] = block_start[cell_block] + (q * m[cell_block] + j) * spans[gb]
    length[cells] = spans[gb]
    threads = np.minimum(WINDOW_THREADS, np.asarray(caps)[tiles["m"]]).astype(np.int32)
    # Each class reserves the threadgroup histogram its tiles use (16-byte multiples).
    ints = (
        (tiles["lanes"].astype(np.int64) * tiles["bins"] * (1 + 2 * tiles["m"]) + 3)
        // 4
        * 4
    )
    sort = np.lexsort((ints, threads, tiles["m"]))
    tiles, threads, ints = tiles[sort], threads[sort], ints[sort]
    starts, begin = _classes(
        tiles["m"].astype(np.int64) << 32 | threads.astype(np.int64) << 16 | ints
    )
    classes = (
        starts,
        threads[begin],
        tiles["m"][begin].astype(np.int32),
        ints[begin].astype(np.int32),
    )
    partials = int(block_size.sum())
    return tiles, classes, groups, group_classes, partials, first, length


def _float_floor(value):
    """The largest float32 at most `value` (-inf below the float range)."""
    value = float(value)
    rounded = np.float32(value)
    if float(rounded) > value:
        rounded = np.nextafter(rounded, np.float32(-np.inf))
    return float(rounded)


def _largest(values):
    """Largest magnitude of each duration's row (1 for an all-zero row)."""
    largest = np.abs(values).max(axis=1)
    return np.where(largest > 0, largest, 1.0)


def _period_max(values, widths):
    """Maximum of each period's consecutive cells (-inf without cells)."""
    out = np.full(len(widths), -np.inf)
    has = widths > 0
    if has.any():
        starts = (np.cumsum(widths) - widths)[has]
        out[has] = np.maximum.reduceat(values, starts)
    return out


_KEYS = itertools.count()


class Context:
    """One pass's GPU inputs, built once and shared by its period chunks."""

    def __init__(self, t0_time, num, den):
        self.key = next(_KEYS)
        self.t0_time = np.ascontiguousarray(t0_time, dtype=np.float64)
        self.num = np.ascontiguousarray(num, dtype=np.float64)
        self.den = np.ascontiguousarray(den, dtype=np.float64)
        t = self.t0_time
        self.usable = (
            available()
            and len(t) > 1
            and _plain(self.num, self.den, 0.0)
            and not np.any(np.diff(t) < 0)
            and 2.0**-46 * t[-1] * 4 <= EDGE_DAYS
        )
        if not self.usable:
            return
        th, tl = _split(t)
        self.times = np.ascontiguousarray(np.stack([th, tl], axis=1))
        self.table = np.searchsorted(
            t, np.arange(int(t[-1] / GRID_DAYS) + 3) * GRID_DAYS
        ).astype(np.int32)
        # Cadence-major pairs; rint(x / step) is within step / 2 of x.
        step_num = 2.0 ** np.ceil(np.log2(_largest(self.num) / INT_LIMIT))
        step_den = 2.0 ** np.ceil(np.log2(_largest(self.den) / INT_LIMIT))
        with np.errstate(over="ignore", under="ignore"):
            self.factors = (step_num / np.sqrt(step_den)).astype(np.float32)
        # Bounds need normal float32 factors; this cap keeps every lower bound finite.
        self.usable = bool(
            np.all(self.factors >= np.finfo(np.float32).tiny)
            and np.all(self.factors <= 2.0**95)
            and np.all(self.den.min(axis=1) / step_den >= 1e-37)
        )
        if not self.usable:
            return
        # pairs[parity, p, i] holds durations 2p + parity and 2p + parity + 1 of cadence i.
        n_dur, half = self.num.shape[0], self.num.shape[0] // 2 + 1
        q = np.zeros((2 * half + 1, len(t), 2), dtype=np.int32)
        q[:n_dur, :, 0] = np.rint(self.num / step_num[:, None])
        q[:n_dur, :, 1] = np.rint(self.den / step_den[:, None])
        first = 2 * np.arange(half)
        self.pairs = np.empty((2, half, len(t), 4), dtype=np.int32)
        for parity in (0, 1):
            self.pairs[parity, :, :, :2] = q[first + parity]
            self.pairs[parity, :, :, 2:] = q[first + parity + 1]

    def screen(self, order, nbins, lo, hi, n_trans, threshold=-np.inf):
        """Per-cell (upper, lower, second upper, bin) bounds; uncertain periods."""
        return self.dispatch(self.prepare(order, nbins, lo, hi, n_trans, threshold))

    def prepare(self, order, nbins, lo, hi, n_trans, threshold=-np.inf, layouts=None,
                chunk=None):  # fmt: skip
        """The CPU side of screen(): layout and period records, built before the caller claims the GPU."""
        lib = _library()
        caps = (0,) + tuple(
            max(32, min(1024, lib.v3_gpu_max_threads(m, 0)) // 32 * 32)
            for m in range(1, CHUNK + 1)
        )
        entry = layouts.get(chunk) if layouts is not None else None
        if entry is None or entry[0] != caps:
            entry = (caps, _layout(order, nbins, lo, hi, caps))
            if layouts is not None:
                layouts[chunk] = entry
        layout = entry[1]
        periods = np.zeros(len(order), dtype=PERIOD)
        periods["nb"] = nbins
        periods["epochs"] = np.floor(self.t0_time[-1] / order).astype(np.int64) + 1
        periods["ph"], periods["pl"] = _split(order)
        periods["wh"], periods["wl"] = _split(order / nbins)
        return layout, periods, n_trans, _float_floor(threshold)

    def dispatch(self, prepared):
        """The device side of screen()."""
        lib = _library()
        layout, periods, n_trans, threshold = prepared
        tiles, (starts, sizes, durations, ints), groups, (group_starts, group_m) = (
            layout[:4]
        )
        n_partials, first, length = layout[4:]
        cells = np.zeros((len(first), 4), dtype=np.float32)
        flags = np.zeros(len(periods), dtype=np.int32)
        seconds = ctypes.c_double()
        status = lib.v3_gpu_screen(
            self.key, len(self.t0_time), self.times.ctypes.data, len(self.table),
            self.table.ctypes.data, GRID_DAYS, self.num.shape[0], self.pairs.ctypes.data,
            self.pairs.nbytes, self.factors.ctypes.data,
            len(periods), periods.ctypes.data, PERIOD.itemsize, len(tiles),
            tiles.ctypes.data, TILE.itemsize, len(sizes), starts.ctypes.data,
            sizes.ctypes.data, durations.ctypes.data, ints.ctypes.data, len(groups),
            groups.ctypes.data,
            GROUP.itemsize, len(group_m), group_starts.ctypes.data, group_m.ctypes.data,
            FOLD_THREADS, n_partials, len(first), first.ctypes.data, length.ctypes.data,
            cells.ctypes.data, flags.ctypes.data, EDGE_DAYS, float(n_trans),
            threshold, ctypes.byref(seconds),
        )  # fmt: skip
        if status:
            raise RuntimeError(f"GPU screening failed with status {status}")
        return cells, flags.astype(bool), seconds.value

    def supported(self, order, nbins, threshold):
        """Whether the integer and phase error bounds cover these inputs."""
        # Keep headroom above float32 subnormals; the CPU keeps the requested threshold.
        if (
            not self.usable
            or len(order) == 0
            or not threshold >= 4.0 * np.finfo(np.float32).tiny
        ):
            return False
        if nbins.max() >= 2**24:
            return False
        if np.floor(self.t0_time[-1] / order.min()) + 1 > MAX_EPOCHS:
            return False
        return True

    def fold_search(self, order, nbins, lo, hi, n_trans, threshold, budget):
        """kernel_bindings.fold_search through GPU bounds; None when bounds do not apply."""
        if not self.supported(order, nbins, threshold):
            return None
        cells, uncertain, seconds = self.screen(
            order, nbins, lo, hi, n_trans, threshold
        )
        return self.resolve(
            order, nbins, lo, hi, n_trans, threshold, budget, cells, uncertain, seconds
        )

    def resolve(
        self,
        order,
        nbins,
        lo,
        hi,
        n_trans,
        threshold,
        budget,
        cells,
        uncertain,
        seconds,
    ):
        """Resolve the returned bounds on CPU without occupying the GPU."""
        t0_time, num, den = self.t0_time, self.num, self.den
        widths = (hi - lo).astype(np.int64)
        period_of = np.repeat(np.arange(len(order)), widths)
        starts = np.cumsum(widths) - widths
        duration_of = lo[period_of] + (np.arange(len(period_of)) - starts[period_of])
        upper = cells[:, 0].astype(np.float64)
        lower = cells[:, 1].astype(np.float64)
        second = cells[:, 2].astype(np.float64)
        arg = cells[:, 3].view(np.int32)
        lower_max = _period_max(lower, widths)
        best = np.full(len(order), float(threshold))
        winner = np.full(len(order), -1, dtype=np.int32)
        position = np.full(len(order), -1, dtype=np.int32)
        # Uncertain phase assignments: exact search of the whole period.
        exact = np.flatnonzero(uncertain)
        if len(exact):
            b, w, p = kernel_bindings.fold_search(
                t0_time, num, den, order[exact], nbins[exact], lo[exact], hi[exact],
                n_trans, threshold, budget,
            )  # fmt: skip
            best[exact], winner[exact], position[exact] = b, w, p
        floor_ = np.maximum(float(threshold), lower_max)
        candidate = (
            (~uncertain[period_of]) & (upper > threshold) & (upper >= floor_[period_of])
        )
        unique = candidate & (second < floor_[period_of])
        value = np.full(len(upper), -np.inf)
        where = np.flatnonzero(unique)
        if len(where):
            mes = np.empty(len(where))
            kernel_bindings.library().v3_bins_mes(
                ctypes.c_int64(len(t0_time)), t0_time, num, den,
                ctypes.c_int64(len(where)),
                np.ascontiguousarray(order[period_of[where]]),
                np.ascontiguousarray(nbins[period_of[where]], dtype=np.int32),
                np.ascontiguousarray(duration_of[where], dtype=np.int32),
                np.ascontiguousarray(arg[where], dtype=np.int32), float(n_trans),
                self.table, len(self.table), GRID_DAYS, mes,
            )  # fmt: skip
            value[where] = mes
        cell_arg = arg.astype(np.int32).copy()
        shared = np.flatnonzero(candidate & ~unique)
        if len(shared):
            b, w, p = kernel_bindings.fold_search(
                t0_time, num, den, order[period_of[shared]], nbins[period_of[shared]],
                duration_of[shared].astype(np.int32),
                (duration_of[shared] + 1).astype(np.int32), n_trans, -np.inf, budget,
            )  # fmt: skip
            value[shared] = b
            cell_arg[shared] = p
        resolved = np.flatnonzero(candidate & (value > threshold))
        if len(resolved):
            # First duration attaining each period's maximum, as the upstream loop.
            periods = period_of[resolved]
            top = np.full(len(order), -np.inf)
            np.maximum.at(top, periods, value[resolved])
            hit = resolved[value[resolved] == top[periods]]
            keep = np.ones(len(hit), dtype=bool)
            keep[1:] = period_of[hit[1:]] != period_of[hit[:-1]]
            hit = hit[keep]
            best[period_of[hit]] = value[hit]
            winner[period_of[hit]] = duration_of[hit]
            position[period_of[hit]] = cell_arg[hit]
        statistics = {
            "gpu_seconds": seconds,
            "uncertain_periods": int(len(exact)),
            "bin_resolutions": int(len(where)),
            "cell_resolutions": int(len(shared)),
        }
        return best, winner, position, statistics


def fold_search(t0_time, num, den, order, nbins, lo, hi, n_trans, threshold, budget):
    """One-shot Context(t0_time, num, den).fold_search(...)."""
    return Context(t0_time, num, den).fold_search(
        order, nbins, lo, hi, n_trans, threshold, budget
    )
