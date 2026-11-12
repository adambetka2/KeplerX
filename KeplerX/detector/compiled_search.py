"""The Detector's search, compiled."""

import copy
import importlib
import os
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy import fft as sp_fft
from scipy.signal import _signaltools

from detector import kernel_bindings
from detector.gap_padding import (
    MIN_SEARCH_CADENCES,
    DegenerateCurve,
    Segments,
    padding,
)
from detector.repeat_rule import Repeats
from detector.upstream import dump as upstream
from detector.upstream import ses as upstream_ses

# Each period block's folded sums stay within this many bytes (L2-resident).
FOLD_BUDGET = 1 << 20
# Periods per GPU scheduling chunk: concurrent workers share the GPU in short turns.
GPU_CHUNK = 16384
# Screen MES; refined MES must reach the threshold (docs/keplerX-2026-11-02-detection-at-the-information-limit.md).
SCREEN_THRESHOLD = 6.6


def kep_period_sampling(time, P_min=None, P_max=None, n_tr=3.0, rho_star=1.0,
                        d_leastsq=0.075):  # fmt: skip
    """upstream.kep_period_sampling with its recurrence compiled (bitwise)."""
    if P_min is None:
        P_min = 0.25 / np.sqrt(rho_star)
    if P_max is None:
        P_max = np.inf
    baseline = max(time) - min(time)
    if n_tr > 1:
        n_tr -= 1
    limit = min(baseline / n_tr, P_max)
    return kernel_bindings.period_sampling(P_min, limit, baseline, d_leastsq, rho_star)


@contextmanager
def deferred_period_sampling(module):
    """Construct a Dump of `module` without sampling its periods."""
    original = module.kep_period_sampling
    module.kep_period_sampling = lambda **_: np.empty(0)
    try:
        yield
    finally:
        module.kep_period_sampling = original


def search_grid(periods, tdurs, texp, rho_star, dur_range):
    """Per-period bin counts and admitted duration index ranges, as upstream."""
    min_dur_buff = (
        (13.0 / 24.0) * (1.0 / 365.0) ** (1.0 / 3.0) * rho_star ** (-1.0 / 3.0)
    ) * dur_range[0]
    max_dur_buff = (
        (13.0 / 24.0) * (1.0 / 365.0) ** (1.0 / 3.0) * rho_star ** (-1.0 / 3.0)
    ) * dur_range[1]
    periods = np.asarray(periods)
    root = periods ** (1.0 / 3.0)
    min_dur = root * min_dur_buff
    max_dur = root * max_dur_buff
    tdurs = np.asarray(tdurs)
    if not (np.isfinite(tdurs).all() and np.all(np.diff(tdurs) > 0)):
        raise ValueError("Duration gate needs a sorted template bank")
    lo = np.searchsorted(tdurs, min_dur, side="left")
    hi = np.minimum(
        np.searchsorted(tdurs, max_dur, side="right"),
        np.searchsorted(tdurs, periods / 3.0, side="left"),
    )
    count = np.maximum(hi - lo, 0)
    lo = np.where(count > 0, lo, 0).astype(np.int32)
    hi = (lo + count).astype(np.int32)
    nbins = (np.floor_divide(periods, texp) + 1).astype(np.int64)
    if nbins.max(initial=0) >= 2**31:
        raise ValueError("Phase-bin count exceeds the compiled kernel range")
    return nbins.astype(np.int32), lo, hi


class SearchGrid:
    """A search's trial periods, longest first."""

    def __init__(self, periods, tdurs, texp, rho_star, dur_range):
        self.periods = periods
        self.inputs = (np.asarray(tdurs), float(texp), float(rho_star),
                       tuple(map(float, dur_range)))  # fmt: skip
        self.order = np.asarray(periods)[::-1]
        self.nbins, self.lo, self.hi = search_grid(
            self.order, tdurs, texp, rho_star, dur_range
        )
        self.layouts = {}

    @classmethod
    def serving(cls, grid, periods, tdurs, texp, rho_star, dur_range):
        """grid when it serves these inputs, else a new grid for them."""
        if grid is not None and grid.serves(periods, tdurs, texp, rho_star, dur_range):
            return grid
        return cls(periods, tdurs, texp, rho_star, dur_range)

    def serves(self, periods, tdurs, texp, rho_star, dur_range):
        bank, step, density, window = self.inputs
        return (
            (periods is self.periods or np.array_equal(periods, self.periods))
            and np.array_equal(tdurs, bank)
            and (float(texp), float(rho_star), tuple(map(float, dur_range)))
            == (step, density, window)
        )


USAGE = {
    "gpu_chunks": 0, "cpu_chunks": 0, "gpu_failures": 0, "gpu_seconds": 0.0,
    "resolved_periods": 0, "omitted_duration_convolutions": 0,
}  # fmt: skip


GPU_AHEAD = 8  # bounded screening results; one GPU dispatch thread
# Periods per GPU dispatch of the longest-period chunk: keeps its partial buffer to tens of MB.
LONG_BLOCK = 2048
_UNSCREENED = object()


def fold_search(t0_time, num, den, order, nbins, lo, hi, num_transits, threshold,
                layouts=None):  # fmt: skip
    """kernel_bindings.fold_search, with supported chunks queued for GPU screening."""
    context = None
    if os.environ.get("KEPLERX_GPU", "1") != "0":
        from detector import gpu_screening

        if gpu_screening.available():
            context = gpu_screening.Context(t0_time, num, den)
            if not context.usable:
                context = None
    out = (np.empty(len(order)), np.empty(len(order), dtype=np.int32),
           np.empty(len(order), dtype=np.int32))  # fmt: skip
    blocks = _blocks(order, nbins, lo, hi, num_transits, threshold, context is not None)
    if context is None:
        for part, args in blocks:
            _cpu_block(t0_time, num, den, part, args, out)
    else:
        _gpu_blocks(context, t0_time, num, den, blocks, layouts, out)
    return out


def _blocks(order, nbins, lo, hi, num_transits, threshold, gpu):
    """(slice, fold arguments) per chunk; on the GPU the longest periods go in LONG_BLOCK blocks."""
    blocks = []
    for start in range(0, len(order), GPU_CHUNK):
        stop = min(start + GPU_CHUNK, len(order))
        step = LONG_BLOCK if gpu and start == 0 else GPU_CHUNK
        for first in range(start, stop, step):
            part = slice(first, min(first + step, stop))
            blocks.append((part, (order[part], nbins[part], lo[part], hi[part],
                                  num_transits, threshold)))  # fmt: skip
    return blocks


def _gpu_blocks(context, t0_time, num, den, blocks, layouts, out):
    """Screen blocks on the GPU, short CPU-heavy ones first; results keep their slices."""
    blocks = blocks[::-1]
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = []
        for index, (part, args) in enumerate(blocks):
            while len(futures) < min(index + GPU_AHEAD, len(blocks)):
                ahead, ahead_args = blocks[len(futures)]
                key = (ahead.start, ahead.stop)
                futures.append(
                    pool.submit(_gpu_screen, context, ahead_args, layouts, key)
                )
            screened = _gpu_chunk(context, args, futures[index].result())
            futures[index] = None
            if screened is None:
                _cpu_block(t0_time, num, den, part, args, out)
            else:
                for target, value in zip(out, screened[:3]):
                    target[part] = value


def _cpu_block(t0_time, num, den, part, args, out):
    USAGE["cpu_chunks"] += 1
    folded = kernel_bindings.fold_search(t0_time, num, den, *args, FOLD_BUDGET)
    for target, value in zip(out, folded[:3]):
        target[part] = value


def _gpu_screen(context, args, layouts=None, chunk=None):
    from detector import gpu_screening

    if not context.supported(args[0], args[1], args[-1]):
        return None
    try:
        # Only the device work holds the shared GPU claim.
        prepared = context.prepare(*args, layouts=layouts, chunk=chunk)
        with gpu_screening.claim():
            return context.dispatch(prepared)
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        USAGE["gpu_failures"] += 1
        return None


def _gpu_chunk(context, args, screened=_UNSCREENED):
    if screened is _UNSCREENED:
        screened = _gpu_screen(context, args)
    if screened is None:
        return None
    try:
        out = context.resolve(*args, FOLD_BUDGET, *screened)
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        USAGE["gpu_failures"] += 1
        return None
    if out is not None:
        stats = out[3]
        USAGE["gpu_chunks"] += 1
        USAGE["gpu_seconds"] += stats["gpu_seconds"]
        USAGE["resolved_periods"] += (
            stats["uncertain_periods"] + stats["bin_resolutions"]
            + stats["cell_resolutions"]
        )  # fmt: skip
    return out


def _kept_winners(t0_time, ses, order, nbins, best, winner, position, t_durs,
                  single_frac, threshold):  # fmt: skip
    """Yield (P, max_mes, duration, peak) for unvetoed winners in search order."""
    found = np.flatnonzero(winner >= 0)
    peaks, vetoed = kernel_bindings.single_events(
        t0_time, ses, order[found], nbins[found], position[found], winner[found],
        t_durs, best[found], frac=single_frac, threshold=threshold,
    )  # fmt: skip
    for slot, index in enumerate(found):
        P, i, max_mes = order[index], int(winner[index]), best[index]
        if vetoed[slot] < 0:
            # Upstream's np.max over an empty in-transit selection.
            raise ValueError(
                "zero-size array to reduction operation maximum which has no identity"
            )
        if not vetoed[slot]:
            yield P, max_mes, t_durs[i], np.float64(peaks[slot])


def _epoch_search(t0, t0_time, num, den, order, nbins, lo, hi, t_durs, threshold,
                  texp, num_transits, kicid, return_df, layouts=None,
                  masked_events=None):  # fmt: skip
    """search_all_durations' search."""
    screen = min(SCREEN_THRESHOLD, threshold)
    best, winner, position = fold_search(
        t0_time, num, den, order, nbins, lo, hi, num_transits, screen, layouts
    )
    rows = _epoch_candidates(
        t0, t0_time, num, den, order, nbins, lo, hi, best, winner, position, t_durs,
        threshold, texp, num_transits, kicid, masked_events,
    )  # fmt: skip
    frame = pd.DataFrame(rows, columns=["star_id", "period", "mes", "t0", "tdur"])
    if not len(frame):
        return np.array([[kicid, np.nan, np.nan, np.nan, np.nan]])
    return frame if return_df else frame.to_numpy()


def _epoch_candidates(t0, t0_time, num, den, order, nbins, lo, hi, best, winner,
                      position, t_durs, threshold, texp, num_transits, kicid,
                      masked_events=None):  # fmt: skip
    """Rows (star_id, period, mes, t0, tdur)."""
    from detector import candidate_ephemerides as ce
    from detector import single_event_veto

    found = np.flatnonzero(winner >= 0)
    if not len(found):
        return []
    periods = order[found]
    centre = (position[found] + 0.5) * (periods / nbins[found])
    survivors = found[
        ce.vetted_clean(
            np.ascontiguousarray(periods, dtype=np.float64),
            np.ascontiguousarray(best[found], dtype=np.float64),
            np.ascontiguousarray(t0 + centre, dtype=np.float64),
            np.ascontiguousarray(t_durs[winner[found]], dtype=np.float64),
            float(t0 + 0.5 * t0_time[-1]), float(t0_time[-1]), ce.MULTIPLES,
            ce.SAME_PERIOD, ce.HARMONIC, ce.EPOCH_DURATIONS, t0_time,
            np.ascontiguousarray(num, dtype=np.float64),
            np.ascontiguousarray(den, dtype=np.float64),
            nbins[found].astype(np.int64), winner[found].astype(np.int64),
            position[found].astype(np.int64),
            single_event_veto.expected_maxima(int(t0_time[-1] / periods.min()) + 2),
            single_event_veto.OUTLIER_MARGIN, float(threshold),
        )
    ]  # fmt: skip
    if len(survivors) > ce.REFINED:
        strongest = np.argsort(-best[survivors], kind="stable")[: ce.REFINED]
        survivors = np.sort(survivors[strongest])
    num = np.ascontiguousarray(num, dtype=np.float64)
    den = np.ascontiguousarray(den, dtype=np.float64)
    use = np.ones(len(t0_time), dtype=bool)
    rows = []
    for index in survivors:
        P = float(order[index])
        neighbours = order[max(index - 1, 0) : index + 2]
        step = float(np.abs(np.diff(neighbours)).mean()) if len(neighbours) > 1 else 0.0
        phase = (position[index] + 0.5) * (P / nbins[index])
        mes, period, epoch, row, _ = ce.refine(
            t0_time, num, den, use, P, step, ce.PERIOD_STEPS,
            float(ce.mid_epoch(t0_time, P, phase)), float(texp), ce.EPOCH_STEPS,
            int(lo[index]), int(hi[index]), int(num_transits),
        )  # fmt: skip
        if not np.isfinite(mes):
            mes, period, epoch, row = best[index], P, phase, int(winner[index])
        if mes < threshold:
            continue
        mes, period, epoch, row = ce.strongest_alias(
            t0_time, num, den, use, order, lo, hi, float(texp), int(num_transits),
            (mes, period, epoch, row),
        )  # fmt: skip
        nums, dens, centres = ce.event_terms(
            t0_time, num, den, use, period, epoch, float(texp), row
        )
        carrying = ce.carrying_events(nums, dens, threshold, int(num_transits))
        if len(carrying):
            if masked_events is not None:
                masked_events.extend(
                    (t0 + float(centres[i]), float(t_durs[row])) for i in carrying
                )
            continue
        rows.append(
            (kicid, period, float(mes), t0 + float(ce.first_epoch(period, epoch)),
             t_durs[row])
        )  # fmt: skip
    return rows


def search_all_durations(
    time, num, den, ses, period_sampling, kicid, t_durs, rho_star=1.0,
    dur_range=(0.25, 1.5), outputfile=None, print_updates=True, texp=0.0204,
    return_df=False, progbar=True, num_transits=2, check_vetoes=False,
    single_frac=0.9, norm_mes=False, threshold=7.0, module=upstream, grid=None,
    upstream_reference=False, masked_events=None,
):  # fmt: skip
    """upstream.Search_for_TCEs_in_all_Tdur_models with a compiled period loop."""
    if norm_mes:
        raise ValueError("norm_mes has no compiled equivalent")
    if check_vetoes:
        raise ValueError("check_vetoes has no compiled equivalent")
    t0 = np.min(time)
    t0_time = np.ascontiguousarray(time - t0)
    detections = {"star_id": [], "period": [], "mes": [], "t0": [], "tdur": []}
    grid = SearchGrid.serving(grid, period_sampling, t_durs, texp, rho_star, dur_range)
    order, nbins, lo, hi = grid.order, grid.nbins, grid.lo, grid.hi
    if not upstream_reference:
        return _epoch_search(
            t0, t0_time, num, den, order, nbins, lo, hi, np.asarray(t_durs),
            threshold, texp, num_transits, kicid, return_df, grid.layouts,
            masked_events,
        )  # fmt: skip
    best, winner, position = fold_search(
        t0_time, num, den, order, nbins, lo, hi, num_transits, threshold, grid.layouts
    )
    kept = _kept_winners(
        t0_time, ses, order, nbins, best, winner, position, t_durs, single_frac,
        threshold,
    )  # fmt: skip
    for P, max_mes, detected_duration, detected_peak_loc in kept:
        if print_updates:
            print(
                "\rKIC{4:09d}: TCE at P={0:.7f} days,    MES={1:.2f},    "
                "Tdur={2:.3f} days,    t0={3:.2f}".format(
                    P, max_mes, detected_duration, detected_peak_loc + t0, kicid
                )
            )
        detections["star_id"].append(kicid)
        detections["period"].append(P)
        detections["mes"].append(max_mes)
        detections["t0"].append(detected_peak_loc + t0)
        detections["tdur"].append(detected_duration)
    df = pd.DataFrame(detections)
    if len(df) > 0:
        cleaned_df = module.clean_tces_of_harmonics(df)
        if print_updates:
            print(cleaned_df)
        if return_df:
            return cleaned_df
        return cleaned_df.to_numpy()
    return np.array([[kicid, np.nan, np.nan, np.nan, np.nan]])


def calc_var_stat(x, window_size, exp_time, method="mad", slat_frac=0.2,
                  n_mad_window=15_001, ses=upstream_ses, ranks=None):  # fmt: skip
    """ses.calc_var_stat (MAD branch) with the compiled running median."""
    if method != "mad" or np.isnan(x).any():
        return ses.calc_var_stat(
            x, window_size, exp_time, method, slat_frac, n_mad_window
        )
    window_points = 2 * int(window_size / exp_time)
    if window_points < n_mad_window:
        shared = ranks() if ranks is not None else None
        sig = 1.4826 * kernel_bindings.running_median(np.abs(x), window_points, shared)
    else:
        dx = window_points / n_mad_window
        indices = np.arange(len(x))
        skip_indices = np.linspace(0, len(x), int(len(x) / dx))
        x_skipped = np.interp(skip_indices, indices, x)
        sig_decimated = 1.4826 * kernel_bindings.running_median(
            np.abs(x_skipped), n_mad_window
        )
        sig = np.interp(indices, skip_indices, sig_decimated)
    if (sig == 0).any():
        nbad = np.sum(sig == 0)
        print("WARNING! VAR STAT == 0 AT {}/{} CADENCES".format(nbad, len(sig)))
        sig[np.where(sig == 0)[0]] = np.nanmedian(sig)
    return sig**2.0


class Levels:
    """Lazy ranking of |transform level| shared by every template's MAD."""

    def __init__(self, transform):
        self.transform = transform
        self.cache = {}

    def ranks(self, level):
        def compute():
            if level not in self.cache:
                self.cache[level] = kernel_bindings.ranking(
                    np.abs(self.transform[level])
                )
            return self.cache[level]

        return compute


class Template:
    """A template's wavelet levels with cached convolution operands."""

    def __init__(self, transform):
        self.transform = transform
        self.spectra = {}
        self.kernels = {}

    def kernel(self, level, squared):
        reverse = self.transform[level, ::-1]
        return reverse**2.0 if squared else reverse

    def spectrum(self, level, squared, fshape):
        key = (level, squared)
        if key not in self.spectra:
            self.spectra[key] = sp_fft.rfftn(
                self.kernel(level, squared), fshape, axes=[0]
            )
        return self.spectra[key]

    def correlate_kernel(self, level, squared):
        key = (level, squared)
        if key not in self.kernels:
            ap2 = np.ascontiguousarray(self.kernel(level, squared)[::-1])
            nonzero = np.flatnonzero(ap2)
            span = (int(nonzero[0]), int(nonzero[-1]) + 1) if len(nonzero) else (0, 0)
            self.kernels[key] = (ap2, span)
        return self.kernels[key]


_METHODS = {}
# Bytes of durations FFT whitening transforms at once.
FFT_BANK_BYTES = 256 * 1024**2
# Template spectra are cached for transforms up to this size (at most 7 MB per duration pair).
SPECTRUM_CACHE_POINTS = 32768


def convolution_method(in1, in2):
    """Cache SciPy's shape/dtype-based direct-versus-FFT decision."""
    key = (in1.shape, in2.shape, in1.dtype, in2.dtype)
    if key not in _METHODS:
        _METHODS[key] = (
            "scipy"
            if in1.ndim != 1
            or in2.ndim != 1
            or len(in1) < 2
            or len(in2) < 2
            or in1.dtype != np.float64
            or in2.dtype != np.float64
            else _signaltools.choose_conv_method(in1, in2, mode="same")
        )
    return _METHODS[key]


def convolve_same(in1, template, level, squared, need, ses=upstream_ses):
    """scipy.signal.convolve(in1, template kernel, mode="same")[need], same bits."""
    in2 = template.kernel(level, squared)
    method = convolution_method(in1, in2)
    if method == "scipy":
        return ses.convolve(in1, in2, mode="same")[need]
    if method == "direct":
        # NumPy's small_correlate takes kernels of at most 11 samples.
        if (
            len(in1) != len(in2)
            or len(in1) <= 16
            or not kernel_bindings.library().direct_convolution
        ):
            return np.convolve(in1, in2, "same")[need]
        ap2, span = template.correlate_kernel(level, squared)
        return kernel_bindings.convolve_same(in1, ap2, span, need)
    shape = len(in1) + len(in2) - 1
    fshape = [sp_fft.next_fast_len(shape, True)]
    sp1 = sp_fft.rfftn(in1, fshape, axes=[0])
    ret = sp_fft.irfftn(
        sp1 * template.spectrum(level, squared, fshape), fshape, axes=[0]
    )
    start = (shape - len(in1)) // 2
    out = ret[:shape][start : start + len(in1)].copy()
    if np.isnan(out.flat[0]) or np.isinf(out.flat[0]):
        warnings.warn(
            "Use of fft convolution on input with NAN or inf results in NAN or inf "
            "output. Consider using method='direct' instead.",
            category=RuntimeWarning,
            stacklevel=2,
        )
    return out.astype(np.float64)[need]


def variance_weights(flux_transform, window_size, var_calc="mad", texp=0.020417,
                     ses_module=upstream_ses, shared=None):  # fmt: skip
    """The unchanged whitening weights for one duration."""
    shared = Levels(flux_transform) if shared is None else shared
    sig2 = [
        1.0
        / calc_var_stat(
            x,
            window_size * 2**i,
            method=var_calc,
            exp_time=texp,
            ses=ses_module,
            ranks=shared.ranks(i),
        )  # fmt: skip
        for i, x in enumerate(flux_transform)
    ]
    return np.array(sig2)


def calculate_ses(flux_transform, template, window_size, need, var_calc="mad",
                  texp=0.020417, ses_module=upstream_ses, shared=None):  # fmt: skip
    """upstream_ses.calculate_SES evaluated at the kept positions `need`."""
    sig2 = variance_weights(
        flux_transform, window_size, var_calc, texp, ses_module, shared
    )
    if (
        flux_transform.shape == template.transform.shape == sig2.shape
        and flux_transform.dtype == template.transform.dtype == sig2.dtype == np.float64
        and sig2.shape[1] > 16
        and convolution_method(flux_transform[0], template.transform[0]) == "direct"
        and kernel_bindings.library().direct_convolution
    ):
        N, D = kernel_bindings.ses_sums(flux_transform, sig2, template.transform, need)
        return N / np.sqrt(D), N, D, sig2
    n_levels = len(flux_transform)
    N = np.zeros(len(need))
    D = np.zeros(len(need))
    levels = np.concatenate([np.arange(1, n_levels), [n_levels - 1]])
    for i, level in enumerate(levels):
        weighted = sig2[i] * flux_transform[i]
        N += 2.0 ** (-level) * convolve_same(
            weighted, template, i, False, need, ses_module
        )
        D += 2.0 ** (-level) * convolve_same(
            sig2[i], template, i, True, need, ses_module
        )
    ses = N / np.sqrt(D)
    return ses, N, D, sig2


def _same_bytes(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _same_input(last, now):
    """Whether a segment's whitening input equals its last one."""
    (keep, time, flux, options), (keep2, time2, flux2, options2) = last, now
    if options.keys() != options2.keys():
        return False
    return (
        _same_bytes(keep, keep2) and _same_bytes(time, time2)
        and _same_bytes(flux, flux2)
        and all(
            _same_bytes(options[k], options2[k])
            if isinstance(options[k], np.ndarray) or isinstance(options2[k], np.ndarray)
            else options[k] == options2[k]
            for k in options
        )
    )  # fmt: skip


class FastSearch(Segments):
    """Mixin over a stage's upstream Dump: compiled whitening and search."""

    _upstream = upstream
    _ses = upstream_ses

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._signal_transforms = {}
        # Each segment's last whitening, by (begin, end).
        self._segment_whitening = {}
        self._grid = None
        # Template spectra by segment shape, shared by segment threads.
        self._bank_spectra = {}
        # The stage runner drops whitening rows that other Dump callers keep.
        self.search_only = False

    def trial_grid(self, dur_range=(0.25, 1.5)):
        """This Dump's SearchGrid for its periods, bank and the window."""
        self._grid = SearchGrid.serving(
            self._grid, self.search_periods, self.tdurs, self.lc.exptime,
            self.star_density, dur_range,
        )  # fmt: skip
        return self._grid

    def _active_search_rows(self, remove_hises, calc_ses, search_options):
        """Rows admitted by at least one actual period, never a smaller search."""
        if (
            not self.search_only
            or remove_hises
            or not calc_ses
            or any(mode in ("raise", "call", "log") for mode in np.geterr().values())
            or getattr(self.Calculate_SES, "__func__", None)
            is not FastSearch.Calculate_SES
        ):
            return None
        durations, periods = np.asarray(self.tdurs), np.asarray(self.search_periods)
        if not (
            durations.ndim == periods.ndim == 1
            and len(durations)
            and len(periods)
            and np.isfinite(durations).all()
            and np.isfinite(periods).all()
            and np.all(durations > 0)
            and np.all(periods > 0)
            and np.all(np.diff(durations) > 0)
        ):
            return None
        grid = self.trial_grid(search_options.get("dur_range", (0.25, 1.5)))
        low, high = grid.lo, grid.hi
        starts = np.bincount(low, minlength=len(durations) + 1)
        stops = np.bincount(high, minlength=len(durations) + 1)
        active = np.cumsum(starts - stops)[:-1] > 0
        return active if active.any() else None

    def _segment_ses(self, starts, mask, calc_ses_kw):
        if (
            getattr(self.Calculate_SES, "__func__", None)
            is not FastSearch.Calculate_SES
        ):
            return super()._segment_ses(starts, mask, calc_ses_kw)
        # Build the shared immutable kernels before worker access.
        kernel_bindings.library()
        kernel_bindings._db2_filters()
        errors = np.geterr()
        bounds = list(zip(starts, starts[1:] + [len(self.lc.time)], strict=True))
        workers = min(getattr(self, "whitening_threads", 1), len(bounds))
        parts = []
        if workers <= 1:
            results = [
                self._whiten_segment(a, b, mask, calc_ses_kw, errors) for a, b in bounds
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = [
                    pool.submit(self._whiten_segment, a, b, mask, calc_ses_kw, errors)
                    for a, b in bounds
                ]
                results = [job.result() for job in jobs]
        for result in results:
            if result is not None:
                segment, cdpp = result
                parts.append(segment)
                self.tdur_cdpp.extend(cdpp)
        return parts

    def _whiten_segment(self, begin, end, mask, calc_ses_kw, errors):
        """One segment's whitening, (begin, end, (ses, num, den)) and its CDPP."""
        keep = mask[begin:end]
        if not keep.any():
            return None
        time = self.lc.time[begin:end][keep]
        flux = self.lc.flux[begin:end][keep]
        last = self._segment_whitening.get((begin, end))
        if last is not None and _same_input(last[0], (keep, time, flux, calc_ses_kw)):
            ses, num, den, cdpp = last[1]
            return (begin, end, (ses.copy(), num.copy(), den.copy())), list(cdpp)
        # Merge per-segment results in original segment order.
        child = copy.copy(self)
        child.tdur_cdpp = []
        with np.errstate(**errors):
            part = child.Calculate_SES(time=time, flux=flux, mask=keep, **calc_ses_kw)
        if calc_ses_kw.get("ses_weights") is None:
            self._segment_whitening[(begin, end)] = (
                (keep.copy(), time.copy(), flux.copy(), dict(calc_ses_kw)),
                tuple(a.copy() for a in part) + (list(child.tdur_cdpp),),
            )
        return (begin, end, part), child.tdur_cdpp

    def _template(self, tdepth, dur, pad):
        key = (tdepth, dur, pad)
        if key not in self._signal_transforms:
            signal = self.get_transit_model(depth=tdepth, width=dur, pad=pad)
            transform = kernel_bindings.ocwt(signal, upstream=self._ses.ocwt)
            self._signal_transforms[key] = Template(transform)
        return self._signal_transforms[key]

    def _fft_whitening(self, flux, need, durations, active, var_calc, window_width,
                       tdepth):  # fmt: skip
        """Calculate_SES's rows by _fft_bank."""
        if (
            var_calc != "mad"
            or flux.dtype != np.float64
            or flux.shape[1] <= 16
            or not np.isfinite(flux).all()
        ):
            return None, None
        rows = np.arange(len(durations)) if active is None else np.flatnonzero(active)
        windows = self._bank_windows(len(flux), durations, rows, window_width)
        cdpp = {} if active is None else None
        bank = self._fft_bank(flux, need, durations, rows, windows, tdepth, cdpp)
        return (None, None) if bank is None else (bank, cdpp)

    def _bank_windows(self, levels, durations, rows, window_width):
        """calc_var_stat's window points for window_size * 2**i, per row."""
        return np.array(
            [
                [2 * int(window_width * durations[d] * 2**i / self.lc.exptime)
                 for i in range(levels)]
                for d in rows
            ],
            dtype=np.int64,
        )  # fmt: skip

    def _fft_bank(self, flux, need, durations, rows, windows, tdepth, cdpp=None):
        """The Detector's whitening."""
        levels, n = flux.shape
        size = sp_fft.next_fast_len(2 * n - 1, True)
        group = max(1, FFT_BANK_BYTES // (32 * levels * (size // 2 + 1)))
        start = (n - 1) // 2 + need
        bank = {}
        for first in range(0, len(rows), group):
            part = windows[first : first + group]
            level = np.broadcast_to(np.arange(levels), part.shape)
            pairs, index = np.unique(
                np.stack([level, part], -1).reshape(-1, 2), axis=0, return_inverse=True
            )
            sig2 = kernel_bindings.mad_pairs(flux, pairs[:, 0], pairs[:, 1])
            if sig2 is None:
                return None
            weighted = sp_fft.rfft(sig2 * flux[pairs[:, 0]], size, axis=1)
            weights = sp_fft.rfft(sig2, size, axis=1)
            index = index.reshape(part.shape)
            for j, d in enumerate(rows[first : first + group]):
                kernel, square = self._bank_spectrum(tdepth, durations[d], n, size)
                own = index[j]
                if cdpp is not None:
                    cdpp[d] = np.median(sig2[own])
                num = sp_fft.irfft(np.einsum("ij,ij->j", weighted[own], kernel), size)
                den = sp_fft.irfft(np.einsum("ij,ij->j", weights[own], square), size)
                num, den = num[start], den[start]
                bank[d] = (num / np.sqrt(den), num, den)
        return bank

    def _bank_spectrum(self, tdepth, duration, n, size):
        """A template's correlation kernels for every level."""
        key = (tdepth, duration, n, size)
        if key in self._bank_spectra:
            return self._bank_spectra[key]
        transform = self._template(tdepth, duration, n).transform
        levels = len(transform)
        weight = 2.0 ** -np.minimum(np.arange(1, levels + 1), levels - 1)
        reverse = transform[:, ::-1]
        spectra = (
            weight[:, None] * sp_fft.rfft(reverse, size, axis=1),
            weight[:, None] * sp_fft.rfft(reverse**2, size, axis=1),
        )
        if size <= SPECTRUM_CACHE_POINTS:
            self._bank_spectra[key] = spectra
        return spectra

    def Calculate_SES(self, time=None, flux=None, mask=None, var_calc="mad",
                      fill_mode="reflect", tdurs=None, ses_weights=None,
                      window_width=7.0, tdepth=100.0, multi_sector=False,
                      _active_rows=None):  # fmt: skip
        mask = self.lc.mask.copy() if mask is None else mask
        time = self.lc.time[mask] if time is None else time
        flux = self._baseline(self.lc.flux[mask] if flux is None else flux)
        dur_iter = self.tdurs if tdurs is None else tdurs
        pad_time, pad_flux, pad_boo = padding(self, time, flux, fill_mode, dur_iter)
        FluxTransform = kernel_bindings.ocwt(pad_flux - 1.0, upstream=self._ses.ocwt)
        if pad_boo.dtype != bool or len(pad_boo) != len(pad_flux):
            # Fail as upstream does (ses[pad_boo]) before any compiled read past the transform.
            np.zeros(len(pad_flux))[pad_boo]
            raise TypeError("pad_time_series returned an unusable keep mask")
        need = np.flatnonzero(pad_boo)
        shared = Levels(FluxTransform)
        ses_alldur = [[]] * len(dur_iter)
        num_alldur = [[]] * len(dur_iter)
        den_alldur = [[]] * len(dur_iter)
        bank = cdpp = None
        if self.reference is None:
            bank, cdpp = self._fft_whitening(
                FluxTransform, need, dur_iter, _active_rows, var_calc, window_width,
                tdepth,
            )  # fmt: skip
        for i, dur in enumerate(dur_iter):
            if _active_rows is not None and not _active_rows[i]:
                # No search or veto reads this row; padding and indices stay unchanged.
                ses_alldur[i] = np.zeros(len(need))
                num_alldur[i] = np.zeros(len(need))
                den_alldur[i] = np.ones(len(need))
                USAGE["omitted_duration_convolutions"] += 1
                continue
            if bank is not None:
                ses_alldur[i], num_alldur[i], den_alldur[i] = bank[i]
                if cdpp is not None:
                    self.tdur_cdpp.append(cdpp[i])
                continue
            template = self._template(tdepth, dur, len(pad_flux))
            ses, num, den, sig2 = calculate_ses(
                FluxTransform, template, texp=self.lc.exptime,
                window_size=window_width * dur, need=need, var_calc=var_calc,
                ses_module=self._ses, shared=shared,
            )  # fmt: skip
            ses_alldur[i] = ses
            num_alldur[i] = num
            den_alldur[i] = den
            # Only the private search path skips CDPP diagnostics.
            if _active_rows is None:
                self.tdur_cdpp.append(np.median(sig2, overwrite_input=True))
        if ses_weights is None:
            ses_alldur = np.array(ses_alldur)
            num_alldur = np.array(num_alldur)
        else:
            ses_alldur = np.array(ses_alldur) / ses_weights
            num_alldur = np.array(num_alldur) / ses_weights
        den_alldur = np.array(den_alldur)
        self.ses = ses_alldur
        self.num = num_alldur
        self.den = den_alldur
        return ses_alldur, num_alldur, den_alldur

    def _check_others(self, TCEs, mask_time, threshold):
        """candidate_ephemerides.check_others."""
        if self.reference is not None:
            return self._upstream.mask_highest_TCE_and_check_others(
                TCEs=TCEs, time=mask_time.copy(), num=self.num.copy(),
                den=self.den.copy(), tdurs=self.tdurs, threshold=threshold,
            )  # fmt: skip
        if len(TCEs) < 2:
            return TCEs
        from detector.candidate_ephemerides import check_others

        return check_others(
            TCEs, mask_time, np.ascontiguousarray(self.num),
            np.ascontiguousarray(self.den), self.tdurs, threshold,
        )  # fmt: skip

    def Search_for_TCEs(self, ses_cut=20.0, threshold=7.0, remove_hises=True,
                        calc_ses=True, return_df=False, use_tce_mask=False,
                        fill_mode="reflect", **tce_search_kw):  # fmt: skip
        mask = self.lc.mask.copy()
        if use_tce_mask:
            mask &= self.tce_mask
        active = self._active_search_rows(remove_hises, calc_ses, tce_search_kw)
        if self.ses is None or calc_ses:
            whitening = {} if active is None else {"_active_rows": active}
            ses, num, den = self.Calculate_SES_by_Segment(
                mask=mask, fill_mode=fill_mode, **whitening
            )
        else:
            ses, num, den = self.ses.copy(), self.num.copy(), self.den.copy()
        periods = self.search_periods
        time = self.lc.time.copy()
        mask_time = time[mask]
        if len(mask_time) < MIN_SEARCH_CADENCES:
            raise DegenerateCurve(
                f"{len(mask_time)} cadences whiten; {MIN_SEARCH_CADENCES} are needed"
            )
        ses_mask = np.ones(len(ses[0]), dtype=bool)
        single_TCE = None
        if remove_hises:
            hises_ind = np.argmax(ses) % len(ses[0])
            hises_tdur_ind = np.argmax(ses) // len(ses[0])
            hises_t0 = mask_time[hises_ind]
            hises_tdur = self.tdurs[hises_tdur_ind]
            hises_mask = self._upstream.make_transit_mask(
                time=mask_time, P=1e10, t0=hises_t0, dur=hises_tdur
            )
            if np.max(ses[hises_tdur_ind][hises_mask]) < 0.5 * np.max(ses):
                print("\nSingle Transit Candidate at t0={:.2f}\n".format(hises_t0))
                ses_mask &= hises_mask
                single_TCE = pd.DataFrame.from_dict(
                    {
                        "star_id": [self.lc.ID],
                        "period": [-1],
                        "mes": [np.max(ses)],
                        "t0": [hises_t0],
                        "tdur": [hises_tdur],
                    }  # fmt: skip
                )
        nd_ses_mask = np.ix_(np.ones(len(self.tdurs), dtype=bool), ses_mask)
        self.masked_events = []
        TCEs = search_all_durations(
            time=mask_time[ses_mask], num=num[nd_ses_mask], den=den[nd_ses_mask],
            ses=ses[nd_ses_mask], period_sampling=periods, kicid=self.lc.ID,
            t_durs=self.tdurs, rho_star=self.star_density,
            outputfile="tce_detection_test.h5", texp=self.lc.exptime,
            num_transits=self.min_transits, return_df=False, threshold=threshold,
            module=self._upstream,
            grid=self.trial_grid(tce_search_kw.get("dur_range", (0.25, 1.5))),
            upstream_reference=self.reference is not None,
            masked_events=self.masked_events, **tce_search_kw,
        )  # fmt: skip
        TCEs_checked = self._check_others(TCEs, mask_time, threshold)
        if single_TCE is not None:
            if np.isnan(TCEs_checked).any():
                TCEs_checked = single_TCE.to_numpy()
            else:
                TCEs_checked = np.append(single_TCE.to_numpy(), TCEs_checked).reshape(
                    -1, 5
                )
        if active is not None:
            # Partial rows are private scratch; public whitening keeps its own complete cache.
            self.ses = self.num = self.den = None
        if return_df:
            return pd.DataFrame(
                TCEs_checked, columns=["star_id", "period", "mes", "t0", "tdur"]
            ).dropna()
        return TCEs_checked


def dump_class(stage="detector"):
    """The compiled Dump for a stage's own upstream copy (identical files)."""
    module = importlib.import_module(f"{stage}.upstream.dump")
    ses = importlib.import_module(f"{stage}.upstream.ses")
    return type(
        "FastDump",
        (Repeats, FastSearch, module.Dump),
        {"_upstream": module, "_ses": ses, "__module__": __name__},
    )
