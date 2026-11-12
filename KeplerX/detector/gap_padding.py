"""Repair upstream boundary failures without changing successful padding."""

import numpy as np

from detector.flattened_templates import KNOT_DAYS, SHORTEST_DAYS, flattened


def _filled(time, flux, cadence, mode):
    """Upstream gap arithmetic with cyclic reflection for arbitrarily long gaps."""
    dx = np.diff(time)
    indices, times, values = [], [], []
    negative = -1 * flux + 2.0 * np.median(flux)
    cycle = np.concatenate([flux, negative[::-1]])
    for j in np.flatnonzero(dx > 1.1 * cadence):
        added = np.arange(time[j] + cadence, time[j + 1] - cadence / 2, cadence)
        count = int(np.round(dx[j] / cadence)) - 1
        if len(added) != count:
            raise ValueError("Gap timestamps and cadence count differ")
        indices.extend([j + 1] * count)
        times.extend(added)
        if count < 3 or mode == "line":
            values.extend(
                flux[j] + ((flux[j + 1] - flux[j]) / dx[j]) * (added - time[j])
            )
        elif mode == "reflect":
            left, right = int(np.ceil(count / 2.0)), int(np.floor(count / 2.0))
            if j - left < 0:
                part = cycle[
                    np.arange(j + len(flux) - left, j + len(flux)) % len(cycle)
                ]
            else:
                part = negative[j - left : j]
            values.extend(part[::-1])
            if j + right > len(flux) - 1:
                part = cycle[np.arange(j, j + right) % len(cycle)]
            else:
                part = negative[j : j + right]
            values.extend(part[::-1])
        elif mode == "mean":
            values.extend([np.mean(flux)] * count)
        else:
            raise ValueError(f"Unsupported repaired gap mode: {mode}")
    return (
        np.insert(time, indices, times),
        np.insert(flux, indices, values),
        np.insert(np.ones(len(time), dtype=bool), indices, np.zeros(len(times), bool)),
    )


def pad_time_series(time, flux, cadence, upstream, in_mode="reflect", minimum=4):
    """Use upstream unchanged when it returns consistent wavelet/template arrays."""
    try:
        result = upstream(
            time, flux, cadence=cadence, in_mode=in_mode, pad_end=True, fill_gaps=True
        )
    except IndexError:
        result = None
    if result is not None:
        x, y, keep = result
        if len(x) == len(y) == len(keep) and len(y) >= minimum:
            return result
    x, y, keep = _filled(time, flux, cadence, in_mode)
    size = 1 << int(np.ceil(np.log2(max(len(x), minimum))))
    right = (size - len(x)) // 2
    left = size - len(x) - right
    # Symmetric end padding: upstream's exact expression when each pad fits.
    lvalues = np.resize(y, left)[::-1]
    rvalues = np.resize(y[::-1], right)
    return (
        np.concatenate(
            [
                x[0] - np.arange(1, left + 1)[::-1] * cadence,
                x,
                x[-1] + np.arange(1, right + 1) * cadence,
            ]
        ),
        np.concatenate([lvalues, y, rvalues]),
        np.concatenate([np.zeros(left, bool), keep, np.zeros(right, bool)]),
    )


def padding(dump, time, flux, fill_mode, tdurs):
    minimum = max(
        4, 2 * len(np.arange(dump.lc.exptime, max(tdurs), dump.lc.exptime)) + 1
    )
    return pad_time_series(
        time,
        flux,
        dump.lc.exptime,
        dump._upstream.pad_time_series,
        in_mode=fill_mode,
        minimum=minimum,
    )


# Whitening denominators above this come only from degenerate segments.
DEGENERATE_DENOMINATOR = 1e12
# Too few cadences to search (about two days); upstream fails on them downstream.
MIN_SEARCH_CADENCES = 100


class DegenerateCurve(ValueError):
    """No search is possible: no segment whitens, or too few cadences remain."""


# Gaps longer than this end a segment (docs/keplerX-2026-11-02-detector-speed-on-survey-curves.md).
LONG_GAP_DAYS = 30.0


class Segments:
    """Upstream's segments and whitening, with three changes and repairs."""

    # The golden tests' upstream reference this Dump runs, or None for the Detector.
    reference = None

    def get_transit_model(self, depth, width, pad=None, b=0.25):
        """upstream's template, through KBonus's flattening from SHORTEST_DAYS (detector.flattened_templates)."""
        model = super().get_transit_model(depth, width, pad=pad, b=b)
        if width < SHORTEST_DAYS:
            return model
        return flattened(model, self.lc.exptime, KNOT_DAYS)

    def _baseline(self, flux):
        return flux - np.mean(flux) + 1.0 if len(flux) else flux

    def Calculate_SES_by_Segment(self, gap_size=None, sector_size=None, mask=None,
                                 **calc_ses_kw):  # fmt: skip
        t = self.lc.time
        mask = self.tce_mask if mask is None else mask
        gap_size = self.gap_size if gap_size is None else gap_size
        sector_size = self.sector_size if sector_size is None else sector_size
        starts = [0]
        for gap in np.flatnonzero(np.diff(t) > gap_size):
            if t[gap + 1] - t[gap] > LONG_GAP_DAYS:
                starts.append(int(gap) + 1)
            elif t[gap] - t[starts[-1]] > sector_size:
                starts.append(int(gap))
        segments = self._segment_ses(starts, mask, calc_ses_kw)
        if not segments:
            raise ValueError("No usable cadences for whitening")
        parts = self._whitened_segments(segments, mask)
        self.ses, self.num, self.den = (
            np.concatenate([part[i] for part in parts], axis=1) for i in range(3)
        )
        self._unit_spread(calc_ses_kw.get("tdurs"))
        return self.ses, self.num, self.den

    def _unit_spread(self, tdurs=None):
        """Divide each flattened template's num and ses by the robust spread of its ses."""
        for row, duration in enumerate(self.tdurs if tdurs is None else tdurs):
            if duration < SHORTEST_DAYS:
                continue
            ses = self.ses[row]
            spread = 1.4826 * np.median(np.abs(ses - np.median(ses)))
            if spread > 1.0:
                self.ses[row] = ses / spread
                self.num[row] = self.num[row] / spread

    @staticmethod
    def _whitened_segments(segments, mask):
        """Drop segments whose whitening is degenerate, clearing their cadences in mask."""
        usable = [
            all(np.isfinite(array).all() for array in part)
            and not (part[2] > DEGENERATE_DENOMINATOR).any()
            for _, _, part in segments
        ]
        if not any(usable):
            raise DegenerateCurve("No segment whitens: every one is degenerate")
        if all(usable):
            return [part for _, _, part in segments]
        for (begin, end, _), ok in zip(segments, usable, strict=True):
            if not ok:
                mask[begin:end] = False
        return [part for (_, _, part), ok in zip(segments, usable, strict=True) if ok]

    def _segment_ses(self, starts, mask, calc_ses_kw):
        t = self.lc.time
        parts = []
        for begin, end in zip(starts, starts[1:] + [len(t)], strict=True):
            keep = mask[begin:end]
            if keep.any():
                part = self.Calculate_SES(
                    time=t[begin:end][keep],
                    flux=self.lc.flux[begin:end][keep],
                    mask=keep,
                    **calc_ses_kw,
                )
                parts.append((begin, end, part))
        return parts

    def Calculate_SES(self, time=None, flux=None, mask=None, var_calc="mad",
                      fill_mode="reflect", tdurs=None, ses_weights=None,
                      window_width=7.0, tdepth=100.0, multi_sector=False):  # fmt: skip
        """Native arithmetic; only boundary padding is repaired."""
        mask = self.lc.mask.copy() if mask is None else mask
        time = self.lc.time[mask] if time is None else time
        flux = self._baseline(self.lc.flux[mask] if flux is None else flux)
        durations = self.tdurs if tdurs is None else tdurs
        _, padded, keep = padding(self, time, flux, fill_mode, durations)
        transform = self._upstream.ocwt(padded - 1.0)
        rows = []
        for duration in durations:
            signal = self.get_transit_model(
                depth=tdepth, width=duration, pad=len(padded)
            )
            ses, num, den, sig2 = self._upstream.calculate_SES(
                transform,
                self._upstream.ocwt(signal),
                texp=self.lc.exptime,
                window_size=window_width * duration,
                var_calc=var_calc,
            )
            rows.append((ses[keep], num[keep], den[keep]))
            self.tdur_cdpp.append(np.median(sig2))
        self.ses, self.num, self.den = (
            np.array([row[i] for row in rows]) for i in range(3)
        )
        if ses_weights is not None:
            self.ses = self.ses / ses_weights
            self.num = self.num / ses_weights
        return self.ses, self.num, self.den
