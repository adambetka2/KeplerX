"""upstream utils.pad_time_series with its reflection array computed once."""

import numpy as np

from vetter.upstream import utils as upstream


def _reflected(y, negative, j, n_add):
    n_left = int(np.ceil(n_add / 2.0))
    n_right = int(np.floor(n_add / 2.0))
    values = []
    if j - n_left < 0:
        y_repeat_begin = np.concatenate([y, negative[::-1]])
        values.extend(y_repeat_begin[range(j + len(y) - n_left, j + len(y))][::-1])
    else:
        values.extend(negative[range(j - n_left, j)][::-1])
    if j + n_right > len(y) - 1:
        y_repeat_end = np.concatenate([y, negative[::-1], y])
        values.extend(y_repeat_end[range(j, j + n_right)][::-1])
    else:
        values.extend(negative[range(j, j + n_right)][::-1])
    return values


def _gap_values(x, y, dx, j, n_add, cadence, mode, constant_val, negative):
    if mode == "line":
        y1 = y[j]
        y2 = y[j + 1]
        added = np.arange(x[j] + cadence, x[j + 1] - cadence / 2, cadence)
        return list(y1 + ((y2 - y1) / dx[j]) * (added - x[j]))
    if mode == "mean":
        return [np.mean(y)] * n_add
    if mode == "constant":
        return [constant_val] * n_add
    if mode == "reflect":
        return _reflected(y, negative, j, n_add)
    return []


def _filled(x, y, cadence, in_mode, constant_val):
    truth_array = np.ones_like(x, dtype=bool)
    dx = x[1:] - x[:-1]
    bad_idx = np.arange(len(dx))[dx > 1.1 * cadence]
    x_add, y_add, idx2add = [], [], []
    negative = None
    for j in bad_idx:
        x_add.extend(list(np.arange(x[j] + cadence, x[j + 1] - cadence / 2, cadence)))
        n_add = int(np.round(dx[j] / cadence)) - 1
        idx2add.extend([j + 1] * n_add)
        mode = "line" if n_add < 3 else in_mode
        if mode == "reflect" and negative is None:
            negative = -1 * y + 2.0 * np.median(y)
        y_add.extend(
            _gap_values(x, y, dx, j, n_add, cadence, mode, constant_val, negative)
        )
    return (
        np.insert(x, idx2add, x_add),
        np.insert(y, idx2add, y_add),
        np.insert(truth_array, idx2add, np.zeros_like(x_add)),
    )


def _padded_ends(padded_x, padded_y, truth_array, cadence, len_pad):
    if len_pad is None:
        padded_len = int(2 ** np.ceil(np.log2(len(padded_x))))
    else:
        padded_len = len_pad
    total2add = padded_len - len(padded_x)
    add_right = int(np.floor(total2add / 2))
    add_left = int(np.ceil(total2add / 2))
    if total2add == 0:
        return padded_x, padded_y, truth_array, True
    x_add_right = padded_x[-1] + np.arange(1, add_right + 1) * cadence
    x_add_left = padded_x[0] - np.arange(1, add_left + 1)[::-1] * cadence
    padded_x = np.concatenate([x_add_left, padded_x, x_add_right])
    y_add_left = padded_y[:add_left]
    y_add_right = padded_y[-add_right:]
    padded_y = np.concatenate([y_add_left[::-1], padded_y, y_add_right[::-1]])
    truth_array = np.concatenate([np.zeros(add_left), truth_array, np.zeros(add_right)])
    return padded_x, padded_y, truth_array, False


def pad_time_series(x, y, cadence, in_mode="reflect", pad_end=False, fill_gaps=True,
                    constant_val=0.0, len_pad=None, small_gap=True):  # fmt: skip
    if in_mode == "AR":
        return upstream.pad_time_series(
            x, y, cadence, in_mode=in_mode, pad_end=pad_end, fill_gaps=fill_gaps,
            constant_val=constant_val, len_pad=len_pad, small_gap=small_gap,
        )  # fmt: skip
    padded_x, padded_y = x, y
    truth_array = np.ones_like(x, dtype=bool)
    if fill_gaps:
        padded_x, padded_y, truth_array = _filled(x, y, cadence, in_mode, constant_val)
    if pad_end:
        padded_x, padded_y, truth_array, unchanged = _padded_ends(
            padded_x, padded_y, truth_array, cadence, len_pad
        )
        if unchanged:
            return padded_x, padded_y, truth_array
    return padded_x, padded_y, np.array(truth_array, dtype=bool)
