import numpy as np
from jesse.helpers import get_candle_source


def bar_duration(
    candles: np.ndarray,
    sequential=False,
) -> np.ndarray:
    timestamps = candles[:, 0]
    durations = np.full_like(timestamps, 0, dtype=np.float64)
    durations[1:] = (timestamps[1:] - timestamps[:-1]) / 60000
    if sequential:
        return durations
    else:
        return durations[-1:]


def bar_open(candles: np.ndarray, sequential=False) -> np.ndarray:
    o = get_candle_source(candles, "open")
    if sequential:
        return o
    else:
        return o[-1:]


def bar_high(candles: np.ndarray, sequential=False) -> np.ndarray:
    h = get_candle_source(candles, "high")
    if sequential:
        return h
    else:
        return h[-1:]


def bar_low(candles: np.ndarray, sequential=False) -> np.ndarray:
    l = get_candle_source(candles, "low")
    if sequential:
        return l
    else:
        return l[-1:]


def bar_close(
    candles: np.ndarray,
    sequential=False,
) -> np.ndarray:
    c = get_candle_source(candles, "close")
    if sequential:
        return c
    else:
        return c[-1:]
