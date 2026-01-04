import numpy as np
from jesse.helpers import slice_candles


def natr(
    candles: np.ndarray,
    period: int = 14,
    sequential: bool = False,
):
    """
    NATR - Normalized Average True Range

    使用 Wilder 平滑递推，避免 np.convolve 引发的底层崩溃。
    """
    candles = slice_candles(candles, sequential)
    n = len(candles)
    if n == 0:
        return np.array([]) if sequential else np.nan

    high = candles[:, 3]
    low = candles[:, 4]
    close = candles[:, 2]

    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    if n > 1:
        diff1 = high[1:] - low[1:]
        diff2 = np.abs(high[1:] - close[:-1])
        diff3 = np.abs(low[1:] - close[:-1])
        tr[1:] = np.maximum(np.maximum(diff1, diff2), diff3)

    atr = np.full(n, np.nan, dtype=np.float64)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    natr_arr = (atr / close) * 100.0
    return natr_arr if sequential else natr_arr[-1]
