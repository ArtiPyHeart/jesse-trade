import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import jit

from custom_indicators.utils.math import deg_cos, deg_sin


@jit(nopython=True)
def _compute_hp(close, n, hplength):
    HP = np.zeros(n)
    angle = 0.707 * 360 / hplength  # 角度制
    alpha1 = (deg_cos(angle) + deg_sin(angle) - 1) / deg_cos(angle)

    for i in range(2, n):
        HP[i] = (
            (1 - alpha1 / 2) ** 2 * (close[i] - 2 * close[i - 1] + close[i - 2])
            + 2 * (1 - alpha1) * HP[i - 1]
            - (1 - alpha1) ** 2 * HP[i - 2]
        )
    return HP


@jit(nopython=True)
def _compute_filt(HP, n, lplength):
    a1 = np.exp(-1.414 * np.pi / lplength)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / lplength)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    Filt = np.zeros(n)
    for i in range(2, n):
        Filt[i] = c1 * (HP[i] + HP[i - 1]) / 2 + c2 * Filt[i - 1] + c3 * Filt[i - 2]
    return Filt


@jit(nopython=True)
def _compute_correlation(r, M, lag):
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for count in range(M):
        X = r[count]
        Y = r[lag + count]
        Sx += X
        Sy += Y
        Sxx += X * X
        Syy += Y * Y
        Sxy += X * Y
    denominator = (M * Sxx - Sx * Sx) * (M * Syy - Sy * Sy)
    corr = 0.0
    if denominator > 0:
        corr = (M * Sxy - Sx * Sy) / np.sqrt(denominator)
    return 0.5 * (corr + 1)


@jit(nopython=True)
def _compute_reversal(r, hplength, avglength):
    sum_deltas = 0
    for lag in range(3, hplength + 1):
        M = avglength if avglength != 0 else lag
        if lag + M > len(r):
            continue

        corr_curr = _compute_correlation(r, M, lag)

        if (lag + M) > (len(r) - 1):
            corr_prev = corr_curr
        else:
            r_prev = r[1:]  # 向前移动一个位置的序列
            corr_prev = _compute_correlation(r_prev, M, lag)

        if (corr_curr > 0.5 and corr_prev < 0.5) or (
            corr_curr < 0.5 and corr_prev > 0.5
        ):
            sum_deltas += 1

    return 1 if sum_deltas > 24 else 0


def autocorrelation_reversals(
    candles: np.ndarray,
    source_type: str = "close",
    hplength: int = 48,
    lplength: int = 10,
    avglength: int = 3,
    sequential: bool = False,
):
    # 参数设置，原始代码中 HPLength = 48, LPLength = 10, AvgLength = 3

    candles = slice_candles(candles, sequential)
    close = get_candle_source(candles, source_type)
    n = len(close)

    # 为确保计算 Pearson 相关时数据足够，至少需要 hplength + avglength + 1 根 K线
    min_required = hplength + avglength + 1
    if n < min_required:
        if sequential:
            return np.full(n, 0.0)
        else:
            return 0.0

    # 1. 计算高通滤波器 HP
    HP = _compute_hp(close, n, hplength)

    # 2. 计算超级平滑滤波器 Filt
    Filt = _compute_filt(HP, n, lplength)

    # 3. 计算 reversal 信号
    if sequential:
        reversal_series = np.full(n, np.nan)
        start_index = hplength + avglength
        for j in range(start_index, n):
            r = Filt[: j + 1][::-1]
            reversal_series[j] = _compute_reversal(r, hplength, avglength)
        reversal_series[:start_index] = 0.0
        return reversal_series
    else:
        r = Filt[::-1]
        reversal = _compute_reversal(r, hplength, avglength)
        return reversal
