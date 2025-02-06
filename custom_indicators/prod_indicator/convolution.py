import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.utils.math import deg_cos, deg_sin


# 新增：先定义子函数，用于进行Corr/Slope/Convolution的计算
@njit
def _calc_corr_slope_conv(Filt: np.ndarray, length: int, maxN: int):
    corr_arr = np.zeros((length, maxN))
    slope_arr = np.zeros((length, maxN))
    conv_arr = np.zeros((length, maxN))

    for i in range(length):
        for N in range(1, maxN + 1):
            if i - (N - 1) < 0:
                continue

            Sx = Sy = Sxx = Syy = Sxy = 0.0
            for II in range(1, N + 1):
                X = Filt[i - (II - 1)]
                Y = Filt[i - ((N - II))]
                Sx += X
                Sy += Y
                Sxx += X * X
                Syy += Y * Y
                Sxy += X * Y

            denom = (N * Sxx - Sx**2) * (N * Syy - Sy**2)
            if denom > 0:
                corr = (N * Sxy - Sx * Sy) / np.sqrt(denom)
            else:
                corr = 0.0

            mid_idx = i - int(0.5 * N)
            if mid_idx >= 0 and Filt[mid_idx] < Filt[i]:
                slope = -1.0
            else:
                slope = 1.0

            e_val = np.exp(3.0 * corr)
            conv = (1.0 + (e_val - 1.0) / (e_val + 1.0)) / 2.0

            corr_arr[i, N - 1] = corr
            slope_arr[i, N - 1] = slope
            conv_arr[i, N - 1] = conv

    return corr_arr, slope_arr, conv_arr


# 新增基于 John Ehlers Convolution 原逻辑的函数
def ehlers_convolution(
    candles: np.ndarray,
    short_period: int = 40,
    long_period: int = 80,
    source_type: str = "close",
    sequential: bool = False,
):
    """
    基于Ehlers Convolution的Python实现
    返回三个二维数组corr、slope、conv，其大小均为 [len(candles), 48]
    - corr[i, n-1], slope[i, n-1], conv[i, n-1] 分别对应第i根K线的N（1至48）的相关系数、斜率与Convolution值
    """
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    length = len(src)
    # 为确保数组索引安全，若K线数量不足3根，直接返回空值
    if length < 3:
        return np.array([]), np.array([]), np.array([])

    # -------------------
    # 1. High Pass Filter (HP)
    # -------------------
    angle = (1.414 * 360.0) / long_period  # 原逻辑角度值，直接以度为单位计算
    alpha1 = (deg_cos(angle) + deg_sin(angle) - 1.0) / deg_cos(angle)

    HP = np.zeros(length)
    # 初始化前两根，简单赋值
    HP[0], HP[1] = src[0], src[1]
    for i in range(2, length):
        HP[i] = (
            ((1 - alpha1 / 2.0) ** 2) * (src[i] - 2 * src[i - 1] + src[i - 2])
            + 2 * (1 - alpha1) * HP[i - 1]
            - (1 - alpha1) ** 2 * HP[i - 2]
        )

    # -------------------
    # 2. Super Smoother Filter (Filt)
    # -------------------
    a1 = np.exp(-1.414 * np.pi / short_period)
    b1 = 2.0 * a1 * deg_cos(1.414 * 180.0 / short_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    Filt = np.zeros(length)
    # 同样初始化前两根
    Filt[0], Filt[1] = HP[0], HP[1]
    for i in range(2, length):
        Filt[i] = c1 * (HP[i] + HP[i - 1]) / 2.0 + c2 * Filt[i - 1] + c3 * Filt[i - 2]

    # -------------------
    # 3. 调用子函数进行 Corr、Slope、Convolution 计算
    # -------------------
    maxN = 48
    corr_arr, slope_arr, conv_arr = _calc_corr_slope_conv(Filt, length, maxN)

    if sequential:
        return corr_arr[:, 2:], slope_arr[:, 2:], conv_arr[:, 2:]
    else:
        return corr_arr[-1, 2:], slope_arr[-1, 2:], conv_arr[-1, 2:]
