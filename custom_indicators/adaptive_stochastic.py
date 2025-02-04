import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.utils.math import deg_cos, deg_sin


@njit
def _calc_loops(
    src: np.ndarray,
    alpha1: float,
    c1: float,
    c2: float,
    c3: float,
    avg_length: int,
    max_lag: int = 48,
) -> np.ndarray:
    """
    用 numba 加速的子函数，包含对 HP、Filt、自相关、Stochastic 等主要循环的计算过程。
    返回与原逻辑一致的 AdaptiveStochastic 序列。
    """
    length = len(src)
    HP = np.zeros(length)
    Filt = np.zeros(length)

    # 1) 计算高通滤波 HP
    HP[0] = src[0]
    if length > 1:
        HP[1] = src[1]
    for i in range(2, length):
        HP[i] = (
            ((1 - alpha1 / 2) ** 2) * (src[i] - 2 * src[i - 1] + src[i - 2])
            + 2 * (1 - alpha1) * HP[i - 1]
            - (1 - alpha1) ** 2 * HP[i - 2]
        )

    # 2) 计算 Super Smoother Filter (Filt)
    Filt[0] = HP[0]
    if length > 1:
        Filt[1] = HP[1]
    for i in range(2, length):
        Filt[i] = c1 * (HP[i] + HP[i - 1]) / 2.0 + c2 * Filt[i - 1] + c3 * Filt[i - 2]

    # 3) 计算自相关 (Corr)
    Corr = np.zeros(max_lag + 1)
    for Lag in range(max_lag + 1):
        if avg_length != 0:
            M = avg_length
        else:
            M = Lag
        if M <= 0 or (length - Lag) < M:
            Corr[Lag] = 0.0
            continue

        if Lag != 0:
            X = Filt[length - (M + Lag) : length - Lag]
        else:
            X = Filt[length - M : length]
        Y = Filt[length - M : length]

        Sx = 0.0
        Sy = 0.0
        Sxx = 0.0
        Syy = 0.0
        Sxy = 0.0
        for j in range(M):
            x_val = X[j]
            y_val = Y[j]
            Sx += x_val
            Sy += y_val
            Sxx += x_val * x_val
            Syy += y_val * y_val
            Sxy += x_val * y_val

        denom = (M * Sxx - Sx * Sx) * (M * Syy - Sy * Sy)
        if denom > 0:
            Corr[Lag] = (M * Sxy - Sx * Sy) / np.sqrt(denom)
        else:
            Corr[Lag] = 0.0

    # 4) 计算余弦、正弦分量及 R
    CosinePart = np.zeros(max_lag + 1)
    SinePart = np.zeros(max_lag + 1)
    SqSum = np.zeros(max_lag + 1)
    for Period in range(10, max_lag + 1):
        for N in range(3, max_lag + 1):
            CosinePart[Period] += Corr[N] * deg_cos(360 * N / Period)
            SinePart[Period] += Corr[N] * deg_sin(360 * N / Period)
        SqSum[Period] = CosinePart[Period] ** 2 + SinePart[Period] ** 2

    R = np.zeros(max_lag + 1)
    old_R = np.zeros(max_lag + 1)
    for Period in range(10, max_lag + 1):
        old_val = old_R[Period]
        R[Period] = 0.2 * (SqSum[Period] ** 2) + 0.8 * old_val
        old_R[Period] = R[Period]

    # 5) 找最大功率并计算 DominantCycle
    MaxPwr = 0.0
    for Period in range(10, max_lag + 1):
        if R[Period] > MaxPwr:
            MaxPwr = R[Period]

    Pwr = np.zeros(max_lag + 1)
    if MaxPwr != 0:
        for Period in range(3, max_lag + 1):
            Pwr[Period] = R[Period] / MaxPwr

    Spx = 0.0
    Sp = 0.0
    for Period in range(10, max_lag + 1):
        if Pwr[Period] >= 0.5:
            Spx += Period * Pwr[Period]
            Sp += Pwr[Period]
    DominantCycle = 10.0
    if Sp != 0.0:
        DominantCycle = Spx / Sp
    if DominantCycle < 10:
        DominantCycle = 10
    elif DominantCycle > 48:
        DominantCycle = 48

    # 6) 计算自适应随机 (AdaptiveStochastic)
    AdaptiveStochastic = np.zeros(length)
    if length > 0:
        AdaptiveStochastic[0] = 0.5

    for i in range(1, length):
        start_idx = i - int(DominantCycle) + 1
        if start_idx < 0:
            start_idx = 0
        window_size = i - start_idx + 1

        highest_val = Filt[i]
        lowest_val = Filt[i]
        for w in range(window_size):
            val = Filt[start_idx + w]
            if val > highest_val:
                highest_val = val
            if val < lowest_val:
                lowest_val = val

        if highest_val == lowest_val:
            stoc_val = 0.5
        else:
            stoc_val = (Filt[i] - lowest_val) / (highest_val - lowest_val)

        if i >= 2:
            AdaptiveStochastic[i] = (
                c1 * (stoc_val + stoc_val) / 2.0
                + c2 * AdaptiveStochastic[i - 1]
                + c3 * AdaptiveStochastic[i - 2]
            )
        elif i == 1:
            AdaptiveStochastic[i] = stoc_val

    return AdaptiveStochastic


def adaptive_stochastic(
    candles: np.ndarray,
    source_type: str = "close",
    avg_length: int = 3,
    sequential: bool = False,
) -> np.ndarray:
    """
    自适应随机 (Adaptive Stochastic) 指标实现，参考原 11_2AdaptiveStochastic.txt 的思路。
    avg_length: 与原脚本中的 AvgLength(3) 对应，用于控制相关计算时的窗口。
    当 sequential=False 时仅返回最后一个值；否则返回整条序列。
    """
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    if len(src) < 3:
        return np.array([np.nan]) if sequential else np.nan

    length = len(src)

    # 计算高通滤波系数与超平滑滤波系数
    alpha1 = (
        deg_cos(0.707 * 360.0 / 48.0) + deg_sin(0.707 * 360.0 / 48.0) - 1
    ) / deg_cos(0.707 * 360.0 / 48.0)

    a1 = np.exp(-1.414 * np.pi / 10)
    b1 = 2 * a1 * deg_cos(1.414 * 180.0 / 10)
    c2, c3 = b1, -a1 * a1
    c1 = 1 - c2 - c3

    # 调用加速子函数进行主要循环计算
    AdaptiveStochastic = _calc_loops(src, alpha1, c1, c2, c3, avg_length, 48)

    if sequential:
        return AdaptiveStochastic
    else:
        return AdaptiveStochastic[-1]
