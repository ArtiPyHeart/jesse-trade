import math

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from src.utils.math_tools import deg_cos, deg_sin


@njit
def _calc_stochastic_matrix(Filt, n, lookback_start, lookback_end, c1, c2, c3):
    """
    使用 numba 优化计算每个 Lookback 下的随机指标矩阵。
    行对应不同 Lookback，列对应每根 K 线。
    """
    num_levels = lookback_end - lookback_start + 1
    # 使用 np.full 直接创建并初始化为 nan 的数组，避免使用循环
    results_matrix = np.full((num_levels, n), np.nan)

    for L in range(lookback_start, lookback_end + 1):
        idx = L - lookback_start
        # 分配 Ratio 和平滑指标 S 的数组，初值设为 nan
        R = np.full(n, np.nan)  # 使用 np.full 直接创建并初始化为 nan 的数组
        S = np.full(n, np.nan)  # 避免使用循环赋值

        # 计算 Ratio：从 t = L-1 开始才有足够的历史数据
        for t in range(L - 1, n):
            highestC = Filt[t - L + 1]
            lowestC = Filt[t - L + 1]
            for j in range(t - L + 1, t + 1):
                val = Filt[j]
                if val > highestC:
                    highestC = val
                if val < lowestC:
                    lowestC = val
            if highestC != lowestC:
                R[t] = (Filt[t] - lowestC) / (highestC - lowestC)
            else:
                R[t] = 0.0

        # 对 Ratio 使用 Super Smoother 做平滑处理得到 S
        if not math.isnan(R[L - 1]):
            S[L - 1] = R[L - 1]
        else:
            S[L - 1] = 0.0

        if L < n:
            if not math.isnan(R[L]) and not math.isnan(S[L - 1]):
                S[L] = c1 * (R[L] + R[L - 1]) / 2.0 + (c2 + c3) * S[L - 1]
                if S[L] < 0.0:
                    S[L] = 0.0
                elif S[L] > 1.0:
                    S[L] = 1.0
            else:
                S[L] = S[L - 1]

            for t in range(L + 1, n):
                if (
                    math.isnan(R[t])
                    or math.isnan(R[t - 1])
                    or math.isnan(S[t - 1])
                    or math.isnan(S[t - 2])
                ):
                    S[t] = float("nan")
                else:
                    S[t] = c1 * (R[t] + R[t - 1]) / 2.0 + c2 * S[t - 1] + c3 * S[t - 2]
                    if S[t] < 0.0:
                        S[t] = 0.0
                    elif S[t] > 1.0:
                        S[t] = 1.0

        for t in range(n):
            results_matrix[idx, t] = S[t]
    return results_matrix


def swamicharts_stochastic(
    candles: np.ndarray, source_type: str = "close", sequential: bool = False
):
    # 切片数据并获取指定数据源（通常是close）
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)
    n = len(src)

    # -------------------------------
    # 1. 计算高通滤波 HP
    # -------------------------------
    # 根据原始代码计算alpha1
    alpha1 = (deg_cos(0.707 * 360 / 48) + deg_sin(0.707 * 360 / 48) - 1) / deg_cos(
        0.707 * 360 / 48
    )

    HP = np.full(n, 0.0)
    # 初始值：t=0, t=1直接设为0
    if n > 0:
        HP[0] = 0.0
    if n > 1:
        HP[1] = 0.0
    # t>=2 递归计算 HP
    for t in range(2, n):
        HP[t] = (
            ((1 - alpha1 / 2) ** 2) * (src[t] - 2 * src[t - 1] + src[t - 2])
            + 2 * (1 - alpha1) * HP[t - 1]
            - ((1 - alpha1) ** 2) * HP[t - 2]
        )

    # -------------------------------
    # 2. 使用 Super Smoother 对 HP 进行平滑，得到 Filt
    # -------------------------------
    # 计算Super Smoother系数
    a1 = np.exp(-1.414 * np.pi / 10)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / 10)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    Filt = np.full(n, 0.0)
    if n > 0:
        Filt[0] = HP[0]
    if n > 1:
        Filt[1] = HP[1]
    for t in range(2, n):
        Filt[t] = c1 * (HP[t] + HP[t - 1]) / 2 + c2 * Filt[t - 1] + c3 * Filt[t - 2]

    # -------------------------------
    # 3. 对每个 Lookback 周期 (5 到 48) 计算随机指标，并返回矩阵形式结果
    # -------------------------------
    lookback_start = 5
    lookback_end = 48
    results_matrix = _calc_stochastic_matrix(
        Filt, n, lookback_start, lookback_end, c1, c2, c3
    )

    lookbacks = np.arange(lookback_start, lookback_end + 1)
    if sequential:
        # 将结果矩阵转置，行对应每根K线，列对应不同Lookback
        output = results_matrix.T
        return lookbacks, output
    else:
        # 返回每个 Lookback 对应的最新指标值
        output = results_matrix[:, -1]
        return lookbacks.reshape(1, -1), output.reshape(1, -1)
