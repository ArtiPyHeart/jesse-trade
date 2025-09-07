import math

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from src.utils.math_tools import deg_cos, deg_sin


@njit
def _calc_swamicharts_rsi_inner(Filt, N, L_start, L_end, c1, c2, c3):
    """
    使用 numba 加速的子函数，计算每个 Lookback 下递归平滑的 MyRSI 值。
    Filt：已计算好的滤波序列
    N：K线总数
    L_start：最小 Lookback（如5）
    L_end：最大 Lookback（如48）
    c1, c2, c3：Super Smoother 的平滑参数
    返回值：二维数组，形状 (num_lookbacks, N)，每行对应一个 Lookback 的计算结果，
             对于 i < Lookback，值为 nan。
    """
    num_L = L_end - L_start + 1
    results = np.empty((num_L, N))
    # 初始化全部为 nan
    for i in range(num_L):
        for j in range(N):
            results[i, j] = float("nan")

    for idx in range(num_L):
        L = L_start + idx
        # 计算初始值：当 i == L 时
        closesUp = 0.0
        closesDn = 0.0
        for count in range(L):
            diff = Filt[L - count] - Filt[L - count - 1]
            if diff > 0:
                closesUp += diff
            elif diff < 0:
                closesDn += -diff
        denom = closesUp + closesDn
        if denom != 0:
            current_ratio = closesUp / denom
        else:
            current_ratio = 0.0

        ratio_prev = current_ratio
        myrsi_prev = current_ratio
        myrsi_prev2 = current_ratio
        results[idx, L] = current_ratio

        for i in range(L + 1, N):
            closesUp = 0.0
            closesDn = 0.0
            for count in range(L):
                diff = Filt[i - count] - Filt[i - count - 1]
                if diff > 0:
                    closesUp += diff
                elif diff < 0:
                    closesDn += -diff
            denom = closesUp + closesDn
            if denom != 0:
                current_ratio = closesUp / denom
            else:
                current_ratio = 0.0

            new_myrsi = (
                c1 * (current_ratio + ratio_prev) / 2
                + c2 * myrsi_prev
                + c3 * myrsi_prev2
            )
            if new_myrsi < 0:
                new_myrsi = 0.0
            elif new_myrsi > 1:
                new_myrsi = 1.0

            results[idx, i] = new_myrsi
            ratio_prev = current_ratio
            myrsi_prev2 = myrsi_prev
            myrsi_prev = new_myrsi
    return results


def swamicharts_rsi(
    candles: np.ndarray, source_type: str = "close", sequential: bool = False
):
    """
    根据 Cycle Analytics for Traders 16_1SwamiChartsRSI.txt 实现的 SwamiCharts RSI 指标
    使用步骤：
      1. 先用高通滤波器计算 HP 序列；
      2. 利用 Super Smoother 滤波计算 Filt 序列；
      3. 对于每个 Lookback（5~48），在每根 K 线上采用历史 L+1 根 Filt 值计算局部 RSI（用 ClosesUp 和 ClosesDn 算出 Ratio），
         随后对 Ratio 应用 10 根 Super Smoother 平滑，得到 MyRSI，其值限定在 [0,1] 内。
    返回值：
      若 sequential=True，则返回 (lookbacks, all_values) ，其中 all_values 为 ndarray，每行对应一根 K 线，每列对应一个 Lookback；
      若 sequential=False，则只返回当前 K 线对应的值。
    """
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)
    N = len(src)
    if N < 50:
        # 为保证递归计算正常，至少需要足够的K线数据
        raise ValueError("数据长度不足，至少需要50根K线才能计算SwamiCharts RSI指标。")

    # 初始化 HP 与 Filt 数组（原始计算中使用 Close 值）
    HP = np.zeros(N)
    Filt = np.zeros(N)

    # ===== 计算 HP 相关参数 =====
    # alpha1 = (cos(0.707*360/48) + sin(0.707*360/48) - 1) / cos(0.707*360/48)
    alpha1 = (deg_cos(0.707 * 360 / 48) + deg_sin(0.707 * 360 / 48) - 1) / deg_cos(
        0.707 * 360 / 48
    )
    factor = (1 - alpha1 / 2) ** 2  # 平方计算

    # ===== 计算 Super Smoother 参数 =====
    # a1 = exp(-1.414*pi/10)
    a1 = math.exp(-1.414 * math.pi / 10)
    # b1 = 2 * a1 * cos(1.414*180/10)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / 10)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # ===== 计算 HP 与 Filt 序列 =====
    for i in range(N):
        if i < 2:
            HP[i] = 0  # 初始值设定为0
            Filt[i] = HP[i]
        else:
            HP[i] = (
                factor * (src[i] - 2 * src[i - 1] + src[i - 2])
                + 2 * (1 - alpha1) * HP[i - 1]
                - factor * HP[i - 2]
            )
            Filt[i] = c1 * (HP[i] + HP[i - 1]) / 2 + c2 * Filt[i - 1] + c3 * Filt[i - 2]

    # ===== 对每个 Lookback（5~48）计算局部 RSI 并作平滑（使用 numba 优化子函数） =====
    lookback_start = 5
    lookback_end = 48
    # 调用优化后的 numba 加速子函数，返回结果矩阵形状为 (num_lookbacks, N)
    results_matrix = _calc_swamicharts_rsi_inner(
        Filt, N, lookback_start, lookback_end, c1, c2, c3
    )

    lookbacks = np.arange(lookback_start, lookback_end + 1)
    if sequential:
        # 转置，构造二维数组：行 = 每根K线，列 = 各 Lookback 值
        output = results_matrix.T
        return lookbacks, output
    else:
        # 只返回当前 K 线（最后一行）的 MyRSI 值，对于每个 Lookback
        num_levels = results_matrix.shape[0]
        output = np.empty(num_levels)
        for idx in range(num_levels):
            output[idx] = results_matrix[idx, -1]
        return lookbacks.reshape(1, -1), output.reshape(1, -1)
