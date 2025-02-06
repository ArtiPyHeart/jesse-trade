import math

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.utils.math import deg_cos, deg_sin


@njit
def _calc_corr(filt, L, avg_length):
    """
    计算给定时刻（K线）下的自相关数组（长度49）。
    """
    corr = np.zeros(49, dtype=np.float64)
    for lag in range(49):
        m = avg_length if avg_length > 0 else lag
        if m == 0 or L - lag - 1 <= 0:
            continue
        max_count = m if m < (L - lag - 1) else (L - lag - 1)
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for count in range(max_count):
            x = filt[L - 1 - count]
            y = filt[L - 1 - lag - count]
            sx += x
            sy += y
            sxx += x * x
            syy += y * y
            sxy += x * y
        denominator = (m * sxx - sx * sx) * (m * syy - sy * sy)
        if denominator > 0:
            corr[lag] = (m * sxy - sx * sy) / math.sqrt(denominator)
    return corr


@njit
def _calc_periodogram(corr):
    """
    计算周期图：返回余弦、正弦分量及其平方和数组。
    """
    cosine_part = np.zeros(49, dtype=np.float64)
    sine_part = np.zeros(49, dtype=np.float64)
    sq_sum = np.zeros(49, dtype=np.float64)
    for period in range(10, 49):
        cp = 0.0
        sp_val = 0.0
        for n in range(3, 49):
            cp += corr[n] * deg_cos(370 * n / period)
            sp_val += corr[n] * deg_sin(370 * n / period)
        cosine_part[period] = cp
        sine_part[period] = sp_val
        sq_sum[period] = cp * cp + sp_val * sp_val
    return cosine_part, sine_part, sq_sum


@njit
def _update_power_and_dom(sq_sum, r_state, max_pwr_state):
    """
    利用周期图的平方和更新递归平滑状态，归一化后计算功率谱。
    """
    pwr = np.zeros(49, dtype=np.float64)
    for period in range(10, 49):
        r_state[period] = 0.2 * (sq_sum[period] ** 2) + 0.8 * r_state[period]
    max_pwr_state = 0.995 * max_pwr_state
    for period in range(10, 49):
        if r_state[period] > max_pwr_state:
            max_pwr_state = r_state[period]
    for period in range(10, 49):
        if max_pwr_state != 0:
            pwr[period] = r_state[period] / max_pwr_state
        else:
            pwr[period] = 0.0
    return pwr, r_state, max_pwr_state


@njit
def _calc_dominant_cycle(pwr):
    """
    基于归一化后的功率谱计算主导周期（加权平均）。
    """
    spx = 0.0
    sp = 0.0
    for period in range(10, 49):
        if pwr[period] >= 0.5:
            spx += period * pwr[period]
            sp += pwr[period]
    if sp != 0:
        return spx / sp
    else:
        return 0.0


def autocorrelation_periodogram(
    candles: np.ndarray,
    source_type: str = "close",
    avg_length: int = 3,
    sequential: bool = False,
):
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type)

    # 计算所需的数组长度
    length = len(source)

    # 初始化输出数组
    hp = np.zeros(length)
    filt = np.zeros(length)

    # 高通滤波器参数
    alpha1 = (deg_cos(0.707 * 360 / 48) + deg_sin(0.707 * 360 / 48) - 1) / deg_cos(
        0.707 * 360 / 48
    )

    # Super Smoother滤波器参数
    a1 = np.exp(-1.414 * np.pi / 10)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / 10)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # 高通滤波
    for i in range(2, length):
        hp[i] = (
            (1 - alpha1 / 2) ** 2 * (source[i] - 2 * source[i - 1] + source[i - 2])
            + 2 * (1 - alpha1) * hp[i - 1]
            - (1 - alpha1) ** 2 * hp[i - 2]
        )

    # Super Smoother滤波
    for i in range(2, length):
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]

    # 计算每根K线对应的主导周期和功率谱（支持 sequential 参数）
    dom_cycle_series = np.full(length, np.nan)
    pwr_series = np.full((length, 49), np.nan)
    r_state = np.zeros(49)  # 用于递归平滑的状态存储
    max_pwr_state = 0
    min_samples = 49  # 至少需要49根K线才能计算完整指标

    for t in range(length):
        if t + 1 < min_samples:
            continue
        L = t + 1
        # 调用子函数计算自相关
        corr = _calc_corr(filt, L, avg_length)
        # 调用子函数计算周期图并获取sq_sum（cosine_part和sine_part此处不再需要）
        _, _, sq_sum = _calc_periodogram(corr)
        # 调用子函数更新功率谱及递归状态，并归一化
        pwr, r_state, max_pwr_state = _update_power_and_dom(
            sq_sum, r_state, max_pwr_state
        )
        # 调用子函数计算主导周期
        dom_cycle = _calc_dominant_cycle(pwr)
        dom_cycle_series[t] = dom_cycle
        pwr_series[t, :] = pwr

    if sequential:
        return dom_cycle_series, pwr_series[:, 10:49]  # 返回所有K线的结果
    else:
        return dom_cycle_series[-1], pwr_series[-1, 10:49]  # 返回最新K线的结果
