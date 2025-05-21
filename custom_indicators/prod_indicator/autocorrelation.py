import math

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.utils.math_tools import deg_cos, deg_sin


@njit
def _apply_filters(src: np.ndarray, alpha1: float, a1: float, b1: float) -> np.ndarray:
    """计算高通滤波和Super Smoother"""
    length = len(src)
    HP = np.zeros(length)
    Filt = np.zeros(length)

    # Super Smoother系数
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    for i in range(length):
        # 高通滤波
        if i >= 2:
            HP[i] = (
                ((1 - alpha1 / 2.0) ** 2) * (src[i] - 2 * src[i - 1] + src[i - 2])
                + 2 * (1 - alpha1) * HP[i - 1]
                - (1 - alpha1) ** 2 * HP[i - 2]
            )
        elif i == 1:
            HP[i] = src[i] - src[i - 1]
        else:
            HP[i] = src[i]

        # Super Smoother
        if i >= 2:
            Filt[i] = (
                c1 * (HP[i] + HP[i - 1]) / 2.0 + c2 * Filt[i - 1] + c3 * Filt[i - 2]
            )
        elif i == 1:
            Filt[i] = (HP[i] + HP[i - 1]) / 2.0
        else:
            Filt[i] = HP[i]

    return Filt


@njit
def _calculate_correlation(Filt: np.ndarray, i: int, lag: int, M: int) -> float:
    """计算单个lag的皮尔逊相关系数"""
    if M <= 0 or i < (lag + M - 1):
        return 0.5

    Sx = Sy = Sxx = Syy = Sxy = 0.0

    for count in range(M):
        X = Filt[i - count]
        Y = Filt[i - lag - count]

        Sx += X
        Sy += Y
        Sxx += X * X
        Syy += Y * Y
        Sxy += X * Y

    denom = (M * Sxx - Sx * Sx) * (M * Syy - Sy * Sy)
    if denom > 0:
        corr = (M * Sxy - Sx * Sy) / np.sqrt(denom)
        # 归一化到[0,1]
        return 0.5 * (corr + 1)
    return 0.5


@njit
def _calculate_all_correlations(
    Filt: np.ndarray, max_lag: int, AvgLength: int
) -> np.ndarray:
    """计算所有时间点的所有lag的相关系数"""
    length = len(Filt)
    Corr_all = np.zeros((length, max_lag + 1))

    for i in range(length):
        for lag in range(max_lag + 1):
            M = AvgLength if AvgLength > 0 else lag
            Corr_all[i, lag] = _calculate_correlation(Filt, i, lag, M)

    return Corr_all


def autocorrelation(
    candles: np.ndarray,
    source_type: str = "close",
    avg_length: int = 0,
    sequential: bool = False,
) -> np.ndarray:
    """主函数：计算自相关指标"""
    # 数据预处理
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    # 修改：使用 EasyLanguage 角度制计算方式
    # 原 EasyLanguage 代码:
    #   alpha1 = (Cosine(.707*360 / 48) + Sine(.707*360 / 48) - 1) / Cosine(.707*360 / 48);
    #   a1 = expvalue(-1.414*3.14159 / 10);
    #   b1 = 2*a1*Cosine(1.414*180 / 10);
    alpha1 = (deg_cos(0.707 * 360 / 48) + deg_sin(0.707 * 360 / 48) - 1) / deg_cos(
        0.707 * 360 / 48
    )
    a1 = np.exp(-1.414 * math.pi / 10)
    b1 = 2.0 * a1 * deg_cos(1.414 * 180 / 10)

    # 应用滤波器
    Filt = _apply_filters(src, alpha1, a1, b1)

    # 计算相关系数
    max_lag = 48
    Corr_all = _calculate_all_correlations(Filt, max_lag, avg_length)

    if sequential:
        return Corr_all[:, 2:]
    else:
        return Corr_all[-1, 2:]
