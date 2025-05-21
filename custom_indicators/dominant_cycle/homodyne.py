import math

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.utils.math_tools import deg_cos, deg_sin


@njit
def _calc_homodyne_core(close, n, alpha1, c1, c2, c3):
    HP = np.zeros(n)
    Filt = np.zeros(n)
    IPeak = np.zeros(n)
    RealSeries = np.zeros(n)
    Quad = np.zeros(n)
    QPeak = np.zeros(n)
    Imag = np.zeros(n)
    PeriodSeries = np.zeros(n)
    DomCycle = np.zeros(n)

    PeriodSeries[0] = 10.0
    PeriodSeries[1] = 10.0
    DomCycle[0] = 10.0
    DomCycle[1] = 10.0
    IPeak[0] = 1e-8
    IPeak[1] = 1e-8
    QPeak[0] = 1e-8
    QPeak[1] = 1e-8

    for i in range(2, n):
        HP[i] = (
            (1 - alpha1 / 2) ** 2 * (close[i] - 2 * close[i - 1] + close[i - 2])
            + 2 * (1 - alpha1) * HP[i - 1]
            - (1 - alpha1) ** 2 * HP[i - 2]
        )

        Filt[i] = c1 * (HP[i] + HP[i - 1]) / 2 + c2 * Filt[i - 1] + c3 * Filt[i - 2]

        IPeak[i] = 0.991 * IPeak[i - 1]
        if abs(Filt[i]) > IPeak[i]:
            IPeak[i] = abs(Filt[i])
        if IPeak[i] < 1e-8:
            IPeak[i] = 1e-8

        RealSeries[i] = Filt[i] / IPeak[i]

        Quad[i] = RealSeries[i] - RealSeries[i - 1]

        QPeak[i] = 0.991 * QPeak[i - 1]
        if abs(Quad[i]) > QPeak[i]:
            QPeak[i] = abs(Quad[i])
        if QPeak[i] < 1e-8:
            QPeak[i] = 1e-8

        Imag[i] = Quad[i] / QPeak[i]

        Re_calc = RealSeries[i] * RealSeries[i - 1] + Imag[i] * Imag[i - 1]
        Im_calc = RealSeries[i - 1] * Imag[i] - RealSeries[i] * Imag[i - 1]

        period = PeriodSeries[i - 1]
        if Re_calc != 0.0 and Im_calc != 0.0:
            period_candidate = 6.28318 / abs(Im_calc / Re_calc)
            period = period_candidate
        if period < 10.0:
            period = 10.0
        if period > 48.0:
            period = 48.0
        PeriodSeries[i] = period

        DomCycle[i] = (
            c1 * (PeriodSeries[i] + PeriodSeries[i - 1]) / 2
            + c2 * DomCycle[i - 1]
            + c3 * DomCycle[i - 2]
        )
    return DomCycle


def homodyne(
    candles: np.ndarray,
    source_type: str = "close",
    lp_period: int = 20,
    sequential: bool = False,
):
    """
    计算基于Ehlers的Homodyne指标的Dominant Cycle
    参数:
        candles: np.ndarray, 蜡烛数据
        source_type: str, 数据源类型, 默认使用收盘价("close")
        sequential: bool, 是否返回整个指标序列, True返回所有值, False仅返回最后一根K线的值
        lp_period: int, Super Smoother Filter的平滑周期, 默认20
    返回:
        sequential为True时返回整个Dominant Cycle序列, 否则返回最后一个指标值
    """
    candles = slice_candles(candles, sequential)
    close = get_candle_source(candles, source_type)
    n = len(close)
    if n < 3:
        return np.zeros(n) if sequential else 0

    # 计算常量，用于Highpass滤波器和Super Smoother Filter
    alpha1 = (deg_cos(0.707 * 360 / 48) + deg_sin(0.707 * 360 / 48) - 1) / deg_cos(
        0.707 * 360 / 48
    )
    a1 = math.exp(-1.414 * math.pi / lp_period)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / lp_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # 使用numba加速的子函数计算Dominant Cycle序列
    DomCycle = _calc_homodyne_core(close, n, alpha1, c1, c2, c3)

    return DomCycle if sequential else DomCycle[-1]
