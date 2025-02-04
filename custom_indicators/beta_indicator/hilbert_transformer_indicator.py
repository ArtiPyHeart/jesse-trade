from math import exp, fabs, pi

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.utils.math import cos_radians, sin_radians


@njit
def _calc_core(src, length, alpha1, c1, c2, c3, c1_im, c2_im, c3_im):
    HP = np.zeros(length)
    Filt = np.zeros(length)
    IPeak = np.zeros(length)
    QPeak = np.zeros(length)
    Real = np.zeros(length)
    Quadrature = np.zeros(length)
    Imag = np.zeros(length)

    # 初始化前两个元素
    HP[0], HP[1] = src[0], src[1]
    Filt[0], Filt[1] = src[0], src[1]
    # 初始化IPeak和QPeak的前两个值
    IPeak[0] = IPeak[1] = 1e-10
    QPeak[0] = QPeak[1] = 1e-10

    for i in range(2, length):
        # 高通滤波
        HP[i] = (
            (
                (1.0 - alpha1 / 2.0)
                * (1.0 - alpha1 / 2.0)
                * (src[i] - 2.0 * src[i - 1] + src[i - 2])
            )
            + 2.0 * (1.0 - alpha1) * HP[i - 1]
            - (1.0 - alpha1) * (1.0 - alpha1) * HP[i - 2]
        )

        # Super Smoother
        Filt[i] = c1 * (HP[i] + HP[i - 1]) / 2.0 + c2 * Filt[i - 1] + c3 * Filt[i - 2]

        # Real
        IPeak[i] = 0.991 * IPeak[i - 1]
        if fabs(Filt[i]) > IPeak[i]:
            IPeak[i] = fabs(Filt[i])
        Real[i] = Filt[i] / (IPeak[i] + 1e-10)

        # Quadrature
        raw_q = Real[i] - Real[i - 1]
        QPeak[i] = 0.991 * QPeak[i - 1]
        if fabs(raw_q) > QPeak[i]:
            QPeak[i] = fabs(raw_q)
        Quadrature[i] = raw_q / (QPeak[i] + 1e-10)

        # Imag
        Imag[i] = (
            c1_im * (Quadrature[i] + Quadrature[i - 1]) / 2.0
            + c2_im * Imag[i - 1]
            + c3_im * Imag[i - 2]
        )

    return Real, Imag


def hilbert_transformer_indicator(
    candles: np.ndarray,
    lp_period: int = 20,
    source_type: str = "close",
    sequential: bool = False,
):
    """
    Hilbert Transformer 指标，参考 Ehlers 的原指标思路。
    参数：
        candles: np.ndarray - OHLCV 数据
        lp_period: int - Super Smoother 滤波周期
        source_type: str - 选择的数据源，比如 'close', 'open', 'high', 'low' 等
        sequential: bool - 是否返回整条序列
    返回:
        (real, imag) 或 (real序列, imag序列)
    """

    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    length = len(src)
    if length < 3:
        # 数据量太少时不做计算
        return (
            (np.nan, np.nan)
            if not sequential
            else (np.zeros(length) * np.nan, np.zeros(length) * np.nan)
        )

    # ------------------------
    # 第一步：高通滤波
    # ------------------------
    angle = 0.707 * 360.0 / 48.0  # 计算角度（度）
    # 将角度转换为弧度再传入 cos 和 sin
    alpha1 = (cos_radians(angle) + sin_radians(angle) - 1.0) / cos_radians(angle)

    # ------------------------
    # 第二步：Super Smoother
    # ------------------------
    a1 = exp(-1.414 * pi / lp_period)
    # 使用 radians 转换角度：1.414*180/lp_period（度）转换为弧度
    b1 = 2.0 * a1 * cos_radians(1.414 * 180.0 / lp_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1.0 - c2 - c3

    # ------------------------
    # 第三步：Imag 部分所需的参数（不同周期）
    # ------------------------
    a1_im = exp(-1.414 * pi / 10.0)
    # 同样转换角度：1.414*180/10（度）转换为弧度
    b1_im = 2.0 * a1_im * cos_radians(1.414 * 180.0 / 10.0)
    c2_im = b1_im
    c3_im = -a1_im * a1_im
    c1_im = 1.0 - c2_im - c3_im

    # ------------------------
    Real, Imag = _calc_core(src, length, alpha1, c1, c2, c3, c1_im, c2_im, c3_im)

    return (Real, Imag) if sequential else (Real[-1], Imag[-1])
