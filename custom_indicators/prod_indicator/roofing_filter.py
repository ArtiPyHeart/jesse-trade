from typing import Union

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.utils.math_tools import deg_cos, deg_sin


@njit
def _roofing_filter(
    source: np.ndarray,
    hp_period: int,
    lp_period: int,
    zero_mean: bool = False,
) -> np.ndarray:
    """
    Roofing Filter的核心计算逻辑

    :param source: np.ndarray - 输入数据
    :param hp_period: int - 高通滤波周期
    :param lp_period: int - 低通滤波周期
    :param zero_mean: bool - 是否为零均值滤波
    :return: np.ndarray
    """
    # 修正高通滤波器参数计算（使用度数为单位的三角函数计算）
    angle = 360 / hp_period
    alpha1 = (deg_cos(angle) + deg_sin(angle) - 1) / deg_cos(angle)

    hp = np.zeros_like(source)
    filt = np.zeros_like(source)
    if zero_mean:
        filt2 = np.zeros_like(source)

    # 修正高通滤波计算（使用一阶差分）
    for i in range(1, len(source)):
        hp[i] = (1 - alpha1 / 2) * (source[i] - source[i - 1]) + (1 - alpha1) * hp[
            i - 1
        ]

    # 计算超级平滑器参数
    rad2 = 1.414 * np.pi / lp_period
    a1 = np.exp(-rad2)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / lp_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # 应用超级平滑器
    for i in range(2, len(hp)):
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]

    if zero_mean:
        # 新增第二级高通滤波
        for i in range(1, len(filt)):
            filt2[i] = (1 - alpha1 / 2) * (filt[i] - filt[i - 1]) + (
                1 - alpha1
            ) * filt2[i - 1]

    return filt2 if zero_mean else filt


def roofing_filter(
    candles: np.ndarray,
    hp_period: int = 80,
    lp_period: int = 40,
    source_type: str = "close",
    zero_mean: bool = True,
    sequential: bool = False,
) -> Union[float, np.ndarray]:
    """
    Roofing Filter - 一个由John Ehlers开发的趋势跟踪指标

    :param candles: np.ndarray
    :param hp_period: int - 高通滤波周期
    :param lp_period: int - 低通滤波周期
    :param source_type: str - 输入数据类型
    :param sequential: bool - 是否返回整个序列
    :return: Union[float, np.ndarray]
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)

    # 参数验证
    if hp_period < 1 or lp_period < 1:
        raise ValueError("周期参数必须大于0")

    if hp_period <= lp_period:
        raise ValueError("高通滤波周期必须大于低通滤波周期")

    filt = _roofing_filter(source, hp_period, lp_period, zero_mean)

    return filt if sequential else filt[-1]
