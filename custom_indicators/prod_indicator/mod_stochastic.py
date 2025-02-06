import math
from typing import Union

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.prod_indicator.roofing_filter import _roofing_filter
from custom_indicators.utils.math import deg_cos, deg_sin


@njit
def _compute_stochastic(source, length):
    # 计算高通滤波器参数
    angle = 0.707 * 360 / 48  # 将角度计算为度数，符合原始代码（0.707*360/48）
    alpha1 = (deg_cos(angle) + deg_sin(angle) - 1) / deg_cos(angle)

    # 初始化数组
    hp = np.zeros_like(source)
    filt = np.zeros_like(source)
    result = np.zeros_like(source)

    # 高通滤波器
    for i in range(2, len(source)):
        hp[i] = (
            (1 - alpha1 / 2) ** 2 * (source[i] - 2 * source[i - 1] + source[i - 2])
            + 2 * (1 - alpha1) * hp[i - 1]
            - (1 - alpha1) ** 2 * hp[i - 2]
        )

    # Super Smoother Filter参数
    a1 = math.exp(-1.414 * math.pi / 10)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / 10)  # 1.414*180/10，角度制，与原始代码一致
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # 应用Super Smoother Filter
    for i in range(2, len(source)):
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]

    # 计算移动窗口内的最高值和最低值
    for i in range(length, len(source)):
        window = filt[i - length + 1 : i + 1]
        highest = np.max(window)
        lowest = np.min(window)
        if highest != lowest:
            stoc = (filt[i] - lowest) / (highest - lowest)
        else:
            stoc = 0.5

        # 最后的平滑
        result[i] = (
            c1 * (stoc + result[i - 1]) / 2 + c2 * result[i - 1] + c3 * result[i - 2]
        )

    return result


def mod_stochastic(
    candles: np.ndarray,
    length: int = 20,
    source_type: str = "close",
    roofing_filter: bool = False,
    hp_period: int = 48,
    lp_period: int = 10,
    sequential: bool = False,
) -> Union[float, np.ndarray]:
    """
    修改版的Stochastic指标

    参数:
        candles (np.ndarray): K线数据
        length (int): 计算周期长度
        source_type (str): 使用的价格类型
        roofing_filter (bool): 是否使用roofing filter
        hp_period (int): 高通滤波周期
        lp_period (int): 低通滤波周期
        sequential (bool): 是否返回序列数据

    返回:
        Union[float, np.ndarray]: 计算结果
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    if roofing_filter:
        source = _roofing_filter(source, hp_period=hp_period, lp_period=lp_period)
    res = _compute_stochastic(source, length)

    return res if sequential else res[-1]
