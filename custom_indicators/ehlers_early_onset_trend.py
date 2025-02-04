from typing import Union

import numpy as np
import numpy.typing as npt
from jesse.helpers import get_candle_source, slice_candles

from custom_indicators.utils.math import deg_cos, deg_sin


def ehlers_early_onset_trend(
    candles: npt.NDArray,
    lp_period: int = 30,
    k: float = 0.85,
    source_type: str = "close",
    sequential: bool = False,
) -> Union[float, npt.NDArray]:
    """
    计算Ehlers Early Onset Trend指标

    参数:
        candles: numpy数组，包含timestamp, open, close, high, low, volume
        lp_period: 平滑周期，默认30
        k: 系数K，默认0.85
        source_type: 使用的价格类型，默认close
        sequential: 是否返回序列，默认False

    返回:
        如果sequential=True，返回指标的整个序列
        如果sequential=False，返回最新的指标值
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)

    hp = np.zeros_like(source)
    filt = np.zeros_like(source)
    peak = np.zeros_like(source)
    quotient = np.zeros_like(source)

    # 高通滤波器参数
    alpha1 = (deg_cos(0.707 * 360 / 48) + deg_sin(0.707 * 360 / 48) - 1) / deg_cos(
        0.707 * 360 / 48
    )

    # SuperSmoother Filter参数
    a1 = np.exp(-1.414 * np.pi / lp_period)
    b1 = 2 * a1 * deg_cos(1.414 * 180 / lp_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    for i in range(2, len(source)):
        # 高通滤波器
        hp[i] = (
            (1 - alpha1 / 2)
            * (1 - alpha1 / 2)
            * (source[i] - 2 * source[i - 1] + source[i - 2])
            + 2 * (1 - alpha1) * hp[i - 1]
            - (1 - alpha1) * (1 - alpha1) * hp[i - 2]
        )

        # SuperSmoother Filter
        filt[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]

        # 快速攻击-慢速衰减算法
        peak[i] = 0.991 * peak[i - 1]
        if abs(filt[i]) > peak[i]:
            peak[i] = abs(filt[i])

        # 归一化roofing filter
        if peak[i] != 0:
            x = filt[i] / peak[i]
            quotient[i] = (x + k) / (k * x + 1)

    if sequential:
        return quotient
    else:
        return quotient[-1]
