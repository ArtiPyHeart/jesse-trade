import math

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit

from custom_indicators.utils.math import cos_radians, sin_radians


@njit
def _calc_wave(
    src: np.ndarray, length: int, alpha1: float, c1: float, c2: float, c3: float
):
    HP = np.zeros(length, dtype=np.float64)
    Filt = np.zeros(length, dtype=np.float64)
    Wave = np.zeros(length, dtype=np.float64)
    Pwr = np.zeros(length, dtype=np.float64)

    for i in range(length):
        if i == 0:
            HP[i] = 0.0
            Filt[i] = 0.0
        else:
            HP[i] = 0.5 * (1 + alpha1) * (src[i] - src[i - 1]) + alpha1 * HP[i - 1]

            if i == 1:
                Filt[i] = 0.0
            else:
                Filt[i] = (
                    c1 * (HP[i] + HP[i - 1]) / 2.0 + c2 * Filt[i - 1] + c3 * Filt[i - 2]
                )

        if i >= 2:
            Wave[i] = (Filt[i] + Filt[i - 1] + Filt[i - 2]) / 3.0
            Pwr[i] = (Filt[i] ** 2 + Filt[i - 1] ** 2 + Filt[i - 2] ** 2) / 3.0

            if Pwr[i] != 0:
                Wave[i] /= math.sqrt(Pwr[i])

    return Wave


def evenbetter_sinewave(
    candles: np.ndarray,
    duration: int = 40,
    source_type: str = "close",
    sequential: bool = False,
):
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)
    length = len(src)

    # 计算 alpha1 (高通滤波器系数)，确保使用角度计算保持与原始代码一致性
    alpha1 = (1 - sin_radians(360.0 / duration)) / cos_radians(360.0 / duration)

    # 超级平滑滤波器系数
    a1 = math.exp(-1.414 * math.pi / 10)
    b1 = 2 * a1 * cos_radians(1.414 * 180.0 / 10)
    c2 = b1
    c3 = -(a1**2)
    c1 = 1 - c2 - c3

    Wave = _calc_wave(src, length, alpha1, c1, c2, c3)

    if sequential:
        return Wave
    else:
        return Wave[-1]
