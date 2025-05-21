import numpy as np
from jesse.helpers import get_candle_source, slice_candles

from custom_indicators.utils.math_tools import deg_cos, deg_sin


def decycler_oscillator(
    candles: np.ndarray,
    HPPeriod1: int = 30,
    HPPeriod2: int = 60,
    source_type: str = "close",
    sequential: bool = False,
) -> np.ndarray:
    """
    根据John F. Ehlers的DeCycler Oscillator实现, 对应原始EasyLanguage版.

    算法说明:
    1. 根据HPPeriod计算alpha值:
       alpha = (cos(0.707*360/HPPeriod)+ sin(0.707*360/HPPeriod) - 1) / cos(0.707*360/HPPeriod)
    2. 分别使用alpha1和alpha2对收盘价进行二阶高通滤波:
       HP = (1 - alpha/2)^2 * (Close - 2*Close[1] + Close[2]) + 2*(1 - alpha)*HP[1] - (1 - alpha)^2*HP[2]
    3. 指标值为两个滤波结果之差：Decycle = HP2 - HP1

    参数:
        close: 包含收盘价数据的numpy数组.
        HPPeriod1: 第一个高通滤波周期，默认值为30.
        HPPeriod2: 第二个高通滤波周期，默认值为60.
        sequential: 布尔值，若为True返回所有计算的指标值，否则返回最新的指标值.

    返回:
        如果sequential为True，返回包含各K线指标值的numpy数组，否则返回最后一个指标值.
    """

    candles = slice_candles(candles, sequential)
    close = get_candle_source(candles, source_type)

    n = len(close)
    if n < 3:
        return np.full(n, np.nan) if sequential else np.nan

    # 计算角度并转换为alpha值
    angle1 = 0.707 * 360 / HPPeriod1
    angle2 = 0.707 * 360 / HPPeriod2
    a1 = (deg_cos(angle1) + deg_sin(angle1) - 1) / deg_cos(angle1)
    a2 = (deg_cos(angle2) + deg_sin(angle2) - 1) / deg_cos(angle2)

    # 初始化高通滤波数组
    HP1 = np.full(n, np.nan)
    HP2 = np.full(n, np.nan)

    # 设置初始条件（通常递归滤波的初始值设置为0）
    HP1[0] = 0
    HP2[0] = 0
    if n > 1:
        HP1[1] = 0
        HP2[1] = 0

    # 预先计算系数，避免在循环中重复计算
    c1_1 = (1 - a1 / 2) ** 2
    c2_1 = 2 * (1 - a1)
    c3_1 = (1 - a1) ** 2

    c1_2 = (1 - a2 / 2) ** 2
    c2_2 = 2 * (1 - a2)
    c3_2 = (1 - a2) ** 2

    # 从第三个数据开始递归计算高通滤波值
    for i in range(2, n):
        HP1[i] = (
            c1_1 * (close[i] - 2 * close[i - 1] + close[i - 2])
            + c2_1 * HP1[i - 1]
            - c3_1 * HP1[i - 2]
        )
        HP2[i] = (
            c1_2 * (close[i] - 2 * close[i - 1] + close[i - 2])
            + c2_2 * HP2[i - 1]
            - c3_2 * HP2[i - 2]
        )

    decycle = HP2 - HP1

    return decycle if sequential else decycle[-1]
