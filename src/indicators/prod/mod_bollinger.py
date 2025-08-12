import numpy as np
import numpy.typing as npt
from jesse.helpers import get_candle_source, slice_candles


def mod_bollinger(
    candles: npt.NDArray, period: int = 20, factor: float = 2.5, sequential=False
):
    """
    修改版布林带指标

    参数:
        candles: n*6的numpy数组, 包含timestamp, open, close, high, low, volume
        period: 周期，默认20
        factor: 标准差系数，默认2.0
        sequential: 是否返回序列，默认False

    返回:
        如果sequential=True，返回四条线的序列[middle, upper, lower, width]
        如果sequential=False，返回最后一个值[middle, upper, lower, width]
        其中width是布林带宽度，计算方式为(upper-lower)/middle
    """
    # 获取收盘价
    candles = slice_candles(candles, sequential)
    close = get_candle_source(candles, source_type="close")

    # 计算平滑系数
    alpha = 2 / (period + 1)

    # 初始化数组
    mt = np.zeros_like(close)
    ut = np.zeros_like(close)
    dt = np.zeros_like(close)
    mt2 = np.zeros_like(close)
    ut2 = np.zeros_like(close)
    dt2 = np.zeros_like(close)
    but = np.zeros_like(close)
    blt = np.zeros_like(close)
    width = np.zeros_like(close)

    # 第一个值初始化
    mt[0] = close[0]
    ut[0] = close[0]
    dt[0] = close[0]
    mt2[0] = 0
    ut2[0] = 0
    dt2[0] = 0

    # 计算指标
    for i in range(1, len(close)):
        # 计算中间变量
        mt[i] = alpha * close[i] + (1 - alpha) * mt[i - 1]
        ut[i] = alpha * mt[i] + (1 - alpha) * ut[i - 1]
        dt[i] = ((2 - alpha) * mt[i] - ut[i]) / (1 - alpha)

        mt2[i] = alpha * abs(close[i] - dt[i]) + (1 - alpha) * mt2[i - 1]
        ut2[i] = alpha * mt2[i] + (1 - alpha) * ut2[i - 1]
        dt2[i] = ((2 - alpha) * mt2[i] - ut2[i]) / (1 - alpha)

        # 计算带状线
        but[i] = dt[i] + factor * dt2[i]
        blt[i] = dt[i] - factor * dt2[i]

        # 计算带宽
        width[i] = np.where(dt[i] != 0, (but[i] - blt[i]) / dt[i], 0)

    if sequential:
        return dt, but, blt, width
    else:
        return dt[-1], but[-1], blt[-1], width[-1]
