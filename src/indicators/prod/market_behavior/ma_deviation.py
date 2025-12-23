"""
MA Deviation Indicator
均线偏离指标

计算当前价格与移动平均线的偏离程度
"""

import numpy as np
from jesse import helpers


def ma_deviation(
    candles: np.ndarray,
    period: int = 5,
    source_type: str = "close",
    sequential: bool = False,
) -> np.ndarray:
    """
    计算均线偏离

    公式: close - close.rolling(period).mean()

    衡量价格偏离均线的程度，正值表示超买，负值表示超卖

    :param candles: np.ndarray - K线数据
    :param period: int - 均线周期 (默认5)
    :param source_type: str - 数据源类型 (默认close)
    :param sequential: bool - 是否返回整个序列
    :return: np.ndarray
    """
    assert period > 0, "period must be positive"

    candles = helpers.slice_candles(candles, sequential)
    src = helpers.get_candle_source(candles, source_type)

    n = len(src)
    result = np.full(n, np.nan)

    # 计算滚动均值并求偏离
    for i in range(period - 1, n):
        ma = np.mean(src[i - period + 1 : i + 1])
        result[i] = src[i] - ma

    if sequential:
        return result
    else:
        return result[-1:]
