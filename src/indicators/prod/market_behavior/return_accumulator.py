"""
Return Accumulator Indicator
收益累加器指标

计算过去N期收益率的累加和
"""

import numpy as np
from jesse import helpers


def return_accumulator(
    candles: np.ndarray,
    period: int = 3,
    source_type: str = "close",
    sequential: bool = False,
) -> np.ndarray:
    """
    计算收益累加器

    公式: (close - close.shift(1)).rolling(period).sum()

    累加过去N期的收益率，用于衡量短期动量

    :param candles: np.ndarray - K线数据
    :param period: int - 累加周期 (默认3)
    :param source_type: str - 数据源类型 (默认close)
    :param sequential: bool - 是否返回整个序列
    :return: np.ndarray
    """
    assert period > 0, "period must be positive"

    candles = helpers.slice_candles(candles, sequential)
    src = helpers.get_candle_source(candles, source_type)

    n = len(src)
    result = np.full(n, np.nan)

    # 计算逐期收益
    returns = np.diff(src, prepend=np.nan)

    # 滚动求和
    for i in range(period, n):
        result[i] = np.sum(returns[i - period + 1 : i + 1])

    if sequential:
        return result
    else:
        return result[-1:]
