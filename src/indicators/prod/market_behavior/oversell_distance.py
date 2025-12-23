"""
Oversell Distance Indicator
超卖距离指标

计算当前价格与周期内最低价的距离
"""

import numpy as np
from jesse import helpers


def oversell_distance(
    candles: np.ndarray,
    period: int = 5,
    sequential: bool = False,
) -> np.ndarray:
    """
    计算超卖距离

    公式: close - low.rolling(period).min()

    该值始终 >= 0，值越接近0表示越接近周期低点（超卖）

    :param candles: np.ndarray - K线数据
    :param period: int - 回看周期 (默认5)
    :param sequential: bool - 是否返回整个序列
    :return: np.ndarray
    """
    assert period > 0, "period must be positive"

    candles = helpers.slice_candles(candles, sequential)
    close = helpers.get_candle_source(candles, "close")
    low = helpers.get_candle_source(candles, "low")

    n = len(close)
    result = np.full(n, np.nan)

    # 计算滚动最低价并求距离
    for i in range(period - 1, n):
        rolling_low = np.min(low[i - period + 1 : i + 1])
        result[i] = close[i] - rolling_low

    if sequential:
        return result
    else:
        return result[-1:]
