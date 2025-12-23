"""
HL Diff Indicator
高低点差指标

计算每根K线的高低点之差，表示波动幅度
"""

import numpy as np
from jesse import helpers


def hl_diff(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    计算高低点差

    公式: high - low

    表示单根K线的波动幅度

    :param candles: np.ndarray - K线数据
    :param sequential: bool - 是否返回整个序列
    :return: np.ndarray
    """
    candles = helpers.slice_candles(candles, sequential)
    high = helpers.get_candle_source(candles, "high")
    low = helpers.get_candle_source(candles, "low")

    result = high - low

    if sequential:
        return result
    else:
        return result[-1:]
