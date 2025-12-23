"""
Reverse Momentum Indicator
短期vs长期反转动量指标

计算短期收益与长期收益的差值，用于捕捉市场反转信号
"""

import numpy as np
from jesse import helpers


def reverse_momentum(
    candles: np.ndarray,
    short_period: int = 1,
    long_period: int = 5,
    source_type: str = "close",
    sequential: bool = False,
) -> np.ndarray:
    """
    计算反转动量指标

    公式: (close - close.shift(short)) - (close - close.shift(long))
    简化: close.shift(long) - close.shift(short)

    当短期下跌但长期上涨时，该值为正，表示可能的反转信号

    :param candles: np.ndarray - K线数据
    :param short_period: int - 短期周期 (默认1)
    :param long_period: int - 长期周期 (默认5)
    :param source_type: str - 数据源类型 (默认close)
    :param sequential: bool - 是否返回整个序列
    :return: np.ndarray
    """
    assert short_period > 0, "short_period must be positive"
    assert long_period > short_period, "long_period must be greater than short_period"

    candles = helpers.slice_candles(candles, sequential)
    src = helpers.get_candle_source(candles, source_type)

    n = len(src)
    result = np.full(n, np.nan)

    # 需要 long_period 个历史数据才能计算
    for i in range(long_period, n):
        # (src[i] - src[i-short]) - (src[i] - src[i-long])
        # = src[i-long] - src[i-short]
        short_return = src[i] - src[i - short_period]
        long_return = src[i] - src[i - long_period]
        result[i] = short_return - long_return

    if sequential:
        return result
    else:
        return result[-1:]
