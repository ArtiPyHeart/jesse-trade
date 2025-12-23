"""
Excess Volatility Indicator
超额波动指标

计算当前波动相对于平均波动的超额部分
"""

import numpy as np
from jesse import helpers


def excess_volatility(
    candles: np.ndarray,
    ma_period: int = 5,
    sequential: bool = False,
) -> np.ndarray:
    """
    计算超额波动

    公式: (high - low) - (high - low).rolling(ma_period).mean()

    正值表示当前波动高于平均，负值表示低于平均

    :param candles: np.ndarray - K线数据
    :param ma_period: int - 均线周期 (默认5)
    :param sequential: bool - 是否返回整个序列
    :return: np.ndarray
    """
    assert ma_period > 0, "ma_period must be positive"

    candles = helpers.slice_candles(candles, sequential)
    high = helpers.get_candle_source(candles, "high")
    low = helpers.get_candle_source(candles, "low")

    # 计算高低点差
    hl = high - low

    n = len(hl)
    result = np.full(n, np.nan)

    # 计算滚动均值并求超额
    for i in range(ma_period - 1, n):
        hl_ma = np.mean(hl[i - ma_period + 1 : i + 1])
        result[i] = hl[i] - hl_ma

    if sequential:
        return result
    else:
        return result[-1:]
