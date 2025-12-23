"""
HL Diff MA Indicator
波动幅度均线指标

计算高低点差的移动平均，表示平均波动幅度
"""

import numpy as np
from jesse import helpers


def hl_diff_ma(
    candles: np.ndarray,
    period: int = 5,
    sequential: bool = False,
) -> np.ndarray:
    """
    计算波动幅度均线

    公式: (high - low).rolling(period).mean()

    表示过去N期的平均波动幅度

    :param candles: np.ndarray - K线数据
    :param period: int - 均线周期 (默认5)
    :param sequential: bool - 是否返回整个序列
    :return: np.ndarray
    """
    assert period > 0, "period must be positive"

    candles = helpers.slice_candles(candles, sequential)
    high = helpers.get_candle_source(candles, "high")
    low = helpers.get_candle_source(candles, "low")

    # 计算高低点差
    hl = high - low

    n = len(hl)
    result = np.full(n, np.nan)

    # 计算滚动均值
    for i in range(period - 1, n):
        result[i] = np.mean(hl[i - period + 1 : i + 1])

    if sequential:
        return result
    else:
        return result[-1:]
