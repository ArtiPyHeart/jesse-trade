import numba
import numpy as np
from jesse import helpers


@numba.njit
def _calculate_money_flow_volume(high, low, close, volume):
    """计算单个K线的资金流量值"""
    if high > low:
        return ((2.0 * close - high - low) / (high - low)) * volume
    else:
        return 0.0


@numba.njit
def _calculate_chaiken_money_flow(highs, lows, closes, volumes, period):
    """计算整个序列的Chaiken's Money Flow值"""
    n = len(closes)
    result = np.zeros(n)

    # 计算每个K线的money flow
    money_flows = np.zeros(n)
    for i in range(n):
        money_flows[i] = _calculate_money_flow_volume(
            highs[i], lows[i], closes[i], volumes[i]
        )

    # 计算前置未定义值
    front_bad = period - 1

    # 初始未定义值设为0
    for i in range(front_bad):
        result[i] = 0.0

    # 计算CMF值
    for i in range(front_bad, n):
        sum_money_flow = 0.0
        sum_volume = 0.0

        # 计算周期内的money flow总和和交易量总和
        for j in range(i - period + 1, i + 1):
            sum_money_flow += money_flows[j]
            sum_volume += volumes[j]

        # 计算CMF
        if sum_volume > 0:
            result[i] = sum_money_flow / sum_volume
        else:
            result[i] = 0.0

    return result


def chaiken_money_flow(
    candles: np.ndarray,
    period: int = 21,
    sequential: bool = False,
):
    """
    Chaiken's Money Flow

    将资金流量乘数与成交量相结合的技术指标，用于衡量市场中的买卖压力

    :param candles: np.ndarray
    :param period: 周期长度
    :param sequential: 是否返回整个序列
    :return: float | np.ndarray
    """
    candles = helpers.slice_candles(candles, sequential)

    # 提取所需的价格和成交量数据
    highs = helpers.get_candle_source(candles, "high")
    lows = helpers.get_candle_source(candles, "low")
    closes = helpers.get_candle_source(candles, "close")
    volumes = helpers.get_candle_source(candles, "volume")

    # 使用numba加速的函数计算CMF
    result = _calculate_chaiken_money_flow(highs, lows, closes, volumes, period)

    # 根据sequential参数返回结果
    if sequential:
        return result
    else:
        return result[-1:]
