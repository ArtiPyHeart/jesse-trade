import numba
import numpy as np
from jesse import helpers
from scipy import stats


@numba.njit
def _calculate_reactivity(highs, lows, closes, volumes, period, multiplier):
    """计算整个序列的Reactivity值"""
    n = len(closes)
    result = np.zeros(n)

    # 计算指数平滑常数
    alpha = 2.0 / (period * multiplier + 1)

    # 前置未定义值设为0
    front_bad = period
    for i in range(front_bad):
        result[i] = 0.0

    # 初始化：找到前period个K线的价格范围，同时开始平滑
    lowest = lows[0]
    highest = highs[0]
    smoothed_range = highest - lowest
    smoothed_volume = volumes[0]

    # 初始化计算前period个K线的最高最低价
    for i in range(1, period):
        if highs[i] > highest:
            highest = highs[i]
        if lows[i] < lowest:
            lowest = lows[i]
        smoothed_range = alpha * (highest - lowest) + (1.0 - alpha) * smoothed_range
        smoothed_volume = alpha * volumes[i] + (1.0 - alpha) * smoothed_volume

    # 计算每个有效K线的Reactivity值
    for i in range(front_bad, n):
        # 计算当前窗口的价格范围
        window_lowest = lows[i]
        window_highest = highs[i]

        # 查找period+1个K线中的最高最低价（包括当前K线）
        for j in range(1, period + 1):
            if i - j >= 0:
                if highs[i - j] > window_highest:
                    window_highest = highs[i - j]
                if lows[i - j] < window_lowest:
                    window_lowest = lows[i - j]

        # 更新平滑范围和平滑成交量
        smoothed_range = (
            alpha * (window_highest - window_lowest) + (1.0 - alpha) * smoothed_range
        )
        smoothed_volume = alpha * volumes[i] + (1.0 - alpha) * smoothed_volume

        # 计算纵横比
        aspect_ratio = (window_highest - window_lowest) / (smoothed_range + 1e-10)

        # 处理成交量为零的情况
        if volumes[i] > 0.0 and smoothed_volume > 0.0:
            aspect_ratio /= volumes[i] / smoothed_volume
        else:
            aspect_ratio = 1.0

        # 计算Reactivity值
        price_change = closes[i] - closes[i - period]
        raw_reactivity = aspect_ratio * price_change

        # 归一化
        normalized_reactivity = raw_reactivity / (smoothed_range + 1e-10)

        # 存储结果（不应用正态CDF，因为numba不支持）
        result[i] = normalized_reactivity

    return result, front_bad


def reactivity(
    candles: np.ndarray,
    period: int = 14,
    multiplier: int = 5,
    sequential: bool = False,
):
    """
    Reactivity (反应性指标)

    测量价格变化与价格区间和成交量关系的技术指标

    :param candles: np.ndarray
    :param period: 周期长度
    :param multiplier: 平滑乘数
    :param sequential: 是否返回整个序列
    :return: float | np.ndarray
    """
    candles = helpers.slice_candles(candles, sequential)

    # 提取所需的价格和成交量数据
    highs = helpers.get_candle_source(candles, "high")
    lows = helpers.get_candle_source(candles, "low")
    closes = helpers.get_candle_source(candles, "close")
    volumes = helpers.get_candle_source(candles, "volume")

    # 使用numba加速的函数计算Reactivity
    result, front_bad = _calculate_reactivity(
        highs, lows, closes, volumes, period, multiplier
    )

    # 应用正态CDF转换
    for i in range(front_bad, len(result)):
        result[i] = 100.0 * stats.norm.cdf(0.6 * result[i]) - 50.0

    # 根据sequential参数返回结果
    if sequential:
        return result
    else:
        return result[-1]
