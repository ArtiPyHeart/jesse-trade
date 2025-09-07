import numpy as np
from jesse import helpers
from jesse.indicators import atr
from numba import njit
from scipy.stats import norm


@njit
def _compute_rolling_avgs(price_changes, n, short_length, long_length, front_bad):
    short_sum = np.zeros(n)
    long_sum = np.zeros(n)
    for i in range(front_bad, n):
        s = 0.0
        for j in range(i - short_length + 1, i + 1):
            s += price_changes[j]
        short_sum[i] = s / short_length
        s2 = 0.0
        for j in range(i - long_length + 1, i + 1):
            s2 += price_changes[j]
        long_sum[i] = s2 / long_length
    return short_sum, long_sum


def price_change_oscillator(
    candles: np.ndarray,
    period: int = 10,
    multiplier: int = 10,
    source_type: str = "close",
    sequential: bool = False,
) -> np.ndarray:
    """
    价格变化震荡指标(Price Change Oscillator)

    参数:
        candles (np.ndarray): 价格K线数据
        period (int): 短周期长度
        multiplier (int): 长周期与短周期的倍数，默认为2
        source_type (str): 价格数据类型，默认为收盘价
        sequential (bool): 是否返回完整序列数据

    返回:
        np.ndarray: 价格变化震荡指标值
    """
    candles = helpers.slice_candles(candles, sequential)
    src = helpers.get_candle_source(candles, source_type)

    # 确保multiplier至少为2
    if multiplier < 2:
        multiplier = 2

    # 计算长短周期长度
    short_length = period
    long_length = short_length * multiplier

    # 计算价格变化的绝对对数值
    price_changes = np.abs(np.log(src[1:] / src[:-1]))
    price_changes = np.insert(price_changes, 0, 0)  # 第一个值设为0

    # 初始化结果数组
    result = np.zeros_like(src)

    # 计算前部无效值的数量
    front_bad = long_length

    # 使用numba加速的私有函数计算滚动均值
    short_sum, long_sum = _compute_rolling_avgs(
        price_changes, len(src), short_length, long_length, front_bad
    )

    # 计算标准化因子
    denom = 0.36 + 1.0 / short_length  # 适用于multiplier = 2
    v = np.log(0.5 * multiplier) / 1.609  # multiplier=2时为0，multiplier=10时为1
    denom += 0.7 * v  # 适用于multiplier = 2-10

    # 获取ATR值
    atr_values = atr(candles, long_length, True)

    # 计算最终的震荡指标值
    valid_indices = np.where((denom * atr_values) > 1.0e-20)[0]
    valid_indices = valid_indices[valid_indices >= front_bad]

    if len(valid_indices) > 0:
        diff = short_sum[valid_indices] - long_sum[valid_indices]
        normalized_diff = diff / (denom * atr_values[valid_indices])
        result[valid_indices] = 100.0 * norm.cdf(4.0 * normalized_diff) - 50.0

    if sequential:
        return result
    else:
        return result[-1:]
