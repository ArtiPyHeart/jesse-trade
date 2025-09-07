import numba
import numpy as np
from jesse import helpers
from scipy import stats


@numba.njit
def _calculate_norm_obv(closes, volumes, period):
    """计算整个序列的Normalized On Balance Volume值"""
    n = len(closes)
    result = np.zeros(n)

    # 前置未定义值设为0
    front_bad = period
    for i in range(front_bad):
        result[i] = 0.0

    # 计算每个有效点的Normalized OBV
    for i in range(front_bad, n):
        signed_volume = 0.0
        total_volume = 0.0

        # 计算周期内的有符号交易量和总交易量
        for j in range(period):
            k = i - j
            k_prev = i - j - 1

            # 累加有符号交易量和总交易量
            if closes[k] > closes[k_prev]:
                signed_volume += volumes[k]
            elif closes[k] < closes[k_prev]:
                signed_volume -= volumes[k]

            # 总是累加总交易量
            total_volume += volumes[k]

        # 计算标准化值
        if total_volume > 0.0:
            # 计算比值并乘以周期长度的平方根
            value = signed_volume / total_volume
            value *= np.sqrt(period)

            # 存储原始值，稍后应用正态CDF
            result[i] = value
        else:
            result[i] = 0.0

    return result, front_bad


def norm_on_balance_volume(
    candles: np.ndarray,
    period: int = 14,
    sequential: bool = False,
):
    """
    Normalized On Balance Volume (标准化累积交易量)

    衡量价格趋势中的买卖压力平衡的技术指标。
    不同于传统OBV，这个指标计算了特定周期内的买卖成交量比例，并进行了归一化处理。

    :param candles: np.ndarray
    :param period: 计算周期
    :param sequential: 是否返回整个序列
    :return: float | np.ndarray
    """
    candles = helpers.slice_candles(candles, sequential)

    # 提取收盘价和成交量
    closes = helpers.get_candle_source(candles, "close")
    volumes = helpers.get_candle_source(candles, "volume")

    # 使用numba加速的函数计算标准化OBV
    result, front_bad = _calculate_norm_obv(closes, volumes, period)

    # 应用正态CDF转换
    for i in range(front_bad, len(result)):
        result[i] = 100.0 * stats.norm.cdf(0.6 * result[i]) - 50.0

    # 根据sequential参数返回结果
    if sequential:
        return result
    else:
        return result[-1:]
