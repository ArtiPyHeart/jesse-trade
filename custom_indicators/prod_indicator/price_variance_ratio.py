import numba
import numpy as np
from jesse import helpers
from scipy import stats


@numba.njit
def _calculate_variance(prices, length, use_change=False):
    """计算价格或价格变化的对数方差"""
    if use_change:
        # 使用价格变化(对数收益率)
        terms = np.log(prices[1:] / prices[:-1])
        if len(terms) < length:
            return 0.0
        terms = terms[-length:]
    else:
        # 直接使用价格的对数
        terms = np.log(prices[-length:])

    # 计算均值
    mean = np.mean(terms)

    # 计算方差
    sum_squared = 0.0
    for i in range(len(terms)):
        sum_squared += (terms[i] - mean) ** 2

    return sum_squared / length


@numba.njit
def _calculate_price_variance_ratio(src, period, long_period):
    """计算整个时间序列的价格方差比率"""
    n = len(src)
    result = np.zeros(n)
    front_bad = long_period - 1

    # 初始未定义值设为0
    for i in range(front_bad):
        result[i] = 0.0

    # 计算指标值
    for i in range(front_bad, n):
        # 计算长期方差(分母)
        denom = _calculate_variance(src[: i + 1], long_period, False)

        if denom > 0.0:
            # 计算短期方差(分子)
            numerator = _calculate_variance(src[: i + 1], period, False)
            # 计算方差比率
            ratio = numerator / denom
        else:
            ratio = 1.0

        # 由于numba不支持stats.f.cdf，我们先存储比率，在主函数中应用CDF转换
        result[i] = ratio

    return result


def price_variance_ratio(
    candles: np.ndarray,
    period: int = 14,
    multiplier: int = 2,
    source_type: str = "close",
    sequential: bool = False,
):
    """
    价格方差比率 (Price Variance Ratio)

    基于短期价格方差与长期价格方差的比率来测量价格波动性的变化

    :param candles: np.ndarray
    :param period: 短期周期长度
    :param multiplier: 长期周期乘数 (长期周期 = 短期周期 * 乘数)
    :param source_type: 蜡烛图数据源
    :param sequential: 是否返回整个序列
    :return: float | np.ndarray
    """
    candles = helpers.slice_candles(candles, sequential)
    src = helpers.get_candle_source(candles, source_type)

    # 确保乘数至少为2
    if multiplier < 2:
        multiplier = 2

    # 计算长期周期
    long_period = period * multiplier

    # 使用numba加速的函数计算方差比率
    result = _calculate_price_variance_ratio(src, period, long_period)

    # 应用F分布CDF转换，将结果调整到-50到+50的范围
    for i in range(long_period - 1, len(src)):
        result[i] = (
            100.0 * stats.f.cdf(multiplier * result[i], 2, 2 * multiplier) - 50.0
        )

    # 根据sequential参数返回结果
    if sequential:
        return result
    else:
        return result[-1]
