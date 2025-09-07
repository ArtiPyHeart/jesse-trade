"""
First generation features (Roll Measure/Impact, Corwin-Schultz spread estimator)
"""

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import njit


@njit
def _get_roll_measure(close_prices: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Advances in Financial Machine Learning, page 282.
    Get Roll Measure - 使用Numba加速的核心计算函数

    Roll Measure gives the estimate of effective bid-ask spread
    without using quote-data.
    """
    length = len(close_prices)
    roll_measure = np.zeros(length)

    for i in range(window, length):
        # 手动计算价格差分，替代np.diff
        price_diff = np.zeros(window)
        for j in range(window):
            price_diff[j] = (
                close_prices[i - window + j + 1] - close_prices[i - window + j]
            )

        # 计算price_diff和其lag之间的协方差
        price_diff_lag = price_diff[:-1]
        price_diff = price_diff[1:]

        # 手动计算协方差
        mean_diff = np.mean(price_diff)
        mean_lag = np.mean(price_diff_lag)
        cov = np.mean((price_diff - mean_diff) * (price_diff_lag - mean_lag))

        roll_measure[i] = 2 * np.sqrt(abs(cov))

    return roll_measure


@njit
def _get_beta(high: np.ndarray, low: np.ndarray, window: int) -> np.ndarray:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.
    Get beta estimate from Corwin-Schultz algorithm - 使用Numba加速的核心计算函数
    """
    length = len(high)
    beta = np.zeros(length)

    for i in range(window, length):
        ret = np.log(high[i - window : i] / low[i - window : i])
        high_low_ret = ret * ret
        # 每两个值求和
        pairs_sum = np.zeros(window // 2)
        for j in range(window // 2):
            pairs_sum[j] = high_low_ret[j * 2] + high_low_ret[j * 2 + 1]
        beta[i] = np.mean(pairs_sum)

    return beta


@njit
def _get_gamma(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.
    Get gamma estimate from Corwin-Schultz algorithm - 使用Numba加速的核心计算函数
    """
    length = len(high)
    gamma = np.zeros(length)

    for i in range(2, length):
        high_max = max(high[i], high[i - 1])
        low_min = min(low[i], low[i - 1])
        gamma[i] = np.log(high_max / low_min) ** 2

    return gamma


@njit
def _get_alpha(beta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.
    Get alpha from Corwin-Schultz algorithm - 使用Numba加速的核心计算函数
    """
    length = len(beta)
    alpha = np.zeros(length)
    den = 3 - 2 * 2**0.5

    for i in range(len(beta)):
        if beta[i] > 0:
            alpha[i] = ((2**0.5 - 1) * beta[i] ** 0.5) / den
            if gamma[i] > 0:
                alpha[i] -= (gamma[i] / den) ** 0.5
            if alpha[i] < 0:
                alpha[i] = 0

    return alpha


def roll_measure(
    candles: np.ndarray, window: int = 20, sequential: bool = False
) -> np.ndarray:
    """
    Roll Measure指标的Jesse接口函数

    :param candles: np.ndarray - Jesse K线数据
    :param window: int - 计算窗口大小
    :param sequential: bool - 是否返回完整序列
    :return: np.ndarray - Roll measure指标值
    """
    candles = slice_candles(candles, sequential)
    close = get_candle_source(candles, source_type="close")
    res = _get_roll_measure(close, window)

    return res if sequential else res[-1]


def roll_impact(
    candles: np.ndarray, window: int = 20, sequential: bool = False
) -> np.ndarray:
    """
    Roll Impact指标的Jesse接口函数

    :param candles: np.ndarray - Jesse K线数据
    :param window: int - 计算窗口大小
    :param sequential: bool - 是否返回完整序列
    :return: np.ndarray - Roll impact指标值
    """
    candles = slice_candles(candles, sequential)
    close = get_candle_source(candles, source_type="close")
    volume = get_candle_source(candles, source_type="volume")

    roll_measure_values = _get_roll_measure(close, window)
    dollar_volume = close * volume

    # 避免除以0
    dollar_volume = np.where(dollar_volume == 0, np.inf, dollar_volume)
    res = roll_measure_values / dollar_volume

    return res if sequential else res[-1]


def corwin_schultz_estimator(
    candles: np.ndarray, window: int = 20, sequential: bool = False
) -> np.ndarray:
    """
    Corwin-Schultz spread estimator指标的Jesse接口函数

    :param candles: np.ndarray - Jesse K线数据
    :param window: int - 计算窗口大小
    :param sequential: bool - 是否返回完整序列
    :return: np.ndarray - Corwin-Schultz spread estimator指标值
    """
    candles = slice_candles(candles, sequential)
    high = get_candle_source(candles, source_type="high")
    low = get_candle_source(candles, source_type="low")

    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)
    alpha = _get_alpha(beta, gamma)

    spread = np.zeros_like(alpha)
    valid = alpha > 0
    spread[valid] = 2 * (np.exp(alpha[valid]) - 1) / (1 + np.exp(alpha[valid]))

    return spread if sequential else spread[-1]


def bekker_parkinson_vol(
    candles: np.ndarray, window: int = 20, sequential: bool = False
) -> np.ndarray:
    """
    Bekker-Parkinson volatility指标的Jesse接口函数

    :param candles: np.ndarray - Jesse K线数据
    :param window: int - 计算窗口大小
    :param sequential: bool - 是否返回完整序列
    :return: np.ndarray - Bekker-Parkinson volatility指标值
    """
    candles = slice_candles(candles, sequential)
    high = get_candle_source(candles, source_type="high")
    low = get_candle_source(candles, source_type="low")

    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)

    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2**0.5

    sigma = np.zeros_like(beta)
    valid = (beta > 0) | (gamma > 0)
    sigma[valid] = (2**-0.5 - 1) * beta[valid] ** 0.5 / (k2 * den)
    sigma[valid] += (gamma[valid] / (k2**2 * den)) ** 0.5
    sigma[sigma < 0] = 0

    # 确保窗口期之前的值为0
    sigma[:window] = 0

    return sigma if sequential else sigma[-1:]
