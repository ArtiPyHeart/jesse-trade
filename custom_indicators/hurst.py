from typing import Union

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from jesse.indicators import supersmoother
from numba import njit


@njit
def _hurst_calculator(source: np.ndarray, period: int) -> tuple:
    """
    Numba optimized function to calculate Hurst coefficient
    """
    length = len(source)
    dimen = np.zeros(length)
    hurst = np.zeros(length)
    smooth_hurst = np.zeros(length)

    # Super Smoother Filter coefficients
    a1 = np.exp(-1.414 * np.pi / 20)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / 20)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # Calculate for each point
    for i in range(period, length):
        # Calculate N3 using the entire period
        period_high = np.max(source[i - period + 1 : i + 1])
        period_low = np.min(source[i - period + 1 : i + 1])
        n3 = (period_high - period_low) / period

        # Calculate N1 using first half of the period
        half_period = period // 2
        first_half_high = np.max(source[i - period + 1 : i - period + half_period + 1])
        first_half_low = np.min(source[i - period + 1 : i - period + half_period + 1])
        n1 = (first_half_high - first_half_low) / half_period

        # Calculate N2 using second half of the period
        second_half_high = np.max(source[i - half_period + 1 : i + 1])
        second_half_low = np.min(source[i - half_period + 1 : i + 1])
        n2 = (second_half_high - second_half_low) / half_period

        # Calculate dimension and Hurst
        if n1 > 0 and n2 > 0 and n3 > 0:
            if i > period:
                dimen[i] = 0.5 * (
                    (np.log(n1 + n2) - np.log(n3)) / np.log(2) + dimen[i - 1]
                )
            else:
                dimen[i] = 0.5 * ((np.log(n1 + n2) - np.log(n3)) / np.log(2))

        hurst[i] = 2 - dimen[i]

        # Apply Super Smoother
        if i >= 2:
            smooth_hurst[i] = (
                c1 * (hurst[i] + hurst[i - 1]) / 2
                + c2 * smooth_hurst[i - 1]
                + c3 * smooth_hurst[i - 2]
            )

    return dimen, hurst, smooth_hurst


def hurst_coefficient(
    candles: np.ndarray,
    source_type: str = "close",
    period: int = 30,
    sequential: bool = False,
) -> Union[float, np.ndarray]:
    """
    Hurst Coefficient - Measures the long term memory of time series
    :param candles: np.ndarray
    :param source_type: str - default: 'close'
    :param period: int - default: 30
    :param sequential: bool - default: False
    :return: Union[float, np.ndarray]
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)
    source = supersmoother(
        source, period=14, sequential=True
    )  # Always get sequential for internal calculation

    # Calculate using numba optimized function
    dimen, hurst, smooth_hurst = _hurst_calculator(source, period)

    if sequential:
        return smooth_hurst
    else:
        return smooth_hurst[-1]
