import numpy as np
from jesse.helpers import get_candle_source, slice_candles

from src.utils.math_tools import rolling_sum_with_nan


def yang_zhang_volatility(
    candles: np.ndarray,
    period: int = 20,
    sequential=False,
):
    """
    Calculate Yang-Zhang volatility
    """
    candles = slice_candles(candles, sequential=sequential)
    o = get_candle_source(candles, "open")
    h = get_candle_source(candles, "high")
    l = get_candle_source(candles, "low")
    c = get_candle_source(candles, "close")

    c_shift_1 = np.roll(c, 1)
    c_shift_1[0] = np.nan
    o_shift_1 = np.roll(o, 1)
    o_shift_1[0] = np.nan

    high_close_ret = np.log(h / c)
    high_open_ret = np.log(h / o)
    low_close_ret = np.log(l / c)
    low_open_ret = np.log(l / o)

    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    sigma_open_sq = 1 / (period - 1) * rolling_sum_with_nan(high_open_ret**2, period)
    sigma_close_sq = 1 / (period - 1) * rolling_sum_with_nan(low_open_ret**2, period)
    sigma_rs_sq = (
        1
        / (period - 1)
        * rolling_sum_with_nan(
            high_close_ret * high_open_ret + low_close_ret * low_open_ret, period
        )
    )

    res = np.sqrt(sigma_open_sq + k * sigma_close_sq + (1 - k) * sigma_rs_sq)

    if sequential:
        return res
    else:
        return res[-1:]
