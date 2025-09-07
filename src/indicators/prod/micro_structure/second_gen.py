"""
Second generation models features: Kyle lambda, Amihud Lambda, Hasbrouck lambda (bar and trade based)
"""

import numpy as np
import pandas as pd
from jesse.helpers import get_candle_source, slice_candles

# pylint: disable=invalid-name


def kyle_lambda(
    candles: np.ndarray, window: int = 20, sequential: bool = False
) -> np.ndarray:
    """
    Advances in Financial Machine Learning, p. 286-288.

    Get Kyle lambda from bars data using the Jesse framework.

    :param candles: np.ndarray - Jesse candles data.
    :param window: int - Rolling window used for estimation.
    :param sequential: bool - Whether to return the full sequence.
    :return: np.ndarray - Kyle lambdas.
    """
    candles = slice_candles(candles, sequential)
    close = pd.Series(get_candle_source(candles, source_type="close"))
    volume = pd.Series(get_candle_source(candles, source_type="volume"))

    close_diff = close.diff()
    # Ensure the sign series has the same index as close_diff before filling
    close_diff_sign = (
        close_diff.apply(np.sign)
        .reindex(close_diff.index)
        .replace(0, np.nan)
        .ffill()
        .fillna(0)
    )
    volume_mult_trade_signs = volume * close_diff_sign  # bt * Vt

    # Avoid division by zero, replace inf with nan
    raw_lambda = (close_diff / volume_mult_trade_signs).replace(
        [np.inf, -np.inf], np.nan
    )
    res_pd = raw_lambda.rolling(window=window, min_periods=window).mean()

    # Convert to numpy array and replace NaN with 0
    res = np.nan_to_num(res_pd.to_numpy())

    return res if sequential else res[-1]


def amihud_lambda(
    candles: np.ndarray, window: int = 20, sequential: bool = False
) -> np.ndarray:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from bars data using the Jesse framework.

    :param candles: np.ndarray - Jesse candles data.
    :param window: int - Rolling window used for estimation.
    :param sequential: bool - Whether to return the full sequence.
    :return: np.ndarray - Amihud lambdas.
    """
    candles = slice_candles(candles, sequential)
    close = pd.Series(get_candle_source(candles, source_type="close"))
    volume = pd.Series(get_candle_source(candles, source_type="volume"))
    dollar_volume = close * volume

    returns_abs = np.log(close / close.shift(1)).abs()

    # Avoid division by zero, replace inf with nan
    raw_lambda = (returns_abs / dollar_volume).replace([np.inf, -np.inf], np.nan)
    res_pd = raw_lambda.rolling(window=window, min_periods=window).mean()

    # Convert to numpy array and replace NaN with 0
    res = np.nan_to_num(res_pd.to_numpy())

    return res if sequential else res[-1]


def hasbrouck_lambda(
    candles: np.ndarray, window: int = 20, sequential: bool = False
) -> np.ndarray:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from bars data using the Jesse framework.

    :param candles: np.ndarray - Jesse candles data.
    :param window: int - Rolling window used for estimation.
    :param sequential: bool - Whether to return the full sequence.
    :return: np.ndarray - Hasbrouck lambdas.
    """
    candles = slice_candles(candles, sequential)
    close = pd.Series(get_candle_source(candles, source_type="close"))
    volume = pd.Series(get_candle_source(candles, source_type="volume"))
    dollar_volume = close * volume

    log_ret = np.log(close / close.shift(1))
    # Ensure the sign series has the same index as log_ret before filling
    log_ret_sign = (
        log_ret.apply(np.sign)
        .reindex(log_ret.index)
        .replace(0, np.nan)
        .ffill()
        .fillna(0)
    )

    # Avoid issues with sqrt of zero or negative dollar volume (though unlikely)
    dollar_volume_sqrt = np.sqrt(
        dollar_volume.replace(0, np.nan)
    )  # Replace 0 with NaN before sqrt
    signed_dollar_volume_sqrt = log_ret_sign * dollar_volume_sqrt

    # Avoid division by zero, replace inf with nan
    raw_lambda = (log_ret / signed_dollar_volume_sqrt).replace(
        [np.inf, -np.inf], np.nan
    )
    res_pd = raw_lambda.rolling(window=window, min_periods=window).mean()

    # Convert to numpy array and replace NaN with 0
    res = np.nan_to_num(res_pd.to_numpy())

    return res if sequential else res[-1:]
