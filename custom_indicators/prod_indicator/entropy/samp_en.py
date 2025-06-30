import numpy as np
from joblib import Parallel, delayed
from jesse.helpers import slice_candles, get_candle_source

from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.utils.math_tools import (
    log_ret_from_array_price,
    log_ret_from_current_price,
)


def sample_entropy_indicator(
    candles: np.ndarray,
    period: int = 32,
    use_array_price: bool = False,
    source_type: str = "close",
    sequential: bool = False,
):
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    if sequential:
        if use_array_price:
            log_ret_list = log_ret_from_array_price(src, period)
        else:
            log_ret_list = log_ret_from_current_price(src, period)

        entropy_array = Parallel(n_jobs=-2)(
            delayed(sample_entropy_numba)(i) for i in log_ret_list
        )
        entropy_array = np.hstack(([np.nan] * period, entropy_array))
        return entropy_array
    else:
        if use_array_price:
            log_ret = src[-period:-1] / src[-period - 1 : -2]
        else:
            log_ret = src[-1] / src[-period - 1 : -1]
        return sample_entropy_numba(log_ret)
