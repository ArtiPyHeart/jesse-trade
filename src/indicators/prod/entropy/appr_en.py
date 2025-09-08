import numpy as np
from jesse.helpers import get_candle_source
from joblib import delayed, Parallel

from src.data_process.entropy.apen_sampen import approximate_entropy_numba
from src.utils.math_tools import (
    log_ret_from_array_price,
    log_ret_from_current_price,
)


def approximate_entropy_indicator(
    candles: np.ndarray,
    period: int = 32,
    use_array_price: bool = True,
    source_type: str = "close",
    sequential: bool = False,
):
    src = get_candle_source(candles, source_type)

    if sequential:
        if use_array_price:
            log_ret_list = log_ret_from_array_price(src, period)
        else:
            log_ret_list = log_ret_from_current_price(src, period)

        entropy_array = Parallel()(
            delayed(approximate_entropy_numba)(i) for i in log_ret_list
        )
        entropy_array = np.hstack(([np.nan] * period, entropy_array))
        return entropy_array
    else:
        if use_array_price:
            log_ret = src[-period:-1] / src[-period - 1 : -2]
        else:
            log_ret = src[-1] / src[-period - 1 : -1]
        return np.array([approximate_entropy_numba(log_ret)])
