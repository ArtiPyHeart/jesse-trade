import numpy as np
from jesse.helpers import get_candle_source, slice_candles


def autocorrelation_periodogram(
    candles: np.ndarray, source_type: str = "close", sequential: bool = False
):
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    # 根据sequential参数决定返回值，可以返回单个值或多个值，True返回序列，False返回最后一个值
    if sequential:
        return src
    else:
        return src[-1]
