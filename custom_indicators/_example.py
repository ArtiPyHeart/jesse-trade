import numpy as np
from jesse.helpers import get_candle_source, slice_candles


def _indicator_name(
    candles: np.ndarray, source_type: str = "close", sequential: bool = False
):
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)
    another_src = get_candle_source(candles, "high")

    # 根据sequential参数决定返回值，可以返回单个值或多个值，True返回序列，False返回最后一个值
    if sequential:
        return src, another_src
    else:
        return src[-1], another_src[-1]
