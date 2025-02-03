import numpy as np
from jesse.helpers import get_candle_source, slice_candles


def _indicator_name(
    candles: np.ndarray, source_type: str = "close", sequential: bool = False
):
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)
    return src
