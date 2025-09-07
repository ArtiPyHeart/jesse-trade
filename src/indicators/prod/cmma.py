import numpy as np
from jesse import helpers
from jesse.indicators import atr, ma


def cmma(candles, period=14, matype=0, C=1.0, source_type="close", sequential=False):
    candles = helpers.slice_candles(candles, sequential)
    ma_data = ma(
        candles,
        period=period,
        matype=matype,
        source_type=source_type,
        sequential=sequential,
    )
    atr_data = atr(candles, period=period + 1, sequential=sequential)
    src = helpers.get_candle_source(candles, source_type)

    cmma = np.full_like(src, np.nan)
    if len(src) < period:
        return cmma if sequential else cmma[-1]

    cmma_raw = (np.log(src) - np.log(ma_data)) / (atr_data / np.sqrt(period + 1))
    cmma = 100 * (C * cmma_raw) - 50
    if sequential:
        return cmma
    else:
        return cmma[-1:]
