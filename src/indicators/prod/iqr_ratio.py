import numpy as np
from jesse import helpers


def iqr_ratio(candles, source_type="close", period=20, sequential=False):
    candles = helpers.slice_candles(candles, sequential)
    src = helpers.get_candle_source(candles, source_type)

    iqr = np.full_like(src, np.nan)
    if len(src) < period:
        return iqr if sequential else iqr[-1]

    for i in range(len(src)):
        if i < period:
            continue

        raw_data = src[i - period : i]

        raw_iqr = np.percentile(raw_data, 75) - np.percentile(raw_data, 25)
        # 标准化
        iqr[i] = (raw_iqr - np.mean(raw_data)) / np.std(raw_data)

    if sequential:
        return iqr
    else:
        return iqr[-1:]
