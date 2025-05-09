import os
from multiprocessing import Pool

import numpy as np

from custom_indicators.toolbox.bar.build import build_bar_by_threshold_greater_than
from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy


def build_entropy_bar(
    candles: np.ndarray,
    lag: int,
    entropy_threshold: float,
    max_bars: int = -1,
):
    lag_ret = [
        np.log(candles[i, 2] / candles[i - lag : i, 2])
        for i in range(lag, len(candles))
    ]
    with Pool(processes=max(1, os.cpu_count() - 1)) as pool:
        entropy_array = np.array(
            list(pool.imap(sample_entropy, lag_ret)), dtype=np.float64
        )

    candles = candles[lag:]

    merged_candles = build_bar_by_threshold_greater_than(
        candles, entropy_array, entropy_threshold, max_bars=max_bars, reverse=True
    )
    return merged_candles
