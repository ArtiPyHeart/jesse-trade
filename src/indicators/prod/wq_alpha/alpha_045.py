"""
Alpha #45: Close-Volume Correlation Product

Formula: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
Note: rank() is ignored for single asset

Type: Correlation-based
Description: Multiple correlation and moving average product.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delay, ts_sum, ts_corr, ts_mean
except ImportError:
    from _operators import ts_delay, ts_sum, ts_corr, ts_mean


def alpha_045(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #45: Close-Volume Correlation Product.

    Formula: -1 * (mean(delay(close, 5), 20) * correlation(close, volume, 2) * correlation(sum(close, 5), sum(close, 20), 2))

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    volume = get_candle_source(candles, "volume")

    # mean(delay(close, 5), 20)
    delayed_close = ts_delay(close, 5)
    mean_delayed = ts_mean(delayed_close, 20)

    # correlation(close, volume, 2)
    corr_cv = ts_corr(close, volume, 2)

    # sum(close, 5) and sum(close, 20)
    sum_5 = ts_sum(close, 5)
    sum_20 = ts_sum(close, 20)

    # correlation(sum(close, 5), sum(close, 20), 2)
    corr_sums = ts_corr(sum_5, sum_20, 2)

    result = -1.0 * mean_delayed * corr_cv * corr_sums

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #45...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_045(candles, sequential=True)
    single_result = alpha_045(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.2f}, {valid.max():.2f}]")
    print("\nAlpha #45 all tests passed!")
