"""
Alpha #27: Volume-VWAP Correlation

Formula: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1) : 1)
Note: rank() is ignored for single asset, VWAP approximated as (H+L+C)/3

Type: Correlation-based (VWAP)
Description: Signal based on volume-vwap correlation threshold.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_corr, ts_sum, get_vwap
except ImportError:
    from _operators import ts_corr, ts_sum, get_vwap


def alpha_027(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #27: Volume-VWAP Correlation.

    Formula: if (sum(correlation(volume, vwap, 6), 2) / 2.0) > 0.5: -1 else: 1

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")
    volume = get_candle_source(candles, "volume")

    vwap = get_vwap(high, low, close)
    corr = ts_corr(volume, vwap, 6)
    sum_corr = ts_sum(corr, 2)
    avg_corr = sum_corr / 2.0

    # Condition: avg_corr > 0.5
    result = np.where(avg_corr > 0.5, -1.0, 1.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #27...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_027(candles, sequential=True)
    single_result = alpha_027(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  +1 count: {np.sum(valid == 1)}, -1 count: {np.sum(valid == -1)}")
    print("\nAlpha #27 all tests passed!")
