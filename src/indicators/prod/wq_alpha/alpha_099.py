"""
Alpha #99: Correlation Comparison

Formula: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136))
          < rank(correlation(low, volume, 6.28259))) * -1)

Type: Correlation-based
Description: Mid-ADV correlation vs low-volume correlation.
Note: rank() ignored for single asset.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_sum, ts_corr, get_adv
except ImportError:
    from _operators import ts_sum, ts_corr, get_adv


def alpha_099(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #99: Correlation Comparison.

    Simplified: (correlation(sum(mid, 20), sum(adv60, 20), 9)
                < correlation(low, volume, 6)) * -1

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")
    volume = get_candle_source(candles, "volume")

    adv60 = get_adv(volume, 60)
    mid = (high + low) / 2.0

    # Part 1: correlation(sum(mid, 20), sum(adv60, 20), 9)
    sum_mid = ts_sum(mid, 20)
    sum_adv = ts_sum(adv60, 20)
    corr1 = ts_corr(sum_mid, sum_adv, 9)

    # Part 2: correlation(low, volume, 6)
    corr2 = ts_corr(low, volume, 6)

    # Comparison
    result = np.where(corr1 < corr2, -1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #99...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_099(candles, sequential=True)
    single_result = alpha_099(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Signal distribution: -1 count: {np.sum(valid == -1)}, 0 count: {np.sum(valid == 0)}")
    print("\nAlpha #99 all tests passed!")
