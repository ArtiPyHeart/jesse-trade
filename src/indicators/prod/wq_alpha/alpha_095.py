"""
Alpha #95: Open Range Comparison

Formula: (rank((open - ts_min(open, 12.4105)))
          < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))

Type: Correlation-based
Description: Open range rank vs mid-ADV correlation rank power.
Note: rank() ignored for single asset.
      Returns 1 if condition true, 0 if false.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_min, ts_sum, ts_rank, ts_corr, get_adv
except ImportError:
    from _operators import ts_min, ts_sum, ts_rank, ts_corr, get_adv


def alpha_095(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #95: Open Range Comparison.

    Simplified: (open - ts_min(open, 12))
                < ts_rank(correlation(sum(mid, 19), sum(adv40, 19), 13)^5, 12)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array (0 or 1)
    """
    open_ = get_candle_source(candles, "open")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")
    volume = get_candle_source(candles, "volume")

    adv40 = get_adv(volume, 40)
    mid = (high + low) / 2.0

    # Part 1: open - ts_min(open, 12)
    min_open = ts_min(open_, 12)
    open_range = open_ - min_open

    # Part 2: ts_rank(correlation(sum(mid, 19), sum(adv40, 19), 13)^5, 12)
    sum_mid = ts_sum(mid, 19)
    sum_adv = ts_sum(adv40, 19)
    corr = ts_corr(sum_mid, sum_adv, 13)
    corr_pow5 = corr ** 5
    rank_corr = ts_rank(corr_pow5, 12)

    # Comparison
    result = np.where(open_range < rank_corr, 1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #95...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_095(candles, sequential=True)
    single_result = alpha_095(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Signal distribution: 1 count: {np.sum(valid == 1)}, 0 count: {np.sum(valid == 0)}")
    print("\nAlpha #95 all tests passed!")
