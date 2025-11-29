"""
Alpha #68: High-ADV Correlation

Formula: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333)
          < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)

Type: Correlation-based
Description: High-volume correlation rank vs weighted price delta.
Note: rank() ignored for single asset.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta, ts_rank, ts_corr, get_adv
except ImportError:
    from _operators import ts_delta, ts_rank, ts_corr, get_adv


def alpha_068(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #68: High-ADV Correlation.

    Simplified: (ts_rank(correlation(high, adv15, 9), 14)
                < delta(close * 0.518 + low * 0.482, 1)) * -1

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

    adv15 = get_adv(volume, 15)

    w = 0.518371

    # Part 1: ts_rank(correlation(high, adv15, 9), 14)
    corr = ts_corr(high, adv15, 9)
    rank_corr = ts_rank(corr, 14)

    # Part 2: delta(weighted_price, 1)
    weighted = close * w + low * (1 - w)
    delta_price = ts_delta(weighted, 1)

    # Comparison
    result = np.where(rank_corr < delta_price, -1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #68...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_068(candles, sequential=True)
    single_result = alpha_068(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Signal distribution: -1 count: {np.sum(valid == -1)}, 0 count: {np.sum(valid == 0)}")
    print("\nAlpha #68 all tests passed!")
