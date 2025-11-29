"""
Alpha #85: Correlation Power

Formula: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))
          ^ rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))

Type: Correlation-based
Description: Weighted price-ADV correlation powered by mid-volume rank correlation.
Note: rank() ignored for single asset.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank, ts_corr, get_adv
except ImportError:
    from _operators import ts_rank, ts_corr, get_adv


def alpha_085(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #85: Correlation Power.

    Simplified: correlation(weighted, adv30, 10) ^ correlation(ts_rank(mid, 4), ts_rank(vol, 10), 7)

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

    adv30 = get_adv(volume, 30)
    mid = (high + low) / 2.0

    w = 0.876703

    # Part 1: correlation(weighted, adv30, 10)
    weighted = high * w + close * (1 - w)
    corr1 = ts_corr(weighted, adv30, 10)

    # Part 2: correlation(ts_rank(mid, 4), ts_rank(volume, 10), 7)
    rank_mid = ts_rank(mid, 4)
    rank_vol = ts_rank(volume, 10)
    corr2 = ts_corr(rank_mid, rank_vol, 7)

    # Power with protection
    result = np.where(
        corr1 >= 0,
        np.power(np.maximum(corr1, 1e-10), corr2),
        -np.power(np.maximum(np.abs(corr1), 1e-10), corr2)
    )

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #85...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_085(candles, sequential=True)
    single_result = alpha_085(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #85 all tests passed!")
