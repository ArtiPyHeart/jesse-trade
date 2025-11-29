"""
Alpha #78: Correlation Power (VWAP)

Formula: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))
          ^ rank(correlation(rank(vwap), rank(volume), 5.77492)))

Type: Correlation-based (VWAP)
Description: Product of two correlation-based signals via power.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_sum, ts_corr, get_adv, get_vwap
except ImportError:
    from _operators import ts_sum, ts_corr, get_adv, get_vwap


def alpha_078(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #78: Correlation Power.

    Simplified: correlation(sum(weighted, 20), sum(adv40, 20), 7)
                ^ correlation(vwap, volume, 6)

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
    adv40 = get_adv(volume, 40)

    w = 0.352233

    # Part 1: correlation(sum(weighted, 20), sum(adv40, 20), 7)
    weighted = low * w + vwap * (1 - w)
    sum_weighted = ts_sum(weighted, 20)
    sum_adv = ts_sum(adv40, 20)
    corr1 = ts_corr(sum_weighted, sum_adv, 7)

    # Part 2: correlation(vwap, volume, 6)
    corr2 = ts_corr(vwap, volume, 6)

    # Power with protection for negative base
    result = np.where(
        corr1 >= 0,
        np.power(corr1, corr2),
        -np.power(np.abs(corr1), corr2)
    )

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #78...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_078(candles, sequential=True)
    single_result = alpha_078(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #78 all tests passed!")
