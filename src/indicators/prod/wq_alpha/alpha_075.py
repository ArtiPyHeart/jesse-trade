"""
Alpha #75: VWAP-Volume Correlation Comparison (VWAP)

Formula: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))

Type: Correlation-based (VWAP)
Description: VWAP-volume correlation vs low-ADV correlation.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
      Returns 1 if condition true, 0 if false.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_corr, get_adv, get_vwap
except ImportError:
    from _operators import ts_corr, get_adv, get_vwap


def alpha_075(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #75: VWAP-Volume Correlation Comparison.

    Simplified: (correlation(vwap, volume, 4) < correlation(low, adv50, 12))
    Returns 1 if true, 0 if false.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array (0 or 1)
    """
    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")
    volume = get_candle_source(candles, "volume")

    vwap = get_vwap(high, low, close)
    adv50 = get_adv(volume, 50)

    # Part 1: correlation(vwap, volume, 4)
    corr1 = ts_corr(vwap, volume, 4)

    # Part 2: correlation(low, adv50, 12)
    corr2 = ts_corr(low, adv50, 12)

    # Comparison
    result = np.where(corr1 < corr2, 1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #75...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_075(candles, sequential=True)
    single_result = alpha_075(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Signal distribution: 1 count: {np.sum(valid == 1)}, 0 count: {np.sum(valid == 0)}")
    print("\nAlpha #75 all tests passed!")
