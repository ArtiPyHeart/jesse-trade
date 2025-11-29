"""
Alpha #74: Correlation Comparison (VWAP)

Formula: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365))
          < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)

Type: Correlation-based (VWAP)
Description: Close-ADV correlation vs weighted VWAP-volume correlation.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_sum, ts_corr, get_adv, get_vwap
except ImportError:
    from _operators import ts_sum, ts_corr, get_adv, get_vwap


def alpha_074(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #74: Correlation Comparison.

    Simplified: (correlation(close, sum(adv30, 37), 15)
                < correlation(high*0.026 + vwap*0.974, volume, 11)) * -1

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
    adv30 = get_adv(volume, 30)

    w = 0.0261661

    # Part 1: correlation(close, sum(adv30, 37), 15)
    sum_adv = ts_sum(adv30, 37)
    corr1 = ts_corr(close, sum_adv, 15)

    # Part 2: correlation(weighted, volume, 11)
    weighted = high * w + vwap * (1 - w)
    corr2 = ts_corr(weighted, volume, 11)

    # Comparison
    result = np.where(corr1 < corr2, -1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #74...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_074(candles, sequential=True)
    single_result = alpha_074(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Signal distribution: -1 count: {np.sum(valid == -1)}, 0 count: {np.sum(valid == 0)}")
    print("\nAlpha #74 all tests passed!")
