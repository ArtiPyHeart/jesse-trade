"""
Alpha #65: VWAP-ADV Correlation vs Open Range (VWAP)

Formula: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374))
          < rank((open - ts_min(open, 13.635)))) * -1)

Type: Correlation-based (VWAP)
Description: Weighted VWAP correlation vs open price range.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_sum, ts_min, ts_corr, get_adv, get_vwap
except ImportError:
    from _operators import ts_sum, ts_min, ts_corr, get_adv, get_vwap


def alpha_065(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #65: VWAP-ADV Correlation vs Open Range.

    Simplified comparison of weighted VWAP correlation and open range.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")
    volume = get_candle_source(candles, "volume")

    vwap = get_vwap(high, low, close)
    adv60 = get_adv(volume, 60)

    w = 0.00817205

    # Part 1: correlation(weighted_vwap, sum(adv60, 9), 6)
    weighted = open_ * w + vwap * (1 - w)
    sum_adv = ts_sum(adv60, 9)
    corr = ts_corr(weighted, sum_adv, 6)

    # Part 2: open - ts_min(open, 14)
    min_open = ts_min(open_, 14)
    open_range = open_ - min_open

    # Comparison
    result = np.where(corr < open_range, -1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #65...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_065(candles, sequential=True)
    single_result = alpha_065(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Signal distribution: -1 count: {np.sum(valid == -1)}, 0 count: {np.sum(valid == 0)}")
    print("\nAlpha #65 all tests passed!")
