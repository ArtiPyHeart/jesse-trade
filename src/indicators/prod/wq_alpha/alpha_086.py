"""
Alpha #86: ADV Correlation Comparison (VWAP)

Formula: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195)
          < rank(((open + close) - (vwap + open)))) * -1)

Type: Correlation-based (VWAP)
Description: Close-ADV correlation rank vs price deviation.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_sum, ts_rank, ts_corr, get_adv, get_vwap
except ImportError:
    from _operators import ts_sum, ts_rank, ts_corr, get_adv, get_vwap


def alpha_086(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #86: ADV Correlation Comparison.

    Simplified: (ts_rank(correlation(close, sum(adv20, 15), 6), 20)
                < ((open + close) - (vwap + open))) * -1
    Note: (open + close) - (vwap + open) = close - vwap

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
    adv20 = get_adv(volume, 20)

    # Part 1: ts_rank(correlation(close, sum(adv20, 15), 6), 20)
    sum_adv = ts_sum(adv20, 15)
    corr = ts_corr(close, sum_adv, 6)
    rank_corr = ts_rank(corr, 20)

    # Part 2: (open + close) - (vwap + open) = close - vwap
    price_dev = close - vwap

    # Comparison
    result = np.where(rank_corr < price_dev, -1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #86...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_086(candles, sequential=True)
    single_result = alpha_086(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Signal distribution: -1 count: {np.sum(valid == -1)}, 0 count: {np.sum(valid == 0)}")
    print("\nAlpha #86 all tests passed!")
