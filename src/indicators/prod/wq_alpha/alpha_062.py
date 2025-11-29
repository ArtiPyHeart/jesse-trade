"""
Alpha #62: VWAP-ADV Correlation (VWAP)

Formula: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009))
          < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)

Type: Correlation-based (VWAP)
Description: VWAP correlation compared with price level comparison.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_sum, ts_corr, get_adv, get_vwap
except ImportError:
    from _operators import ts_sum, ts_corr, get_adv, get_vwap


def alpha_062(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #62: VWAP-ADV Correlation.

    Simplified: (correlation(vwap, sum(adv20, 22), 10) < ((2*open) < (mid + high))) * -1
    Returns -1 if correlation < price comparison, 0 otherwise.

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

    # Part 1: correlation(vwap, sum(adv20, 22), 10)
    sum_adv = ts_sum(adv20, 22)
    corr = ts_corr(vwap, sum_adv, 10)

    # Part 2: (2*open) < ((high + low) / 2 + high)
    mid = (high + low) / 2.0
    price_cond = (2.0 * open_) < (mid + high)

    # Comparison: corr < price_cond (converted to float)
    price_cond_float = price_cond.astype(np.float64)
    result = np.where(corr < price_cond_float, -1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #62...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_062(candles, sequential=True)
    single_result = alpha_062(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Signal distribution: -1 count: {np.sum(valid == -1)}, 0 count: {np.sum(valid == 0)}")
    print("\nAlpha #62 all tests passed!")
