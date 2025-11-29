"""
Alpha #64: Complex Correlation (VWAP)

Formula: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),
                           sum(adv120, 12.7054), 16.6208))
          < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)

Type: Correlation-based (VWAP)
Description: Weighted price correlation vs weighted price delta.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_sum, ts_delta, ts_corr, get_adv, get_vwap
except ImportError:
    from _operators import ts_sum, ts_delta, ts_corr, get_adv, get_vwap


def alpha_064(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #64: Complex Correlation.

    Simplified comparison of weighted price correlation and delta.

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
    adv120 = get_adv(volume, 120)

    w = 0.178404

    # Part 1: correlation(sum(weighted_price, 13), sum(adv120, 13), 17)
    weighted_price1 = open_ * w + low * (1 - w)
    sum_price = ts_sum(weighted_price1, 13)
    sum_adv = ts_sum(adv120, 13)
    corr = ts_corr(sum_price, sum_adv, 17)

    # Part 2: delta(weighted_vwap, 4)
    mid = (high + low) / 2.0
    weighted_price2 = mid * w + vwap * (1 - w)
    delta_price = ts_delta(weighted_price2, 4)

    # Comparison
    result = np.where(corr < delta_price, -1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #64...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_064(candles, sequential=True)
    single_result = alpha_064(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Signal distribution: -1 count: {np.sum(valid == -1)}, 0 count: {np.sum(valid == 0)}")
    print("\nAlpha #64 all tests passed!")
