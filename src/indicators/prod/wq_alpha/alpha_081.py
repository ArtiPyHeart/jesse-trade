"""
Alpha #81: VWAP-ADV Correlation Power (VWAP)

Formula: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655)))
          < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)

Type: Correlation-based (VWAP)
Description: Log of product of correlation powers vs VWAP-volume correlation.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_sum, ts_corr, ts_product, get_adv, get_vwap
except ImportError:
    from _operators import ts_sum, ts_corr, ts_product, get_adv, get_vwap


def alpha_081(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #81: VWAP-ADV Correlation Power.

    Simplified: (log(product(correlation(vwap, sum(adv10, 50), 8)^4, 15))
                < correlation(vwap, volume, 5)) * -1

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
    adv10 = get_adv(volume, 10)

    # Part 1: log(product(correlation(vwap, sum(adv10, 50), 8)^4, 15))
    sum_adv = ts_sum(adv10, 50)
    corr1 = ts_corr(vwap, sum_adv, 8)
    corr1_pow4 = corr1 ** 4  # Always positive
    prod = ts_product(corr1_pow4, 15)
    log_prod = np.where(prod > 0, np.log(prod), np.nan)

    # Part 2: correlation(vwap, volume, 5)
    corr2 = ts_corr(vwap, volume, 5)

    # Comparison
    result = np.where(log_prod < corr2, -1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #81...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_081(candles, sequential=True)
    single_result = alpha_081(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Signal distribution: -1 count: {np.sum(valid == -1)}, 0 count: {np.sum(valid == 0)}")
    print("\nAlpha #81 all tests passed!")
