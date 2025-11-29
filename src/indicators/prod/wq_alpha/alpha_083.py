"""
Alpha #83: Complex Ratio (VWAP)

Formula: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume)))
          / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))

Type: Rank-based (VWAP)
Description: Delayed range ratio times volume rank divided by VWAP deviation.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delay, ts_mean, get_vwap
except ImportError:
    from _operators import ts_delay, ts_mean, get_vwap


def alpha_083(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #83: Complex Ratio.

    Simplified: (delay(range_ratio, 2) * volume) / (range_ratio / (vwap - close))
    where range_ratio = (high - low) / ma5

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
    ma_5 = ts_mean(close, 5)

    # Range ratio
    range_ratio = np.where(ma_5 != 0, (high - low) / ma_5, 0.0)

    # Delayed range ratio
    delay_ratio = ts_delay(range_ratio, 2)

    # VWAP deviation
    vwap_dev = vwap - close

    # Denominator: range_ratio / (vwap - close)
    denom = np.where(vwap_dev != 0, range_ratio / vwap_dev, 0.0)

    # Final result
    result = np.where(denom != 0, (delay_ratio * volume) / denom, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #83...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_083(candles, sequential=True)
    single_result = alpha_083(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #83 all tests passed!")
