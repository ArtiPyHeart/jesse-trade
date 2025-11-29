"""
Alpha #50: Volume-VWAP Correlation Max

Formula: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))

Type: Correlation-based (VWAP)
Description: Max of volume-vwap correlation over window.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_corr, ts_max, get_vwap
except ImportError:
    from _operators import ts_corr, ts_max, get_vwap


def alpha_050(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #50: Volume-VWAP Correlation Max.

    Simplified: -1 * ts_max(correlation(volume, vwap, 5), 5)

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
    corr = ts_corr(volume, vwap, 5)
    max_corr = ts_max(corr, 5)

    result = -1.0 * max_corr

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #50...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_050(candles, sequential=True)
    single_result = alpha_050(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #50 all tests passed!")
