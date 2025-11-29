"""
Alpha #42: VWAP-Close Ratio

Formula: (vwap - close) / (vwap + close)
Note: rank() is ignored for single asset

Type: Rank-based (VWAP Approximation)
Description: Normalized difference between VWAP and close price.
VWAP approximated as (high + low + close) / 3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import get_vwap
except ImportError:
    from _operators import get_vwap


def alpha_042(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #42: VWAP-Close Ratio.

    Formula: (vwap - close) / (vwap + close)
    Note: rank() simplified for single asset

    Measures the normalized difference between VWAP and close.
    Positive when VWAP > close (price below average).
    Negative when VWAP < close (price above average).

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")
    close = get_candle_source(candles, "close")

    # VWAP approximation
    vwap = get_vwap(high, low, close)

    # (vwap - close) / (vwap + close)
    denom = vwap + close
    result = np.where(denom != 0, (vwap - close) / denom, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #42...")

    _, candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0,
        caching=True,
        is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_042(candles, sequential=True)
    single_result = alpha_042(candles, sequential=False)

    assert len(seq_result) == len(candles), "Output length mismatch"
    assert abs(seq_result[-1] - single_result[0]) < 1e-10, "Sequential inconsistency"
    print("  Sequential consistency: OK")

    # Manual verification
    idx = 100
    h, l, c = candles[idx, 3], candles[idx, 4], candles[idx, 2]
    vwap = (h + l + c) / 3
    expected = (vwap - c) / (vwap + c)
    assert abs(seq_result[idx] - expected) < 1e-10, "Manual calculation mismatch"
    print("  Manual verification: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    # Result should be bounded since it's a ratio
    assert valid.min() >= -1.0 and valid.max() <= 1.0, "Value out of expected range"
    print("  Value range valid: OK")

    print("\nAlpha #42 all tests passed!")
