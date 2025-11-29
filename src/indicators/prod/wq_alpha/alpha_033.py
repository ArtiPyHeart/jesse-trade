"""
Alpha #33: Open-Close Ratio

Formula: -1 * ((1 - (open / close))^1)
Note: rank() is ignored for single asset

Type: Rank-based (simplified to Price)
Description: Measures the ratio of open to close, negated.
"""

import numpy as np
from jesse.helpers import get_candle_source


def alpha_033(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #33: Open-Close Ratio.

    Formula: -1 * ((1 - (open / close))^1)
    Simplified: -1 * (1 - open/close) = open/close - 1

    This alpha is negative when close > open (bullish candle),
    and positive when close < open (bearish candle).

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    close = get_candle_source(candles, "close")

    # -1 * (1 - open/close) = open/close - 1
    # Avoid division by zero
    result = np.where(close != 0, open_ / close - 1.0, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #33...")

    # Load test data from jesse
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

    seq_result = alpha_033(candles, sequential=True)
    single_result = alpha_033(candles, sequential=False)

    assert len(seq_result) == len(candles), "Output length mismatch"
    assert abs(seq_result[-1] - single_result[0]) < 1e-10, "Sequential inconsistency"
    print("  Sequential consistency: OK")

    # Manual verification
    idx = 100
    expected = candles[idx, 1] / candles[idx, 2] - 1.0
    assert abs(seq_result[idx] - expected) < 1e-10, "Manual calculation mismatch"
    print("  Manual verification: OK")

    valid_values = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid_values.min():.6f}, {valid_values.max():.6f}]")

    print("\nAlpha #33 all tests passed!")
