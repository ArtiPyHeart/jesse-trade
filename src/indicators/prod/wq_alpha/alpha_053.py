"""
Alpha #53: Price Position Change

Formula: -1 * delta((((close - low) - (high - close)) / (close - low)), 9)

Type: Price/Volume
Description: Measures the change in price position within the bar's range over 9 periods.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta
except ImportError:
    from _operators import ts_delta


def alpha_053(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #53: Price Position Change.

    Formula: -1 * delta((((close - low) - (high - close)) / (close - low)), 9)

    The inner expression ((close - low) - (high - close)) / (close - low) measures
    where the close is positioned within the bar. This alpha tracks the 9-period
    change in that position.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")

    # ((close - low) - (high - close)) / (close - low)
    # = (2*close - low - high) / (close - low)
    # When close == low (doji at low), use a small epsilon or set to 0
    denom = close - low
    # Use relative epsilon to avoid extreme values
    epsilon = np.maximum(np.abs(close) * 1e-10, 1e-10)
    safe_denom = np.where(np.abs(denom) < epsilon, epsilon, denom)

    inner = ((close - low) - (high - close)) / safe_denom
    # Clip extreme values that can occur when denom is very small
    inner = np.clip(inner, -10, 10)

    # -1 * delta(inner, 9)
    result = -1.0 * ts_delta(inner, 9)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #53...")

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

    # Test 1: Sequential consistency
    seq_result = alpha_053(candles, sequential=True)
    single_result = alpha_053(candles, sequential=False)

    assert len(seq_result) == len(candles), "Output length mismatch"
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10, "Sequential inconsistency"
    print("  Sequential consistency: OK")

    # Test 2: NaN check (first 9 values should be NaN due to delta(9))
    nan_count = np.sum(np.isnan(seq_result))
    assert nan_count == 9, f"Expected 9 NaN, got {nan_count}"
    print(f"  NaN count: {nan_count} (expected 9)")

    # Test 3: Value statistics
    valid_values = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
    print(f"  Mean: {valid_values.mean():.6f}, Std: {valid_values.std():.4f}")

    print("\nAlpha #53 all tests passed!")
