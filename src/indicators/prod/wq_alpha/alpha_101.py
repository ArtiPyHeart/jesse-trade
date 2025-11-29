"""
Alpha #101: Intraday Price Move Ratio

Formula: (close - open) / ((high - low) + 0.001)

Type: Price/Volume
Description: Measures intraday price movement relative to the trading range.
"""

import numpy as np
from jesse.helpers import get_candle_source


def alpha_101(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #101: Intraday price move ratio.

    Formula: (close - open) / ((high - low) + 0.001)

    This alpha measures the directional movement within a candle relative to
    its total range. A value close to 1 means price closed near the high,
    close to -1 means price closed near the low.

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

    # (close - open) / ((high - low) + 0.001)
    # Adding 0.001 to avoid division by zero when high == low
    result = (close - open_) / ((high - low) + 0.001)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #101...")

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
    seq_result = alpha_101(candles, sequential=True)
    single_result = alpha_101(candles, sequential=False)

    assert len(seq_result) == len(candles), "Output length mismatch"
    assert abs(seq_result[-1] - single_result[0]) < 1e-10, "Sequential inconsistency"
    print("  Sequential consistency: OK")

    # Test 2: Value range check (should be roughly between -1 and 1)
    # With the 0.001 denominator, extreme values are possible but rare
    valid_values = seq_result[~np.isnan(seq_result)]
    assert len(valid_values) == len(candles), "Should have no NaN values"
    print(f"  Value range: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
    print("  No NaN values: OK")

    # Test 3: Manual calculation verification
    idx = 100
    open_val = candles[idx, 1]
    close_val = candles[idx, 2]
    high_val = candles[idx, 3]
    low_val = candles[idx, 4]
    expected = (close_val - open_val) / ((high_val - low_val) + 0.001)
    assert abs(seq_result[idx] - expected) < 1e-10, "Manual calculation mismatch"
    print("  Manual verification: OK")

    # Test 4: Different slice sizes
    print("  Testing different slice sizes:")
    for size in [100, 500, 1000]:
        if size <= len(candles):
            slice_result = alpha_101(candles[-size:], sequential=False)
            full_result = alpha_101(candles, sequential=True)
            diff = abs(slice_result[0] - full_result[-1])
            assert diff < 1e-10, f"Slice size {size} mismatch"
            print(f"    Size {size}: OK (diff={diff:.2e})")

    print("\nAlpha #101 all tests passed!")
