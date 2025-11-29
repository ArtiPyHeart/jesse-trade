"""
Alpha #12: Volume-Price Divergence

Formula: sign(delta(volume, 1)) * (-1 * delta(close, 1))

Type: Price/Volume
Description: Captures volume-price divergence. When volume increases but price
decreases (or vice versa), this alpha generates a signal.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta, sign_array
except ImportError:
    from _operators import ts_delta, sign_array


def alpha_012(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #12: Volume-Price Divergence.

    Formula: sign(delta(volume, 1)) * (-1 * delta(close, 1))

    This alpha captures volume-price divergence:
    - When volume increases (sign=1) and price decreases (delta<0),
      result is positive (potential reversal signal)
    - When volume decreases (sign=-1) and price increases (delta>0),
      result is positive (potential reversal signal)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    volume = get_candle_source(candles, "volume")

    # delta(volume, 1) = volume[t] - volume[t-1]
    delta_volume = ts_delta(volume, 1)

    # delta(close, 1) = close[t] - close[t-1]
    delta_close = ts_delta(close, 1)

    # sign(delta(volume, 1)) * (-1 * delta(close, 1))
    result = sign_array(delta_volume) * (-1.0 * delta_close)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #12...")

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
    seq_result = alpha_012(candles, sequential=True)
    single_result = alpha_012(candles, sequential=False)

    assert len(seq_result) == len(candles), "Output length mismatch"
    # Handle NaN comparison
    if np.isnan(seq_result[-1]) and np.isnan(single_result[0]):
        pass  # Both NaN is OK
    else:
        assert abs(seq_result[-1] - single_result[0]) < 1e-10, "Sequential inconsistency"
    print("  Sequential consistency: OK")

    # Test 2: NaN check (first value should be NaN due to delta)
    assert np.isnan(seq_result[0]), "First value should be NaN"
    nan_count = np.sum(np.isnan(seq_result))
    assert nan_count == 1, f"Expected 1 NaN, got {nan_count}"
    print(f"  NaN count: {nan_count} (expected 1)")

    # Test 3: Manual calculation verification
    idx = 100
    close = candles[:, 2]
    volume = candles[:, 5]
    delta_vol = volume[idx] - volume[idx - 1]
    delta_close = close[idx] - close[idx - 1]
    expected = np.sign(delta_vol) * (-1.0 * delta_close)
    assert abs(seq_result[idx] - expected) < 1e-10, "Manual calculation mismatch"
    print("  Manual verification: OK")

    # Test 4: Value statistics
    valid_values = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
    print(f"  Mean: {valid_values.mean():.4f}, Std: {valid_values.std():.4f}")

    print("\nAlpha #12 all tests passed!")
