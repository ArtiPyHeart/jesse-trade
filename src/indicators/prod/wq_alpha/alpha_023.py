"""
Alpha #23: High Breakout

Formula: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)

Type: Price/Volume
Description: When high exceeds 20-period average high, signal based on 2-period high change.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_mean, ts_delta
except ImportError:
    from _operators import ts_mean, ts_delta


def alpha_023(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #23: High Breakout.

    Formula: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)

    When current high exceeds 20-period average high (breakout),
    return negative of 2-period high change. Otherwise return 0.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    high = get_candle_source(candles, "high")

    avg_high_20 = ts_mean(high, 20)
    delta_high_2 = ts_delta(high, 2)

    # Breakout condition: current high > 20-period average
    breakout = avg_high_20 < high

    result = np.where(breakout, -1.0 * delta_high_2, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #23...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_023(candles, sequential=True)
    single_result = alpha_023(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    zero_count = np.sum(valid == 0)
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print(f"  Zero count (no breakout): {zero_count}/{len(valid)}")
    print("\nAlpha #23 all tests passed!")
