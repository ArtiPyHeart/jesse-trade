"""
Alpha #15: High-Volume Correlation Sum

Formula: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
Note: rank() is ignored for single asset

Type: Correlation-based
Description: Rolling sum of high-volume correlation over 3 periods.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_corr, ts_sum
except ImportError:
    from _operators import ts_corr, ts_sum


def alpha_015(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #15: High-Volume Correlation Sum.

    Formula: -1 * sum(correlation(high, volume, 3), 3)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    high = get_candle_source(candles, "high")
    volume = get_candle_source(candles, "volume")

    corr = ts_corr(high, volume, 3)
    result = -1.0 * ts_sum(corr, 3)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #15...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_015(candles, sequential=True)
    single_result = alpha_015(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #15 all tests passed!")
