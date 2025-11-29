"""
Alpha #16: High-Volume Covariance

Formula: (-1 * rank(covariance(rank(high), rank(volume), 5)))
Note: rank() is ignored for single asset

Type: Rank-based
Description: Negative covariance between high price and volume.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_cov
except ImportError:
    from _operators import ts_cov


def alpha_016(
    candles: np.ndarray,
    period: int = 5,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #16: High-Volume Covariance.

    Formula: -1 * covariance(high, volume, 5)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        period: Covariance window (default 5)
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    high = get_candle_source(candles, "high")
    volume = get_candle_source(candles, "volume")

    result = -1.0 * ts_cov(high, volume, period)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #16...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_016(candles, sequential=True)
    single_result = alpha_016(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.2f}, {valid.max():.2f}]")
    print("\nAlpha #16 all tests passed!")
