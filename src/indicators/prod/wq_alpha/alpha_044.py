"""
Alpha #44: High-Volume Correlation

Formula: -1 * correlation(high, rank(volume), 5)
Note: rank() is ignored for single asset

Type: Correlation-based
Description: Negative correlation between high price and volume over 5 periods.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_corr
except ImportError:
    from _operators import ts_corr


def alpha_044(
    candles: np.ndarray,
    period: int = 5,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #44: High-Volume Correlation.

    Formula: -1 * correlation(high, volume, 5)
    Note: rank(volume) simplified to volume for single asset

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        period: Correlation window (default 5)
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    high = get_candle_source(candles, "high")
    volume = get_candle_source(candles, "volume")

    # -1 * correlation(high, volume, 5)
    result = -1.0 * ts_corr(high, volume, period)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #44...")

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

    seq_result = alpha_044(candles, sequential=True)
    single_result = alpha_044(candles, sequential=False)

    assert len(seq_result) == len(candles), "Output length mismatch"
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10, "Sequential inconsistency"
    print("  Sequential consistency: OK")

    nan_count = np.sum(np.isnan(seq_result))
    print(f"  NaN count: {nan_count} (expected 4 for period=5)")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")

    print("\nAlpha #44 all tests passed!")
