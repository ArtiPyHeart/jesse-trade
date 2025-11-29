"""
Alpha #3: Open-Volume Correlation

Formula: (-1 * correlation(rank(open), rank(volume), 10))
Note: rank() is ignored for single asset

Type: Correlation-based
Description: Negative correlation between open price and volume.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_corr
except ImportError:
    from _operators import ts_corr


def alpha_003(
    candles: np.ndarray,
    period: int = 10,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #3: Open-Volume Correlation.

    Formula: -1 * correlation(open, volume, 10)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        period: Correlation window (default 10)
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    volume = get_candle_source(candles, "volume")

    result = -1.0 * ts_corr(open_, volume, period)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #3...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_003(candles, sequential=True)
    single_result = alpha_003(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #3 all tests passed!")
