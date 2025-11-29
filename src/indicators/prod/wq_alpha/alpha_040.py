"""
Alpha #40: High Volatility with High-Volume Correlation

Formula: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))

Type: Correlation-based
Description: Combines high price volatility with high-volume correlation.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_stddev, ts_corr
except ImportError:
    from _operators import ts_stddev, ts_corr


def alpha_040(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #40: High Volatility with High-Volume Correlation.

    Formula: (-1 * stddev(high, 10)) * correlation(high, volume, 10)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    high = get_candle_source(candles, "high")
    volume = get_candle_source(candles, "volume")

    std_high = ts_stddev(high, 10)
    corr = ts_corr(high, volume, 10)

    result = (-1.0 * std_high) * corr

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #40...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_040(candles, sequential=True)
    single_result = alpha_040(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #40 all tests passed!")
