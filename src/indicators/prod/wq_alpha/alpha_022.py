"""
Alpha #22: High-Volume Correlation Delta

Formula: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
Note: rank() is ignored for single asset

Type: Correlation-based
Description: Change in high-volume correlation weighted by close volatility.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_corr, ts_delta, ts_stddev
except ImportError:
    from _operators import ts_corr, ts_delta, ts_stddev


def alpha_022(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #22: High-Volume Correlation Delta.

    Formula: -1 * delta(correlation(high, volume, 5), 5) * stddev(close, 20)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    volume = get_candle_source(candles, "volume")

    corr = ts_corr(high, volume, 5)
    delta_corr = ts_delta(corr, 5)
    std_close = ts_stddev(close, 20)

    result = -1.0 * delta_corr * std_close

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #22...")
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

    seq_result = alpha_022(candles, sequential=True)
    single_result = alpha_022(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #22 all tests passed!")
