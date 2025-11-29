"""
Alpha #55: Price Position Volume Correlation

Formula: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))

Type: Correlation-based
Description: Negative correlation of price position in range with volume.
Note: rank() ignored for single asset.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_min, ts_max, ts_corr
except ImportError:
    from _operators import ts_min, ts_max, ts_corr


def alpha_055(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #55: Price Position Volume Correlation.

    Simplified: -1 * correlation((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)), volume, 6)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")
    volume = get_candle_source(candles, "volume")

    min_low_12 = ts_min(low, 12)
    max_high_12 = ts_max(high, 12)

    # Price position in range (like Williams %R)
    range_12 = max_high_12 - min_low_12
    # Avoid division by zero
    position = np.where(
        range_12 != 0,
        (close - min_low_12) / range_12,
        0.5
    )

    corr = ts_corr(position, volume, 6)
    result = -1.0 * corr

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #55...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_055(candles, sequential=True)
    single_result = alpha_055(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #55 all tests passed!")
