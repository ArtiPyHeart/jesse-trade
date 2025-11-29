"""
Alpha #6: Open-Volume Correlation

Formula: -1 * correlation(open, volume, 10)

Type: Correlation-based
Description: Negative correlation between open price and volume over 10 periods.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_corr
except ImportError:
    from _operators import ts_corr


def alpha_006(
    candles: np.ndarray,
    period: int = 10,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #6: Open-Volume Correlation.

    Formula: -1 * correlation(open, volume, 10)

    This alpha captures the negative correlation between opening price
    and trading volume. A strong negative correlation suggests mean reversion.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        period: Correlation window (default 10)
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    volume = get_candle_source(candles, "volume")

    # -1 * correlation(open, volume, 10)
    result = -1.0 * ts_corr(open_, volume, period)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #6...")

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

    seq_result = alpha_006(candles, sequential=True)
    single_result = alpha_006(candles, sequential=False)

    assert len(seq_result) == len(candles), "Output length mismatch"
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10, "Sequential inconsistency"
    print("  Sequential consistency: OK")

    nan_count = np.sum(np.isnan(seq_result))
    print(f"  NaN count: {nan_count} (expected 9 for period=10)")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    # Correlation should be between -1 and 1, so result should be between -1 and 1
    assert valid.min() >= -1.0 and valid.max() <= 1.0, "Correlation out of range"
    print("  Value range valid: OK")

    print("\nAlpha #6 all tests passed!")
