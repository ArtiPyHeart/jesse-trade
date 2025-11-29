"""
Alpha #10: Momentum Consistency (ranked version of Alpha 9)

Formula: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
         ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
Note: rank() is ignored for single asset

Type: Rank-based
Description: Similar to Alpha 9 but with 4-period window instead of 5.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta, ts_min, ts_max
except ImportError:
    from _operators import ts_delta, ts_min, ts_max


def alpha_010(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #10: Momentum Consistency (4-period).

    Similar to Alpha 9 but uses 4-period window.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    delta_close = ts_delta(close, 1)
    min_delta = ts_min(delta_close, 4)
    max_delta = ts_max(delta_close, 4)

    cond1 = 0 < min_delta
    cond2 = max_delta < 0

    result = np.where(cond1, delta_close,
                      np.where(cond2, delta_close, -1.0 * delta_close))

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #10...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_010(candles, sequential=True)
    single_result = alpha_010(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #10 all tests passed!")
