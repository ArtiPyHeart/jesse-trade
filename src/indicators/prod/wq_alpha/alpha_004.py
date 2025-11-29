"""
Alpha #4: Low Price Rank

Formula: (-1 * Ts_Rank(rank(low), 9))
Note: rank() is ignored for single asset -> (-1 * Ts_Rank(low, 9))

Type: Rank-based
Description: Negative time series rank of low prices over 9 periods.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank
except ImportError:
    from _operators import ts_rank


def alpha_004(
    candles: np.ndarray,
    period: int = 9,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #4: Low Price Rank.

    Formula: -1 * Ts_Rank(low, 9)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        period: Rank window (default 9)
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    low = get_candle_source(candles, "low")
    result = -1.0 * ts_rank(low, period)
    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #4...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_004(candles, sequential=True)
    single_result = alpha_004(candles, sequential=False)
    assert len(seq_result) == len(candles)
    assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #4 all tests passed!")
