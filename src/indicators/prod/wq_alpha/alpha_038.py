"""
Alpha #38: Close Rank with Close-Open Ratio

Formula: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
Note: rank() is ignored for single asset

Type: Rank-based
Description: Combines close ranking with close/open ratio.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank
except ImportError:
    from _operators import ts_rank


def alpha_038(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #38: Close Rank with Close-Open Ratio.

    Formula: (-1 * Ts_Rank(close, 10)) * (close / open)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    close = get_candle_source(candles, "close")

    close_rank = ts_rank(close, 10)
    close_open_ratio = np.where(open_ != 0, close / open_, 1.0)

    result = (-1.0 * close_rank) * close_open_ratio

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #38...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_038(candles, sequential=True)
    single_result = alpha_038(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #38 all tests passed!")
