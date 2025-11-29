"""
Alpha #37: Correlation with Intraday Move

Formula: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))

Type: Correlation-based
Description: Correlation of lagged intraday move with close plus current intraday move.
Note: rank() ignored for single asset.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delay, ts_corr
except ImportError:
    from _operators import ts_delay, ts_corr


def alpha_037(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #37: Correlation with Intraday Move.

    Formula: correlation(delay((open - close), 1), close, 200) + (open - close)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    close = get_candle_source(candles, "close")

    # Intraday move
    intraday_move = open_ - close

    # Delayed intraday move
    delayed_move = ts_delay(intraday_move, 1)

    # Correlation
    corr = ts_corr(delayed_move, close, 200)

    result = corr + intraday_move

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #37...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_037(candles, sequential=True)
    single_result = alpha_037(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #37 all tests passed!")
