"""
Alpha #18: Intraday Volatility and Correlation

Formula: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
Note: rank() is ignored for single asset

Type: Correlation-based
Description: Combines intraday volatility with close-open correlation.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_stddev, ts_corr
except ImportError:
    from _operators import ts_stddev, ts_corr


def alpha_018(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #18: Intraday Volatility and Correlation.

    Formula: -1 * (stddev(abs(close - open), 5) + (close - open) + correlation(close, open, 10))

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    close = get_candle_source(candles, "close")

    # Intraday move
    intraday = close - open_

    # stddev(abs(close - open), 5)
    std_intraday = ts_stddev(np.abs(intraday), 5)

    # correlation(close, open, 10)
    corr = ts_corr(close, open_, 10)

    # -1 * (std + intraday + corr)
    result = -1.0 * (std_intraday + intraday + corr)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #18...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_018(candles, sequential=True)
    single_result = alpha_018(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #18 all tests passed!")
