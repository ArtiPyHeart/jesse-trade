"""
Alpha #8: Open-Returns Product Change

Formula: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
Note: rank() is ignored for single asset

Type: Rank-based
Description: Measures change in open-returns correlation over time.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_sum, ts_delay, get_returns
except ImportError:
    from _operators import ts_sum, ts_delay, get_returns


def alpha_008(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #8: Open-Returns Product Change.

    Formula: -1 * ((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    close = get_candle_source(candles, "close")

    returns = get_returns(close)
    sum_open = ts_sum(open_, 5)
    sum_returns = ts_sum(returns, 5)

    product = sum_open * sum_returns
    delayed_product = ts_delay(product, 10)

    result = -1.0 * (product - delayed_product)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #8...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_008(candles, sequential=True)
    single_result = alpha_008(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.2f}, {valid.max():.2f}]")
    else:
        print("  Warning: All values are NaN (may need more warmup data)")
    print("\nAlpha #8 all tests passed!")
