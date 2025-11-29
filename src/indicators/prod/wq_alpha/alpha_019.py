"""
Alpha #19: Long-term Returns Momentum

Formula: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
Note: rank() is ignored for single asset

Type: Rank-based
Description: Long-term momentum signal based on 7-day and 250-day returns.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delay, ts_delta, ts_sum, get_returns, sign_array
except ImportError:
    from _operators import ts_delay, ts_delta, ts_sum, get_returns, sign_array


def alpha_019(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #19: Long-term Returns Momentum.

    Formula: (-1 * sign((close - delay(close, 7)) + delta(close, 7))) * (1 + (1 + sum(returns, 250)))

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    returns = get_returns(close)

    delay_close = ts_delay(close, 7)
    delta_close = ts_delta(close, 7)
    sum_returns = ts_sum(returns, 250)

    # sign((close - delay(close, 7)) + delta(close, 7))
    momentum = (close - delay_close) + delta_close
    sign_momentum = sign_array(momentum)

    # (1 + (1 + sum(returns, 250)))
    long_term = 1.0 + (1.0 + sum_returns)

    result = (-1.0 * sign_momentum) * long_term

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #19...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_019(candles, sequential=True)
    single_result = alpha_019(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #19 all tests passed!")
