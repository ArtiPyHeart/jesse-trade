"""
Alpha #52: Returns Momentum with Volume

Formula: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
          rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))

Type: Rank-based
Description: Low price change combined with returns momentum and volume rank.
Note: rank() ignored for single asset.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_min, ts_delay, ts_sum, ts_rank, get_returns
except ImportError:
    from _operators import ts_min, ts_delay, ts_sum, ts_rank, get_returns


def alpha_052(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #52: Returns Momentum with Volume.

    Simplified: ((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
                ((sum(returns, 240) - sum(returns, 20)) / 220) * ts_rank(volume, 5)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    low = get_candle_source(candles, "low")
    volume = get_candle_source(candles, "volume")

    returns = get_returns(close)

    # Part 1: ((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5))
    min_low_5 = ts_min(low, 5)
    delay_min_low = ts_delay(min_low_5, 5)
    part1 = (-1.0 * min_low_5) + delay_min_low

    # Part 2: (sum(returns, 240) - sum(returns, 20)) / 220
    sum_ret_240 = ts_sum(returns, 240)
    sum_ret_20 = ts_sum(returns, 20)
    part2 = (sum_ret_240 - sum_ret_20) / 220.0

    # Part 3: ts_rank(volume, 5)
    part3 = ts_rank(volume, 5)

    result = part1 * part2 * part3

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #52...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_052(candles, sequential=True)
    single_result = alpha_052(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #52 all tests passed!")
