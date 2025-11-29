"""
Alpha #35: Triple Rank Product

Formula: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))

Type: Rank-based
Description: Product of volume rank, price position rank, and returns rank.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank, get_returns
except ImportError:
    from _operators import ts_rank, get_returns


def alpha_035(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #35: Triple Rank Product.

    Formula: ts_rank(volume, 32) * (1 - ts_rank((close + high) - low, 16)) * (1 - ts_rank(returns, 32))

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

    returns = get_returns(close)

    # Part 1: ts_rank(volume, 32)
    rank_vol = ts_rank(volume, 32)

    # Part 2: (1 - ts_rank((close + high) - low, 16))
    price_range = (close + high) - low
    rank_price = ts_rank(price_range, 16)
    part2 = 1.0 - rank_price

    # Part 3: (1 - ts_rank(returns, 32))
    rank_ret = ts_rank(returns, 32)
    part3 = 1.0 - rank_ret

    result = rank_vol * part2 * part3

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #35...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_035(candles, sequential=True)
    single_result = alpha_035(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #35 all tests passed!")
