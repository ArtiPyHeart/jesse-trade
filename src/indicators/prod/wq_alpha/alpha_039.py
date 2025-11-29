"""
Alpha #39: Delta Close with Decay Linear Volume

Formula: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))

Type: Rank-based
Description: Price change weighted by volume pattern and returns momentum.
Note: rank() ignored for single asset.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta, ts_sum, decay_linear, get_adv, get_returns
except ImportError:
    from _operators import ts_delta, ts_sum, decay_linear, get_adv, get_returns


def alpha_039(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #39: Delta Close with Decay Linear Volume.

    Simplified: (-1 * delta(close, 7) * (1 - decay_linear(volume / adv20, 9))) * (1 + sum(returns, 250))

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    volume = get_candle_source(candles, "volume")

    returns = get_returns(close)
    adv20 = get_adv(volume, 20)

    # Volume ratio with decay linear
    vol_ratio = np.where(adv20 != 0, volume / adv20, 1.0)
    decay_vol = decay_linear(vol_ratio, 9)

    # Delta close
    delta_7 = ts_delta(close, 7)

    # Sum of returns
    sum_ret = ts_sum(returns, 250)

    # Part 1: -1 * delta(close, 7) * (1 - decay_vol)
    part1 = -1.0 * delta_7 * (1.0 - decay_vol)

    # Part 2: (1 + sum(returns, 250))
    part2 = 1.0 + sum_ret

    result = part1 * part2

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #39...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_039(candles, sequential=True)
    single_result = alpha_039(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #39 all tests passed!")
