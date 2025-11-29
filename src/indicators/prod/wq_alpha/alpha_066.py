"""
Alpha #66: VWAP Decay Linear

Formula: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052))
          + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)

Type: Rank-based (VWAP)
Description: VWAP delta decay plus price deviation decay rank.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta, ts_rank, decay_linear, get_vwap
except ImportError:
    from _operators import ts_delta, ts_rank, decay_linear, get_vwap


def alpha_066(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #66: VWAP Decay Linear.

    Simplified:
    decay_linear(delta(vwap, 4), 7) + ts_rank(decay_linear((low - vwap) / (open - mid), 11), 7)
    All multiplied by -1.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")

    vwap = get_vwap(high, low, close)
    mid = (high + low) / 2.0

    # Part 1: decay_linear(delta(vwap, 4), 7)
    delta_vwap = ts_delta(vwap, 4)
    part1 = decay_linear(delta_vwap, 7)

    # Part 2: ts_rank(decay_linear((low - vwap) / (open - mid), 11), 7)
    # Note: 0.96633 + (1-0.96633) = 1, so it's just low
    denom = open_ - mid
    ratio = np.where(denom != 0, (low - vwap) / denom, 0.0)
    decay_ratio = decay_linear(ratio, 11)
    part2 = ts_rank(decay_ratio, 7)

    result = -1.0 * (part1 + part2)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #66...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_066(candles, sequential=True)
    single_result = alpha_066(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #66 all tests passed!")
