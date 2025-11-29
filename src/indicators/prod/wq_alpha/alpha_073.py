"""
Alpha #73: Max VWAP Delta Decay (VWAP)

Formula: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
             Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608)
             / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)

Type: Rank-based (VWAP)
Description: Max of VWAP delta decay and weighted price return decay.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta, ts_rank, decay_linear, get_vwap
except ImportError:
    from _operators import ts_delta, ts_rank, decay_linear, get_vwap


def alpha_073(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #73: Max VWAP Delta Decay.

    Simplified: max(decay_linear(delta(vwap, 5), 3),
                   ts_rank(decay_linear(-return_weighted, 3), 17)) * -1

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

    w = 0.147155

    # Part 1: decay_linear(delta(vwap, 5), 3)
    delta_vwap = ts_delta(vwap, 5)
    part1 = decay_linear(delta_vwap, 3)

    # Part 2: ts_rank(decay_linear(-return_weighted, 3), 17)
    weighted = open_ * w + low * (1 - w)
    delta_weighted = ts_delta(weighted, 2)
    return_weighted = np.where(weighted != 0, delta_weighted / weighted, 0.0)
    neg_return = -1.0 * return_weighted
    decay_ret = decay_linear(neg_return, 3)
    part2 = ts_rank(decay_ret, 17)

    result = -1.0 * np.maximum(part1, part2)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #73...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_073(candles, sequential=True)
    single_result = alpha_073(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #73 all tests passed!")
