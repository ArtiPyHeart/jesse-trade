"""
Alpha #96: Max Decay Correlation (VWAP)

Formula: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151),
             Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)

Type: Correlation-based (VWAP)
Description: Max of two complex decay-rank signals.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank, ts_corr, ts_argmax, decay_linear, get_adv, get_vwap
except ImportError:
    from _operators import ts_rank, ts_corr, ts_argmax, decay_linear, get_adv, get_vwap


def alpha_096(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #96: Max Decay Correlation.

    Simplified: max(ts_rank(decay_linear(correlation(vwap, volume, 4), 4), 8),
                   ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7), ts_rank(adv60, 4), 4), 13), 14), 13)) * -1

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

    vwap = get_vwap(high, low, close)
    adv60 = get_adv(volume, 60)

    # Part 1: ts_rank(decay_linear(correlation(vwap, volume, 4), 4), 8)
    corr1 = ts_corr(vwap, volume, 4)
    decay1 = decay_linear(corr1, 4)
    part1 = ts_rank(decay1, 8)

    # Part 2: ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7), ts_rank(adv60, 4), 4), 13), 14), 13)
    rank_close = ts_rank(close, 7)
    rank_adv = ts_rank(adv60, 4)
    corr2 = ts_corr(rank_close, rank_adv, 4)
    argmax_corr = ts_argmax(corr2, 13)
    decay2 = decay_linear(argmax_corr, 14)
    part2 = ts_rank(decay2, 13)

    result = -1.0 * np.maximum(part1, part2)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #96...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_096(candles, sequential=True)
    single_result = alpha_096(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #96 all tests passed!")
