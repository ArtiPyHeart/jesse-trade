"""
Alpha #98: Decay Correlation Difference (VWAP)

Formula: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088))
          - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))

Type: Correlation-based (VWAP)
Description: VWAP-ADV correlation decay minus argmin decay rank.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_sum, ts_rank, ts_corr, ts_argmin, decay_linear, get_adv, get_vwap
except ImportError:
    from _operators import ts_sum, ts_rank, ts_corr, ts_argmin, decay_linear, get_adv, get_vwap


def alpha_098(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #98: Decay Correlation Difference.

    Simplified: decay_linear(correlation(vwap, sum(adv5, 26), 5), 7)
                - decay_linear(ts_rank(ts_argmin(correlation(open, adv15, 21), 9), 7), 8)

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
    volume = get_candle_source(candles, "volume")

    vwap = get_vwap(high, low, close)
    adv5 = get_adv(volume, 5)
    adv15 = get_adv(volume, 15)

    # Part 1: decay_linear(correlation(vwap, sum(adv5, 26), 5), 7)
    sum_adv5 = ts_sum(adv5, 26)
    corr1 = ts_corr(vwap, sum_adv5, 5)
    part1 = decay_linear(corr1, 7)

    # Part 2: decay_linear(ts_rank(ts_argmin(correlation(open, adv15, 21), 9), 7), 8)
    corr2 = ts_corr(open_, adv15, 21)
    argmin_corr = ts_argmin(corr2, 9)
    rank_argmin = ts_rank(argmin_corr, 7)
    part2 = decay_linear(rank_argmin, 8)

    result = part1 - part2

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #98...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_098(candles, sequential=True)
    single_result = alpha_098(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #98 all tests passed!")
