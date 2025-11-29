"""
Alpha #72: Ratio of Decay Correlations (VWAP)

Formula: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519))
          / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))

Type: Correlation-based (VWAP)
Description: Ratio of two decay-weighted correlations.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank, ts_corr, decay_linear, get_adv, get_vwap
except ImportError:
    from _operators import ts_rank, ts_corr, decay_linear, get_adv, get_vwap


def alpha_072(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #72: Ratio of Decay Correlations.

    Simplified ratio of two decay_linear(correlation(...)) components.

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
    adv40 = get_adv(volume, 40)

    mid = (high + low) / 2.0

    # Part 1: decay_linear(correlation(mid, adv40, 9), 10)
    corr1 = ts_corr(mid, adv40, 9)
    part1 = decay_linear(corr1, 10)

    # Part 2: decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3)
    rank_vwap = ts_rank(vwap, 4)
    rank_vol = ts_rank(volume, 19)
    corr2 = ts_corr(rank_vwap, rank_vol, 7)
    part2 = decay_linear(corr2, 3)

    # Ratio with protection
    result = np.where(part2 != 0, part1 / part2, 0.0)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #72...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_072(candles, sequential=True)
    single_result = alpha_072(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #72 all tests passed!")
