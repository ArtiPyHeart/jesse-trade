"""
Alpha #88: Min Decay Correlations

Formula: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)),
             Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))

Type: Correlation-based
Description: Min of OHLC rank decay and close-volume correlation decay rank.
Note: rank() ignored for single asset.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank, ts_corr, decay_linear, get_adv
except ImportError:
    from _operators import ts_rank, ts_corr, decay_linear, get_adv


def alpha_088(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #88: Min Decay Correlations.

    Simplified: min(decay_linear((open + low) - (high + close), 8),
                   ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3))

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

    adv60 = get_adv(volume, 60)

    # Part 1: decay_linear((open + low) - (high + close), 8)
    ohlc_diff = (open_ + low) - (high + close)
    part1 = decay_linear(ohlc_diff, 8)

    # Part 2: ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3)
    rank_close = ts_rank(close, 8)
    rank_adv = ts_rank(adv60, 21)
    corr = ts_corr(rank_close, rank_adv, 8)
    decay_corr = decay_linear(corr, 7)
    part2 = ts_rank(decay_corr, 3)

    result = np.minimum(part1, part2)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #88...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_088(candles, sequential=True)
    single_result = alpha_088(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #88 all tests passed!")
