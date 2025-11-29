"""
Alpha #92: Min Decay Correlations

Formula: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683),
             Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))

Type: Correlation-based
Description: Min of price comparison decay rank and low-volume correlation decay rank.
Note: rank() ignored for single asset. Boolean converted to float (1.0 or 0.0).
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank, ts_corr, decay_linear, get_adv
except ImportError:
    from _operators import ts_rank, ts_corr, decay_linear, get_adv


def alpha_092(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #92: Min Decay Correlations.

    Simplified: min(ts_rank(decay_linear((mid + close) < (low + open), 15), 19),
                   ts_rank(decay_linear(correlation(low, adv30, 8), 7), 7))

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

    adv30 = get_adv(volume, 30)
    mid = (high + low) / 2.0

    # Part 1: ts_rank(decay_linear((mid + close) < (low + open), 15), 19)
    cond = ((mid + close) < (low + open_)).astype(np.float64)
    decay_cond = decay_linear(cond, 15)
    part1 = ts_rank(decay_cond, 19)

    # Part 2: ts_rank(decay_linear(correlation(low, adv30, 8), 7), 7)
    corr = ts_corr(low, adv30, 8)
    decay_corr = decay_linear(corr, 7)
    part2 = ts_rank(decay_corr, 7)

    result = np.minimum(part1, part2)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #92...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_092(candles, sequential=True)
    single_result = alpha_092(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #92 all tests passed!")
