"""
Alpha #71: Max Decay Correlations (VWAP)

Formula: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948),
             Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))

Type: Correlation-based (VWAP)
Description: Max of two decay linear rank signals.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank, ts_corr, decay_linear, get_adv, get_vwap
except ImportError:
    from _operators import ts_rank, ts_corr, decay_linear, get_adv, get_vwap


def alpha_071(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #71: Max Decay Correlations.

    Simplified: max of two ts_rank(decay_linear(...)) components.

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
    adv180 = get_adv(volume, 180)

    # Part 1: ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16)
    rank_close = ts_rank(close, 3)
    rank_adv = ts_rank(adv180, 12)
    corr = ts_corr(rank_close, rank_adv, 18)
    decay1 = decay_linear(corr, 4)
    part1 = ts_rank(decay1, 16)

    # Part 2: ts_rank(decay_linear(((low + open) - 2*vwap)^2, 16), 4)
    price_diff = (low + open_) - (2.0 * vwap)
    price_diff_sq = price_diff ** 2
    decay2 = decay_linear(price_diff_sq, 16)
    part2 = ts_rank(decay2, 4)

    result = np.maximum(part1, part2)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #71...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_071(candles, sequential=True)
    single_result = alpha_071(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #71 all tests passed!")
