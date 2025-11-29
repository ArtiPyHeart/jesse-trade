"""
Alpha #94: VWAP Rank Power (VWAP)

Formula: ((rank((vwap - ts_min(vwap, 11.5783)))
          ^ Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)

Type: Correlation-based (VWAP)
Description: VWAP range rank powered by VWAP-volume correlation rank.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_min, ts_rank, ts_corr, get_adv, get_vwap
except ImportError:
    from _operators import ts_min, ts_rank, ts_corr, get_adv, get_vwap


def alpha_094(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #94: VWAP Rank Power.

    Simplified: ((vwap - ts_min(vwap, 12))
                ^ ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3)) * -1

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

    # Part 1: vwap - ts_min(vwap, 12)
    min_vwap = ts_min(vwap, 12)
    vwap_range = vwap - min_vwap

    # Part 2: ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3)
    rank_vwap = ts_rank(vwap, 20)
    rank_adv = ts_rank(adv60, 4)
    corr = ts_corr(rank_vwap, rank_adv, 18)
    rank_corr = ts_rank(corr, 3)

    # Power with protection
    result = np.where(
        vwap_range >= 0,
        -1.0 * np.power(np.maximum(vwap_range, 1e-10), rank_corr),
        -1.0 * (-np.power(np.maximum(np.abs(vwap_range), 1e-10), rank_corr))
    )

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #94...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_094(candles, sequential=True)
    single_result = alpha_094(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #94 all tests passed!")
