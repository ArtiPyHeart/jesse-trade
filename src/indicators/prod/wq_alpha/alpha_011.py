"""
Alpha #11: VWAP-Close Range with Volume Delta

Formula: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
Note: rank() is ignored for single asset, VWAP approximated as (H+L+C)/3

Type: Rank-based (VWAP)
Description: Combines VWAP-close range extremes with volume momentum.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import get_vwap, ts_max, ts_min, ts_delta
except ImportError:
    from _operators import get_vwap, ts_max, ts_min, ts_delta


def alpha_011(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #11: VWAP-Close Range with Volume Delta.

    Formula: (ts_max(vwap - close, 3) + ts_min(vwap - close, 3)) * delta(volume, 3)

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
    vwap_close_diff = vwap - close

    max_diff = ts_max(vwap_close_diff, 3)
    min_diff = ts_min(vwap_close_diff, 3)
    delta_vol = ts_delta(volume, 3)

    result = (max_diff + min_diff) * delta_vol

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #11...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_011(candles, sequential=True)
    single_result = alpha_011(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.2f}, {valid.max():.2f}]")
    print("\nAlpha #11 all tests passed!")
