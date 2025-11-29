"""
Alpha #5: Open-VWAP Deviation

Formula: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
Note: rank() is ignored for single asset, VWAP approximated as (H+L+C)/3

Type: Rank-based (VWAP)
Description: Measures open price deviation from average VWAP, weighted by close-vwap deviation.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import get_vwap, ts_mean
except ImportError:
    from _operators import get_vwap, ts_mean


def alpha_005(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #5: Open-VWAP Deviation.

    Formula: (open - mean(vwap, 10)) * (-1 * abs(close - vwap))

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
    avg_vwap = ts_mean(vwap, 10)

    # (open - avg_vwap) * (-1 * abs(close - vwap))
    result = (open_ - avg_vwap) * (-1.0 * np.abs(close - vwap))

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #5...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_005(candles, sequential=True)
    single_result = alpha_005(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #5 all tests passed!")
