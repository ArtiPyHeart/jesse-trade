"""
Alpha #26: Volume-High Rank Correlation Max

Formula: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))

Type: Correlation-based
Description: Maximum of rolling correlation between volume and high rankings.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank, ts_corr, ts_max
except ImportError:
    from _operators import ts_rank, ts_corr, ts_max


def alpha_026(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #26: Volume-High Rank Correlation Max.

    Formula: -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    high = get_candle_source(candles, "high")
    volume = get_candle_source(candles, "volume")

    vol_rank = ts_rank(volume, 5)
    high_rank = ts_rank(high, 5)
    corr = ts_corr(vol_rank, high_rank, 5)
    max_corr = ts_max(corr, 3)

    result = -1.0 * max_corr

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #26...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_026(candles, sequential=True)
    single_result = alpha_026(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #26 all tests passed!")
