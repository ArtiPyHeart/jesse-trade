"""
Alpha #43: Volume-ADV Rank with Close Delta

Formula: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))

Type: Rank-based
Description: Combines volume/ADV ranking with negative close change ranking.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank, ts_delta, get_adv
except ImportError:
    from _operators import ts_rank, ts_delta, get_adv


def alpha_043(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #43: Volume-ADV Rank with Close Delta.

    Formula: ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    volume = get_candle_source(candles, "volume")

    adv20 = get_adv(volume, 20)
    vol_ratio = np.where(adv20 != 0, volume / adv20, 1.0)
    vol_rank = ts_rank(vol_ratio, 20)

    delta_close = ts_delta(close, 7)
    delta_rank = ts_rank(-1.0 * delta_close, 8)

    result = vol_rank * delta_rank

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #43...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_043(candles, sequential=True)
    single_result = alpha_043(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #43 all tests passed!")
