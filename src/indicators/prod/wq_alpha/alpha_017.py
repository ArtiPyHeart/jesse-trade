"""
Alpha #17: Close Rank with Volume-ADV Rank

Formula: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
Note: rank() is ignored for single asset

Type: Rank-based
Description: Combines close ranking with second-order close delta and volume/adv ranking.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_rank, ts_delta, get_adv
except ImportError:
    from _operators import ts_rank, ts_delta, get_adv


def alpha_017(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #17: Close Rank with Volume-ADV Rank.

    Formula: ((-1 * ts_rank(close, 10)) * delta(delta(close, 1), 1)) * ts_rank(volume / adv20, 5)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    volume = get_candle_source(candles, "volume")

    # ts_rank(close, 10)
    close_rank = ts_rank(close, 10)

    # delta(delta(close, 1), 1) = second order difference
    delta1 = ts_delta(close, 1)
    delta2 = ts_delta(delta1, 1)

    # volume / adv20
    adv20 = get_adv(volume, 20)
    vol_ratio = np.where(adv20 != 0, volume / adv20, 1.0)

    # ts_rank(vol_ratio, 5)
    vol_rank = ts_rank(vol_ratio, 5)

    result = ((-1.0 * close_rank) * delta2) * vol_rank

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #17...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_017(candles, sequential=True)
    single_result = alpha_017(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #17 all tests passed!")
