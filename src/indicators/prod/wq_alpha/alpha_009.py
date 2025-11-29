"""
Alpha #9: Momentum Consistency

Formula: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
          ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))

Type: Price/Volume
Description: Returns price momentum, with direction depending on consistency of recent moves.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta, ts_min, ts_max
except ImportError:
    from _operators import ts_delta, ts_min, ts_max


def alpha_009(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #9: Momentum Consistency.

    Formula: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
              ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))

    Logic:
    - If all 5-period deltas are positive (min > 0): return delta (trending up)
    - Else if all 5-period deltas are negative (max < 0): return delta (trending down)
    - Else (mixed signals): return -delta (contrarian)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")

    # delta(close, 1) = close[t] - close[t-1]
    delta_close = ts_delta(close, 1)

    # ts_min(delta(close, 1), 5): minimum delta over last 5 periods
    min_delta = ts_min(delta_close, 5)

    # ts_max(delta(close, 1), 5): maximum delta over last 5 periods
    max_delta = ts_max(delta_close, 5)

    # Conditions
    # Condition 1: 0 < ts_min(...) -> all deltas positive
    cond1 = 0 < min_delta

    # Condition 2: ts_max(...) < 0 -> all deltas negative
    cond2 = max_delta < 0

    # Nested conditional:
    # if cond1: delta_close
    # elif cond2: delta_close
    # else: -delta_close
    result = np.where(
        cond1,
        delta_close,
        np.where(cond2, delta_close, -1.0 * delta_close)
    )

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #9...")

    _, candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0,
        caching=True,
        is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_009(candles, sequential=True)
    single_result = alpha_009(candles, sequential=False)

    assert len(seq_result) == len(candles), "Output length mismatch"
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10, "Sequential inconsistency"
    print("  Sequential consistency: OK")

    # Check NaN count (should have NaN for warmup: 1 (delta) + 4 (ts_min/max) = 5)
    nan_count = np.sum(np.isnan(seq_result))
    print(f"  NaN count: {nan_count}")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print(f"  Mean: {valid.mean():.6f}, Std: {valid.std():.4f}")

    print("\nAlpha #9 all tests passed!")
