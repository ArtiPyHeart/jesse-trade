"""
Alpha #24: Long-term Close Momentum

Formula: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) |
          ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ?
          (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))

Type: Price/Volume
Description: Conditional based on 100-day moving average change.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_mean, ts_delta, ts_delay, ts_min
except ImportError:
    from _operators import ts_mean, ts_delta, ts_delay, ts_min


def alpha_024(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #24: Long-term Close Momentum.

    Formula: if delta(ma100, 100) / delay(close, 100) <= 0.05:
               -1 * (close - ts_min(close, 100))
             else:
               -1 * delta(close, 3)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")

    ma_100 = ts_mean(close, 100)
    delta_ma = ts_delta(ma_100, 100)
    delay_close = ts_delay(close, 100)
    min_close = ts_min(close, 100)
    delta_close_3 = ts_delta(close, 3)

    # delta(ma100, 100) / delay(close, 100)
    ratio = np.where(delay_close != 0, delta_ma / delay_close, 0.0)

    # Condition: ratio <= 0.05
    cond = ratio <= 0.05

    result = np.where(cond,
                      -1.0 * (close - min_close),
                      -1.0 * delta_close_3)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #24...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_024(candles, sequential=True)
    single_result = alpha_024(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #24 all tests passed!")
