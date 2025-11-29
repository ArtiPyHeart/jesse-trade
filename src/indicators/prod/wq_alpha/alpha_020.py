"""
Alpha #20: Gap Analysis

Formula: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
Note: rank() is ignored for single asset

Type: Rank-based
Description: Measures gaps between open and previous high/close/low.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delay
except ImportError:
    from _operators import ts_delay


def alpha_020(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #20: Gap Analysis.

    Formula: (-1 * (open - delay(high, 1))) * (open - delay(close, 1)) * (open - delay(low, 1))

    Measures the product of gaps from open to previous high, close, and low.

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

    delay_high = ts_delay(high, 1)
    delay_close = ts_delay(close, 1)
    delay_low = ts_delay(low, 1)

    result = ((-1.0 * (open_ - delay_high)) *
              (open_ - delay_close) *
              (open_ - delay_low))

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #20...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_020(candles, sequential=True)
    single_result = alpha_020(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.2f}, {valid.max():.2f}]")
    print("\nAlpha #20 all tests passed!")
