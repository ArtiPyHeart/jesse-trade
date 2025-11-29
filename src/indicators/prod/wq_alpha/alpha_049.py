"""
Alpha #49: Momentum Threshold

Formula: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ?
          1 : ((-1) * (close - delay(close, 1))))

Type: Price/Volume
Description: Momentum threshold signal with price change fallback.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delay
except ImportError:
    from _operators import ts_delay


def alpha_049(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #49: Momentum Threshold.

    Formula: if momentum_diff < -0.1: 1 else: -1 * (close - delay(close, 1))

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")

    delay_1 = ts_delay(close, 1)
    delay_10 = ts_delay(close, 10)
    delay_20 = ts_delay(close, 20)

    # Momentum rate difference
    momentum_1 = (delay_20 - delay_10) / 10.0
    momentum_2 = (delay_10 - close) / 10.0
    diff = momentum_1 - momentum_2

    # Condition: diff < -0.1
    cond = diff < -0.1

    result = np.where(
        cond,
        1.0,
        -1.0 * (close - delay_1)
    )

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #49...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_049(candles, sequential=True)
    single_result = alpha_049(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #49 all tests passed!")
