"""
Alpha #46: Close Momentum Conditional

Formula: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?
          (-1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ?
          1 : ((-1) * (close - delay(close, 1)))))

Type: Price/Volume
Description: Conditional based on 10-day momentum acceleration.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delay
except ImportError:
    from _operators import ts_delay


def alpha_046(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #46: Close Momentum Conditional.

    Formula: Based on momentum rate of change comparison.

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

    # Momentum rate: (delay20 - delay10) / 10 - (delay10 - close) / 10
    momentum_1 = (delay_20 - delay_10) / 10.0
    momentum_2 = (delay_10 - close) / 10.0
    diff = momentum_1 - momentum_2

    # Conditions
    cond1 = diff > 0.25
    cond2 = diff < 0

    result = np.where(
        cond1,
        -1.0,
        np.where(
            cond2,
            1.0,
            -1.0 * (close - delay_1)
        )
    )

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #46...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_046(candles, sequential=True)
    single_result = alpha_046(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #46 all tests passed!")
