"""
Alpha #30: Sign-based Volume Ratio

Formula: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2))))
          + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))

Type: Rank-based
Description: Price momentum sign combined with volume ratio.
Note: rank() ignored for single asset.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delay, ts_sum, sign_array
except ImportError:
    from _operators import ts_delay, ts_sum, sign_array


def alpha_030(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #30: Sign-based Volume Ratio.

    Formula: ((1.0 - (sign(close - delay(close, 1)) + sign(delay(close, 1) - delay(close, 2))
              + sign(delay(close, 2) - delay(close, 3)))) * sum(volume, 5)) / sum(volume, 20)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    volume = get_candle_source(candles, "volume")

    # Delayed closes
    delay_1 = ts_delay(close, 1)
    delay_2 = ts_delay(close, 2)
    delay_3 = ts_delay(close, 3)

    # Signs of price changes
    sign1 = sign_array(close - delay_1)
    sign2 = sign_array(delay_1 - delay_2)
    sign3 = sign_array(delay_2 - delay_3)

    # Sum of signs (can be -3, -1, 1, or 3)
    sum_signs = sign1 + sign2 + sign3

    # Volume sums
    sum_vol_5 = ts_sum(volume, 5)
    sum_vol_20 = ts_sum(volume, 20)

    # Avoid division by zero
    result = np.where(
        sum_vol_20 != 0,
        ((1.0 - sum_signs) * sum_vol_5) / sum_vol_20,
        0.0
    )

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #30...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_030(candles, sequential=True)
    single_result = alpha_030(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #30 all tests passed!")
