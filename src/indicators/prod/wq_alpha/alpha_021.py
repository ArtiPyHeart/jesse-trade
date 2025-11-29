"""
Alpha #21: Volume-ADV Conditional

Formula: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1) :
          (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 :
          (((1 < (volume / adv20)) | ((volume / adv20) == 1)) ? 1 : (-1))))

Type: Price/Volume
Description: Complex conditional based on close moving averages, volatility, and volume.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_mean, ts_stddev, get_adv
except ImportError:
    from _operators import ts_mean, ts_stddev, get_adv


def alpha_021(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #21: Volume-ADV Conditional.

    Complex nested conditional based on price patterns and volume.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    volume = get_candle_source(candles, "volume")

    # Moving averages
    ma_8 = ts_mean(close, 8)
    ma_2 = ts_mean(close, 2)
    std_8 = ts_stddev(close, 8)
    adv20 = get_adv(volume, 20)

    # Volume ratio
    vol_ratio = np.where(adv20 != 0, volume / adv20, 1.0)

    # Conditions
    # Condition 1: (ma_8 + std_8) < ma_2
    cond1 = (ma_8 + std_8) < ma_2

    # Condition 2: ma_2 < (ma_8 - std_8)
    cond2 = ma_2 < (ma_8 - std_8)

    # Condition 3: vol_ratio >= 1
    cond3 = vol_ratio >= 1.0

    # Nested conditional
    result = np.where(cond1, -1.0,
                      np.where(cond2, 1.0,
                               np.where(cond3, 1.0, -1.0)))

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #21...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_021(candles, sequential=True)
    single_result = alpha_021(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
        print(f"  +1 count: {np.sum(valid == 1)}, -1 count: {np.sum(valid == -1)}")
    print("\nAlpha #21 all tests passed!")
