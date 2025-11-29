"""
Alpha #47: Complex Volume-Price (VWAP)

Formula: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5)))
          - rank((vwap - delay(vwap, 5))))

Type: Rank-based (VWAP)
Description: Volume-price intensity minus VWAP momentum.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delay, ts_mean, get_adv, get_vwap
except ImportError:
    from _operators import ts_delay, ts_mean, get_adv, get_vwap


def alpha_047(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #47: Complex Volume-Price.

    Simplified: (((1 / close) * volume / adv20) * (high * (high - close) / ma_high_5))
                - (vwap - delay(vwap, 5))

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")
    volume = get_candle_source(candles, "volume")

    vwap = get_vwap(high, low, close)
    adv20 = get_adv(volume, 20)

    # Part 1: ((1 / close) * volume / adv20) * (high * (high - close) / ma_high_5)
    inv_close = np.where(close != 0, 1.0 / close, 0.0)
    vol_ratio = np.where(adv20 != 0, volume / adv20, 1.0)
    ma_high_5 = ts_mean(high, 5)
    high_factor = np.where(ma_high_5 != 0, high * (high - close) / ma_high_5, 0.0)
    part1 = inv_close * vol_ratio * high_factor

    # Part 2: vwap - delay(vwap, 5)
    delay_vwap = ts_delay(vwap, 5)
    part2 = vwap - delay_vwap

    result = part1 - part2

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #47...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_047(candles, sequential=True)
    single_result = alpha_047(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #47 all tests passed!")
