"""
Alpha #25: Returns-ADV-VWAP-High

Formula: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
Note: rank() is ignored for single asset, VWAP approximated as (H+L+C)/3

Type: Rank-based (VWAP)
Description: Product of returns, volume, vwap and high-close spread.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import get_returns, get_adv, get_vwap
except ImportError:
    from _operators import get_returns, get_adv, get_vwap


def alpha_025(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #25: Returns-ADV-VWAP-High.

    Formula: ((-1 * returns) * adv20) * vwap * (high - close)

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

    returns = get_returns(close)
    adv20 = get_adv(volume, 20)
    vwap = get_vwap(high, low, close)

    result = ((-1.0 * returns) * adv20) * vwap * (high - close)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #25...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_025(candles, sequential=True)
    single_result = alpha_025(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.2e}, {valid.max():.2e}]")
    print("\nAlpha #25 all tests passed!")
