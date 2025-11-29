"""
Alpha #41: Geometric Mean vs VWAP

Formula: ((high * low)^0.5) - vwap

Type: Price/Volume (VWAP Approximation)
Description: Difference between geometric mean of high/low and VWAP.
VWAP approximated as (high + low + close) / 3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import get_vwap
except ImportError:
    from _operators import get_vwap


def alpha_041(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #41: Geometric Mean vs VWAP.

    Formula: ((high * low)^0.5) - vwap

    Compares the geometric mean of high and low prices with the VWAP.
    VWAP is approximated as (high + low + close) / 3.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")
    close = get_candle_source(candles, "close")

    # Geometric mean of high and low
    geo_mean = np.sqrt(high * low)

    # VWAP approximation
    vwap = get_vwap(high, low, close)

    # ((high * low)^0.5) - vwap
    result = geo_mean - vwap

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #41...")

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

    seq_result = alpha_041(candles, sequential=True)
    single_result = alpha_041(candles, sequential=False)

    assert len(seq_result) == len(candles), "Output length mismatch"
    assert abs(seq_result[-1] - single_result[0]) < 1e-10, "Sequential inconsistency"
    print("  Sequential consistency: OK")

    # Manual verification
    idx = 100
    h, l, c = candles[idx, 3], candles[idx, 4], candles[idx, 2]
    expected = np.sqrt(h * l) - (h + l + c) / 3
    assert abs(seq_result[idx] - expected) < 1e-10, "Manual calculation mismatch"
    print("  Manual verification: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")

    print("\nAlpha #41 all tests passed!")
