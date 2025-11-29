"""
Alpha #57: Close-VWAP Decay

Formula: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))

Type: Rank-based (VWAP)
Description: Close deviation from VWAP normalized by argmax decay.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_argmax, decay_linear, get_vwap
except ImportError:
    from _operators import ts_argmax, decay_linear, get_vwap


def alpha_057(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #57: Close-VWAP Decay.

    Simplified: -1 * (close - vwap) / decay_linear(ts_argmax(close, 30), 2)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")

    vwap = get_vwap(high, low, close)

    # ts_argmax(close, 30)
    argmax_30 = ts_argmax(close, 30)

    # decay_linear of argmax
    decay_argmax = decay_linear(argmax_30, 2)

    # Avoid division by zero
    result = np.where(
        decay_argmax != 0,
        -1.0 * (close - vwap) / decay_argmax,
        0.0
    )

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #57...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_057(candles, sequential=True)
    single_result = alpha_057(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #57 all tests passed!")
