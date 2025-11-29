"""
Alpha #77: Min Decay Correlations (VWAP)

Formula: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),
             rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))

Type: Correlation-based (VWAP)
Description: Min of price deviation decay and mid-ADV correlation decay.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_corr, decay_linear, get_adv, get_vwap
except ImportError:
    from _operators import ts_corr, decay_linear, get_adv, get_vwap


def alpha_077(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #77: Min Decay Correlations.

    Simplified: min(decay_linear((mid + high) - (vwap + high), 20),
                   decay_linear(correlation(mid, adv40, 3), 6))

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
    adv40 = get_adv(volume, 40)
    mid = (high + low) / 2.0

    # Part 1: decay_linear((mid + high) - (vwap + high), 20)
    # Simplifies to: decay_linear(mid - vwap, 20)
    diff = mid - vwap
    part1 = decay_linear(diff, 20)

    # Part 2: decay_linear(correlation(mid, adv40, 3), 6)
    corr = ts_corr(mid, adv40, 3)
    part2 = decay_linear(corr, 6)

    result = np.minimum(part1, part2)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #77...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_077(candles, sequential=True)
    single_result = alpha_077(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #77 all tests passed!")
