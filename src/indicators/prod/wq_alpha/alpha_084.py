"""
Alpha #84: VWAP Rank Power (VWAP)

Formula: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))

Type: Rank-based (VWAP)
Description: Signed power of VWAP deviation rank with close delta as exponent.
Note: VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta, ts_max, ts_rank, signed_power, get_vwap
except ImportError:
    from _operators import ts_delta, ts_max, ts_rank, signed_power, get_vwap


def alpha_084(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #84: VWAP Rank Power.

    Formula: SignedPower(ts_rank(vwap - ts_max(vwap, 15), 21), delta(close, 5))

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

    # VWAP deviation from max
    max_vwap = ts_max(vwap, 15)
    vwap_dev = vwap - max_vwap

    # Rank of deviation
    rank_dev = ts_rank(vwap_dev, 21)

    # Delta of close
    delta_close = ts_delta(close, 5)

    # Signed power (element-wise)
    result = np.empty_like(close, dtype=np.float64)
    for i in range(len(close)):
        if np.isnan(rank_dev[i]) or np.isnan(delta_close[i]):
            result[i] = np.nan
        elif rank_dev[i] >= 0:
            result[i] = np.power(rank_dev[i], delta_close[i]) if rank_dev[i] > 0 or delta_close[i] >= 0 else 0.0
        else:
            result[i] = -np.power(np.abs(rank_dev[i]), delta_close[i]) if np.abs(rank_dev[i]) > 0 else 0.0

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #84...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_084(candles, sequential=True)
    single_result = alpha_084(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.6f}, {valid.max():.6f}]")
    print("\nAlpha #84 all tests passed!")
