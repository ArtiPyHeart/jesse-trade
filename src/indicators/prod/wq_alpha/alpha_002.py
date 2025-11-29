"""
Alpha #2: Volume-Price Delta Correlation

Formula: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
Note: rank() is ignored for single asset

Type: Correlation-based
Description: Negative correlation between volume change and intraday return.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta, ts_corr
except ImportError:
    from _operators import ts_delta, ts_corr


def alpha_002(
    candles: np.ndarray,
    period: int = 6,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #2: Volume-Price Delta Correlation.

    Formula: -1 * correlation(delta(log(volume), 2), (close - open) / open, 6)

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        period: Correlation window (default 6)
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    close = get_candle_source(candles, "close")
    volume = get_candle_source(candles, "volume")

    # delta(log(volume), 2)
    log_volume = np.log(np.maximum(volume, 1e-10))  # Avoid log(0)
    delta_log_vol = ts_delta(log_volume, 2)

    # (close - open) / open (intraday return)
    intraday_ret = np.where(open_ != 0, (close - open_) / open_, 0.0)

    # -1 * correlation
    result = -1.0 * ts_corr(delta_log_vol, intraday_ret, period)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #2...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_002(candles, sequential=True)
    single_result = alpha_002(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #2 all tests passed!")
