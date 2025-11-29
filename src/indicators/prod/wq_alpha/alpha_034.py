"""
Alpha #34: Volatility Ratio and Close Delta

Formula: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
Note: rank() is ignored for single asset

Type: Rank-based
Description: Combines short-term vs medium-term volatility ratio with price momentum.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import get_returns, ts_stddev, ts_delta
except ImportError:
    from _operators import get_returns, ts_stddev, ts_delta


def alpha_034(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #34: Volatility Ratio and Close Delta.

    Formula: (1 - (stddev(returns, 2) / stddev(returns, 5))) + (1 - delta(close, 1))

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    returns = get_returns(close)

    std_2 = ts_stddev(returns, 2)
    std_5 = ts_stddev(returns, 5)
    delta_close = ts_delta(close, 1)

    # Avoid division by zero
    vol_ratio = np.where(std_5 != 0, std_2 / std_5, 1.0)

    result = (1.0 - vol_ratio) + (1.0 - delta_close)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #34...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_034(candles, sequential=True)
    single_result = alpha_034(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #34 all tests passed!")
