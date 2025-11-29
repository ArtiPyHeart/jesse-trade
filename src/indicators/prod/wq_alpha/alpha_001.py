"""
Alpha #1: Conditional Volatility

Formula: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
Note: rank() is ignored for single asset

Type: Rank-based
Description: Measures volatility conditioned on negative returns.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import get_returns, ts_stddev, ts_argmax, signed_power
except ImportError:
    from _operators import get_returns, ts_stddev, ts_argmax, signed_power


def alpha_001(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #1: Conditional Volatility.

    Formula: Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2), 5) - 0.5

    When returns are negative, use volatility; otherwise use close.
    Then find argmax of squared values over 5 periods.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    returns = get_returns(close)
    std_returns = ts_stddev(returns, 20)

    # Conditional: if returns < 0, use stddev(returns, 20), else use close
    cond_val = np.where(returns < 0, std_returns, close)

    # SignedPower(x, 2)
    powered = signed_power(cond_val, 2.0)

    # Ts_ArgMax over 5 periods
    argmax_val = ts_argmax(powered, 5)

    # Normalize to around 0
    result = argmax_val - 0.5

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #1...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_001(candles, sequential=True)
    single_result = alpha_001(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #1 all tests passed!")
