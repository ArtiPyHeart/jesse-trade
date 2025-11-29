"""
Alpha #54: Price Structure

Formula: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

Type: Price/Volume
Description: Complex price structure ratio based on OHLC relationships.
"""

import numpy as np
from jesse.helpers import get_candle_source


def alpha_054(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #54: Price Structure.

    Formula: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    open_ = get_candle_source(candles, "open")
    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")

    # Numerator: -1 * (low - close) * open^5
    numer = -1.0 * (low - close) * np.power(open_, 5)

    # Denominator: (low - high) * close^5
    # Note: (low - high) is always <= 0
    denom = (low - high) * np.power(close, 5)

    # Avoid division by zero
    epsilon = 1e-10
    safe_denom = np.where(np.abs(denom) < epsilon, epsilon, denom)

    result = numer / safe_denom
    # Clip extreme values
    result = np.clip(result, -100, 100)

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #54...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_054(candles, sequential=True)
    single_result = alpha_054(candles, sequential=False)
    assert len(seq_result) == len(candles)
    assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #54 all tests passed!")
