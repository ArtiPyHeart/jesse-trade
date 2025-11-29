"""
Alpha #7: Volume Surge Momentum

Formula: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))

Type: Rank-based
Description: When volume exceeds 20-day average, signals based on 7-day price momentum.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import ts_delta, ts_rank, sign_array, get_adv
except ImportError:
    from _operators import ts_delta, ts_rank, sign_array, get_adv


def alpha_007(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #7: Volume Surge Momentum.

    Formula: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))

    When volume exceeds the 20-day average:
    - Calculate 7-day price change momentum
    - Rank the absolute change over 60 periods
    - Apply directional sign
    Otherwise, return -1.

    Args:
        candles: Jesse candles [timestamp, open, close, high, low, volume]
        sequential: True returns full array, False returns latest value

    Returns:
        Alpha values array
    """
    close = get_candle_source(candles, "close")
    volume = get_candle_source(candles, "volume")

    # adv20: 20-day average volume
    adv20 = get_adv(volume, 20)

    # delta(close, 7)
    delta_close_7 = ts_delta(close, 7)

    # ts_rank(abs(delta(close, 7)), 60)
    rank_abs_delta = ts_rank(np.abs(delta_close_7), 60)

    # sign(delta(close, 7))
    sign_delta = sign_array(delta_close_7)

    # Volume surge condition
    volume_surge = adv20 < volume

    # (-1 * ts_rank(...)) * sign(...)
    surge_signal = (-1.0 * rank_abs_delta) * sign_delta

    # Apply condition: volume surge -> surge_signal, else -1
    result = np.where(volume_surge, surge_signal, -1.0)

    # 确保warmup期统一为NaN（ts_rank需要60期，adv需要20期）
    # warmup = max(60-1, 7, 20-1) = 59
    warmup = 59
    result[:warmup] = np.nan

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #7...")

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

    seq_result = alpha_007(candles, sequential=True)
    single_result = alpha_007(candles, sequential=False)

    assert len(seq_result) == len(candles), "Output length mismatch"
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10, "Sequential inconsistency"
    print("  Sequential consistency: OK")

    # Check NaN count (should have NaN for warmup period: max(20-1, 7, 60-1) = 59)
    nan_count = np.sum(np.isnan(seq_result))
    print(f"  NaN count: {nan_count}")

    valid = seq_result[~np.isnan(seq_result)]
    print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")

    # Check that -1 values exist (non-surge cases)
    neg_one_count = np.sum(valid == -1.0)
    print(f"  Non-surge cases (-1): {neg_one_count}")

    print("\nAlpha #7 all tests passed!")
