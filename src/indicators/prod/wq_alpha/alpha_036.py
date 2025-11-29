"""
Alpha #36: Complex Multi-Factor (VWAP)

Formula: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15)))
          + (0.7 * rank((open - close))))
          + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5))))
          + rank(abs(correlation(vwap, adv20, 6))))
          + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))

Type: Correlation-based (VWAP)
Description: Multi-factor combination with intraday moves and correlations.
Note: rank() ignored for single asset, VWAP approximated as (H+L+C)/3.
"""

import numpy as np
from jesse.helpers import get_candle_source

try:
    from ._operators import (ts_corr, ts_delay, ts_rank, ts_mean,
                             get_adv, get_vwap, get_returns)
except ImportError:
    from _operators import (ts_corr, ts_delay, ts_rank, ts_mean,
                            get_adv, get_vwap, get_returns)


def alpha_036(
    candles: np.ndarray,
    sequential: bool = False,
) -> np.ndarray:
    """
    Alpha #36: Complex Multi-Factor.

    Simplified:
    2.21 * correlation((close - open), delay(volume, 1), 15)
    + 0.7 * (open - close)
    + 0.73 * ts_rank(delay(-returns, 6), 5)
    + abs(correlation(vwap, adv20, 6))
    + 0.6 * ((ma200 - open) * (close - open))

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
    volume = get_candle_source(candles, "volume")

    returns = get_returns(close)
    vwap = get_vwap(high, low, close)
    adv20 = get_adv(volume, 20)

    # Part 1: 2.21 * correlation((close - open), delay(volume, 1), 15)
    intraday = close - open_
    delay_vol = ts_delay(volume, 1)
    corr1 = ts_corr(intraday, delay_vol, 15)
    part1 = 2.21 * corr1

    # Part 2: 0.7 * (open - close)
    part2 = 0.7 * (open_ - close)

    # Part 3: 0.73 * ts_rank(delay(-returns, 6), 5)
    neg_returns = -1.0 * returns
    delay_ret = ts_delay(neg_returns, 6)
    rank_ret = ts_rank(delay_ret, 5)
    part3 = 0.73 * rank_ret

    # Part 4: abs(correlation(vwap, adv20, 6))
    corr2 = ts_corr(vwap, adv20, 6)
    part4 = np.abs(corr2)

    # Part 5: 0.6 * ((ma200 - open) * (close - open))
    ma_200 = ts_mean(close, 200)
    part5 = 0.6 * ((ma_200 - open_) * (close - open_))

    result = part1 + part2 + part3 + part4 + part5

    return result if sequential else result[-1:]


if __name__ == "__main__":
    from jesse import helpers, research

    print("Testing Alpha #36...")
    _, candles = research.get_candles(
        "Binance Perpetual Futures", "BTC-USDT", "1m",
        helpers.date_to_timestamp("2024-01-01"),
        helpers.date_to_timestamp("2024-01-07"),
        warmup_candles_num=0, caching=True, is_for_jesse=False,
    )
    print(f"  Loaded {len(candles)} candles")

    seq_result = alpha_036(candles, sequential=True)
    single_result = alpha_036(candles, sequential=False)
    assert len(seq_result) == len(candles)
    if not (np.isnan(seq_result[-1]) and np.isnan(single_result[0])):
        assert abs(seq_result[-1] - single_result[0]) < 1e-10
    print("  Sequential consistency: OK")

    valid = seq_result[~np.isnan(seq_result)]
    if len(valid) > 0:
        print(f"  Value range: [{valid.min():.4f}, {valid.max():.4f}]")
    print("\nAlpha #36 all tests passed!")
