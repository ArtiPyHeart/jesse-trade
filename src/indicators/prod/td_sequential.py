from typing import Tuple

import numpy as np
from jesse.helpers import get_candle_source, slice_candles
from numba import jit


@jit(nopython=True, fastmath=True)
def _td_setup(
    close: np.ndarray, stealth_actions: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(close)

    buy_set = np.zeros(N, dtype=np.float64)  # 1~9
    sell_set = np.zeros(N, dtype=np.float64)  # 1~9

    if stealth_actions:
        # 记录最近一次出现 buySet==8、sellSet==8 的索引，用于判断“barssince(...) <= 1”
        last_buy8_index = -1
        last_sell8_index = -1

    if N < 4:
        return buy_set, sell_set

    for i in range(4, N):
        # ========== (a) sellSet 逻辑 ==========
        if close[i] > close[i - 4]:
            # 延续 sellSet 计数
            if np.abs(sell_set[i - 1] - 9) < 1e-8:
                sell_set[i] = 1  # 若前面是9，再次出现继续条件，则从1重新开始
            else:
                sell_set[i] = sell_set[i - 1] + 1
        else:
            sell_set[i] = 0

        if stealth_actions:
            if np.abs(sell_set[i] - 8) < 1e-8:
                last_sell8_index = i

            if (
                last_sell8_index >= 0
                and (i - last_sell8_index <= 1)
                and np.abs(buy_set[i] - 1) < 1e-8
            ):
                sell_set[i] = 9

        # ========== (b) buySet 逻辑 ==========
        if close[i] < close[i - 4]:
            if np.abs(buy_set[i - 1] - 9) < 1e-8:
                buy_set[i] = 1
            else:
                buy_set[i] = buy_set[i - 1] + 1
        else:
            buy_set[i] = 0

        if stealth_actions:
            if np.abs(buy_set[i] - 8) < 1e-8:
                last_buy8_index = i

            if (
                last_buy8_index >= 0
                and (i - last_buy8_index <= 1)
                and np.abs(sell_set[i] - 1) < 1e-8
            ):
                buy_set[i] = 9

    return buy_set, sell_set


@jit(nopython=True, fastmath=True)
def _td_countdown(
    buy_set: np.ndarray,
    sell_set: np.ndarray,
    close: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    aggressive: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(close)
    buy_count = np.zeros(N, dtype=np.float64)  # 1~13 (标准)
    sell_count = np.zeros(N, dtype=np.float64)  # 1~13 (标准)
    if N < 2:
        return buy_count, sell_count

    for i in range(2, N):
        bc_prev = buy_count[i - 1]
        sc_prev = sell_count[i - 1]

        # Pine 脚本中 buySet=9 代表“开始买方 countdown”
        # 如果刚好出现 buySet=9, 则:
        #   - 如果满足 isBuyCondition，就把 buy_count 设为 1
        #   - 否则设为 0
        #   - 如果前面已经在进行 countdown，那么通常会在出现对方9时终止
        if aggressive:
            is_buy_cond = low[i] < low[i - 2]
        else:
            is_buy_cond = close[i] < low[i - 2]
        if np.abs(buy_set[i] - 9) < 1e-8:
            buy_count[i] = 1 if is_buy_cond else 0
        else:
            # 如果之前处于 countdown 过程, 并且还没到 13，就根据条件往下数
            # 脚本里若出现 sellSet == 9，就会把 buy_count = 14 强制结束
            if np.abs(sc_prev - 9) < 1e-8:  # 对方9出现, 强制中断
                buy_count[i] = 14
            elif 0 < bc_prev < 13:
                if is_buy_cond:
                    buy_count[i] = bc_prev + 1
                else:
                    buy_count[i] = bc_prev
            else:
                buy_count[i] = bc_prev

        # 同理, 对 sell_count 做类似处理
        if aggressive:
            is_sell_cond = high[i] > high[i - 2]
        else:
            is_sell_cond = close[i] > high[i - 2]
        if np.abs(sell_set[i] - 9) < 1e-8:
            sell_count[i] = 1 if is_sell_cond else 0
        else:
            if np.abs(bc_prev - 9) < 1e-8:  # 对方9出现, 强制中断
                sell_count[i] = 14
            elif 0 < sc_prev < 13:
                if is_sell_cond:
                    sell_count[i] = sc_prev + 1
                else:
                    sell_count[i] = sc_prev
            else:
                sell_count[i] = sc_prev

    return buy_count, sell_count


def td_sequential(
    candles: np.ndarray, sequential=False, aggressive=False, stealth_actions=False
):
    """
    Demark TD Sequential 指标
        candles: jesse 的 candles 数据
        sequential: 是否返回序列数据
        aggressive: 是否使用 aggressive 条件判定买卖count计数增加
        stealth_actions: 是否使用 stealth 模式判定td setup中对方计数打断了8到9的过程
    """
    candles = slice_candles(candles, sequential)

    close = get_candle_source(candles, "close")
    high = get_candle_source(candles, "high")
    low = get_candle_source(candles, "low")

    buy_set, sell_set = _td_setup(close, stealth_actions)
    buy_count, sell_count = _td_countdown(
        buy_set, sell_set, close, low, high, aggressive
    )

    if sequential:
        return buy_count, sell_count
    else:
        return buy_count[-1], sell_count[-1]
