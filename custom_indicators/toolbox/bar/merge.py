import numpy as np
from numba import njit


@njit
def _get_new_bar(bar1: np.ndarray, bar2: np.ndarray) -> np.ndarray:
    # 时序上要保证bar1在bar2之前
    t = bar2[0]  # 取最后的时间戳
    o = bar1[1]
    c = bar2[2]
    h = max(bar1[3], bar2[3])
    l = min(bar1[4], bar2[4])
    v = bar1[5] + bar2[5]
    return np.array([t, o, c, h, l, v])


@njit
def _delete_row_2d(arr: np.ndarray, row_to_delete: int) -> np.ndarray:
    upper = arr[:row_to_delete, :]
    lower = arr[row_to_delete + 1 :, :]
    return np.vstack((upper, lower))


@njit
def _delete_row_1d(arr: np.ndarray, row_to_delete: int) -> np.ndarray:
    upper = arr[:row_to_delete]
    lower = arr[row_to_delete + 1 :]
    return np.concatenate((upper, lower))


@njit
def _nb_merge_bar(
    candles: np.ndarray,
    candles_return: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    action_bar_index = np.abs(candles_return).argmin()
    action_bar = candles[action_bar_index]
    if action_bar_index - 1 >= 0:
        last_bar = candles[action_bar_index - 1]
        last_range = np.abs(
            candles_return[action_bar_index - 1] - candles_return[action_bar_index]
        )
    else:
        last_bar = None

    if action_bar_index + 1 <= candles.shape[0] - 1:
        next_bar = candles[action_bar_index + 1]
        next_range = np.abs(
            candles_return[action_bar_index] - candles_return[action_bar_index + 1]
        )
    else:
        next_bar = None

    if last_bar is None:
        new_bar = _get_new_bar(action_bar, next_bar)
        candles[action_bar_index] = new_bar
        candles = _delete_row_2d(candles, action_bar_index + 1)
        candles_return[action_bar_index] = new_bar[2] / new_bar[1]
        candles_return = _delete_row_1d(candles_return, action_bar_index + 1)
    elif next_bar is None:
        new_bar = _get_new_bar(last_bar, action_bar)
        candles[action_bar_index - 1] = new_bar
        candles = _delete_row_2d(candles, action_bar_index)
        candles_return[action_bar_index - 1] = new_bar[2] / new_bar[1]
        candles_return = _delete_row_1d(candles_return, action_bar_index)
    else:
        if last_range < next_range:
            new_bar = _get_new_bar(last_bar, action_bar)
            candles[action_bar_index - 1] = new_bar
            candles = _delete_row_2d(candles, action_bar_index)
            candles_return[action_bar_index - 1] = new_bar[2] / new_bar[1]
            candles_return = _delete_row_1d(candles_return, action_bar_index)
        else:
            new_bar = _get_new_bar(action_bar, next_bar)
            candles[action_bar_index] = new_bar
            candles = _delete_row_2d(candles, action_bar_index + 1)
            candles_return[action_bar_index] = new_bar[2] / new_bar[1]
            candles_return = _delete_row_1d(candles_return, action_bar_index + 1)
    return candles, candles_return


# ================================================================
# Optimised, in-place implementation
# ================================================================


@njit(cache=True, fastmath=True)
def _nb_merge_bars_inplace(
    candles: np.ndarray,
    bars_limit: int,
    lag: int = 1,
) -> np.ndarray:
    """高性能实现——完全在 numba 内部循环，避免 Python for-loop 与频繁重新分配。

    与 `np_merge_bars` (旧实现) 在逻辑上一致。
    ``candles`` 按 numba 的要求应当是 float64 的二维数组，形状为 *(N, 6)*。
    返回值为合并后的 *view*，因此无需再进行额外复制。
    """

    # 预处理：先裁剪出 lag 之后的视图，与旧实现保持一致
    candles_view = candles[lag:].copy()
    n = candles_view.shape[0]
    candles_return = np.zeros(n, dtype=np.float64)

    # 主循环：不断合并直到满足 bars_limit
    while n > bars_limit:
        # 0. 准备 candles_return (长度 == candles_view)
        for i in range(n):
            candles_return[i] = np.log(candles[i + lag, 2] / candles[i, 2])
        candles_return = candles_return[:n]

        # 1. 找到 |return| 最小的 bar 下标(对应 action_bar)
        abs_min = np.abs(candles_return[0])
        action_idx = 0
        for i in range(1, n):
            cur = np.abs(candles_return[i])
            if cur < abs_min:
                abs_min = cur
                action_idx = i

        # 2. 依据 last_range / next_range 规则决定与前还是后合并
        if action_idx == 0:
            merge_with_prev = False  # 只能与下一根合并
        elif action_idx == n - 1:
            merge_with_prev = True  # 只能与上一根合并
        else:
            last_range = np.abs(
                candles_return[action_idx - 1] - candles_return[action_idx]
            )
            next_range = np.abs(
                candles_return[action_idx] - candles_return[action_idx + 1]
            )
            merge_with_prev = last_range < next_range

        # 3. 执行合并 (获得新 bar 并整体左移数组) -------------------------
        if merge_with_prev:
            # i1 与 i2 合并 => 结果写回 i1 位置，删除 i2
            i1 = action_idx - 1
            i2 = action_idx
        else:
            i1 = action_idx
            i2 = action_idx + 1

        # 3.1 计算合并后的新 bar，各字段含义与旧实现一致
        t = candles_view[i2, 0]  # 取后一根的时间戳
        o = candles_view[i1, 1]
        c = candles_view[i2, 2]
        h = (
            candles_view[i1, 3]
            if candles_view[i1, 3] > candles_view[i2, 3]
            else candles_view[i2, 3]
        )
        l = (
            candles_view[i1, 4]
            if candles_view[i1, 4] < candles_view[i2, 4]
            else candles_view[i2, 4]
        )
        v = candles_view[i1, 5] + candles_view[i2, 5]

        # 3.2 将新 bar 写回 i1 位置
        candles_view[i1, 0] = t
        candles_view[i1, 1] = o
        candles_view[i1, 2] = c
        candles_view[i1, 3] = h
        candles_view[i1, 4] = l
        candles_view[i1, 5] = v

        # 3.3 左移 candles_view[i2+1 : n] 到 i2 ... (覆盖删除 i2)
        for j in range(i2 + 1, n):
            candles_view[j - 1, 0] = candles_view[j, 0]
            candles_view[j - 1, 1] = candles_view[j, 1]
            candles_view[j - 1, 2] = candles_view[j, 2]
            candles_view[j - 1, 3] = candles_view[j, 3]
            candles_view[j - 1, 4] = candles_view[j, 4]
            candles_view[j - 1, 5] = candles_view[j, 5]

        # n、candles_return 的有效长度各缩短 1
        n -= 1

        # 打印进度
        if n % 50000 == 0:
            print("n = " + str(n) + ", target = " + str(bars_limit))

    # 返回前 n 条记录（视图切片，无额外复制）
    return candles_view[:n]


def np_merge_bars(
    candles: np.ndarray,
    bars_limit: int,
    lag: int = 1,
) -> np.ndarray:
    """在保持算法逻辑不变的前提下，对原实现进行显著提速。

    参数
    ------
    candles : np.ndarray
        形状 *(N, 6)* 的 K 线数组 (timestamp, open, close, high, low, volume)。
    bars_limit : int
        目标 bar 数量 (合并停止条件)。
    lag : int, default 1
        与原实现保持一致。
    """

    assert candles.shape[0] > bars_limit, (
        f"bars_limit must be less than candles.shape[0], "
        f"but got {candles.shape[0] = } < {bars_limit = }"
    )

    return _nb_merge_bars_inplace(candles, bars_limit, lag)
