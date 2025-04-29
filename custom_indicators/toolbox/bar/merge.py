import numpy as np
from numba import njit
from tqdm.auto import tqdm


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
    sign_first: bool,
) -> tuple[np.ndarray, np.ndarray]:
    action_bar_index = np.abs(candles_return).argmin()
    action_bar = candles[action_bar_index]
    if action_bar_index - 1 >= 0:
        last_bar = candles[action_bar_index - 1]
        is_same_sign_last = (
            candles_return[action_bar_index - 1] * candles_return[action_bar_index] > 0
        )
        last_range = np.abs(
            candles_return[action_bar_index - 1] - candles_return[action_bar_index]
        )
    else:
        last_bar = None

    if action_bar_index + 1 <= candles.shape[0] - 1:
        next_bar = candles[action_bar_index + 1]
        is_same_sign_next = (
            candles_return[action_bar_index] * candles_return[action_bar_index + 1] > 0
        )
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
        if sign_first:
            if is_same_sign_last is is_same_sign_next:
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
                    candles_return = _delete_row_1d(
                        candles_return, action_bar_index + 1
                    )
            else:
                if is_same_sign_next:
                    new_bar = _get_new_bar(action_bar, next_bar)
                    candles[action_bar_index] = new_bar
                    candles = _delete_row_2d(candles, action_bar_index + 1)
                    candles_return[action_bar_index] = new_bar[2] / new_bar[1]
                    candles_return = _delete_row_1d(
                        candles_return, action_bar_index + 1
                    )
                else:
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


@njit
def _batch_nb_merge_bar(
    candles: np.ndarray,
    candles_return: np.ndarray,
    sign_first: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(batch_size):
        candles, candles_return = _nb_merge_bar(candles, candles_return, sign_first)
    return candles, candles_return


def np_merge_bars(
    candles: np.ndarray, bars_limit: int, lag: int = 1, sign_first: bool = False
) -> np.ndarray:
    assert candles.shape[0] > bars_limit, (
        f"bars_limit must be less than candles.shape[0], "
        f"but got {candles.shape[0] = } < {bars_limit = }"
    )
    candles_return = candles[lag:, 2] / candles[:-lag, 2]
    candles = candles[lag:]
    rounds = candles.shape[0] - bars_limit

    BATCH_SIZE = 10000

    for _ in tqdm(range(rounds // BATCH_SIZE)):
        candles, candles_return = _batch_nb_merge_bar(
            candles, candles_return, sign_first, BATCH_SIZE
        )
        assert candles.shape[0] == candles_return.shape[0]

    # 处理剩下的
    candles, candles_return = _batch_nb_merge_bar(
        candles, candles_return, sign_first, rounds % BATCH_SIZE
    )
    assert candles.shape[0] == candles_return.shape[0]

    return candles
