import numpy as np
from numba import njit


def z_score_filter_np(
    raw_array: np.ndarray,
    mean_window: int = 20,
    std_window: int = 20,
    z_score: float = 3,
) -> np.ndarray:
    """
    适用于 Jesse 的 z_score_filter 函数示例。
    输入：一维 numpy array。
    输出：同长度的 numpy array，对应索引位置的值为 1 或 0 或 np.nan。

    :param raw_array: (np.ndarray) 任意一维 numpy array。
    :param mean_window: (int) 计算 rolling mean 的窗口大小。
    :param std_window: (int) 计算 rolling std 的窗口大小。
    :param z_score: (float) 标准差倍数阈值。
    :return: (np.ndarray) 长度与 raw_array 相同的结果数组。
             内容：1（满足条件）、0（不满足条件）、np.nan（无法计算）。
    """

    length = len(raw_array)
    result = np.full(length, np.nan, dtype=float)

    # 对于中前期不足以计算滚动窗口的位置，直接返回 np.nan
    # 当 i >= mean_window-1 和 i >= std_window-1 时才可以正常计算
    min_idx = max(mean_window, std_window) - 1

    for i in range(min_idx, length):
        window_slice_mean = raw_array[i - mean_window + 1 : i + 1]
        window_slice_std = raw_array[i - std_window + 1 : i + 1]

        rolling_mean = np.mean(window_slice_mean)
        rolling_std = np.std(window_slice_std, ddof=1)  # ddof=1 => sample std

        # 若 rolling_std 为 0 或者 NaN，则无法比较
        if np.isnan(rolling_mean) or np.isnan(rolling_std) or rolling_std == 0:
            result[i] = np.nan
        else:
            if raw_array[i] >= rolling_mean + z_score * rolling_std:
                result[i] = 1.0
            else:
                result[i] = 0.0

    return result
