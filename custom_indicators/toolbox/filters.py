import numpy as np
from jesse.indicators.high_pass import high_pass_fast
from jesse.indicators.high_pass_2_pole import high_pass_2_pole_fast
from jesse.indicators.supersmoother import supersmoother_fast
from jesse.indicators.supersmoother_3_pole import (
    supersmoother_fast as super_smoother_3_pole_fast,
)


def high_pass_filter(source: np.ndarray, period: int = 48) -> np.ndarray:
    """
    高通滤波器
    """
    return high_pass_fast(source, period)


def high_pass_2_pole_filter(source: np.ndarray, period: int = 48) -> np.ndarray:
    """
    2 阶高通滤波器
    """
    return high_pass_2_pole_fast(source, period)


def super_smoother_filter(source: np.ndarray, period: int = 10) -> np.ndarray:
    """
    超平滑滤波器
    """
    return supersmoother_fast(source, period)


def super_smoother_3_pole_filter(source: np.ndarray, period: int = 10) -> np.ndarray:
    """
    3 阶超平滑滤波器
    """
    return super_smoother_3_pole_fast(source, period)


def z_score_filter_np(
    raw_array: np.ndarray,
    mean_window: int = 20,
    std_window: int = 20,
    z_score: float = 3,
    sequential: bool = False,
) -> np.ndarray:
    """
    适用于 Jesse 的 z_score_filter 函数示例。
    输入：一维 numpy array。
    输出：当 sequential=True 时返回同长度的 numpy array，值为 1 或 0 或 np.nan；
         当 sequential=False 时返回单个值。

    :param raw_array: (np.ndarray) 任意一维 numpy array。
    :param mean_window: (int) 计算 rolling mean 的窗口大小。
    :param std_window: (int) 计算 rolling std 的窗口大小。
    :param z_score: (float) 标准差倍数阈值。
    :param sequential: (bool) 是否返回完整序列。默认为 False，只返回最后一个值。
    :return: (np.ndarray) 当 sequential=True 时返回与 raw_array 相同长度的结果数组，
             当 sequential=False 时返回最后一个计算结果。
             内容：1（满足条件）、0（不满足条件）、np.nan（无法计算）。
    """
    length = len(raw_array)
    min_idx = max(mean_window, std_window) - 1

    if not sequential:
        # 只计算最后一个值
        if length <= min_idx:
            return np.nan

        window_slice_mean = raw_array[-mean_window:]
        window_slice_std = raw_array[-std_window:]

        rolling_mean = np.mean(window_slice_mean)
        rolling_std = np.std(window_slice_std, ddof=1)

        if np.isnan(rolling_mean) or np.isnan(rolling_std):
            return np.nan

        return 1.0 if raw_array[-1] >= rolling_mean + z_score * rolling_std else 0.0

    # sequential=True 时计算完整序列
    result = np.full(length, np.nan, dtype=float)

    for i in range(min_idx, length):
        window_slice_mean = raw_array[i - mean_window + 1 : i + 1]
        window_slice_std = raw_array[i - std_window + 1 : i + 1]

        rolling_mean = np.mean(window_slice_mean)
        rolling_std = np.std(window_slice_std, ddof=1)

        if np.isnan(rolling_mean) or np.isnan(rolling_std) or rolling_std == 0:
            result[i] = np.nan
        else:
            result[i] = (
                1.0 if raw_array[i] >= rolling_mean + z_score * rolling_std else 0.0
            )

    return result
