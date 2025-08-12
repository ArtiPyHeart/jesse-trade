import numpy as np
import scipy.signal as signal
from numba import njit


@njit(cache=True, fastmath=True)
def _update_boundary_points(
    imf: np.ndarray, left_idx: int, right_idx: int, tol: float
) -> tuple:
    """使用Newton-Raphson方法更新边界点"""
    n = len(imf)

    # 边界检查
    if left_idx <= 0 or left_idx >= n - 1 or right_idx <= 0 or right_idx >= n - 1:
        return imf[left_idx], imf[right_idx], False

    # 一阶导数（中心差分）
    df_left = (imf[left_idx + 1] - imf[left_idx - 1]) * 0.5
    df_right = (imf[right_idx + 1] - imf[right_idx - 1]) * 0.5

    # 二阶导数（中心差分）
    d2f_left = imf[left_idx + 1] - 2 * imf[left_idx] + imf[left_idx - 1]
    d2f_right = imf[right_idx + 1] - 2 * imf[right_idx] + imf[right_idx - 1]

    # 避免除零错误
    if abs(d2f_left) < 1e-10 or abs(d2f_right) < 1e-10:
        return imf[left_idx], imf[right_idx], False

    # Newton-Raphson更新
    new_left = imf[left_idx] - df_left / d2f_left
    new_right = imf[right_idx] - df_right / d2f_right

    # 检查收敛
    converged = (
        abs(new_left - imf[left_idx]) < tol and abs(new_right - imf[right_idx]) < tol
    )

    return new_left, new_right, converged


@njit(cache=True)
def _find_extrema_indices(peaks: np.ndarray, valleys: np.ndarray) -> np.ndarray:
    """高效合并和排序极值点索引"""
    n_peaks = len(peaks)
    n_valleys = len(valleys)
    total = n_peaks + n_valleys

    if total == 0:
        return np.empty(0, dtype=np.int64)

    # 预分配数组
    extrema = np.empty(total, dtype=np.int64)

    # 合并数组
    extrema[:n_peaks] = peaks
    extrema[n_peaks:] = valleys

    # 排序
    extrema.sort()

    return extrema


def nrbo(imf: np.ndarray, max_iter: int = 10, tol: float = 1e-6) -> np.ndarray:
    """
    Newton-Raphson Boundary Optimization (NRBO)

    使用Newton-Raphson方法优化IMF的边界点，以改善边界效应。

    Parameters:
    -----------
    imf : np.ndarray
        输入的IMF（本征模态函数）
    max_iter : int, default=10
        最大迭代次数
    tol : float, default=1e-6
        收敛容差

    Returns:
    --------
    np.ndarray
        优化后的IMF

    Notes:
    ------
    该算法通过迭代调整边界极值点的值来减少边界效应。
    使用Newton-Raphson方法基于一阶和二阶导数来更新边界点。
    """
    # 输入验证
    if len(imf) < 3:
        return imf.copy()

    # 创建副本以避免修改原始数据
    imf_copy = np.ascontiguousarray(imf.copy(), dtype=np.float64)

    for iteration in range(max_iter):
        # 查找极值点
        peaks, _ = signal.find_peaks(imf_copy)
        valleys, _ = signal.find_peaks(-imf_copy)

        # 使用numba优化的函数合并极值点
        extrema = _find_extrema_indices(peaks, valleys)

        if len(extrema) < 2:
            break

        # 获取边界极值点
        left_extrema = extrema[0]
        right_extrema = extrema[-1]

        # 使用numba加速的函数更新边界点
        new_left, new_right, converged = _update_boundary_points(
            imf_copy, left_extrema, right_extrema, tol
        )

        if converged:
            break

        # 更新边界点的值
        imf_copy[left_extrema] = new_left
        imf_copy[right_extrema] = new_right

    return imf_copy
