import math

import numpy as np
from numba import njit


@njit
def deg_sin(degrees: float) -> float:
    res = np.sin(np.deg2rad(degrees))
    return res


@njit
def deg_cos(degrees: float) -> float:
    res = np.cos(np.deg2rad(degrees))
    return res


@njit
def deg_tan(degrees: float) -> float:
    res = np.tan(np.deg2rad(degrees))
    return res


@njit
def dt(array: np.ndarray) -> np.ndarray:
    # 创建结果数组，与输入数组大小相同
    res = np.empty_like(array)
    # 第一个元素设为nan
    res[0] = np.nan
    # 计算差分
    res[1:] = np.diff(array)
    return res


@njit
def ddt(array: np.ndarray) -> np.ndarray:
    res = np.empty_like(array)
    res[0] = np.nan
    res[1:] = np.diff(dt(array))
    return res


@njit
def lag(array: np.ndarray, n: int) -> np.ndarray:
    result = np.full_like(array, np.nan)
    if n > 0:
        result[n:] = array[:-n]
    elif n < 0:
        result[:n] = array[-n:]
    else:
        result = array.copy()
    return result


@njit
def std(array: np.ndarray, n: int = 20) -> np.ndarray:
    # 使用cumsum方法创建rolling window
    ret = np.full_like(array, np.nan)

    # 计算移动窗口的标准差
    for i in range(n - 1, len(array)):
        ret[i] = np.std(array[i - n + 1 : i + 1], ddof=1)
    return ret


@njit
def skew(array: np.ndarray, n: int = 20) -> np.ndarray:
    ret = np.full_like(array, np.nan)

    for i in range(n - 1, len(array)):
        window = array[i - n + 1 : i + 1]
        # 计算中心矩
        m3 = np.mean((window - np.mean(window)) ** 3)
        _std = np.std(window, ddof=1)
        # 偏度计算公式
        ret[i] = m3 / (_std**3) if std != 0 else 0
    return ret


@njit
def kurtosis(array: np.ndarray, n: int = 20) -> np.ndarray:
    ret = np.full_like(array, np.nan)

    for i in range(n - 1, len(array)):
        window = array[i - n + 1 : i + 1]
        # 计算中心矩
        m4 = np.mean((window - np.mean(window)) ** 4)
        var = np.var(window, ddof=1)
        # 峰度计算公式
        ret[i] = (m4 / var**2) - 3 if var != 0 else 0
    return ret
