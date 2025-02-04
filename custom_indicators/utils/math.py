import math

import numpy as np
from numba import njit


@njit
def sin_radians(degrees: float) -> float:
    return math.sin(math.radians(degrees))


@njit
def cos_radians(degrees: float) -> float:
    return math.cos(math.radians(degrees))


@njit
def tan_radians(degrees: float) -> float:
    return math.tan(math.radians(degrees))


@njit
def dt(array: np.ndarray) -> np.ndarray:
    return np.diff(array, prepend=np.nan, axis=0)


@njit
def ddt(array: np.ndarray) -> np.ndarray:
    return np.diff(dt(array), prepend=np.nan, axis=0)


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
        std = np.std(window, ddof=1)
        # 偏度计算公式
        ret[i] = m3 / (std**3) if std != 0 else 0
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
