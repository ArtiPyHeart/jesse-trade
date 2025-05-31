import numpy as np
from numba import njit


@njit
def log_ret(
    candles: np.ndarray, window_on_vol: np.ndarray | list[int] | int
) -> list[np.ndarray]:
    log_ret_list = []

    if isinstance(window_on_vol, (int, float)):
        window_on_vol = [int(window_on_vol)] * len(candles)
    else:
        window_on_vol = np.array(window_on_vol)

    for idx, w_on_vol in enumerate(window_on_vol):
        if np.isfinite(w_on_vol):
            w_on_vol = round(w_on_vol)
            if idx - w_on_vol >= 0:
                log_ret_list.append(
                    np.log(candles[idx, 2] / candles[idx - w_on_vol : idx, 2])
                )
    return log_ret_list


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


@njit
def np_shift(array: np.ndarray, n: int) -> np.ndarray:
    res = np.full_like(array, np.nan)
    if n > 0:
        res[n:] = array[:-n]
    elif n < 0:
        res[:n] = array[-n:]
    return res


@njit
def np_array_fill_nan(todo: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    在todo数组开头填充nan，使得todo和target数组长度相同
    """
    res = np.concatenate((np.full(len(target) - len(todo), np.nan), todo))
    return res


@njit
def rolling_sum_with_nan(arr: np.ndarray, window: int) -> np.ndarray:
    if window > len(arr):
        # 窗口大于数组长度时，全部返回nan
        return np.full_like(arr, np.nan, dtype=np.float64)

    # 创建结果数组
    result = np.full_like(arr, np.nan, dtype=np.float64)

    # 计算第一个窗口的和
    current_sum = np.sum(arr[:window])
    result[window - 1] = current_sum

    # 使用滑动窗口计算剩余的和
    for i in range(window, len(arr)):
        current_sum = current_sum - arr[i - window] + arr[i]
        result[i] = current_sum

    return result


if __name__ == "__main__":
    arr = np.array([1, 2, 3, 4, 5])
    print(rolling_sum_with_nan(arr, 2))
