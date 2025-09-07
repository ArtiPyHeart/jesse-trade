"""
独立的转换函数

所有转换函数直接操作numpy array，不涉及特征名解析
"""

from typing import Optional

import numpy as np
from numba import njit


@njit(cache=True)
def dt(array: np.ndarray) -> np.ndarray:
    """
    一阶差分

    Args:
        array: 输入数组

    Returns:
        一阶差分结果，第一个值为nan
    """
    if array.ndim == 1:
        result = np.empty_like(array)
        result[0] = np.nan
        result[1:] = array[1:] - array[:-1]
        return result
    else:
        # 处理2D数组，对每列分别计算
        result = np.empty_like(array)
        result[0, :] = np.nan
        result[1:, :] = array[1:, :] - array[:-1, :]
        return result


@njit(cache=True)
def ddt(array: np.ndarray) -> np.ndarray:
    """
    二阶差分

    Args:
        array: 输入数组

    Returns:
        二阶差分结果，前两个值为nan
    """
    if array.ndim == 1:
        result = np.empty_like(array)
        result[:2] = np.nan
        # 先计算一阶差分
        dt_result = np.empty_like(array)
        dt_result[0] = np.nan
        dt_result[1:] = array[1:] - array[:-1]
        # 再计算二阶差分
        result[2:] = dt_result[2:] - dt_result[1:-1]
        return result
    else:
        # 处理2D数组
        result = np.empty_like(array)
        result[:2, :] = np.nan
        # 先计算一阶差分
        dt_result = np.empty_like(array)
        dt_result[0, :] = np.nan
        dt_result[1:, :] = array[1:, :] - array[:-1, :]
        # 再计算二阶差分
        result[2:, :] = dt_result[2:, :] - dt_result[1:-1, :]
        return result


@njit(cache=True)
def lag(array: np.ndarray, n: int) -> np.ndarray:
    """
    滞后n期

    Args:
        array: 输入数组
        n: 滞后期数（正数表示向后滞后，负数表示向前）

    Returns:
        滞后结果，滞后部分填充nan
    """
    if array.ndim == 1:
        result = np.full_like(array, np.nan)
        if n > 0:
            # 向后滞后
            if n < len(array):
                result[n:] = array[:-n]
        elif n < 0:
            # 向前（实际是lead）
            if -n < len(array):
                result[:n] = array[-n:]
        else:
            # n == 0，直接复制
            result = array.copy()
        return result
    else:
        # 处理2D数组
        result = np.full_like(array, np.nan)
        if n > 0:
            if n < array.shape[0]:
                result[n:, :] = array[:-n, :]
        elif n < 0:
            if -n < array.shape[0]:
                result[:n, :] = array[-n:, :]
        else:
            result = array.copy()
        return result


@njit(cache=True)
def rolling_mean(array: np.ndarray, window: int) -> np.ndarray:
    """
    滚动均值

    Args:
        array: 输入数组
        window: 窗口大小

    Returns:
        滚动均值，前window-1个值为nan
    """
    if array.ndim == 1:
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            result[i] = np.mean(array[i - window + 1 : i + 1])
        return result
    else:
        # 处理2D数组，对每列分别计算
        result = np.full_like(array, np.nan)
        for col in range(array.shape[1]):
            for i in range(window - 1, array.shape[0]):
                result[i, col] = np.mean(array[i - window + 1 : i + 1, col])
        return result


@njit(cache=True)
def rolling_std(array: np.ndarray, window: int) -> np.ndarray:
    """
    滚动标准差

    Args:
        array: 输入数组
        window: 窗口大小

    Returns:
        滚动标准差，前window-1个值为nan
    """
    if array.ndim == 1:
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            result[i] = np.std(array[i - window + 1 : i + 1])
        return result
    else:
        result = np.full_like(array, np.nan)
        for col in range(array.shape[1]):
            for i in range(window - 1, array.shape[0]):
                result[i, col] = np.std(array[i - window + 1 : i + 1, col])
        return result


@njit(cache=True)
def rolling_max(array: np.ndarray, window: int) -> np.ndarray:
    """
    滚动最大值

    Args:
        array: 输入数组
        window: 窗口大小

    Returns:
        滚动最大值，前window-1个值为nan
    """
    if array.ndim == 1:
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            result[i] = np.max(array[i - window + 1 : i + 1])
        return result
    else:
        result = np.full_like(array, np.nan)
        for col in range(array.shape[1]):
            for i in range(window - 1, array.shape[0]):
                result[i, col] = np.max(array[i - window + 1 : i + 1, col])
        return result


@njit(cache=True)
def rolling_min(array: np.ndarray, window: int) -> np.ndarray:
    """
    滚动最小值

    Args:
        array: 输入数组
        window: 窗口大小

    Returns:
        滚动最小值，前window-1个值为nan
    """
    if array.ndim == 1:
        result = np.full_like(array, np.nan)
        for i in range(window - 1, len(array)):
            result[i] = np.min(array[i - window + 1 : i + 1])
        return result
    else:
        result = np.full_like(array, np.nan)
        for col in range(array.shape[1]):
            for i in range(window - 1, array.shape[0]):
                result[i, col] = np.min(array[i - window + 1 : i + 1, col])
        return result


class TransformChain:
    """转换链处理器"""

    # 支持的转换函数映射
    TRANSFORMS = {
        "dt": dt,
        "ddt": ddt,
        "lag": lag,
        "mean": rolling_mean,
        "std": rolling_std,
        "max": rolling_max,
        "min": rolling_min,
    }

    @classmethod
    def parse_transform_name(cls, transform_str: str) -> tuple[str, Optional[int]]:
        """
        解析转换字符串，提取转换名和参数

        例如:
        - "dt" -> ("dt", None)
        - "lag5" -> ("lag", 5)
        - "mean20" -> ("mean", 20)

        Args:
            transform_str: 转换字符串

        Returns:
            (转换名, 参数)
        """
        # 检查是否是纯转换名（无参数）
        if transform_str in cls.TRANSFORMS:
            return transform_str, None

        # 尝试解析带参数的转换
        for transform_name in cls.TRANSFORMS:
            if transform_str.startswith(transform_name):
                param_str = transform_str[len(transform_name) :]
                if param_str.isdigit():
                    return transform_name, int(param_str)

        # 无法识别的转换
        return None, None

    @classmethod
    def apply(cls, data: np.ndarray, transform_str: str) -> np.ndarray:
        """
        应用单个转换

        Args:
            data: 输入数据
            transform_str: 转换字符串（如"dt", "lag5", "mean20"）

        Returns:
            转换后的数据
        """
        transform_name, param = cls.parse_transform_name(transform_str)

        if transform_name is None:
            raise ValueError(f"Unknown transform: {transform_str}")

        transform_func = cls.TRANSFORMS[transform_name]

        if param is not None:
            # 带参数的转换
            return transform_func(data, param)
        else:
            # 无参数的转换
            return transform_func(data)

    @classmethod
    def apply_chain(cls, data: np.ndarray, transforms: list[str]) -> np.ndarray:
        """
        应用转换链

        Args:
            data: 输入数据
            transforms: 转换列表，如["mean20", "dt", "lag5"]

        Returns:
            转换后的数据
        """
        result = data.copy()
        for transform_str in transforms:
            result = cls.apply(result, transform_str)
        return result
