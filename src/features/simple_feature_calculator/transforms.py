"""
独立的转换函数

所有转换函数直接操作numpy array，不涉及特征名解析
"""

from typing import Optional

import numpy as np
from numba import njit

# 延迟导入可选依赖
_gph_ripser_parallel = None
_scipy_entropy = None


def _ensure_gph_imported():
    """确保giotto-ph已导入"""
    global _gph_ripser_parallel, _scipy_entropy
    if _gph_ripser_parallel is None:
        try:
            from gph import ripser_parallel
            from scipy.stats import entropy

            _gph_ripser_parallel = ripser_parallel
            _scipy_entropy = entropy
        except ImportError:
            raise ImportError(
                "需要安装giotto-ph才能使用拓扑持续熵功能。\n"
                "请运行: pip install giotto-ph"
            )
    return _gph_ripser_parallel, _scipy_entropy


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


@njit(cache=True)
def rolling_skew(array: np.ndarray, window: int) -> np.ndarray:
    """
    滚动偏度（Skewness）

    偏度衡量分布的不对称性：
    - 正偏度：右尾较长，大部分值在左侧
    - 负偏度：左尾较长，大部分值在右侧
    - 零偏度：对称分布

    Args:
        array: 输入数组
        window: 窗口大小

    Returns:
        滚动偏度，前window-1个值为nan
    """
    if array.ndim == 1:
        result = np.full_like(array, np.nan, dtype=np.float64)
        for i in range(window - 1, len(array)):
            window_data = array[i - window + 1 : i + 1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std > 0:
                # 计算三阶中心矩
                m3 = np.mean((window_data - mean) ** 3)
                result[i] = m3 / (std**3)
            else:
                result[i] = np.nan
        return result
    else:
        result = np.full_like(array, np.nan, dtype=np.float64)
        for col in range(array.shape[1]):
            for i in range(window - 1, array.shape[0]):
                window_data = array[i - window + 1 : i + 1, col]
                mean = np.mean(window_data)
                std = np.std(window_data)
                if std > 0:
                    m3 = np.mean((window_data - mean) ** 3)
                    result[i, col] = m3 / (std**3)
                else:
                    result[i, col] = np.nan
        return result


@njit(cache=True)
def rolling_kurt(array: np.ndarray, window: int) -> np.ndarray:
    """
    滚动峰度（Kurtosis）

    峰度衡量分布的尾部厚度：
    - 正峰度（>0）：厚尾，极端值较多
    - 负峰度（<0）：薄尾，极端值较少
    - 零峰度：正态分布（使用超额峰度，正态分布为0）

    Args:
        array: 输入数组
        window: 窗口大小

    Returns:
        滚动超额峰度（减去3），前window-1个值为nan
    """
    if array.ndim == 1:
        result = np.full_like(array, np.nan, dtype=np.float64)
        for i in range(window - 1, len(array)):
            window_data = array[i - window + 1 : i + 1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            if std > 0:
                # 计算四阶中心矩
                m4 = np.mean((window_data - mean) ** 4)
                # 返回超额峰度（减去3，使正态分布的峰度为0）
                result[i] = m4 / (std**4) - 3.0
            else:
                result[i] = np.nan
        return result
    else:
        result = np.full_like(array, np.nan, dtype=np.float64)
        for col in range(array.shape[1]):
            for i in range(window - 1, array.shape[0]):
                window_data = array[i - window + 1 : i + 1, col]
                mean = np.mean(window_data)
                std = np.std(window_data)
                if std > 0:
                    m4 = np.mean((window_data - mean) ** 4)
                    result[i, col] = m4 / (std**4) - 3.0
                else:
                    result[i, col] = np.nan
        return result


@njit(cache=True)
def rolling_median(array: np.ndarray, window: int) -> np.ndarray:
    """
    滚动中位数

    Args:
        array: 输入数组
        window: 窗口大小

    Returns:
        滚动中位数，前window-1个值为nan
    """
    if array.ndim == 1:
        result = np.full_like(array, np.nan, dtype=np.float64)
        for i in range(window - 1, len(array)):
            result[i] = np.median(array[i - window + 1 : i + 1])
        return result
    else:
        result = np.full_like(array, np.nan, dtype=np.float64)
        for col in range(array.shape[1]):
            for i in range(window - 1, array.shape[0]):
                result[i, col] = np.median(array[i - window + 1 : i + 1, col])
        return result


@njit(cache=True)
def rolling_curvature(array: np.ndarray, window: int) -> np.ndarray:
    """
    滚动曲率

    曲率衡量曲线的弯曲程度：
    - 高曲率：急转弯，可能是趋势反转点
    - 低曲率：平滑趋势或直线运动
    - 中等曲率：正常的价格波动

    异常处理策略：
    - 数据不足（<3点）: 返回0（无弯曲）
    - 恒定序列: 返回0（直线无弯曲）
    - 分母接近0: 返回0（避免除零）

    Args:
        array: 输入数组
        window: 窗口大小

    Returns:
        滚动曲率均值，前window-1个值为0
    """
    if array.ndim == 1:
        result = np.full_like(array, 0.0, dtype=np.float64)

        for i in range(window - 1, len(array)):
            window_data = array[i - window + 1 : i + 1]

            # 检查数据有效性
            valid_mask = ~np.isnan(window_data)
            valid_data = window_data[valid_mask]

            # 需要至少3个点计算曲率
            if len(valid_data) < 3:
                result[i] = 0.0
                continue

            # 创建时间索引
            x = np.arange(len(valid_data), dtype=np.float64)
            y = valid_data.astype(np.float64)

            # 计算一阶导数（使用中心差分）
            dx_dt = np.zeros_like(x)
            dy_dt = np.zeros_like(y)

            # 边界使用前向/后向差分
            dx_dt[0] = x[1] - x[0]
            dy_dt[0] = y[1] - y[0]

            dx_dt[-1] = x[-1] - x[-2]
            dy_dt[-1] = y[-1] - y[-2]

            # 中间使用中心差分
            for j in range(1, len(x) - 1):
                dx_dt[j] = (x[j + 1] - x[j - 1]) / 2.0
                dy_dt[j] = (y[j + 1] - y[j - 1]) / 2.0

            # 计算二阶导数
            d2x_dt2 = np.zeros_like(x)
            d2y_dt2 = np.zeros_like(y)

            # 边界使用前向/后向差分
            d2x_dt2[0] = dx_dt[1] - dx_dt[0]
            d2y_dt2[0] = dy_dt[1] - dy_dt[0]

            d2x_dt2[-1] = dx_dt[-1] - dx_dt[-2]
            d2y_dt2[-1] = dy_dt[-1] - dy_dt[-2]

            # 中间使用中心差分
            for j in range(1, len(dx_dt) - 1):
                d2x_dt2[j] = (dx_dt[j + 1] - dx_dt[j - 1]) / 2.0
                d2y_dt2[j] = (dy_dt[j + 1] - dy_dt[j - 1]) / 2.0

            # 计算曲率
            curvature_values = np.zeros(len(x))
            for j in range(len(x)):
                numerator = np.abs(d2x_dt2[j] * dy_dt[j] - d2y_dt2[j] * dx_dt[j])
                denominator = (dx_dt[j] ** 2 + dy_dt[j] ** 2) ** 1.5

                if denominator > 1e-10:
                    curvature_values[j] = numerator / denominator
                else:
                    curvature_values[j] = 0.0

            # 返回平均曲率
            result[i] = np.mean(curvature_values)

        return result
    else:
        # 处理2D数组
        result = np.full_like(array, 0.0, dtype=np.float64)

        for col in range(array.shape[1]):
            for i in range(window - 1, array.shape[0]):
                window_data = array[i - window + 1 : i + 1, col]

                # 检查数据有效性
                valid_mask = ~np.isnan(window_data)
                valid_data = window_data[valid_mask]

                # 需要至少3个点计算曲率
                if len(valid_data) < 3:
                    result[i, col] = 0.0
                    continue

                # 创建时间索引
                x = np.arange(len(valid_data), dtype=np.float64)
                y = valid_data.astype(np.float64)

                # 计算一阶导数
                dx_dt = np.zeros_like(x)
                dy_dt = np.zeros_like(y)

                dx_dt[0] = x[1] - x[0]
                dy_dt[0] = y[1] - y[0]

                dx_dt[-1] = x[-1] - x[-2]
                dy_dt[-1] = y[-1] - y[-2]

                for j in range(1, len(x) - 1):
                    dx_dt[j] = (x[j + 1] - x[j - 1]) / 2.0
                    dy_dt[j] = (y[j + 1] - y[j - 1]) / 2.0

                # 计算二阶导数
                d2x_dt2 = np.zeros_like(x)
                d2y_dt2 = np.zeros_like(y)

                d2x_dt2[0] = dx_dt[1] - dx_dt[0]
                d2y_dt2[0] = dy_dt[1] - dy_dt[0]

                d2x_dt2[-1] = dx_dt[-1] - dx_dt[-2]
                d2y_dt2[-1] = dy_dt[-1] - dy_dt[-2]

                for j in range(1, len(dx_dt) - 1):
                    d2x_dt2[j] = (dx_dt[j + 1] - dx_dt[j - 1]) / 2.0
                    d2y_dt2[j] = (dy_dt[j + 1] - dy_dt[j - 1]) / 2.0

                # 计算曲率
                curvature_values = np.zeros(len(x))
                for j in range(len(x)):
                    numerator = np.abs(d2x_dt2[j] * dy_dt[j] - d2y_dt2[j] * dx_dt[j])
                    denominator = (dx_dt[j] ** 2 + dy_dt[j] ** 2) ** 1.5

                    if denominator > 1e-10:
                        curvature_values[j] = numerator / denominator
                    else:
                        curvature_values[j] = 0.0

                # 返回平均曲率
                result[i, col] = np.mean(curvature_values)

        return result


@njit(cache=True)
def rolling_hurst(array: np.ndarray, window: int, min_lag: int = 2) -> np.ndarray:
    """
    滚动Hurst指数（使用R/S分析法）

    Hurst指数衡量时间序列的长期记忆性：
    - H < 0.5: 均值回复（反持续性）
    - H = 0.5: 随机游走（布朗运动）
    - H > 0.5: 趋势持续（正持续性）

    拟合异常处理策略：
    - 数据恒定（无波动）: 返回0.5，表示无趋势特征
    - 数据量不足: 返回0.5，保守估计为随机游走
    - 拟合失败: 返回0.5，避免引入极端值

    Args:
        array: 输入数组
        window: 窗口大小
        min_lag: 最小滞后期数（默认2）

    Returns:
        滚动Hurst指数，前window-1个值为0.5（中性值）
    """
    if array.ndim == 1:
        result = np.full_like(array, 0.5, dtype=np.float64)

        for i in range(window - 1, len(array)):
            window_data = array[i - window + 1 : i + 1]

            # 检查数据有效性
            if np.any(np.isnan(window_data)):
                # 如果有nan，使用前一个值或保持0.5
                if i > window - 1:
                    result[i] = result[i - 1]
                continue

            # 计算均值调整后的序列
            mean = np.mean(window_data)
            centered = window_data - mean

            # 检查是否为恒定序列
            if np.std(centered) < 1e-10:
                result[i] = 0.5
                continue

            # R/S 分析
            max_lag = min(window // 2, 20)  # 限制最大滞后
            lags = []
            rs_values = []

            for lag in range(min_lag, max_lag + 1):
                if lag >= window:
                    break

                # 将数据分成多个子序列
                n_segments = window // lag
                if n_segments < 1:
                    continue

                # 预分配数组存储R/S值
                rs_segment = np.zeros(n_segments)
                valid_count = 0

                for seg in range(n_segments):
                    start = seg * lag
                    end = start + lag
                    if end > window:
                        break

                    segment = window_data[start:end]

                    # 计算均值调整的累积和
                    seg_mean = np.mean(segment)
                    deviations = segment - seg_mean
                    cumsum = np.cumsum(deviations)

                    # 计算Range
                    if len(cumsum) > 0:
                        R = np.max(cumsum) - np.min(cumsum)
                    else:
                        R = 0.0

                    # 计算标准差
                    S = np.std(segment)

                    # 计算R/S
                    if S > 1e-10:
                        rs_segment[valid_count] = R / S
                        valid_count += 1

                if valid_count > 0:
                    lags.append(float(lag))
                    rs_values.append(np.mean(rs_segment[:valid_count]))

            # 至少需要3个点进行拟合
            if len(rs_values) < 3:
                result[i] = 0.5
                continue

            # 拟合 log(R/S) = H * log(lag) + c
            try:
                log_lags = np.log(np.array(lags))
                log_rs = np.log(np.array(rs_values))

                # 过滤掉无效值
                valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
                if np.sum(valid_mask) < 3:
                    result[i] = 0.5
                    continue

                log_lags = log_lags[valid_mask]
                log_rs = log_rs[valid_mask]

                # 最小二乘法拟合
                x_mean = np.mean(log_lags)
                y_mean = np.mean(log_rs)

                numerator = np.sum((log_lags - x_mean) * (log_rs - y_mean))
                denominator = np.sum((log_lags - x_mean) ** 2)

                if abs(denominator) < 1e-10:
                    result[i] = 0.5
                    continue

                hurst = numerator / denominator

                # 确保Hurst值在合理范围内
                if 0.0 < hurst < 1.0:
                    result[i] = hurst
                elif hurst <= 0.0:
                    result[i] = 0.1
                elif hurst >= 1.0:
                    result[i] = 0.9
                else:
                    result[i] = 0.5

            except:
                # 拟合异常，使用前一个值或0.5
                if i > window - 1:
                    result[i] = result[i - 1]
                else:
                    result[i] = 0.5

        return result
    else:
        # 处理2D数组
        result = np.full_like(array, 0.5, dtype=np.float64)

        for col in range(array.shape[1]):
            for i in range(window - 1, array.shape[0]):
                window_data = array[i - window + 1 : i + 1, col]

                if np.any(np.isnan(window_data)):
                    if i > window - 1:
                        result[i, col] = result[i - 1, col]
                    continue

                # 计算均值调整后的序列
                mean = np.mean(window_data)
                centered = window_data - mean

                # 检查是否为恒定序列
                if np.std(centered) < 1e-10:
                    result[i, col] = 0.5
                    continue

                # R/S 分析
                max_lag = min(window // 2, 20)
                lags = []
                rs_values = []

                for lag in range(min_lag, max_lag + 1):
                    if lag >= window:
                        break

                    # 将数据分成多个子序列
                    n_segments = window // lag
                    if n_segments < 1:
                        continue

                    # 预分配数组存储R/S值
                    rs_segment = np.zeros(n_segments)
                    valid_count = 0

                    for seg in range(n_segments):
                        start = seg * lag
                        end = start + lag
                        if end > window:
                            break

                        segment = window_data[start:end]

                        # 计算均值调整的累积和
                        seg_mean = np.mean(segment)
                        deviations = segment - seg_mean
                        cumsum = np.cumsum(deviations)

                        # 计算Range
                        if len(cumsum) > 0:
                            R = np.max(cumsum) - np.min(cumsum)
                        else:
                            R = 0.0

                        # 计算标准差
                        S = np.std(segment)

                        # 计算R/S
                        if S > 1e-10:
                            rs_segment[valid_count] = R / S
                            valid_count += 1

                    if valid_count > 0:
                        lags.append(float(lag))
                        rs_values.append(np.mean(rs_segment[:valid_count]))

                # 至少需要3个点进行拟合
                if len(rs_values) < 3:
                    result[i, col] = 0.5
                    continue

                # 拟合 log(R/S) = H * log(lag) + c
                try:
                    log_lags = np.log(np.array(lags))
                    log_rs = np.log(np.array(rs_values))

                    # 过滤掉无效值
                    valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
                    if np.sum(valid_mask) < 3:
                        result[i, col] = 0.5
                        continue

                    log_lags = log_lags[valid_mask]
                    log_rs = log_rs[valid_mask]

                    # 最小二乘法拟合
                    x_mean = np.mean(log_lags)
                    y_mean = np.mean(log_rs)

                    numerator = np.sum((log_lags - x_mean) * (log_rs - y_mean))
                    denominator = np.sum((log_lags - x_mean) ** 2)

                    if abs(denominator) < 1e-10:
                        result[i, col] = 0.5
                        continue

                    hurst = numerator / denominator

                    # 确保Hurst值在合理范围内
                    if 0.0 < hurst < 1.0:
                        result[i, col] = hurst
                    elif hurst <= 0.0:
                        result[i, col] = 0.1
                    elif hurst >= 1.0:
                        result[i, col] = 0.9
                    else:
                        result[i, col] = 0.5

                except:
                    if i > window - 1:
                        result[i, col] = result[i - 1, col]
                    else:
                        result[i, col] = 0.5

        return result


def rolling_persistent_homology_entropy(array: np.ndarray, window: int) -> np.ndarray:
    """
    滚动持久同调熵（Persistent Homology Entropy）

    使用拓扑数据分析计算时间序列的持续熵，
    衡量序列中拓扑特征的复杂度和持续性模式。

    持续熵的含义：
    - 高熵：复杂的拓扑结构，多样的特征尺度
    - 低熵：简单的拓扑结构，特征尺度集中
    - 可用于检测市场状态变化、异常模式等

    Args:
        array: 输入数组
        window: 窗口大小（建议>=20）

    Returns:
        滚动持续熵，前window-1个值为nan

    Note:
        需要安装giotto-ph: pip install giotto-ph
    """
    # 确保导入了必要的库
    ripser_parallel, scipy_entropy = _ensure_gph_imported()

    if array.ndim == 1:
        result = np.full_like(array, np.nan, dtype=np.float64)

        for i in range(window - 1, len(array)):
            window_data = array[i - window + 1 : i + 1]

            # 跳过包含nan的窗口
            valid_mask = ~np.isnan(window_data)
            if not np.all(valid_mask):
                continue

            # 如果数据方差太小，返回0（无拓扑特征）
            if np.std(window_data) < 1e-10:
                result[i] = 0.0
                continue

            # 转换为点云格式 (n_points, 1)
            X = window_data.reshape(-1, 1)

            try:
                # 计算持久性图
                # 注意：对于1D点云（时间序列），maxdim=0与maxdim=1数学等价
                # 因为1D点集无法形成真正的1-cycles（H₁恒为0）
                # 使用maxdim=0可提升性能，尤其是在大窗口时
                ph_result = ripser_parallel(
                    X,
                    maxdim=0,  # 对1D时间序列，仅需0维同调
                    metric="euclidean",
                    n_threads=1,  # 小窗口时单线程最优
                )

                # 提取所有维度的寿命
                all_lifetimes = []
                for dim_dgm in ph_result["dgms"]:
                    if len(dim_dgm) > 0:
                        # 计算寿命（death - birth）
                        lifetimes = dim_dgm[:, 1] - dim_dgm[:, 0]
                        # 过滤掉零寿命和无穷寿命
                        valid_lifetimes = lifetimes[
                            (lifetimes > 1e-10) & (lifetimes != np.inf)
                        ]
                        if len(valid_lifetimes) > 0:
                            all_lifetimes.extend(valid_lifetimes)

                # 计算熵
                if len(all_lifetimes) > 0:
                    lifetimes_array = np.array(all_lifetimes)
                    # 归一化为概率分布
                    probabilities = lifetimes_array / np.sum(lifetimes_array)
                    # 计算香农熵（base 2）
                    result[i] = scipy_entropy(probabilities, base=2)
                else:
                    result[i] = 0.0

            except Exception:
                # 如果计算失败，使用前一个值或保持nan
                if i > window - 1 and not np.isnan(result[i - 1]):
                    result[i] = result[i - 1]
                else:
                    result[i] = np.nan

        return result
    else:
        # 处理2D数组，对每列分别计算
        result = np.full_like(array, np.nan, dtype=np.float64)
        for col in range(array.shape[1]):
            result[:, col] = rolling_persistent_homology_entropy(array[:, col], window)
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
        "skew": rolling_skew,
        "kurt": rolling_kurt,
        "median": rolling_median,
        "hurst": rolling_hurst,
        "curv": rolling_curvature,
        "phent": rolling_persistent_homology_entropy,
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
