"""Hurst指数计算模块

使用R/S分析（Rescaled Range Analysis）计算Hurst指数。
- H > 0.5: 趋势性市场（持续上升或下降）
- H ≈ 0.5: 随机游走
- H < 0.5: 反趋势市场（均值回归）
"""

from typing import Optional

import numpy as np
from numba import njit


@njit(cache=True)
def _compute_rs_values(
    log_returns: np.ndarray,
    min_lag: int,
    max_lag: int,
) -> tuple[np.ndarray, int]:
    n_lags = max_lag - min_lag + 1
    rs_values = np.empty(n_lags, dtype=np.float64)
    count = 0
    n = len(log_returns)

    for lag in range(min_lag, max_lag + 1):
        total = 0.0
        valid = 0
        window_count = n - lag + 1
        for i in range(window_count):
            mean_ = 0.0
            for j in range(lag):
                mean_ += log_returns[i + j]
            mean_ /= lag

            cumulative = 0.0
            min_cum = 0.0
            max_cum = 0.0
            var = 0.0
            for j in range(lag):
                diff = log_returns[i + j] - mean_
                var += diff * diff
                cumulative += diff
                if cumulative < min_cum:
                    min_cum = cumulative
                if cumulative > max_cum:
                    max_cum = cumulative

            if lag > 1:
                std_ = np.sqrt(var / (lag - 1))
            else:
                std_ = 0.0

            if std_ > 0.0:
                total += (max_cum - min_cum) / std_
                valid += 1

        if valid > 0:
            rs_values[count] = total / valid
            count += 1

    return rs_values, count


@njit(cache=True)
def _linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return np.nan

    x_mean = 0.0
    y_mean = 0.0
    for i in range(n):
        x_mean += x[i]
        y_mean += y[i]
    x_mean /= n
    y_mean /= n

    sxx = 0.0
    sxy = 0.0
    for i in range(n):
        dx = x[i] - x_mean
        sxx += dx * dx
        sxy += dx * (y[i] - y_mean)

    if sxx == 0.0:
        return np.nan
    return sxy / sxx


def _calculate_hurst_from_log_returns(
    log_returns: np.ndarray,
    min_lag: int,
    max_lag: int,
    series_length: int,
    lags: Optional[np.ndarray] = None,
    log_lags: Optional[np.ndarray] = None,
) -> float:
    assert log_returns.ndim == 1, f"log_returns must be 1D, got {log_returns.ndim}D"
    assert min_lag >= 2, f"min_lag must be >= 2, got {min_lag}"
    assert max_lag >= min_lag, f"max_lag({max_lag}) must >= min_lag({min_lag})"

    max_lag = min(max_lag, series_length // 2)
    if max_lag < min_lag:
        return np.nan

    log_returns = log_returns[~np.isnan(log_returns)]
    if len(log_returns) <= max_lag:
        return np.nan

    rs_values, count = _compute_rs_values(log_returns, min_lag, max_lag)
    if count < 3:
        return np.nan

    rs_values = rs_values[:count]
    valid_mask = rs_values > 0
    if valid_mask.sum() < 3:
        return np.nan

    expected_len = max_lag - min_lag + 1
    if lags is None or len(lags) != expected_len:
        lags = np.arange(min_lag, max_lag + 1)
    if log_lags is None or len(log_lags) != expected_len:
        log_lags = np.log(lags)

    valid_log_lags = log_lags[:count][valid_mask]
    valid_rs = rs_values[valid_mask]

    log_valid_rs = np.log(valid_rs)
    slope = _linear_regression_slope(valid_log_lags, log_valid_rs)
    return float(slope)


def calculate_hurst(
    series: np.ndarray,
    min_lag: int,
    max_lag: int,
) -> float:
    """使用R/S分析计算Hurst指数

    Args:
        series: 价格序列（一维数组）
        min_lag: 最小滞后周期
        max_lag: 最大滞后周期

    Returns:
        Hurst指数，异常情况返回NaN
    """
    assert series.ndim == 1, f"series must be 1D, got {series.ndim}D"
    assert min_lag >= 2, f"min_lag must be >= 2, got {min_lag}"
    assert max_lag >= min_lag, f"max_lag({max_lag}) must >= min_lag({min_lag})"

    # 对数收益率（与notebook一致，清理NaN）
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.diff(np.log(series))
    return _calculate_hurst_from_log_returns(
        log_returns,
        min_lag,
        max_lag,
        len(series),
    )


def get_lag_params(window_size: int) -> tuple[int, int]:
    """根据窗口大小计算推荐的lag参数

    Args:
        window_size: 滑动窗口大小

    Returns:
        (min_lag, max_lag) 元组
    """
    min_lag = max(5, window_size // 6)
    max_lag = max(10, window_size // 3)
    return min_lag, max_lag
