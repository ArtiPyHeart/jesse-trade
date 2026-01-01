"""Hurst指数计算模块

使用R/S分析（Rescaled Range Analysis）计算Hurst指数。
- H > 0.5: 趋势性市场（持续上升或下降）
- H ≈ 0.5: 随机游走
- H < 0.5: 反趋势市场（均值回归）
"""

import numpy as np
from scipy.stats import linregress


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

    # 确保max_lag不超过序列长度的一半
    max_lag = min(max_lag, len(series) // 2)

    if max_lag < min_lag:
        return np.nan

    # 对数收益率（与notebook一致，清理NaN）
    with np.errstate(divide="ignore", invalid="ignore"):
        log_returns = np.diff(np.log(series))
    # 移除NaN（与notebook的dropna一致）
    log_returns = log_returns[~np.isnan(log_returns)]

    if len(log_returns) <= max_lag:
        return np.nan

    lags = list(range(min_lag, max_lag + 1))
    rs_values = []

    for lag in lags:
        rs_list = []
        for i in range(len(log_returns) - lag + 1):
            segment = log_returns[i : i + lag]
            mean_ = np.mean(segment)
            deviations = segment - mean_
            cumulative = np.cumsum(deviations)
            range_ = np.max(cumulative) - np.min(cumulative)
            std_ = np.std(segment, ddof=1)  # 与pandas默认一致
            if std_ > 0:
                rs_list.append(range_ / std_)

        if rs_list:
            rs_values.append(np.mean(rs_list))

    # 至少需要3个点进行线性回归
    if len(rs_values) < 3:
        return np.nan

    # 过滤掉非正值（log要求正数）
    valid_mask = np.array(rs_values) > 0
    if valid_mask.sum() < 3:
        return np.nan

    valid_lags = np.array(lags)[: len(rs_values)][valid_mask]
    valid_rs = np.array(rs_values)[valid_mask]

    try:
        slope, _, _, _, _ = linregress(np.log(valid_lags), np.log(valid_rs))
        return float(slope)
    except Exception:
        return np.nan


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
