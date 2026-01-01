"""平稳性检验模块

包含ADF（Augmented Dickey-Fuller）和KPSS（Kwiatkowski-Phillips-Schmidt-Shin）检验。

ADF检验:
- p > 0.05: 序列非平稳（有单位根）
- p <= 0.05: 序列平稳

KPSS检验:
- p < 0.05: 序列非平稳
- p >= 0.05: 序列平稳
"""

import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


def run_adf_test(series: np.ndarray) -> tuple[float, float]:
    """ADF检验（Augmented Dickey-Fuller Test）

    原假设: 序列存在单位根（非平稳）
    p > 0.05 表示非平稳，p <= 0.05 表示平稳

    Args:
        series: 价格序列（一维数组）

    Returns:
        (statistic, p_value) 元组
    """
    assert series.ndim == 1, f"series must be 1D, got {series.ndim}D"
    assert len(series) >= 10, f"series too short: {len(series)}"

    # 移除NaN
    clean_series = series[~np.isnan(series)]
    if len(clean_series) < 10:
        return np.nan, np.nan

    try:
        result = adfuller(clean_series, autolag="AIC")
        return float(result[0]), float(result[1])
    except Exception:
        return np.nan, np.nan


def run_kpss_test(
    series: np.ndarray,
    regression: str = "ct",
) -> tuple[float, float]:
    """KPSS检验（Kwiatkowski-Phillips-Schmidt-Shin Test）

    原假设: 序列围绕常数或趋势平稳
    p < 0.05 表示非平稳，p >= 0.05 表示平稳

    Args:
        series: 价格序列（一维数组）
        regression: 回归类型，'c'(常数) 或 'ct'(常数+趋势)

    Returns:
        (statistic, p_value) 元组
    """
    assert series.ndim == 1, f"series must be 1D, got {series.ndim}D"
    assert len(series) >= 10, f"series too short: {len(series)}"
    assert regression in ("c", "ct"), (
        f"regression must be 'c' or 'ct', got {regression}"
    )

    # 移除NaN
    clean_series = series[~np.isnan(series)]
    if len(clean_series) < 10:
        return np.nan, np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(clean_series, regression=regression)
            return float(result[0]), float(result[1])
    except Exception:
        return np.nan, np.nan
