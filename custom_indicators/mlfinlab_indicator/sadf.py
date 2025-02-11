from typing import Tuple, Union

import numpy as np
from jesse import helpers
from numba import njit, prange


def _get_sadf_at_t(
    X: np.ndarray, y: np.ndarray, min_length: int, model: str, phi: float
) -> float:
    """
    Advances in Financial Machine Learning, Snippet 17.2, page 258.

    SADF's Inner Loop (get SADF value at t)

    :param X: (np.ndarray) Lagged values, constants, trend coefficients
    :param y: (np.ndarray) Y values (either y or y.diff())
    :param min_length: (int) Minimum number of samples needed for estimation
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :return: (float) SADF statistics for y.index[-1]
    """
    rows = y.shape[0]
    bsadf = -np.inf
    for start in range(0, rows - min_length + 1):
        y_sub = y[start:]
        X_sub = X[start:]
        b_mean, b_var = get_betas(X_sub, y_sub)
        if not np.isnan(b_mean[0]):
            b_mean_0 = b_mean[0, 0] if b_mean.ndim > 1 else b_mean[0]
            b_std_0 = (b_var[0, 0] ** 0.5) if b_var.size > 1 else np.nan
            with np.errstate(invalid="ignore"):
                all_adf = b_mean_0 / b_std_0
            if model[:2] == "sm":
                all_adf = np.abs(all_adf) / (rows**phi)
            if all_adf > bsadf:
                bsadf = all_adf
    return float(bsadf)


def _get_y_x(
    series: np.ndarray, model: str, lags: Union[int, list], add_const: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advances in Financial Machine Learning, Snippet 17.2, page 258-259.

    Preparing The Datasets

    :param series: (np.ndarray) Series to prepare for test statistics generation (for example log prices)
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param lags: (int or list) Either number of lags to use or array of specified lags
    :param add_const: (bool) Flag to add constant
    :return: (np.ndarray, np.ndarray) Prepared y and X for SADF generation
    """
    # 计算一阶差分
    diff_series = np.concatenate(([np.nan], np.diff(series)))
    # 准备滞后项
    x_lagged = _lag_ndarray(diff_series, lags)
    # 去掉开头因滞后导致的 NaN
    valid_idx = ~np.isnan(x_lagged).any(axis=1)
    x_lagged = x_lagged[valid_idx]
    y = diff_series[valid_idx]
    # 加入一列 y_(t-1)
    y_lag = np.concatenate(([np.nan], series[:-1]))[valid_idx]
    x_with_y_lag = np.column_stack((y_lag, x_lagged))

    # 若需要常数项
    if add_const:
        const_col = np.ones(len(x_with_y_lag))
        x_with_y_lag = np.column_stack((x_with_y_lag, const_col))

    # 根据 model 不同添加多项式/对数等列
    # 以保证与原有A/DF逻辑一致
    if model == "linear":
        trend_col = np.arange(len(x_with_y_lag))
        x_with_y_lag = np.column_stack((x_with_y_lag, trend_col))
    elif model == "quadratic":
        trend_col = np.arange(len(x_with_y_lag))
        quad_col = trend_col**2
        x_with_y_lag = np.column_stack((x_with_y_lag, trend_col, quad_col))
    elif model == "sm_poly_1":
        # y 替换为原生 series
        y = series[valid_idx]
        trend_col = np.arange(len(y))
        quad_col = trend_col**2
        x_with_y_lag = np.column_stack((np.ones(len(y)), trend_col, quad_col))
    elif model == "sm_poly_2":
        # y 改为 log(series)
        y = np.log(series[valid_idx])
        trend_col = np.arange(len(y))
        quad_col = trend_col**2
        x_with_y_lag = np.column_stack((np.ones(len(y)), trend_col, quad_col))
    elif model == "sm_exp":
        y = np.log(series[valid_idx])
        trend_col = np.arange(len(y))
        x_with_y_lag = np.column_stack((np.ones(len(y)), trend_col))
    elif model == "sm_power":
        y = np.log(series[valid_idx])
        # log_trend 可能会出现对0取 log 的情况，需要保护
        trend = np.arange(len(y))
        with np.errstate(divide="ignore"):
            log_trend = np.log(
                trend, out=np.zeros_like(trend, dtype=float), where=(trend > 0)
            )
        x_with_y_lag = np.column_stack((np.ones(len(y)), log_trend))
    else:
        raise ValueError("Unknown model")

    return x_with_y_lag, y.reshape(-1, 1)


def _lag_ndarray(values: np.ndarray, lags: Union[int, list]) -> np.ndarray:
    """
    Advances in Financial Machine Learning, Snipet 17.3, page 259.

    Apply Lags to DataFrame

    :param values: (np.ndarray) Series to apply lags
    :param lags: (int or list) Lag(s) to use
    :return: (np.ndarray) Dataframe with lags
    """
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    rows = len(values)
    # 收集所有滞后列
    lagged_cols = []
    for lag in lags:
        # 向下移动 lag 行，相当于 shift(lag)
        shifted = np.concatenate((np.full(lag, np.nan), values[:-lag]))
        lagged_cols.append(shifted)
    return np.array(lagged_cols).T  # 形状: (rows, len(lags))


def get_betas(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advances in Financial Machine Learning, Snippet 17.4, page 259.

    Fitting The ADF Specification (get beta estimate and estimate variance)

    :param X: (np.ndarray) Features(factors)
    :param y: (np.ndarray) Outcomes
    :return: (np.ndarray, np.ndarray) Betas and variances of estimates
    """
    # X.T X
    xx = X.T @ X
    # X.T y
    xy = X.T @ y
    # 防止奇异矩阵的异常
    try:
        xx_inv = np.linalg.inv(xx)
    except np.linalg.LinAlgError:
        return np.array([[np.nan]]), np.array([[np.nan, np.nan]])

    b_mean = xx_inv @ xy
    # 残差
    err = y - X @ b_mean
    dof = X.shape[0] - X.shape[1]
    if dof <= 0:
        return np.array([[np.nan]]), np.array([[np.nan, np.nan]])

    b_var = (err.T @ err) / dof * xx_inv
    return b_mean, b_var


def _sadf_outer_loop(
    X: np.ndarray,
    y: np.ndarray,
    min_length: int,
    model: str,
    phi: float,
    molecule: list,
) -> np.ndarray:
    """
    This function gets SADF for t times from molecule

    :param X: (np.ndarray) Features(factors)
    :param y: (np.ndarray) Outcomes
    :param min_length: (int) Minimum number of observations
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :param molecule: (list) Indices to get SADF
    :return: (np.ndarray) SADF statistics
    """
    sadf_series = np.full(len(molecule), np.nan, dtype=float)
    for index in molecule:
        X_subset = X[:index]
        y_subset = y[:index]
        value = _get_sadf_at_t(X_subset, y_subset, min_length, model, phi)
        sadf_series[index] = value
    return sadf_series


@njit
def get_betas_jit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    与原 get_betas 相同的逻辑，但用 numba 加速。
    若 X 维度过于动态或 np.linalg.inv 出现问题，可考虑稍加改造。
    """
    xx = X.T @ X
    xy = X.T @ y
    try:
        xx_inv = np.linalg.inv(xx)
    except np.linalg.LinAlgError:
        return np.array([[np.nan]]), np.array([[np.nan, np.nan]])

    b_mean = xx_inv @ xy
    err = y - X @ b_mean
    dof = X.shape[0] - X.shape[1]
    if dof <= 0:
        return np.array([[np.nan]]), np.array([[np.nan, np.nan]])

    b_var = (err.T @ err) / dof * xx_inv
    return b_mean, b_var


@njit(parallel=True)
def _get_sadf_at_t_jit(
    X: np.ndarray, y: np.ndarray, min_length: int, model: str, phi: float
) -> float:
    """
    numba并行加速版本，与 _get_sadf_at_t 类似。
    并行体现在内层start循环上，需使用 prange。
    """
    rows = y.shape[0]
    bsadf = -np.inf
    for start in prange(rows - min_length + 1):
        y_sub = y[start:]
        X_sub = X[start:]
        b_mean, b_var = get_betas_jit(X_sub, y_sub)
        if not np.isnan(b_mean[0]):
            b_mean_0 = b_mean[0, 0] if b_mean.ndim > 1 else b_mean[0]
            b_std_0 = (b_var[0, 0] ** 0.5) if b_var.size > 1 else np.nan
            if np.isnan(b_std_0) or b_std_0 == 0:
                continue
            all_adf = b_mean_0 / b_std_0
            if model[:2] == "sm":
                all_adf = np.abs(all_adf) / (rows**phi)
            if all_adf > bsadf:
                bsadf = all_adf
    return float(bsadf)


def get_sadf(
    candles: np.ndarray,
    model: str,
    lags: Union[int, list],
    min_length: int,
    add_const: bool = False,
    phi: float = 0,
    sequential: bool = False,
    source_type: str = "close",
) -> Union[float, np.ndarray]:
    """
    Advances in Financial Machine Learning, p. 258-259.

    Multithread implementation of SADF

    SADF fits the ADF regression at each end point t with backwards expanding start points. For the estimation
    of SADF(t), the right side of the window is fixed at t. SADF recursively expands the beginning of the sample
    up to t - min_length, and returns the sup of this set.

    When doing with sub- or super-martingale test, the variance of beta of a weak long-run bubble may be smaller than
    one of a strong short-run bubble, hence biasing the method towards long-run bubbles. To correct for this bias,
    ADF statistic in samples with large lengths can be penalized with the coefficient phi in [0, 1] such that:

    ADF_penalized = ADF / (sample_length ^ phi)

    :param candles: (np.ndarray) Jesse 的蜡烛图数据
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param lags: (int or list) Either number of lags to use or array of specified lags
    :param min_length: (int) Minimum number of observations needed for estimation
    :param add_const: (bool) Flag to add constant
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :param sequential: (bool) 若为False，只返回最后一根K线的结果；若为True，则返回整条序列
    :param source_type: (str) 取用蜡烛图的哪种价格/数据，默认'close'
    :return: (np.ndarray或float) 返回一条SADF指标值或整条序列
    """
    # 获取蜡烛图的目标数据源
    source = helpers.get_candle_source(candles, source_type=source_type)
    # 去除开头可能的 NaN
    source = source[~np.isnan(source)]
    if len(source) <= min_length:
        raise ValueError("数据长度必须大于最小样本长度 min_length。")

    # 准备 X, Y
    X_all, y_all = _get_y_x(source, model, lags, add_const)

    # min_length 需在拼合后的 X, y 范围内
    total_len = len(y_all)
    if total_len < min_length:
        raise ValueError("处理后的数据长度不足以进行 SADF 计算。")

    # 逐点计算
    sadf_vals = np.full(total_len, np.nan, dtype=float)
    for i in range(min_length - 1, total_len):
        X_subset = X_all[: i + 1]
        y_subset = y_all[: i + 1]
        sadf_val = _get_sadf_at_t_jit(X_subset, y_subset, min_length, model, phi)
        sadf_vals[i] = sadf_val

    # 若 sequential = False，则只返回最后一个值
    return sadf_vals if sequential else sadf_vals[-1]
