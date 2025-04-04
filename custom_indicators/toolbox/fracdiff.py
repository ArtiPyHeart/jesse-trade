import numpy as np
from numba import njit
from statsmodels.tsa.stattools import adfuller


@njit
def get_weights_ffd(diff_amt: float, thresh: float, lim: int):
    """
    计算固定宽度窗口分数差分的权重

    参数:
    ----------
    diff_amt : float
        差分系数
    thresh : float
        权重的最小阈值
    lim : int
        权重向量的最大长度

    返回:
    ----------
    np.ndarray
        权重向量
    """
    weights = [1.0]
    k = 1

    # 迭代计算权重
    ctr = 0
    while True:
        # 计算下一个权重
        weights_ = -weights[-1] * (diff_amt - k + 1) / k

        if abs(weights_) < thresh:
            break

        weights.append(weights_)
        k += 1
        ctr += 1
        if ctr == lim - 1:  # 达到大小限制时退出循环
            break

    # 反转列表，转换为列向量
    weights = np.array(weights[::-1]).reshape(-1, 1)
    return weights


def frac_diff_ffd(series: np.ndarray, diff_amt: float, thresh=1e-5, sequential=True):
    """
    分数差分函数 - 固定宽度窗口（Fixed-width Window Fractional Differentiation）

    该函数实现了分数差分，可以使时间序列平稳化的同时尽可能保留记忆。
    这是通过用一个固定宽度的窗口，对过去的数据点进行加权平均来实现的。

    参数:
    ----------
    series : np.ndarray
        一维数组，需要进行分数差分的时间序列
    diff_amt : float
        差分系数，可以是任意正小数
    thresh : float, 可选
        权重的最小阈值，决定窗口截断点，默认1e-5
    sequential : bool, 可选
        是否返回整个序列，若为False则只返回最后一个值，默认True

    返回:
    ----------
    np.ndarray 或 float
        分数差分后的序列，若sequential=False则返回最后一个值
    """
    # 确保series是一维数组
    series = np.asarray(series)
    if series.ndim > 1:
        series = series.flatten()

    # 1. 计算权重
    weights = get_weights_ffd(diff_amt, thresh, len(series))
    width = len(weights) - 1

    # 2. 应用权重计算分数差分
    # 创建结果数组，前width个位置无法计算，设为NaN
    result = np.full_like(series, np.nan, dtype=float)

    # 从width位置开始计算分数差分
    for i in range(width, len(series)):
        # 取当前位置减去width到当前位置的数据
        window = series[i - width : i + 1]
        # 计算当前位置的分数差分
        result[i] = np.sum(weights.flatten() * window)

    # 3. 根据sequential参数决定返回值
    if sequential:
        return result
    else:
        if np.isnan(result[-1]):
            return 0.0  # 如果最后一个值是NaN，返回0
        return result[-1]


def find_min_diff_amt(
    series: np.ndarray, adf_thresh=0.05, step=0.01, max_diff=1.0, ffd_thresh=1e-5
):
    """
    寻找使时间序列平稳的最小分数差分系数 d

    通过迭代增加差分系数 d，并使用ADF检验判断序列平稳性，
    找到使序列在统计上显著平稳的最小 d 值。

    参数:
    ----------
    series : np.ndarray
        一维数组，需要进行处理的时间序列
    adf_thresh : float, 可选
        ADF检验的p值阈值，用于判断平稳性，默认0.05
    step : float, 可选
        每次迭代增加的差分系数值，默认0.01
    max_diff : float, 可选
        尝试的最大差分系数值，默认1.0
    ffd_thresh : float, 可选
        传递给 frac_diff_ffd 的权重阈值，默认1e-5

    返回:
    ----------
    float or None
        找到的最小差分系数 d。如果在 max_diff 范围内未找到，则返回 None。
    """
    series = np.asarray(series)
    if series.ndim > 1:
        series = series.flatten()

    current_diff = 0.0
    while current_diff <= max_diff:
        diff_series = frac_diff_ffd(
            series, current_diff, thresh=ffd_thresh, sequential=True
        )
        # 移除前导的 NaN 值
        diff_series_clean = diff_series[~np.isnan(diff_series)]

        if len(diff_series_clean) < 20:  # ADF检验至少需要一些数据点
            print(
                f"Warning: Series too short for ADF test after differentiation with d={current_diff:.2f}. Skipping."
            )
            current_diff += step
            continue

        try:
            adf_result = adfuller(
                diff_series_clean, maxlag=1, regression="c", autolag=None
            )
            p_value = adf_result[1]

            if p_value < adf_thresh:
                return current_diff
        except Exception as e:
            print(f"Error during ADF test for d={current_diff:.2f}: {e}")
            # 如果ADF检验出错，可以选择跳过或停止
            pass  # 继续尝试下一个 d 值

        current_diff += step
        # 浮点数精度问题处理
        current_diff = round(current_diff, 8)

    print(f"Warning: Minimum diff_amt not found within the range [0, {max_diff}].")
    return None


def stationary_test(series: np.ndarray, adf_thresh=0.05):
    """
    使用ADF检验判断时间序列是否平稳。

    参数:
    ----------
    series : np.ndarray
        一维数组，需要检验的时间序列。
    adf_thresh : float, 可选
        ADF检验的p值阈值，用于判断平稳性，默认0.05。

    返回:
    ----------
    bool
        如果序列根据ADF检验是平稳的，则返回 True，否则返回 False。
    """
    series = np.asarray(series)
    if series.ndim > 1:
        series = series.flatten()

    # 移除NaN值
    series_clean = series[~np.isnan(series)]

    if len(series_clean) < 20:  # ADF检验至少需要一些数据点
        raise ValueError(
            f"Series is too short {len(series_clean) = } for ADF test after cleaning NaNs."
        )

    adf_result = adfuller(series_clean, maxlag=1, regression="c", autolag=None)
    p_value = adf_result[1]
    return p_value
