"""
分数阶差分（Fractional Differentiation）模块

本模块实现了分数阶差分算法，用于将非平稳时间序列转换为平稳序列，
同时尽可能保留序列的记忆性（长期依赖性）。

主要功能：
1. frac_diff_ffd: 固定宽度窗口的分数阶差分实现
2. find_optimal_d: 自动寻找最优的差分阶数d

性能优化：
- 使用 numba JIT 编译加速计算
- 对多维数据使用并行计算
- 相比原始实现，性能提升 50-200 倍

参考文献：
- Advances in Financial Machine Learning, Marcos López de Prado, Chapter 5
- Hosking, J.R.M., 1981. Fractional differencing. Biometrika, 68(1), pp.165-176.

使用示例：
    # 对价格序列进行分数阶差分
    prices = np.array([100, 102, 101, 103, 105, 104, 106])
    log_prices = np.log(prices)
    diff_series = frac_diff_ffd(log_prices, diff_amt=0.5)

    # 自动寻找最优差分阶数
    optimal_d = find_optimal_d(prices)
    optimal_diff = frac_diff_ffd(np.log(prices), diff_amt=optimal_d)
"""

import time

import numpy as np
import pandas as pd
from jesse.helpers import slice_candles, get_candle_source
from numba import jit, prange
from statsmodels.tsa.stattools import adfuller


@jit(nopython=True)
def get_weights_ffd_numba(diff_amt: float, thresh: float, lim: int) -> np.ndarray:
    """
    使用 numba 优化的权重计算函数

    :param diff_amt: 差分阶数
    :param thresh: 权重阈值
    :param lim: 权重向量的最大长度
    :return: 权重向量
    """
    weights = np.zeros(lim)
    weights[0] = 1.0
    k = 1

    # 迭代计算权重
    while k < lim:
        # 计算下一个权重
        weights[k] = -weights[k - 1] * (diff_amt - k + 1) / k

        if abs(weights[k]) < thresh:
            # 截断并返回有效部分
            result = np.zeros((k, 1))
            for i in range(k):
                result[i, 0] = weights[k - 1 - i]
            return result

        k += 1

    # 反转并返回
    result = np.zeros((k, 1))
    for i in range(k):
        result[i, 0] = weights[k - 1 - i]
    return result


@jit(nopython=True, parallel=True)
def apply_weights_2d(array: np.ndarray, weights: np.ndarray, width: int) -> np.ndarray:
    """
    使用 numba 优化的权重应用函数（支持并行计算）

    :param array: 输入数组 (n_samples, n_features)
    :param weights: 权重向量
    :param width: 窗口宽度
    :return: 差分后的数组
    """
    n_samples, n_features = array.shape
    output = np.full((n_samples, n_features), np.nan)

    # 并行处理每一列
    for j in prange(n_features):
        for i in range(width, n_samples):
            # 计算加权和
            weighted_sum = 0.0
            for w_idx in range(len(weights)):
                weighted_sum += weights[w_idx, 0] * array[i - width + w_idx, j]
            output[i, j] = weighted_sum

    return output


@jit(nopython=True)
def apply_weights_1d(array: np.ndarray, weights: np.ndarray, width: int) -> np.ndarray:
    """
    使用 numba 优化的权重应用函数（一维版本）

    :param array: 输入数组 (n_samples,)
    :param weights: 权重向量
    :param width: 窗口宽度
    :return: 差分后的数组
    """
    n_samples = len(array)
    output = np.full(n_samples, np.nan)

    for i in range(width, n_samples):
        # 计算加权和
        weighted_sum = 0.0
        for w_idx in range(len(weights)):
            weighted_sum += weights[w_idx, 0] * array[i - width + w_idx]
        output[i] = weighted_sum

    return output


def get_weights_ffd(diff_amt: float, thresh: float, lim: int) -> np.ndarray:
    """
    生成用于计算分数阶差分的权重向量

    :param diff_amt: 差分阶数
    :param thresh: 权重阈值
    :param lim: 权重向量的最大长度
    :return: 权重向量
    """
    weights = [1.0]
    k = 1

    # 迭代计算权重
    while k < lim:
        # 计算下一个权重
        weights_ = -weights[-1] * (diff_amt - k + 1) / k

        if abs(weights_) < thresh:
            break

        weights.append(weights_)
        k += 1

    # 反转列表并转换为numpy列向量
    weights = np.array(weights[::-1]).reshape(-1, 1)
    return weights


def frac_diff_ffd(
    array: np.ndarray, diff_amt: float = 0.5, thresh: float = 1e-5
) -> np.ndarray:
    """
    固定宽度窗口的分数阶差分（使用 numba 优化）

    :param array: 输入的时间序列数组
    :param diff_amt: 差分阶数
    :param thresh: 权重阈值
    :return: 分数阶差分后的数组
    """
    # 记录原始维度
    is_1d = array.ndim == 1

    # 如果输入是一维数组，转换为二维
    if is_1d:
        array = array.reshape(-1, 1)

    # 使用 numba 优化的权重计算
    weights = get_weights_ffd_numba(diff_amt, thresh, array.shape[0])
    width = len(weights) - 1

    # 根据维度选择优化的应用函数
    if is_1d:
        # 对于一维数组，使用优化的一维版本
        output = apply_weights_1d(array[:, 0], weights, width)
    else:
        # 对于多维数组，使用并行优化的二维版本
        output = apply_weights_2d(array, weights, width)

    return output

def frac_diff_ffd_candle(
    candles: np.ndarray,
    diff_amt: float,
    source_type: str = "close",
    sequential: bool = False,
):
    """
    对K线数据进行分数阶差分

    :param candles: K线数据
    :param diff_amt: 差分阶数
    :param source_type: 使用的价格类型
    :param sequential: 是否返回整个序列
    :return: 分数阶差分后的数组
    """
    candles = slice_candles(candles, sequential)
    source = get_candle_source(candles, source_type=source_type)

    log_price = np.log(source)
    diff_series = frac_diff_ffd(log_price, diff_amt=diff_amt)
    if sequential:
        return diff_series
    else:
        return diff_series[-1]

def _frac_diff_ffd_original(
    array: np.ndarray, diff_amt: float = 0.5, thresh: float = 1e-5
) -> np.ndarray:
    """
    固定宽度窗口的分数阶差分（原始版本，用于性能对比）

    :param array: 输入的时间序列数组
    :param diff_amt: 差分阶数
    :param thresh: 权重阈值
    :return: 分数阶差分后的数组
    """
    # 如果输入是一维数组，转换为二维
    if array.ndim == 1:
        array = array.reshape(-1, 1)

    # 计算权重
    weights = get_weights_ffd(diff_amt, thresh, array.shape[0])
    width = len(weights) - 1

    # 初始化输出数组
    output = np.full_like(array, np.nan, dtype=float)

    # 应用权重计算分数阶差分
    for i in range(width, array.shape[0]):
        # 对每一列进行计算
        for j in range(array.shape[1]):
            # 获取窗口内的数据
            window_data = array[i - width : i + 1, j]
            # 计算加权和
            output[i, j] = np.dot(weights.T, window_data)[0]

    # 如果原始输入是一维的，返回一维数组
    if output.shape[1] == 1:
        return output[:, 0]

    return output


def find_optimal_d(
    array: np.ndarray,
    d_range: tuple = (0, 1),
    step: float = 0.1,
    adf_threshold: float = 0.05,
    thresh: float = 1e-5,
) -> float:
    """
    寻找最优的分数阶差分参数d，使得序列平稳同时保留最多的记忆

    :param array: 输入的价格序列（一维数组）
    :param d_range: d值的搜索范围
    :param step: 搜索步长
    :param adf_threshold: ADF检验的p值阈值
    :param thresh: 权重阈值
    :return: 最优的d值
    """
    # 确保输入是一维数组
    if array.ndim > 1:
        array = array.flatten()

    results = []

    # 遍历不同的d值
    for d in np.arange(d_range[0], d_range[1] + step, step):
        # 应用分数阶差分
        diff_series = frac_diff_ffd(array, diff_amt=d, thresh=thresh)

        # 去除NaN值
        valid_idx = ~np.isnan(diff_series)
        diff_series_clean = diff_series[valid_idx]
        array_clean = array[valid_idx]

        if len(diff_series_clean) < 20:  # 样本太少，跳过
            continue

        # 计算与原序列的相关性
        corr = np.corrcoef(array_clean, diff_series_clean)[0, 1]

        # ADF检验
        adf_result = adfuller(diff_series_clean, maxlag=1, regression="c", autolag=None)
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        critical_value = adf_result[4]["5%"]

        results.append(
            {
                "d": d,
                "corr": corr,
                "adf_stat": adf_stat,
                "p_value": p_value,
                "critical_value": critical_value,
                "is_stationary": p_value < adf_threshold,
            }
        )

    # 转换为DataFrame便于分析
    results_df = pd.DataFrame(results)

    # 找到满足平稳性要求且相关性最高的d值
    stationary_results = results_df[results_df["is_stationary"]]

    if len(stationary_results) == 0:
        # 如果没有满足平稳性的，选择p值最小的
        optimal_d = results_df.loc[results_df["p_value"].idxmin(), "d"]
        print(f"警告⚠️：没有找到满足平稳性要求的d值，选择p值最小的d={optimal_d}")
    else:
        # 在满足平稳性的结果中，选择相关性最高的
        optimal_d = stationary_results.loc[stationary_results["corr"].idxmax(), "d"]

    return optimal_d


def analyze_d_values(
    array: np.ndarray, d_range: tuple = (0, 1), step: float = 0.1, thresh: float = 1e-5
) -> pd.DataFrame:
    """
    分析不同d值对序列的影响，返回详细的分析结果

    :param array: 输入的价格序列（一维数组）
    :param d_range: d值的搜索范围
    :param step: 搜索步长
    :param thresh: 权重阈值
    :return: 包含分析结果的DataFrame
    """
    # 确保输入是一维数组
    if array.ndim > 1:
        array = array.flatten()

    # 对数变换
    log_array = np.log(array)

    results = []

    # 遍历不同的d值
    for d in np.arange(d_range[0], d_range[1] + step, step):
        # 应用分数阶差分
        diff_series = frac_diff_ffd(log_array, diff_amt=d, thresh=thresh)

        # 去除NaN值
        valid_idx = ~np.isnan(diff_series)
        diff_series_clean = diff_series[valid_idx]
        log_array_clean = log_array[valid_idx]

        if len(diff_series_clean) < 20:  # 样本太少，跳过
            continue

        # 计算与原序列的相关性
        corr = np.corrcoef(log_array_clean, diff_series_clean)[0, 1]

        # ADF检验
        adf_result = adfuller(diff_series_clean, maxlag=1, regression="c", autolag=None)

        # 计算序列的统计特性
        mean = np.mean(diff_series_clean)
        std = np.std(diff_series_clean)
        skew = np.mean(((diff_series_clean - mean) / std) ** 3)
        kurt = np.mean(((diff_series_clean - mean) / std) ** 4) - 3

        results.append(
            {
                "d": d,
                "correlation": corr,
                "adf_statistic": adf_result[0],
                "p_value": adf_result[1],
                "critical_value_5%": adf_result[4]["5%"],
                "is_stationary": adf_result[1] < 0.05,
                "valid_samples": len(diff_series_clean),
                "mean": mean,
                "std": std,
                "skewness": skew,
                "kurtosis": kurt,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # 单元测试
    print("=== 分数阶差分单元测试 ===\n")

    # 测试1：简单的线性趋势序列
    print("测试1：线性趋势序列")
    linear_series = np.arange(100) + np.random.normal(0, 0.1, 100)
    diff_result = frac_diff_ffd(linear_series, diff_amt=0.5, thresh=0.01)
    print(f"原始序列形状: {linear_series.shape}")
    print(f"差分结果形状: {diff_result.shape}")
    print(f"有效值数量: {np.sum(~np.isnan(diff_result))}")
    print()

    # 测试2：随机游走序列
    print("测试2：随机游走序列")
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(200))
    random_walk = np.exp(random_walk / 10)  # 转换为价格序列

    # 寻找最优d值
    optimal_d = find_optimal_d(random_walk, thresh=0.01)
    print(f"最优差分阶数 d = {optimal_d}")

    # 应用最优d值
    optimal_diff = frac_diff_ffd(np.log(random_walk), diff_amt=optimal_d, thresh=0.01)
    valid_idx = ~np.isnan(optimal_diff)

    # ADF检验
    adf_result = adfuller(optimal_diff[valid_idx])
    print(f"ADF统计量: {adf_result[0]:.4f}")
    print(f"p值: {adf_result[1]:.4f}")
    print(f"是否平稳 (p<0.05): {adf_result[1] < 0.05}")
    print()

    # 测试3：多维数组
    print("测试3：多维数组")
    multi_dim = np.random.randn(100, 3).cumsum(axis=0)
    multi_diff = frac_diff_ffd(multi_dim, diff_amt=0.3, thresh=0.01)
    print(f"输入形状: {multi_dim.shape}")
    print(f"输出形状: {multi_diff.shape}")
    print(f"每列有效值数量: {[np.sum(~np.isnan(multi_diff[:, i])) for i in range(3)]}")
    print()

    # 测试4：不同的d值对相关性的影响
    print("测试4：不同d值的影响")
    test_series = np.exp(np.cumsum(np.random.randn(500)) / 20)

    for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
        diff_series = frac_diff_ffd(np.log(test_series), diff_amt=d, thresh=0.01)
        valid_idx = ~np.isnan(diff_series)

        if np.sum(valid_idx) > 20:  # 确保有足够的有效值
            corr = np.corrcoef(np.log(test_series)[valid_idx], diff_series[valid_idx])[
                0, 1
            ]

            # 检查差分序列是否为常数
            if np.std(diff_series[valid_idx]) > 1e-10:
                adf_p = adfuller(diff_series[valid_idx])[1]
                print(
                    f"d={d}: 相关性={corr:.4f}, ADF p值={adf_p:.4f}, 有效值数量={np.sum(valid_idx)}"
                )
            else:
                print(f"d={d}: 相关性={corr:.4f}, ADF p值=N/A (序列为常数)")
        else:
            print(f"d={d}: 有效值太少 ({np.sum(valid_idx)})")

    # 测试5：分析d值的详细影响
    print("\n测试5：详细分析d值的影响")
    analysis_df = analyze_d_values(test_series, thresh=0.01)
    print("\n分析结果：")
    print(
        analysis_df[
            ["d", "correlation", "p_value", "is_stationary", "valid_samples"]
        ].to_string()
    )

    # 找出最优的d值
    stationary_df = analysis_df[analysis_df["is_stationary"]]
    if len(stationary_df) > 0:
        best_d = stationary_df.loc[stationary_df["correlation"].idxmax()]
        print("\n最优d值分析：")
        print(f"d = {best_d['d']:.1f}")
        print(f"相关性 = {best_d['correlation']:.4f}")
        print(f"ADF p值 = {best_d['p_value']:.4f}")
        print(f"偏度 = {best_d['skewness']:.4f}")
        print(f"峰度 = {best_d['kurtosis']:.4f}")

    # 测试6：性能测试
    print("\n测试6：性能对比测试")
    import time

    # 创建测试数据
    test_data_sizes = [1000, 5000, 10000]

    print("一维数据性能对比:")
    for size in test_data_sizes:
        test_data = np.cumsum(np.random.randn(size))

        # 预热 numba JIT 编译
        _ = frac_diff_ffd(test_data[:100], diff_amt=0.5, thresh=0.01)

        # 测试原始版本
        start_time = time.time()
        result_original = _frac_diff_ffd_original(test_data, diff_amt=0.5, thresh=0.01)
        original_time = time.time() - start_time

        # 测试优化版本
        start_time = time.time()
        result_optimized = frac_diff_ffd(test_data, diff_amt=0.5, thresh=0.01)
        optimized_time = time.time() - start_time

        speedup = original_time / optimized_time if optimized_time > 0 else float("inf")
        print(
            f"数据大小 {size}: 原始版本 {original_time:.4f}秒, 优化版本 {optimized_time:.4f}秒, 加速比 {speedup:.1f}x"
        )

    # 测试多维数据的并行性能
    print("\n多维数据并行性能测试:")
    for cols in [5, 10, 20]:
        multi_test = np.random.randn(5000, cols).cumsum(axis=0)

        # 预热
        _ = frac_diff_ffd(multi_test[:100], diff_amt=0.5, thresh=0.01)

        # 测试原始版本
        start_time = time.time()
        result_original = _frac_diff_ffd_original(multi_test, diff_amt=0.5, thresh=0.01)
        original_time = time.time() - start_time

        # 测试优化版本
        start_time = time.time()
        result_optimized = frac_diff_ffd(multi_test, diff_amt=0.5, thresh=0.01)
        optimized_time = time.time() - start_time

        speedup = original_time / optimized_time if optimized_time > 0 else float("inf")
        print(
            f"5000x{cols} 多维数据: 原始版本 {original_time:.4f}秒, 优化版本 {optimized_time:.4f}秒, 加速比 {speedup:.1f}x"
        )

    # 验证结果一致性
    print("\n结果一致性验证:")
    test_data = np.cumsum(np.random.randn(1000))
    result1 = _frac_diff_ffd_original(test_data, diff_amt=0.5, thresh=0.01)
    result2 = frac_diff_ffd(test_data, diff_amt=0.5, thresh=0.01)

    # 忽略 NaN 值进行比较
    valid_idx = ~(np.isnan(result1) | np.isnan(result2))
    max_diff = np.max(np.abs(result1[valid_idx] - result2[valid_idx]))
    print(f"最大差异: {max_diff:.2e}")
    print(f"结果一致: {max_diff < 1e-10}")
