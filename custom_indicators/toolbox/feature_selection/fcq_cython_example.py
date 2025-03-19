"""
Cython版FCQSelector使用示例
"""

import time
import numpy as np
import pandas as pd
from fcq_selector_cython import CythonFCQSelector
from custom_indicators.toolbox.feature_selction.fcq_selector import FCQSelector


def generate_test_data(n_samples=1000, n_features=100, random_state=42):
    """生成测试数据"""
    np.random.seed(random_state)

    # 生成特征矩阵
    X = np.random.randn(n_samples, n_features)

    # 生成目标变量 (只与前10个特征相关)
    y = X[:, :10].sum(axis=1) + np.random.randn(n_samples) * 0.5

    # 转换为DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


def compare_performance(n_samples=1000, n_features=100, random_state=42):
    """比较原始FCQSelector和Cython版FCQSelector的性能"""
    print(f"生成测试数据: {n_samples}样本, {n_features}特征...")
    X, y = generate_test_data(n_samples, n_features, random_state)

    # 测试原始版本
    print("\n测试原始FCQSelector:")
    selector_orig = FCQSelector(max_features=20, verbose=False)
    start_time = time.time()
    selected_orig = selector_orig.fit_transform(X, y)
    orig_time = time.time() - start_time
    print(f"原始版本运行时间: {orig_time:.4f}秒")
    print(f"选择的特征数量: {selected_orig.shape[1]}")

    # 测试Cython版本
    print("\n测试Cython版FCQSelector:")
    selector_cython = CythonFCQSelector(max_features=20, verbose=False)
    start_time = time.time()
    selected_cython = selector_cython.fit_transform(X, y)
    cython_time = time.time() - start_time
    print(f"Cython版本运行时间: {cython_time:.4f}秒")
    print(f"选择的特征数量: {selected_cython.shape[1]}")

    # 比较结果
    speedup = orig_time / cython_time
    print(f"\n速度提升: {speedup:.2f}x")

    # 检查结果一致性
    same_features = set(selected_orig.columns) == set(selected_cython.columns)
    print(f"特征选择结果一致: {same_features}")

    return {
        "original_time": orig_time,
        "cython_time": cython_time,
        "speedup": speedup,
        "same_features": same_features,
    }


def benchmark_different_sizes():
    """对不同数据规模进行基准测试"""
    print("对不同数据规模进行基准测试...\n")

    results = []

    # 测试不同样本量
    sample_sizes = [1000, 5000, 10000]
    for n_samples in sample_sizes:
        print(f"测试 {n_samples} 样本, 100 特征:")
        result = compare_performance(n_samples=n_samples, n_features=100)
        results.append({"n_samples": n_samples, "n_features": 100, **result})
        print("-" * 80)

    # 测试不同特征数量
    feature_sizes = [100, 500, 1000]
    for n_features in feature_sizes:
        print(f"测试 5000 样本, {n_features} 特征:")
        result = compare_performance(n_samples=5000, n_features=n_features)
        results.append({"n_samples": 5000, "n_features": n_features, **result})
        print("-" * 80)

    # 打印摘要
    print("\n基准测试摘要:")
    print(
        f"{'数据规模':>15} | {'原始版本(秒)':>14} | {'Cython版本(秒)':>14} | {'加速比':>10}"
    )
    print("-" * 60)

    for result in results:
        size = f"{result['n_samples']}x{result['n_features']}"
        print(
            f"{size:>15} | {result['original_time']:>14.4f} | {result['cython_time']:>14.4f} | {result['speedup']:>10.2f}x"
        )


if __name__ == "__main__":
    print("=" * 80)
    print("Cython版FCQSelector性能测试")
    print("=" * 80)

    # 运行单次性能对比
    print("\n运行单次性能对比...")
    compare_performance(n_samples=5000, n_features=200)

    # 是否运行全面基准测试
    run_benchmark = False
    if run_benchmark:
        print("\n\n" + "=" * 80)
        benchmark_different_sizes()
