"""
RFCQSelector 性能基准测试

测量特征选择的速度，用于优化前后对比。

运行方式:
    python scripts/benchmark_rfcq_speed.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.feature_selection.rfcq_selector import RFCQSelector


def generate_test_data(n_samples: int = 5000, n_features: int = 100, seed: int = 42):
    """生成测试数据（与一致性测试相同）"""
    np.random.seed(seed)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(
        X["feat_0"] * 0.5
        + X["feat_1"] * 0.3
        + X["feat_2"] * 0.1
        + np.random.randn(n_samples) * 0.1
    )
    return X, y


def benchmark(n_runs: int = 3, max_features: int = 20):
    """运行基准测试"""
    print(f"RFCQSelector 性能基准测试")
    print(f"=" * 50)

    # 生成数据
    print("生成测试数据...")
    X, y = generate_test_data()
    print(f"数据规模: {X.shape[0]} 样本 x {X.shape[1]} 特征")
    print(f"目标特征数: {max_features}")
    print()

    # 预热（编译 numba 函数）
    print("预热 numba 函数...")
    selector = RFCQSelector(max_features=5, random_state=42, verbose=False)
    selector.fit(X, y)
    print()

    # 基准测试
    times = []
    for i in range(n_runs):
        print(f"运行 {i + 1}/{n_runs}...")
        start = time.perf_counter()
        selector = RFCQSelector(max_features=max_features, random_state=42, verbose=False)
        selector.fit(X, y)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  耗时: {elapsed:.2f}s")

    # 统计结果
    avg_time = np.mean(times)
    std_time = np.std(times)
    time_per_feature = avg_time / max_features

    print()
    print(f"结果统计:")
    print(f"-" * 50)
    print(f"平均耗时: {avg_time:.2f}s ± {std_time:.2f}s")
    print(f"每特征耗时: {time_per_feature:.3f}s")
    print(f"特征/秒: {1 / time_per_feature:.2f}")

    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "time_per_feature": time_per_feature,
        "features_per_second": 1 / time_per_feature,
    }


def benchmark_large(n_runs: int = 2, max_features: int = 100):
    """大规模基准测试"""
    print(f"RFCQSelector 大规模性能基准测试")
    print(f"=" * 50)

    # 生成大规模数据
    print("生成大规模测试数据...")
    np.random.seed(42)
    n_samples = 10000
    n_features = 500
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(
        X["feat_0"] * 0.5
        + X["feat_1"] * 0.3
        + X["feat_2"] * 0.1
        + np.random.randn(n_samples) * 0.1
    )
    print(f"数据规模: {X.shape[0]} 样本 x {X.shape[1]} 特征")
    print(f"目标特征数: {max_features}")
    print()

    # 基准测试
    times = []
    for i in range(n_runs):
        print(f"运行 {i + 1}/{n_runs}...")
        start = time.perf_counter()
        selector = RFCQSelector(max_features=max_features, random_state=42, verbose=False)
        selector.fit(X, y)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  耗时: {elapsed:.2f}s")

    # 统计结果
    avg_time = np.mean(times)
    time_per_feature = avg_time / max_features

    print()
    print(f"结果统计:")
    print(f"-" * 50)
    print(f"平均耗时: {avg_time:.2f}s")
    print(f"每特征耗时: {time_per_feature:.3f}s")
    print(f"特征/秒: {1 / time_per_feature:.2f}")


def benchmark_detailed():
    """详细计时分析，找出性能瓶颈"""
    print("\n" + "=" * 60)
    print("详细计时分析")
    print("=" * 60)

    # 生成数据
    X, y = generate_test_data(n_samples=5000, n_features=100)
    print(f"数据规模: {X.shape[0]} 样本 x {X.shape[1]} 特征")

    # 分步计时
    selector = RFCQSelector(max_features=20, random_state=42, verbose=False)

    # 1. 计算 relevance（LightGBM 原生 CV）
    start = time.perf_counter()
    selector.variables_ = selector._find_numerical_variables(X)
    X_numeric = X[selector.variables_]
    selector.relevance_ = selector._calculate_relevance(X_numeric, y)
    relevance_time = time.perf_counter() - start

    # 2. MRMR 循环
    start = time.perf_counter()
    X_data = np.asarray(X_numeric.values, dtype=np.float32, order="C")
    n_features = len(selector.variables_)
    mask = np.ones(n_features, dtype=bool)
    relevance = selector.relevance_.copy()

    first_idx = np.argmax(relevance)
    selected_indices = [first_idx]
    mask[first_idx] = False

    from src.features.feature_selection.rfcq_selector import fast_corrwith_numba

    X_remaining = X_data[:, mask]
    y_values = X_data[:, first_idx]
    initial_redundance = fast_corrwith_numba(X_remaining, y_values)

    running_mean = np.zeros(n_features, dtype=np.float64)
    running_mean[mask] = initial_redundance
    redundance_count = 1
    eps = 1e-10
    n_to_select = 19  # 20 - 1

    for _ in range(n_to_select):
        if not mask.any():
            break
        safe_redundance = np.maximum(running_mean, eps)
        mrmr_scores = np.where(mask, relevance / safe_redundance, -np.inf)
        best_idx = np.argmax(mrmr_scores)
        selected_indices.append(best_idx)
        mask[best_idx] = False
        if not mask.any():
            break
        X_remaining = X_data[:, mask]
        y_values = X_data[:, best_idx]
        new_redundance = fast_corrwith_numba(X_remaining, y_values)
        redundance_count += 1
        running_mean[mask] += (new_redundance - running_mean[mask]) / redundance_count

    mrmr_time = time.perf_counter() - start

    print(f"\n时间分解:")
    print(f"-" * 50)
    print(f"LightGBM CV + RF训练: {relevance_time:.2f}s ({relevance_time/(relevance_time+mrmr_time)*100:.1f}%)")
    print(f"MRMR循环 (选19个特征): {mrmr_time:.2f}s ({mrmr_time/(relevance_time+mrmr_time)*100:.1f}%)")
    print(f"总计: {relevance_time + mrmr_time:.2f}s")


if __name__ == "__main__":
    benchmark_detailed()

    print("\n" + "=" * 60)
    print("小规模测试 (100特征, 选20个)")
    print("=" * 60)
    benchmark(n_runs=3, max_features=20)
