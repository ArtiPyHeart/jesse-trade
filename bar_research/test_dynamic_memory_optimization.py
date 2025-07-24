"""
测试动态内存优化功能

演示memory_efficient模式如何根据实际数据大小自动调整参数
"""

import numpy as np
from bar_research.symbolic_regression_deap_advanced import (
    AdvancedSymbolicRegressionDEAP,
)


def test_dynamic_optimization():
    """测试不同数据大小下的动态优化"""

    # 测试场景
    test_cases = [
        {
            "name": "小数据集",
            "n_samples": 1000,
            "n_features": 3,
            "population_size": 500,
        },
        {
            "name": "中等数据集",
            "n_samples": 10000,
            "n_features": 5,
            "population_size": 2000,
        },
        {
            "name": "大数据集",
            "n_samples": 50000,
            "n_features": 10,
            "population_size": 5000,
        },
    ]

    for case in test_cases:
        print(f"\n{'='*50}")
        print(f"测试: {case['name']}")
        print(f"数据大小: {case['n_samples']} × {case['n_features']}")
        print(f"种群大小: {case['population_size']}")
        print("=" * 50)

        # 创建测试数据
        X = np.random.randn(case["n_samples"], case["n_features"])
        feature_names = [f"f{i}" for i in range(case["n_features"])]

        # 创建模型（启用动态内存优化）
        model = AdvancedSymbolicRegressionDEAP(
            population_size=case["population_size"],
            generations=5,  # 少量代数用于快速测试
            memory_efficient=True,  # 启用动态优化
            verbose=True,
            random_state=42,
        )

        print("\n初始参数:")
        print(f"  缓存大小: {model.max_cache_size}")
        print(f"  批大小: {model.batch_size}")
        print(f"  进程数: {model.n_jobs}")

        # 注意：实际的动态调整发生在fit()调用时
        print("\n动态优化将在fit()时根据实际内存使用情况进行...")

        # 如果要实际运行，取消下面的注释
        # model.fit(
        #     X=X,
        #     feature_names=feature_names,
        #     NA_MAX_NUM=100,
        #     candles_path="data/btc_1m.npy"
        # )


def demonstrate_memory_estimation():
    """演示内存估算功能"""
    print("\n" + "=" * 60)
    print("内存估算演示")
    print("=" * 60)

    # 创建不同大小的数据
    sizes = [
        (5000, 5),  # 5K × 5
        (20000, 10),  # 20K × 10
        (100000, 20),  # 100K × 20
    ]

    for n_samples, n_features in sizes:
        X = np.random.randn(n_samples, n_features)
        data_size_mb = X.nbytes / 1024 / 1024

        print(f"\n数据规模: {n_samples} 样本 × {n_features} 特征")
        print(f"数据大小: {data_size_mb:.2f} MB")

        # 估算不同种群大小的内存需求
        for pop_size in [500, 2000, 5000]:
            # 简化的内存估算
            per_individual = data_size_mb * 2  # 粗略估计
            population_memory = pop_size * per_individual
            total_memory = data_size_mb + population_memory

            print(f"  种群 {pop_size}: 预计需要 {total_memory:.0f} MB")


if __name__ == "__main__":
    print("动态内存优化测试\n")

    # 测试动态优化
    test_dynamic_optimization()

    # 演示内存估算
    demonstrate_memory_estimation()

    print("\n\n提示:")
    print("- memory_efficient=True 时会自动根据数据大小调整参数")
    print("- 内存压力大时会自动减少进程数、缓存和批大小")
    print("- 内存充足时会自动提高性能相关参数")
    print("- 不再依赖硬编码的种群大小阈值！")
