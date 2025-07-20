"""
测试符号回归实现
比较基础版本和高级版本的性能
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from symbolic_regression_deap import SymbolicRegressionDEAP, prepare_features
from symbolic_regression_deap_advanced import AdvancedSymbolicRegressionDEAP


def test_basic_version():
    """测试基础版本"""
    print("=" * 50)
    print("测试基础DEAP符号回归实现")
    print("=" * 50)

    # 准备数据
    X, feature_names, NA_MAX_NUM = prepare_features(use_last_n=10000)
    print(f"数据准备完成: {X.shape[0]} 样本, {X.shape[1]} 特征")

    # 创建模型
    model = SymbolicRegressionDEAP(
        population_size=100,
        generations=20,
        tournament_size=5,
        crossover_prob=0.8,
        mutation_prob=0.15,
        max_depth=8,
        init_depth=(2, 5),
        parsimony_coefficient=0.005,
        elite_size=5,
        verbose=True,
        random_state=42,
    )

    # 训练
    start_time = time.time()
    model.fit(X, feature_names)
    training_time = time.time() - start_time

    # 评估
    results = model.evaluate_best_individual()

    print(f"\n训练时间: {training_time:.2f} 秒")
    print(f"最佳表达式: {results['expression'][:100]}...")
    print(f"适应度: {results['fitness']:.6f}")
    print(f"峰度: {results['kurtosis']:.4f}")
    print(f"Bar数量: {results['num_bars']}")
    print(f"树大小: {results['tree_size']}")

    return model, results


def test_advanced_version():
    """测试高级版本"""
    print("\n" + "=" * 50)
    print("测试高级DEAP符号回归实现（多目标优化）")
    print("=" * 50)

    # 准备数据
    X, feature_names, NA_MAX_NUM = prepare_features(use_last_n=10000)
    print(f"数据准备完成: {X.shape[0]} 样本, {X.shape[1]} 特征")

    # 创建模型
    model = AdvancedSymbolicRegressionDEAP(
        population_size=160,  # 总种群大小
        generations=20,
        n_islands=4,
        migration_rate=0.1,
        local_search_prob=0.05,
        adaptive_mutation=True,
        verbose=True,
        random_state=42,
    )

    # 训练
    start_time = time.time()
    model.fit(X, feature_names)
    training_time = time.time() - start_time

    # 获取最佳表达式
    best_expressions = model.get_best_expressions(n=3)

    print(f"\n训练时间: {training_time:.2f} 秒")
    print(f"Pareto前沿大小: {len(model.pareto_front)}")

    print("\n=== Top 3 解 ===")
    for expr in best_expressions:
        print(f"\n解 {expr['rank']}:")
        print(f"  峰度偏差: {expr['kurtosis_deviation']:.6f}")
        print(f"  复杂度: {expr['complexity']}")
        print(f"  实际峰度: {expr['actual_kurtosis']:.4f}")
        print(f"  Bar数量: {expr['num_bars']}")

    return model, best_expressions


def compare_bar_distributions(
    basic_model, advanced_model, X, candles_path="data/btc_1m.npy"
):
    """比较两个模型生成的bar分布"""
    # 加载数据
    candles = np.load(candles_path)
    candles = candles[candles[:, 5] > 0]
    if len(candles) > 10000:
        candles = candles[-10000:]

    from custom_indicators.toolbox.bar.build import build_bar_by_cumsum

    # 基础模型预测
    y_pred_basic = basic_model.predict(X)
    cumsum_threshold_basic = np.sum(np.abs(y_pred_basic)) / (len(candles) // 120)
    bars_basic = build_bar_by_cumsum(
        candles, np.abs(y_pred_basic), cumsum_threshold_basic
    )

    # 高级模型预测
    y_pred_advanced = advanced_model.predict(X)
    cumsum_threshold_advanced = np.sum(np.abs(y_pred_advanced)) / (len(candles) // 120)
    bars_advanced = build_bar_by_cumsum(
        candles, np.abs(y_pred_advanced), cumsum_threshold_advanced
    )

    # 计算收益率
    def calculate_returns(bars):
        close_arr = bars[:, 2]
        returns = np.log(close_arr[1:] / close_arr[:-1])
        return returns[~np.isnan(returns) & ~np.isinf(returns)]

    returns_basic = calculate_returns(bars_basic)
    returns_advanced = calculate_returns(bars_advanced)

    # 原始1分钟bar的收益率
    returns_original = calculate_returns(candles)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 收益率分布
    axes[0, 0].hist(
        returns_original, bins=50, alpha=0.7, density=True, label="Original 1m"
    )
    axes[0, 0].set_title("原始1分钟Bar收益率分布")
    axes[0, 0].set_xlabel("收益率")
    axes[0, 0].set_ylabel("密度")

    axes[0, 1].hist(
        returns_basic,
        bins=50,
        alpha=0.7,
        density=True,
        label="Basic Model",
        color="orange",
    )
    axes[0, 1].set_title("基础模型Bar收益率分布")
    axes[0, 1].set_xlabel("收益率")

    axes[0, 2].hist(
        returns_advanced,
        bins=50,
        alpha=0.7,
        density=True,
        label="Advanced Model",
        color="green",
    )
    axes[0, 2].set_title("高级模型Bar收益率分布")
    axes[0, 2].set_xlabel("收益率")

    # Q-Q图
    from scipy import stats

    stats.probplot(returns_original, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("原始1分钟Bar Q-Q图")

    stats.probplot(returns_basic, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("基础模型Bar Q-Q图")

    stats.probplot(returns_advanced, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title("高级模型Bar Q-Q图")

    plt.tight_layout()
    plt.savefig("bar_comparison.png", dpi=150)
    plt.show()

    # 打印统计信息
    print("\n=== 分布统计对比 ===")
    print(f"{'指标':<15} {'原始1m':<15} {'基础模型':<15} {'高级模型':<15}")
    print("-" * 60)

    metrics = [
        ("Bar数量", len(candles), len(bars_basic), len(bars_advanced)),
        (
            "峰度",
            stats.kurtosis(returns_original, fisher=False),
            stats.kurtosis(returns_basic, fisher=False),
            stats.kurtosis(returns_advanced, fisher=False),
        ),
        (
            "偏度",
            abs(stats.skew(returns_original)),
            abs(stats.skew(returns_basic)),
            abs(stats.skew(returns_advanced)),
        ),
        (
            "Shapiro-Wilk p值",
            (
                stats.shapiro(returns_original[:5000])[1]
                if len(returns_original) > 5000
                else stats.shapiro(returns_original)[1]
            ),
            (
                stats.shapiro(returns_basic[:5000])[1]
                if len(returns_basic) > 5000
                else stats.shapiro(returns_basic)[1]
            ),
            (
                stats.shapiro(returns_advanced[:5000])[1]
                if len(returns_advanced) > 5000
                else stats.shapiro(returns_advanced)[1]
            ),
        ),
    ]

    for name, val1, val2, val3 in metrics:
        print(f"{name:<15} {val1:<15.4f} {val2:<15.4f} {val3:<15.4f}")


if __name__ == "__main__":
    # 测试基础版本
    basic_model, basic_results = test_basic_version()

    # 测试高级版本
    advanced_model, advanced_results = test_advanced_version()

    # 准备数据用于比较
    X, _, _ = prepare_features(use_last_n=10000)

    # 比较结果
    compare_bar_distributions(basic_model, advanced_model, X[:-1])

    # 可视化Pareto前沿（仅高级版本）
    if hasattr(advanced_model, "visualize_pareto_front"):
        print("\n生成Pareto前沿可视化...")
        advanced_model.visualize_pareto_front(save_path="pareto_front_test.png")
