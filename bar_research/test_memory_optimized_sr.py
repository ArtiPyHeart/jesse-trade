"""
测试内存优化后的符号回归

这个脚本测试优化后的AdvancedSymbolicRegressionDEAP类，
特别是在大种群规模下的内存使用情况。
"""

import numpy as np
import psutil
import time
from bar_research.symbolic_regression_deap_advanced import (
    AdvancedSymbolicRegressionDEAP,
)


def get_memory_usage():
    """获取当前进程的内存使用情况（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def test_large_population():
    """测试大种群规模"""
    print("=== 测试内存优化的符号回归 ===\n")

    # 创建测试数据
    print("准备测试数据...")
    n_samples = 5000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    NA_MAX_NUM = 100

    # 测试配置
    test_configs = [
        {
            "name": "小种群（基准）",
            "population_size": 500,
            "memory_efficient": False,
            "generations": 20,
        },
        {
            "name": "大种群（内存高效模式）",
            "population_size": 5000,
            "memory_efficient": True,
            "generations": 20,
        },
        {
            "name": "超大种群（极限测试）",
            "population_size": 10000,
            "memory_efficient": True,
            "generations": 10,
            "n_jobs": 1,  # 单进程模式
        },
    ]

    for config in test_configs:
        print(f"\n--- {config['name']} ---")
        print(f"种群大小: {config['population_size']}")
        print(f"内存高效模式: {config.get('memory_efficient', False)}")

        # 记录初始内存
        initial_memory = get_memory_usage()
        print(f"初始内存使用: {initial_memory:.2f} MB")

        try:
            # 创建模型
            model = AdvancedSymbolicRegressionDEAP(
                population_size=config["population_size"],
                generations=config["generations"],
                n_islands=4,
                elite_size=10,
                memory_efficient=config.get("memory_efficient", False),
                max_cache_size=500,  # 限制缓存大小
                batch_size=50,  # 小批量评估
                n_jobs=config.get("n_jobs", -1),
                verbose=True,
                random_state=42,
            )

            # 开始训练
            start_time = time.time()
            model.fit(
                X=X,
                feature_names=feature_names,
                NA_MAX_NUM=NA_MAX_NUM,
                stand_scale=True,
                candles_path="../data/btc_1m.npy",
            )

            # 记录结果
            end_time = time.time()
            final_memory = get_memory_usage()
            peak_memory = final_memory  # 简化，实际应该监控峰值

            print(f"\n训练完成！")
            print(f"训练时间: {end_time - start_time:.2f} 秒")
            print(f"最终内存使用: {final_memory:.2f} MB")
            print(f"内存增长: {final_memory - initial_memory:.2f} MB")

            # 获取最佳表达式
            best_expressions = model.get_best_expressions(n=3)
            print(f"\nTop 3 表达式:")
            for expr in best_expressions:
                print(f"  {expr['rank']}. {expr['expression'][:50]}...")
                print(
                    f"     峰度偏差: {expr['kurtosis_deviation']:.4f}, 复杂度: {expr['complexity']}"
                )

        except Exception as e:
            print(f"\n错误: {e}")
            print("如果内存不足，请尝试:")
            print("1. 减小 population_size")
            print("2. 启用 memory_efficient=True")
            print("3. 设置 n_jobs=1 使用单进程")
            print("4. 减小 max_cache_size 和 batch_size")

        # 强制垃圾回收
        import gc

        gc.collect()
        time.sleep(2)  # 等待内存释放


def test_memory_efficient_features():
    """测试内存高效特性"""
    print("\n=== 测试内存高效特性 ===\n")

    # 准备数据
    n_samples = 3000
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    feature_names = [f"f{i}" for i in range(n_features)]

    # 测试不同的内存优化配置
    print("1. 测试缓存管理...")
    model = AdvancedSymbolicRegressionDEAP(
        population_size=1000,
        generations=10,
        max_cache_size=100,  # 小缓存
        memory_efficient=True,
        verbose=False,
    )

    print("2. 测试批量评估...")
    model = AdvancedSymbolicRegressionDEAP(
        population_size=2000,
        generations=10,
        batch_size=20,  # 小批量
        memory_efficient=True,
        verbose=False,
    )

    print("3. 测试共享内存（多进程）...")
    model = AdvancedSymbolicRegressionDEAP(
        population_size=3000,
        generations=10,
        memory_efficient=True,
        n_jobs=2,
        verbose=False,
    )

    print("\n所有内存优化特性测试完成！")


if __name__ == "__main__":
    # 打印系统信息
    print(f"系统内存: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB")
    print(f"可用内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")
    print(f"CPU核心数: {psutil.cpu_count()}\n")

    # 运行测试
    test_large_population()
    # test_memory_efficient_features()  # 可选的额外测试

    print("\n=== 测试完成 ===")
