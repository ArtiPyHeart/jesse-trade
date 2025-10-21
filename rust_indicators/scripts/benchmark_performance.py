#!/usr/bin/env python3
"""
性能基准测试：对比 Python 和 Rust 实现的性能

测试内容：
1. NRBO 性能测试
2. VMD 性能测试
3. 不同信号长度下的性能对比
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入 Python 实现
from src.indicators.prod.emd.vmdpy import VMD
from src.indicators.prod.emd.nrbo import nrbo as nrbo_python

# 导入 Rust 实现
import _rust_indicators


def benchmark_nrbo(signal_lengths=[100, 500, 1000, 5000], num_runs=10):
    """
    NRBO 性能基准测试
    """
    print("=" * 80)
    print("NRBO 性能基准测试")
    print("=" * 80)

    results = []

    for n in signal_lengths:
        print(f"\n信号长度: {n}")

        # 生成测试信号
        t = np.linspace(0, 10, n)
        imf = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

        # Python 版本
        print("  测试 Python 版本...", end=" ")
        times_py = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = nrbo_python(imf, max_iter=10, tol=1e-6)
            times_py.append(time.perf_counter() - start)

        avg_py = np.mean(times_py)
        std_py = np.std(times_py)
        print(f"{avg_py*1000:.2f}±{std_py*1000:.2f} ms")

        # Rust 版本
        print("  测试 Rust 版本...", end=" ")
        times_rs = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = _rust_indicators.nrbo_py(imf, max_iter=10, tol=1e-6)
            times_rs.append(time.perf_counter() - start)

        avg_rs = np.mean(times_rs)
        std_rs = np.std(times_rs)
        print(f"{avg_rs*1000:.2f}±{std_rs*1000:.2f} ms")

        speedup = avg_py / avg_rs
        print(f"  加速比: {speedup:.2f}x")

        results.append({
            'signal_length': n,
            'python_time_ms': avg_py * 1000,
            'python_std_ms': std_py * 1000,
            'rust_time_ms': avg_rs * 1000,
            'rust_std_ms': std_rs * 1000,
            'speedup': speedup
        })

    return pd.DataFrame(results)


def benchmark_vmd(signal_lengths=[100, 500, 1000], K_values=[2, 3, 5], num_runs=5):
    """
    VMD 性能基准测试
    """
    print("\n" + "=" * 80)
    print("VMD 性能基准测试")
    print("=" * 80)

    results = []

    for n in signal_lengths:
        for K in K_values:
            print(f"\n信号长度: {n}, K={K}")

            # 生成测试信号
            t = np.linspace(0, 1, n)
            f = np.sin(2 * np.pi * 5 * t)

            # Python 版本
            print("  测试 Python 版本...", end=" ")
            times_py = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = VMD(f, alpha=2000, tau=0.0, K=K, DC=False, init=1, tol=1e-7)
                times_py.append(time.perf_counter() - start)

            avg_py = np.mean(times_py)
            std_py = np.std(times_py)
            print(f"{avg_py*1000:.2f}±{std_py*1000:.2f} ms")

            # Rust 版本
            print("  测试 Rust 版本...", end=" ")
            times_rs = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = _rust_indicators.vmd_py(f, alpha=2000, tau=0.0, k=K, dc=False, init=1, tol=1e-7)
                times_rs.append(time.perf_counter() - start)

            avg_rs = np.mean(times_rs)
            std_rs = np.std(times_rs)
            print(f"{avg_rs*1000:.2f}±{std_rs*1000:.2f} ms")

            speedup = avg_py / avg_rs
            print(f"  加速比: {speedup:.2f}x")

            results.append({
                'signal_length': n,
                'K': K,
                'python_time_ms': avg_py * 1000,
                'python_std_ms': std_py * 1000,
                'rust_time_ms': avg_rs * 1000,
                'rust_std_ms': std_rs * 1000,
                'speedup': speedup
            })

    return pd.DataFrame(results)


def print_summary(nrbo_df, vmd_df):
    """
    打印性能测试总结
    """
    print("\n" + "=" * 80)
    print("性能测试总结")
    print("=" * 80)

    print("\n📊 NRBO 性能汇总:")
    print(nrbo_df.to_string(index=False))
    print(f"\n平均加速比: {nrbo_df['speedup'].mean():.2f}x")
    print(f"最大加速比: {nrbo_df['speedup'].max():.2f}x")
    print(f"最小加速比: {nrbo_df['speedup'].min():.2f}x")

    print("\n📊 VMD 性能汇总:")
    print(vmd_df.to_string(index=False))
    print(f"\n平均加速比: {vmd_df['speedup'].mean():.2f}x")
    print(f"最大加速比: {vmd_df['speedup'].max():.2f}x")
    print(f"最小加速比: {vmd_df['speedup'].min():.2f}x")

    # 保存结果
    output_dir = Path(__file__).parent.parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)

    nrbo_df.to_csv(output_dir / "nrbo_benchmark.csv", index=False)
    vmd_df.to_csv(output_dir / "vmd_benchmark.csv", index=False)

    print(f"\n结果已保存到: {output_dir}")


def main():
    """主函数"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║           Rust Indicators 性能基准测试                          ║
║                                                                ║
║  对比 Python (NumPy/Numba) vs Rust 实现的性能                  ║
╚════════════════════════════════════════════════════════════════╝
    """)

    # NRBO 基准测试
    nrbo_results = benchmark_nrbo(
        signal_lengths=[100, 500, 1000, 5000],
        num_runs=10
    )

    # VMD 基准测试
    vmd_results = benchmark_vmd(
        signal_lengths=[100, 500, 1000],
        K_values=[2, 3, 5],
        num_runs=5
    )

    # 打印总结
    print_summary(nrbo_results, vmd_results)

    print("\n✅ 性能基准测试完成！")


if __name__ == "__main__":
    main()
