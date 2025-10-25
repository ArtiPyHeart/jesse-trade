#!/usr/bin/env python3
"""
Ripser 性能测试

测试不同规模点云的计算时间
"""

import time
import numpy as np
from pyrs_indicators.topology import ripser

def benchmark_ripser(n_points, max_dim, n_runs=3):
    """性能基准测试"""
    print(f"\n{'='*70}")
    print(f"Benchmark: {n_points} points, max_dim={max_dim}")
    print(f"{'='*70}")

    # 生成随机点云
    np.random.seed(42)
    points = np.random.randn(n_points, 2).astype(np.float64)

    times = []
    for run in range(n_runs):
        start = time.time()
        result = ripser(points, max_dim=max_dim, threshold=2.0)
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"  Run {run+1}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n  Average: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  Persistence dimensions: {len(result['persistence'])}")
    for dim in range(len(result['persistence'])):
        n_pairs = result['persistence'][dim].shape[0]
        print(f"    H_{dim}: {n_pairs} pairs")

    return avg_time

def main():
    print("Ripser Performance Benchmark")
    print("="*70)

    # 测试不同规模
    test_cases = [
        (10, 1, 5),   # 小规模
        (20, 1, 3),   # 中等规模
        (30, 1, 3),   # 较大规模
        (50, 1, 2),   # 大规模
        (10, 2, 3),   # 小规模 + 高维
        (20, 2, 2),   # 中等规模 + 高维
    ]

    results = []
    for n_points, max_dim, n_runs in test_cases:
        try:
            avg_time = benchmark_ripser(n_points, max_dim, n_runs)
            results.append((n_points, max_dim, avg_time))
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append((n_points, max_dim, None))

    # 总结
    print(f"\n{'='*70}")
    print("Performance Summary")
    print(f"{'='*70}")
    print(f"{'Points':<10} {'MaxDim':<10} {'Time (s)':<15}")
    print("-"*70)
    for n_points, max_dim, avg_time in results:
        if avg_time is not None:
            print(f"{n_points:<10} {max_dim:<10} {avg_time:<15.3f}")
        else:
            print(f"{n_points:<10} {max_dim:<10} {'FAILED':<15}")

    print(f"\n✓ Performance benchmark complete")

if __name__ == "__main__":
    main()
