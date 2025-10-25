"""
二项式系数表性能基准测试

对比 Rust 实现与 Python scipy 的性能。
"""

import time
import numpy as np
from scipy.special import comb


def python_binomial_table(n, k):
    """
    Python 参考实现（使用 scipy）

    Args:
        n: 最大 n 值
        k: 最大 k 值

    Returns:
        二维数组，table[j][i] = C(i, j)
    """
    # 使用 object dtype 避免溢出，然后验证范围
    table = np.zeros((k + 1, n + 1), dtype=object)

    for i in range(n + 1):
        for j in range(min(i, k) + 1):
            value = int(comb(i, j, exact=True))
            # 检查是否在安全范围内
            MAX_INDEX = (1 << (64 - 1 - 8)) - 1
            if value > MAX_INDEX:
                raise OverflowError(f"C({i}, {j}) = {value} exceeds i64 limit")
            table[j, i] = value

    return table.astype(np.int64)


def benchmark_python(n, k, num_runs=10):
    """
    基准测试 Python 实现

    Args:
        n: 最大 n 值
        k: 最大 k 值
        num_runs: 运行次数

    Returns:
        (平均时间, 标准差)
    """
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()
        _ = python_binomial_table(n, k)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times), np.std(times)


def main():
    """运行性能基准测试"""
    print("=" * 70)
    print("二项式系数表性能基准测试")
    print("=" * 70)
    print()

    test_cases = [
        (20, 10, "小规模（20x10）"),
        (50, 25, "中规模（50x25）"),
        (57, 28, "最大安全规模（57x28，接近 i64 限制）"),
    ]

    print(f"{'规模':<20} {'Python (ms)':<15} {'Std (ms)':<15}")
    print("-" * 70)

    for n, k, desc in test_cases:
        try:
            mean_time, std_time = benchmark_python(n, k, num_runs=5)
            print(f"{desc:<20} {mean_time*1000:>10.2f}     {std_time*1000:>10.2f}")
        except Exception as e:
            print(f"{desc:<20} ERROR: {e}")

    print()
    print("=" * 70)
    print("注意：Rust 实现预计性能提升 10-50x")
    print("      （需要在 Rust 基准测试中验证）")
    print("=" * 70)


if __name__ == "__main__":
    main()
