#!/usr/bin/env python3
"""
VMD 性能基准测试

测试全局 FFT 缓存优化效果：
- 首次调用（冷启动）vs 后续调用（缓存命中）
- 模拟实际使用场景（多窗口顺序调用）
"""

import time
import numpy as np
from pyrs_indicators.ind_decomposition import vmd

# VMD 参数
ALPHA = 2000
TAU = 0.0
K = 5
DC = False
INIT = 1
TOL = 1e-7


def generate_price_signal(n: int) -> np.ndarray:
    """生成模拟价格信号"""
    t = np.linspace(0, 1, n)
    trend = 100 + 20 * t
    cycle = 5 * np.sin(2 * np.pi * 10 * t)
    noise = np.random.randn(n) * 0.5
    return trend + cycle + noise


def benchmark_single_call(signal: np.ndarray, name: str, warmup: bool = True) -> float:
    """测试单次 VMD 调用"""
    if warmup:
        # 预热（触发全局缓存）
        _ = vmd(signal, alpha=ALPHA, tau=TAU, K=K, DC=DC, init=INIT, tol=TOL)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = vmd(signal, alpha=ALPHA, tau=TAU, K=K, DC=DC, init=INIT, tol=TOL)
        times.append(time.perf_counter() - t0)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"  {name}: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    return avg_time


def benchmark_sequential_calls(signals: list, name: str) -> float:
    """测试顺序调用多个窗口（模拟特征计算）"""
    # 预热第一个
    _ = vmd(signals[0], alpha=ALPHA, tau=TAU, K=K, DC=DC, init=INIT, tol=TOL)

    t0 = time.perf_counter()
    results = []
    for sig in signals:
        u = vmd(sig, alpha=ALPHA, tau=TAU, K=K, DC=DC, init=INIT, tol=TOL)
        results.append(u)
    total_time = time.perf_counter() - t0

    n_calls = len(signals)
    per_call = total_time / n_calls
    print(f"  {name}:")
    print(f"    总耗时: {total_time:.2f}s")
    print(f"    调用次数: {n_calls}")
    print(f"    每次调用: {per_call*1000:.2f}ms")
    return total_time


def main():
    np.random.seed(42)

    print("=" * 60)
    print("VMD 性能基准测试（全局 FFT 缓存优化）")
    print("=" * 60)
    print()

    # 测试1：不同信号长度的单次调用
    print("【测试1】不同信号长度的单次 VMD 调用")
    print("-" * 40)
    for n in [64, 128, 256, 512]:
        signal = generate_price_signal(n)
        benchmark_single_call(signal, f"N={n}")
    print()

    # 测试2：首次调用 vs 后续调用（验证缓存效果）
    print("【测试2】冷启动 vs 缓存命中（N=512）")
    print("-" * 40)
    signal_512 = generate_price_signal(512)

    # 冷启动（使用新的信号长度触发新的 FFT Plan 创建）
    # 注意：由于全局缓存，只有第一次使用新长度时才会创建
    signal_new = generate_price_signal(513)  # 新长度
    t0 = time.perf_counter()
    _ = vmd(signal_new, alpha=ALPHA, tau=TAU, K=K, DC=DC, init=INIT, tol=TOL)
    cold_time = time.perf_counter() - t0
    print(f"  冷启动 (N=513): {cold_time*1000:.2f}ms")

    # 缓存命中
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = vmd(signal_new, alpha=ALPHA, tau=TAU, K=K, DC=DC, init=INIT, tol=TOL)
        times.append(time.perf_counter() - t0)
    warm_time = np.mean(times)
    print(f"  缓存命中 (N=513): {warm_time*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")
    print(f"  加速比: {cold_time/warm_time:.2f}x")
    print()

    # 测试3：模拟特征计算场景
    print("【测试3】模拟特征计算场景")
    print("-" * 40)

    # 场景：1000 根 K 线，window=512，产生 489 个窗口
    n_candles = 1000
    window = 512
    full_signal = generate_price_signal(n_candles)
    windows = [full_signal[i : i + window] for i in range(n_candles - window + 1)]
    print(f"  K线数量: {n_candles}, 窗口大小: {window}")
    print(f"  窗口数量: {len(windows)}")
    benchmark_sequential_calls(windows, f"顺序处理 {len(windows)} 个窗口")
    print()

    # 测试4：不同窗口大小
    print("【测试4】不同窗口大小（各100个窗口）")
    print("-" * 40)
    for window_size in [32, 64, 128, 256, 512]:
        windows = [generate_price_signal(window_size) for _ in range(100)]
        benchmark_sequential_calls(windows, f"Window={window_size}")
    print()

    print("=" * 60)
    print("基准测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
