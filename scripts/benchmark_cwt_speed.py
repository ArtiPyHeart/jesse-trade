"""
CWT (Continuous Wavelet Transform) 性能基准测试

用于评估 Rust CWT 实现的性能优化效果
"""

import time
import numpy as np

try:
    from pyrs_indicators.ind_wavelets import cwt
    from pyrs_indicators import HAS_RUST
except ImportError:
    HAS_RUST = False
    print("⚠️  pyrs_indicators not available")
    print("   Run: cd rust_indicators && cargo clean && maturin develop --release")
    exit(1)


def benchmark_cwt(signal_len: int, num_scales: int, num_runs: int = 5) -> dict:
    """运行 CWT 基准测试"""
    np.random.seed(42)

    # 生成测试信号
    t = np.linspace(0, 1, signal_len)
    signal = 100 + np.cumsum(np.random.randn(signal_len) * 0.5)

    # 生成 scales
    scales = np.logspace(np.log2(8), np.log2(128), num=num_scales, base=2)
    wavelet = 'cmor1.5-1.0'
    sampling_period = 0.5
    pad_width = int(max(scales))

    # 预热
    _ = cwt(signal, scales, wavelet=wavelet, sampling_period=sampling_period,
            precision=12, pad_width=pad_width, verbose=False)

    # 计时
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result, freqs = cwt(signal, scales, wavelet=wavelet, sampling_period=sampling_period,
                           precision=12, pad_width=pad_width, verbose=False)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'signal_len': signal_len,
        'num_scales': num_scales,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'output_shape': result.shape,
    }


def main():
    print("=" * 70)
    print("CWT 性能基准测试")
    print("=" * 70)

    # 测试配置
    test_configs = [
        (500, 32),    # 小规模
        (1000, 64),   # 中等规模
        (2000, 64),   # 较大规模
        (4000, 128),  # 大规模
    ]

    results = []
    for signal_len, num_scales in test_configs:
        print(f"\n测试: signal_len={signal_len}, num_scales={num_scales}")
        result = benchmark_cwt(signal_len, num_scales, num_runs=5)
        results.append(result)

        print(f"  平均时间: {result['mean_time']*1000:.2f} ms")
        print(f"  标准差:   {result['std_time']*1000:.2f} ms")
        print(f"  最小时间: {result['min_time']*1000:.2f} ms")
        print(f"  最大时间: {result['max_time']*1000:.2f} ms")
        print(f"  输出形状: {result['output_shape']}")

    # 汇总表格
    print("\n" + "=" * 70)
    print("性能汇总")
    print("=" * 70)
    print(f"{'配置':<20} {'平均时间 (ms)':<15} {'标准差 (ms)':<15}")
    print("-" * 50)
    for r in results:
        config = f"{r['signal_len']}x{r['num_scales']}"
        print(f"{config:<20} {r['mean_time']*1000:<15.2f} {r['std_time']*1000:<15.2f}")

    print("\n✅ 基准测试完成")
    return results


if __name__ == "__main__":
    main()
