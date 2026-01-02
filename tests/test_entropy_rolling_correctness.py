"""Entropy Rolling 正确性测试

验证 Rust Rayon 并行滑动窗口计算的正确性。
"""

import numpy as np
import pytest
import time

from pyrs_indicators.util_entropy import (
    approximate_entropy,
    approximate_entropy_rolling,
    sample_entropy,
    sample_entropy_rolling,
)


# ============================================================================
# 测试数据生成
# ============================================================================


def generate_test_data(n: int, seed: int = 42) -> np.ndarray:
    """生成测试数据序列"""
    np.random.seed(seed)
    return np.random.randn(n).astype(np.float64)


def python_approximate_entropy_rolling(
    data: np.ndarray, period: int
) -> np.ndarray:
    """Python 参考实现（使用 Rust 单点函数）"""
    n = len(data)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        result[i] = approximate_entropy(window, m=2, r_ratio=0.3, mode="range")

    return result


def python_sample_entropy_rolling(
    data: np.ndarray, period: int
) -> np.ndarray:
    """Python 参考实现（使用 Rust 单点函数）"""
    n = len(data)
    result = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = data[i - period + 1 : i + 1]
        result[i] = sample_entropy(window, m=2, r_ratio=0.3, mode="range")

    return result


# ============================================================================
# 正确性测试
# ============================================================================


class TestApproximateEntropyRolling:
    """ApEn Rolling 正确性测试"""

    def test_shape_and_nan_pattern(self):
        """测试输出形状和 NaN 模式"""
        data = generate_test_data(200)
        period = 32

        result = approximate_entropy_rolling(data, period)

        assert result.shape == (200,), f"Expected (200,), got {result.shape}"
        assert np.all(np.isnan(result[: period - 1])), "前 period-1 个应为 NaN"
        assert np.all(~np.isnan(result[period - 1 :])), "后面应为有效值"

    def test_matches_python_reference(self):
        """验证与 Python 参考实现一致"""
        data = generate_test_data(100)
        period = 16

        rust_result = approximate_entropy_rolling(data, period)
        python_result = python_approximate_entropy_rolling(data, period)

        # 比较非 NaN 部分
        valid_mask = ~np.isnan(rust_result)
        np.testing.assert_allclose(
            rust_result[valid_mask],
            python_result[valid_mask],
            rtol=1e-10,
            err_msg="ApEn rolling 结果不一致",
        )

    def test_edge_case_short_sequence(self):
        """边界情况：序列长度 < period"""
        data = generate_test_data(20)
        period = 32

        result = approximate_entropy_rolling(data, period)

        assert result.shape == (20,)
        assert np.all(np.isnan(result)), "短序列应全为 NaN"

    def test_edge_case_minimum_length(self):
        """边界情况：序列长度刚好为 period"""
        data = generate_test_data(32)
        period = 32

        result = approximate_entropy_rolling(data, period)

        assert result.shape == (32,)
        assert np.sum(np.isnan(result)) == 31
        assert np.sum(~np.isnan(result)) == 1


class TestSampleEntropyRolling:
    """SampEn Rolling 正确性测试"""

    def test_shape_and_nan_pattern(self):
        """测试输出形状和 NaN 模式"""
        data = generate_test_data(200)
        period = 32

        result = sample_entropy_rolling(data, period)

        assert result.shape == (200,), f"Expected (200,), got {result.shape}"
        assert np.all(np.isnan(result[: period - 1])), "前 period-1 个应为 NaN"

    def test_matches_python_reference(self):
        """验证与 Python 参考实现一致"""
        data = generate_test_data(100)
        period = 16

        rust_result = sample_entropy_rolling(data, period)
        python_result = python_sample_entropy_rolling(data, period)

        # 比较非 NaN 部分
        valid_mask = ~np.isnan(rust_result) & ~np.isnan(python_result)
        np.testing.assert_allclose(
            rust_result[valid_mask],
            python_result[valid_mask],
            rtol=1e-10,
            err_msg="SampEn rolling 结果不一致",
        )

    def test_edge_case_short_sequence(self):
        """边界情况：序列长度 < period"""
        data = generate_test_data(20)
        period = 32

        result = sample_entropy_rolling(data, period)

        assert result.shape == (20,)
        assert np.all(np.isnan(result)), "短序列应全为 NaN"


# ============================================================================
# 性能测试
# ============================================================================


class TestPerformance:
    """性能对比测试"""

    @pytest.mark.slow
    def test_rust_rolling_vs_python_loop(self):
        """验证 Rust rolling 比 Python 循环快"""
        data = generate_test_data(500)
        period = 32
        n_runs = 3

        # Rust 计时
        rust_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = approximate_entropy_rolling(data, period)
            rust_times.append(time.perf_counter() - start)
        rust_avg = np.mean(rust_times)

        # Python 参考实现计时
        python_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = python_approximate_entropy_rolling(data, period)
            python_times.append(time.perf_counter() - start)
        python_avg = np.mean(python_times)

        speedup = python_avg / rust_avg
        print(f"\n性能对比 (N={len(data)}, period={period}):")
        print(f"  Python loop: {python_avg*1000:.2f}ms")
        print(f"  Rust rolling: {rust_avg*1000:.2f}ms")
        print(f"  Speedup: {speedup:.1f}x")

        # Rust rolling 应该比 Python 循环快
        assert speedup >= 1.0, f"Rust 应该不比 Python 慢"


# ============================================================================
# 通用性测试
# ============================================================================


class TestGenerality:
    """验证 rolling 函数可用于任意数据"""

    def test_with_random_data(self):
        """测试随机数据"""
        np.random.seed(123)
        data = np.random.randn(100)
        result = approximate_entropy_rolling(data, period=20)
        assert not np.all(np.isnan(result))

    def test_with_sine_wave(self):
        """测试正弦波数据"""
        data = np.sin(np.linspace(0, 8 * np.pi, 200))
        result = approximate_entropy_rolling(data, period=25)
        assert not np.all(np.isnan(result))
        # 周期信号应有较低的熵值
        valid = result[~np.isnan(result)]
        assert np.mean(valid) < 1.0  # 周期信号熵值较低

    def test_with_log_returns(self):
        """测试 log returns 数据（典型用例）"""
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200)) + 100
        log_ret = np.log(prices[1:] / prices[:-1])
        result = approximate_entropy_rolling(log_ret, period=31)
        assert result.shape == (199,)
        assert not np.all(np.isnan(result))


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Entropy Rolling 正确性测试")
    print("=" * 60)

    # 基础测试
    print("\n1. ApEn Rolling 形状测试...")
    test = TestApproximateEntropyRolling()
    test.test_shape_and_nan_pattern()
    print("   ✓ 通过")

    print("\n2. ApEn Rolling 正确性测试...")
    test.test_matches_python_reference()
    print("   ✓ 通过")

    print("\n3. SampEn Rolling 形状测试...")
    test = TestSampleEntropyRolling()
    test.test_shape_and_nan_pattern()
    print("   ✓ 通过")

    print("\n4. SampEn Rolling 正确性测试...")
    test.test_matches_python_reference()
    print("   ✓ 通过")

    # 通用性测试
    print("\n5. 通用性测试...")
    test = TestGenerality()
    test.test_with_random_data()
    test.test_with_sine_wave()
    test.test_with_log_returns()
    print("   ✓ 通过")

    # 性能测试
    print("\n6. 性能对比测试...")
    perf_test = TestPerformance()
    perf_test.test_rust_rolling_vs_python_loop()
    print("   ✓ 通过")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
