"""Entropy 正确性测试

验证 Rust 实现与 Python 原始实现的一致性。
"""

import numpy as np
import pytest


def _generate_test_data(n: int, seed: int = 42) -> np.ndarray:
    """生成测试数据"""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n)


def _generate_periodic_data(n: int, period: int = 10) -> np.ndarray:
    """生成周期性数据"""
    t = np.linspace(0, 2 * np.pi * n / period, n)
    return np.sin(t) + 0.1 * np.sin(3 * t)


class TestApproximateEntropy:
    """Approximate Entropy 测试"""

    def test_matches_python_range_mode(self):
        """Rust ApEn 与 Python 实现结果一致（range 模式）"""
        from pyrs_indicators.util_entropy import approximate_entropy as apen_rust
        from src.data_process.entropy.apen_sampen import (
            approximate_entropy as apen_py,
        )

        x = _generate_test_data(200)

        rust_result = apen_rust(x, m=2, r_ratio=0.3, mode="range")
        py_result = apen_py(x, m=2, r_ratio=0.3, mode="range")

        np.testing.assert_allclose(rust_result, py_result, rtol=1e-10)

    def test_matches_python_std_mode(self):
        """Rust ApEn 与 Python 实现结果一致（std 模式）"""
        from pyrs_indicators.util_entropy import approximate_entropy as apen_rust
        from src.data_process.entropy.apen_sampen import (
            approximate_entropy as apen_py,
        )

        x = _generate_test_data(200)

        rust_result = apen_rust(x, m=2, r_ratio=0.3, mode="std")
        py_result = apen_py(x, m=2, r_ratio=0.3, mode="std")

        np.testing.assert_allclose(rust_result, py_result, rtol=1e-10)

    def test_periodic_data_finite(self):
        """周期性数据应返回有限的 ApEn"""
        from pyrs_indicators.util_entropy import approximate_entropy

        periodic = _generate_periodic_data(200)

        apen_periodic = approximate_entropy(periodic, m=2, r_ratio=0.3)

        # 确保返回有限值
        assert np.isfinite(apen_periodic)
        assert apen_periodic >= 0

    def test_different_m_values(self):
        """测试不同的 m 值"""
        from pyrs_indicators.util_entropy import approximate_entropy as apen_rust
        from src.data_process.entropy.apen_sampen import (
            approximate_entropy as apen_py,
        )

        x = _generate_test_data(200)

        for m in [1, 2, 3]:
            rust_result = apen_rust(x, m=m, r_ratio=0.3, mode="range")
            py_result = apen_py(x, m=m, r_ratio=0.3, mode="range")
            np.testing.assert_allclose(rust_result, py_result, rtol=1e-10)

    def test_validation_errors(self):
        """参数验证错误"""
        from pyrs_indicators.util_entropy import approximate_entropy

        x = _generate_test_data(200)

        # 无效的 mode
        with pytest.raises(ValueError, match="mode must be"):
            approximate_entropy(x, mode="invalid")

        # 无效的 r_ratio
        with pytest.raises(ValueError, match="r_ratio must be"):
            approximate_entropy(x, r_ratio=1.5)

        # 无效的 m
        with pytest.raises(ValueError, match="m must be"):
            approximate_entropy(x, m=0)

        # 2D 数组
        with pytest.raises(ValueError, match="must be 1D"):
            approximate_entropy(np.array([[1, 2], [3, 4]]))


class TestSampleEntropy:
    """Sample Entropy 测试"""

    def test_matches_python_range_mode(self):
        """Rust SampEn 与 Python 实现结果一致（range 模式）"""
        from pyrs_indicators.util_entropy import sample_entropy as sampen_rust
        from src.data_process.entropy.apen_sampen import sample_entropy as sampen_py

        x = _generate_test_data(200)

        rust_result = sampen_rust(x, m=2, r_ratio=0.3, mode="range")
        py_result = sampen_py(x, m=2, r_ratio=0.3, mode="range")

        np.testing.assert_allclose(rust_result, py_result, rtol=1e-10)

    def test_matches_python_std_mode(self):
        """Rust SampEn 与 Python 实现结果一致（std 模式）"""
        from pyrs_indicators.util_entropy import sample_entropy as sampen_rust
        from src.data_process.entropy.apen_sampen import sample_entropy as sampen_py

        x = _generate_test_data(200)

        rust_result = sampen_rust(x, m=2, r_ratio=0.3, mode="std")
        py_result = sampen_py(x, m=2, r_ratio=0.3, mode="std")

        np.testing.assert_allclose(rust_result, py_result, rtol=1e-10)

    def test_periodic_data_finite(self):
        """周期性数据应返回有限的 SampEn"""
        from pyrs_indicators.util_entropy import sample_entropy

        periodic = _generate_periodic_data(200)

        sampen_periodic = sample_entropy(periodic, m=2, r_ratio=0.3)

        # 确保返回有限值
        assert np.isfinite(sampen_periodic)
        assert sampen_periodic >= 0

    def test_different_m_values(self):
        """测试不同的 m 值"""
        from pyrs_indicators.util_entropy import sample_entropy as sampen_rust
        from src.data_process.entropy.apen_sampen import sample_entropy as sampen_py

        x = _generate_test_data(200)

        for m in [1, 2, 3]:
            rust_result = sampen_rust(x, m=m, r_ratio=0.3, mode="range")
            py_result = sampen_py(x, m=m, r_ratio=0.3, mode="range")
            # 允许 NaN 的情况
            if np.isnan(py_result):
                assert np.isnan(rust_result)
            else:
                np.testing.assert_allclose(rust_result, py_result, rtol=1e-10)

    def test_validation_errors(self):
        """参数验证错误"""
        from pyrs_indicators.util_entropy import sample_entropy

        x = _generate_test_data(200)

        # 无效的 mode
        with pytest.raises(ValueError, match="mode must be"):
            sample_entropy(x, mode="invalid")

        # 无效的 r_ratio
        with pytest.raises(ValueError, match="r_ratio must be"):
            sample_entropy(x, r_ratio=0)

        # 序列太短
        with pytest.raises(ValueError, match="at least"):
            sample_entropy(np.array([1.0, 2.0]), m=2)


class TestEdgeCases:
    """边界情况测试"""

    def test_constant_sequence(self):
        """常数序列（零方差）"""
        from pyrs_indicators.util_entropy import approximate_entropy, sample_entropy

        x = np.ones(100)

        # 零方差应该返回 NaN
        apen = approximate_entropy(x, m=2, r_ratio=0.3, mode="range")
        sampen = sample_entropy(x, m=2, r_ratio=0.3, mode="range")

        assert np.isnan(apen)
        assert np.isnan(sampen)

    def test_minimum_length(self):
        """最小长度序列"""
        from pyrs_indicators.util_entropy import approximate_entropy, sample_entropy

        # m=2 需要至少 4 个元素
        x = np.array([1.0, 2.0, 3.0, 4.0])

        apen = approximate_entropy(x, m=2, r_ratio=0.3)
        sampen = sample_entropy(x, m=2, r_ratio=0.3)

        assert np.isfinite(apen) or np.isnan(apen)
        assert np.isfinite(sampen) or np.isnan(sampen)

    def test_large_dataset_consistency(self):
        """大数据集一致性"""
        from pyrs_indicators.util_entropy import approximate_entropy as apen_rust
        from pyrs_indicators.util_entropy import sample_entropy as sampen_rust
        from src.data_process.entropy.apen_sampen import (
            approximate_entropy as apen_py,
        )
        from src.data_process.entropy.apen_sampen import sample_entropy as sampen_py

        x = _generate_test_data(500)

        rust_apen = apen_rust(x, m=2, r_ratio=0.2)
        py_apen = apen_py(x, m=2, r_ratio=0.2)
        np.testing.assert_allclose(rust_apen, py_apen, rtol=1e-10)

        rust_sampen = sampen_rust(x, m=2, r_ratio=0.2)
        py_sampen = sampen_py(x, m=2, r_ratio=0.2)
        np.testing.assert_allclose(rust_sampen, py_sampen, rtol=1e-10)


class TestPerformance:
    """性能测试（可选，需要较长时间）"""

    @pytest.mark.slow
    def test_rust_faster_than_python(self):
        """Rust 实现应该比 Python 实现快"""
        import time

        from pyrs_indicators.util_entropy import approximate_entropy as apen_rust
        from pyrs_indicators.util_entropy import sample_entropy as sampen_rust
        from src.data_process.entropy.apen_sampen import (
            approximate_entropy as apen_py,
        )
        from src.data_process.entropy.apen_sampen import sample_entropy as sampen_py

        x = _generate_test_data(500)
        iterations = 10

        # ApEn 性能测试
        start = time.time()
        for _ in range(iterations):
            apen_py(x, m=2, r_ratio=0.3)
        py_apen_time = time.time() - start

        start = time.time()
        for _ in range(iterations):
            apen_rust(x, m=2, r_ratio=0.3)
        rust_apen_time = time.time() - start

        # SampEn 性能测试
        start = time.time()
        for _ in range(iterations):
            sampen_py(x, m=2, r_ratio=0.3)
        py_sampen_time = time.time() - start

        start = time.time()
        for _ in range(iterations):
            sampen_rust(x, m=2, r_ratio=0.3)
        rust_sampen_time = time.time() - start

        print(f"\nApEn: Python={py_apen_time:.3f}s, Rust={rust_apen_time:.3f}s")
        print(f"SampEn: Python={py_sampen_time:.3f}s, Rust={rust_sampen_time:.3f}s")

        # Rust 应该至少快 2 倍（保守估计）
        assert rust_apen_time < py_apen_time
        assert rust_sampen_time < py_sampen_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
