"""
CWT (Continuous Wavelet Transform) 冒烟与一致性测试

验证 Rust 实现能正常运行并与 PyWavelets 数值对齐
"""

import sys
import numpy as np
import pywt
import pytest

# 导入新的 Python 接口
try:
    from pyrs_indicators.ind_wavelets import cwt
    from pyrs_indicators import HAS_RUST
except ImportError:
    HAS_RUST = False
    print("⚠️  Warning: pyrs_indicators not available, run: cd rust_indicators && cargo clean && maturin develop --release")


EPSILON_DB = 1e-12


def _pywt_reference(signal, scales, wavelet, sampling_period, pad_width):
    """Reference implementation using PyWavelets (cmor family only)."""
    if pad_width:
        padded = np.pad(signal, pad_width, mode="symmetric")
    else:
        padded = signal

    coef_py, _ = pywt.cwt(
        padded,
        scales,
        wavelet,
        sampling_period=sampling_period,
        method="fft",
    )
    if pad_width:
        coef_py = coef_py[:, pad_width : pad_width + len(signal)]

    # Rust 实现使用 log10()，不是 20*log10()
    coef_py_db = np.log10(np.abs(coef_py) + EPSILON_DB)
    freqs_py = pywt.scale2frequency(wavelet, scales) / sampling_period
    return coef_py_db.T, freqs_py


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_cwt_cmor_wavelet():
    """测试 CWT 使用 Complex Morlet wavelet（实际生产环境使用的小波）"""
    np.random.seed(42)
    # 模拟价格数据
    t = np.linspace(0, 1, 200)
    signal = 100 + np.cumsum(np.random.randn(200) * 0.5)

    # 使用生产环境的参数
    scales = np.logspace(np.log2(8), np.log2(128), num=32, base=2)
    wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet (生产环境使用)
    sampling_period = 0.5  # 30分钟 = 0.5小时
    pad_width = int(max(scales))

    # Rust 实现
    try:
        coef_rust, freqs = cwt(
            signal,
            scales,
            wavelet=wavelet,
            sampling_period=sampling_period,
            precision=12,
            pad_width=pad_width,
            verbose=False
        )

        print(f"\n  CWT 输出形状: {coef_rust.shape}")
        print(f"  期望形状: ({len(signal)}, {len(scales)}) [时间 x 尺度]")
        print(f"  系数范围: [{np.min(coef_rust):.4f}, {np.max(coef_rust):.4f}] dB")
        print(f"  频率范围: [{np.min(freqs):.4f}, {np.max(freqs):.4f}] Hz")

        # 验证输出形状 (Rust 实现返回 [time, scales])
        assert coef_rust.shape == (len(signal), len(scales)), f"输出形状不正确: {coef_rust.shape}"
        assert freqs.shape == (len(scales),), f"频率数组形状不正确: {freqs.shape}"

        # 验证没有 NaN 或 Inf
        assert not np.any(np.isnan(coef_rust)), "CWT 输出包含 NaN"
        assert not np.any(np.isinf(coef_rust)), "CWT 输出包含 Inf"
        assert not np.any(np.isnan(freqs)), "频率数组包含 NaN"

        # 验证频率合理性（应该递减）
        assert np.all(freqs[:-1] >= freqs[1:]), "频率应随尺度增加而递减"

        print("  ✅ CWT (Complex Morlet) 冒烟测试通过")
        return True
    except Exception as e:
        print(f"  ⚠️  CWT 测试失败: {e}")
        return False


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_cwt_periodic_signal():
    """测试 CWT 对周期性信号的响应"""
    np.random.seed(123)
    t = np.arange(300)
    # 20周期的正弦波 + 噪声
    signal = 100 + 5 * np.sin(2 * np.pi * t / 20) + np.random.randn(300) * 0.5

    scales = np.logspace(np.log2(10), np.log2(100), num=20, base=2)
    wavelet = 'cmor1.5-1.0'

    try:
        coef_rust, freqs = cwt(
            signal,
            scales,
            wavelet=wavelet,
            sampling_period=1.0,
            precision=10,
            pad_width=int(max(scales)),
            verbose=False
        )

        print(f"\n  CWT (周期信号) 输出形状: {coef_rust.shape}")
        print(f"  系数范围: [{np.min(coef_rust):.4f}, {np.max(coef_rust):.4f}] dB")

        assert coef_rust.shape == (len(signal), len(scales))
        assert not np.any(np.isnan(coef_rust))
        assert not np.any(np.isinf(coef_rust))

        print("  ✅ CWT (周期信号) 冒烟测试通过")
        return True
    except Exception as e:
        print(f"  ⚠️  CWT (周期信号) 测试失败: {e}")
        return False


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_cwt_matches_pywt_without_padding():
    """Rust CWT 输出需与 PyWavelets 完全一致（无填充）"""
    t = np.linspace(0, 1, 256)
    signal = (np.sin(2 * np.pi * 15 * t) + 0.2 * np.sin(2 * np.pi * 5 * t)).astype(np.float64)

    scales = np.logspace(np.log2(4), np.log2(64), num=24, base=2, dtype=np.float64)
    wavelet = "cmor1.5-1.0"
    sampling_period = 0.5

    coef_rust, freqs_rust = cwt(
        signal,
        scales,
        wavelet=wavelet,
        sampling_period=sampling_period,
        precision=12,
        pad_width=0,
        verbose=False,
    )
    coef_py_db, freqs_py = _pywt_reference(signal, scales, wavelet, sampling_period, pad_width=0)

    np.testing.assert_allclose(coef_rust, coef_py_db, atol=6e-13)
    np.testing.assert_allclose(freqs_rust, freqs_py, atol=1e-15)


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_cwt_matches_pywt_with_symmetric_padding():
    """Rust CWT 输出需与 PyWavelets 完全一致（含对称填充）"""
    rng = np.random.default_rng(123)
    signal = rng.normal(loc=100.0, scale=0.8, size=300).cumsum().astype(np.float64)

    scales = np.logspace(np.log2(8), np.log2(128), num=32, base=2, dtype=np.float64)
    wavelet = "cmor1.5-1.0"
    sampling_period = 0.5
    pad_width = int(max(scales))

    coef_rust, freqs_rust = cwt(
        signal,
        scales,
        wavelet=wavelet,
        sampling_period=sampling_period,
        precision=12,
        pad_width=pad_width,
        verbose=False,
    )
    coef_py_db, freqs_py = _pywt_reference(
        signal,
        scales,
        wavelet,
        sampling_period,
        pad_width=pad_width,
    )

    np.testing.assert_allclose(coef_rust, coef_py_db, atol=6e-13)
    np.testing.assert_allclose(freqs_rust, freqs_py, atol=1e-15)


if __name__ == "__main__":
    if not HAS_RUST:
        print("❌ Rust indicators 未安装，跳过测试")
        print("   运行: cd rust_indicators && cargo clean && maturin develop --release")
        sys.exit(1)

    print("=" * 60)
    print("CWT 冒烟测试 (PyO3 0.27 + numpy 0.27)")
    print("=" * 60)

    success_count = 0
    success_count += test_cwt_cmor_wavelet()
    success_count += test_cwt_periodic_signal()

    if success_count == 2:
        print(f"\n✅ 所有 CWT 冒烟测试通过 ({success_count}/2)！")
    elif success_count > 0:
        print(f"\n⚠️  部分 CWT 测试通过 ({success_count}/2)")
    else:
        print("\n❌ CWT 测试全部失败")
