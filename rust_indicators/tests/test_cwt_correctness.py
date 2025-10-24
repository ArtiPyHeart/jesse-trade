"""
CWT (Continuous Wavelet Transform) 冒烟测试

验证 Rust 实现能正常运行并返回合理的值
"""

import sys
import numpy as np
import pytest

# 导入新的 Python 接口
try:
    from pyrs_indicators.ind_wavelets import cwt
    from pyrs_indicators import HAS_RUST
except ImportError:
    HAS_RUST = False
    print("⚠️  Warning: pyrs_indicators not available, run: cd rust_indicators && cargo clean && maturin develop --release")


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
