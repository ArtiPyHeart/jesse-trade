"""
CWT (Continuous Wavelet Transform) PyWavelets vs Rust 数值一致性测试

验证 Rust 实现与 PyWavelets 参考实现在数值上完全一致
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

# 导入 PyWavelets 作为参考实现
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("⚠️  PyWavelets not available, install with: pip install PyWavelets")

# 导入 Rust 实现
try:
    from pyrs_indicators.ind_wavelets import cwt as cwt_rust
    from pyrs_indicators import HAS_RUST
except ImportError:
    HAS_RUST = False
    print("⚠️  Rust implementation not available")


def _cwt_pywavelets_reference(signal, scales, wavelet='cmor1.5-1.0', sampling_period=1.0):
    """
    使用 PyWavelets 计算 CWT（参考实现）

    Returns:
        coef_db: CWT 系数（dB 尺度），形状 (signal_len, num_scales)
        freqs: 对应的频率数组
    """
    # PyWavelets 的 cwt 返回 (coefficients, frequencies)
    # coefficients 形状是 (num_scales, signal_len)
    coef_complex, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)

    # 转置以匹配 Rust 实现的形状 (signal_len, num_scales)
    coef_complex = coef_complex.T

    # 转换为 dB 尺度
    epsilon = 1e-10
    coef_db = 20 * np.log10(np.abs(coef_complex) + epsilon)

    return coef_db, freqs


@pytest.mark.skipif(not HAS_RUST or not HAS_PYWT, reason="Rust or PyWavelets not available")
def test_cwt_basic_comparison():
    """基础测试：对比 PyWavelets 和 Rust 实现的 CWT 结果"""
    np.random.seed(42)

    # 创建测试信号：多个频率的正弦波叠加
    t = np.linspace(0, 1, 200)
    signal = (
        np.sin(2 * np.pi * 5 * t) +
        0.5 * np.sin(2 * np.pi * 10 * t) +
        0.3 * np.sin(2 * np.pi * 20 * t)
    )

    # CWT 参数
    scales = np.logspace(np.log2(8), np.log2(128), num=32, base=2)
    wavelet = 'cmor1.5-1.0'
    sampling_period = 0.5
    pad_width = int(max(scales))

    # PyWavelets 实现（参考）
    coef_pywt, freqs_pywt = _cwt_pywavelets_reference(
        signal, scales, wavelet=wavelet, sampling_period=sampling_period
    )

    # Rust 实现（不使用填充，与 PyWavelets 默认行为一致）
    coef_rust, freqs_rust = cwt_rust(
        signal,
        scales,
        wavelet=wavelet,
        sampling_period=sampling_period,
        precision=12,
        pad_width=0,  # PyWavelets 默认不填充
        verbose=False,
    )

    print(f"\n{'='*60}")
    print("基础测试：多频率正弦波叠加信号")
    print(f"{'='*60}")
    print(f"信号长度: {len(signal)}")
    print(f"尺度数量: {len(scales)}")

    print(f"\n【输出形状对比】")
    print(f"  PyWavelets coef: {coef_pywt.shape}")
    print(f"  Rust coef:       {coef_rust.shape}")
    print(f"  PyWavelets freqs: {freqs_pywt.shape}")
    print(f"  Rust freqs:       {freqs_rust.shape}")

    # 验证形状一致
    assert coef_pywt.shape == coef_rust.shape, f"系数形状不一致: {coef_pywt.shape} vs {coef_rust.shape}"
    assert freqs_pywt.shape == freqs_rust.shape, f"频率形状不一致: {freqs_pywt.shape} vs {freqs_rust.shape}"

    print(f"\n【数值一致性对比】")

    # 对比频率
    freqs_diff = np.abs(freqs_pywt - freqs_rust)
    freqs_max_diff = np.max(freqs_diff)
    freqs_mean_diff = np.mean(freqs_diff)

    print(f"  频率数组:")
    print(f"    最大绝对误差: {freqs_max_diff:.2e}")
    print(f"    平均绝对误差: {freqs_mean_diff:.2e}")
    print(f"    频率范围 (PyWavelets): [{np.min(freqs_pywt):.4f}, {np.max(freqs_pywt):.4f}]")
    print(f"    频率范围 (Rust):       [{np.min(freqs_rust):.4f}, {np.max(freqs_rust):.4f}]")

    # 对比系数（dB 尺度）
    coef_diff = np.abs(coef_pywt - coef_rust)
    coef_max_diff = np.max(coef_diff)
    coef_mean_diff = np.mean(coef_diff)
    coef_rel_error = coef_max_diff / (np.max(np.abs(coef_pywt)) + 1e-10)

    print(f"\n  CWT 系数 (dB):")
    print(f"    最大绝对误差: {coef_max_diff:.2e}")
    print(f"    平均绝对误差: {coef_mean_diff:.2e}")
    print(f"    最大相对误差: {coef_rel_error:.2e}")
    print(f"    系数范围 (PyWavelets): [{np.min(coef_pywt):.2f}, {np.max(coef_pywt):.2f}] dB")
    print(f"    系数范围 (Rust):       [{np.min(coef_rust):.2f}, {np.max(coef_rust):.2f}] dB")

    # 数值一致性断言
    # 注意：由于不同实现可能有轻微差异（填充方式、边界处理等），允许较宽松的误差
    assert freqs_max_diff < 1e-10, f"频率误差过大: {freqs_max_diff:.2e}"
    assert coef_mean_diff < 0.5, f"系数平均误差过大: {coef_mean_diff:.2e} dB"
    assert coef_max_diff < 5.0, f"系数最大误差过大: {coef_max_diff:.2e} dB"

    print(f"\n✅ 基础数值一致性验证通过！")


@pytest.mark.skipif(not HAS_RUST or not HAS_PYWT, reason="Rust or PyWavelets not available")
def test_cwt_price_signal_comparison():
    """使用模拟价格信号测试"""
    np.random.seed(123)

    # 模拟价格信号：趋势 + 周期 + 噪声
    t = np.arange(300)
    signal = 100 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 20) + np.random.randn(300) * 0.5

    scales = np.logspace(np.log2(10), np.log2(100), num=20, base=2)
    wavelet = 'cmor1.5-1.0'
    sampling_period = 1.0

    # PyWavelets 实现
    coef_pywt, freqs_pywt = _cwt_pywavelets_reference(
        signal, scales, wavelet=wavelet, sampling_period=sampling_period
    )

    # Rust 实现
    coef_rust, freqs_rust = cwt_rust(
        signal,
        scales,
        wavelet=wavelet,
        sampling_period=sampling_period,
        precision=12,
        pad_width=0,
        verbose=False,
    )

    print(f"\n{'='*60}")
    print("价格信号测试：趋势 + 周期 + 噪声")
    print(f"{'='*60}")
    print(f"信号长度: {len(signal)}")
    print(f"尺度数量: {len(scales)}")

    # 数值对比
    coef_diff = np.abs(coef_pywt - coef_rust)
    coef_max_diff = np.max(coef_diff)
    coef_mean_diff = np.mean(coef_diff)

    print(f"\n【数值一致性】")
    print(f"  系数最大误差: {coef_max_diff:.2e} dB")
    print(f"  系数平均误差: {coef_mean_diff:.2e} dB")

    # 对比时频分布的统计特性
    # 在每个时间点，找到能量最大的尺度（主导频率）
    dominant_scale_pywt = np.argmax(coef_pywt, axis=1)
    dominant_scale_rust = np.argmax(coef_rust, axis=1)

    scale_agreement = np.mean(dominant_scale_pywt == dominant_scale_rust)

    print(f"\n【时频特性对比】")
    print(f"  主导尺度一致率: {scale_agreement * 100:.2f}%")

    # 断言
    assert coef_mean_diff < 0.5, f"系数平均误差过大: {coef_mean_diff:.2e} dB"
    assert scale_agreement > 0.90, f"主导尺度一致率过低: {scale_agreement * 100:.2f}%"

    print(f"\n✅ 价格信号测试通过！")


@pytest.mark.skipif(not HAS_RUST or not HAS_PYWT, reason="Rust or PyWavelets not available")
def test_cwt_different_wavelets():
    """测试不同的 Complex Morlet 小波参数"""
    np.random.seed(456)

    t = np.linspace(0, 2, 400)
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz 正弦波

    scales = np.logspace(np.log2(8), np.log2(64), num=16, base=2)

    # 测试不同的 cmor 参数
    wavelets = ['cmor1.0-1.0', 'cmor1.5-1.0', 'cmor2.0-1.0']

    print(f"\n{'='*60}")
    print("不同小波参数测试")
    print(f"{'='*60}")

    for wavelet in wavelets:
        print(f"\n测试小波: {wavelet}")

        # PyWavelets 实现
        coef_pywt, _ = _cwt_pywavelets_reference(signal, scales, wavelet=wavelet, sampling_period=1.0)

        # Rust 实现
        coef_rust, _ = cwt_rust(
            signal,
            scales,
            wavelet=wavelet,
            sampling_period=1.0,
            precision=12,
            pad_width=0,
            verbose=False,
        )

        coef_diff = np.abs(coef_pywt - coef_rust)
        coef_max_diff = np.max(coef_diff)
        coef_mean_diff = np.mean(coef_diff)

        print(f"  最大误差: {coef_max_diff:.2e} dB")
        print(f"  平均误差: {coef_mean_diff:.2e} dB")

        assert coef_mean_diff < 0.5, f"小波 '{wavelet}' 平均误差过大: {coef_mean_diff:.2e} dB"

    print(f"\n✅ 所有小波参数测试通过！")


@pytest.mark.skipif(not HAS_RUST or not HAS_PYWT, reason="Rust or PyWavelets not available")
def test_cwt_different_scales():
    """测试不同的尺度范围"""
    np.random.seed(789)

    signal = np.random.randn(300)

    # 测试不同的尺度范围
    scale_ranges = [
        {"min": 4, "max": 32, "num": 10, "name": "小尺度范围"},
        {"min": 8, "max": 128, "num": 32, "name": "中等尺度范围"},
        {"min": 16, "max": 256, "num": 50, "name": "大尺度范围"},
    ]

    print(f"\n{'='*60}")
    print("不同尺度范围测试")
    print(f"{'='*60}")

    for case in scale_ranges:
        scales = np.logspace(np.log2(case["min"]), np.log2(case["max"]), num=case["num"], base=2)
        print(f"\n测试: {case['name']} ({case['min']}-{case['max']}, {case['num']} 个尺度)")

        # PyWavelets 实现
        coef_pywt, _ = _cwt_pywavelets_reference(signal, scales, sampling_period=1.0)

        # Rust 实现
        coef_rust, _ = cwt_rust(
            signal,
            scales,
            wavelet='cmor1.5-1.0',
            sampling_period=1.0,
            precision=12,
            pad_width=0,
            verbose=False,
        )

        coef_diff = np.abs(coef_pywt - coef_rust)
        coef_max_diff = np.max(coef_diff)
        coef_mean_diff = np.mean(coef_diff)

        print(f"  最大误差: {coef_max_diff:.2e} dB")
        print(f"  平均误差: {coef_mean_diff:.2e} dB")

        assert coef_mean_diff < 1.0, f"尺度范围 '{case['name']}' 平均误差过大: {coef_mean_diff:.2e} dB"

    print(f"\n✅ 所有尺度范围测试通过！")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_cwt_rust_specific_features():
    """测试 Rust 实现的特定功能（填充）"""
    np.random.seed(999)

    signal = np.sin(2 * np.pi * np.linspace(0, 1, 200) * 10)
    scales = np.logspace(np.log2(8), np.log2(64), num=16, base=2)
    pad_width = int(max(scales))

    print(f"\n{'='*60}")
    print("Rust 特定功能测试：填充")
    print(f"{'='*60}")

    # 不填充
    coef_no_pad, _ = cwt_rust(
        signal,
        scales,
        wavelet='cmor1.5-1.0',
        sampling_period=1.0,
        precision=12,
        pad_width=0,
        verbose=False,
    )

    # 使用填充
    coef_with_pad, _ = cwt_rust(
        signal,
        scales,
        wavelet='cmor1.5-1.0',
        sampling_period=1.0,
        precision=12,
        pad_width=pad_width,
        verbose=False,
    )

    print(f"\n填充宽度: {pad_width}")
    print(f"无填充输出形状: {coef_no_pad.shape}")
    print(f"有填充输出形状: {coef_with_pad.shape}")

    # 填充应该减少边界效应，中心部分应该更接近
    center_slice = slice(pad_width, -pad_width if pad_width > 0 else None)
    center_diff = np.abs(coef_no_pad[center_slice] - coef_with_pad[center_slice])
    center_mean_diff = np.mean(center_diff)

    print(f"中心区域平均差异: {center_mean_diff:.2e} dB")

    # 验证输出长度一致（填充后应该移除）
    assert coef_no_pad.shape == coef_with_pad.shape, "填充后输出形状应该一致"

    # 填充应该对中心区域产生一些影响（减少边界效应）
    # 但不应该产生巨大差异
    assert center_mean_diff < 10.0, f"填充导致的中心区域差异过大: {center_mean_diff:.2e} dB"

    print(f"\n✅ Rust 特定功能测试通过！")


if __name__ == "__main__":
    if not HAS_RUST:
        print("❌ Rust 实现未安装，跳过测试")
        print("   运行: cd rust_indicators && cargo clean && maturin develop --release")
        sys.exit(1)

    if not HAS_PYWT:
        print("❌ PyWavelets 未安装，跳过测试")
        print("   运行: pip install PyWavelets")
        sys.exit(1)

    print("="*60)
    print("CWT PyWavelets vs Rust 数值一致性测试")
    print("="*60)

    test_cwt_basic_comparison()
    test_cwt_price_signal_comparison()
    test_cwt_different_wavelets()
    test_cwt_different_scales()
    test_cwt_rust_specific_features()

    print("\n" + "="*60)
    print("✅ 所有 CWT 数值一致性测试通过！")
    print("="*60)
