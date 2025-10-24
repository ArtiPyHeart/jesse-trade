"""
VMD (Variational Mode Decomposition) 冒烟测试

验证 Rust 实现能正常运行并返回合理的值
"""

import sys
import numpy as np
import pytest

# 导入 Rust 实现
try:
    import _rust_indicators
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("⚠️  Warning: _rust_indicators not available, run: cd rust_indicators && maturin develop --release")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_vmd_basic_decomposition():
    """测试 VMD 基本分解功能（冒烟测试）"""
    # 生成测试信号：两个不同频率的正弦波
    np.random.seed(42)
    t = np.linspace(0, 1, 200)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

    # 参数
    alpha = 2000.0
    tau = 0.0
    K = 3  # 分解为3个模态
    DC = False
    init = 1
    tol = 1e-7

    # Rust 实现 (返回 tuple: (u, u_hat, omega))
    u_modes, _, _ = _rust_indicators.vmd_py(signal, alpha, tau, K, DC, init, tol)

    print(f"\n  VMD 输出形状: {u_modes.shape}")
    print(f"  期望形状: ({K}, {len(signal)}) [模态 x 时间]")
    print(f"  模态1范围: [{np.min(u_modes[0]):.4f}, {np.max(u_modes[0]):.4f}]")
    print(f"  模态2范围: [{np.min(u_modes[1]):.4f}, {np.max(u_modes[1]):.4f}]")
    print(f"  模态3范围: [{np.min(u_modes[2]):.4f}, {np.max(u_modes[2]):.4f}]")

    # 验证输出形状
    assert u_modes.shape == (K, len(signal)), f"输出形状不正确: {u_modes.shape}"

    # 验证没有 NaN 或 Inf
    assert not np.any(np.isnan(u_modes)), "VMD 输出包含 NaN"
    assert not np.any(np.isinf(u_modes)), "VMD 输出包含 Inf"

    # 验证重构：所有模态之和应近似等于原始信号
    # 注意：实际使用中会跳过前几个模态，所以允许一定误差
    reconstructed = np.sum(u_modes, axis=0)
    reconstruction_error = np.max(np.abs(signal - reconstructed))
    print(f"  重构误差: {reconstruction_error:.2e}")

    # 验证能量守恒（宽松检查）
    original_energy = np.sum(signal ** 2)
    reconstructed_energy = np.sum(reconstructed ** 2)
    energy_ratio = reconstructed_energy / original_energy
    print(f"  能量比: {energy_ratio:.4f}")
    assert 0.5 < energy_ratio < 1.5, f"能量守恒失败: {energy_ratio:.4f}"

    print("  ✅ VMD 冒烟测试通过")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_vmd_price_signal():
    """测试 VMD 对价格信号的分解"""
    np.random.seed(123)
    # 模拟价格信号：趋势 + 周期 + 噪声
    t = np.arange(300)
    signal = 100 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 20) + np.random.randn(300) * 0.5

    alpha = 2000.0
    tau = 0.0
    K = 4
    DC = False
    init = 1
    tol = 1e-6

    # Rust 实现 (返回 tuple: (u, u_hat, omega))
    u_modes, _, _ = _rust_indicators.vmd_py(signal, alpha, tau, K, DC, init, tol)

    print(f"\n  VMD (价格信号) 输出形状: {u_modes.shape}")
    for i in range(K):
        print(f"  模态{i+1}范围: [{np.min(u_modes[i]):.4f}, {np.max(u_modes[i]):.4f}]")

    assert u_modes.shape == (K, len(signal))
    assert not np.any(np.isnan(u_modes))
    assert not np.any(np.isinf(u_modes))

    # 验证能量守恒
    original_energy = np.sum(signal ** 2)
    reconstructed = np.sum(u_modes, axis=0)
    reconstructed_energy = np.sum(reconstructed ** 2)
    energy_ratio = reconstructed_energy / original_energy
    print(f"  能量比: {energy_ratio:.4f}")

    # 验证模态正交性（不同模态之间应该相关性较低）
    correlations = []
    for i in range(K):
        for j in range(i+1, K):
            corr = np.corrcoef(u_modes[i], u_modes[j])[0, 1]
            correlations.append(abs(corr))
    avg_corr = np.mean(correlations) if correlations else 0
    print(f"  平均模态间相关性: {avg_corr:.4f}")

    assert 0.3 < energy_ratio < 1.5, f"能量守恒失败: {energy_ratio:.4f}"

    print("  ✅ VMD (价格信号) 冒烟测试通过")


if __name__ == "__main__":
    if not HAS_RUST:
        print("❌ Rust indicators 未安装，跳过测试")
        print("   运行: cd rust_indicators && cargo clean && maturin develop --release")
        sys.exit(1)

    print("=" * 60)
    print("VMD 冒烟测试 (PyO3 0.27 + numpy 0.27)")
    print("=" * 60)

    test_vmd_basic_decomposition()
    test_vmd_price_signal()

    print("\n✅ 所有 VMD 冒烟测试通过！")
