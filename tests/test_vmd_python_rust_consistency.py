"""
VMD (Variational Mode Decomposition) Python vs Rust 数值一致性测试

验证 Rust 实现与 Python 参考实现在数值上完全一致
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

# 导入 Python 参考实现
from src.indicators.prod.emd.vmdpy import VMD as VMD_python

# 导入 Rust 实现
try:
    from pyrs_indicators.ind_decomposition import vmd as vmd_rust
    from pyrs_indicators import HAS_RUST
except ImportError:
    HAS_RUST = False
    print("⚠️  Rust implementation not available")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_vmd_basic_comparison():
    """基础测试：对比 Python 和 Rust 实现的 VMD 分解结果"""
    np.random.seed(42)

    # 创建测试信号：两个不同频率的正弦波叠加
    t = np.linspace(0, 1, 200)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

    # VMD 参数
    alpha = 2000.0
    tau = 0.0
    K = 3
    DC = False
    init = 1
    tol = 1e-7

    # Python 实现
    u_py, u_hat_py, omega_py = VMD_python(
        signal, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol
    )

    # Rust 实现
    u_rs, u_hat_rs, omega_rs = vmd_rust(
        signal, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol, return_full=True
    )

    print(f"\n{'='*60}")
    print("基础测试：两个正弦波叠加信号")
    print(f"{'='*60}")
    print(f"信号长度: {len(signal)}")
    print(f"模态数量 K: {K}")

    print(f"\n【输出形状对比】")
    print(f"  Python u:     {u_py.shape}")
    print(f"  Rust u:       {u_rs.shape}")
    print(f"  Python u_hat: {u_hat_py.shape}")
    print(f"  Rust u_hat:   {u_hat_rs.shape}")
    print(f"  Python omega: {omega_py.shape}")
    print(f"  Rust omega:   {omega_rs.shape}")

    # 验证形状一致
    assert u_py.shape == u_rs.shape, f"u 形状不一致: {u_py.shape} vs {u_rs.shape}"
    assert u_hat_py.shape == u_hat_rs.shape, f"u_hat 形状不一致: {u_hat_py.shape} vs {u_hat_rs.shape}"

    print(f"\n【数值一致性对比】")

    # 对比模态 u
    u_diff = np.abs(u_py - u_rs)
    u_max_diff = np.max(u_diff)
    u_mean_diff = np.mean(u_diff)
    u_rel_error = u_max_diff / (np.max(np.abs(u_py)) + 1e-10)

    print(f"  模态 u:")
    print(f"    最大绝对误差: {u_max_diff:.2e}")
    print(f"    平均绝对误差: {u_mean_diff:.2e}")
    print(f"    最大相对误差: {u_rel_error:.2e}")

    for k in range(K):
        mode_diff = np.max(np.abs(u_py[k] - u_rs[k]))
        print(f"    模态 {k+1} 最大误差: {mode_diff:.2e}")

    # 对比频谱 u_hat（复数）
    u_hat_diff = np.abs(u_hat_py - u_hat_rs)
    u_hat_max_diff = np.max(u_hat_diff)
    u_hat_mean_diff = np.mean(u_hat_diff)

    print(f"\n  频谱 u_hat (复数):")
    print(f"    最大绝对误差: {u_hat_max_diff:.2e}")
    print(f"    平均绝对误差: {u_hat_mean_diff:.2e}")

    # 对比中心频率 omega
    # 注意：omega 的迭代次数可能不同，只对比最终收敛值
    omega_py_final = omega_py[-1, :]
    omega_rs_final = omega_rs[-1, :]
    omega_diff = np.abs(omega_py_final - omega_rs_final)
    omega_max_diff = np.max(omega_diff)

    print(f"\n  中心频率 omega (最终值):")
    print(f"    Python: {omega_py_final}")
    print(f"    Rust:   {omega_rs_final}")
    print(f"    最大误差: {omega_max_diff:.2e}")

    # 数值一致性断言（允许浮点误差）
    assert u_max_diff < 1e-10, f"模态 u 误差过大: {u_max_diff:.2e}"
    assert u_hat_max_diff < 1e-10, f"频谱 u_hat 误差过大: {u_hat_max_diff:.2e}"
    assert omega_max_diff < 1e-10, f"中心频率 omega 误差过大: {omega_max_diff:.2e}"

    print(f"\n✅ 数值一致性验证通过！")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_vmd_price_signal_comparison():
    """使用模拟价格信号测试"""
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

    # Python 实现
    u_py, u_hat_py, omega_py = VMD_python(
        signal, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol
    )

    # Rust 实现
    u_rs, u_hat_rs, omega_rs = vmd_rust(
        signal, alpha=alpha, tau=tau, K=K, DC=DC, init=init, tol=tol, return_full=True
    )

    print(f"\n{'='*60}")
    print("价格信号测试：趋势 + 周期 + 噪声")
    print(f"{'='*60}")
    print(f"信号长度: {len(signal)}")
    print(f"模态数量 K: {K}")

    # 数值对比
    u_diff = np.abs(u_py - u_rs)
    u_max_diff = np.max(u_diff)
    u_mean_diff = np.mean(u_diff)

    print(f"\n【数值一致性】")
    print(f"  模态 u 最大误差: {u_max_diff:.2e}")
    print(f"  模态 u 平均误差: {u_mean_diff:.2e}")

    # 验证重构误差一致
    reconstructed_py = np.sum(u_py, axis=0)
    reconstructed_rs = np.sum(u_rs, axis=0)

    recon_error_py = np.max(np.abs(signal - reconstructed_py))
    recon_error_rs = np.max(np.abs(signal - reconstructed_rs))

    print(f"\n【信号重构误差】")
    print(f"  Python 重构误差: {recon_error_py:.2e}")
    print(f"  Rust 重构误差:   {recon_error_rs:.2e}")
    print(f"  重构误差差异:     {abs(recon_error_py - recon_error_rs):.2e}")

    # 能量守恒对比
    original_energy = np.sum(signal ** 2)

    reconstructed_py_energy = np.sum(reconstructed_py ** 2)
    reconstructed_rs_energy = np.sum(reconstructed_rs ** 2)

    energy_ratio_py = reconstructed_py_energy / original_energy
    energy_ratio_rs = reconstructed_rs_energy / original_energy

    print(f"\n【能量守恒】")
    print(f"  原始信号能量:     {original_energy:.2e}")
    print(f"  Python 重构能量:  {reconstructed_py_energy:.2e} (比率: {energy_ratio_py:.6f})")
    print(f"  Rust 重构能量:    {reconstructed_rs_energy:.2e} (比率: {energy_ratio_rs:.6f})")
    print(f"  能量比率差异:      {abs(energy_ratio_py - energy_ratio_rs):.2e}")

    # 断言
    assert u_max_diff < 1e-10, f"模态误差过大: {u_max_diff:.2e}"
    assert abs(recon_error_py - recon_error_rs) < 1e-10, "重构误差不一致"
    assert abs(energy_ratio_py - energy_ratio_rs) < 1e-10, "能量守恒不一致"

    print(f"\n✅ 价格信号测试通过！")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_vmd_different_parameters():
    """测试不同参数组合下的数值一致性"""
    np.random.seed(456)

    t = np.linspace(0, 2, 400)
    signal = (
        np.sin(2 * np.pi * 3 * t) +
        0.5 * np.sin(2 * np.pi * 7 * t) +
        0.3 * np.sin(2 * np.pi * 15 * t)
    )

    # 测试不同的参数组合
    test_cases = [
        {"alpha": 2000.0, "K": 3, "init": 0, "DC": False, "name": "alpha=2000, init=0"},
        {"alpha": 5000.0, "K": 4, "init": 1, "DC": False, "name": "alpha=5000, K=4"},
        {"alpha": 2000.0, "K": 3, "init": 1, "DC": True, "name": "DC mode"},
    ]

    print(f"\n{'='*60}")
    print("不同参数组合测试")
    print(f"{'='*60}")

    for case in test_cases:
        print(f"\n测试案例: {case['name']}")

        # Python 实现
        u_py, _, _ = VMD_python(
            signal,
            alpha=case["alpha"],
            tau=0.0,
            K=case["K"],
            DC=case["DC"],
            init=case["init"],
            tol=1e-7,
        )

        # Rust 实现
        u_rs, _, _ = vmd_rust(
            signal,
            alpha=case["alpha"],
            tau=0.0,
            K=case["K"],
            DC=case["DC"],
            init=case["init"],
            tol=1e-7,
            return_full=True,
        )

        u_diff = np.abs(u_py - u_rs)
        u_max_diff = np.max(u_diff)

        print(f"  最大误差: {u_max_diff:.2e}")

        assert u_max_diff < 1e-10, f"参数组合 '{case['name']}' 误差过大: {u_max_diff:.2e}"

    print(f"\n✅ 所有参数组合测试通过！")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_vmd_edge_cases():
    """边界情况测试"""
    print(f"\n{'='*60}")
    print("边界情况测试")
    print(f"{'='*60}")

    # 测试 1: 最小信号长度
    print(f"\n测试 1: 较短信号 (N=50)")
    signal_short = np.random.randn(50)

    u_py, _, _ = VMD_python(signal_short, alpha=2000.0, tau=0.0, K=2, DC=False, init=1, tol=1e-6)
    u_rs = vmd_rust(signal_short, alpha=2000.0, tau=0.0, K=2, DC=False, init=1, tol=1e-6)

    diff = np.max(np.abs(u_py - u_rs))
    print(f"  最大误差: {diff:.2e}")
    assert diff < 1e-10

    # 测试 2: 常数信号
    print(f"\n测试 2: 常数信号")
    signal_const = np.ones(200) * 100.0

    u_py, _, _ = VMD_python(signal_const, alpha=2000.0, tau=0.0, K=2, DC=False, init=1, tol=1e-6)
    u_rs = vmd_rust(signal_const, alpha=2000.0, tau=0.0, K=2, DC=False, init=1, tol=1e-6)

    diff = np.max(np.abs(u_py - u_rs))
    print(f"  最大误差: {diff:.2e}")
    assert diff < 1e-10

    # 测试 3: 单一模态 K=1
    print(f"\n测试 3: 单一模态 (K=1)")
    signal_single = np.sin(2 * np.pi * np.linspace(0, 1, 200) * 5)

    u_py, _, _ = VMD_python(signal_single, alpha=2000.0, tau=0.0, K=1, DC=False, init=1, tol=1e-6)
    u_rs = vmd_rust(signal_single, alpha=2000.0, tau=0.0, K=1, DC=False, init=1, tol=1e-6)

    diff = np.max(np.abs(u_py - u_rs))
    print(f"  最大误差: {diff:.2e}")
    assert diff < 1e-10

    print(f"\n✅ 所有边界情况测试通过！")


if __name__ == "__main__":
    if not HAS_RUST:
        print("❌ Rust 实现未安装，跳过测试")
        print("   运行: cd rust_indicators && cargo clean && maturin develop --release")
        sys.exit(1)

    print("="*60)
    print("VMD Python vs Rust 数值一致性测试")
    print("="*60)

    test_vmd_basic_comparison()
    test_vmd_price_signal_comparison()
    test_vmd_different_parameters()
    test_vmd_edge_cases()

    print("\n" + "="*60)
    print("✅ 所有 VMD 数值一致性测试通过！")
    print("="*60)
