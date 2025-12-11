"""
NRBO (Newton-Raphson Boundary Optimization) Python vs Rust 数值一致性测试

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
from src.indicators.prod.emd.nrbo import nrbo as nrbo_python

# 导入 Rust 实现
try:
    from pyrs_indicators._core import _rust_nrbo, HAS_RUST
except ImportError:
    HAS_RUST = False
    print("⚠️  Rust implementation not available")


def nrbo_rust(imf: np.ndarray, max_iter: int = 10, tol: float = 1e-6) -> np.ndarray:
    """Rust NRBO 的 Python 包装"""
    if not HAS_RUST:
        raise ImportError("Rust implementation not available")
    return _rust_nrbo(imf, max_iter, tol)


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_nrbo_basic_comparison():
    """基础测试：对比 Python 和 Rust 实现的 NRBO 结果"""
    np.random.seed(42)

    # 创建测试 IMF：正弦波叠加噪声
    t = np.linspace(0, 4 * np.pi, 200)
    imf = np.sin(t) + 0.1 * np.random.randn(200)

    max_iter = 10
    tol = 1e-6

    # Python 实现
    result_py = nrbo_python(imf, max_iter=max_iter, tol=tol)

    # Rust 实现
    result_rs = nrbo_rust(imf, max_iter=max_iter, tol=tol)

    print(f"\n{'='*60}")
    print("基础测试：正弦波 + 噪声")
    print(f"{'='*60}")
    print(f"信号长度: {len(imf)}")

    print(f"\n【输出形状对比】")
    print(f"  Python: {result_py.shape}")
    print(f"  Rust:   {result_rs.shape}")

    # 验证形状一致
    assert result_py.shape == result_rs.shape, f"形状不一致: {result_py.shape} vs {result_rs.shape}"

    print(f"\n【数值一致性对比】")

    # 对比结果
    diff = np.abs(result_py - result_rs)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_error = max_diff / (np.max(np.abs(result_py)) + 1e-10)

    print(f"  最大绝对误差: {max_diff:.2e}")
    print(f"  平均绝对误差: {mean_diff:.2e}")
    print(f"  最大相对误差: {rel_error:.2e}")

    # 数值一致性断言（允许浮点误差）
    assert max_diff < 1e-10, f"误差过大: {max_diff:.2e}"

    print(f"\n✅ 数值一致性验证通过！")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_nrbo_real_imf():
    """使用真实 VMD 分解后的 IMF 测试"""
    np.random.seed(123)

    # 模拟真实 IMF：周期性成分
    t = np.arange(300)
    imf = 5 * np.sin(2 * np.pi * t / 20) + 0.3 * np.random.randn(300)

    max_iter = 10
    tol = 1e-6

    # Python 实现
    result_py = nrbo_python(imf, max_iter=max_iter, tol=tol)

    # Rust 实现
    result_rs = nrbo_rust(imf, max_iter=max_iter, tol=tol)

    print(f"\n{'='*60}")
    print("真实 IMF 测试")
    print(f"{'='*60}")
    print(f"信号长度: {len(imf)}")

    # 数值对比
    diff = np.abs(result_py - result_rs)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\n【数值一致性】")
    print(f"  最大误差: {max_diff:.2e}")
    print(f"  平均误差: {mean_diff:.2e}")

    # 验证能量守恒
    input_energy = np.sum(imf ** 2)
    output_py_energy = np.sum(result_py ** 2)
    output_rs_energy = np.sum(result_rs ** 2)

    print(f"\n【能量对比】")
    print(f"  输入能量:     {input_energy:.4e}")
    print(f"  Python 输出:  {output_py_energy:.4e}")
    print(f"  Rust 输出:    {output_rs_energy:.4e}")
    print(f"  Python vs Rust 差异: {abs(output_py_energy - output_rs_energy):.2e}")

    # 断言
    assert max_diff < 1e-10, f"误差过大: {max_diff:.2e}"
    assert abs(output_py_energy - output_rs_energy) < 1e-10, "能量不一致"

    print(f"\n✅ 真实 IMF 测试通过！")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_nrbo_different_parameters():
    """测试不同参数组合下的数值一致性"""
    np.random.seed(456)

    t = np.linspace(0, 6 * np.pi, 400)
    imf = np.sin(t) * np.exp(-t / (6 * np.pi)) + 0.05 * np.random.randn(400)

    # 测试不同的参数组合
    test_cases = [
        {"max_iter": 5, "tol": 1e-6, "name": "max_iter=5"},
        {"max_iter": 20, "tol": 1e-6, "name": "max_iter=20"},
        {"max_iter": 10, "tol": 1e-8, "name": "tol=1e-8"},
        {"max_iter": 10, "tol": 1e-4, "name": "tol=1e-4"},
    ]

    print(f"\n{'='*60}")
    print("不同参数组合测试")
    print(f"{'='*60}")

    for case in test_cases:
        print(f"\n测试案例: {case['name']}")

        # Python 实现
        result_py = nrbo_python(imf, max_iter=case["max_iter"], tol=case["tol"])

        # Rust 实现
        result_rs = nrbo_rust(imf, max_iter=case["max_iter"], tol=case["tol"])

        diff = np.abs(result_py - result_rs)
        max_diff = np.max(diff)

        print(f"  最大误差: {max_diff:.2e}")

        assert max_diff < 1e-10, f"参数组合 '{case['name']}' 误差过大: {max_diff:.2e}"

    print(f"\n✅ 所有参数组合测试通过！")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_nrbo_edge_cases():
    """边界情况测试"""
    print(f"\n{'='*60}")
    print("边界情况测试")
    print(f"{'='*60}")

    # 测试 1: 短信号
    print(f"\n测试 1: 短信号 (N=10)")
    imf_short = np.array([1.0, 2.0, 1.5, 3.0, 2.0, 1.0, 2.5, 1.5, 2.0, 1.0])

    result_py = nrbo_python(imf_short, max_iter=10, tol=1e-6)
    result_rs = nrbo_rust(imf_short, max_iter=10, tol=1e-6)

    diff = np.max(np.abs(result_py - result_rs))
    print(f"  最大误差: {diff:.2e}")
    assert diff < 1e-10

    # 测试 2: 很短信号 (N < 3)
    print(f"\n测试 2: 很短信号 (N=2)")
    imf_tiny = np.array([1.0, 2.0])

    result_py = nrbo_python(imf_tiny, max_iter=10, tol=1e-6)
    result_rs = nrbo_rust(imf_tiny, max_iter=10, tol=1e-6)

    # 短输入应该直接返回原值
    diff = np.max(np.abs(result_py - result_rs))
    print(f"  最大误差: {diff:.2e}")
    assert diff < 1e-10

    # 测试 3: 常数信号
    print(f"\n测试 3: 常数信号")
    imf_const = np.ones(100) * 5.0

    result_py = nrbo_python(imf_const, max_iter=10, tol=1e-6)
    result_rs = nrbo_rust(imf_const, max_iter=10, tol=1e-6)

    diff = np.max(np.abs(result_py - result_rs))
    print(f"  最大误差: {diff:.2e}")
    assert diff < 1e-10

    # 测试 4: 无极值点信号（单调）
    print(f"\n测试 4: 单调信号")
    imf_mono = np.linspace(0, 10, 50)

    result_py = nrbo_python(imf_mono, max_iter=10, tol=1e-6)
    result_rs = nrbo_rust(imf_mono, max_iter=10, tol=1e-6)

    diff = np.max(np.abs(result_py - result_rs))
    print(f"  最大误差: {diff:.2e}")
    assert diff < 1e-10

    print(f"\n✅ 所有边界情况测试通过！")


@pytest.mark.skipif(not HAS_RUST, reason="Rust implementation not available")
def test_nrbo_with_vmd_output():
    """使用实际 VMD 分解结果进行测试"""
    try:
        from pyrs_indicators.ind_decomposition import vmd
    except ImportError:
        pytest.skip("VMD not available")

    np.random.seed(789)

    print(f"\n{'='*60}")
    print("VMD + NRBO 组合测试")
    print(f"{'='*60}")

    # 创建测试信号
    t = np.linspace(0, 2, 400)
    signal = (
        np.sin(2 * np.pi * 3 * t) +
        0.5 * np.sin(2 * np.pi * 7 * t) +
        0.2 * np.random.randn(400)
    )

    # VMD 分解
    u_modes, _, _ = vmd(signal, alpha=2000.0, tau=0.0, K=3, DC=False, init=1, tol=1e-7, return_full=True)

    print(f"VMD 分解后模态数: {u_modes.shape[0]}")

    # 对每个模态应用 NRBO
    for k in range(u_modes.shape[0]):
        imf = u_modes[k]

        result_py = nrbo_python(imf, max_iter=10, tol=1e-6)
        result_rs = nrbo_rust(imf, max_iter=10, tol=1e-6)

        diff = np.abs(result_py - result_rs)
        max_diff = np.max(diff)

        print(f"  模态 {k+1} 最大误差: {max_diff:.2e}")

        assert max_diff < 1e-10, f"模态 {k+1} 误差过大: {max_diff:.2e}"

    print(f"\n✅ VMD + NRBO 组合测试通过！")


if __name__ == "__main__":
    if not HAS_RUST:
        print("❌ Rust 实现未安装，跳过测试")
        print("   运行: cd rust_indicators && cargo clean && maturin develop --release")
        sys.exit(1)

    print("="*60)
    print("NRBO Python vs Rust 数值一致性测试")
    print("="*60)

    test_nrbo_basic_comparison()
    test_nrbo_real_imf()
    test_nrbo_different_parameters()
    test_nrbo_edge_cases()
    test_nrbo_with_vmd_output()

    print("\n" + "="*60)
    print("✅ 所有 NRBO 数值一致性测试通过！")
    print("="*60)
