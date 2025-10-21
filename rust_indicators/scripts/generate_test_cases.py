"""
生成 VMD/NRBO 测试数据

从 Python 参考实现导出测试数据，用于 Rust 实现的数值对齐验证。
"""

import pickle
from pathlib import Path
import numpy as np
import sys

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.indicators.prod.emd.vmdpy import VMD as VMD_Python
from src.indicators.prod.emd.nrbo import nrbo as nrbo_python


def generate_test_signal(name: str, length: int = 1000) -> np.ndarray:
    """生成测试信号"""
    t = np.linspace(0, 1, length)

    signals = {
        "simple_sine": np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t),
        "noisy_signal": (
            np.sin(2 * np.pi * 5 * t)
            + np.sin(2 * np.pi * 20 * t)
            + 0.1 * np.random.randn(length)
        ),
        "three_components": (
            np.sin(2 * np.pi * 2 * t)
            + np.sin(2 * np.pi * 10 * t)
            + np.sin(2 * np.pi * 30 * t)
        ),
        "low_frequency": np.sin(2 * np.pi * 1 * t),
        "high_frequency": np.sin(2 * np.pi * 50 * t),
    }

    return signals.get(name, signals["simple_sine"])


def generate_vmd_test_cases():
    """生成 VMD 测试用例"""
    print("=" * 60)
    print("生成 VMD 测试数据")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "test_data" / "vmd"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        {
            "name": "simple_sine",
            "signal": generate_test_signal("simple_sine", 1000),
            "alpha": 2000,
            "tau": 0,
            "K": 2,
            "DC": False,
            "init": 1,
            "tol": 1e-7,
        },
        {
            "name": "three_components",
            "signal": generate_test_signal("three_components", 1000),
            "alpha": 2000,
            "tau": 0,
            "K": 3,
            "DC": False,
            "init": 1,
            "tol": 1e-7,
        },
        {
            "name": "with_dc",
            "signal": generate_test_signal("simple_sine", 500) + 1.0,  # 添加 DC 分量
            "alpha": 2000,
            "tau": 0,
            "K": 3,
            "DC": True,  # 测试 DC 模式
            "init": 1,
            "tol": 1e-7,
        },
        {
            "name": "odd_length",
            "signal": generate_test_signal("simple_sine", 999),  # 奇数长度
            "alpha": 2000,
            "tau": 0,
            "K": 2,
            "DC": False,
            "init": 1,
            "tol": 1e-7,
        },
        {
            "name": "small_signal",
            "signal": generate_test_signal("simple_sine", 100),  # 小信号
            "alpha": 2000,
            "tau": 0,
            "K": 2,
            "DC": False,
            "init": 1,
            "tol": 1e-6,
        },
    ]

    for case in test_cases:
        print(f"\n处理测试用例: {case['name']}")
        print(f"  信号长度: {len(case['signal'])}")
        print(f"  参数: K={case['K']}, alpha={case['alpha']}, DC={case['DC']}")

        # 运行 Python VMD
        u, u_hat, omega = VMD_Python(
            case["signal"],
            case["alpha"],
            case["tau"],
            case["K"],
            case["DC"],
            case["init"],
            case["tol"],
        )

        # 保存结果
        test_data = {
            "name": case["name"],
            "input": {
                "signal": case["signal"],
                "alpha": case["alpha"],
                "tau": case["tau"],
                "K": case["K"],
                "DC": case["DC"],
                "init": case["init"],
                "tol": case["tol"],
            },
            "output": {"u": u, "u_hat": u_hat, "omega": omega},
            "metadata": {
                "signal_length": len(case["signal"]),
                "num_modes": case["K"],
                "num_iterations": omega.shape[0],
            },
        }

        output_file = output_dir / f"{case['name']}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  ✓ 已保存到 {output_file}")
        print(f"  输出形状: u={u.shape}, u_hat={u_hat.shape}, omega={omega.shape}")

    print(f"\n✅ 完成！共生成 {len(test_cases)} 个 VMD 测试用例")
    print(f"保存目录: {output_dir}")


def generate_nrbo_test_cases():
    """生成 NRBO 测试用例"""
    print("\n" + "=" * 60)
    print("生成 NRBO 测试数据")
    print("=" * 60)

    output_dir = Path(__file__).parent.parent / "test_data" / "nrbo"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        {
            "name": "simple_sine",
            "imf": np.sin(np.linspace(0, 10, 100)),
            "max_iter": 10,
            "tol": 1e-6,
        },
        {
            "name": "complex_signal",
            "imf": np.sin(np.linspace(0, 20, 200)) + 0.1 * np.random.randn(200),
            "max_iter": 10,
            "tol": 1e-6,
        },
        {
            "name": "short_signal",
            "imf": np.sin(np.linspace(0, 5, 10)),
            "max_iter": 10,
            "tol": 1e-6,
        },
        {
            "name": "high_precision",
            "imf": np.sin(np.linspace(0, 10, 100)),
            "max_iter": 50,
            "tol": 1e-10,
        },
    ]

    for case in test_cases:
        print(f"\n处理测试用例: {case['name']}")
        print(f"  IMF 长度: {len(case['imf'])}")
        print(f"  参数: max_iter={case['max_iter']}, tol={case['tol']}")

        # 运行 Python NRBO
        result = nrbo_python(case["imf"], case["max_iter"], case["tol"])

        # 保存结果
        test_data = {
            "name": case["name"],
            "input": {
                "imf": case["imf"],
                "max_iter": case["max_iter"],
                "tol": case["tol"],
            },
            "output": result,
            "metadata": {"imf_length": len(case["imf"])},
        }

        output_file = output_dir / f"{case['name']}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  ✓ 已保存到 {output_file}")
        print(f"  输出形状: {result.shape}")

    print(f"\n✅ 完成！共生成 {len(test_cases)} 个 NRBO 测试用例")
    print(f"保存目录: {output_dir}")


if __name__ == "__main__":
    print("🚀 开始生成测试数据\n")

    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 生成 VMD 测试数据
    generate_vmd_test_cases()

    # 生成 NRBO 测试数据
    generate_nrbo_test_cases()

    print("\n" + "=" * 60)
    print("🎉 所有测试数据生成完成！")
    print("=" * 60)
    print("\n使用方法:")
    print("  1. Rust 测试: cargo test --package vmd -- --nocapture")
    print("  2. 查看数据: python -c 'import pickle; print(pickle.load(open(\"test_data/vmd/simple_sine.pkl\", \"rb\")))'")
