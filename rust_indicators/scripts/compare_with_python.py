"""
Rust vs Python 数值对齐对比工具

加载测试数据并对比 Rust 和 Python 实现的输出。
"""

import pickle
from pathlib import Path
import numpy as np
import sys

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_test_case(test_type: str, test_name: str) -> dict:
    """加载测试用例

    Args:
        test_type: "vmd" or "nrbo"
        test_name: 测试用例名称

    Returns:
        测试数据字典
    """
    test_data_dir = Path(__file__).parent.parent / "test_data" / test_type
    test_file = test_data_dir / f"{test_name}.pkl"

    if not test_file.exists():
        raise FileNotFoundError(f"测试文件不存在: {test_file}")

    with open(test_file, "rb") as f:
        return pickle.load(f)


def compare_arrays(
    arr1: np.ndarray, arr2: np.ndarray, name: str, atol: float = 1e-10
) -> bool:
    """对比两个数组

    Args:
        arr1: Python 输出
        arr2: Rust 输出
        name: 数组名称
        atol: 绝对容差

    Returns:
        是否通过对齐测试
    """
    print(f"\n对比 {name}:")
    print(f"  形状: Python={arr1.shape}, Rust={arr2.shape}")

    if arr1.shape != arr2.shape:
        print(f"  ❌ 形状不匹配!")
        return False

    # 计算误差
    abs_diff = np.abs(arr1 - arr2)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    rel_diff = np.max(abs_diff / (np.abs(arr1) + 1e-10))

    print(f"  最大绝对误差: {max_diff:.2e}")
    print(f"  平均绝对误差: {mean_diff:.2e}")
    print(f"  最大相对误差: {rel_diff:.2e}")

    if max_diff < atol:
        print(f"  ✅ 通过 (误差 < {atol:.0e})")
        return True
    else:
        print(f"  ❌ 失败 (误差 >= {atol:.0e})")

        # 显示最大误差的位置
        max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"  最大误差位置: {max_idx}")
        print(f"    Python 值: {arr1[max_idx]}")
        print(f"    Rust 值: {arr2[max_idx]}")
        print(f"    差值: {abs_diff[max_idx]:.2e}")

        return False


def compare_vmd_output(python_output: dict, rust_output: dict) -> bool:
    """对比 VMD 输出

    Returns:
        是否全部通过
    """
    print("=" * 60)
    print("VMD 输出对比")
    print("=" * 60)

    results = []

    # 对比 u (时域模态)
    results.append(
        compare_arrays(
            python_output["u"], rust_output["u"], "u (时域模态)", atol=1e-6
        )
    )

    # 对比 u_hat (频域模态) - 复数数组
    # 分别对比实部和虚部
    results.append(
        compare_arrays(
            python_output["u_hat"].real,
            rust_output["u_hat"].real,
            "u_hat.real (频域实部)",
            atol=1e-6,
        )
    )
    results.append(
        compare_arrays(
            python_output["u_hat"].imag,
            rust_output["u_hat"].imag,
            "u_hat.imag (频域虚部)",
            atol=1e-6,
        )
    )

    # 对比 omega (中心频率)
    results.append(
        compare_arrays(python_output["omega"], rust_output["omega"], "omega (中心频率)")
    )

    all_passed = all(results)
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)

    return all_passed


def compare_nrbo_output(python_output, rust_output) -> bool:
    """对比 NRBO 输出"""
    print("=" * 60)
    print("NRBO 输出对比")
    print("=" * 60)
    print()
    print(f"对比 NRBO 优化后的 IMF:")

    # Handle different data formats
    py_result = python_output['result'] if isinstance(python_output, dict) else python_output
    rs_result = rust_output['result'] if isinstance(rust_output, dict) else rust_output

    passed = compare_arrays(py_result, rs_result, "NRBO 优化后的 IMF")

    print("\n" + "=" * 60)
    if passed:
        print("🎉 测试通过！")
    else:
        print("❌ 测试失败")
    print("=" * 60)

    return passed


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="对比 Rust 和 Python 实现")
    parser.add_argument(
        "test_type", choices=["vmd", "nrbo"], help="测试类型"
    )
    parser.add_argument("test_name", help="测试用例名称 (如: simple_sine)")
    parser.add_argument(
        "--rust-output", help="Rust 输出文件 (.pkl)", default=None
    )

    args = parser.parse_args()

    # 加载 Python 测试数据
    test_data = load_test_case(args.test_type, args.test_name)

    print(f"\n📦 加载测试用例: {args.test_name}")
    print(f"   类型: {args.test_type.upper()}")
    print(f"   Python 输出已加载")

    if args.rust_output:
        # 加载 Rust 输出
        with open(args.rust_output, "rb") as f:
            rust_output = pickle.load(f)

        print(f"   Rust 输出已加载\n")

        # 对比
        if args.test_type == "vmd":
            compare_vmd_output(test_data["output"], rust_output["output"])
        else:  # nrbo
            compare_nrbo_output(test_data["output"], rust_output["output"])
    else:
        print("\n⚠️  未提供 Rust 输出文件，仅显示 Python 测试数据\n")

        # 显示测试数据摘要
        print("=" * 60)
        print("测试数据摘要")
        print("=" * 60)
        print(f"\n输入参数:")
        for key, value in test_data["input"].items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")

        print(f"\n输出:")
        if args.test_type == "vmd":
            print(f"  u: shape={test_data['output']['u'].shape}")
            print(f"  u_hat: shape={test_data['output']['u_hat'].shape}")
            print(f"  omega: shape={test_data['output']['omega'].shape}")
        else:
            print(f"  result: shape={test_data['output'].shape}")

        print("\n使用方法:")
        print(f"  python compare_with_python.py {args.test_type} {args.test_name} --rust-output <rust_output.pkl>")


if __name__ == "__main__":
    main()
