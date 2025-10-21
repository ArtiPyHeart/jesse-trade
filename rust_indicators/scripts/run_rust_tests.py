#!/usr/bin/env python3
"""
运行 Rust 实现并保存输出，用于与 Python 对比
"""

import pickle
import numpy as np
from pathlib import Path
import _rust_indicators

def run_nrbo_tests():
    """运行所有 NRBO 测试用例"""
    test_dir = Path(__file__).parent.parent / "test_data" / "nrbo"

    for test_file in test_dir.glob("*.pkl"):
        test_name = test_file.stem
        print(f"Running NRBO test: {test_name}")

        # 加载测试数据
        with open(test_file, 'rb') as f:
            data = pickle.load(f)

        # 运行 Rust 实现
        imf = data['input']['imf']
        max_iter = data['input']['max_iter']
        tol = data['input']['tol']

        rust_result = _rust_indicators.nrbo_py(imf, max_iter=max_iter, tol=tol)

        # 保存 Rust 输出
        output_data = {
            'input': data['input'],
            'output': {'result': rust_result}
        }

        output_file = test_dir / f"{test_name}_rust.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)

        print(f"  ✅ Saved to {output_file}")

def run_vmd_tests():
    """运行所有 VMD 测试用例"""
    test_dir = Path(__file__).parent.parent / "test_data" / "vmd"

    for test_file in test_dir.glob("*.pkl"):
        test_name = test_file.stem
        print(f"Running VMD test: {test_name}")

        # 加载测试数据
        with open(test_file, 'rb') as f:
            data = pickle.load(f)

        # 运行 Rust 实现
        signal = data['input']['signal']
        alpha = data['input']['alpha']
        tau = data['input']['tau']
        K = data['input']['K']
        DC = data['input']['DC']
        init = data['input']['init']
        tol = data['input']['tol']

        u, u_hat, omega = _rust_indicators.vmd_py(
            signal, alpha=alpha, tau=tau, k=K, dc=DC, init=init, tol=tol
        )

        # 保存 Rust 输出
        output_data = {
            'input': data['input'],
            'output': {'u': u, 'u_hat': u_hat, 'omega': omega}
        }

        output_file = test_dir / f"{test_name}_rust.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)

        print(f"  ✅ Saved to {output_file}")

if __name__ == "__main__":
    print("=" * 60)
    print("Running NRBO tests...")
    print("=" * 60)
    run_nrbo_tests()

    print()
    print("=" * 60)
    print("Running VMD tests...")
    print("=" * 60)
    run_vmd_tests()

    print()
    print("✅ All Rust tests completed!")
