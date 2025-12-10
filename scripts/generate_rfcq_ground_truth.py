#!/usr/bin/env python
"""
在 RFCQSelector 改造前运行此脚本，生成基准测试数据。

用法:
    python scripts/generate_rfcq_ground_truth.py

输出:
    tests/data/rfcq_ground_truth.pkl
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.feature_selection.rfcq_selector import RFCQSelector


def generate_test_data(n_samples: int = 5000, n_features: int = 100, seed: int = 42):
    """生成测试数据（固定随机种子确保可重复）"""
    np.random.seed(seed)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # 创建与前几个特征相关的目标变量
    y = pd.Series(
        X["feat_0"] * 0.5
        + X["feat_1"] * 0.3
        + X["feat_2"] * 0.1
        + np.random.randn(n_samples) * 0.1
    )
    return X, y


def generate_ground_truth():
    """生成所有测试用例的 ground truth"""
    results = {}

    print("生成测试数据...")
    X, y = generate_test_data()

    # 基础测试
    print("运行基础测试 (max_features=20)...")
    selector = RFCQSelector(max_features=20, random_state=42, verbose=False)
    selector.fit(X, y)

    results["selected_features"] = {
        "features_to_drop_": selector.features_to_drop_,
        "variables_": list(selector.variables_),
    }
    results["relevance"] = selector.relevance_.copy()
    results["transformed_columns"] = list(selector.transform(X).columns)

    # 参数组合测试
    for max_features in [10, 20, 50]:
        for task_type in ["auto", "regression"]:
            print(f"运行参数组合测试: max_features={max_features}, task_type={task_type}...")
            selector = RFCQSelector(
                max_features=max_features,
                task_type=task_type,
                random_state=42,
                verbose=False,
            )
            selector.fit(X, y)
            key = f"params_{max_features}_{task_type}"
            results[key] = {"features_to_drop_": selector.features_to_drop_}

    # 边界条件测试
    print("运行边界条件测试 (n_features=5)...")
    X_small, y_small = generate_test_data(n_samples=100, n_features=5, seed=42)
    selector_small = RFCQSelector(max_features=3, random_state=42, verbose=False)
    selector_small.fit(X_small, y_small)
    results["edge_case_small"] = {
        "features_to_drop_": selector_small.features_to_drop_,
        "n_dropped": len(selector_small.features_to_drop_),
    }

    return results


def verify_reproducibility(results: dict, n_runs: int = 3):
    """验证结果可重复性"""
    print(f"\n验证可重复性 ({n_runs} 次运行)...")
    X, y = generate_test_data()

    for i in range(n_runs):
        selector = RFCQSelector(max_features=20, random_state=42, verbose=False)
        selector.fit(X, y)

        if selector.features_to_drop_ != results["selected_features"]["features_to_drop_"]:
            print(f"  运行 {i+1}: 结果不一致!")
            return False
        print(f"  运行 {i+1}: 结果一致 ✓")

    return True


def main():
    output_path = project_root / "tests" / "data" / "rfcq_ground_truth.pkl"

    print("=" * 60)
    print("RFCQSelector Ground Truth 生成器")
    print("=" * 60)

    # 生成 ground truth
    results = generate_ground_truth()

    # 验证可重复性
    if not verify_reproducibility(results):
        print("\n错误: 结果不可重复，请检查随机种子设置")
        sys.exit(1)

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nGround truth 已保存到: {output_path}")
    print(f"包含 {len(results)} 个测试用例:")
    for key in results:
        print(f"  - {key}")


if __name__ == "__main__":
    main()
