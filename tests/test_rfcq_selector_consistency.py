"""
RFCQSelector 一致性测试

用于验证内存优化改造后，特征选择结果与改造前完全一致。

运行方式:
    pytest tests/test_rfcq_selector_consistency.py -v
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_selection.rfcq_selector import RFCQSelector


# Ground truth 文件路径
GROUND_TRUTH_PATH = Path(__file__).parent / "data" / "rfcq_ground_truth.pkl"


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


def _fit_selector(
    X: pd.DataFrame,
    y: pd.Series,
    max_features: int,
    task_type: str = "auto",
) -> RFCQSelector:
    selector = RFCQSelector(
        max_features=max_features,
        task_type=task_type,
        random_state=42,
        verbose=False,
        n_jobs=1,
    )
    selector.fit(X, y)
    return selector


def _build_ground_truth() -> dict:
    data: dict = {}
    X, y = generate_test_data()

    base_selector = _fit_selector(X, y, max_features=20)
    data["selected_features"] = {
        "features_to_drop_": base_selector.features_to_drop_,
        "variables_": list(base_selector.variables_),
    }
    data["relevance"] = base_selector.relevance_
    data["transformed_columns"] = list(base_selector.transform(X).columns)

    for max_features in [10, 20, 50]:
        for task_type in ["auto", "regression"]:
            selector = _fit_selector(
                X, y, max_features=max_features, task_type=task_type
            )
            data[f"params_{max_features}_{task_type}"] = {
                "features_to_drop_": selector.features_to_drop_,
            }

    X_small, y_small = generate_test_data(n_samples=100, n_features=5, seed=42)
    selector_small = _fit_selector(X_small, y_small, max_features=3)
    data["edge_case_small"] = {
        "n_dropped": len(selector_small.features_to_drop_),
        "features_to_drop_": selector_small.features_to_drop_,
    }

    return data


def _ensure_ground_truth() -> None:
    if GROUND_TRUTH_PATH.exists():
        return

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _build_ground_truth()
    with open(GROUND_TRUTH_PATH, "wb") as f:
        pickle.dump(data, f)


def load_ground_truth(key: str):
    """加载 ground truth 数据"""
    _ensure_ground_truth()

    with open(GROUND_TRUTH_PATH, "rb") as f:
        data = pickle.load(f)

    if key not in data:
        raise KeyError(f"Ground truth 中不存在 key: {key}")

    return data[key]


class TestSelectedFeaturesConsistency:
    """测试选择的特征与 ground truth 一致"""

    def test_features_to_drop_consistency(self):
        """测试 features_to_drop_ 完全一致"""
        X, y = generate_test_data()
        selector = _fit_selector(X, y, max_features=20)

        expected = load_ground_truth("selected_features")
        assert selector.features_to_drop_ == expected["features_to_drop_"], (
            f"features_to_drop_ 不一致:\n"
            f"期望: {expected['features_to_drop_'][:5]}...\n"
            f"实际: {selector.features_to_drop_[:5]}..."
        )

    def test_variables_consistency(self):
        """测试 variables_ 完全一致"""
        X, y = generate_test_data()
        selector = _fit_selector(X, y, max_features=20)

        expected = load_ground_truth("selected_features")
        assert list(selector.variables_) == expected["variables_"]


class TestRelevanceConsistency:
    """测试 relevance 数值一致性"""

    def test_relevance_values(self):
        """测试 relevance_ 数值一致（允许小误差）"""
        X, y = generate_test_data()
        selector = _fit_selector(X, y, max_features=20)

        expected = load_ground_truth("relevance")
        np.testing.assert_allclose(
            selector.relevance_,
            expected,
            rtol=1e-5,
            err_msg="relevance_ 数值不一致",
        )


class TestTransformConsistency:
    """测试 transform 输出一致性"""

    def test_transformed_columns(self):
        """测试 transform 后的列名完全一致"""
        X, y = generate_test_data()
        selector = RFCQSelector(
            max_features=20,
            random_state=42,
            verbose=False,
            n_jobs=1,
        )
        X_transformed = selector.fit_transform(X, y)

        expected_columns = load_ground_truth("transformed_columns")
        assert list(X_transformed.columns) == expected_columns


class TestParameterCombinations:
    """测试不同参数组合"""

    @pytest.mark.parametrize("max_features", [10, 20, 50])
    @pytest.mark.parametrize("task_type", ["auto", "regression"])
    def test_parameter_combinations(self, max_features, task_type):
        """测试不同 max_features 和 task_type 组合"""
        X, y = generate_test_data()
        selector = _fit_selector(X, y, max_features=max_features, task_type=task_type)

        key = f"params_{max_features}_{task_type}"
        expected = load_ground_truth(key)
        assert selector.features_to_drop_ == expected["features_to_drop_"], (
            f"参数组合 {key} 的 features_to_drop_ 不一致"
        )


class TestEdgeCases:
    """测试边界条件"""

    def test_small_feature_set(self):
        """测试少量特征的情况"""
        X_small, y_small = generate_test_data(n_samples=100, n_features=5, seed=42)
        selector = _fit_selector(X_small, y_small, max_features=3)

        expected = load_ground_truth("edge_case_small")
        assert len(selector.features_to_drop_) == expected["n_dropped"]
        assert selector.features_to_drop_ == expected["features_to_drop_"]


class TestReproducibility:
    """测试结果可重复性"""

    def test_multiple_runs_same_result(self):
        """测试多次运行结果相同"""
        X, y = generate_test_data()

        results = []
        for _ in range(3):
            selector = _fit_selector(X, y, max_features=20)
            results.append(selector.features_to_drop_.copy())

        for i in range(1, len(results)):
            assert results[0] == results[i], f"运行 {i + 1} 的结果与第 1 次不同"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
