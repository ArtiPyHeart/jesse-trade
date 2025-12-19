"""
GrootCVSelector 单元测试

运行方式:
    pytest tests/test_grootcv_selector.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_selection.grootcv_selector import (
    GrootCVConfig,
    GrootCVSelector,
)


# ============== Fixtures ==============


@pytest.fixture
def sample_classification_data():
    """生成分类测试数据"""
    np.random.seed(42)
    n_samples = 500
    n_features = 15

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # 前 3 个特征与目标相关
    y = pd.Series(
        (X["feat_0"] + X["feat_1"] * 0.5 + X["feat_2"] * 0.3 > 0).astype(int)
    )
    return X, y


@pytest.fixture
def sample_regression_data():
    """生成回归测试数据"""
    np.random.seed(42)
    n_samples = 500
    n_features = 15

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(
        X["feat_0"] * 0.5 + X["feat_1"] * 0.3 + np.random.randn(n_samples) * 0.1
    )
    return X, y


# ============== Config Tests ==============


class TestGrootCVConfig:
    """配置类测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = GrootCVConfig()
        assert config.objective == "auto"  # 默认改为 auto
        assert config.cutoff == 1.0  # 改为 float
        assert config.n_folds == 5
        assert config.n_iter == 5
        assert config.silent is True
        assert config.fastshap is True  # 默认启用 fasttreeshap 加速

    def test_custom_config(self):
        """测试自定义配置"""
        config = GrootCVConfig(
            objective="rmse",
            cutoff=2,
            n_folds=3,
            lgbm_params={"min_data_in_leaf": 10},
        )
        assert config.objective == "rmse"
        assert config.cutoff == 2
        assert config.lgbm_params == {"min_data_in_leaf": 10}

    def test_invalid_objective(self):
        """测试无效 objective 报错"""
        with pytest.raises(ValueError):
            GrootCVConfig(objective="invalid")

    def test_invalid_cutoff(self):
        """测试无效 cutoff 报错"""
        with pytest.raises(ValueError):
            GrootCVConfig(cutoff=-1)
        # cutoff=0 也无效（gt=0）
        with pytest.raises(ValueError):
            GrootCVConfig(cutoff=0)

    def test_float_cutoff(self):
        """测试 float 类型 cutoff"""
        config = GrootCVConfig(cutoff=0.8)
        assert config.cutoff == 0.8
        assert isinstance(config.cutoff, float)

    def test_auto_objective(self):
        """测试 auto objective"""
        config = GrootCVConfig(objective="auto")
        assert config.objective == "auto"


# ============== Selector Tests ==============


class TestGrootCVSelectorBasic:
    """基本功能测试"""

    def test_fit_classification(self, sample_classification_data):
        """测试分类任务拟合"""
        X, y = sample_classification_data
        selector = GrootCVSelector(task_type="classification", verbose=False)
        selector.fit(X, y)

        assert selector.variables_ is not None
        assert selector.selected_features_ is not None
        assert selector.features_to_drop_ is not None
        assert (
            len(selector.selected_features_) + len(selector.features_to_drop_)
            == len(selector.variables_)
        )

    def test_fit_regression(self, sample_regression_data):
        """测试回归任务拟合"""
        X, y = sample_regression_data
        selector = GrootCVSelector(task_type="regression", verbose=False)
        selector.fit(X, y)

        assert selector.selected_features_ is not None
        assert len(selector.selected_features_) > 0

    def test_auto_task_detection(self, sample_classification_data):
        """测试自动任务类型检测"""
        X, y = sample_classification_data
        selector = GrootCVSelector(task_type="auto", verbose=False)
        selector.fit(X, y)

        # 二分类数据应该被检测为 classification
        assert selector.selected_features_ is not None


class TestGrootCVSelectorTransform:
    """transform 方法测试"""

    def test_transform(self, sample_classification_data):
        """测试 transform 输出"""
        X, y = sample_classification_data
        selector = GrootCVSelector(verbose=False)
        selector.fit(X, y)

        X_transformed = selector.transform(X)
        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed.columns) == len(selector.selected_features_)

    def test_fit_transform(self, sample_classification_data):
        """测试 fit_transform"""
        X, y = sample_classification_data
        selector = GrootCVSelector(verbose=False)
        X_transformed = selector.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert all(col in selector.selected_features_ for col in X_transformed.columns)

    def test_transform_before_fit_raises(self, sample_classification_data):
        """测试未拟合就 transform 报错"""
        X, y = sample_classification_data
        selector = GrootCVSelector()

        with pytest.raises(ValueError, match="fit"):
            selector.transform(X)


class TestGrootCVSelectorGetSupport:
    """get_support 方法测试"""

    def test_get_support_mask(self, sample_classification_data):
        """测试返回布尔掩码"""
        X, y = sample_classification_data
        selector = GrootCVSelector(verbose=False)
        selector.fit(X, y)

        mask = selector.get_support(indices=False)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(selector.variables_)
        assert mask.sum() == len(selector.selected_features_)

    def test_get_support_indices(self, sample_classification_data):
        """测试返回索引列表"""
        X, y = sample_classification_data
        selector = GrootCVSelector(verbose=False)
        selector.fit(X, y)

        indices = selector.get_support(indices=True)
        assert isinstance(indices, list)
        assert len(indices) == len(selector.selected_features_)


class TestGrootCVSelectorEdgeCases:
    """边界条件测试"""

    def test_dataframe_type_check(self):
        """测试输入类型检查"""
        selector = GrootCVSelector()
        with pytest.raises(TypeError, match="DataFrame"):
            selector.fit(np.array([[1, 2], [3, 4]]), [0, 1])

    def test_invalid_task_type(self):
        """测试无效 task_type 报错"""
        with pytest.raises(ValueError, match="task_type"):
            GrootCVSelector(task_type="invalid")

    def test_task_type_overrides_objective(self):
        """测试 task_type 覆盖 config.objective"""
        # task_type='classification' 应该覆盖 config.objective
        config = GrootCVConfig(objective="rmse")
        selector = GrootCVSelector(config=config, task_type="classification")
        assert selector.config.objective == "binary"

        # task_type='regression' 应该覆盖 config.objective
        config2 = GrootCVConfig(objective="binary")
        selector2 = GrootCVSelector(config=config2, task_type="regression")
        assert selector2.config.objective == "rmse"

        # task_type='auto' 不应该覆盖 config.objective
        config3 = GrootCVConfig(objective="rmse")
        selector3 = GrootCVSelector(config=config3, task_type="auto")
        assert selector3.config.objective == "rmse"

    def test_single_feature(self):
        """测试单特征情况"""
        np.random.seed(42)
        X = pd.DataFrame({"feat_0": np.random.randn(100)})
        y = pd.Series(np.random.randint(0, 2, 100))

        selector = GrootCVSelector(verbose=False)
        selector.fit(X, y)

        # 单特征应该被保留或丢弃，不应报错
        assert len(selector.variables_) == 1


class TestGrootCVSelectorCompatibility:
    """与 RFImportanceSelector 兼容性测试"""

    def test_has_relevance_attribute(self, sample_classification_data):
        """测试存在 relevance_ 属性"""
        X, y = sample_classification_data
        selector = GrootCVSelector(verbose=False)
        selector.fit(X, y)

        assert hasattr(selector, "relevance_")
        assert selector.relevance_ is not None

    def test_has_selected_features_attribute(self, sample_classification_data):
        """测试存在 selected_features_ 属性"""
        X, y = sample_classification_data
        selector = GrootCVSelector(verbose=False)
        selector.fit(X, y)

        assert hasattr(selector, "selected_features_")
        assert isinstance(selector.selected_features_, list)

    def test_has_features_to_drop_attribute(self, sample_classification_data):
        """测试存在 features_to_drop_ 属性"""
        X, y = sample_classification_data
        selector = GrootCVSelector(verbose=False)
        selector.fit(X, y)

        assert hasattr(selector, "features_to_drop_")
        assert isinstance(selector.features_to_drop_, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
