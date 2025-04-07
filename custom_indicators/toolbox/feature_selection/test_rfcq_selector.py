import unittest

import numpy as np
import pandas as pd
from feature_engine.selection import MRMR
from rfcq_selector import RFCQSelector
from sklearn.datasets import make_classification


class TestRFCQSelector(unittest.TestCase):
    """RFCQ特征选择器的单元测试"""

    def setUp(self):
        """创建测试数据"""
        # 创建一个合成数据集，用于测试
        X, y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=5,  # 只有5个特征是有信息量的
            n_redundant=15,  # 15个特征是冗余的
            random_state=42,
        )

        # 转换为DataFrame和Series
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        self.X = pd.DataFrame(X, columns=feature_names)
        self.y = pd.Series(y)

    def test_basic_functionality(self):
        """测试基本功能"""
        # 创建选择器
        selector = RFCQSelector(max_features=5, random_state=42, verbose=False)

        # 训练选择器
        selector.fit(self.X, self.y)

        # 检查是否正确选择了5个特征
        X_transformed = selector.transform(self.X)
        self.assertEqual(X_transformed.shape[1], 5)

        # 测试fit_transform
        X_fit_transformed = selector.fit_transform(self.X, self.y)
        self.assertEqual(X_fit_transformed.shape[1], 5)

        # 测试get_support
        mask = selector.get_support()
        self.assertEqual(mask.sum(), 5)

        indices = selector.get_support(indices=True)
        self.assertEqual(len(indices), 5)

    def test_default_max_features(self):
        """测试默认max_features（20%的特征）"""
        selector = RFCQSelector(random_state=42, verbose=False)
        selector.fit(self.X, self.y)

        expected_features = int(0.2 * self.X.shape[1])
        X_transformed = selector.transform(self.X)
        self.assertEqual(X_transformed.shape[1], expected_features)

    def test_custom_param_grid(self):
        """测试自定义参数网格"""
        param_grid = {"n_estimators": [50, 100], "max_depth": [2, 3]}
        selector = RFCQSelector(
            max_features=3, param_grid=param_grid, random_state=42, verbose=False
        )
        selector.fit(self.X, self.y)

        X_transformed = selector.transform(self.X)
        self.assertEqual(X_transformed.shape[1], 3)

    def test_feature_engine_consistency(self):
        """测试与feature-engine库的MRMR(方法='RFCQ')结果一致性"""
        # 创建自定义选择器
        rfcq_selector = RFCQSelector(
            max_features=5,
            scoring="accuracy",
            cv=3,
            param_grid={"max_depth": [1, 2, 3, 4]},
            random_state=42,
            verbose=False,
        )

        # 创建feature-engine的MRMR选择器
        fe_selector = MRMR(
            method="RFCQ",
            max_features=5,
            scoring="accuracy",
            cv=3,
            param_grid={"max_depth": [1, 2, 3, 4]},
            regression=False,
            random_state=42,
        )

        # 使用相同数据拟合两个选择器
        rfcq_selector.fit(self.X, self.y)
        fe_selector.fit(self.X, self.y)

        # 获取两个选择器选择的特征
        rfcq_features = set(self.X.columns[rfcq_selector.get_support()])
        fe_features = set(self.X.columns[fe_selector.get_support()])

        # 打印选择的特征以便调试
        print(f"RFCQ选择的特征: {rfcq_features}")
        print(f"Feature-engine选择的特征: {fe_features}")

        # 验证选择的特征集合是否相同
        self.assertEqual(rfcq_features, fe_features)

    def test_error_conditions(self):
        """测试错误条件"""
        selector = RFCQSelector(verbose=False)

        # 在fit之前调用transform应该引发ValueError
        with self.assertRaises(ValueError):
            selector.transform(self.X)

        # 在fit之前调用get_support应该引发ValueError
        with self.assertRaises(ValueError):
            selector.get_support()

        # 传入一个不是DataFrame的X应该引发TypeError
        with self.assertRaises(TypeError):
            selector.fit(self.X.values, self.y)

        # 创建一个只有1个数值型特征的数据框
        X_single = pd.DataFrame({"a": np.random.randn(10)})
        y_single = pd.Series(np.random.randint(0, 2, 10))

        # 传入一个只有1个特征的数据框应该引发ValueError
        with self.assertRaises(ValueError):
            selector.fit(X_single, y_single)


if __name__ == "__main__":
    unittest.main()
