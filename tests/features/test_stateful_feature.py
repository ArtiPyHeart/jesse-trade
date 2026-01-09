"""
有状态特征测试

测试场景：
1. sequential=True 自动训练并缓存
2. sequential=False 加载缓存成功
3. sequential=False 无缓存时报错
4. 参数不匹配时报错
5. 版本过期 + sequential=True 重训练
6. 版本过期 + sequential=False 报错
7. clear_stateful_cache() 清除缓存
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from src.features.simple_feature_calculator import (
    SimpleFeatureCalculator,
    StatefulFeatureBase,
    get_global_registry,
)


class MockStatefulFeature(StatefulFeatureBase):
    """用于测试的模拟有状态特征"""

    VERSION = "1.0.0"

    def __init__(self, candles: np.ndarray, sequential: bool = True, param_a: int = 10):
        super().__init__(candles, sequential)
        self.param_a = param_a
        self._trained_mean: float = 0.0

    def get_params(self) -> Dict[str, Any]:
        return {"param_a": self.param_a}

    def get_version(self) -> str:
        return self.VERSION

    def train(self, candles: np.ndarray) -> None:
        # 简单训练：计算 close 价格的均值
        self._trained_mean = float(np.mean(candles[:, 2]))

    def inference(self, candles: np.ndarray, sequential: bool) -> np.ndarray:
        # 推理：返回 close 减去训练时的均值
        result = candles[:, 2] - self._trained_mean
        if sequential:
            return result
        return result[-1:]

    def get_state_dict(self) -> Dict[str, Any]:
        return {"trained_mean": np.array([self._trained_mean])}

    def set_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._trained_mean = float(state_dict["trained_mean"][0])


class MockStatefulFeatureV2(MockStatefulFeature):
    """版本 2.0.0 的模拟特征，用于测试版本过期"""

    VERSION = "2.0.0"


def make_candles(n: int = 100) -> np.ndarray:
    """生成模拟 Jesse 格式的 K 线数据"""
    np.random.seed(42)
    return np.column_stack(
        [
            np.arange(n),  # timestamp
            np.random.rand(n) + 100,  # open
            np.random.rand(n) + 100,  # close
            np.random.rand(n) + 101,  # high
            np.random.rand(n) + 99,  # low
            np.random.rand(n) * 1000,  # volume
        ]
    )


@pytest.fixture
def cache_dir():
    """临时缓存目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def candles():
    """测试用 K 线数据"""
    return make_candles(100)


@pytest.fixture
def registry():
    """获取全局注册中心并在测试后清理"""
    reg = get_global_registry()
    yield reg


class TestStatefulFeature:
    """有状态特征测试类"""

    def test_train_on_sequential_true(
        self, cache_dir: Path, candles: np.ndarray, registry
    ):
        """测试 sequential=True 时自动训练"""
        # 注册测试特征
        registry.register_stateful_class(
            "mock_stateful",
            MockStatefulFeature,
            params={"param_a": 10},
        )

        calc = SimpleFeatureCalculator(
            stateful_cache_dir=cache_dir,
            load_buildin=False,
        )
        calc.load(candles, sequential=True)

        result = calc.get(["mock_stateful"])
        assert "mock_stateful" in result
        assert len(result["mock_stateful"]) == len(candles)

        # 验证缓存已创建
        assert (cache_dir / "mock_stateful" / "meta.json").exists()
        assert (cache_dir / "mock_stateful" / "state.safetensors").exists()

    def test_load_cache_on_sequential_false(
        self, cache_dir: Path, candles: np.ndarray, registry
    ):
        """测试 sequential=False 时加载缓存"""
        # 注册并训练
        registry.register_stateful_class(
            "mock_stateful_load",
            MockStatefulFeature,
            params={"param_a": 20},
        )

        calc = SimpleFeatureCalculator(
            stateful_cache_dir=cache_dir,
            load_buildin=False,
        )
        calc.load(candles, sequential=True)
        train_result = calc.get(["mock_stateful_load"])

        # 清空内存缓存，模拟新会话
        calc.clear_cache()

        # 以 sequential=False 加载
        calc.load(candles, sequential=False)
        infer_result = calc.get(["mock_stateful_load"])

        assert len(infer_result["mock_stateful_load"]) == 1
        # 验证推理结果与训练结果最后一行一致
        np.testing.assert_allclose(
            infer_result["mock_stateful_load"],
            train_result["mock_stateful_load"][-1:],
        )

    def test_error_on_no_cache_sequential_false(
        self, cache_dir: Path, candles: np.ndarray, registry
    ):
        """测试无缓存时 sequential=False 报错"""
        registry.register_stateful_class(
            "mock_stateful_no_cache",
            MockStatefulFeature,
            params={"param_a": 30},
        )

        calc = SimpleFeatureCalculator(
            stateful_cache_dir=cache_dir,
            load_buildin=False,
        )
        calc.load(candles, sequential=False)

        with pytest.raises(RuntimeError, match="cache not found"):
            calc.get(["mock_stateful_no_cache"])

    def test_error_on_params_mismatch(
        self, cache_dir: Path, candles: np.ndarray, registry
    ):
        """测试参数不匹配时报错"""
        # 用 param_a=10 训练
        registry.register_stateful_class(
            "mock_stateful_params",
            MockStatefulFeature,
            params={"param_a": 10},
        )

        calc = SimpleFeatureCalculator(
            stateful_cache_dir=cache_dir,
            load_buildin=False,
        )
        calc.load(candles, sequential=True)
        calc.get(["mock_stateful_params"])

        # 修改参数重新注册
        registry.register_stateful_class(
            "mock_stateful_params",
            MockStatefulFeature,
            params={"param_a": 999},  # 不同参数
        )

        # 清空内存缓存
        calc.clear_cache()

        with pytest.raises(RuntimeError, match="params mismatch"):
            calc.get(["mock_stateful_params"])

    def test_retrain_on_version_outdated_sequential_true(
        self, cache_dir: Path, candles: np.ndarray, registry
    ):
        """测试版本过期时 sequential=True 重训练"""
        # 先用 V1 训练
        registry.register_stateful_class(
            "mock_stateful_version",
            MockStatefulFeature,  # VERSION = "1.0.0"
            params={"param_a": 10},
        )

        calc = SimpleFeatureCalculator(
            stateful_cache_dir=cache_dir,
            load_buildin=False,
        )
        calc.load(candles, sequential=True)
        calc.get(["mock_stateful_version"])

        # 用 V2 重新注册
        registry.register_stateful_class(
            "mock_stateful_version",
            MockStatefulFeatureV2,  # VERSION = "2.0.0"
            params={"param_a": 10},
        )

        # 清空内存缓存
        calc.clear_cache()

        # sequential=True 应该自动重训练
        result = calc.get(["mock_stateful_version"])
        assert len(result["mock_stateful_version"]) == len(candles)

    def test_error_on_version_outdated_sequential_false(
        self, cache_dir: Path, candles: np.ndarray, registry
    ):
        """测试版本过期时 sequential=False 报错"""
        # 先用 V1 训练
        registry.register_stateful_class(
            "mock_stateful_version_err",
            MockStatefulFeature,  # VERSION = "1.0.0"
            params={"param_a": 10},
        )

        calc = SimpleFeatureCalculator(
            stateful_cache_dir=cache_dir,
            load_buildin=False,
        )
        calc.load(candles, sequential=True)
        calc.get(["mock_stateful_version_err"])

        # 用 V2 重新注册
        registry.register_stateful_class(
            "mock_stateful_version_err",
            MockStatefulFeatureV2,  # VERSION = "2.0.0"
            params={"param_a": 10},
        )

        # 清空内存缓存
        calc.clear_cache()

        # sequential=False 应该报错
        calc.load(candles, sequential=False)
        with pytest.raises(RuntimeError, match="version outdated"):
            calc.get(["mock_stateful_version_err"])

    def test_clear_stateful_cache_single(
        self, cache_dir: Path, candles: np.ndarray, registry
    ):
        """测试清除单个特征缓存"""
        registry.register_stateful_class(
            "mock_stateful_clear1",
            MockStatefulFeature,
            params={"param_a": 10},
        )
        registry.register_stateful_class(
            "mock_stateful_clear2",
            MockStatefulFeature,
            params={"param_a": 20},
        )

        calc = SimpleFeatureCalculator(
            stateful_cache_dir=cache_dir,
            load_buildin=False,
        )
        calc.load(candles, sequential=True)
        calc.get(["mock_stateful_clear1", "mock_stateful_clear2"])

        assert (cache_dir / "mock_stateful_clear1").exists()
        assert (cache_dir / "mock_stateful_clear2").exists()

        # 只清除 clear1
        calc.clear_stateful_cache("mock_stateful_clear1")

        assert not (cache_dir / "mock_stateful_clear1").exists()
        assert (cache_dir / "mock_stateful_clear2").exists()

    def test_clear_stateful_cache_all(
        self, cache_dir: Path, candles: np.ndarray, registry
    ):
        """测试清除所有缓存"""
        registry.register_stateful_class(
            "mock_stateful_clear_all1",
            MockStatefulFeature,
            params={"param_a": 10},
        )
        registry.register_stateful_class(
            "mock_stateful_clear_all2",
            MockStatefulFeature,
            params={"param_a": 20},
        )

        calc = SimpleFeatureCalculator(
            stateful_cache_dir=cache_dir,
            load_buildin=False,
        )
        calc.load(candles, sequential=True)
        calc.get(["mock_stateful_clear_all1", "mock_stateful_clear_all2"])

        assert (cache_dir / "mock_stateful_clear_all1").exists()
        assert (cache_dir / "mock_stateful_clear_all2").exists()

        # 清除所有
        calc.clear_stateful_cache()

        assert not (cache_dir / "mock_stateful_clear_all1").exists()
        assert not (cache_dir / "mock_stateful_clear_all2").exists()
        # 目录本身应该被重新创建
        assert cache_dir.exists()

    def test_error_without_cache_dir(self, candles: np.ndarray, registry):
        """测试没有设置 cache_dir 时报错"""
        registry.register_stateful_class(
            "mock_stateful_no_dir",
            MockStatefulFeature,
            params={"param_a": 10},
        )

        calc = SimpleFeatureCalculator(
            stateful_cache_dir=None,  # 不设置
            load_buildin=False,
        )
        calc.load(candles, sequential=True)

        with pytest.raises(RuntimeError, match="stateful_cache_dir"):
            calc.get(["mock_stateful_no_dir"])

    def test_multi_column_stateful_feature(
        self, cache_dir: Path, candles: np.ndarray, registry
    ):
        """测试多列有状态特征"""

        class MultiColumnStateful(StatefulFeatureBase):
            def __init__(self, candles, sequential=True):
                super().__init__(candles, sequential)
                self._weights: np.ndarray = np.zeros(3)

            def get_params(self):
                return {}

            def get_version(self):
                return "1.0.0"

            def train(self, candles):
                self._weights = np.array([1.0, 2.0, 3.0])

            def inference(self, candles, sequential):
                result = np.outer(candles[:, 2], self._weights)  # [len(candles), 3]
                if sequential:
                    return result
                return result[-1:]

            def get_state_dict(self):
                return {"weights": self._weights}

            def set_state_dict(self, state_dict):
                self._weights = state_dict["weights"]

        registry.register_stateful_class(
            "mock_multi_col",
            MultiColumnStateful,
            returns_multiple=True,
        )

        calc = SimpleFeatureCalculator(
            stateful_cache_dir=cache_dir,
            load_buildin=False,
        )
        calc.load(candles, sequential=True)
        result = calc.get(["mock_multi_col"])

        assert result["mock_multi_col"].shape == (len(candles), 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
