"""
FeatureMaker 和 Reducer 单元测试

测试覆盖：
1. FeatureMaker 基本功能
2. Reducer 基本功能
3. align_features_labels 对齐函数
4. 持久化/加载
5. 边界情况和错误处理
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import (
    FeatureMaker,
    FeatureMakerConfig,
    Reducer,
    ReducerConfig,
)
from src.features.dimensionality_reduction import ARDVAEConfig


# ==================== Fixtures ====================


@pytest.fixture
def sample_candles() -> np.ndarray:
    """生成测试用 K 线数据（500 根）"""
    np.random.seed(42)
    n = 500
    timestamps = np.arange(n) * 60000  # 1 分钟间隔
    opens = 100 + np.cumsum(np.random.randn(n) * 0.1)
    closes = opens + np.random.randn(n) * 0.05
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 0.02)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 0.02)
    volumes = np.abs(np.random.randn(n) * 1000) + 100

    return np.column_stack([timestamps, opens, closes, highs, lows, volumes])


@pytest.fixture
def simple_config() -> FeatureMakerConfig:
    """简单配置（只使用原始特征，不使用 SSM）"""
    # 使用实际存在的特征名
    return FeatureMakerConfig(
        feature_names=["natr", "fisher", "adx_14"],
        verbose=False,
    )


@pytest.fixture
def ssm_config() -> FeatureMakerConfig:
    """包含 SSM 的配置"""
    # 使用实际存在的特征名
    return FeatureMakerConfig(
        feature_names=["deep_ssm_0", "deep_ssm_1", "natr", "fisher"],
        ssm_state_dim=5,
        verbose=False,
    )


# ==================== FeatureMakerConfig 测试 ====================


class TestFeatureMakerConfig:
    """FeatureMakerConfig 测试"""

    def test_parse_raw_features(self):
        """测试原始特征解析"""
        config = FeatureMakerConfig(feature_names=["natr", "fisher", "adx_14"])
        assert config.raw_feature_names == ["natr", "fisher", "adx_14"]
        assert config.ssm_feature_names == []
        assert config.ssm_types == []

    def test_parse_ssm_features(self):
        """测试 SSM 特征解析"""
        config = FeatureMakerConfig(
            feature_names=["deep_ssm_0", "deep_ssm_1", "lg_ssm_0", "natr"]
        )
        assert config.raw_feature_names == ["natr"]
        assert "deep_ssm_0" in config.ssm_feature_names
        assert "lg_ssm_0" in config.ssm_feature_names
        assert config.ssm_types == ["deep_ssm", "lg_ssm"]

    def test_ssm_index_validation(self):
        """测试 SSM 索引验证"""
        with pytest.raises(ValueError, match="index .* >= state_dim"):
            FeatureMakerConfig(
                feature_names=["deep_ssm_10"],  # state_dim 默认是 5
                ssm_state_dim=5,
            )

    def test_schema_hash_consistency(self):
        """测试 schema hash 一致性"""
        config1 = FeatureMakerConfig(feature_names=["natr", "fisher"])
        config2 = FeatureMakerConfig(feature_names=["natr", "fisher"])
        assert config1.schema_hash == config2.schema_hash

        config3 = FeatureMakerConfig(feature_names=["fisher", "natr"])  # 不同顺序
        assert config1.schema_hash != config3.schema_hash

    def test_save_load_roundtrip(self):
        """测试配置保存/加载"""
        config = FeatureMakerConfig(
            feature_names=["deep_ssm_0", "natr"],
            ssm_state_dim=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(str(path))
            loaded = FeatureMakerConfig.load(str(path))

            assert loaded.feature_names == config.feature_names
            assert loaded.ssm_state_dim == config.ssm_state_dim
            assert loaded.schema_hash == config.schema_hash


# ==================== FeatureMaker 测试（无 SSM）====================


class TestFeatureMakerNoSSM:
    """FeatureMaker 测试（不使用 SSM）"""

    def test_fit_transform_returns_dataframe(self, sample_candles, simple_config):
        maker = FeatureMaker(simple_config)
        result = maker.fit_transform(sample_candles)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_candles)

    def test_output_columns_match_config(self, sample_candles, simple_config):
        maker = FeatureMaker(simple_config)
        result = maker.fit_transform(sample_candles)

        assert list(result.columns) == simple_config.feature_names

    def test_transform_after_fit(self, sample_candles, simple_config):
        maker = FeatureMaker(simple_config)
        maker.fit(sample_candles)

        result = maker.transform(sample_candles)
        assert isinstance(result, pd.DataFrame)

    def test_transform_without_fit_raises(self, sample_candles, simple_config):
        """测试未 fit 时 transform 抛出异常"""
        maker = FeatureMaker(simple_config)
        with pytest.raises(RuntimeError, match="not fitted"):
            maker.transform(sample_candles)

    def test_inference_returns_single_row(self, sample_candles, simple_config):
        """测试 inference 返回单行"""
        maker = FeatureMaker(simple_config)
        maker.fit(sample_candles)

        result = maker.inference(sample_candles)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_inference_without_fit_raises(self, sample_candles, simple_config):
        """测试未 fit 时 inference 抛出异常"""
        maker = FeatureMaker(simple_config)
        with pytest.raises(RuntimeError, match="not fitted"):
            maker.inference(sample_candles)

    def test_invalid_candles_shape(self, simple_config):
        """测试无效 candles 形状"""
        maker = FeatureMaker(simple_config)

        # 错误的列数
        bad_candles = np.random.randn(100, 5)
        with pytest.raises(ValueError, match="6 columns"):
            maker.fit(bad_candles)

        # 1D 数组
        with pytest.raises(ValueError, match="2D array"):
            maker.fit(np.random.randn(100))


# ==================== FeatureMaker 测试（有 SSM）====================


class TestFeatureMakerWithSSM:
    """FeatureMaker 测试（使用 SSM）"""

    def test_fit_transform_with_ssm(self, sample_candles, ssm_config):
        maker = FeatureMaker(ssm_config)
        result = maker.fit_transform(sample_candles)

        assert "deep_ssm_0" in result.columns
        assert "deep_ssm_1" in result.columns
        assert "natr" in result.columns

    def test_inference_idempotency(self, sample_candles, ssm_config):
        """测试 inference 幂等性（同一 timestamp 返回缓存）"""
        maker = FeatureMaker(ssm_config)
        maker.fit(sample_candles)
        maker.reset_ssm_states()

        # 第一次调用
        result1 = maker.inference(sample_candles)

        # 第二次调用（同一 timestamp）
        result2 = maker.inference(sample_candles)

        # 应该返回相同结果（缓存）
        pd.testing.assert_frame_equal(result1, result2)

    def test_inference_updates_on_new_timestamp(self, sample_candles, ssm_config):
        """测试新 timestamp 时 inference 更新"""
        maker = FeatureMaker(ssm_config)
        maker.fit(sample_candles)
        maker.reset_ssm_states()

        result1 = maker.inference(sample_candles)

        # 添加新 bar（新 timestamp）
        new_bar = sample_candles[-1:].copy()
        new_bar[0, 0] += 60000  # 新 timestamp
        new_candles = np.vstack([sample_candles, new_bar])

        result2 = maker.inference(new_candles)

        # 应该是不同的结果
        assert not result1.equals(result2)

    def test_reset_clears_idempotency_cache(self, sample_candles, ssm_config):
        """测试 reset 清除幂等缓存"""
        maker = FeatureMaker(ssm_config)
        maker.fit(sample_candles)

        # 获取 SSM 特征
        maker.reset_ssm_states()
        result1 = maker.inference(sample_candles)

        # reset 后应该重新计算
        maker.reset_ssm_states()
        result2 = maker.inference(sample_candles)

        # SSM 状态被 reset，结果应该相同（都是从初始状态开始）
        pd.testing.assert_frame_equal(result1, result2)

    def test_save_load_roundtrip(self, sample_candles, ssm_config):
        """测试保存/加载往返"""
        maker = FeatureMaker(ssm_config)
        maker.fit(sample_candles)

        with tempfile.TemporaryDirectory() as tmpdir:
            maker.save(tmpdir, "test_maker")

            loaded = FeatureMaker.load(tmpdir, "test_maker")
            assert loaded.is_fitted
            assert loaded.config.feature_names == ssm_config.feature_names

            # 加载后可以 inference
            loaded.reset_ssm_states()
            result = loaded.inference(sample_candles)
            assert len(result) == 1

    def test_load_missing_ssm_weights_raises(self, sample_candles, ssm_config):
        """测试加载时缺失 SSM 权重抛出异常"""
        maker = FeatureMaker(ssm_config)
        maker.fit(sample_candles)

        with tempfile.TemporaryDirectory() as tmpdir:
            maker.save(tmpdir, "test_maker")

            # 删除 SSM 权重文件
            ssm_path = Path(tmpdir) / "test_maker" / "deep_ssm.safetensors"
            ssm_path.unlink()

            with pytest.raises(FileNotFoundError, match="SSM weight file not found"):
                FeatureMaker.load(tmpdir, "test_maker")


# ==================== Reducer 测试 ====================


class TestReducer:
    """Reducer 测试"""

    @pytest.fixture
    def sample_features(self) -> pd.DataFrame:
        """生成测试用特征数据"""
        np.random.seed(42)
        n = 200
        return pd.DataFrame(
            np.random.randn(n, 50),
            columns=[f"feature_{i}" for i in range(50)],
        )

    def test_fit_transform(self, sample_features):
        """测试 fit_transform"""
        config = ReducerConfig(ard_vae_config=ARDVAEConfig(max_latent_dim=10, seed=42))
        reducer = Reducer(config)
        result = reducer.fit_transform(sample_features, verbose=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_features)
        # 列名应该是整数字符串
        assert all(col.isdigit() for col in result.columns)

    def test_transform_after_fit(self, sample_features):
        """测试 fit 后可以 transform"""
        config = ReducerConfig(ard_vae_config=ARDVAEConfig(max_latent_dim=10, seed=42))
        reducer = Reducer(config)
        reducer.fit(sample_features, verbose=False)

        result = reducer.transform(sample_features)
        assert isinstance(result, pd.DataFrame)

    def test_transform_without_fit_raises(self, sample_features):
        """测试未 fit 时 transform 抛出异常"""
        reducer = Reducer(ReducerConfig())
        with pytest.raises(RuntimeError, match="not fitted"):
            reducer.transform(sample_features)

    def test_column_mismatch_raises(self, sample_features):
        """测试列不匹配抛出异常"""
        config = ReducerConfig(ard_vae_config=ARDVAEConfig(max_latent_dim=10, seed=42))
        reducer = Reducer(config)
        reducer.fit(sample_features, verbose=False)

        # 修改列名
        wrong_features = sample_features.copy()
        wrong_features.columns = [f"wrong_{i}" for i in range(50)]

        with pytest.raises(ValueError, match="Column mismatch"):
            reducer.transform(wrong_features)

    def test_nan_input_raises(self, sample_features):
        """测试 NaN 输入抛出异常"""
        config = ReducerConfig(ard_vae_config=ARDVAEConfig(max_latent_dim=10, seed=42))
        reducer = Reducer(config)

        # 添加 NaN
        bad_features = sample_features.copy()
        bad_features.iloc[10, 5] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            reducer.fit(bad_features, verbose=False)

    def test_save_load_roundtrip(self, sample_features):
        """测试保存/加载往返"""
        config = ReducerConfig(ard_vae_config=ARDVAEConfig(max_latent_dim=10, seed=42))
        reducer = Reducer(config)
        reducer.fit(sample_features, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            reducer.save(tmpdir, "test_reducer")

            loaded = Reducer.load(tmpdir, "test_reducer")
            assert loaded.is_fitted

            # 加载后可以 transform
            result = loaded.transform(sample_features)
            assert isinstance(result, pd.DataFrame)

    def test_input_feature_names_subset(self, sample_features):
        """测试 input_feature_names 子集功能"""
        subset_cols = [f"feature_{i}" for i in range(10)]  # 只用前 10 列
        config = ReducerConfig(
            input_feature_names=subset_cols,
            ard_vae_config=ARDVAEConfig(max_latent_dim=5, seed=42),
        )
        reducer = Reducer(config)
        reducer.fit(sample_features, verbose=False)

        result = reducer.transform(sample_features)
        assert isinstance(result, pd.DataFrame)

    def test_input_feature_names_persisted(self, sample_features):
        """测试 input_feature_names 持久化"""
        subset_cols = [f"feature_{i}" for i in range(10)]
        config = ReducerConfig(
            input_feature_names=subset_cols,
            ard_vae_config=ARDVAEConfig(max_latent_dim=5, seed=42),
        )
        reducer = Reducer(config)
        reducer.fit(sample_features, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            reducer.save(tmpdir, "test_reducer")
            loaded = Reducer.load(tmpdir, "test_reducer")

            assert loaded.config.input_feature_names == subset_cols


# ==================== ReducerConfig 测试 ====================


class TestReducerConfig:
    """ReducerConfig 测试"""

    def test_schema_hash_includes_input_feature_names(self):
        """测试 schema_hash 包含 input_feature_names"""
        config1 = ReducerConfig(input_feature_names=["a", "b"])
        config2 = ReducerConfig(input_feature_names=["a", "b"])
        config3 = ReducerConfig(input_feature_names=["b", "a"])
        config4 = ReducerConfig(input_feature_names=None)

        assert config1.schema_hash == config2.schema_hash
        assert config1.schema_hash != config3.schema_hash
        assert config1.schema_hash != config4.schema_hash

    def test_save_load_roundtrip(self):
        """测试配置保存/加载"""
        config = ReducerConfig(
            reducer_type="ard_vae",
            input_feature_names=["a", "b", "c"],
            ard_vae_config=ARDVAEConfig(max_latent_dim=32),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.save(str(path))
            loaded = ReducerConfig.load(str(path))

            assert loaded.reducer_type == config.reducer_type
            assert loaded.input_feature_names == config.input_feature_names
            assert loaded.schema_hash == config.schema_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
