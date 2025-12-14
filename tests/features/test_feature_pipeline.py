"""
FeaturePipeline 单元测试

测试 FeaturePipeline 的三种使用模式和持久化功能。
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FeaturePipeline, PipelineConfig
from src.features.ssm import DeepSSMAdapter, LGSSMAdapter
from src.models.deep_ssm import DeepSSMConfig
from src.models.lgssm import LGSSMConfig


def make_jesse_candles(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    生成符合 Jesse 格式的假 K 线数据

    Args:
        n_samples: 样本数量
        seed: 随机种子

    Returns:
        二维数组 (n_samples, 6)，列为 [timestamp, open, close, high, low, volume]
    """
    np.random.seed(seed)

    # 生成时间戳（每分钟一根 K 线）
    timestamps = np.arange(n_samples) * 60000 + 1600000000000

    # 生成价格（随机游走）
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    # 生成 OHLC
    noise = np.random.randn(n_samples, 4) * 0.001
    open_prices = prices * (1 + noise[:, 0])
    close_prices = prices * (1 + noise[:, 1])
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(noise[:, 2]))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(noise[:, 3]))

    # 生成成交量
    volumes = np.random.rand(n_samples) * 1000 + 100

    return np.column_stack([
        timestamps, open_prices, close_prices, high_prices, low_prices, volumes
    ])


class TestPipelineConfig:
    """PipelineConfig 单元测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = PipelineConfig()

        assert config.feature_names == []
        assert config.ssm_state_dim == 5
        assert len(config.ssm_input_features) == 80  # fracdiff 特征
        assert not config.use_dimension_reducer

    def test_custom_config_raw_features_only(self):
        """测试只有一阶特征的配置"""
        config = PipelineConfig(
            feature_names=["rsi", "macd", "atr"]
        )

        assert config.raw_feature_names == ["rsi", "macd", "atr"]
        assert config.ssm_feature_names == []
        assert config.ssm_types == []

    def test_custom_config_ssm_features_only(self):
        """测试只有 SSM 特征的配置"""
        config = PipelineConfig(
            feature_names=["deep_ssm_0", "deep_ssm_1", "lg_ssm_0"]
        )

        assert config.raw_feature_names == []
        assert config.ssm_feature_names == ["deep_ssm_0", "deep_ssm_1", "lg_ssm_0"]
        assert set(config.ssm_types) == {"deep_ssm", "lg_ssm"}

    def test_custom_config_mixed_features(self):
        """测试混合特征的配置"""
        config = PipelineConfig(
            feature_names=["deep_ssm_0", "rsi", "lg_ssm_1", "macd"]
        )

        assert config.raw_feature_names == ["rsi", "macd"]
        assert config.ssm_feature_names == ["deep_ssm_0", "lg_ssm_1"]
        assert set(config.ssm_types) == {"deep_ssm", "lg_ssm"}

    def test_all_calculator_features_with_ssm(self):
        """测试带 SSM 时的 all_calculator_features"""
        config = PipelineConfig(
            feature_names=["deep_ssm_0", "rsi"]
        )

        # 应该包含 rsi 和所有 ssm_input_features
        calc_features = set(config.all_calculator_features)
        assert "rsi" in calc_features
        assert "frac_o_o1_diff" in calc_features  # SSM 输入特征之一

    def test_all_calculator_features_without_ssm(self):
        """测试不带 SSM 时的 all_calculator_features"""
        config = PipelineConfig(
            feature_names=["rsi", "macd"]
        )

        # 只包含原始特征，不包含 SSM 输入特征
        assert set(config.all_calculator_features) == {"rsi", "macd"}

    def test_invalid_ssm_index(self):
        """测试无效的 SSM 索引"""
        with pytest.raises(ValueError, match="index 5 >= state_dim 5"):
            PipelineConfig(
                feature_names=["deep_ssm_5"],  # 默认 state_dim=5，最大索引为 4
                ssm_state_dim=5,
            )

    def test_valid_ssm_index_with_custom_state_dim(self):
        """测试自定义 state_dim 后的有效索引"""
        config = PipelineConfig(
            feature_names=["deep_ssm_9"],
            ssm_state_dim=10,
        )
        assert "deep_ssm_9" in config.ssm_feature_names

    def test_save_load(self):
        """测试配置保存和加载"""
        config = PipelineConfig(
            feature_names=["deep_ssm_0", "rsi", "lg_ssm_1"],
            ssm_state_dim=5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "config.json"

            config.save(str(save_path))
            loaded = PipelineConfig.load(str(save_path))

            assert loaded.feature_names == config.feature_names
            assert loaded.ssm_state_dim == config.ssm_state_dim
            assert loaded.raw_feature_names == config.raw_feature_names
            assert loaded.ssm_feature_names == config.ssm_feature_names

    def test_copy(self):
        """测试配置复制"""
        config = PipelineConfig(
            feature_names=["deep_ssm_0", "rsi"],
            ssm_state_dim=5,
        )

        copied = config.copy(ssm_state_dim=10)

        assert copied.feature_names == ["deep_ssm_0", "rsi"]
        assert copied.ssm_state_dim == 10
        assert config.ssm_state_dim == 5  # 原配置不变


class TestFeaturePipelineBasic:
    """FeaturePipeline 基础功能测试"""

    def test_init_default(self):
        """测试默认初始化"""
        pipeline = FeaturePipeline()

        assert pipeline.config is not None
        assert not pipeline.is_fitted
        assert pipeline.ssm_processors == {}
        assert pipeline.dimension_reducer is None

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = PipelineConfig(
            feature_names=["deep_ssm_0", "rsi"],
        )
        pipeline = FeaturePipeline(config=config)

        assert pipeline.config.feature_names == ["deep_ssm_0", "rsi"]
        assert pipeline.config.raw_feature_names == ["rsi"]
        assert pipeline.config.ssm_feature_names == ["deep_ssm_0"]

    def test_init_with_ssm_processors(self):
        """测试带 SSM 处理器初始化"""
        # 创建 mock SSM 处理器
        deep_ssm = MagicMock(spec=DeepSSMAdapter)
        deep_ssm.is_fitted = True

        ssm_processors = {"deep_ssm": deep_ssm}
        pipeline = FeaturePipeline(ssm_processors=ssm_processors)

        assert "deep_ssm" in pipeline.ssm_processors
        assert pipeline.ssm_processors["deep_ssm"] == deep_ssm

    def test_get_all_feature_names(self):
        """测试 get_all_feature_names 方法"""
        config = PipelineConfig(
            feature_names=["deep_ssm_0", "rsi", "lg_ssm_1"]
        )
        pipeline = FeaturePipeline(config=config)

        names = pipeline.get_all_feature_names()
        assert names == ["deep_ssm_0", "rsi", "lg_ssm_1"]


class TestFeaturePipelineWithMockedCalculator:
    """使用 Mock 的 FeaturePipeline 测试"""

    @pytest.fixture
    def mock_calculator(self):
        """创建 mock 特征计算器"""
        calculator = MagicMock()

        # 模拟 get 方法返回特征字典
        def mock_get(feature_names):
            n_samples = 100
            return {name: np.random.randn(n_samples) for name in feature_names}

        calculator.get.side_effect = mock_get
        return calculator

    @pytest.fixture
    def sample_candles(self):
        """创建测试 candles"""
        np.random.seed(42)
        n_samples = 100

        return np.column_stack(
            [
                np.arange(n_samples) * 60000,  # timestamp
                np.random.randn(n_samples).cumsum() + 100,  # open
                np.random.randn(n_samples).cumsum() + 100,  # close
                np.random.randn(n_samples).cumsum() + 101,  # high
                np.random.randn(n_samples).cumsum() + 99,  # low
                np.abs(np.random.randn(n_samples)) * 1000,  # volume
            ]
        )

    def test_pipeline_with_mocked_ssm(self, mock_calculator, sample_candles):
        """测试带 mock SSM 的流水线"""
        # 配置：请求 deep_ssm_0, deep_ssm_1, rsi
        config = PipelineConfig(
            feature_names=["deep_ssm_0", "deep_ssm_1", "rsi"],
            ssm_state_dim=3,
        )

        # 创建 mock SSM
        mock_ssm = MagicMock(spec=DeepSSMAdapter)
        mock_ssm.is_fitted = True
        mock_ssm.state_dim = 3
        mock_ssm.prefix = "deep_ssm"

        def mock_transform(df):
            return pd.DataFrame(
                np.random.randn(len(df), 3),
                index=df.index,
                columns=["deep_ssm_0", "deep_ssm_1", "deep_ssm_2"],
            )

        mock_ssm.transform.side_effect = mock_transform

        # 创建流水线
        pipeline = FeaturePipeline(
            config=config,
            raw_calculator=mock_calculator,
            ssm_processors={"deep_ssm": mock_ssm},
        )
        pipeline._is_fitted = True

        # 测试 transform
        result = pipeline.transform(sample_candles)

        assert isinstance(result, pd.DataFrame)
        # 只应该包含请求的特征
        assert list(result.columns) == ["deep_ssm_0", "deep_ssm_1", "rsi"]
        assert len(result.columns) == 3


class TestFeaturePipelineSaveLoad:
    """FeaturePipeline 持久化测试"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """创建测试数据"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        t = np.linspace(0, 4 * np.pi, n_samples)
        data = np.column_stack(
            [np.sin(t + i * np.pi / n_features) for i in range(n_features)]
        )
        data += np.random.randn(n_samples, n_features) * 0.1

        return pd.DataFrame(data, columns=[f"feat_{i}" for i in range(n_features)])

    def test_save_load_ssm_adapters(self, sample_data):
        """测试 SSM 适配器的保存和加载"""
        # 创建并训练 SSM
        deep_config = DeepSSMConfig(
            obs_dim=sample_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        deep_ssm = DeepSSMAdapter(config=deep_config)
        deep_ssm.fit(sample_data)

        lg_config = LGSSMConfig(
            obs_dim=sample_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        lg_ssm = LGSSMAdapter(config=lg_config)
        lg_ssm.fit(sample_data)

        # 创建配置 - 使用新的 feature_names 格式
        config = PipelineConfig(
            feature_names=["deep_ssm_0", "deep_ssm_1", "lg_ssm_0", "lg_ssm_1"],
            ssm_state_dim=3,
            # 覆盖默认的 ssm_input_features，使用测试数据的列名
            ssm_input_features=list(sample_data.columns),
        )

        # 创建流水线
        pipeline = FeaturePipeline(
            config=config,
            ssm_processors={"deep_ssm": deep_ssm, "lg_ssm": lg_ssm},
        )
        pipeline._is_fitted = True

        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存（使用新的 path + name 接口）
            pipeline.save(tmpdir, "test_pipeline")

            # 验证文件存在（在子目录中）
            assert (Path(tmpdir) / "test_pipeline" / "pipeline_config.json").exists()
            assert (Path(tmpdir) / "test_pipeline" / "deep_ssm.safetensors").exists()
            assert (Path(tmpdir) / "test_pipeline" / "lg_ssm.safetensors").exists()

            # 加载（使用新的 path + name 接口）
            loaded = FeaturePipeline.load(tmpdir, "test_pipeline")

            assert loaded.is_fitted
            assert "deep_ssm" in loaded.ssm_processors
            assert "lg_ssm" in loaded.ssm_processors
            assert loaded.ssm_processors["deep_ssm"].state_dim == 3
            assert loaded.ssm_processors["lg_ssm"].state_dim == 3


class TestFeaturePipelineSSMIntegration:
    """FeaturePipeline SSM 集成测试"""

    @pytest.fixture
    def candles(self) -> np.ndarray:
        """创建符合 Jesse 格式的 K 线数据"""
        return make_jesse_candles(100, seed=42)

    @pytest.fixture
    def feature_data(self) -> pd.DataFrame:
        """创建用于 SSM 训练的特征数据"""
        np.random.seed(42)
        n_samples = 100
        n_features = 3

        t = np.linspace(0, 4 * np.pi, n_samples)
        data = np.column_stack(
            [np.sin(t + i * np.pi / n_features) for i in range(n_features)]
        )
        data += np.random.randn(n_samples, n_features) * 0.1

        return pd.DataFrame(data, columns=[f"feat_{i}" for i in range(n_features)])

    def test_ssm_warmup(self, candles, feature_data):
        """测试 SSM 预热功能"""
        # 创建并训练 LG-SSM
        lg_config = LGSSMConfig(
            obs_dim=feature_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        lg_ssm = LGSSMAdapter(config=lg_config)
        lg_ssm.fit(feature_data)

        # 使用 mock calculator
        mock_calculator = MagicMock()
        mock_calculator.get.return_value = {
            col: feature_data[col].values for col in feature_data.columns
        }

        # 创建配置 - 请求 SSM 特征
        config = PipelineConfig(
            feature_names=["lg_ssm_0", "lg_ssm_1", "lg_ssm_2"],
            ssm_state_dim=3,
            ssm_input_features=list(feature_data.columns),
        )

        # 创建流水线
        pipeline = FeaturePipeline(
            config=config,
            raw_calculator=mock_calculator,
            ssm_processors={"lg_ssm": lg_ssm},
        )
        pipeline._is_fitted = True

        # 记录初始状态
        lg_ssm.reset_state()
        initial_state = lg_ssm._state.copy()

        # 预热（传入符合 Jesse 格式的 candles）
        pipeline.warmup_ssm(candles)

        # 状态应该已改变
        assert not np.allclose(lg_ssm._state, initial_state)

    def test_inference_vs_transform_consistency(self, feature_data):
        """测试单步推理与批量转换的一致性（SSM adapter 层面）"""
        # 创建并训练 LG-SSM
        lg_config = LGSSMConfig(
            obs_dim=feature_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        lg_ssm = LGSSMAdapter(config=lg_config)
        lg_ssm.fit(feature_data)

        # 批量转换结果
        batch_result = lg_ssm.transform(feature_data)

        # 重置并逐行推理
        lg_ssm.reset_state()
        inference_results = []
        for i in range(len(feature_data)):
            state = lg_ssm.inference(feature_data.iloc[i].values)
            inference_results.append(state.copy())

        inference_array = np.array(inference_results)

        # 比较
        np.testing.assert_array_almost_equal(
            inference_array,
            batch_result.values,
            decimal=5,
        )

    def test_warmup_then_inference_numerical_consistency(self, feature_data):
        """测试 warmup 后继续 inference 的数值一致性（SSM adapter 层面）

        验证：warmup(data[:n]) + inference(data[n:]) == transform(data) 的后半部分
        """
        warmup_len = 50

        # 创建并训练 LG-SSM
        lg_config = LGSSMConfig(
            obs_dim=feature_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        lg_ssm = LGSSMAdapter(config=lg_config)
        lg_ssm.fit(feature_data)

        # 方式1：完整 transform
        full_transform = lg_ssm.transform(feature_data)

        # 方式2：warmup + inference
        lg_ssm.reset_state()

        # warmup 阶段
        for i in range(warmup_len):
            lg_ssm.inference(feature_data.iloc[i].values)

        # inference 阶段
        inference_results = []
        for i in range(warmup_len, len(feature_data)):
            state = lg_ssm.inference(feature_data.iloc[i].values)
            inference_results.append(state.copy())

        inference_array = np.array(inference_results)

        # 比较后半部分
        np.testing.assert_array_almost_equal(
            inference_array,
            full_transform.values[warmup_len:],
            decimal=5,
            err_msg="warmup + inference should match transform results",
        )


class TestFeaturePipelineEndToEnd:
    """FeaturePipeline 端到端数值正确性测试"""

    @pytest.fixture
    def candles(self) -> np.ndarray:
        """创建符合 Jesse 格式的 K 线数据"""
        return make_jesse_candles(100, seed=42)

    @pytest.fixture
    def feature_data(self) -> pd.DataFrame:
        """创建用于 SSM 训练的特征数据"""
        np.random.seed(42)
        n_samples = 100
        n_features = 3

        t = np.linspace(0, 4 * np.pi, n_samples)
        data = np.column_stack(
            [np.sin(t + i * np.pi / n_features) for i in range(n_features)]
        )
        data += np.random.randn(n_samples, n_features) * 0.1

        return pd.DataFrame(data, columns=[f"feat_{i}" for i in range(n_features)])

    def test_pipeline_transform_vs_individual_ssm(self, candles, feature_data):
        """测试 Pipeline transform 与单独 SSM transform 的一致性"""
        # 创建并训练独立的 SSM
        deep_config = DeepSSMConfig(
            obs_dim=feature_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        deep_ssm = DeepSSMAdapter(config=deep_config)
        deep_ssm.fit(feature_data)

        lg_config = LGSSMConfig(
            obs_dim=feature_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        lg_ssm = LGSSMAdapter(config=lg_config)
        lg_ssm.fit(feature_data)

        # 获取独立 SSM 的 transform 结果
        deep_transform = deep_ssm.transform(feature_data)
        lg_transform = lg_ssm.transform(feature_data)

        # 使用 mock calculator 创建 Pipeline
        mock_calculator = MagicMock()
        mock_calculator.get.return_value = {
            col: feature_data[col].values for col in feature_data.columns
        }

        # 配置：请求所有 SSM 特征
        config = PipelineConfig(
            feature_names=[
                "deep_ssm_0", "deep_ssm_1", "deep_ssm_2",
                "lg_ssm_0", "lg_ssm_1", "lg_ssm_2",
            ],
            ssm_state_dim=3,
            ssm_input_features=list(feature_data.columns),
        )

        pipeline = FeaturePipeline(
            config=config,
            raw_calculator=mock_calculator,
            ssm_processors={"deep_ssm": deep_ssm, "lg_ssm": lg_ssm},
        )
        pipeline._is_fitted = True

        # Pipeline transform（传入符合 Jesse 格式的 candles）
        pipeline_result = pipeline.transform(candles)

        # 验证 SSM 特征一致
        for col in deep_transform.columns:
            np.testing.assert_array_almost_equal(
                pipeline_result[col].values,
                deep_transform[col].values,
                decimal=5,
                err_msg=f"DeepSSM column {col} mismatch",
            )

        for col in lg_transform.columns:
            np.testing.assert_array_almost_equal(
                pipeline_result[col].values,
                lg_transform[col].values,
                decimal=5,
                err_msg=f"LGSSM column {col} mismatch",
            )

    def test_pipeline_inference_vs_transform_consistency(self, candles, feature_data):
        """测试 Pipeline inference 与 transform 的数值一致性"""
        # 创建并训练 LG-SSM
        lg_config = LGSSMConfig(
            obs_dim=feature_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        lg_ssm = LGSSMAdapter(config=lg_config)
        lg_ssm.fit(feature_data)

        # 使用 mock calculator
        mock_calculator = MagicMock()

        def mock_get(feature_names):
            # 返回最后一行
            return {name: feature_data[name].values[-1:] for name in feature_names}

        mock_calculator.get.side_effect = mock_get

        config = PipelineConfig(
            feature_names=["lg_ssm_0", "lg_ssm_1", "lg_ssm_2"],
            ssm_state_dim=3,
            ssm_input_features=list(feature_data.columns),
        )

        pipeline = FeaturePipeline(
            config=config,
            raw_calculator=mock_calculator,
            ssm_processors={"lg_ssm": lg_ssm},
        )
        pipeline._is_fitted = True

        # 获取 transform 结果
        transform_result = lg_ssm.transform(feature_data)

        # 重置并逐行 inference
        lg_ssm.reset_state()
        for i in range(len(feature_data) - 1):
            lg_ssm.inference(feature_data.iloc[i].values)

        # 最后一步通过 Pipeline inference（传入符合 Jesse 格式的 candles）
        inference_result = pipeline.inference(candles)

        # 比较最后一行
        for col in transform_result.columns:
            np.testing.assert_array_almost_equal(
                inference_result[col].values,
                transform_result[col].values[-1:],
                decimal=5,
                err_msg=f"Pipeline inference vs transform mismatch for {col}",
            )

    def test_output_only_contains_requested_features(self, candles, feature_data):
        """测试输出只包含请求的特征"""
        # 创建并训练 SSM
        lg_config = LGSSMConfig(
            obs_dim=feature_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        lg_ssm = LGSSMAdapter(config=lg_config)
        lg_ssm.fit(feature_data)

        # 使用 mock calculator
        mock_calculator = MagicMock()
        mock_calculator.get.return_value = {
            col: feature_data[col].values for col in feature_data.columns
        }

        # 只请求部分特征
        config = PipelineConfig(
            feature_names=["lg_ssm_0", "lg_ssm_2"],  # 只要 0 和 2，不要 1
            ssm_state_dim=3,
            ssm_input_features=list(feature_data.columns),
        )

        pipeline = FeaturePipeline(
            config=config,
            raw_calculator=mock_calculator,
            ssm_processors={"lg_ssm": lg_ssm},
        )
        pipeline._is_fitted = True

        # Transform（传入符合 Jesse 格式的 candles）
        result = pipeline.transform(candles)

        # 只应该包含请求的特征
        assert list(result.columns) == ["lg_ssm_0", "lg_ssm_2"]
        assert "lg_ssm_1" not in result.columns


class TestFeaturePipelineDimensionReduction:
    """FeaturePipeline 降维功能测试"""

    @pytest.fixture
    def candles(self) -> np.ndarray:
        """创建符合 Jesse 格式的 K 线数据"""
        return make_jesse_candles(100, seed=42)

    @pytest.fixture
    def feature_data(self) -> pd.DataFrame:
        """创建用于 SSM 训练的特征数据"""
        np.random.seed(42)
        n_samples = 100
        n_features = 3

        t = np.linspace(0, 4 * np.pi, n_samples)
        data = np.column_stack(
            [np.sin(t + i * np.pi / n_features) for i in range(n_features)]
        )
        data += np.random.randn(n_samples, n_features) * 0.1

        return pd.DataFrame(data, columns=[f"feat_{i}" for i in range(n_features)])

    def test_inference_applies_dimension_reducer(self, candles, feature_data):
        """测试 inference 方法正确应用降维器"""
        import torch

        from src.features.dimensionality_reduction import DimensionReducerProtocol

        # 确保测试隔离性 - 重置随机种子
        np.random.seed(42)
        torch.manual_seed(42)

        # 创建并训练 LG-SSM
        lg_config = LGSSMConfig(
            obs_dim=feature_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        lg_ssm = LGSSMAdapter(config=lg_config)
        lg_ssm.fit(feature_data)

        # 创建 mock dimension reducer
        mock_reducer = MagicMock(spec=DimensionReducerProtocol)
        mock_reducer.is_fitted = True
        mock_reducer.n_components = 2

        def mock_transform(df):
            # 简单的降维：返回固定列名
            return pd.DataFrame(
                df.iloc[:, :2].values,
                index=df.index,
                columns=["reduced_0", "reduced_1"],
            )

        mock_reducer.transform.side_effect = mock_transform

        # 使用 mock calculator
        mock_calculator = MagicMock()

        def mock_get(feature_names):
            return {name: feature_data[name].values[-1:] for name in feature_names}

        mock_calculator.get.side_effect = mock_get

        # 创建配置（启用降维）
        # 注意：feature_names 指定降维前的特征，降维后输出由 dimension_reducer 决定
        config = PipelineConfig(
            feature_names=["lg_ssm_0", "lg_ssm_1", "lg_ssm_2"],  # SSM 特征，将被降维
            ssm_state_dim=3,
            ssm_input_features=list(feature_data.columns),
            use_dimension_reducer=True,
        )

        # 创建流水线
        pipeline = FeaturePipeline(
            config=config,
            raw_calculator=mock_calculator,
            ssm_processors={"lg_ssm": lg_ssm},
            dimension_reducer=mock_reducer,
        )
        pipeline._is_fitted = True

        # warmup SSM
        lg_ssm.reset_state()
        for i in range(len(feature_data) - 1):
            lg_ssm.inference(feature_data.iloc[i].values)

        # 调用 inference（传入符合 Jesse 格式的 candles）
        result = pipeline.inference(candles)

        # 验证 dimension reducer 被调用
        assert mock_reducer.transform.called, "Dimension reducer should be called in inference"

        # 验证结果列名
        assert "reduced_0" in result.columns
        assert "reduced_1" in result.columns
        assert len(result.columns) == 2

    def test_inference_vs_transform_with_dimension_reducer(self, candles, feature_data):
        """测试启用降维时 inference 与 transform 都正确应用降维器且输出格式一致"""
        from src.features.dimensionality_reduction import DimensionReducerProtocol

        # 使用确定性 mock SSM
        mock_ssm = MagicMock()
        mock_ssm.is_fitted = True
        mock_ssm.prefix = "lg_ssm"

        def mock_ssm_transform(df):
            """批量处理：返回固定值"""
            n_rows = len(df)
            return pd.DataFrame(
                {
                    "lg_ssm_0": np.ones(n_rows),
                    "lg_ssm_1": np.ones(n_rows) * 2,
                    "lg_ssm_2": np.ones(n_rows) * 3,
                },
                index=df.index,
            )

        def mock_ssm_inference(obs):
            """单步推理：返回固定值"""
            return np.array([1.0, 2.0, 3.0])

        mock_ssm.transform.side_effect = mock_ssm_transform
        mock_ssm.inference.side_effect = mock_ssm_inference

        # 创建 mock dimension reducer
        mock_reducer = MagicMock(spec=DimensionReducerProtocol)
        mock_reducer.is_fitted = True
        mock_reducer.n_components = 2

        def deterministic_transform(df):
            # 简单的确定性变换：保证相同输入产生相同输出
            col1 = df.iloc[:, 0].values * 10
            col2 = df.iloc[:, 1].values * 20
            return pd.DataFrame(
                {"reduced_0": col1, "reduced_1": col2},
                index=df.index,
            )

        mock_reducer.transform.side_effect = deterministic_transform

        # Transform 用
        mock_calculator_transform = MagicMock()
        mock_calculator_transform.get.return_value = {
            col: feature_data[col].values for col in feature_data.columns
        }

        # Inference 用
        mock_calculator_inference = MagicMock()

        def mock_get_last(feature_names):
            return {name: feature_data[name].values[-1:] for name in feature_names}

        mock_calculator_inference.get.side_effect = mock_get_last

        # 配置
        config = PipelineConfig(
            feature_names=["lg_ssm_0", "lg_ssm_1", "lg_ssm_2"],
            ssm_state_dim=3,
            ssm_input_features=list(feature_data.columns),
            use_dimension_reducer=True,
        )

        # ===== Transform 流程 =====
        pipeline_transform = FeaturePipeline(
            config=config,
            raw_calculator=mock_calculator_transform,
            ssm_processors={"lg_ssm": mock_ssm},
            dimension_reducer=mock_reducer,
        )
        pipeline_transform._is_fitted = True

        # 传入符合 Jesse 格式的 candles
        transform_result = pipeline_transform.transform(candles)

        # 验证 transform 调用了降维器且输出正确列名
        assert mock_reducer.transform.called
        assert "reduced_0" in transform_result.columns
        assert "reduced_1" in transform_result.columns

        # ===== Inference 流程 =====
        pipeline_inference = FeaturePipeline(
            config=config,
            raw_calculator=mock_calculator_inference,
            ssm_processors={"lg_ssm": mock_ssm},
            dimension_reducer=mock_reducer,
        )
        pipeline_inference._is_fitted = True

        # 传入符合 Jesse 格式的 candles
        inference_result = pipeline_inference.inference(candles)

        # 验证 inference 也调用了降维器且输出相同列名
        assert mock_reducer.transform.call_count >= 2
        assert "reduced_0" in inference_result.columns
        assert "reduced_1" in inference_result.columns

        # 验证 transform 和 inference 输出列名一致
        assert list(transform_result.columns) == list(inference_result.columns)


class TestFeaturePipelineNoSSM:
    """FeaturePipeline 无 SSM 配置场景测试

    验证当 config.ssm_types 为空时，Pipeline 各方法能正常工作不报错。
    """

    @pytest.fixture
    def candles(self) -> np.ndarray:
        """创建符合 Jesse 格式的 K 线数据"""
        return make_jesse_candles(100, seed=42)

    @pytest.fixture
    def raw_only_config(self) -> PipelineConfig:
        """只有原始特征的配置（无 SSM）"""
        return PipelineConfig(
            feature_names=["rsi", "macd", "atr"]
        )

    def test_config_no_ssm(self, raw_only_config):
        """验证无 SSM 配置正确解析"""
        assert raw_only_config.ssm_types == []
        assert raw_only_config.ssm_feature_names == []
        assert raw_only_config.raw_feature_names == ["rsi", "macd", "atr"]

    def test_fit_no_ssm(self, candles, raw_only_config):
        """验证无 SSM 时 fit 正常工作"""
        # 使用 mock calculator
        mock_calculator = MagicMock()
        mock_calculator.get.return_value = {
            "rsi": np.random.rand(len(candles)),
            "macd": np.random.rand(len(candles)),
            "atr": np.random.rand(len(candles)),
        }

        pipeline = FeaturePipeline(
            config=raw_only_config,
            raw_calculator=mock_calculator,
        )

        # fit 应该正常完成，不报错
        pipeline.fit(candles)

        assert pipeline.is_fitted
        assert len(pipeline.ssm_processors) == 0

    def test_fit_transform_no_ssm(self, candles, raw_only_config):
        """验证无 SSM 时 fit_transform 正常工作"""
        mock_calculator = MagicMock()
        mock_calculator.get.return_value = {
            "rsi": np.random.rand(len(candles)),
            "macd": np.random.rand(len(candles)),
            "atr": np.random.rand(len(candles)),
        }

        pipeline = FeaturePipeline(
            config=raw_only_config,
            raw_calculator=mock_calculator,
        )

        # fit_transform 应该正常完成
        result = pipeline.fit_transform(candles)

        assert pipeline.is_fitted
        assert list(result.columns) == ["rsi", "macd", "atr"]
        assert len(result) == len(candles)

    def test_transform_no_ssm(self, candles, raw_only_config):
        """验证无 SSM 时 transform 正常工作"""
        mock_calculator = MagicMock()
        mock_calculator.get.return_value = {
            "rsi": np.random.rand(len(candles)),
            "macd": np.random.rand(len(candles)),
            "atr": np.random.rand(len(candles)),
        }

        pipeline = FeaturePipeline(
            config=raw_only_config,
            raw_calculator=mock_calculator,
        )
        pipeline._is_fitted = True  # 模拟已训练

        # transform 应该正常完成
        result = pipeline.transform(candles)

        assert list(result.columns) == ["rsi", "macd", "atr"]
        assert len(result) == len(candles)

    def test_inference_no_ssm(self, candles, raw_only_config):
        """验证无 SSM 时 inference 正常工作"""
        mock_calculator = MagicMock()

        def mock_get(feature_names):
            return {name: np.random.rand(1) for name in feature_names}

        mock_calculator.get.side_effect = mock_get

        pipeline = FeaturePipeline(
            config=raw_only_config,
            raw_calculator=mock_calculator,
        )
        pipeline._is_fitted = True

        # inference 应该正常完成
        result = pipeline.inference(candles)

        assert list(result.columns) == ["rsi", "macd", "atr"]
        assert len(result) == 1

    def test_warmup_ssm_no_ssm(self, candles, raw_only_config):
        """验证无 SSM 时 warmup_ssm 直接返回不报错"""
        mock_calculator = MagicMock()

        pipeline = FeaturePipeline(
            config=raw_only_config,
            raw_calculator=mock_calculator,
        )
        pipeline._is_fitted = True

        # warmup_ssm 应该直接返回，不调用 calculator
        pipeline.warmup_ssm(candles)

        # mock_calculator 不应该被调用
        assert not mock_calculator.load.called
        assert not mock_calculator.get.called

    def test_reset_ssm_states_no_ssm(self, raw_only_config):
        """验证无 SSM 时 reset_ssm_states 不报错"""
        pipeline = FeaturePipeline(config=raw_only_config)
        pipeline._is_fitted = True

        # 应该不报错
        pipeline.reset_ssm_states()


class TestSimpleFeatureCalculatorCachePreservation:
    """测试 SimpleFeatureCalculator 的缓存保留逻辑"""

    def test_load_same_candles_reference_preserves_cache(self):
        """同一 candles 引用应保留缓存"""
        from src.features.simple_feature_calculator import SimpleFeatureCalculator

        calc = SimpleFeatureCalculator(verbose=False)
        candles = np.random.rand(100, 6)

        # 第一次 load 并计算特征
        calc.load(candles, sequential=True)
        calc.get(["fisher"])  # 使用注册表中存在的特征

        # 验证缓存已建立
        assert len(calc.cache) > 0
        cache_before = dict(calc.cache)

        # 再次 load 同一引用
        calc.load(candles, sequential=True)

        # 缓存应该被保留
        assert calc.cache == cache_before

    def test_load_same_candles_content_preserves_cache(self):
        """相同内容的不同对象应保留缓存"""
        from src.features.simple_feature_calculator import SimpleFeatureCalculator

        calc = SimpleFeatureCalculator(verbose=False)
        candles = np.random.rand(100, 6)

        # 第一次 load 并计算特征
        calc.load(candles, sequential=True)
        calc.get(["fisher"])  # 使用注册表中存在的特征

        # 验证缓存已建立
        assert len(calc.cache) > 0
        cache_before = dict(calc.cache)

        # 创建内容相同的新数组
        candles_copy = candles.copy()
        assert candles is not candles_copy  # 确认是不同对象
        assert np.array_equal(candles, candles_copy)  # 确认内容相同

        # load 新对象
        calc.load(candles_copy, sequential=True)

        # 缓存应该被保留
        assert calc.cache == cache_before

    def test_load_different_candles_clears_cache(self):
        """不同内容的 candles 应清空缓存"""
        from src.features.simple_feature_calculator import SimpleFeatureCalculator

        calc = SimpleFeatureCalculator(verbose=False)
        candles1 = np.random.rand(100, 6)

        # 第一次 load 并计算特征
        calc.load(candles1, sequential=True)
        calc.get(["fisher"])  # 使用注册表中存在的特征

        # 验证缓存已建立
        assert len(calc.cache) > 0

        # load 不同数据
        candles2 = np.random.rand(100, 6)
        calc.load(candles2, sequential=True)

        # 缓存应该被清空
        assert len(calc.cache) == 0

    def test_load_sequential_true_can_serve_false(self):
        """sequential=True 的缓存可以服务 sequential=False"""
        from src.features.simple_feature_calculator import SimpleFeatureCalculator

        calc = SimpleFeatureCalculator(verbose=False)
        candles = np.random.rand(100, 6)

        # 以 sequential=True 加载
        calc.load(candles, sequential=True)
        calc.get(["fisher"])  # 使用注册表中存在的特征

        cache_before = dict(calc.cache)

        # 以 sequential=False 再次加载同一数据
        calc.load(candles, sequential=False)

        # 缓存应该被保留
        assert calc.cache == cache_before


class TestShareRawCalculator:
    """测试 share_raw_calculator_from 方法"""

    @pytest.fixture
    def raw_only_config(self):
        """只有原始特征的配置（无 SSM）"""
        return PipelineConfig(
            feature_names=["rsi", "macd", "atr"],
            verbose=False,
        )

    @pytest.fixture
    def raw_only_config_subset(self):
        """原始特征子集的配置"""
        return PipelineConfig(
            feature_names=["rsi", "macd"],
            verbose=False,
        )

    def test_share_raw_calculator_from_basic(self, raw_only_config, raw_only_config_subset):
        """基本共享功能测试"""
        source_pipeline = FeaturePipeline(config=raw_only_config)
        target_pipeline = FeaturePipeline(config=raw_only_config_subset)

        # 共享前是不同的 calculator
        assert source_pipeline._raw_calculator is not target_pipeline._raw_calculator

        # 执行共享
        result = target_pipeline.share_raw_calculator_from(source_pipeline)

        # 验证返回 self（支持链式调用）
        assert result is target_pipeline

        # 验证是同一个 calculator
        assert source_pipeline._raw_calculator is target_pipeline._raw_calculator

    def test_share_raw_calculator_chain_call(self, raw_only_config, raw_only_config_subset):
        """链式调用测试"""
        source_pipeline = FeaturePipeline(config=raw_only_config)
        target_pipeline = FeaturePipeline(config=raw_only_config_subset)

        # 链式调用
        target_pipeline.share_raw_calculator_from(source_pipeline)

        # 验证共享成功
        assert source_pipeline._raw_calculator is target_pipeline._raw_calculator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
