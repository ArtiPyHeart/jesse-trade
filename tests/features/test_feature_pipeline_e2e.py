"""
FeaturePipeline 端到端集成测试

验证训练、批量转换、实时推理的数值一致性。

测试层级:
1. 不降维一致性 (fit_transform vs transform vs warmup+inference)
2. 降维一致性
3. 保存加载 + 批量推理
4. 保存加载 + 实时推理
5. 边界条件

特点：
- 使用真实 SimpleFeatureCalculator 特征（排除 cwt/vmd）
- 同时测试一阶特征（原始特征）和二阶特征（SSM 特征）
- fracdiff 特征需要约 372+ 行历史数据才能产生有效值
"""

import random
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

from src.features.pipeline import FeaturePipeline, PipelineConfig
from src.features.ssm import DeepSSMAdapter, LGSSMAdapter
from src.models.deep_ssm import DeepSSMConfig
from src.models.lgssm import LGSSMConfig


# ============== 常量定义 ==============

# 一阶特征（快速计算，排除 cwt/vmd）
FIRST_ORDER_FEATURES = [
    "adx_14",
    "natr",
    "stc",
    "williams_r",
    "roofing_filter",
]

# 二阶特征（SSM 输出）
SECOND_ORDER_FEATURES = [
    "deep_ssm_0",
    "deep_ssm_1",
    "deep_ssm_2",
    "lg_ssm_0",
    "lg_ssm_1",
    "lg_ssm_2",
]

# 混合特征（一阶 + 二阶）
MIXED_FEATURES = FIRST_ORDER_FEATURES + SECOND_ORDER_FEATURES

# 数值容差
# DeepSSM 由于神经网络的数值不稳定性，需要较宽松的容差
PIPELINE_RTOL = 1e-2  # 1% 相对误差
PIPELINE_ATOL = 1e-4


# ============== 工具函数 ==============


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

    return np.column_stack(
        [timestamps, open_prices, close_prices, high_prices, low_prices, volumes]
    )


def seed_everything(seed: int = 42) -> None:
    """统一设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_valid_data_range(df: pd.DataFrame) -> tuple:
    """
    获取 DataFrame 中有效数据的范围（跳过开头的 NaN）

    Args:
        df: 待检查的 DataFrame

    Returns:
        (first_valid_row, last_row) 元组
    """
    valid_rows = ~df.isna().any(axis=1)
    valid_indices = valid_rows[valid_rows].index.tolist()
    if len(valid_indices) == 0:
        return len(df), len(df)
    return valid_indices[0], len(df)


def create_pipeline_with_ssm(
    config: PipelineConfig,
    lgssm_config: LGSSMConfig,
    deepssm_config: DeepSSMConfig,
) -> FeaturePipeline:
    """创建带有 SSM 处理器的 Pipeline

    按 config.ssm_types 顺序创建 SSM 处理器，确保顺序一致性。
    """
    ssm_processors = {}

    # 按 config.ssm_types 顺序创建，确保与 inference/transform 列顺序一致
    for ssm_type in config.ssm_types:
        if ssm_type == "deep_ssm":
            ssm_processors["deep_ssm"] = DeepSSMAdapter(
                config=deepssm_config, prefix="deep_ssm"
            )
        elif ssm_type == "lg_ssm":
            ssm_processors["lg_ssm"] = LGSSMAdapter(
                config=lgssm_config, prefix="lg_ssm"
            )

    return FeaturePipeline(
        config=config,
        ssm_processors=ssm_processors,
    )


# ============== Fixtures ==============


@pytest.fixture
def realistic_candles():
    """生成 800 行 K 线数据（fracdiff 需要约 372+ 行才有有效值）"""
    return make_jesse_candles(800, seed=42)


@pytest.fixture
def lgssm_config():
    """LGSSM 配置（80 维 fracdiff 输入）"""
    return LGSSMConfig(obs_dim=80, state_dim=5)


@pytest.fixture
def deepssm_config():
    """DeepSSM 配置（简化以加速测试）"""
    return DeepSSMConfig(
        obs_dim=80,
        state_dim=5,
        lstm_hidden=16,
        max_epochs=3,
    )


@pytest.fixture
def config_no_reduction():
    """不降维配置"""
    return PipelineConfig(
        feature_names=MIXED_FEATURES.copy(),
        ssm_state_dim=5,
        use_dimension_reducer=False,
    )


@pytest.fixture
def config_with_reduction():
    """降维配置"""
    return PipelineConfig(
        feature_names=MIXED_FEATURES.copy(),
        ssm_state_dim=5,
        use_dimension_reducer=True,
        dimension_reducer_type="ard_vae",
        dimension_reducer_config={
            "max_latent_dim": 32,
            "max_epochs": 20,
            "patience": 5,
            "seed": 42,
        },
    )


# ============== 测试类 ==============


class TestNoReductionConsistency:
    """不降维场景测试"""

    def test_fit_transform_equals_transform_no_reduction(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证不降维时 fit_transform 和 transform 输出完全一致

        这是最基本的一致性要求：训练时计算的特征 == 训练后重新计算的特征
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        # fit_transform
        result1 = pipeline.fit_transform(realistic_candles)

        # transform
        result2 = pipeline.transform(realistic_candles)

        # 验证输出行数 = candles 行数
        assert len(result1) == len(realistic_candles)
        assert len(result2) == len(realistic_candles)

        # 获取有效数据范围
        first_valid, _ = get_valid_data_range(result1)
        assert first_valid < len(realistic_candles), "Should have some valid data"

        # 比较有效数据部分
        np.testing.assert_allclose(
            result1.iloc[first_valid:].values,
            result2.iloc[first_valid:].values,
            rtol=PIPELINE_RTOL,
            atol=PIPELINE_ATOL,
            err_msg="fit_transform vs transform should produce identical results",
        )

    def test_transform_equals_warmup_inference_no_reduction(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证不降维时 transform 结果 == warmup + 逐行 inference 结果

        这是实盘场景的核心一致性：历史预热 + 流式推理 == 批量处理
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        pipeline.fit_transform(realistic_candles)

        # Ground truth: transform
        transform_result = pipeline.transform(realistic_candles)
        first_valid, _ = get_valid_data_range(transform_result)

        # warmup 长度需要在有效数据之后
        warmup_len = first_valid + 20
        assert warmup_len < len(realistic_candles) - 50, "Not enough data for test"

        # warmup + inference
        pipeline.warmup_ssm(realistic_candles[:warmup_len])

        inference_results = []
        for i in range(warmup_len, len(realistic_candles)):
            result = pipeline.inference(realistic_candles[: i + 1])
            inference_results.append(result.iloc[0].values)

        inference_array = np.array(inference_results)
        expected = transform_result.iloc[warmup_len:].values

        np.testing.assert_allclose(
            inference_array,
            expected,
            rtol=PIPELINE_RTOL,
            atol=PIPELINE_ATOL,
            err_msg="warmup + inference should match transform",
        )

    def test_output_columns_match_config(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证不降维时输出列名与 config.feature_names 完全一致
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        result = pipeline.fit_transform(realistic_candles)

        # 列名应与配置完全一致
        assert list(result.columns) == config_no_reduction.feature_names

    def test_output_columns_consistent_across_methods(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证 fit_transform / transform / inference 三个方法的输出列名和顺序完全一致

        这是列一致性的核心测试：
        - fit_transform 用于训练
        - transform 用于批量推理
        - inference 用于实时推理
        三者的输出格式必须完全一致，否则下游模型会出错
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        # 1. fit_transform 输出
        fit_transform_result = pipeline.fit_transform(realistic_candles)
        fit_transform_columns = list(fit_transform_result.columns)

        # 2. transform 输出
        transform_result = pipeline.transform(realistic_candles)
        transform_columns = list(transform_result.columns)

        # 3. inference 输出（需要先 warmup）
        first_valid, _ = get_valid_data_range(fit_transform_result)
        warmup_len = first_valid + 30
        pipeline.warmup_ssm(realistic_candles[:warmup_len])
        inference_result = pipeline.inference(realistic_candles[: warmup_len + 1])
        inference_columns = list(inference_result.columns)

        # 验证三者列名完全一致（包括顺序）
        assert fit_transform_columns == transform_columns, (
            f"fit_transform vs transform columns mismatch:\n"
            f"  fit_transform: {fit_transform_columns}\n"
            f"  transform: {transform_columns}"
        )
        assert transform_columns == inference_columns, (
            f"transform vs inference columns mismatch:\n"
            f"  transform: {transform_columns}\n"
            f"  inference: {inference_columns}"
        )

        # 验证与配置一致
        assert fit_transform_columns == config_no_reduction.feature_names, (
            f"Output columns don't match config.feature_names:\n"
            f"  output: {fit_transform_columns}\n"
            f"  config: {config_no_reduction.feature_names}"
        )

    def test_output_row_count_equals_candles(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证 fit_transform / transform 的输出行数与 candles 行数完全一致

        这是行数一致性的核心测试：输出 DataFrame 行数必须等于输入 candles 行数
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        # fit_transform 输出行数
        fit_transform_result = pipeline.fit_transform(realistic_candles)
        assert len(fit_transform_result) == len(realistic_candles), (
            f"fit_transform row count mismatch: "
            f"output={len(fit_transform_result)}, candles={len(realistic_candles)}"
        )

        # transform 输出行数
        transform_result = pipeline.transform(realistic_candles)
        assert len(transform_result) == len(realistic_candles), (
            f"transform row count mismatch: "
            f"output={len(transform_result)}, candles={len(realistic_candles)}"
        )

        # inference 输出行数（应为 1）
        first_valid, _ = get_valid_data_range(fit_transform_result)
        warmup_len = first_valid + 30
        pipeline.warmup_ssm(realistic_candles[:warmup_len])
        inference_result = pipeline.inference(realistic_candles[: warmup_len + 1])
        assert len(inference_result) == 1, (
            f"inference row count should be 1, got {len(inference_result)}"
        )

    def test_first_second_order_features_coexist(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证一阶特征和二阶特征同时正确计算

        一阶特征（原始特征）和二阶特征（SSM 特征）应该都出现在输出中
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        result = pipeline.fit_transform(realistic_candles)

        # 验证一阶特征存在
        for feat in FIRST_ORDER_FEATURES:
            assert feat in result.columns, f"Missing first-order feature: {feat}"

        # 验证二阶特征存在
        for feat in SECOND_ORDER_FEATURES:
            assert feat in result.columns, f"Missing second-order feature: {feat}"

        # 获取有效数据范围
        first_valid, _ = get_valid_data_range(result)
        valid_result = result.iloc[first_valid:]

        # 有效数据部分应无 NaN
        assert not valid_result.isna().any().any(), "Valid data should have no NaN"

        # 值应在合理范围内
        assert np.isfinite(valid_result.values).all(), "All values should be finite"


class TestWithReductionConsistency:
    """降维场景测试"""

    def test_fit_transform_equals_transform_with_reduction(
        self,
        realistic_candles,
        config_with_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证降维时 fit_transform 和 transform 输出完全一致
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_with_reduction, lgssm_config, deepssm_config
        )

        # fit_transform
        result1 = pipeline.fit_transform(realistic_candles)

        # transform
        result2 = pipeline.transform(realistic_candles)

        # 验证输出行数
        assert len(result1) == len(realistic_candles)
        assert len(result2) == len(realistic_candles)

        # 获取有效数据范围
        first_valid, _ = get_valid_data_range(result1)

        # 比较有效数据部分
        np.testing.assert_allclose(
            result1.iloc[first_valid:].values,
            result2.iloc[first_valid:].values,
            rtol=PIPELINE_RTOL,
            atol=PIPELINE_ATOL,
            err_msg="fit_transform vs transform should match with reduction",
        )

    def test_transform_equals_warmup_inference_with_reduction(
        self,
        realistic_candles,
        config_with_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证降维时 transform 结果 == warmup + 逐行 inference 结果
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_with_reduction, lgssm_config, deepssm_config
        )

        pipeline.fit_transform(realistic_candles)

        # Ground truth
        transform_result = pipeline.transform(realistic_candles)
        first_valid, _ = get_valid_data_range(transform_result)

        warmup_len = first_valid + 20
        assert warmup_len < len(realistic_candles) - 50

        # warmup + inference
        pipeline.warmup_ssm(realistic_candles[:warmup_len])

        inference_results = []
        for i in range(warmup_len, len(realistic_candles)):
            result = pipeline.inference(realistic_candles[: i + 1])
            inference_results.append(result.iloc[0].values)

        inference_array = np.array(inference_results)
        expected = transform_result.iloc[warmup_len:].values

        np.testing.assert_allclose(
            inference_array,
            expected,
            rtol=PIPELINE_RTOL,
            atol=PIPELINE_ATOL,
            err_msg="warmup + inference should match transform with reduction",
        )

    def test_output_columns_are_numeric(
        self,
        realistic_candles,
        config_with_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证降维后输出列名为数字字符串 "0", "1", "2", ...
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_with_reduction, lgssm_config, deepssm_config
        )

        result = pipeline.fit_transform(realistic_candles)

        # 列名应为数字字符串
        expected_cols = [str(i) for i in range(len(result.columns))]
        assert list(result.columns) == expected_cols, (
            f"Columns should be numeric strings, got {list(result.columns)}"
        )

    def test_dimension_reduction_actually_reduces(
        self,
        realistic_candles,
        config_with_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证降维后输出列数 <= max_latent_dim

        注意：ARDVAE 在所有可用特征上训练（SSM 输出 + calculator 特征），
        而不仅仅是 config.feature_names。因此降维后列数可能多于 feature_names，
        但应该 <= max_latent_dim。
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_with_reduction, lgssm_config, deepssm_config
        )

        result = pipeline.fit_transform(realistic_candles)

        # 降维后列数应 <= max_latent_dim
        max_latent_dim = config_with_reduction.dimension_reducer_config[
            "max_latent_dim"
        ]
        n_output_features = len(result.columns)

        assert n_output_features <= max_latent_dim, (
            f"Output should have <= {max_latent_dim} features, got {n_output_features}"
        )

        # 列名应为数字字符串
        expected_cols = [str(i) for i in range(n_output_features)]
        assert list(result.columns) == expected_cols

    def test_output_columns_consistent_across_methods_with_reduction(
        self,
        realistic_candles,
        config_with_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证降维时 fit_transform / transform / inference 三个方法的输出列名和顺序完全一致
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_with_reduction, lgssm_config, deepssm_config
        )

        # 1. fit_transform 输出
        fit_transform_result = pipeline.fit_transform(realistic_candles)
        fit_transform_columns = list(fit_transform_result.columns)

        # 2. transform 输出
        transform_result = pipeline.transform(realistic_candles)
        transform_columns = list(transform_result.columns)

        # 3. inference 输出（需要先 warmup）
        first_valid, _ = get_valid_data_range(fit_transform_result)
        warmup_len = first_valid + 30
        pipeline.warmup_ssm(realistic_candles[:warmup_len])
        inference_result = pipeline.inference(realistic_candles[: warmup_len + 1])
        inference_columns = list(inference_result.columns)

        # 验证三者列名完全一致（包括顺序）
        assert fit_transform_columns == transform_columns, (
            f"fit_transform vs transform columns mismatch:\n"
            f"  fit_transform: {fit_transform_columns}\n"
            f"  transform: {transform_columns}"
        )
        assert transform_columns == inference_columns, (
            f"transform vs inference columns mismatch:\n"
            f"  transform: {transform_columns}\n"
            f"  inference: {inference_columns}"
        )


class TestSaveLoadBatchTransform:
    """保存加载 + 批量推理测试"""

    def test_save_load_transform_no_reduction(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证不降维时 save → load → transform 结果不变
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        original = pipeline.fit_transform(realistic_candles)
        first_valid, _ = get_valid_data_range(original)

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save(tmpdir, "e2e_test")
            loaded = FeaturePipeline.load(tmpdir, "e2e_test")

            restored = loaded.transform(realistic_candles)

            np.testing.assert_allclose(
                original.iloc[first_valid:].values,
                restored.iloc[first_valid:].values,
                rtol=PIPELINE_RTOL,
                atol=PIPELINE_ATOL,
                err_msg="Transform after save/load should match original",
            )

    def test_save_load_transform_with_reduction(
        self,
        realistic_candles,
        config_with_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证降维时 save → load → transform 结果不变
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_with_reduction, lgssm_config, deepssm_config
        )

        original = pipeline.fit_transform(realistic_candles)
        first_valid, _ = get_valid_data_range(original)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存（降维器自动保存）
            pipeline.save(tmpdir, "e2e_test_reduction")

            # 加载（降维器自动加载）
            loaded = FeaturePipeline.load(tmpdir, "e2e_test_reduction")

            restored = loaded.transform(realistic_candles)

            # 列数应一致
            assert len(original.columns) == len(restored.columns)

            np.testing.assert_allclose(
                original.iloc[first_valid:].values,
                restored.iloc[first_valid:].values,
                rtol=PIPELINE_RTOL,
                atol=PIPELINE_ATOL,
                err_msg="Transform after save/load should match with reduction",
            )


class TestSaveLoadRealtimeInference:
    """保存加载 + 实时推理测试"""

    def test_save_load_warmup_inference_no_reduction(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证不降维时 save → load → warmup → inference 结果不变
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        pipeline.fit_transform(realistic_candles)

        transform_result = pipeline.transform(realistic_candles)
        first_valid, _ = get_valid_data_range(transform_result)

        warmup_len = first_valid + 30
        assert warmup_len < len(realistic_candles)

        # 原始 pipeline: warmup + inference
        pipeline.warmup_ssm(realistic_candles[:warmup_len])
        original_inference = (
            pipeline.inference(realistic_candles[: warmup_len + 1]).iloc[0].values
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save(tmpdir, "e2e_inference_test")
            loaded = FeaturePipeline.load(tmpdir, "e2e_inference_test")

            # 加载后: warmup + inference
            loaded.warmup_ssm(realistic_candles[:warmup_len])
            restored_inference = (
                loaded.inference(realistic_candles[: warmup_len + 1]).iloc[0].values
            )

            np.testing.assert_allclose(
                original_inference,
                restored_inference,
                rtol=PIPELINE_RTOL,
                atol=PIPELINE_ATOL,
                err_msg="Inference after save/load should match original",
            )

    def test_save_load_warmup_inference_with_reduction(
        self,
        realistic_candles,
        config_with_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证降维时 save → load → warmup → inference 结果不变
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_with_reduction, lgssm_config, deepssm_config
        )

        pipeline.fit_transform(realistic_candles)

        transform_result = pipeline.transform(realistic_candles)
        first_valid, _ = get_valid_data_range(transform_result)

        warmup_len = first_valid + 30
        assert warmup_len < len(realistic_candles)

        # 原始 pipeline
        pipeline.warmup_ssm(realistic_candles[:warmup_len])
        original_inference = (
            pipeline.inference(realistic_candles[: warmup_len + 1]).iloc[0].values
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存（降维器自动保存）
            pipeline.save(tmpdir, "e2e_inference_reduction")

            # 加载（降维器自动加载）
            loaded = FeaturePipeline.load(tmpdir, "e2e_inference_reduction")

            # 加载后
            loaded.warmup_ssm(realistic_candles[:warmup_len])
            restored_inference = (
                loaded.inference(realistic_candles[: warmup_len + 1]).iloc[0].values
            )

            np.testing.assert_allclose(
                original_inference,
                restored_inference,
                rtol=PIPELINE_RTOL,
                atol=PIPELINE_ATOL,
                err_msg="Inference after save/load should match with reduction",
            )


class TestEdgeCases:
    """边界条件测试"""

    def test_nan_handling_warmup_period(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证开头 NaN（fracdiff warmup 期）正确处理

        输出行数应等于 candles 行数，开头是 NaN，有效数据后无 NaN
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        result = pipeline.fit_transform(realistic_candles)

        # 输出行数应等于 candles 行数
        assert len(result) == len(realistic_candles)

        # 找到有效数据范围
        first_valid, _ = get_valid_data_range(result)

        # 应该有一些 NaN 行（fracdiff warmup）
        assert first_valid > 0, "Should have some NaN warmup rows"

        # 有效数据后不应有 NaN
        valid_result = result.iloc[first_valid:]
        nan_cols = valid_result.columns[valid_result.isna().any()].tolist()
        assert len(nan_cols) == 0, f"Intermediate NaN in columns: {nan_cols}"

    def test_inference_nan_zero_tolerance(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证 inference() 对 NaN 零容忍

        如果传入的 candles 太短导致特征有 NaN，应该立即报错
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        pipeline.fit_transform(realistic_candles)

        # 使用太短的历史数据应该触发 NaN 错误
        with pytest.raises(ValueError, match="inference.*received NaN"):
            pipeline.inference(realistic_candles[:50])

    def test_mixed_ssm_types(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证 DeepSSM 和 LGSSM 同时工作

        两种 SSM 应该都能正确计算并输出特征
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        result = pipeline.fit_transform(realistic_candles)

        # 验证两种 SSM 特征都存在
        deep_ssm_cols = [c for c in result.columns if c.startswith("deep_ssm_")]
        lg_ssm_cols = [c for c in result.columns if c.startswith("lg_ssm_")]

        assert len(deep_ssm_cols) == 3, (
            f"Expected 3 deep_ssm columns, got {len(deep_ssm_cols)}"
        )
        assert len(lg_ssm_cols) == 3, (
            f"Expected 3 lg_ssm columns, got {len(lg_ssm_cols)}"
        )

        # 有效数据部分两种 SSM 特征都应有值
        first_valid, _ = get_valid_data_range(result)
        valid_result = result.iloc[first_valid:]

        for col in deep_ssm_cols + lg_ssm_cols:
            assert not valid_result[col].isna().any(), f"NaN in {col}"
            assert np.isfinite(valid_result[col].values).all(), f"Inf in {col}"


class TestFitTransformSemantics:
    """验证 fit_transform 与 fit + transform 语义一致性"""

    def test_fit_transform_equals_fit_then_transform_no_reduction(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证不降维时 fit_transform(X) 与 fit(X).transform(X) 结果一致

        这是 sklearn 惯例的核心语义要求。
        """
        seed_everything(42)

        # 方法1: fit_transform
        pipeline1 = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )
        result1 = pipeline1.fit_transform(realistic_candles)

        # 方法2: fit + transform（使用相同的随机种子）
        seed_everything(42)
        pipeline2 = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )
        pipeline2.fit(realistic_candles)
        result2 = pipeline2.transform(realistic_candles)

        # 列名应该完全一致
        assert list(result1.columns) == list(result2.columns), (
            f"Column mismatch:\n"
            f"  fit_transform: {list(result1.columns)}\n"
            f"  fit+transform: {list(result2.columns)}"
        )

        # 行数应该完全一致
        assert len(result1) == len(result2), (
            f"Row count mismatch: fit_transform={len(result1)}, fit+transform={len(result2)}"
        )

        # 有效数据范围内数值应该完全一致
        first_valid, _ = get_valid_data_range(result1)

        np.testing.assert_allclose(
            result1.iloc[first_valid:].values,
            result2.iloc[first_valid:].values,
            rtol=PIPELINE_RTOL,
            atol=PIPELINE_ATOL,
            err_msg="fit_transform vs fit+transform should produce identical results",
        )

    def test_fit_transform_equals_fit_then_transform_with_reduction(
        self,
        realistic_candles,
        config_with_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证降维时 fit_transform(X) 与 fit(X).transform(X) 结果一致
        """
        seed_everything(42)

        # 方法1: fit_transform
        pipeline1 = create_pipeline_with_ssm(
            config_with_reduction, lgssm_config, deepssm_config
        )
        result1 = pipeline1.fit_transform(realistic_candles)

        # 方法2: fit + transform（使用相同的随机种子）
        seed_everything(42)
        pipeline2 = create_pipeline_with_ssm(
            config_with_reduction, lgssm_config, deepssm_config
        )
        pipeline2.fit(realistic_candles)
        result2 = pipeline2.transform(realistic_candles)

        # 列名应该完全一致
        assert list(result1.columns) == list(result2.columns), (
            f"Column mismatch:\n"
            f"  fit_transform: {list(result1.columns)}\n"
            f"  fit+transform: {list(result2.columns)}"
        )

        # 行数应该完全一致
        assert len(result1) == len(result2), (
            f"Row count mismatch: fit_transform={len(result1)}, fit+transform={len(result2)}"
        )

        # 有效数据范围内数值应该完全一致
        first_valid, _ = get_valid_data_range(result1)

        np.testing.assert_allclose(
            result1.iloc[first_valid:].values,
            result2.iloc[first_valid:].values,
            rtol=PIPELINE_RTOL,
            atol=PIPELINE_ATOL,
            err_msg="fit_transform vs fit+transform should produce identical results with reduction",
        )

    def test_fit_handles_nan_like_fit_transform(
        self,
        realistic_candles,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证 fit 方法与 fit_transform 一样处理 NaN

        两者应该在相同位置开始处理有效数据。
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        # fit_transform 应该成功并返回结果
        result = pipeline.fit_transform(realistic_candles)
        first_valid_fit_transform, _ = get_valid_data_range(result)

        # 新 pipeline: fit 也应该成功处理相同的数据
        seed_everything(42)
        pipeline2 = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        # fit 不应该报错（即使数据开头有 NaN）
        pipeline2.fit(realistic_candles)

        # transform 应该产生相同的结果
        result2 = pipeline2.transform(realistic_candles)
        first_valid_fit, _ = get_valid_data_range(result2)

        # 两者的第一个有效行应该相同
        assert first_valid_fit_transform == first_valid_fit, (
            f"First valid row mismatch: "
            f"fit_transform={first_valid_fit_transform}, fit+transform={first_valid_fit}"
        )

    def test_fit_rejects_all_nan_data(
        self,
        config_no_reduction,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证 fit 对全 NaN 数据报错

        与 fit_transform 行为一致。

        注意：使用 200 行数据，足够指标计算但 fracdiff 特征会全部是 NaN。
        """
        seed_everything(42)

        pipeline = create_pipeline_with_ssm(
            config_no_reduction, lgssm_config, deepssm_config
        )

        # 创建中等长度的数据（足够指标计算，但 fracdiff 特征全为 NaN）
        # fracdiff 需要约 372+ 行才能产生有效值
        short_candles = make_jesse_candles(n_samples=200, seed=42)

        # fit 应该报错（因为所有特征行都包含 NaN）
        with pytest.raises(ValueError, match="All rows contain NaN"):
            pipeline.fit(short_candles)


class TestSSMCopy:
    """验证 SSM 模型复制功能"""

    def test_copy_ssm_success(
        self,
        realistic_candles,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证成功复制 SSM 模型

        复制后的 Pipeline 应该能正常 fit_transform，且 SSM 训练被跳过。
        """
        seed_everything(42)

        # 1. 创建并训练全量 Pipeline
        full_config = PipelineConfig(
            feature_names=MIXED_FEATURES.copy(),
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        full_pipeline = create_pipeline_with_ssm(
            full_config, lgssm_config, deepssm_config
        )
        full_pipeline.fit_transform(realistic_candles)

        # 2. 创建模型特定 Pipeline（只选择部分特征）
        model_config = PipelineConfig(
            feature_names=["deep_ssm_0", "lg_ssm_0", "adx_14"],
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        model_pipeline = FeaturePipeline(model_config)

        # 3. 复制 SSM
        model_pipeline.copy_ssm_from(full_pipeline)

        # 验证 SSM 已复制且已 fit
        assert "deep_ssm" in model_pipeline.ssm_processors
        assert "lg_ssm" in model_pipeline.ssm_processors
        assert model_pipeline.ssm_processors["deep_ssm"].is_fitted
        assert model_pipeline.ssm_processors["lg_ssm"].is_fitted

        # 4. fit_transform 应该跳过 SSM 训练（因为已复制）
        model_result = model_pipeline.fit_transform(realistic_candles)

        # 验证输出格式
        assert model_result.shape[1] == 3  # deep_ssm_0, lg_ssm_0, adx_14
        assert set(model_result.columns) == {"deep_ssm_0", "lg_ssm_0", "adx_14"}

        # 验证有效数据存在
        first_valid, last_valid = get_valid_data_range(model_result)
        assert first_valid < last_valid, "Should have valid data"

    def test_copy_ssm_then_fit_with_reducer(
        self,
        realistic_candles,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证复制 SSM 后可以正常训练降维器

        这是典型的使用场景：复制 SSM + 训练降维器。
        """
        seed_everything(42)

        # 1. 训练全量 Pipeline（不降维）
        full_config = PipelineConfig(
            feature_names=MIXED_FEATURES.copy(),
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        full_pipeline = create_pipeline_with_ssm(
            full_config, lgssm_config, deepssm_config
        )
        full_pipeline.fit_transform(realistic_candles)

        # 2. 创建带降维的模型 Pipeline
        model_config = PipelineConfig(
            feature_names=["deep_ssm_0", "deep_ssm_1", "lg_ssm_0", "adx_14", "natr"],
            ssm_state_dim=5,
            use_dimension_reducer=True,
            dimension_reducer_config={
                "max_latent_dim": 4,
                "max_epochs": 5,
                "patience": 3,
                "seed": 42,
            },
        )
        model_pipeline = FeaturePipeline(model_config)
        model_pipeline.copy_ssm_from(full_pipeline)

        # 3. fit_transform 应该只训练降维器
        model_result = model_pipeline.fit_transform(realistic_candles)

        # 验证降维器已训练
        assert model_pipeline.dimension_reducer is not None
        assert model_pipeline.dimension_reducer.is_fitted

        # 验证输出是降维后的特征
        # 列名应该是 "0", "1", ... 而非原始特征名
        assert all(col.isdigit() for col in model_result.columns)

    def test_copy_ssm_input_features_mismatch(
        self,
        realistic_candles,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证 ssm_input_features 不匹配时报错
        """
        seed_everything(42)

        # 1. 训练源 Pipeline
        source_config = PipelineConfig(
            feature_names=["deep_ssm_0"],
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        source_pipeline = create_pipeline_with_ssm(
            source_config, lgssm_config, deepssm_config
        )
        source_pipeline.fit_transform(realistic_candles)

        # 2. 创建目标 Pipeline，使用不同的 ssm_input_features
        target_config = PipelineConfig(
            feature_names=["deep_ssm_0"],
            ssm_state_dim=5,
            ssm_input_features=["frac_o_o1_diff", "frac_o_o2_diff"],  # 不同的输入特征
            use_dimension_reducer=False,
        )
        target_pipeline = FeaturePipeline(target_config)

        # 3. 复制应该报错
        with pytest.raises(ValueError, match="SSM input features mismatch"):
            target_pipeline.copy_ssm_from(source_pipeline)

    def test_copy_ssm_state_dim_mismatch(
        self,
        realistic_candles,
    ):
        """
        验证 ssm_state_dim 不匹配时报错
        """
        seed_everything(42)

        # 1. 训练源 Pipeline（state_dim=5）
        source_config = PipelineConfig(
            feature_names=["lg_ssm_0", "lg_ssm_1"],
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        lgssm_config = LGSSMConfig(obs_dim=80, state_dim=5)
        source_pipeline = FeaturePipeline(source_config)
        source_pipeline._ssm_processors["lg_ssm"] = LGSSMAdapter(config=lgssm_config)
        source_pipeline.fit_transform(realistic_candles)

        # 2. 创建目标 Pipeline（state_dim=10）
        target_config = PipelineConfig(
            feature_names=["lg_ssm_0", "lg_ssm_1"],
            ssm_state_dim=10,  # 不同的 state_dim
            use_dimension_reducer=False,
        )
        target_pipeline = FeaturePipeline(target_config)

        # 3. 复制应该报错
        with pytest.raises(ValueError, match="SSM state_dim mismatch"):
            target_pipeline.copy_ssm_from(source_pipeline)

    def test_copy_ssm_source_not_fitted(
        self,
        lgssm_config,
        deepssm_config,
    ):
        """
        验证源 Pipeline 未 fit 时报错
        """
        # 1. 创建未训练的源 Pipeline
        source_config = PipelineConfig(
            feature_names=["deep_ssm_0"],
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        source_pipeline = create_pipeline_with_ssm(
            source_config, lgssm_config, deepssm_config
        )
        # 不调用 fit_transform

        # 2. 创建目标 Pipeline
        target_config = PipelineConfig(
            feature_names=["deep_ssm_0"],
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        target_pipeline = FeaturePipeline(target_config)

        # 3. 复制应该报错
        with pytest.raises(RuntimeError, match="Source pipeline not fitted"):
            target_pipeline.copy_ssm_from(source_pipeline)

    def test_copy_ssm_type_not_found(
        self,
        realistic_candles,
    ):
        """
        验证源 Pipeline 缺少所需 SSM 类型时报错
        """
        seed_everything(42)

        # 1. 训练只有 lg_ssm 的源 Pipeline
        source_config = PipelineConfig(
            feature_names=["lg_ssm_0"],  # 只有 lg_ssm
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        lgssm_config = LGSSMConfig(obs_dim=80, state_dim=5)
        source_pipeline = FeaturePipeline(source_config)
        source_pipeline._ssm_processors["lg_ssm"] = LGSSMAdapter(config=lgssm_config)
        source_pipeline.fit_transform(realistic_candles)

        # 2. 创建需要 deep_ssm 的目标 Pipeline
        target_config = PipelineConfig(
            feature_names=["deep_ssm_0"],  # 需要 deep_ssm
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        target_pipeline = FeaturePipeline(target_config)

        # 3. 复制应该报错
        with pytest.raises(KeyError, match="Source pipeline lacks SSM type: deep_ssm"):
            target_pipeline.copy_ssm_from(source_pipeline)

    def test_copied_ssm_independent_state(
        self,
        realistic_candles,
    ):
        """
        验证复制的 SSM 拥有独立的内部状态

        修改目标 Pipeline 的 SSM 状态不应影响源 Pipeline。
        """
        seed_everything(42)

        # 1. 训练源 Pipeline
        source_config = PipelineConfig(
            feature_names=["lg_ssm_0", "lg_ssm_1"],
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        lgssm_config = LGSSMConfig(obs_dim=80, state_dim=5)
        source_pipeline = FeaturePipeline(source_config)
        source_pipeline._ssm_processors["lg_ssm"] = LGSSMAdapter(config=lgssm_config)
        source_pipeline.fit_transform(realistic_candles)

        # 记录源 SSM 的状态
        source_ssm = source_pipeline.ssm_processors["lg_ssm"]
        source_state_before = (
            source_ssm._state.copy() if source_ssm._state is not None else None
        )

        # 2. 复制 SSM
        target_config = PipelineConfig(
            feature_names=["lg_ssm_0"],
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        target_pipeline = FeaturePipeline(target_config)
        target_pipeline.copy_ssm_from(source_pipeline)

        # 3. 使用目标 Pipeline 进行推理（会修改状态）
        target_pipeline._is_fitted = True  # 标记为已 fit
        target_pipeline.warmup_ssm(realistic_candles)

        # 4. 验证源 SSM 状态未被修改
        source_state_after = (
            source_ssm._state.copy() if source_ssm._state is not None else None
        )

        if source_state_before is not None and source_state_after is not None:
            np.testing.assert_array_equal(
                source_state_before,
                source_state_after,
                err_msg="Source SSM state should not be modified by target pipeline",
            )


class TestShareRawCalculatorE2E:
    """测试 share_raw_calculator_from 完整工作流"""

    def test_share_calculator_then_fit_transform_no_ssm(self, realistic_candles):
        """共享 calculator 后 fit_transform 应复用缓存（无 SSM 场景）"""
        import time

        # 1. 源 Pipeline 计算所有特征（使用注册表中存在的特征）
        source_config = PipelineConfig(
            feature_names=["fisher", "natr", "vwap", "adx_7", "adx_14"],
            use_dimension_reducer=False,
        )
        source_pipeline = FeaturePipeline(source_config)

        start_time = time.perf_counter()
        source_result = source_pipeline.fit_transform(realistic_candles)
        source_time = time.perf_counter() - start_time

        # 验证缓存已建立
        assert len(source_pipeline._raw_calculator.cache) > 0

        # 2. 目标 Pipeline 共享 calculator
        target_config = PipelineConfig(
            feature_names=["fisher", "natr"],  # 子集
            use_dimension_reducer=False,
        )
        target_pipeline = FeaturePipeline(target_config)
        target_pipeline.share_raw_calculator_from(source_pipeline)

        # 3. fit_transform 应该复用缓存
        start_time = time.perf_counter()
        target_result = target_pipeline.fit_transform(realistic_candles)
        target_time = time.perf_counter() - start_time

        # 4. 验证结果正确
        assert list(target_result.columns) == ["fisher", "natr"]
        assert len(target_result) == len(source_result)

        # 验证非 NaN 值一致（fit_transform 的 NaN 填充可能导致 NaN 位置略有不同）
        for col in ["fisher", "natr"]:
            source_vals = source_result[col].values
            target_vals = target_result[col].values
            # 找到两者都非 NaN 的位置
            valid_mask = ~np.isnan(source_vals) & ~np.isnan(target_vals)
            np.testing.assert_array_almost_equal(
                target_vals[valid_mask],
                source_vals[valid_mask],
                decimal=10,
            )

        # 5. 目标 Pipeline 应该显著更快（缓存复用）
        # 注意：由于缓存复用，目标 Pipeline 几乎不需要计算
        assert target_time < source_time

    def test_share_calculator_with_copy_ssm(
        self, realistic_candles, lgssm_config, deepssm_config
    ):
        """共享 calculator + 复制 SSM 的完整工作流"""
        # 1. 源 Pipeline：全量特征
        source_config = PipelineConfig(
            feature_names=["lg_ssm_0", "lg_ssm_1", "deep_ssm_0", "fisher", "natr"],
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        source_pipeline = FeaturePipeline(source_config)
        source_pipeline._ssm_processors["lg_ssm"] = LGSSMAdapter(config=lgssm_config)
        source_pipeline._ssm_processors["deep_ssm"] = DeepSSMAdapter(
            config=deepssm_config
        )

        source_result = source_pipeline.fit_transform(realistic_candles)

        # 2. 目标 Pipeline：特征子集，共享 calculator + 复制 SSM
        target_config = PipelineConfig(
            feature_names=["lg_ssm_0", "fisher"],  # 子集
            ssm_state_dim=5,
            use_dimension_reducer=False,
        )
        target_pipeline = FeaturePipeline(target_config)
        target_pipeline.share_raw_calculator_from(source_pipeline)
        target_pipeline.copy_ssm_from(source_pipeline, ssm_types=["lg_ssm"])

        target_result = target_pipeline.fit_transform(realistic_candles)

        # 3. 验证结果
        assert list(target_result.columns) == ["lg_ssm_0", "fisher"]

        # SSM 特征值应该一致（因为复制了训练好的 SSM）
        np.testing.assert_array_almost_equal(
            target_result["lg_ssm_0"].values,
            source_result["lg_ssm_0"].values,
            decimal=5,
        )

        # 原始特征值应该一致（因为共享了 calculator 缓存）
        np.testing.assert_array_equal(
            target_result["fisher"].values,
            source_result["fisher"].values,
        )

    def test_shared_calculator_is_same_instance(self, realistic_candles):
        """验证共享的 calculator 是同一个实例"""
        source_config = PipelineConfig(
            feature_names=["fisher"],
            use_dimension_reducer=False,
        )
        source_pipeline = FeaturePipeline(source_config)

        target_config = PipelineConfig(
            feature_names=["natr"],
            use_dimension_reducer=False,
        )
        target_pipeline = FeaturePipeline(target_config)

        # 共享前
        assert source_pipeline._raw_calculator is not target_pipeline._raw_calculator

        # 共享
        target_pipeline.share_raw_calculator_from(source_pipeline)

        # 共享后是同一实例
        assert source_pipeline._raw_calculator is target_pipeline._raw_calculator

        # 对其中一个操作会影响另一个
        source_pipeline._raw_calculator.load(realistic_candles, sequential=True)
        assert target_pipeline._raw_calculator.candles is realistic_candles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
