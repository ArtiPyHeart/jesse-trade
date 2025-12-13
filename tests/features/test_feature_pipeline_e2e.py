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
            ssm_processors["lg_ssm"] = LGSSMAdapter(config=lgssm_config, prefix="lg_ssm")

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
        result1 = pipeline.fit_transform(realistic_candles, verbose=False)

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

        pipeline.fit_transform(realistic_candles, verbose=False)

        # Ground truth: transform
        transform_result = pipeline.transform(realistic_candles)
        first_valid, _ = get_valid_data_range(transform_result)

        # warmup 长度需要在有效数据之后
        warmup_len = first_valid + 20
        assert warmup_len < len(realistic_candles) - 50, "Not enough data for test"

        # warmup + inference
        pipeline.warmup_ssm(realistic_candles[:warmup_len], verbose=False)

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

        result = pipeline.fit_transform(realistic_candles, verbose=False)

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
        fit_transform_result = pipeline.fit_transform(realistic_candles, verbose=False)
        fit_transform_columns = list(fit_transform_result.columns)

        # 2. transform 输出
        transform_result = pipeline.transform(realistic_candles)
        transform_columns = list(transform_result.columns)

        # 3. inference 输出（需要先 warmup）
        first_valid, _ = get_valid_data_range(fit_transform_result)
        warmup_len = first_valid + 30
        pipeline.warmup_ssm(realistic_candles[:warmup_len], verbose=False)
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
        fit_transform_result = pipeline.fit_transform(realistic_candles, verbose=False)
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
        pipeline.warmup_ssm(realistic_candles[:warmup_len], verbose=False)
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

        result = pipeline.fit_transform(realistic_candles, verbose=False)

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
        result1 = pipeline.fit_transform(realistic_candles, verbose=False)

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

        pipeline.fit_transform(realistic_candles, verbose=False)

        # Ground truth
        transform_result = pipeline.transform(realistic_candles)
        first_valid, _ = get_valid_data_range(transform_result)

        warmup_len = first_valid + 20
        assert warmup_len < len(realistic_candles) - 50

        # warmup + inference
        pipeline.warmup_ssm(realistic_candles[:warmup_len], verbose=False)

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

        result = pipeline.fit_transform(realistic_candles, verbose=False)

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

        result = pipeline.fit_transform(realistic_candles, verbose=False)

        # 降维后列数应 <= max_latent_dim
        max_latent_dim = config_with_reduction.dimension_reducer_config["max_latent_dim"]
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
        fit_transform_result = pipeline.fit_transform(realistic_candles, verbose=False)
        fit_transform_columns = list(fit_transform_result.columns)

        # 2. transform 输出
        transform_result = pipeline.transform(realistic_candles)
        transform_columns = list(transform_result.columns)

        # 3. inference 输出（需要先 warmup）
        first_valid, _ = get_valid_data_range(fit_transform_result)
        warmup_len = first_valid + 30
        pipeline.warmup_ssm(realistic_candles[:warmup_len], verbose=False)
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

        original = pipeline.fit_transform(realistic_candles, verbose=False)
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

        original = pipeline.fit_transform(realistic_candles, verbose=False)
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

        pipeline.fit_transform(realistic_candles, verbose=False)

        transform_result = pipeline.transform(realistic_candles)
        first_valid, _ = get_valid_data_range(transform_result)

        warmup_len = first_valid + 30
        assert warmup_len < len(realistic_candles)

        # 原始 pipeline: warmup + inference
        pipeline.warmup_ssm(realistic_candles[:warmup_len], verbose=False)
        original_inference = pipeline.inference(
            realistic_candles[: warmup_len + 1]
        ).iloc[0].values

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.save(tmpdir, "e2e_inference_test")
            loaded = FeaturePipeline.load(tmpdir, "e2e_inference_test")

            # 加载后: warmup + inference
            loaded.warmup_ssm(realistic_candles[:warmup_len], verbose=False)
            restored_inference = loaded.inference(
                realistic_candles[: warmup_len + 1]
            ).iloc[0].values

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

        pipeline.fit_transform(realistic_candles, verbose=False)

        transform_result = pipeline.transform(realistic_candles)
        first_valid, _ = get_valid_data_range(transform_result)

        warmup_len = first_valid + 30
        assert warmup_len < len(realistic_candles)

        # 原始 pipeline
        pipeline.warmup_ssm(realistic_candles[:warmup_len], verbose=False)
        original_inference = pipeline.inference(
            realistic_candles[: warmup_len + 1]
        ).iloc[0].values

        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存（降维器自动保存）
            pipeline.save(tmpdir, "e2e_inference_reduction")

            # 加载（降维器自动加载）
            loaded = FeaturePipeline.load(tmpdir, "e2e_inference_reduction")

            # 加载后
            loaded.warmup_ssm(realistic_candles[:warmup_len], verbose=False)
            restored_inference = loaded.inference(
                realistic_candles[: warmup_len + 1]
            ).iloc[0].values

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

        result = pipeline.fit_transform(realistic_candles, verbose=False)

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

        pipeline.fit_transform(realistic_candles, verbose=False)

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

        result = pipeline.fit_transform(realistic_candles, verbose=False)

        # 验证两种 SSM 特征都存在
        deep_ssm_cols = [c for c in result.columns if c.startswith("deep_ssm_")]
        lg_ssm_cols = [c for c in result.columns if c.startswith("lg_ssm_")]

        assert len(deep_ssm_cols) == 3, f"Expected 3 deep_ssm columns, got {len(deep_ssm_cols)}"
        assert len(lg_ssm_cols) == 3, f"Expected 3 lg_ssm columns, got {len(lg_ssm_cols)}"

        # 有效数据部分两种 SSM 特征都应有值
        first_valid, _ = get_valid_data_range(result)
        valid_result = result.iloc[first_valid:]

        for col in deep_ssm_cols + lg_ssm_cols:
            assert not valid_result[col].isna().any(), f"NaN in {col}"
            assert np.isfinite(valid_result[col].values).all(), f"Inf in {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
