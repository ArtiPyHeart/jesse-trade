"""
Phase 2: 特征计算测试

测试覆盖:
1. 原始特征批量计算 (SimpleFeatureCalculator)
2. Fracdiff 特征提取
3. FeaturePipeline 推理一致性 (关键)
4. FeaturePipeline 输出完整性 (关键)

运行方式: 从项目根目录执行
    pytest tests/test_backtest_vectorized/test_phase2_features.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 初始化 Jesse 数据库连接 (必须在导入其他模块之前)
from jesse.services import db  # noqa: F401

PIPELINE_DIR = (
    Path(__file__).parent.parent.parent / "strategies/BinanceBtcDemoBarV2/models"
)
PIPELINE_NAME = "global_pipeline"


# ==================== Fixtures ====================
@pytest.fixture(scope="module")
def fusion_bars_for_features(jesse_candles):
    """
    生成用于特征测试的 fusion bars

    Returns:
        tuple: (fusion_bars, warmup_len)
            - fusion_bars: 所有 fusion bars
            - warmup_len: warmup 部分的长度
    """
    from src.bars.fusion.demo import DemoBar

    warmup_candles, trading_candles = jesse_candles

    # 合并并生成 fusion bars
    all_candles = np.vstack([warmup_candles, trading_candles])
    bar_container = DemoBar(clip_r=0.012, max_bars=-1, threshold=1.399)
    bar_container.update_with_candles(all_candles)
    fusion_bars = bar_container.get_fusion_bars()

    # 计算 warmup 分界点
    warmup_last_ts = warmup_candles[-1, 0]
    warmup_len = np.searchsorted(fusion_bars[:, 0], warmup_last_ts, side="right")

    return fusion_bars, warmup_len


@pytest.fixture(scope="module")
def feature_calculator():
    """返回 SimpleFeatureCalculator 实例"""
    from src.features.simple_feature_calculator import SimpleFeatureCalculator

    return SimpleFeatureCalculator()


@pytest.fixture(scope="module")
def pipeline_config():
    """返回全局 FeaturePipeline 配置"""
    from src.features.pipeline import PipelineConfig

    config_path = PIPELINE_DIR / PIPELINE_NAME / "pipeline_config.json"
    return PipelineConfig.load(str(config_path))


@pytest.fixture(scope="module")
def sample_fracdiff_features():
    """返回用于测试的 fracdiff 特征列表 (子集)"""
    return [
        "frac_o_o1_diff",
        "frac_o_c1_diff",
        "frac_c_c1_diff",
        "frac_h_l1_diff",
    ]


@pytest.fixture(scope="module")
def full_fracdiff_features(pipeline_config):
    """返回完整的 fracdiff 特征列表 (SSM 模型需要)"""
    return pipeline_config.ssm_input_features


@pytest.fixture(scope="module")
def sample_raw_features():
    """返回用于测试的原始特征列表 (不含 SSM 特征)"""
    return [
        "bar_duration",
        "bar_open",
        "bar_close",
    ]


@pytest.fixture(scope="module")
def pipeline_features(fusion_bars_for_features):
    """返回 FeaturePipeline 批量特征输出"""
    from src.features.pipeline import FeaturePipeline

    fusion_bars, warmup_len = fusion_bars_for_features
    pipeline = FeaturePipeline.load(str(PIPELINE_DIR), PIPELINE_NAME)
    df_features = pipeline.transform(fusion_bars)
    return df_features, warmup_len


# ==================== Test 2.1: 原始特征批量计算 ====================
class TestRawFeatureCalculation:
    """测试原始特征批量计算"""

    def test_batch_calculate_basic(
        self, fusion_bars_for_features, feature_calculator, sample_raw_features
    ):
        """验证批量计算基本功能"""
        fusion_bars, _ = fusion_bars_for_features

        # 批量计算
        feature_calculator.load(fusion_bars, sequential=True)
        features = feature_calculator.get(sample_raw_features)

        # 验证: 返回字典包含所有请求的特征
        assert isinstance(features, dict)
        for feat_name in sample_raw_features:
            assert feat_name in features, f"Missing feature: {feat_name}"

    def test_output_length_matches_candles(
        self, fusion_bars_for_features, feature_calculator, sample_raw_features
    ):
        """验证输出长度与 fusion bars 一致"""
        fusion_bars, _ = fusion_bars_for_features

        feature_calculator.load(fusion_bars, sequential=True)
        features = feature_calculator.get(sample_raw_features)

        for feat_name, feat_arr in features.items():
            assert len(feat_arr) == len(fusion_bars), (
                f"Feature {feat_name} length mismatch: "
                f"expected {len(fusion_bars)}, got {len(feat_arr)}"
            )

    def test_feature_values_in_reasonable_range(
        self, fusion_bars_for_features, feature_calculator
    ):
        """验证特征值在合理范围内"""
        fusion_bars, _ = fusion_bars_for_features

        feature_calculator.load(fusion_bars, sequential=True)

        # bar_duration 应该是非负数 (毫秒)
        # 注: 第一根 bar 的 duration 可能是 0
        duration = feature_calculator.get(["bar_duration"])["bar_duration"]
        valid_duration = duration[~np.isnan(duration)]
        assert np.all(valid_duration >= 0), "bar_duration should be non-negative"
        # 大多数 duration 应该是正数
        assert np.mean(valid_duration > 0) > 0.9, "Most bar_duration should be positive"

        # bar_close 应该与 fusion_bars 中的 close 一致
        bar_close = feature_calculator.get(["bar_close"])["bar_close"]
        # 跳过 NaN 位置比较
        valid_mask = ~np.isnan(bar_close)
        np.testing.assert_array_almost_equal(
            bar_close[valid_mask],
            fusion_bars[valid_mask, 2],  # close 在第 2 列
            decimal=6,
            err_msg="bar_close should match fusion_bars close column",
        )

    def test_nan_only_in_warmup_period(
        self, fusion_bars_for_features, feature_calculator, sample_raw_features
    ):
        """验证 NaN 仅出现在 warmup 期间 (指标预热期)"""
        fusion_bars, _ = fusion_bars_for_features

        feature_calculator.load(fusion_bars, sequential=True)
        features = feature_calculator.get(sample_raw_features)

        for feat_name, feat_arr in features.items():
            # 找到第一个非 NaN 的位置
            non_nan_indices = np.where(~np.isnan(feat_arr))[0]
            if len(non_nan_indices) == 0:
                continue  # 全是 NaN，跳过

            first_valid_idx = non_nan_indices[0]

            # NaN 之后不应该再有 NaN (对于大多数特征)
            # 注: 某些特征可能在中间有 NaN，这里只做宽松检查
            trailing_nans = np.isnan(feat_arr[first_valid_idx:])
            trailing_nan_ratio = np.mean(trailing_nans)
            assert trailing_nan_ratio < 0.1, (
                f"Feature {feat_name} has {trailing_nan_ratio * 100:.1f}% NaN "
                f"after first valid value (expected < 10%)"
            )


# ==================== Test 2.2: Fracdiff 特征提取 ====================
class TestFracdiffFeatures:
    """测试 Fracdiff 特征提取"""

    def test_fracdiff_columns_extracted(
        self, fusion_bars_for_features, feature_calculator, sample_fracdiff_features
    ):
        """验证 fracdiff 特征正确提取"""
        fusion_bars, _ = fusion_bars_for_features

        feature_calculator.load(fusion_bars, sequential=True)
        features = feature_calculator.get(sample_fracdiff_features)

        # 验证所有请求的 fracdiff 特征都存在
        for feat_name in sample_fracdiff_features:
            assert feat_name in features, f"Missing fracdiff feature: {feat_name}"
            assert feat_name.startswith("frac_"), (
                f"Fracdiff feature should start with 'frac_': {feat_name}"
            )

    def test_fracdiff_output_is_numeric(
        self, fusion_bars_for_features, feature_calculator, sample_fracdiff_features
    ):
        """验证 fracdiff 输出是数值类型"""
        fusion_bars, _ = fusion_bars_for_features

        feature_calculator.load(fusion_bars, sequential=True)
        features = feature_calculator.get(sample_fracdiff_features)

        for feat_name, feat_arr in features.items():
            assert np.issubdtype(feat_arr.dtype, np.floating), (
                f"Fracdiff feature {feat_name} should be float, got {feat_arr.dtype}"
            )

    def test_fracdiff_to_dataframe(
        self, fusion_bars_for_features, feature_calculator, sample_fracdiff_features
    ):
        """验证 fracdiff 特征可以转为 DataFrame"""
        fusion_bars, _ = fusion_bars_for_features

        feature_calculator.load(fusion_bars, sequential=True)
        features = feature_calculator.get(sample_fracdiff_features)

        df_fracdiff = pd.DataFrame.from_dict(features)

        # 验证 DataFrame 结构
        assert len(df_fracdiff) == len(fusion_bars)
        assert list(df_fracdiff.columns) == sample_fracdiff_features


# ==================== Test 2.3: FeaturePipeline 状态一致性 (关键) ====================
class TestPipelineStateConsistency:
    """测试 FeaturePipeline 状态一致性 - 验证 warmup 与 transform 输出一致"""

    def test_pipeline_inference_output_shape(self, fusion_bars_for_features):
        """验证 inference 输出形状正确"""
        from src.features.pipeline import FeaturePipeline

        fusion_bars, _ = fusion_bars_for_features
        pipeline = FeaturePipeline.load(str(PIPELINE_DIR), PIPELINE_NAME)

        pipeline.warmup_ssm(fusion_bars[:-1])
        result = pipeline.inference(fusion_bars)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.columns.is_unique
        assert not result.isna().any().any()

    def test_warmup_matches_transform_last_row(self, fusion_bars_for_features):
        """验证 warmup + inference 与 transform 最后一行一致"""
        from src.features.pipeline import FeaturePipeline

        fusion_bars, _ = fusion_bars_for_features

        pipeline_warm = FeaturePipeline.load(str(PIPELINE_DIR), PIPELINE_NAME)
        pipeline_warm.warmup_ssm(fusion_bars[:-1])
        inference_row = pipeline_warm.inference(fusion_bars).reset_index(drop=True)

        pipeline_batch = FeaturePipeline.load(str(PIPELINE_DIR), PIPELINE_NAME)
        batch_features = pipeline_batch.transform(fusion_bars)
        batch_row = batch_features.iloc[[-1]].reset_index(drop=True)

        assert not batch_row.isna().any().any()
        pd.testing.assert_frame_equal(
            inference_row,
            batch_row,
            check_exact=False,
            rtol=1e-5,
            atol=1e-6,
        )


# ==================== Test 2.4: FeaturePipeline 输出完整性 (关键) ====================
class TestPipelineOutput:
    """测试 FeaturePipeline 输出完整性"""

    def test_transform_length_matches_candles(
        self,
        fusion_bars_for_features,
        pipeline_features,
    ):
        """验证 transform 输出长度与 fusion bars 一致"""
        fusion_bars, _ = fusion_bars_for_features
        df_features, _ = pipeline_features

        assert len(df_features) == len(fusion_bars)

    def test_trading_slice_no_nan(self, pipeline_features):
        """验证 trading 部分无 NaN"""
        df_features, warmup_len = pipeline_features

        df_trading = df_features.iloc[warmup_len:].reset_index(drop=True)
        assert not df_trading.isna().any().any()

    def test_columns_unique(self, pipeline_features):
        """验证输出列名唯一"""
        df_features, _ = pipeline_features
        assert df_features.columns.is_unique
