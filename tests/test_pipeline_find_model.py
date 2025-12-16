"""
Pipeline Find Model 端到端测试

使用 2 个月真实 candles 数据测试 pipeline_find_model.py 的每个环节
运行命令: pytest tests/test_pipeline_find_model.py -v -s

前置条件：
1. 从项目根目录运行: python scripts/save_test_candles.py
2. 确保 data/test_candles/fusion_candles_2025-04-01_2025-06-01.npy 存在

测试策略：
- 使用 module scope fixtures 避免重复加载/计算耗时资源
- 按流水线顺序逐步验证每个环节
- 重点验证 share_raw_calculator_from 的正确性
"""

from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# 配置：使用 2 个月数据
TEST_START = "2025-04-01"
TEST_END = "2025-06-01"

# candles 文件路径（相对于项目根目录）
CANDLES_FILE = Path(__file__).parent.parent / "data/test_candles" / f"fusion_candles_{TEST_START}_{TEST_END}.npy"

# 测试参数
TEST_LOG_RETURN_LAG = 5
TEST_PRED_NEXT = 2
TEST_SSM_STATE_DIM = 5

# 测试用精简特征集（约 50 个特征，覆盖主要类型）
TEST_FEATS = [
    # 基础 OHLC 特征
    "bar_open_dt", "bar_high_dt", "bar_low_dt", "bar_close_dt",
    # 波动率指标
    "natr", "bekker_parkinson_vol",
    # 动量指标
    "fisher", "williams_r", "mod_rsi",
    # 统计特征
    "natr_mean20", "natr_std20", "fisher_skew20", "fisher_kurt20",
    # 归一化特征
    "natr_norm20", "fisher_zscore20",
    # 差分特征
    "natr_dt", "fisher_dt", "natr_ddt",
    # 滞后特征
    "natr_lag1", "natr_lag2", "fisher_lag1",
    # 高级特征
    "bar_close_dt_hurst20", "bar_close_dt_curv20",
]

# ARDVAE 降维器配置（测试用简化配置）
TEST_REDUCER_CONFIG = {
    "max_latent_dim": 32,  # 测试用小维度
    "kl_threshold": 0.01,
    "max_epochs": 50,  # 测试用少 epochs
    "patience": 5,
    "seed": 42,
}


# =============================================================================
# Module-level fixtures（共享耗时资源）
# =============================================================================


@pytest.fixture(scope="module")
def candles():
    """从文件加载 fusion candles（约 2 个月数据）"""
    if not CANDLES_FILE.exists():
        pytest.skip(
            f"Candles 文件不存在: {CANDLES_FILE}\n"
            f"请先从项目根目录运行: python scripts/save_test_candles.py"
        )

    print(f"\n[Fixture] 从文件加载 candles: {CANDLES_FILE}")
    result = np.load(CANDLES_FILE)
    print(f"[Fixture] 加载完成: {len(result)} 条 fusion bars")
    return result


@pytest.fixture(scope="module")
def global_config():
    """全局 Pipeline 配置（不降维，使用精简特征集）"""
    from research.model_pick.feature_utils import build_full_feature_config

    return build_full_feature_config(TEST_FEATS, ssm_state_dim=TEST_SSM_STATE_DIM)


@pytest.fixture(scope="module")
def global_pipeline(global_config):
    """创建全局 FeaturePipeline（不训练）"""
    from src.features.pipeline import FeaturePipeline

    print(f"\n[Fixture] 创建全局 FeaturePipeline，特征数: {len(global_config.feature_names)}")
    return FeaturePipeline(global_config)


@pytest.fixture(scope="module")
def global_features(global_pipeline, candles):
    """计算全局特征（训练 SSM）"""
    print("\n[Fixture] 计算全局特征 (fit_transform)...")
    result = global_pipeline.fit_transform(candles)
    print(f"[Fixture] 全局特征计算完成: {result.shape}")
    return result


@pytest.fixture(scope="module")
def labeler(candles):
    """创建标签器"""
    from research.model_pick.labeler import PipelineLabeler

    print(f"\n[Fixture] 创建 PipelineLabeler, lag={TEST_LOG_RETURN_LAG}")
    return PipelineLabeler(candles, TEST_LOG_RETURN_LAG)


@pytest.fixture(scope="module")
def aligned_data(global_features, labeler, candles):
    """对齐特征和标签"""
    from research.model_pick.feature_utils import align_features_and_labels

    print("\n[Fixture] 对齐特征和标签...")
    raw_label = labeler.label_hard
    aligned_features, aligned_labels = align_features_and_labels(
        global_features, raw_label, TEST_PRED_NEXT, candles[:, 0]
    )
    print(f"[Fixture] 对齐完成: features={aligned_features.shape}, labels={len(aligned_labels)}")
    return aligned_features, aligned_labels


@pytest.fixture(scope="module")
def selection_result(aligned_data):
    """特征筛选结果"""
    from research.model_pick.feature_utils import select_features

    aligned_features, aligned_labels = aligned_data
    print("\n[Fixture] 执行特征筛选...")
    result = select_features(aligned_features, aligned_labels)
    print(f"[Fixture] 筛选完成: {result.n_total} -> {result.n_selected}")
    return result


# =============================================================================
# Step 1: FusionCandles 测试
# =============================================================================


class TestFusionCandles:
    """Step 1: 验证 FusionCandles 输出"""

    def test_candles_shape(self, candles):
        """K线数据应为 6 列 [timestamp, open, close, high, low, volume]"""
        assert candles.ndim == 2, f"Expected 2D array, got {candles.ndim}D"
        assert candles.shape[1] == 6, f"Expected 6 columns, got {candles.shape[1]}"

    def test_candles_not_empty(self, candles):
        """数据不应为空"""
        assert len(candles) > 0, "Candles should not be empty"
        # 2 个月数据应该至少有 1000 条记录
        assert len(candles) > 1000, f"Expected >1000 bars for 2 months, got {len(candles)}"

    def test_candles_no_all_nan_rows(self, candles):
        """不应有全 NaN 的行"""
        all_nan_rows = np.all(np.isnan(candles), axis=1)
        assert not all_nan_rows.any(), "Found rows with all NaN values"

    def test_candles_timestamp_monotonic(self, candles):
        """时间戳应单调递增"""
        timestamps = candles[:, 0]
        assert np.all(np.diff(timestamps) > 0), "Timestamps should be monotonically increasing"


# =============================================================================
# Step 2: Global Pipeline 测试
# =============================================================================


class TestGlobalPipeline:
    """Step 2: 验证全局 Pipeline"""

    def test_fit_transform_output_shape(self, global_features, candles):
        """输出行数应与 candles 一致"""
        assert len(global_features) == len(candles), (
            f"Output rows {len(global_features)} != candles rows {len(candles)}"
        )

    def test_feature_columns_match_config(self, global_features, global_config):
        """输出列应与配置一致"""
        expected_cols = set(global_config.feature_names)
        actual_cols = set(global_features.columns)
        assert actual_cols == expected_cols, (
            f"Column mismatch: missing={expected_cols - actual_cols}, "
            f"extra={actual_cols - expected_cols}"
        )

    def test_ssm_features_exist(self, global_features):
        """应包含 SSM 特征"""
        ssm_cols = [c for c in global_features.columns if c.startswith(("deep_ssm_", "lg_ssm_"))]
        expected_ssm_count = TEST_SSM_STATE_DIM * 2  # deep_ssm + lg_ssm
        assert len(ssm_cols) == expected_ssm_count, (
            f"Expected {expected_ssm_count} SSM features, got {len(ssm_cols)}: {ssm_cols}"
        )

    def test_no_all_nan_columns(self, global_features):
        """不应有全 NaN 的列（除了 warmup 期间）"""
        # 检查后半部分数据（跳过 warmup）
        half_idx = len(global_features) // 2
        data_half = global_features.iloc[half_idx:]
        all_nan_cols = data_half.columns[data_half.isna().all()]
        assert len(all_nan_cols) == 0, f"Found all-NaN columns after warmup: {list(all_nan_cols)}"


# =============================================================================
# Step 3: Labeler 测试
# =============================================================================


class TestLabeler:
    """Step 3: 验证标签生成"""

    def test_label_hard_not_empty(self, labeler):
        """硬标签不应为空"""
        label = labeler.label_hard
        assert len(label) > 0, "label_hard should not be empty"

    def test_label_hard_valid_values(self, labeler):
        """硬标签应只包含 -1, 0, 1 或 NaN"""
        label = labeler.label_hard
        valid_mask = np.isnan(label) | np.isin(label, [-1, 0, 1])
        assert valid_mask.all(), f"Invalid label values found: {np.unique(label[~valid_mask])}"

    def test_label_distribution_reasonable(self, labeler):
        """标签分布应合理（每类至少占 5%）"""
        label = labeler.label_hard
        valid_labels = label[~np.isnan(label)].astype(int)
        _, counts = np.unique(valid_labels, return_counts=True)
        min_ratio = counts.min() / counts.sum()
        assert min_ratio > 0.05, f"Label distribution too imbalanced: {dict(zip(*np.unique(valid_labels, return_counts=True)))}"

    def test_label_direction_continuous(self, labeler):
        """方向标签应为连续值"""
        label = labeler.label_direction
        valid_labels = label[~np.isnan(label)]
        # 应有足够的唯一值（连续分布）
        n_unique = len(np.unique(valid_labels))
        assert n_unique > 100, f"label_direction should be continuous, got only {n_unique} unique values"


# =============================================================================
# Step 4: Feature Alignment 测试
# =============================================================================


class TestFeatureAlignment:
    """Step 4: 验证特征-标签对齐"""

    def test_alignment_removes_nan(self, aligned_data):
        """对齐后不应有 NaN"""
        aligned_features, aligned_labels = aligned_data
        assert not aligned_features.isna().any().any(), "Features should have no NaN after alignment"
        assert not np.isnan(aligned_labels).any(), "Labels should have no NaN after alignment"

    def test_alignment_shape_correct(self, aligned_data):
        """特征和标签长度应一致"""
        aligned_features, aligned_labels = aligned_data
        assert len(aligned_features) == len(aligned_labels), (
            f"Length mismatch: features={len(aligned_features)}, labels={len(aligned_labels)}"
        )

    def test_alignment_index_is_timestamp(self, aligned_data):
        """对齐后 index 应为时间戳"""
        aligned_features, _ = aligned_data
        # 时间戳应为正整数（毫秒级）
        assert aligned_features.index.dtype in [np.int64, np.int32], (
            f"Index should be integer timestamp, got {aligned_features.index.dtype}"
        )
        assert (aligned_features.index > 0).all(), "Index should be positive timestamps"


# =============================================================================
# Step 5: Feature Selection 测试
# =============================================================================


class TestFeatureSelection:
    """Step 5: 验证特征筛选"""

    def test_selected_features_not_empty(self, selection_result):
        """应选出至少一些特征"""
        assert selection_result.n_selected > 0, "Should select at least some features"

    def test_selected_features_valid_names(self, selection_result, global_features):
        """选出的特征名应在全局特征中"""
        valid_names = set(global_features.columns)
        for feat in selection_result.selected_features:
            assert feat in valid_names, f"Selected feature '{feat}' not in global features"

    def test_selected_includes_ssm(self, selection_result):
        """应选出一些 SSM 特征"""
        ssm_selected = [f for f in selection_result.selected_features if f.startswith(("deep_ssm_", "lg_ssm_"))]
        # SSM 特征应该是重要的，至少选出一个
        assert len(ssm_selected) > 0, "Should select at least one SSM feature"


# =============================================================================
# Step 6: share_raw_calculator_from 测试
# =============================================================================


class TestShareRawCalculator:
    """Step 6: 验证 calculator 共享"""

    def test_calculator_is_same_instance(self, global_pipeline, selection_result):
        """共享后 calculator 应为同一实例"""
        from src.features.pipeline import FeaturePipeline
        from research.model_pick.feature_utils import build_model_config

        model_config = build_model_config(
            selection_result.selected_features,
            ssm_state_dim=TEST_SSM_STATE_DIM,
            reducer_config=TEST_REDUCER_CONFIG,
        )
        model_pipeline = FeaturePipeline(model_config)
        model_pipeline.share_raw_calculator_from(global_pipeline)

        assert model_pipeline._raw_calculator is global_pipeline._raw_calculator, (
            "After share_raw_calculator_from, calculators should be the same instance"
        )

    def test_cache_preserved_after_share(self, global_pipeline, selection_result, candles):
        """共享后缓存应被保留（相同 candles 不清空缓存）"""
        from src.features.pipeline import FeaturePipeline
        from research.model_pick.feature_utils import build_model_config

        # 获取 global_pipeline 的缓存大小
        original_cache_size = len(global_pipeline._raw_calculator.cache)
        assert original_cache_size > 0, "Global pipeline should have cached features"

        # 创建 model_pipeline 并共享 calculator
        model_config = build_model_config(
            selection_result.selected_features,
            ssm_state_dim=TEST_SSM_STATE_DIM,
            reducer_config=TEST_REDUCER_CONFIG,
        )
        model_pipeline = FeaturePipeline(model_config)
        model_pipeline.share_raw_calculator_from(global_pipeline)

        # 用相同 candles 调用 load - 缓存应保留
        model_pipeline._raw_calculator.load(candles, sequential=True)
        cache_size_after_load = len(model_pipeline._raw_calculator.cache)

        assert cache_size_after_load == original_cache_size, (
            f"Cache should be preserved for same candles: "
            f"before={original_cache_size}, after={cache_size_after_load}"
        )


# =============================================================================
# Step 7: copy_ssm_from 测试
# =============================================================================


class TestCopySSM:
    """Step 7: 验证 SSM 复制"""

    def test_ssm_copied_successfully(self, global_pipeline, selection_result):
        """SSM 应成功复制"""
        from src.features.pipeline import FeaturePipeline
        from research.model_pick.feature_utils import build_model_config

        model_config = build_model_config(
            selection_result.selected_features,
            ssm_state_dim=TEST_SSM_STATE_DIM,
            reducer_config=TEST_REDUCER_CONFIG,
        )
        model_pipeline = FeaturePipeline(model_config)
        model_pipeline.copy_ssm_from(global_pipeline)

        # 验证 SSM 已复制（使用 ssm_processors 属性或 _ssm_processors）
        for ssm_type in ["deep_ssm", "lg_ssm"]:
            if ssm_type in model_pipeline.ssm_processors:
                processor = model_pipeline.ssm_processors[ssm_type]
                assert processor._model is not None, f"{ssm_type} should have a copied model"
                assert processor.is_fitted, f"{ssm_type} should be fitted after copy"

    def test_ssm_states_independent(self, global_pipeline, selection_result, candles):
        """复制后 SSM 状态应独立（修改一个不影响另一个）"""
        from src.features.pipeline import FeaturePipeline
        from research.model_pick.feature_utils import build_model_config

        model_config = build_model_config(
            selection_result.selected_features,
            ssm_state_dim=TEST_SSM_STATE_DIM,
            reducer_config=TEST_REDUCER_CONFIG,
        )
        model_pipeline = FeaturePipeline(model_config)
        model_pipeline.copy_ssm_from(global_pipeline)

        # 验证 processor 和内部 model 都不是同一实例
        for ssm_type in ["deep_ssm", "lg_ssm"]:
            if ssm_type in global_pipeline.ssm_processors and ssm_type in model_pipeline.ssm_processors:
                global_processor = global_pipeline.ssm_processors[ssm_type]
                model_processor = model_pipeline.ssm_processors[ssm_type]
                assert global_processor is not model_processor, f"{ssm_type} processors should be different instances"
                # 内部 model 也应该是独立的深拷贝
                assert global_processor._model is not model_processor._model, (
                    f"{ssm_type} internal models should be different instances (deep copy)"
                )


# =============================================================================
# Step 8: Model Pipeline 测试
# =============================================================================


class TestModelPipeline:
    """Step 8: 验证模型 Pipeline"""

    def test_fit_transform_with_shared_calculator(self, global_pipeline, selection_result, candles):
        """使用共享 calculator 的 fit_transform 应正常工作"""
        from src.features.pipeline import FeaturePipeline
        from research.model_pick.feature_utils import build_model_config

        model_config = build_model_config(
            selection_result.selected_features,
            ssm_state_dim=TEST_SSM_STATE_DIM,
            reducer_config=TEST_REDUCER_CONFIG,
        )
        model_pipeline = FeaturePipeline(model_config)
        model_pipeline.share_raw_calculator_from(global_pipeline)
        model_pipeline.copy_ssm_from(global_pipeline)

        # fit_transform 应正常完成
        model_features = model_pipeline.fit_transform(candles)

        assert model_features is not None, "fit_transform should return features"
        assert len(model_features) == len(candles), (
            f"Output rows {len(model_features)} != candles rows {len(candles)}"
        )

    def test_dimension_reduction_works(self, global_pipeline, selection_result, candles):
        """降维应实际减少维度"""
        from src.features.pipeline import FeaturePipeline
        from research.model_pick.feature_utils import build_model_config

        model_config = build_model_config(
            selection_result.selected_features,
            ssm_state_dim=TEST_SSM_STATE_DIM,
            reducer_config=TEST_REDUCER_CONFIG,
        )
        model_pipeline = FeaturePipeline(model_config)
        model_pipeline.share_raw_calculator_from(global_pipeline)
        model_pipeline.copy_ssm_from(global_pipeline)

        model_features = model_pipeline.fit_transform(candles)

        # 降维后维度应小于输入维度
        n_input_features = selection_result.n_selected
        n_output_features = model_features.shape[1]

        print(f"\n降维: {n_input_features} -> {n_output_features}")
        assert n_output_features <= n_input_features, (
            f"Dimension reduction should reduce dimensions: "
            f"input={n_input_features}, output={n_output_features}"
        )


# =============================================================================
# Step 9: 数值一致性测试
# =============================================================================


class TestNumericalConsistency:
    """Step 9: 数值一致性验证"""

    def test_shared_vs_unshared_same_result(self, global_pipeline, selection_result, candles):
        """共享 calculator 与不共享应产生相同结果"""
        from src.features.pipeline import FeaturePipeline
        from research.model_pick.feature_utils import build_model_config

        # 方案 A: 使用 share_raw_calculator_from
        model_config_a = build_model_config(
            selection_result.selected_features,
            ssm_state_dim=TEST_SSM_STATE_DIM,
            reducer_config=TEST_REDUCER_CONFIG,
        )
        pipeline_a = FeaturePipeline(model_config_a)
        pipeline_a.share_raw_calculator_from(global_pipeline)
        pipeline_a.copy_ssm_from(global_pipeline)
        features_a = pipeline_a.fit_transform(candles)

        # 方案 B: 不使用 share_raw_calculator_from（独立计算）
        model_config_b = build_model_config(
            selection_result.selected_features,
            ssm_state_dim=TEST_SSM_STATE_DIM,
            reducer_config=TEST_REDUCER_CONFIG,
        )
        pipeline_b = FeaturePipeline(model_config_b)
        # 只复制 SSM，不共享 calculator
        pipeline_b.copy_ssm_from(global_pipeline)
        features_b = pipeline_b.fit_transform(candles)

        # 比较结果（降维后的特征）
        # 注意：ARDVAE 有随机性，但 seed 固定后应一致
        assert features_a.shape == features_b.shape, (
            f"Shape mismatch: shared={features_a.shape}, unshared={features_b.shape}"
        )

        # 比较非 NaN 值
        mask_a = ~features_a.isna().values
        mask_b = ~features_b.isna().values
        assert np.array_equal(mask_a, mask_b), "NaN patterns should match"

        # 数值应完全一致（因为 seed 相同）
        valid_mask = mask_a.all(axis=1)
        if valid_mask.sum() > 0:
            np.testing.assert_array_almost_equal(
                features_a.loc[valid_mask].values,
                features_b.loc[valid_mask].values,
                decimal=6,
                err_msg="Feature values should be identical with same seed",
            )

    def test_raw_features_identical_after_share(self, global_pipeline, global_features, selection_result, candles):
        """共享 calculator 后，原始特征计算结果应完全一致"""
        from src.features.pipeline import FeaturePipeline
        from research.model_pick.feature_utils import build_model_config

        # 从 selection_result 中取几个原始特征（非 SSM）进行验证
        raw_features_to_check = [
            f for f in selection_result.selected_features
            if not f.startswith(("deep_ssm_", "lg_ssm_"))
        ][:5]  # 取前 5 个原始特征

        if not raw_features_to_check:
            pytest.skip("No raw features selected")

        # 使用共享 calculator 计算
        model_config = build_model_config(
            selection_result.selected_features,
            ssm_state_dim=TEST_SSM_STATE_DIM,
            reducer_config=None,  # 不降维，直接比较原始特征
        )
        # 不降维的配置
        model_config.use_dimension_reducer = False

        pipeline = FeaturePipeline(model_config)
        pipeline.share_raw_calculator_from(global_pipeline)
        pipeline.copy_ssm_from(global_pipeline)

        # 直接从共享的 calculator 获取特征
        calculator = pipeline._raw_calculator
        calculator.load(candles, sequential=True)
        shared_features = calculator.get(raw_features_to_check)

        # 与 global_features 比较
        for feat_name in raw_features_to_check:
            expected = global_features[feat_name].values
            actual = shared_features[feat_name]

            # 比较非 NaN 值
            valid_mask = ~np.isnan(expected) & ~np.isnan(actual)
            np.testing.assert_array_almost_equal(
                actual[valid_mask],
                expected[valid_mask],
                decimal=10,
                err_msg=f"Feature '{feat_name}' values differ between shared and original",
            )
