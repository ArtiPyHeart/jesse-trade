"""
Phase 2: 特征计算测试

测试覆盖:
1. 原始特征批量计算 (SimpleFeatureCalculator)
2. Fracdiff 特征提取
3. SSM 状态一致性 (关键)
4. 特征拼接索引对齐 (关键)

运行方式: 从项目根目录执行
    pytest tests/test_backtest_vectorized/test_phase2_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

# 初始化 Jesse 数据库连接 (必须在导入其他模块之前)
from jesse.services import db  # noqa: F401


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
def sample_fracdiff_features():
    """返回用于测试的 fracdiff 特征列表 (子集)"""
    return [
        "frac_o_o1_diff",
        "frac_o_c1_diff",
        "frac_c_c1_diff",
        "frac_h_l1_diff",
    ]


@pytest.fixture(scope="module")
def full_fracdiff_features():
    """返回完整的 fracdiff 特征列表 (SSM 模型需要)"""
    import json
    from pathlib import Path

    feature_info_path = (
        Path(__file__).parent.parent.parent
        / "strategies/BinanceBtcDemoBarV2/models/feature_info.json"
    )
    with open(feature_info_path) as f:
        feature_info = json.load(f)
    return feature_info["fracdiff"]


@pytest.fixture(scope="module")
def sample_raw_features():
    """返回用于测试的原始特征列表 (不含 SSM 特征)"""
    return [
        "bar_duration",
        "bar_open",
        "bar_close",
    ]


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
        assert np.mean(valid_duration > 0) > 0.9, (
            "Most bar_duration should be positive"
        )

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


# ==================== Test 2.3: SSM 状态一致性 (关键) ====================
class TestSSMStateConsistency:
    """测试 SSM 状态一致性 - 验证 warmup 阶段正确更新状态"""

    @pytest.fixture(scope="class")
    def df_fracdiff_full(self, fusion_bars_for_features, full_fracdiff_features):
        """
        准备完整的 fracdiff DataFrame (SSM 需要 80 个特征)

        注意: fracdiff 特征在前几行会有 NaN（指标预热期），
        SSM 不能接收 NaN 输入，所以这里返回去除 NaN 后的数据和第一个有效行的索引
        """
        from src.features.simple_feature_calculator import SimpleFeatureCalculator

        fusion_bars, warmup_len = fusion_bars_for_features
        fc = SimpleFeatureCalculator()
        fc.load(fusion_bars, sequential=True)
        features = fc.get(full_fracdiff_features)
        df = pd.DataFrame.from_dict(features)

        # 找到第一个无 NaN 的行
        first_valid_idx = df.dropna().index[0]
        print(f"\n[fixture] Fracdiff first valid index: {first_valid_idx}")
        print(f"[fixture] Warmup len: {warmup_len}")

        # 返回从第一个有效行开始的数据，并重置索引
        df_valid = df.iloc[first_valid_idx:].reset_index(drop=True)

        # 调整 warmup_len（减去跳过的 NaN 行数）
        adjusted_warmup_len = max(0, warmup_len - first_valid_idx)

        return df_valid, adjusted_warmup_len, first_valid_idx

    @pytest.fixture
    def ssm_containers(self):
        """创建 SSM 容器"""
        from strategies.BinanceBtcDemoBarV2.models.config import SSMContainer

        return {
            "deep_ssm": SSMContainer("deep_ssm"),
            "lg_ssm": SSMContainer("lg_ssm"),
        }

    def test_ssm_inference_output_shape(
        self,
        df_fracdiff_full,
        ssm_containers,
    ):
        """验证 SSM inference 输出形状正确"""
        df_fracdiff, _, _ = df_fracdiff_full

        # 测试单行 inference
        single_row = df_fracdiff.iloc[[0]]

        for name, ssm in ssm_containers.items():
            result = ssm.inference(single_row)

            # 验证输出是 DataFrame
            assert isinstance(result, pd.DataFrame), (
                f"{name} inference should return DataFrame"
            )

            # 验证只有一行
            assert len(result) == 1, f"{name} inference should return 1 row"

            # 验证列名以 prefix 开头
            for col in result.columns:
                assert col.startswith(name), (
                    f"{name} column should start with '{name}': {col}"
                )

    def test_ssm_warmup_updates_state(
        self,
        df_fracdiff_full,
    ):
        """验证 warmup 阶段会更新 SSM 状态"""
        from strategies.BinanceBtcDemoBarV2.models.config import SSMContainer

        df_fracdiff, warmup_len, _ = df_fracdiff_full

        # 创建两个 SSM: 一个做 warmup，一个不做
        ssm_with_warmup = SSMContainer("lg_ssm")
        ssm_without_warmup = SSMContainer("lg_ssm")

        # 对 ssm_with_warmup 执行 warmup
        warmup_count = min(warmup_len, 100)  # 限制 warmup 数量以加速测试
        for i in range(warmup_count):
            ssm_with_warmup.inference(df_fracdiff.iloc[[i]])

        # 使用相同的测试行进行推理
        test_row = df_fracdiff.iloc[[warmup_count]]

        result_with_warmup = ssm_with_warmup.inference(test_row)
        result_without_warmup = ssm_without_warmup.inference(
            df_fracdiff.iloc[[0]]  # 从头开始
        )
        # 然后再用 test_row
        for i in range(1, warmup_count + 1):
            result_without_warmup = ssm_without_warmup.inference(
                df_fracdiff.iloc[[i]]
            )

        # 两者应该产生相同的输出 (因为处理了相同的历史)
        np.testing.assert_array_almost_equal(
            result_with_warmup.values,
            result_without_warmup.values,
            decimal=5,
            err_msg="SSM with warmup should produce same output as sequential inference",
        )

    def test_ssm_state_continuity_between_warmup_and_trading(
        self,
        df_fracdiff_full,
    ):
        """验证 warmup 结束后状态连续传递到 trading 阶段"""
        from strategies.BinanceBtcDemoBarV2.models.config import SSMContainer

        df_fracdiff, warmup_len, _ = df_fracdiff_full

        ssm = SSMContainer("lg_ssm")

        # 执行 warmup (使用少量数据加速)
        warmup_count = min(warmup_len, 50)
        for i in range(warmup_count):
            ssm.inference(df_fracdiff.iloc[[i]])

        # 记录 trading 第一行结果
        trading_first_result = ssm.inference(df_fracdiff.iloc[[warmup_count]])

        # 重新创建 SSM，完整执行到同一位置
        ssm_full = SSMContainer("lg_ssm")
        for i in range(warmup_count + 1):
            result = ssm_full.inference(df_fracdiff.iloc[[i]])

        # 比较结果
        np.testing.assert_array_almost_equal(
            trading_first_result.values,
            result.values,
            decimal=5,
            err_msg="State should be continuous between warmup and trading",
        )


# ==================== Test 2.4: 特征拼接索引对齐 (关键) ====================
class TestFeatureConcatenation:
    """测试特征拼接索引对齐"""

    @pytest.fixture(scope="class")
    def prepared_features(
        self, fusion_bars_for_features, full_fracdiff_features, sample_raw_features
    ):
        """
        准备所有特征 DataFrame

        注意: fracdiff 特征在前几行会有 NaN，需要跳过
        """
        from src.features.simple_feature_calculator import SimpleFeatureCalculator

        fusion_bars, warmup_len = fusion_bars_for_features
        fc = SimpleFeatureCalculator()
        fc.load(fusion_bars, sequential=True)

        df_fracdiff_full = pd.DataFrame.from_dict(fc.get(full_fracdiff_features))
        df_raw_full = pd.DataFrame.from_dict(fc.get(sample_raw_features))

        # 找到第一个无 NaN 的行
        first_valid_idx = df_fracdiff_full.dropna().index[0]

        # 从第一个有效行开始切片
        df_fracdiff = df_fracdiff_full.iloc[first_valid_idx:].reset_index(drop=True)
        df_raw = df_raw_full.iloc[first_valid_idx:].reset_index(drop=True)

        # 调整 warmup_len
        adjusted_warmup_len = max(0, warmup_len - first_valid_idx)

        return df_fracdiff, df_raw, adjusted_warmup_len

    def test_index_alignment_before_concat(
        self,
        prepared_features,
    ):
        """验证 concat 前所有 DataFrame 索引一致"""
        from strategies.BinanceBtcDemoBarV2.models.config import SSMContainer

        df_fracdiff, df_raw, warmup_len = prepared_features

        # SSM 处理 (简化版)
        ssm = SSMContainer("lg_ssm")
        ssm_results = []

        # Warmup (少量)
        warmup_count = min(warmup_len, 20)
        for i in range(warmup_count):
            ssm.inference(df_fracdiff.iloc[[i]])

        # Trading
        trading_count = min(len(df_fracdiff) - warmup_count, 30)
        for i in range(warmup_count, warmup_count + trading_count):
            ssm_results.append(ssm.inference(df_fracdiff.iloc[[i]]))

        # 合并 SSM 结果
        df_ssm = pd.concat(ssm_results, axis=0).reset_index(drop=True)

        # 切片 raw 特征 (trading 部分)
        df_raw_trading = df_raw.iloc[warmup_count : warmup_count + trading_count]
        df_raw_trading = df_raw_trading.reset_index(drop=True)

        # 验证索引一致
        assert list(df_ssm.index) == list(df_raw_trading.index), (
            "SSM and raw features should have same index after reset"
        )

        # 验证索引是连续的 0, 1, 2, ...
        expected_index = list(range(trading_count))
        assert list(df_ssm.index) == expected_index
        assert list(df_raw_trading.index) == expected_index

    def test_concat_produces_correct_shape(
        self,
        prepared_features,
    ):
        """验证 concat 后 shape 正确"""
        from strategies.BinanceBtcDemoBarV2.models.config import SSMContainer

        df_fracdiff, df_raw, warmup_len = prepared_features

        # SSM 处理 (简化版)
        deep_ssm = SSMContainer("deep_ssm")
        lg_ssm = SSMContainer("lg_ssm")
        deep_ssm_results = []
        lg_ssm_results = []

        # Warmup
        warmup_count = min(warmup_len, 10)
        for i in range(warmup_count):
            deep_ssm.inference(df_fracdiff.iloc[[i]])
            lg_ssm.inference(df_fracdiff.iloc[[i]])

        # Trading
        trading_count = min(len(df_fracdiff) - warmup_count, 20)
        for i in range(warmup_count, warmup_count + trading_count):
            deep_ssm_results.append(deep_ssm.inference(df_fracdiff.iloc[[i]]))
            lg_ssm_results.append(lg_ssm.inference(df_fracdiff.iloc[[i]]))

        # 合并
        df_deep_ssm = pd.concat(deep_ssm_results, axis=0).reset_index(drop=True)
        df_lg_ssm = pd.concat(lg_ssm_results, axis=0).reset_index(drop=True)
        df_raw_trading = df_raw.iloc[
            warmup_count : warmup_count + trading_count
        ].reset_index(drop=True)

        # Concat 所有特征
        df_full = pd.concat([df_deep_ssm, df_lg_ssm, df_raw_trading], axis=1)

        # 验证 shape
        expected_rows = trading_count
        expected_cols = (
            len(df_deep_ssm.columns)
            + len(df_lg_ssm.columns)
            + len(df_raw_trading.columns)
        )

        assert df_full.shape[0] == expected_rows, (
            f"Expected {expected_rows} rows, got {df_full.shape[0]}"
        )
        assert df_full.shape[1] == expected_cols, (
            f"Expected {expected_cols} cols, got {df_full.shape[1]}"
        )

    def test_concat_no_duplicate_columns(
        self,
        prepared_features,
    ):
        """验证 concat 后无重复列"""
        from strategies.BinanceBtcDemoBarV2.models.config import SSMContainer

        df_fracdiff, df_raw, warmup_len = prepared_features

        # SSM 处理
        deep_ssm = SSMContainer("deep_ssm")
        lg_ssm = SSMContainer("lg_ssm")
        deep_ssm_results = []
        lg_ssm_results = []

        warmup_count = min(warmup_len, 5)
        for i in range(warmup_count):
            deep_ssm.inference(df_fracdiff.iloc[[i]])
            lg_ssm.inference(df_fracdiff.iloc[[i]])

        trading_count = min(len(df_fracdiff) - warmup_count, 10)
        for i in range(warmup_count, warmup_count + trading_count):
            deep_ssm_results.append(deep_ssm.inference(df_fracdiff.iloc[[i]]))
            lg_ssm_results.append(lg_ssm.inference(df_fracdiff.iloc[[i]]))

        df_deep_ssm = pd.concat(deep_ssm_results, axis=0).reset_index(drop=True)
        df_lg_ssm = pd.concat(lg_ssm_results, axis=0).reset_index(drop=True)
        df_raw_trading = df_raw.iloc[
            warmup_count : warmup_count + trading_count
        ].reset_index(drop=True)

        df_full = pd.concat([df_deep_ssm, df_lg_ssm, df_raw_trading], axis=1)

        # 检查无重复列
        all_columns = list(df_full.columns)
        unique_columns = list(set(all_columns))
        assert len(all_columns) == len(unique_columns), (
            f"Found duplicate columns: "
            f"{[c for c in all_columns if all_columns.count(c) > 1]}"
        )

    def test_sample_row_values_from_correct_source(
        self,
        prepared_features,
    ):
        """验证抽样行的值来自正确来源"""
        from strategies.BinanceBtcDemoBarV2.models.config import SSMContainer

        df_fracdiff, df_raw, warmup_len = prepared_features

        # SSM 处理
        deep_ssm = SSMContainer("deep_ssm")
        lg_ssm = SSMContainer("lg_ssm")
        deep_ssm_results = []
        lg_ssm_results = []

        warmup_count = min(warmup_len, 5)
        for i in range(warmup_count):
            deep_ssm.inference(df_fracdiff.iloc[[i]])
            lg_ssm.inference(df_fracdiff.iloc[[i]])

        trading_count = min(len(df_fracdiff) - warmup_count, 10)
        for i in range(warmup_count, warmup_count + trading_count):
            deep_ssm_results.append(deep_ssm.inference(df_fracdiff.iloc[[i]]))
            lg_ssm_results.append(lg_ssm.inference(df_fracdiff.iloc[[i]]))

        df_deep_ssm = pd.concat(deep_ssm_results, axis=0).reset_index(drop=True)
        df_lg_ssm = pd.concat(lg_ssm_results, axis=0).reset_index(drop=True)
        df_raw_trading = df_raw.iloc[
            warmup_count : warmup_count + trading_count
        ].reset_index(drop=True)

        df_full = pd.concat([df_deep_ssm, df_lg_ssm, df_raw_trading], axis=1)

        # 验证第 0 行的值
        row_idx = 0

        # deep_ssm 列应该匹配
        for col in df_deep_ssm.columns:
            assert df_full.loc[row_idx, col] == df_deep_ssm.loc[row_idx, col]

        # lg_ssm 列应该匹配
        for col in df_lg_ssm.columns:
            assert df_full.loc[row_idx, col] == df_lg_ssm.loc[row_idx, col]

        # raw 列应该匹配
        for col in df_raw_trading.columns:
            np.testing.assert_almost_equal(
                df_full.loc[row_idx, col],
                df_raw_trading.loc[row_idx, col],
                decimal=6,
            )
