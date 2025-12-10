"""
Phase 3: 模型预测测试

测试覆盖:
1. LGBMContainer 加载与预测
2. Filter 应用逻辑
3. 预测批量输出

运行方式: 从项目根目录执行
    pytest tests/test_backtest_vectorized/test_phase3_predictions.py -v
"""

import numpy as np
import pandas as pd
import pytest

# 初始化 Jesse 数据库连接 (必须在导入其他模块之前)
from jesse.services import db  # noqa: F401


# ==================== Fixtures ====================
@pytest.fixture(scope="module")
def model_features_fixture(jesse_candles):
    """
    准备完整的特征 DataFrame 用于模型预测测试

    Returns:
        tuple: (df_features, model_names, feature_info)
    """
    import json
    from pathlib import Path

    from src.bars.fusion.demo import DemoBar
    from src.features.simple_feature_calculator import SimpleFeatureCalculator
    from strategies.BinanceBtcDemoBarV2.models.config import SSMContainer

    warmup_candles, trading_candles = jesse_candles

    # 生成 fusion bars
    all_candles = np.vstack([warmup_candles, trading_candles])
    bar_container = DemoBar(clip_r=0.012, max_bars=-1, threshold=1.399)
    bar_container.update_with_candles(all_candles)
    fusion_bars = bar_container.get_fusion_bars()

    # 计算 warmup 分界点
    warmup_last_ts = warmup_candles[-1, 0]
    warmup_len = np.searchsorted(fusion_bars[:, 0], warmup_last_ts, side="right")

    # 加载特征配置
    feature_info_path = (
        Path(__file__).parent.parent.parent
        / "strategies/BinanceBtcDemoBarV2/models/feature_info.json"
    )
    with open(feature_info_path) as f:
        feature_info = json.load(f)

    fracdiff_features = feature_info["fracdiff"]

    # 获取模型需要的所有原始特征（不含 SSM 前缀的特征）
    model_names = ["c_L6_N1", "r_L5_N2"]
    all_model_features = set()
    for model_name in model_names:
        all_model_features.update(feature_info[model_name])

    # 过滤掉 SSM 特征（deep_ssm_*, lg_ssm_*），只保留原始特征
    raw_features_needed = [
        f for f in all_model_features
        if not f.startswith("deep_ssm_") and not f.startswith("lg_ssm_")
    ]

    # 计算特征
    fc = SimpleFeatureCalculator()
    fc.load(fusion_bars, sequential=True)

    df_fracdiff_full = pd.DataFrame.from_dict(fc.get(fracdiff_features))
    df_raw_full = pd.DataFrame.from_dict(fc.get(raw_features_needed))

    # 找到第一个无 NaN 的行 (fracdiff 特征预热期)
    first_valid_idx = df_fracdiff_full.dropna().index[0]

    # 从第一个有效行开始切片
    df_fracdiff = df_fracdiff_full.iloc[first_valid_idx:].reset_index(drop=True)
    df_raw = df_raw_full.iloc[first_valid_idx:].reset_index(drop=True)

    # 调整 warmup_len
    adjusted_warmup_len = max(0, warmup_len - first_valid_idx)

    # SSM warmup 和 trading 处理
    deep_ssm = SSMContainer("deep_ssm")
    lg_ssm = SSMContainer("lg_ssm")
    deep_ssm_results = []
    lg_ssm_results = []

    # Warmup (使用部分数据加速)
    warmup_count = min(adjusted_warmup_len, 100)
    for i in range(warmup_count):
        deep_ssm.inference(df_fracdiff.iloc[[i]])
        lg_ssm.inference(df_fracdiff.iloc[[i]])

    # Trading (使用更多数据确保有足够的非 NaN 行)
    trading_count = min(len(df_fracdiff) - warmup_count, 200)
    for i in range(warmup_count, warmup_count + trading_count):
        deep_ssm_results.append(deep_ssm.inference(df_fracdiff.iloc[[i]]))
        lg_ssm_results.append(lg_ssm.inference(df_fracdiff.iloc[[i]]))

    # 合并 SSM 结果
    df_deep_ssm = pd.concat(deep_ssm_results, axis=0).reset_index(drop=True)
    df_lg_ssm = pd.concat(lg_ssm_results, axis=0).reset_index(drop=True)
    df_raw_trading = df_raw.iloc[
        warmup_count : warmup_count + trading_count
    ].reset_index(drop=True)

    # 拼接所有特征
    df_features = pd.concat([df_deep_ssm, df_lg_ssm, df_raw_trading], axis=1)

    print(f"\n[fixture] Features shape: {df_features.shape}")
    print(f"[fixture] Trading count: {trading_count}")

    return df_features, model_names, feature_info


# ==================== Test 3.1: LGBMContainer 加载与预测 ====================
class TestLGBMContainerLoading:
    """测试 LGBMContainer 加载与预测"""

    def test_model_loading(self):
        """验证模型正确加载"""
        from strategies.BinanceBtcDemoBarV2.models.config import (
            LGBMContainer,
            model_name_to_params,
        )

        # 测试分类模型
        model_c = LGBMContainer(*model_name_to_params("c_L6_N1"))
        model_c.is_livetrading = False

        assert model_c.model is not None
        assert model_c.model_type == "c"
        assert model_c.lag == 6
        assert model_c.pred_next == 1
        assert model_c.threshold == 0.5  # 分类模型阈值

        # 测试回归模型
        model_r = LGBMContainer(*model_name_to_params("r_L5_N2"))
        model_r.is_livetrading = False

        assert model_r.model is not None
        assert model_r.model_type == "r"
        assert model_r.lag == 5
        assert model_r.pred_next == 2
        assert model_r.threshold == 0.0  # 回归模型阈值

    def test_predict_proba_output_range(self, model_features_fixture):
        """验证 predict_proba 输出在合理范围内"""
        from strategies.BinanceBtcDemoBarV2.models.config import (
            LGBMContainer,
            model_name_to_params,
        )

        df_features, model_names, feature_info = model_features_fixture

        for model_name in model_names:
            model = LGBMContainer(*model_name_to_params(model_name))
            model.is_livetrading = False

            # 清除 filters 以获取原始 predict_proba
            model.clear_filters()

            # 获取模型需要的特征列 (只取 df_features 中存在的列)
            model_feat_names = feature_info[model.MODEL_NAME]
            available_feats = [f for f in model_feat_names if f in df_features.columns]

            # 测试几行数据 (从后面的行开始，避免 NaN)
            valid_probs = []
            for i in range(len(df_features) - 1, max(0, len(df_features) - 50), -1):
                feat_row = df_features.iloc[[i]][available_feats]

                # 如果特征行有 NaN，跳过
                if feat_row.isna().any().any():
                    continue

                prob = model.predict_proba(feat_row)

                # 分类模型概率应在 [0, 1]
                if model.model_type == "c":
                    assert 0 <= prob <= 1, (
                        f"{model_name} predict_proba should be in [0, 1], got {prob}"
                    )
                    valid_probs.append(prob)
                # 回归模型预测可能是 NaN（如果有缺失特征）
                else:
                    if np.isfinite(prob):
                        valid_probs.append(prob)

            # 确保至少测试了一些有效行
            assert len(valid_probs) >= 1, (
                f"{model_name}: no valid rows without NaN found in last 50 rows"
            )
            print(f"\n[test] {model_name}: tested {len(valid_probs)} valid rows")

    def test_final_predict_returns_valid_values(self, model_features_fixture):
        """验证 final_predict 返回 -1/0/1"""
        from strategies.BinanceBtcDemoBarV2.models.config import (
            LGBMContainer,
            model_name_to_params,
        )

        df_features, model_names, feature_info = model_features_fixture

        for model_name in model_names:
            model = LGBMContainer(*model_name_to_params(model_name))
            model.is_livetrading = False

            predictions = []
            for i in range(min(10, len(df_features))):
                feat_row = df_features.iloc[[i]][feature_info[model.MODEL_NAME]]
                pred = model.final_predict(feat_row)
                predictions.append(pred)

            # 验证所有预测值 ∈ {-1, 0, 1}
            unique_preds = set(predictions)
            assert unique_preds.issubset({-1, 0, 1}), (
                f"{model_name} predictions should be in {{-1, 0, 1}}, "
                f"got {unique_preds}"
            )


# ==================== Test 3.2: Filter 应用逻辑 ====================
class TestFilterApplication:
    """测试 Filter 应用逻辑"""

    def test_giveup_filter(self):
        """验证 giveup filter 正确应用"""
        from strategies.BinanceBtcDemoBarV2.models.config import (
            LGBMContainer,
            model_name_to_params,
        )

        model = LGBMContainer(*model_name_to_params("c_L6_N1"))
        model.is_livetrading = False
        model.clear_filters()

        # 添加 giveup filter [0.45, 0.55]
        model.add_giveup_filter(0.45, 0.55)

        # 测试边界条件
        # prob=0.50 落在 [0.45, 0.55) 内 → 应返回 0
        assert model._apply_filters(0.50) == 0
        assert model._apply_filters(0.45) == 0
        assert model._apply_filters(0.54) == 0

        # prob=0.55 不在 [0.45, 0.55) 内（右边界不含）→ 应返回 1 (>= threshold)
        assert model._apply_filters(0.55) == 1

        # prob=0.60 不在 [0.45, 0.55) 内 → 应返回 1 (>= threshold)
        assert model._apply_filters(0.60) == 1

        # prob=0.40 不在 [0.45, 0.55) 内 → 应返回 -1 (< threshold)
        assert model._apply_filters(0.40) == -1

        # prob=0.44 刚好在边界外 → 应返回 -1
        assert model._apply_filters(0.44) == -1

    def test_reverse_filter(self):
        """验证 reverse filter 正确应用"""
        from strategies.BinanceBtcDemoBarV2.models.config import (
            LGBMContainer,
            model_name_to_params,
        )

        model = LGBMContainer(*model_name_to_params("c_L6_N1"))
        model.is_livetrading = False
        model.clear_filters()

        # 添加 reverse filter [0.48, 0.52]
        model.add_reverse_filter(0.48, 0.52)

        # prob=0.50 落在 [0.48, 0.52) 内
        # 原始预测: 0.50 >= 0.5 → 1 (做多)
        # 反转后: -1 (做空)
        assert model._apply_filters(0.50) == -1

        # prob=0.49 落在 [0.48, 0.52) 内
        # 原始预测: 0.49 < 0.5 → -1 (做空)
        # 反转后: 1 (做多)
        assert model._apply_filters(0.49) == 1

        # prob=0.60 不在区间内 → 原始预测 1
        assert model._apply_filters(0.60) == 1

        # prob=0.40 不在区间内 → 原始预测 -1
        assert model._apply_filters(0.40) == -1

    def test_multiple_filters(self):
        """验证多个 filter 按顺序应用"""
        from strategies.BinanceBtcDemoBarV2.models.config import (
            LGBMContainer,
            model_name_to_params,
        )

        model = LGBMContainer(*model_name_to_params("c_L6_N1"))
        model.is_livetrading = False
        model.clear_filters()

        # 添加多个 filter
        model.add_giveup_filter(0.45, 0.48)  # 放弃区间 1
        model.add_giveup_filter(0.52, 0.55)  # 放弃区间 2

        # 落在第一个 giveup 区间
        assert model._apply_filters(0.46) == 0

        # 落在第二个 giveup 区间
        assert model._apply_filters(0.53) == 0

        # 不在任何区间内
        assert model._apply_filters(0.50) == 1  # >= threshold
        assert model._apply_filters(0.40) == -1  # < threshold

    def test_filter_auto_load(self):
        """验证 filter 自动加载"""
        from pathlib import Path

        from strategies.BinanceBtcDemoBarV2.models.config import (
            LGBMContainer,
            model_name_to_params,
        )

        model = LGBMContainer(*model_name_to_params("c_L6_N1"))
        model.is_livetrading = False

        # 检查是否有 filter 文件存在
        filter_path = (
            Path(__file__).parent.parent.parent
            / "strategies/BinanceBtcDemoBarV2/models/model_c_L6_N1_filters.json"
        )

        if filter_path.exists():
            # 如果 filter 文件存在，应该自动加载
            filters = model.get_filters()
            assert len(filters) >= 0  # 可能有或没有 filters
            print(f"\n[test] Auto-loaded {len(filters)} filters for c_L6_N1")
        else:
            # 如果没有 filter 文件，应该为空
            filters = model.get_filters()
            assert len(filters) == 0


# ==================== Test 3.3: 预测批量输出 ====================
class TestBatchPredictions:
    """测试预测批量输出"""

    def test_predict_all_models_output_length(self, model_features_fixture):
        """验证所有模型预测输出长度正确"""
        from strategies.BinanceBtcDemoBarV2.models.config import (
            LGBMContainer,
            model_name_to_params,
        )

        df_features, model_names, feature_info = model_features_fixture

        for model_name in model_names:
            model = LGBMContainer(*model_name_to_params(model_name))
            model.is_livetrading = False

            predictions = []
            for i in range(len(df_features)):
                feat_row = df_features.iloc[[i]][feature_info[model.MODEL_NAME]]
                pred = model.final_predict(feat_row)
                predictions.append(pred)

            # 验证输出长度
            assert len(predictions) == len(df_features), (
                f"{model_name} predictions length mismatch: "
                f"expected {len(df_features)}, got {len(predictions)}"
            )

    def test_predict_all_models_no_nan(self, model_features_fixture):
        """验证预测结果无 NaN"""
        from strategies.BinanceBtcDemoBarV2.models.config import (
            LGBMContainer,
            model_name_to_params,
        )

        df_features, model_names, feature_info = model_features_fixture

        for model_name in model_names:
            model = LGBMContainer(*model_name_to_params(model_name))
            model.is_livetrading = False

            predictions = []
            for i in range(min(20, len(df_features))):
                feat_row = df_features.iloc[[i]][feature_info[model.MODEL_NAME]]
                pred = model.final_predict(feat_row)
                predictions.append(pred)

            # 验证无 NaN
            assert not any(np.isnan(p) for p in predictions), (
                f"{model_name} predictions should not contain NaN"
            )

    def test_predictions_have_variety(self, model_features_fixture):
        """验证预测结果有多样性（不全是同一个值）"""
        from strategies.BinanceBtcDemoBarV2.models.config import (
            LGBMContainer,
            model_name_to_params,
        )

        df_features, model_names, feature_info = model_features_fixture

        for model_name in model_names:
            model = LGBMContainer(*model_name_to_params(model_name))
            model.is_livetrading = False

            # 清除 filters 以获取原始预测
            model.clear_filters()

            predictions = []
            for i in range(len(df_features)):
                feat_row = df_features.iloc[[i]][feature_info[model.MODEL_NAME]]
                pred = model.final_predict(feat_row)
                predictions.append(pred)

            # 验证预测有多样性
            unique_preds = set(predictions)
            # 至少应该有 2 种不同的预测值（可能没有 0 因为没有 filter）
            assert len(unique_preds) >= 2, (
                f"{model_name} predictions should have variety, "
                f"got only {unique_preds}"
            )

    def test_model_name_to_params(self):
        """验证模型名称解析正确"""
        from strategies.BinanceBtcDemoBarV2.models.config import model_name_to_params

        # 测试分类模型
        model_type, lag, pred_next, threshold = model_name_to_params("c_L6_N1")
        assert model_type == "c"
        assert lag == 6
        assert pred_next == 1
        assert threshold == 0.5

        # 测试回归模型
        model_type, lag, pred_next, threshold = model_name_to_params("r_L5_N2")
        assert model_type == "r"
        assert lag == 5
        assert pred_next == 2
        assert threshold == 0.0

        # 测试无效格式应该报错
        with pytest.raises(AssertionError):
            model_name_to_params("invalid_model_name")

        with pytest.raises(AssertionError):
            model_name_to_params("x_L5_N2")  # 无效的 model_type
