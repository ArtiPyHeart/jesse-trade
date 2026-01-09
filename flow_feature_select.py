"""
Flow Feature Select - 批量特征筛选流水线

遍历 LOG_RETURN_LAG x PRED_NEXT 组合，使用 GMMLabeler 生成标签，
通过 SimpleFeatureCalculator 计算特征，执行全量特征筛选，结果记录到 CSV。

Usage:
    python flow_feature_select.py
"""

import json
from datetime import datetime
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
from jesse import helpers, research

from research.labeler.gmm_labeler import GMMLabeler
from research.utils import align_features_labels
from src.bars.fusion.demo import DemoBar
from src.features.feature_selection import GrootCVConfig, GrootCVSelector
from src.features.simple_feature_calculator import SimpleFeatureCalculator
from src.features.simple_feature_calculator.buildin.feature_names import (
    BUILDIN_FEATURES,
)

# ============================================================================
# 配置参数
# ============================================================================
# 数据范围
START = "2022-08-01"
END = "2025-06-01"

# 搜索参数
LOG_RETURN_LAGS = [4, 5, 6, 7, 8, 9]  # GMMLabeler 的 lag_n
PRED_NEXT_STEPS = [1, 2, 3]  # 预测时间范围
LABEL_TYPES: list[Literal["hard", "direction"]] = ["hard", "direction"]

# 特征筛选配置
GROOTCV_CUTOFF = 5

# 输出文件
OUTPUT_FILE = "feature_selection_results.csv"

# ============================================================================
# 特征列表构建
# ============================================================================
WINDOW = 20

# 基础 OHLC dt 特征
BASIC = ["bar_open_dt", "bar_high_dt", "bar_low_dt", "bar_close_dt"]

# 关键波动率和动量指标（用于极值特征）
KEY_VOLATILITY_INDICATORS = ["natr", "bekker_parkinson_vol", "corwin_schultz_estimator"]
KEY_MOMENTUM_INDICATORS = ["williams_r", "fisher", "mod_rsi", "adaptive_rsi"]
KEY_INDICATORS = BASIC + KEY_VOLATILITY_INDICATORS + KEY_MOMENTUM_INDICATORS

# 基础 OHLC 的高级变换特征
basic_hurst_feats = [f"{i}_hurst{WINDOW}" for i in BASIC]
basic_curv_feats = [f"{i}_curv{WINDOW}" for i in BASIC]
basic_phent_feats = [f"{i}_phent{WINDOW}" for i in BASIC]

# 所有 BUILDIN_FEATURES 的统计特征
mean_feats = [f"{i}_mean{WINDOW}" for i in BUILDIN_FEATURES]
median_feats = [f"{i}_median{WINDOW}" for i in BUILDIN_FEATURES]
std_feats = [f"{i}_std{WINDOW}" for i in BUILDIN_FEATURES]
skew_feats = [f"{i}_skew{WINDOW}" for i in BUILDIN_FEATURES]
kurt_feats = [f"{i}_kurt{WINDOW}" for i in BUILDIN_FEATURES]

# 极值特征（支撑阻力、突破检测）
max_feats = [f"{i}_max{WINDOW}" for i in KEY_INDICATORS]
min_feats = [f"{i}_min{WINDOW}" for i in KEY_INDICATORS]

# 归一化特征（相对位置、超买超卖）
norm_feats = [f"{i}_norm{WINDOW}" for i in KEY_INDICATORS]
zscore_feats = [f"{i}_zscore{WINDOW}" for i in KEY_INDICATORS]

# 高级拓扑/分形特征
hurst_feats = [f"{i}_hurst{WINDOW}" for i in BUILDIN_FEATURES]
curv_feats = [f"{i}_curv{WINDOW}" for i in BUILDIN_FEATURES]
phent_feats = [f"{i}_phent{WINDOW}" for i in BUILDIN_FEATURES]

# 差分特征（动量）
dt_feats = [f"{i}_dt" for i in BUILDIN_FEATURES]
ddt_feats = [f"{i}_ddt" for i in BUILDIN_FEATURES]

# 组合所有非滞后特征
_feats = (
    list(BUILDIN_FEATURES)
    # 基础 OHLC 高级特征
    + basic_hurst_feats
    + basic_curv_feats
    + basic_phent_feats
    # 统计特征
    + mean_feats
    + median_feats
    + std_feats
    + skew_feats
    + kurt_feats
    # 极值特征
    + max_feats
    + min_feats
    # 归一化特征
    + norm_feats
    + zscore_feats
    # 高级特征
    + hurst_feats
    + curv_feats
    # 差分特征
    + dt_feats
    + ddt_feats
)

# 滞后特征（时序信息）
lag_feats = [f"{i}_lag{lag}" for i in _feats for lag in range(1, 4)]

# 完整特征集
FEATURE_NAMES = _feats + phent_feats + lag_feats


def run_single_selection(
    features_df: pd.DataFrame,
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
    label_type: Literal["hard", "direction"],
    cutoff: float,
) -> dict:
    """
    执行单次特征筛选

    Args:
        features_df: 全局特征 DataFrame
        candles: K 线数据
        log_return_lag: GMMLabeler 的 lag_n 参数
        pred_next: 预测时间范围
        label_type: 标签类型 ("hard" 或 "direction")
        cutoff: GrootCV 筛选阈值

    Returns:
        包含筛选结果的字典
    """
    print(f"\n{'=' * 60}")
    print(
        f"[筛选] log_return_lag={log_return_lag}, pred_next={pred_next}, "
        f"label_type={label_type}"
    )
    print("=" * 60)

    # 1. 生成标签
    labeler = GMMLabeler(candles, lag_n=log_return_lag, verbose=False)
    if label_type == "hard":
        raw_labels = labeler.label_hard_state
    else:
        raw_labels = labeler.label_direction_force

    print(f"标签生成完成: {len(raw_labels)} 样本")

    # 2. 对齐特征和标签
    aligned_features, aligned_labels = align_features_labels(
        features_df,
        raw_labels,
        log_return_lag=log_return_lag,
        pred_next=pred_next,
    )
    print(f"对齐后: {len(aligned_features)} 样本, {aligned_features.shape[1]} 特征")

    # 3. 特征筛选
    groot_config = GrootCVConfig(cutoff=cutoff)
    selector = GrootCVSelector(config=groot_config, verbose=True)
    selector.fit(aligned_features, aligned_labels)

    selected = selector.selected_features_
    print(f"筛选结果: {len(selected)}/{aligned_features.shape[1]} 特征")

    return {
        "log_return_lag": log_return_lag,
        "pred_next": pred_next,
        "label_type": label_type,
        "n_total_features": aligned_features.shape[1],
        "n_selected_features": len(selected),
        "selected_features": json.dumps(selected),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    print("=" * 60)
    print("Flow Feature Select - 批量特征筛选")
    print("=" * 60)
    print(f"数据范围: {START} ~ {END}")
    print(f"LOG_RETURN_LAGS: {LOG_RETURN_LAGS}")
    print(f"PRED_NEXT_STEPS: {PRED_NEXT_STEPS}")
    print(f"LABEL_TYPES: {LABEL_TYPES}")
    print(f"GROOTCV_CUTOFF: {GROOTCV_CUTOFF}")
    print(f"特征数量: {len(FEATURE_NAMES)}")
    print("=" * 60)

    # 1. 获取 fusion candles
    print("\n[1/3] 加载 K 线数据...")
    _, raw_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp(START),
        helpers.date_to_timestamp(END),
        warmup_candles_num=0,
        caching=False,
        is_for_jesse=False,
    )
    print(f"原始 K 线数据: {raw_candles.shape}")

    # 转换为 Fusion Bar
    bar = DemoBar(max_bars=-1)
    bar.update_with_candles(raw_candles)
    candles = bar.get_fusion_bars()
    print(f"Fusion K 线数据: {candles.shape}")

    # 2. 计算全局特征（只计算一次）
    print("\n[2/3] 计算全局特征...")
    calc = SimpleFeatureCalculator(verbose=True)
    calc.load(candles, sequential=True)

    # 批量获取特征
    features_dict = calc.get(FEATURE_NAMES)
    features_df = pd.DataFrame(features_dict)
    print(f"全局特征: {features_df.shape}")

    # 3. 遍历所有参数组合进行筛选
    print("\n[3/3] 开始批量特征筛选...")
    combinations = list(product(LOG_RETURN_LAGS, PRED_NEXT_STEPS, LABEL_TYPES))
    total = len(combinations)
    results = []

    for idx, (log_return_lag, pred_next, label_type) in enumerate(combinations, 1):
        print(f"\n进度: {idx}/{total}")
        try:
            result = run_single_selection(
                features_df=features_df,
                candles=candles,
                log_return_lag=log_return_lag,
                pred_next=pred_next,
                label_type=label_type,
                cutoff=GROOTCV_CUTOFF,
            )
            results.append(result)
        except Exception as e:
            print(
                f"[ERROR] log_return_lag={log_return_lag}, pred_next={pred_next}, "
                f"label_type={label_type}: {e}"
            )
            continue

    # 4. 保存结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n{'=' * 60}")
        print(f"完成! 结果已保存到: {OUTPUT_FILE}")
        print(f"共 {len(results)} 条记录")
        print("=" * 60)

        # 打印汇总
        print("\n筛选结果汇总:")
        print(
            df[
                [
                    "log_return_lag",
                    "pred_next",
                    "label_type",
                    "n_total_features",
                    "n_selected_features",
                ]
            ].to_string()
        )
    else:
        print("\n[WARNING] 没有成功的筛选结果")


if __name__ == "__main__":
    main()
