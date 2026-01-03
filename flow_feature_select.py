"""
Flow Feature Select - 批量特征筛选流水线

遍历 LOG_RETURN_LAG x PRED_NEXT 组合，执行全量特征筛选，结果记录到 CSV。

Usage:
    python flow_feature_select.py
"""

import json
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

from research.model_pick.candle_fetch import FusionCandles
from research.model_pick.features import ALL_FEATS
from research.model_pick.labeler import PipelineLabeler
from src.features.feature_selection import GrootCVConfig, GrootCVSelector
from src.features.pipeline import FeatureMaker, FeatureMakerConfig

# ============================================================================
# 配置参数
# ============================================================================
# 数据范围
START = "2022-08-01"
END = "2025-06-01"

# 搜索参数
LOG_RETURN_LAGS = [4, 5, 6, 7, 8]
PRED_NEXT_STEPS = [1, 2, 3]
LABEL_TYPE = "hard"  # "hard" (分类) 或 "direction" (回归)

# 特征筛选配置
GROOTCV_CUTOFF = 5

# 输出文件
OUTPUT_FILE = "feature_selection_results.csv"


def align_features_labels(
    global_features: pd.DataFrame,
    raw_label: np.ndarray,
    log_return_lag: int,
    pred_next: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    对齐特征和标签

    - label 开头缺少 log_return_lag 个值
    - feature 需要去掉开头的 NaN
    - 按 pred_next 进行 shift 对齐
    """
    # 截断 feature 开头 (对齐 label 的 lag)
    features = global_features.iloc[log_return_lag:]

    # shift 以对齐 PRED_NEXT
    features = features.iloc[:-pred_next]
    label = raw_label[pred_next:]

    # 去掉 feature 开头的 NaN
    na_mask = features.isna().any(axis=1).values
    features = features.iloc[~na_mask]
    label = label[~na_mask]

    assert len(features) == len(
        label
    ), f"Length mismatch: {len(features)} vs {len(label)}"
    return features, label


def run_single_selection(
    global_features: pd.DataFrame,
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
    label_type: str,
    cutoff: float,
) -> dict:
    """
    执行单次特征筛选

    Returns:
        包含筛选结果的字典
    """
    print(f"\n{'=' * 60}")
    print(
        f"[筛选] lag={log_return_lag}, pred_next={pred_next}, label_type={label_type}"
    )
    print("=" * 60)

    # 1. 生成标签
    labeler = PipelineLabeler(candles, log_return_lag)
    if label_type == "hard":
        raw_label = labeler.label_hard
    else:
        raw_label = labeler.label_direction

    # 2. 对齐特征和标签
    features, label = align_features_labels(
        global_features, raw_label, log_return_lag, pred_next
    )
    print(f"对齐后: {len(features)} 样本, {features.shape[1]} 特征")

    # 3. 特征筛选
    groot_config = GrootCVConfig(cutoff=cutoff)
    selector = GrootCVSelector(config=groot_config, verbose=True)
    selector.fit(features, label)

    selected = selector.selected_features_
    print(f"筛选结果: {len(selected)}/{features.shape[1]} 特征")

    return {
        "log_return_lag": log_return_lag,
        "pred_next": pred_next,
        "label_type": label_type,
        "n_total_features": features.shape[1],
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
    print(f"LABEL_TYPE: {LABEL_TYPE}")
    print(f"GROOTCV_CUTOFF: {GROOTCV_CUTOFF}")
    print("=" * 60)

    # 1. 获取 fusion candles
    print("\n[1/3] 加载 K 线数据...")
    candle_container = FusionCandles(
        exchange="Binance Perpetual Futures", symbol="BTC-USDT", timeframe="1m"
    )
    candles = candle_container.get_candles(START, END)
    print(f"K 线数据: {candles.shape}")

    # 2. 计算全局特征（只计算一次）
    print("\n[2/3] 计算全局特征...")
    feature_config = FeatureMakerConfig(
        feature_names=ALL_FEATS,
        ssm_state_dim=5,
        verbose=True,
    )
    feature_maker = FeatureMaker(feature_config)
    global_features = feature_maker.fit_transform(candles)
    print(f"全局特征: {global_features.shape}")

    # 3. 遍历所有参数组合进行筛选
    print("\n[3/3] 开始批量特征筛选...")
    combinations = list(product(LOG_RETURN_LAGS, PRED_NEXT_STEPS))
    total = len(combinations)
    results = []

    for idx, (lag, pred_next) in enumerate(combinations, 1):
        print(f"\n进度: {idx}/{total}")
        try:
            result = run_single_selection(
                global_features=global_features,
                candles=candles,
                log_return_lag=lag,
                pred_next=pred_next,
                label_type=LABEL_TYPE,
                cutoff=GROOTCV_CUTOFF,
            )
            results.append(result)
        except Exception as e:
            print(f"[ERROR] lag={lag}, pred_next={pred_next}: {e}")
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
                    "n_total_features",
                    "n_selected_features",
                ]
            ].to_string()
        )
    else:
        print("\n[WARNING] 没有成功的筛选结果")


if __name__ == "__main__":
    main()
