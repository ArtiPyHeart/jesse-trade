"""
pipeline_build_models.py - 使用统一 FeaturePipeline 构建所有 LightGBM 模型

流程：
1. 读取 CSV 所有 selected_features → 去重 + 排序
2. 单一 FeaturePipeline (启用降维) → 一阶特征计算 + SSM 训练/transform + ARDVAE 降维
3. 保存 FeaturePipeline 到 MODEL_DIR/feature_pipeline/
4. 对每个 (lag, pred_next, model_type): 生成标签 → 对齐 → 训练 LightGBM → 保存
"""

from research.model_pick.candle_fetch import FusionCandles
from research.model_pick.labeler import PipelineLabeler
from research.model_pick.feature_utils import (
    build_model_config,
    align_features_and_labels,
)
from src.features.dimensionality_reduction import ARDVAEConfig
from src.features.pipeline import FeaturePipeline

import json
import random
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from jesse.helpers import date_to_timestamp

# 常量配置
MODEL_DIR = Path("./strategies/BinanceBtcDemoBarV2/models")
PIPELINE_NAME = "feature_pipeline"
TRAIN_TEST_SPLIT_DATE = "2025-05-31"
CANDLE_START = "2022-07-01"
CANDLE_END = "2025-11-25"
GLOBAL_SEED = 42

# ARDVAE 降维器配置
REDUCER_CONFIG = ARDVAEConfig(
    max_latent_dim=512,
    kl_threshold=0.01,
    max_epochs=200,
    patience=15,
    seed=GLOBAL_SEED,
)


def collect_all_features_from_csv(df_params: pd.DataFrame) -> list[str]:
    """
    从 CSV 收集所有 selected_features，去重并排序

    Args:
        df_params: 包含 selected_features 列的 DataFrame

    Returns:
        排序后的去重特征列表
    """
    all_features = set()
    for features in df_params["selected_features"]:
        all_features.update(features)
    return sorted(all_features)


def build_unified_pipeline(
    candles: np.ndarray,
    all_features: list[str],
) -> tuple[FeaturePipeline, pd.DataFrame]:
    """
    构建统一的 FeaturePipeline（启用降维）

    单一 Pipeline 完成：一阶特征计算 + SSM 训练/transform + ARDVAE 降维

    Args:
        candles: K线数据
        all_features: 去重后的全量特征列表

    Returns:
        (unified_pipeline, reduced_features)
    """
    config = build_model_config(
        selected_features=all_features,
        ssm_state_dim=5,
        reducer_config=REDUCER_CONFIG,
    )
    pipeline = FeaturePipeline(config)
    reduced_features = pipeline.fit_transform(candles)

    return pipeline, reduced_features


def build_model(
    df_params: pd.DataFrame,
    reduced_features: pd.DataFrame,
    candles: np.ndarray,
    lag: int,
    pred_next: int,
    is_regression: bool = False,
    seed: int = GLOBAL_SEED,
):
    """
    训练单个 LightGBM 模型

    Args:
        df_params: 包含 best_params 的参数 DataFrame
        reduced_features: 降维后的特征 DataFrame（列名为 "0", "1", ...）
        candles: K线数据
        lag: log_return_lag
        pred_next: 预测步数
        is_regression: 是否回归模型
        seed: 随机种子
    """
    model_type = "r" if is_regression else "c"
    MODEL_NAME = f"{model_type}_L{lag}_N{pred_next}"
    model_path = MODEL_DIR / f"model_{MODEL_NAME}.txt"

    if model_path.exists():
        print(f"Model {MODEL_NAME} already exists, skipping")
        return

    # 获取最佳参数
    model_row = df_params[
        (df_params["log_return_lag"] == lag)
        & (df_params["pred_next"] == pred_next)
        & (df_params["model_type"] == ("regressor" if is_regression else "classifier"))
    ]
    best_params = model_row["best_params"].iloc[0].copy()

    # 统一 seed
    best_params.update(
        {
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
        }
    )

    # 生成标签
    labeler = PipelineLabeler(candles, lag)
    raw_label = labeler.label_direction if is_regression else labeler.label_hard

    # 对齐特征与标签
    aligned_features, aligned_labels = align_features_and_labels(
        reduced_features, raw_label, pred_next, candles[:, 0]
    )

    # 划分训练集
    train_mask = aligned_features.index < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x = aligned_features[train_mask]
    train_y = aligned_labels[: train_mask.sum()]

    # 训练并保存
    print(f"Fitting {MODEL_NAME} with {train_x.shape[1]} features")
    model = lgb.train(best_params, lgb.Dataset(train_x, train_y))
    model.save_model(model_path.resolve().as_posix())
    print(f"Saved {MODEL_NAME}")


if __name__ == "__main__":
    # 设置全局 seed
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    print("=" * 60)
    print(f"Global random seed: {GLOBAL_SEED}")
    print("=" * 60 + "\n")

    # 1. 读取 CSV
    print("Loading model search results...")
    df_params = pd.read_csv(Path(__file__).parent / "model_search_results.csv")
    df_params["best_params"] = df_params["best_params"].apply(json.loads)
    df_params["selected_features"] = df_params["selected_features"].apply(json.loads)

    # 2. 加载 candles
    print("Loading candles...")
    candle_container = FusionCandles(
        exchange="Binance Perpetual Futures",
        symbol="BTC-USDT",
        timeframe="1m",
    )
    candles = candle_container.get_candles(CANDLE_START, CANDLE_END)
    print(f"Loaded {len(candles)} candles")

    # 3. 收集所有特征并去重
    all_features = collect_all_features_from_csv(df_params)
    print(f"Total unique features: {len(all_features)}")

    # 4. 构建统一 Pipeline
    print("\nBuilding unified FeaturePipeline...")
    unified_pipeline, reduced_features = build_unified_pipeline(candles, all_features)
    print(f"Reduced features shape: {reduced_features.shape}")

    # 5. 保存 Pipeline
    unified_pipeline.save(str(MODEL_DIR), PIPELINE_NAME)
    print(f"Saved FeaturePipeline to {MODEL_DIR / PIPELINE_NAME}")

    # 6. 训练所有模型
    print("\n" + "=" * 60)
    print("Training LightGBM models...")
    print("=" * 60)

    for lag in range(4, 8):
        for pred_next in range(1, 4):
            # Classifier
            build_model(
                df_params,
                reduced_features,
                candles,
                lag,
                pred_next,
                is_regression=False,
            )
            # Regressor
            build_model(
                df_params, reduced_features, candles, lag, pred_next, is_regression=True
            )

    print("\n" + "=" * 60)
    print("All models built successfully!")
    print("=" * 60)
