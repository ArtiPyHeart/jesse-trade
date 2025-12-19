"""
pipeline_build_models.py - 使用统一 FeaturePipeline 构建所有 LightGBM 模型

流程：
1. 读取 CSV 所有 selected_features → 去重 + 排序
2. 单一 FeaturePipeline (启用降维) → 一阶特征计算 + SSM 训练/transform + ARDVAE 降维
3. 保存 FeaturePipeline 到 MODEL_DIR/feature_pipeline/
4. 对每个 (lag, pred_next, model_type): 生成标签 → 对齐 → 训练 LightGBM → 保存

参数来源：
- 固定参数：TRAIN_TEST_SPLIT_DATE, CANDLE_START, CANDLE_END, GLOBAL_SEED, REDUCER_CONFIG 直接设置
- 调优参数：log_return_lag, pred_next, best_params, selected_features 从 model_search_results.csv 读取
"""

import json
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from jesse.helpers import date_to_timestamp
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold

from research.model_pick.candle_fetch import FusionCandles
from research.model_pick.feature_utils import (
    align_features_and_labels,
    build_model_config,
)
from research.model_pick.labeler import PipelineLabeler
from src.features.dimensionality_reduction import ARDVAEConfig
from src.features.pipeline import FeaturePipeline

# ============================================================
# 固定参数配置（直接设置，与 pipeline_find_model.py 保持一致）
# ============================================================
MODEL_DIR = Path("./strategies/BinanceBtcDemoBarV2/models")
PIPELINE_NAME = "feature_pipeline"
TRAIN_TEST_SPLIT_DATE = "2025-05-31"  # 训练集切分点
CANDLE_START = "2022-08-01"  # 与 pipeline_find_model.py 一致
CANDLE_END = "2025-12-15"  # 生产环境需要更长的数据范围
GLOBAL_SEED = 42
RESULTS_FILE = "model_search_results.csv"

# ARDVAE 降维器配置（固定，与 pipeline_find_model.py 保持一致）
REDUCER_CONFIG = ARDVAEConfig(
    max_latent_dim=512,  # over-complete 设计，ARD prior 自动确定 active dims
    kl_threshold=0.01,  # 判断维度是否 active 的阈值
    max_epochs=150,  # 与 pipeline_find_model.py 一致
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


def get_param_combinations_from_csv(
    df_params: pd.DataFrame,
) -> tuple[list[int], list[int]]:
    """
    从 CSV 获取 log_return_lag 和 pred_next 的唯一值组合

    Args:
        df_params: 包含 log_return_lag 和 pred_next 列的 DataFrame

    Returns:
        (log_return_lags, pred_next_steps) 排序后的唯一值列表
    """
    log_return_lags = sorted(df_params["log_return_lag"].unique().tolist())
    pred_next_steps = sorted(df_params["pred_next"].unique().tolist())
    return log_return_lags, pred_next_steps


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


def _f1_eval(preds: np.ndarray, eval_dataset: lgb.Dataset) -> tuple[str, float, bool]:
    """LightGBM 自定义评估函数：weighted F1 score"""
    y_true = eval_dataset.get_label()
    value = f1_score(y_true, preds > 0.5, average="weighted")
    return "f1", value, True  # (name, value, higher_is_better)


def _r2_eval(preds: np.ndarray, eval_dataset: lgb.Dataset) -> tuple[str, float, bool]:
    """LightGBM 自定义评估函数：R² score"""
    y_true = eval_dataset.get_label()
    ss_res = np.sum((y_true - preds) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return "r2", r2, True  # (name, value, higher_is_better)


def _determine_best_iterations_by_cv(
    params: dict,
    train_x: np.ndarray,
    train_y: np.ndarray,
    is_regression: bool,
    n_splits: int = 5,
    seed: int = GLOBAL_SEED,
) -> tuple[int, float]:
    """
    通过 5-fold CV 确定最佳迭代轮数

    Args:
        params: LightGBM 参数
        train_x: 训练特征
        train_y: 训练标签
        is_regression: 是否回归任务
        n_splits: CV 折数
        seed: 随机种子

    Returns:
        (best_iteration, cv_score): 最佳迭代轮数和 CV 评分
    """
    # 选择 CV 策略
    if is_regression:
        cv_folds = list(
            KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_x)
        )
        feval = _r2_eval
        metric_name = "r2"
    else:
        cv_folds = list(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
                train_x, train_y
            )
        )
        feval = _f1_eval
        metric_name = "f1"

    # 构建 Dataset
    dtrain = lgb.Dataset(train_x, train_y, free_raw_data=True, params={"max_bin": 255})

    # 执行 CV
    cv_results = lgb.cv(
        params,
        dtrain,
        num_boost_round=3000,
        folds=cv_folds,
        feval=feval,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # 获取最佳迭代轮数和评分
    best_iteration = len(cv_results[f"valid {metric_name}-mean"])
    best_score = cv_results[f"valid {metric_name}-mean"][-1]

    return best_iteration, best_score


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

    流程：
    1. 准备数据和参数
    2. 5-fold CV 确定最佳迭代轮数，打印评估指标
    3. 全量训练并保存模型

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
    model_name = f"{model_type}_L{lag}_N{pred_next}"
    model_path = MODEL_DIR / f"model_{model_name}.txt"

    if model_path.exists():
        print(f"Model {model_name} already exists, skipping")
        return

    # 获取最佳参数
    model_row = df_params[
        (df_params["log_return_lag"] == lag)
        & (df_params["pred_next"] == pred_next)
        & (df_params["model_type"] == ("regressor" if is_regression else "classifier"))
    ]
    best_params = model_row["best_params"].iloc[0].copy()

    # 添加调参时使用但未保存到 best_params 的固定参数
    # 这些参数必须与 model_tuning.py 中的设置保持一致
    fixed_params = {
        "boosting": "gbdt",
        "is_unbalance": False,  # 仅分类器使用，回归器会忽略
        "feature_fraction": 1.0,
        "feature_pre_filter": False,
    }
    best_params.update(fixed_params)

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

    # 转换为 numpy 数组（LightGBM 更高效）
    train_x_np = np.ascontiguousarray(train_x.to_numpy(dtype=np.float32))

    # Step 1: 5-fold CV 确定最佳迭代轮数
    metric_name = "R²" if is_regression else "F1"
    print(f"\n[{model_name}] Running 5-fold CV to determine best iterations...")
    best_iteration, cv_score = _determine_best_iterations_by_cv(
        best_params, train_x_np, train_y, is_regression, n_splits=5, seed=seed
    )
    print(
        f"[{model_name}] CV {metric_name}: {cv_score:.4f}, Best iterations: {best_iteration}"
    )

    # Step 2: 全量训练
    print(
        f"[{model_name}] Training final model with {train_x.shape[1]} features, {best_iteration} rounds..."
    )
    model = lgb.train(
        best_params,
        lgb.Dataset(train_x_np, train_y),
        num_boost_round=best_iteration,
    )

    # 保存模型
    model.save_model(model_path.resolve().as_posix())
    print(f"[{model_name}] Saved to {model_path}")


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

    # 3. 从 CSV 获取参数组合（调优参数）
    log_return_lags, pred_next_steps = get_param_combinations_from_csv(df_params)
    print("Parameter combinations from CSV:")
    print(f"  - log_return_lags: {log_return_lags}")
    print(f"  - pred_next_steps: {pred_next_steps}")

    # 4. 收集所有特征并去重
    all_features = collect_all_features_from_csv(df_params)
    print(f"Total unique features: {len(all_features)}")

    # 5. 构建统一 Pipeline
    print("\nBuilding unified FeaturePipeline...")
    unified_pipeline, reduced_features = build_unified_pipeline(candles, all_features)
    print(f"Reduced features shape: {reduced_features.shape}")

    # 6. 保存 Pipeline
    unified_pipeline.save(str(MODEL_DIR), PIPELINE_NAME)
    print(f"Saved FeaturePipeline to {MODEL_DIR / PIPELINE_NAME}")

    # 7. 训练所有模型
    print("\n" + "=" * 60)
    print("Training LightGBM models...")
    print("=" * 60)

    for lag in log_return_lags:
        for pred_next in pred_next_steps:
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
