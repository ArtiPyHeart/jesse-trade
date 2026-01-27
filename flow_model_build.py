"""
Flow Model Build - 批量模型构建流水线

基于 feature_selection_results.csv 构建所有模型：
1. 加载 Jesse candles -> DemoBar -> Fusion candles
2. 从 CSV 提取实际需要的特征 (去重)
3. 全局 SimpleFeatureCalculator 计算特征
4. 遍历每个模型配置：
   - 生成标签 (GMMLabeler)
   - 提取选定特征
   - ARDVAE 降维
   - Optuna 调参
   - 全量训练最终模型
   - 持久化到策略目录

Usage:
    python flow_model_build.py
"""

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from jesse import helpers, research
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from research.labeler.gmm_labeler import GMMLabeler
from research.utils import align_features_labels
from src.bars.fusion.demo import DemoBar
from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig
from src.features.simple_feature_calculator import SimpleFeatureCalculator
from src.utils.drop_na import drop_na_and_align_x_and_y
from src.utils.env_dates import get_env_date, load_env_values

# ============================================================================
# 配置参数
# ============================================================================
# 策略设定
STRATEGY = "BnBtcDemoBar"

# 训练集时间范围（与 flow_feature_select.py 一致）
ENV_VALUES = load_env_values(Path(".env"))
TRAIN_START = get_env_date("TRAIN_START_DATE", ENV_VALUES)
TRAIN_END = get_env_date("TRAIN_END_DATE", ENV_VALUES)

# 输入文件
FEATURE_SELECTION_FILE = "feature_selection_results.csv"

# 模型持久化目录
MODELS_DIR = Path(f"strategies/{STRATEGY}/models")

# ARDVAE 配置
ARDVAE_MAX_LATENT_DIM = 512
ARDVAE_MAX_EPOCHS = 200

# LGB 调参配置
OPTUNA_TRIALS = 200
CV_FOLDS = 5


# ============================================================================
# 特征提取
# ============================================================================
def extract_required_features(csv_path: str) -> list[str]:
    """
    从 feature_selection_results.csv 提取所有模型使用的特征并去重排序

    Args:
        csv_path: CSV 文件路径

    Returns:
        排序后的唯一特征列表
    """
    df = pd.read_csv(csv_path)
    all_features = set()

    for features_json in df["selected_features"]:
        features = json.loads(features_json)
        all_features.update(features)

    return sorted(all_features)


# ============================================================================
# 调参函数
# ============================================================================
METRIC = "f1"


def _eval_metric_f1(preds, eval_dataset):
    """LightGBM 自定义评估指标: weighted F1"""
    y_true = eval_dataset.get_label()
    value = f1_score(y_true, preds > 0.5, average="weighted")
    return METRIC, value, True


def tune_classifier(train_x: pd.DataFrame, train_y: np.ndarray) -> tuple[dict, float]:
    """
    分类模型调参（Optuna + 5-fold CV）

    Args:
        train_x: 特征 DataFrame
        train_y: 标签数组

    Returns:
        (best_params, best_score): 最优参数和最优 F1 分数
    """
    x, y = drop_na_and_align_x_and_y(train_x, train_y)
    print(f"{train_x.shape[1]} features for tuning")

    # LightGBM prefers contiguous float32 arrays
    x = np.ascontiguousarray(x.to_numpy(dtype=np.float32))

    # 固定 max_bin 参数，使用 free_raw_data=True 释放原始数据
    dtrain = lgb.Dataset(x, y, free_raw_data=True, params={"max_bin": 255})
    cv_folds = list(KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42).split(x, y))

    def objective(trial):
        # 参数范围针对降维后 ~20 维特征优化
        max_depth = trial.suggest_int("max_depth", 4, 8)
        max_leaves = 2**max_depth
        num_leaves = trial.suggest_int("num_leaves", 16, min(256, max_leaves))
        bagging_freq = trial.suggest_categorical("bagging_freq", [0, 1])
        if bagging_freq == 1:
            bagging_fraction = trial.suggest_float("bagging_fraction", 0.7, 1.0)
        else:
            bagging_fraction = 1.0

        params = {
            "objective": "binary",
            "num_threads": -1,
            "verbose": -1,
            "boosting": "gbdt",
            "is_unbalance": False,
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 100.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "feature_pre_filter": False,
        }

        pruning_cb = LightGBMPruningCallback(trial, METRIC)
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            pruning_cb,
        ]
        model_res = lgb.cv(
            params,
            dtrain,  # noqa: F821
            num_boost_round=3000,
            folds=cv_folds,
            feval=_eval_metric_f1,
            callbacks=callbacks,
        )
        return model_res[f"valid {METRIC}-mean"][-1]

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=50,
            multivariate=True,
            constant_liar=False,
            warn_independent_sampling=False,
        ),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=1, show_progress_bar=True)

    params = {
        "objective": "binary",
        "num_threads": -1,
        "verbose": -1,
        **study.best_params,
    }
    if params.get("bagging_freq", 0) == 0:
        params.setdefault("bagging_fraction", 1.0)
    best_value = study.best_value

    del study
    del dtrain
    gc.collect()

    return params, best_value


def tune_regressor(train_x: pd.DataFrame, train_y: np.ndarray) -> tuple[dict, float]:
    """
    回归模型调参（Optuna + 5-fold CV）

    Args:
        train_x: 特征 DataFrame
        train_y: 标签数组

    Returns:
        (best_params, best_score): 最优参数和最优 R² 分数
    """
    x, y = drop_na_and_align_x_and_y(train_x, train_y)
    print(f"{train_x.shape[1]} features for tuning")

    # LightGBM prefers contiguous float32 arrays
    x = np.ascontiguousarray(x.to_numpy(dtype=np.float32))

    # 固定 max_bin 参数
    dtrain = lgb.Dataset(x, y, free_raw_data=True, params={"max_bin": 255})
    cv_folds = list(KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42).split(x))

    # 预计算训练集标签的方差，用于计算 R²
    y_var = np.var(y)

    def r2_eval(preds, eval_dataset):
        y_true = eval_dataset.get_label()
        mse = np.mean((y_true - preds) ** 2)
        r2 = 1 - (mse / y_var)
        return "r2", r2, True

    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 4, 8)
        max_leaves = 2**max_depth
        num_leaves = trial.suggest_int("num_leaves", 16, min(256, max_leaves))
        bagging_freq = trial.suggest_categorical("bagging_freq", [0, 1])
        if bagging_freq == 1:
            bagging_fraction = trial.suggest_float("bagging_fraction", 0.7, 1.0)
        else:
            bagging_fraction = 1.0

        params = {
            "objective": "regression",
            "num_threads": -1,
            "verbose": -1,
            "boosting": "gbdt",
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 100.0),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "feature_pre_filter": False,
        }

        pruning_cb = LightGBMPruningCallback(trial, "r2")
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            pruning_cb,
        ]
        model_res = lgb.cv(
            params,
            dtrain,  # noqa: F821
            num_boost_round=3000,
            folds=cv_folds,
            feval=r2_eval,
            callbacks=callbacks,
        )
        return model_res["valid r2-mean"][-1]

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=50,
            multivariate=True,
            constant_liar=False,
            warn_independent_sampling=False,
        ),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=1, show_progress_bar=True)

    params = {
        "objective": "regression",
        "num_threads": -1,
        "verbose": -1,
        **study.best_params,
    }
    if params.get("bagging_freq", 0) == 0:
        params.setdefault("bagging_fraction", 1.0)
    best_value = study.best_value

    del study
    del dtrain
    gc.collect()

    return params, best_value


def _train_final_lgbm_model(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    params: dict,
    num_boost_round: int = 3000,
) -> lgb.Booster:
    """
    训练最终 LightGBM 模型（保留特征名）

    Args:
        train_x: 特征 DataFrame（列名会写入模型）
        train_y: 标签数组
        params: LightGBM 参数
        num_boost_round: 训练轮数

    Returns:
        训练好的 LightGBM Booster
    """
    assert isinstance(train_x, pd.DataFrame), "train_x must be a pandas DataFrame"
    assert len(train_x) == len(train_y), "train_x and train_y length mismatch"

    x = train_x.copy()
    x = x.astype(np.float32)
    x.columns = [str(c) for c in x.columns]

    dtrain = lgb.Dataset(
        x,
        train_y,
        free_raw_data=True,
        feature_name=list(x.columns),
    )
    return lgb.train(params, dtrain, num_boost_round=num_boost_round)


# ============================================================================
# 单模型构建
# ============================================================================
def build_single_model(
    features_df: pd.DataFrame,
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
    label_type: Literal["hard", "direction", "directional_prob"],
    selected_features: list[str],
    gmm_random_state: int,
    models_dir: Path,
) -> dict:
    """
    构建单个模型

    Args:
        features_df: 全局特征 DataFrame
        candles: Fusion K 线数据
        log_return_lag: GMMLabeler 的 lag_n 参数
        pred_next: 预测时间范围
        label_type: 标签类型
        selected_features: 该模型使用的特征列表
        gmm_random_state: GMMLabeler 的随机种子（从特征筛选阶段记录）
        models_dir: 模型保存目录

    Returns:
        包含构建结果的字典
    """
    # 1. 确定模型类型和名称
    # hard -> c (分类), direction -> r (回归), directional_prob -> r2 (回归)
    if label_type == "hard":
        model_type = "c"
    elif label_type == "direction":
        model_type = "r"
    elif label_type == "directional_prob":
        model_type = "r2"
    else:
        raise ValueError(f"Unknown label_type: {label_type}")
    model_name = f"{model_type}_L{log_return_lag}_N{pred_next}"
    model_dir = models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"[构建] {model_name}")
    print(f"特征数: {len(selected_features)}")
    print("=" * 60)

    # 2. 生成标签（使用特征筛选阶段记录的 seed 确保一致性）
    print("\n[1/6] 生成标签...")
    print(f"使用 GMM seed={gmm_random_state} (来自特征筛选阶段)")
    labeler = GMMLabeler(
        candles, lag_n=log_return_lag, verbose=False, random_seed=gmm_random_state
    )
    if label_type == "hard":
        raw_labels = labeler.label_hard_state
    elif label_type == "direction":
        raw_labels = labeler.label_direction_force
    elif label_type == "directional_prob":
        raw_labels = labeler.label_directional_prob
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    # 3. 提取选定特征并对齐
    print("\n[2/6] 对齐特征和标签...")
    selected_df = features_df[selected_features]
    aligned_features, aligned_labels = align_features_labels(
        selected_df,
        raw_labels,
        log_return_lag=log_return_lag,
        pred_next=pred_next,
    )
    print(f"对齐后: {len(aligned_features)} 样本, {aligned_features.shape[1]} 特征")

    # 4. ARDVAE 降维
    print("\n[3/6] ARDVAE 降维...")
    vae_config = ARDVAEConfig(
        max_latent_dim=ARDVAE_MAX_LATENT_DIM,
        max_epochs=ARDVAE_MAX_EPOCHS,
    )
    vae = ARDVAE(config=vae_config)

    # 80% 训练，20% 验证
    val_split = int(len(aligned_features) * 0.8)
    train_data = aligned_features.iloc[:val_split]
    val_data = aligned_features.iloc[val_split:]

    vae.fit(train_data, val_data=val_data, verbose=True)
    reduced_features = vae.transform(aligned_features)
    print(f"降维后维度: {reduced_features.shape[1]}")

    # 5. Optuna 调参
    print("\n[4/6] Optuna 调参...")
    reduced_df = reduced_features

    if label_type == "hard":
        best_params, cv_score = tune_classifier(reduced_df, aligned_labels)
        print(f"最优 F1: {cv_score:.4f}")
    elif label_type in ("direction", "directional_prob"):
        best_params, cv_score = tune_regressor(reduced_df, aligned_labels)
        print(f"最优 R²: {cv_score:.4f}")
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

    # 6. 全量训练最终模型
    print("\n[5/6] 全量训练...")
    final_model = _train_final_lgbm_model(reduced_df, aligned_labels, best_params)

    # 7. 持久化
    print("\n[6/6] 持久化模型...")
    # 保存特征列表
    with open(model_dir / "features.json", "w") as f:
        json.dump(selected_features, f, indent=2)

    # 保存 ARDVAE
    vae.save(model_dir, model_name)

    # 保存 LightGBM
    final_model.save_model(str(model_dir / f"model_{model_name}.txt"))

    # 保存调参结果
    with open(model_dir / "tuning_result.json", "w") as f:
        json.dump(
            {
                "cv_score": cv_score,
                "best_params": best_params,
                "n_latent_dims": int(reduced_features.shape[1]),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )

    print(f"模型已保存到: {model_dir}")

    return {
        "model_name": model_name,
        "n_features": len(selected_features),
        "n_latent_dims": reduced_features.shape[1],
        "cv_score": cv_score,
        "best_params": json.dumps(best_params),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 60)
    print("Flow Model Build - 批量模型构建")
    print("=" * 60)
    print(f"策略: {STRATEGY}")
    print(f"训练集: {TRAIN_START} ~ {TRAIN_END}")
    print(f"模型目录: {MODELS_DIR}")

    # 1. 读取特征筛选结果 + 提取实际需要的特征
    print("\n[1/4] 读取特征筛选结果...")
    selection_df = pd.read_csv(FEATURE_SELECTION_FILE)
    print(f"共 {len(selection_df)} 个模型配置")

    # 提取所有模型使用的特征并去重
    required_features = extract_required_features(FEATURE_SELECTION_FILE)
    print(f"实际需要的特征数: {len(required_features)} (去重后)")

    # 2. 加载训练集 K 线数据
    print("\n[2/4] 加载训练集数据...")
    _, raw_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp(TRAIN_START),
        helpers.date_to_timestamp(TRAIN_END),
        warmup_candles_num=0,
        caching=False,
        is_for_jesse=False,
    )
    print(f"原始 K 线: {raw_candles.shape}")

    bar = DemoBar(max_bars=-1)
    bar.update_with_candles(raw_candles)
    candles = bar.get_fusion_bars()
    print(f"Fusion K 线: {candles.shape}")

    # 3. 只计算实际需要的特征
    print("\n[3/4] 计算实际需要的特征...")
    calc = SimpleFeatureCalculator(verbose=True)
    calc.load(candles, sequential=True)
    features_dict = calc.get(required_features)
    features_df = pd.DataFrame(features_dict)
    print(f"特征矩阵: {features_df.shape}")

    # 4. 遍历构建所有模型
    print("\n[4/4] 开始批量构建模型...")
    results = []
    total = len(selection_df)

    for idx, row in selection_df.iterrows():
        print(f"\n{'#' * 60}")
        print(f"进度: {idx + 1}/{total}")
        print("#" * 60)

        selected_features = json.loads(row["selected_features"])

        try:
            result = build_single_model(
                features_df=features_df,
                candles=candles,
                log_return_lag=int(row["log_return_lag"]),
                pred_next=int(row["pred_next"]),
                label_type=row["label_type"],
                selected_features=selected_features,
                gmm_random_state=int(row["gmm_random_state"]),
                models_dir=MODELS_DIR,
            )
            results.append(result)
        except Exception as e:
            print(f"[ERROR] 构建失败: {e}")
            # 根据 label_type 确定模型前缀
            lt = row["label_type"]
            prefix = (
                "c" if lt == "hard" else ("r2" if lt == "directional_prob" else "r")
            )
            results.append(
                {
                    "model_name": f"{prefix}_L{row['log_return_lag']}_N{row['pred_next']}",
                    "error": str(e),
                }
            )

        gc.collect()

    # 5. 保存构建结果汇总
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_build_results.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"完成! 共构建 {len(results)} 个模型")
    print("结果已保存到: model_build_results.csv")
    print("=" * 60)

    # 打印汇总
    print("\n构建结果汇总:")
    if "cv_score" in results_df.columns:
        print(
            results_df[
                ["model_name", "n_features", "n_latent_dims", "cv_score"]
            ].to_string()
        )


if __name__ == "__main__":
    main()
