"""
Flow Model Tuning - 批量模型调参与训练流水线

基于 feature_selection_results.csv 的筛选结果，完成：
1. 构建并保存全局 FeatureMaker
2. 为每个 (lag, pred_next) 组合训练 Reducer
3. 使用 Optuna 调参
4. 训练最终 LightGBM 模型并保存
5. 测试集评估

Usage:
    python flow_model_tuning.py
"""

import json
import shutil
from datetime import datetime
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from jesse.helpers import date_to_timestamp
from sklearn.metrics import f1_score, r2_score

from research.model_pick.candle_fetch import FusionCandles
from research.model_pick.features import ALL_FEATS
from research.model_pick.labeler import PipelineLabeler
from research.model_pick.model_tuning import ModelTuning
from src.features.pipeline import (
    FeatureMaker,
    FeatureMakerConfig,
    Reducer,
    ReducerConfig,
)

# ============================================================================
# 配置参数
# ============================================================================
# 策略名称
STRATEGY_NAME = "BinanceBtcDemoBar"

# 数据范围
START = "2022-08-01"
END = "2025-12-25"

# 训练/测试分割日期
TRAIN_TEST_SPLIT_DATE = "2025-06-01"

# 搜索参数（控制训练模型的范围）
LOG_RETURN_LAGS = [4, 5, 6, 7, 8]
PRED_NEXT_STEPS = [1, 2, 3]
LABEL_TYPES = ["hard", "direction"]

# 输入文件
INPUT_FILE = "feature_selection_results.csv"

# 训练内部验证集比例（用于 early stopping）
VALID_RATIO = 0.1
MIN_VALID_SAMPLES = 200


def get_models_dir() -> Path:
    """获取策略 models 目录路径"""
    return Path("strategies") / STRATEGY_NAME / "models"


def load_feature_selection_results() -> pd.DataFrame:
    """
    加载特征筛选结果

    Returns:
        特征筛选结果 DataFrame
    """
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(
            f"特征筛选结果文件不存在: {INPUT_FILE}\n"
            "请先运行 flow_feature_select.py 生成特征筛选结果"
        )

    df = pd.read_csv(INPUT_FILE)
    print(f"加载特征筛选结果: {len(df)} 条记录")
    return df


def aggregate_all_features(df: pd.DataFrame) -> list[str]:
    """
    汇总并去重所有选中的特征名称

    Args:
        df: 特征筛选结果 DataFrame

    Returns:
        去重后的特征名称列表
    """
    all_features = set()

    for _, row in df.iterrows():
        selected = json.loads(row["selected_features"])
        all_features.update(selected)

    # 按 ALL_FEATS 中的顺序排序，保证一致性
    all_feats_order = {name: i for i, name in enumerate(ALL_FEATS)}
    sorted_features = sorted(
        all_features,
        key=lambda x: all_feats_order.get(x, len(ALL_FEATS)),
    )

    print(f"汇总特征: {len(sorted_features)} 个（去重后）")
    return sorted_features


def get_selected_features_for_model(
    df: pd.DataFrame, log_return_lag: int, pred_next: int, label_type: str
) -> list[str]:
    """
    获取指定模型的选中特征

    Args:
        df: 特征筛选结果 DataFrame
        log_return_lag: 对数收益 lag
        pred_next: 预测步数
        label_type: 标签类型

    Returns:
        选中的特征名称列表
    """
    mask = (
        (df["log_return_lag"] == log_return_lag)
        & (df["pred_next"] == pred_next)
        & (df["label_type"] == label_type)
    )
    row = df[mask]

    if len(row) == 0:
        raise ValueError(
            f"未找到 lag={log_return_lag}, pred_next={pred_next}, "
            f"label_type={label_type} 的特征筛选结果"
        )

    return json.loads(row.iloc[0]["selected_features"])


def align_features_labels(
    global_features: pd.DataFrame,
    raw_label: np.ndarray,
    candles_ts: np.ndarray,
    log_return_lag: int,
    pred_next: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    对齐特征和标签

    - label 开头缺少 log_return_lag 个值
    - feature 需要去掉开头的 NaN
    - 按 pred_next 进行 shift 对齐
    - 返回 label 对应的时间戳，用于严格切分
    """
    # 截断 feature 开头 (对齐 label 的 lag)
    features = global_features.iloc[log_return_lag:]

    # shift 以对齐 PRED_NEXT
    features = features.iloc[:-pred_next]
    label = raw_label[pred_next:]
    label_ts = candles_ts[log_return_lag + pred_next :]

    # 去掉 feature 开头的 NaN
    na_mask = features.isna().any(axis=1).values
    features = features.iloc[~na_mask]
    label = label[~na_mask]
    label_ts = label_ts[~na_mask]

    assert len(features) == len(label) == len(label_ts), (
        f"Length mismatch: {len(features)} vs {len(label)} vs {len(label_ts)}"
    )
    return features, label, label_ts


def split_train_valid(
    train_x: pd.DataFrame, train_y: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame | None, np.ndarray | None]:
    """
    按时间顺序切分训练/验证集，用于 early stopping。
    """
    n_train = len(train_x)
    n_valid = int(n_train * VALID_RATIO)

    if n_valid < MIN_VALID_SAMPLES:
        return train_x, train_y, None, None

    train_x_fit = train_x.iloc[:-n_valid]
    train_y_fit = train_y[:-n_valid]
    valid_x = train_x.iloc[-n_valid:]
    valid_y = train_y[-n_valid:]

    assert len(train_x_fit) > 0, "Train split empty after validation split."
    return train_x_fit, train_y_fit, valid_x, valid_y


def get_model_name(label_type: str, lag: int, pred_next: int) -> str:
    """
    生成模型名称

    Args:
        label_type: 标签类型 ("hard" 或 "direction")
        lag: 对数收益 lag
        pred_next: 预测步数

    Returns:
        模型名称，格式为 {c|r}_L{lag}_N{pred_next}
    """
    model_type = "c" if label_type == "hard" else "r"
    return f"{model_type}_L{lag}_N{pred_next}"


def train_and_save_model(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    test_x: pd.DataFrame,
    test_y: np.ndarray,
    model_name: str,
    label_type: str,
    models_dir: Path,
) -> dict:
    """
    调参、训练并保存模型

    Args:
        train_x: 训练特征
        train_y: 训练标签
        test_x: 测试特征
        test_y: 测试标签
        model_name: 模型名称
        label_type: 标签类型
        models_dir: 模型保存目录

    Returns:
        包含训练结果的字典
    """
    print(f"\n[调参] {model_name}")
    print(f"  训练集: {len(train_x)} 样本, {train_x.shape[1]} 特征")
    print(f"  测试集: {len(test_x)} 样本")

    tuner = ModelTuning.from_train_data()

    if label_type == "hard":
        best_params, cv_score = tuner.tuning_classifier_direct(train_x, train_y)
        objective = "binary"
    else:
        best_params, cv_score = tuner.tuning_regressor_direct(train_x, train_y)
        objective = "regression"

    print(f"  CV 最佳分数: {cv_score:.4f}")
    print(f"  最佳参数: {best_params}")

    # 使用最佳参数在全量训练集上训练
    print(f"\n[训练] {model_name} (全量训练集)")

    # 训练/验证切分（仅用训练集内部数据）
    train_x_fit, train_y_fit, valid_x, valid_y = split_train_valid(train_x, train_y)

    # 准备训练数据
    train_x_np = np.ascontiguousarray(train_x_fit.to_numpy(dtype=np.float32))
    dtrain = lgb.Dataset(train_x_np, train_y_fit, free_raw_data=True)

    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = []

    if valid_x is not None:
        valid_x_np = np.ascontiguousarray(valid_x.to_numpy(dtype=np.float32))
        dvalid = lgb.Dataset(valid_x_np, valid_y, free_raw_data=True)
        valid_sets.append(dvalid)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
        print(f"  验证集: {len(valid_x)} 样本 (用于 early stopping)")
    else:
        print("  验证集: 未启用 (样本不足)")

    # 添加固定参数
    train_params = {
        **best_params,
        "objective": objective,
        "num_threads": -1,
        "verbose": -1,
    }
    if label_type == "hard":
        train_params["metric"] = "binary_logloss"
    else:
        train_params["metric"] = "l2"

    # 训练模型
    model = lgb.train(
        train_params,
        dtrain,
        num_boost_round=3000,
        callbacks=callbacks,
        valid_sets=valid_sets,
        valid_names=valid_names,
    )

    # 保存模型
    model_dir = models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"model_{model_name}.txt"
    model.save_model(str(model_path))
    print(f"  模型已保存: {model_path}")

    # 测试集评估
    print(f"\n[评估] {model_name} (测试集)")
    test_x_np = np.ascontiguousarray(test_x.to_numpy(dtype=np.float32))
    best_iter = model.best_iteration if model.best_iteration > 0 else None
    test_pred = model.predict(test_x_np, num_iteration=best_iter)

    if label_type == "hard":
        test_pred_binary = (test_pred > 0.5).astype(int)
        test_score = f1_score(test_y, test_pred_binary, average="weighted")
        metric_name = "F1"
    else:
        test_score = r2_score(test_y, test_pred)
        metric_name = "R²"

    print(f"  测试集 {metric_name}: {test_score:.4f}")

    return {
        "model_name": model_name,
        "cv_score": cv_score,
        "test_score": test_score,
        "metric_name": metric_name,
        "n_train": len(train_x),
        "n_test": len(test_x),
        "n_features": train_x.shape[1],
    }


def run_single_model(
    global_features: pd.DataFrame,
    candles: np.ndarray,
    feature_selection_df: pd.DataFrame,
    log_return_lag: int,
    pred_next: int,
    label_type: str,
    split_ts: int,
    models_dir: Path,
) -> dict:
    """
    训练单个模型的完整流程

    Args:
        global_features: 全局特征 DataFrame
        candles: K 线数据
        feature_selection_df: 特征筛选结果
        log_return_lag: 对数收益 lag
        pred_next: 预测步数
        label_type: 标签类型
        split_ts: 训练/测试分割时间戳
        models_dir: 模型保存目录

    Returns:
        包含训练结果的字典
    """
    model_name = get_model_name(label_type, log_return_lag, pred_next)

    print(f"\n{'=' * 60}")
    print(f"[模型] {model_name}")
    print(f"  lag={log_return_lag}, pred_next={pred_next}, label_type={label_type}")
    print("=" * 60)

    selected_features = get_selected_features_for_model(
        feature_selection_df, log_return_lag, pred_next, label_type
    )
    print(f"[1/5] 选中特征: {len(selected_features)} 个")

    print("[2/5] 生成标签...")
    labeler = PipelineLabeler(candles, log_return_lag)
    raw_label = labeler.label_hard if label_type == "hard" else labeler.label_direction

    print("[3/5] 对齐特征和标签...")
    candles_ts = candles[:, 0].astype(int)
    features, label, label_ts = align_features_labels(
        global_features, raw_label, candles_ts, log_return_lag, pred_next
    )
    features = features[selected_features]
    print(f"  对齐后: {len(features)} 样本, {features.shape[1]} 特征")

    train_mask = label_ts < split_ts

    train_x = features[train_mask].reset_index(drop=True)
    train_y = label[train_mask]
    test_x = features[~train_mask].reset_index(drop=True)
    test_y = label[~train_mask]
    assert len(train_x) > 0, "Empty train split after alignment."
    assert len(test_x) > 0, "Empty test split after alignment."
    print(f"  训练集: {len(train_x)}, 测试集: {len(test_x)}")

    print("[4/5] 训练 Reducer...")
    reducer_config = ReducerConfig(verbose=True)
    reducer = Reducer(reducer_config)
    reducer.fit(train_x)
    print(f"  降维: {train_x.shape[1]} → {reducer.n_components}")

    model_dir = models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    reducer.save(str(model_dir), model_name)

    train_x_reduced = reducer.transform(train_x)
    test_x_reduced = reducer.transform(test_x)

    print("[5/5] 调参与训练...")
    result = train_and_save_model(
        train_x=train_x_reduced,
        train_y=train_y,
        test_x=test_x_reduced,
        test_y=test_y,
        model_name=model_name,
        label_type=label_type,
        models_dir=models_dir,
    )

    result["log_return_lag"] = log_return_lag
    result["pred_next"] = pred_next
    result["label_type"] = label_type
    result["n_selected_features"] = len(selected_features)
    result["n_reduced_features"] = reducer.n_components

    return result


def main():
    print("=" * 60)
    print("Flow Model Tuning - 批量模型调参与训练")
    print("=" * 60)
    print(f"策略: {STRATEGY_NAME}")
    print(f"数据范围: {START} ~ {END}")
    print(f"训练/测试分割: {TRAIN_TEST_SPLIT_DATE}")
    print(f"LOG_RETURN_LAGS: {LOG_RETURN_LAGS}")
    print(f"PRED_NEXT_STEPS: {PRED_NEXT_STEPS}")
    print(f"LABEL_TYPES: {LABEL_TYPES}")
    print("=" * 60)

    split_ts = date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    models_dir = get_models_dir()

    # 清理并重建 models 目录
    if models_dir.exists():
        print(f"\n[清理] 删除已有 models 目录: {models_dir}")
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载特征筛选结果
    print("\n[1/5] 加载特征筛选结果...")
    feature_selection_df = load_feature_selection_results()

    valid_combinations = []

    for label_type in LABEL_TYPES:
        for lag, pred_next in product(LOG_RETURN_LAGS, PRED_NEXT_STEPS):
            mask = (
                (feature_selection_df["log_return_lag"] == lag)
                & (feature_selection_df["pred_next"] == pred_next)
                & (feature_selection_df["label_type"] == label_type)
            )
            if mask.any():
                valid_combinations.append((lag, pred_next, label_type))
            else:
                print(
                    f"  [跳过] lag={lag}, pred_next={pred_next}, "
                    f"label_type={label_type}: 无对应筛选结果"
                )

    if not valid_combinations:
        print("\n[ERROR] 没有有效的训练组合，请检查配置或运行 flow_feature_select.py")
        return

    print(f"  有效组合: {len(valid_combinations)} 个")

    filtered_df = feature_selection_df[
        (feature_selection_df["log_return_lag"].isin(LOG_RETURN_LAGS))
        & (feature_selection_df["pred_next"].isin(PRED_NEXT_STEPS))
        & (feature_selection_df["label_type"].isin(LABEL_TYPES))
    ]

    # 2. 汇总所有特征
    print("\n[2/5] 汇总全局特征...")
    all_features = aggregate_all_features(filtered_df)

    # 3. 加载 K 线数据
    print("\n[3/5] 加载 K 线数据...")
    candle_container = FusionCandles(
        exchange="Binance Perpetual Futures", symbol="BTC-USDT", timeframe="1m"
    )
    candles = candle_container.get_candles(START, END)
    print(f"  K 线数据: {candles.shape}")

    # 4. 构建并保存全局 FeatureMaker
    print("\n[4/5] 构建全局 FeatureMaker...")
    feature_config = FeatureMakerConfig(
        feature_names=all_features,
        ssm_state_dim=5,
        verbose=True,
    )
    feature_maker = FeatureMaker(feature_config)
    candles_ts = candles[:, 0].astype(int)
    train_mask = candles_ts < split_ts
    train_candles = candles[train_mask]

    assert len(train_candles) > 0, "Train candles empty after split."
    assert len(candles) > len(train_candles), "Test candles empty after split."

    feature_maker.fit(train_candles)
    global_features = feature_maker.transform(candles)
    print(f"  全局特征: {global_features.shape}")

    # 保存 FeatureMaker
    feature_maker.save(str(models_dir), "feature_maker")

    # 设置特征索引为 timestamp
    global_features.index = candles_ts

    print("\n[5/5] 开始批量模型训练...")
    total = len(valid_combinations)
    results = []

    for idx, (lag, pred_next, label_type) in enumerate(valid_combinations, 1):
        print(f"\n进度: {idx}/{total}")
        try:
            result = run_single_model(
                global_features=global_features,
                candles=candles,
                feature_selection_df=filtered_df,
                log_return_lag=lag,
                pred_next=pred_next,
                label_type=label_type,
                split_ts=split_ts,
                models_dir=models_dir,
            )
            results.append(result)
        except Exception as e:
            print(
                f"[ERROR] lag={lag}, pred_next={pred_next}, label_type={label_type}: {e}"
            )
            import traceback

            traceback.print_exc()
            continue

    # 打印汇总
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    if results:
        print(f"\n共训练 {len(results)} 个模型:")
        print("-" * 80)
        print(
            f"{'模型名称':<15} {'选中特征':>8} {'降维后':>6} "
            f"{'CV分数':>8} {'测试分数':>8} {'指标':>4}"
        )
        print("-" * 80)

        for r in results:
            print(
                f"{r['model_name']:<15} {r['n_selected_features']:>8} "
                f"{r['n_reduced_features']:>6} {r['cv_score']:>8.4f} "
                f"{r['test_score']:>8.4f} {r['metric_name']:>4}"
            )

        print("-" * 80)

        results_df = pd.DataFrame(results)
        output_csv = "model_tuning_results.csv"
        results_df.to_csv(output_csv, index=False)

        print(f"\n模型已保存到: {models_dir}")
        print(f"结果已保存到: {output_csv}")
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("\n[WARNING] 没有成功训练的模型")


if __name__ == "__main__":
    main()
