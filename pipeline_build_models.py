from research.model_pick.candle_fetch import FusionCandles
from research.model_pick.feature_select import FeatureSelector
from research.model_pick.features import FeatureLoader
from research.model_pick.labeler import PipelineLabeler

import json
import random
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from jesse.helpers import date_to_timestamp

# 准备工作
DATA_DIR = Path("./data")
MODEL_DIR = Path("./strategies/BinanceBtcDeapV1Voting/models")
# 已训练的SSM模型路径
MODEL_DEEP_SSM_PATH = MODEL_DIR / "deep_ssm"
MODEL_LG_SSM_PATH = MODEL_DIR / "lg_ssm"
TRAIN_TEST_SPLIT_DATE = "2025-03-01"
df_params = pd.read_csv(Path(__file__).parent / "model_search_results.csv")
df_params["best_params"] = df_params["best_params"].apply(json.loads)
df_params["selected_features"] = df_params["selected_features"].apply(json.loads)
candle_container = FusionCandles(
    exchange="Binance Perpetual Futures", symbol="BTC-USDT", timeframe="1m"
)
candles = candle_container.get_candles("2022-07-01", "2025-09-20")
# 特征生成只关心特征名称和原始数据
feature_loader = FeatureLoader(candles)
# 加载已训练的deep ssm与lg ssm模型
feature_selector = FeatureSelector(model_save_dir=MODEL_DIR, load_existing=True)

fracdiff_features = []
for p1 in ["o", "h", "l", "c"]:
    for p2 in ["o", "h", "l", "c"]:
        for l in range(1, 6):
            fracdiff_features.append(f"frac_{p1}_{p2}{l}_diff")

# 新生成特征配置记录
feature_info_path = MODEL_DIR / "feature_info.json"
if not feature_info_path.exists():
    with open(feature_info_path, "w") as f:
        json.dump({"fracdiff": fracdiff_features}, f, indent=4)
else:
    with open(feature_info_path, "r") as f:
        feature_info = json.load(f)
        feature_info["fracdiff"] = fracdiff_features
    with open(feature_info_path, "w") as f:
        json.dump(feature_info, f, indent=4)


def build_model(lag: int, pred_next: int, is_regression: bool = False, seed: int = 42):
    model_type = "r" if is_regression else "c"
    MODEL_NAME = f"{model_type}_L{lag}_N{pred_next}"

    model_path = MODEL_DIR / f"model_{MODEL_NAME}.txt"
    model_prod_path = MODEL_DIR / f"model_{MODEL_NAME}_prod.txt"
    if model_path.exists() and model_prod_path.exists():
        print(f"{lag = } {pred_next = } {model_type = } already exists, skipping")
        return

    # 获取最佳参数和已选择的特征
    model_row = df_params[
        (df_params["log_return_lag"] == lag)
        & (df_params["pred_next"] == pred_next)
        & (df_params["model_type"] == ("regressor" if is_regression else "classifier"))
    ]
    best_model_param = model_row["best_params"].iloc[0].copy()  # 使用copy避免修改原始数据
    feature_names = model_row["selected_features"].iloc[0]

    # 统一设置random seed以确保prod模型和非prod模型的一致性
    # 这对于置信度切片过滤至关重要
    best_model_param["seed"] = seed
    best_model_param["feature_fraction_seed"] = seed
    best_model_param["bagging_seed"] = seed
    best_model_param["data_random_seed"] = seed
    print(f"Using unified random seed: {seed}")
    # 制作标签
    labeler = PipelineLabeler(candles, lag)
    if is_regression:
        raw_label = labeler.label_direction
    else:
        raw_label = labeler.label_hard

    # 加工全量特征
    df_feat, label = feature_loader.get_feature_label_bundle(raw_label, pred_next)
    train_mask = df_feat.index.to_numpy() < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_y = label[train_mask]

    # 使用调参时已选择的特征名称
    print(f"Using {len(feature_names)} pre-selected features")
    with open(feature_info_path, "r") as f_r:
        feature_info = json.load(f_r)
        feature_info[f"{MODEL_NAME}"] = feature_names

    # 使用已加载的模型生成特征，不重新训练
    full_x = feature_selector.get_all_features_no_fit(df_feat)[feature_names]
    train_x = full_x[train_mask]
    assert full_x.shape[0] == len(label)

    with open(feature_info_path, "w") as f_w:
        json.dump(feature_info, f_w, indent=4)

    print(f"fitting {MODEL_NAME} model with {train_x.shape[1]} features")
    model = lgb.train(best_model_param, lgb.Dataset(train_x, train_y))
    model.save_model(model_path.resolve().as_posix())

    print(f"fitting {MODEL_NAME} prod model with {full_x.shape[1]} features")
    model_prod = lgb.train(best_model_param, lgb.Dataset(full_x, label))
    model_prod.save_model(model_prod_path.resolve().as_posix())


if __name__ == "__main__":
    # 全局统一设置random seed，确保可重复性
    GLOBAL_SEED = 42
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    print(f"=" * 60)
    print(f"Global random seed set to {GLOBAL_SEED}")
    print(f"This ensures consistency between prod and non-prod models")
    print(f"=" * 60 + "\n")

    # classifiers
    for lag in range(5, 8):
        for pred_next in range(1, 4):
            print(f"building classifier for {lag = } and {pred_next = }")
            build_model(lag, pred_next, is_regression=False, seed=GLOBAL_SEED)

    # regressors
    print("building regression model...")
    build_model(7, 1, is_regression=True, seed=GLOBAL_SEED)
    build_model(7, 2, is_regression=True, seed=GLOBAL_SEED)
    build_model(7, 3, is_regression=True, seed=GLOBAL_SEED)

    print("\n" + "=" * 60)
    print("All models built with unified random seed")
    print("This maximizes the effectiveness of confidence slice filters")
    print("=" * 60)
