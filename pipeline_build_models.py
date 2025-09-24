from research.model_pick.candle_fetch import FusionCandles
from research.model_pick.feature_select import FeatureSelector
from research.model_pick.features import FeatureLoader
from research.model_pick.labeler import PipelineLabeler

import json
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from jesse.helpers import date_to_timestamp

# 准备工作
DATA_DIR = Path("./data")
MODEL_DIR = Path("./strategies/BinanceBtcDeapV1Voting/models")
MODEL_DEEP_SSM_PATH = MODEL_DIR / "deep_ssm"
MODEL_LG_SSM_PATH = MODEL_DIR / "lg_ssm"
TRAIN_TEST_SPLIT_DATE = "2025-03-01"
df_params = pd.read_csv(Path(__file__).parent / "model_search_results.csv")
df_params["best_params"] = df_params["best_params"].apply(json.loads)
candle_container = FusionCandles(
    exchange="Binance Perpetual Futures", symbol="BTC-USDT", timeframe="1m"
)
candles = candle_container.get_candles("2022-07-01", "2025-09-20")
# 特征生成只关心特征名称和原始数据
feature_loader = FeatureLoader(candles)
# 由于训练集相同，selector内部的deep ssm与lg ssm只需要训练一次
feature_selector = FeatureSelector()

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


def build_model(lag: int, pred_next: int, is_regression: bool = False):
    model_type = "r" if is_regression else "c"
    model_path = MODEL_DIR / f"model_{model_type}_L{lag}_N{pred_next}.txt"
    model_prod_path = MODEL_DIR / f"model_{model_type}_L{lag}_N{pred_next}_prod.txt"
    if model_path.exists() and model_prod_path.exists():
        print(f"{lag = } {pred_next = } {model_type = } already exists, skipping")
        return

    best_model_param = df_params[
        (df_params["log_return_lag"] == lag)
        & (df_params["pred_next"] == pred_next)
        & (df_params["model_type"] == ("regressor" if is_regression else "classifier"))
    ]["best_params"].iloc[0]
    # 制作标签
    labeler = PipelineLabeler(candles, lag)
    if is_regression:
        raw_label = labeler.label_direction
    else:
        raw_label = labeler.label_hard

    # 加工全量特征
    df_feat, label = feature_loader.get_feature_label_bundle(raw_label, pred_next)
    train_mask = df_feat.index.to_numpy() < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x = df_feat[train_mask]
    train_y = label[train_mask]

    feature_names = feature_selector.select_features(train_x, train_y)
    feature_selector.deep_ssm_model.save(MODEL_DEEP_SSM_PATH.resolve().as_posix())
    feature_selector.lg_ssm_model.save(MODEL_LG_SSM_PATH.resolve().as_posix())
    with open(MODEL_DIR / "feature_info.json", "r") as f_r:
        feature_info = json.load(f_r)
        feature_info[f"L{lag}_N{pred_next}"] = feature_names

    with open(MODEL_DIR / "feature_info.json", "w") as f_w:
        json.dump(feature_info, f_w, indent=4)

    model = lgb.train(best_model_param, lgb.Dataset(train_x, train_y))
    model.save_model(model_path.resolve().as_posix())

    model_prod = lgb.train(best_model_param, lgb.Dataset(df_feat, label))
    model_prod.save_model(model_prod_path.resolve().as_posix())


if __name__ == "__main__":
    # classifiers
    for lag in range(4, 7):
        for pred_next in range(1, 4):
            print(f"building classifier for {lag = } and {pred_next = }")
            build_model(lag, pred_next, is_regression=False)

    # regressors
    print("building regression model...")
    build_model(6, 1, is_regression=True)
    build_model(6, 3, is_regression=True)
    build_model(5, 1, is_regression=True)
    build_model(7, 1, is_regression=True)

    print("done")
