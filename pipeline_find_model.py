import numpy as np
from jesse.helpers import date_to_timestamp

from research.model_pick.candle_fetch import FusionCandles
from research.model_pick.feature_select import FeatureSelector
from research.model_pick.features import FeatureLoader
from research.model_pick.labeler import PipelineLabeler
from research.model_pick.model_tuning import ModelTuning

# 固定训练集切分点，从而固定训练集，节约特征生成和筛选的时间。测试集主要用于回测
TRAIN_TEST_SPLIT_DATE = "2025-03-01"

candle_container = FusionCandles(
    exchange="Binance Perpetual Futures", symbol="BTC-USDT", timeframe="1m"
)
candles = candle_container.get_candles("2022-07-01", "2025-09-15")
# 特征生成只关心特征名称和原始数据
feature_loader = FeatureLoader(candles)
# 由于训练集相同，selector内部的deep ssm与lg ssm只需要训练一次
feature_selector = FeatureSelector()


def evaluate_classifier(
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
):
    labeler = PipelineLabeler(candles, log_return_lag)
    label_for_classifier = labeler.label_hard

    df_feat, label_c = feature_loader.get_feature_label_bundle(
        label_for_classifier, pred_next
    )
    train_mask = df_feat.index.to_numpy() < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x = df_feat[train_mask]
    train_y = label_c[train_mask]

    feature_names = feature_selector.select_features(train_x, train_y)

    model_tuning = ModelTuning(
        TRAIN_TEST_SPLIT_DATE,
        train_x,
        train_y,
    )

    params, best_score = model_tuning.tuning_classifier(feature_selector, feature_names)
    return params, best_score


def evaluate_regressor(
    candles: np.ndarray,
    log_return_lag: int,
    pred_next: int,
):
    labeler = PipelineLabeler(candles, log_return_lag)
    label_for_regressor = labeler.label_direction

    df_feat, label_r = feature_loader.get_feature_label_bundle(
        label_for_regressor, pred_next
    )
    train_mask = df_feat.index.to_numpy() < date_to_timestamp(TRAIN_TEST_SPLIT_DATE)
    train_x = df_feat[train_mask]
    train_y = label_r[train_mask]

    feature_names = feature_selector.select_features(train_x, train_y)

    model_tuning = ModelTuning(
        TRAIN_TEST_SPLIT_DATE,
        train_x,
        train_y,
    )

    params, best_score = model_tuning.tuning_classifier(feature_selector, feature_names)
    return params, best_score


if __name__ == "__main__":
    # 设置不同的log return回看step，至少从4开始
    for i in [4, 5, 6, 7, 8]:
        # 设置不同的pred next step，预测next N个标签
        for p in [1, 2, 3]:
            classifier_params, c_score = evaluate_classifier(candles.copy(), i, p)
            regressor_params, r_score = evaluate_regressor(candles.copy(), i, p)
