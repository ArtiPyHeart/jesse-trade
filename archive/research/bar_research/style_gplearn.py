"""
gplearn风格的符号回归。
目标：找到分布最接近正态分布的表达式。
评估方式：通过评估整合后的bar的kurtosis间接评估。
本文件代码仅仅作为示例，实际运行在jupyter notebook中进行。
"""

import numpy as np
import pandas as pd
from gplearn.fitness import make_fitness
from gplearn.genetic import SymbolicRegressor
from jesse.utils import numpy_candles_to_dataframe
from joblib import Parallel, delayed
from scipy import stats

from src.bars.build import build_bar_by_cumsum
from src.data_process.entropy.apen_sampen import sample_entropy_numba
from src.utils.math_tools import log_ret_from_candles

candles = np.load(
    "data/btc_1m.npy"
)  # 6 columns: timestamp, open, close, high, low, volume
candles = candles[candles[:, 5] > 0]  # ignore 0 volume candles
df = numpy_candles_to_dataframe(
    candles
)  # datetime index, 5 columns: open, close, high, low, volume

feature_and_label = []
# label
label = np.log(df["close"].shift(-1) / df["close"])
label.name = "label"
feature_and_label.append(label)

# 候选特征X
## high low range
feature_and_label.append(np.log(df["high"] / df["low"]))

RANGE = [25, 50, 100, 200]

## log return
for i in RANGE:
    series = np.log(df["close"] / df["close"].shift(i))
    series.name = f"r{i}"
    feature_and_label.append(series)

## entropy
for i in RANGE:
    log_ret_list = log_ret_from_candles(candles, [i] * len(candles))
    entropy_array: list[float] = Parallel(n_jobs=-1)(
        delayed(sample_entropy_numba)(i) for i in log_ret_list
    )
    len_gap = len(df) - len(entropy_array)

    entropy_array = [np.nan] * len_gap + entropy_array
    entropy_series = pd.Series(entropy_array, index=df.index)
    entropy_series.name = f"r{i}_entropy"
    feature_and_label.append(entropy_series)

df_features_and_label = pd.concat(feature_and_label, axis=1)
del feature_and_label

NA_MAX_NUM = df_features_and_label.isna().sum().max()

df_features_and_label = df_features_and_label.iloc[NA_MAX_NUM:]


def get_kurtosis(merged_bar):
    close_arr = merged_bar[:, 2]
    ret = np.log(close_arr[5:] / close_arr[:-5])
    standard = (ret - ret.mean()) / ret.std()
    kurtosis = stats.kurtosis(standard, axis=None, fisher=False, nan_policy="omit")
    return kurtosis


def gp_kurtosis(y, y_pred, w):
    # bypass gplearn's fitness function check
    if len(y_pred) <= 2:
        return 1000

    candles = np.load("data/btc_1m.npy")
    candles = candles[candles[:, 5] > 0]
    candles_in_metrics = candles[NA_MAX_NUM:]

    assert len(y_pred) == len(candles_in_metrics)

    cumsum_threshold = np.sum(y_pred) / (len(candles_in_metrics) // 120)

    merged_bar_cumsum = build_bar_by_cumsum(
        candles_in_metrics,
        y_pred,
        cumsum_threshold,
        reverse=False,
    )
    kurtosis_cumsum = get_kurtosis(merged_bar_cumsum)
    # 防止合成bar数量过少
    if len(merged_bar_cumsum) < len(candles_in_metrics) // 240:
        return 1000
    return kurtosis_cumsum


custom_kurtosis_loss = make_fitness(
    function=gp_kurtosis,
    greater_is_better=False,
    wrap=True,
)

cols = [col for col in df_features_and_label.columns if col != "label"]
X = df_features_and_label[cols].values[:-1]
y = df_features_and_label["label"].values[:-1]

# 训练
est_gp = SymbolicRegressor(
    init_method="full",
    metric=custom_kurtosis_loss,
    population_size=20000,
    generations=30,
    tournament_size=50,
    stopping_criteria=2,
    function_set=["add", "sub", "abs", "neg", "max", "min"],
    p_crossover=0.7,
    p_subtree_mutation=0.12,
    p_hoist_mutation=0.06,
    p_point_mutation=0.12,
    max_samples=1,
    parsimony_coefficient=0.009,
    feature_names=cols,
    verbose=1,
    n_jobs=12,
    # random_state=233,
)
est_gp.fit(X, y)
# 在jupyter中评估模型结果
