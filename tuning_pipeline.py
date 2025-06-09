from contextlib import contextmanager

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from hmmlearn.hmm import GMMHMM
from jesse import helpers
from mpire import WorkerPool

from custom_indicators.all_features import feature_bundle
from custom_indicators.toolbox.bar.fusion.base import FusionBarContainerBase
from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.toolbox.feature_selection.rfcq_selector import RFCQSelector
from custom_indicators.utils.math_tools import log_ret


class OptunaLogManager:
    @staticmethod
    @contextmanager
    def silent_optimization():
        """上下文管理器，在此范围内禁用日志"""
        optuna.logging.disable_default_handler()
        try:
            yield
        finally:
            optuna.logging.enable_default_handler()

    @staticmethod
    @contextmanager
    def verbose_optimization():
        """上下文管理器，在此范围内启用详细日志"""
        original_level = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.INFO)
        try:
            yield
        finally:
            optuna.logging.set_verbosity(original_level)


optuna_log_manager = OptunaLogManager()


class TuningBarContainer(FusionBarContainerBase):
    def __init__(
        self, n1: int, n2: int, n_entropy: int, threshold: float, max_bars=50000
    ):
        super().__init__(max_bars, threshold)
        self.N_1 = n1
        self.N_2 = n2
        self.N_ENTROPY = n_entropy

    @property
    def max_lookback(self) -> int:
        return max(self.N_1, self.N_2, self.N_ENTROPY)

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        log_ret_n_1 = np.log(candles[self.N_1 :, 2] / candles[: -self.N_1, 2])
        if self.max_lookback > self.N_1:
            log_ret_n_1 = log_ret_n_1[self.max_lookback - self.N_1 :]
        log_ret_n_2 = np.log(candles[self.N_2 :, 2] / candles[: -self.N_2, 2])
        if self.max_lookback > self.N_2:
            log_ret_n_2 = log_ret_n_2[self.max_lookback - self.N_2 :]

        if self.max_lookback > self.N_ENTROPY:
            entropy_log_ret_list = log_ret(
                candles[self.max_lookback - self.N_ENTROPY :], self.N_ENTROPY
            )
        else:
            entropy_log_ret_list = log_ret(candles, self.N_ENTROPY)

        with WorkerPool() as pool:
            entropy_array = pool.map(sample_entropy_numba, entropy_log_ret_list)

        return np.min([np.abs(log_ret_n_1), log_ret_n_2 + entropy_array], axis=0)


class BacktestPipeline:
    def __init__(self, raw_bar_path: str):
        raw_candles = np.load(raw_bar_path)
        self.raw_candles = raw_candles[raw_candles[:, 5] > 0]
        self.bar_container = None
        self._merged_bar = None
        self.side_label = None
        self.df_feature = None

    @property
    def train_test_split_timestamp(self):
        return helpers.date_to_timestamp("2025-01-01")

    @property
    def trading_fee(self) -> float:
        return 0.05 / 100

    @property
    def merged_bar(self):
        return self._merged_bar.copy()

    @merged_bar.setter
    def merged_bar(self, value):
        self._merged_bar = value

    def init_bar_container(self, n1, n2, n_entropy, threshold=0.5):
        self.bar_container = TuningBarContainer(n1, n2, n_entropy, threshold)

    def get_threshold_array(self):
        return self.bar_container.get_thresholds(self.raw_candles)

    def set_threshold(self, threshold):
        self.bar_container.THRESHOLD = threshold

    def generate_merged_bar(self):
        self.bar_container.update_with_candles(self.raw_candles)
        self.merged_bar = self.bar_container.get_fusion_bars()

    def _get_gmmhmm_random_state(self):
        def objective(trial: optuna.Trial):
            mix = 3  ### GMM mix参数
            L = 5

            close_arr = self.merged_bar[:, 2]
            high_arr = self.merged_bar[:, 3][L:]
            low_arr = self.merged_bar[:, 4][L:]

            log_return = np.log(close_arr[1:] / close_arr[:-1])[L - 1 :]
            log_return_L = np.log(close_arr[L:] / close_arr[:-L])
            HL_diff = np.log(high_arr / low_arr)

            X = np.column_stack([HL_diff, log_return_L, log_return])

            datelist = np.asarray(
                [
                    pd.Timestamp(helpers.timestamp_to_time(i))
                    for i in self.merged_bar[:, 0][L:]
                ]
            )
            closeidx = self.merged_bar[:, 2][L:]

            assert len(datelist) == len(closeidx)
            assert len(datelist) == len(X)

            gmm = GMMHMM(
                n_components=2,
                n_mix=mix,
                covariance_type="diag",
                n_iter=1000,
                # weights_prior=2,
                means_weight=0.5,
                random_state=trial.suggest_int("random_state", 0, 1000),
            )
            gmm.fit(X)
            latent_states_sequence = gmm.predict(X)
            data = pd.DataFrame(
                {
                    "datelist": datelist,
                    "logreturn": log_return,
                    "state": latent_states_sequence,
                }
            ).set_index("datelist")

            final_ret = 0
            for i in data["state"].unique():
                ret = data[data["state"] == i]["logreturn"].sum()
                final_ret += np.abs(ret)

            return final_ret

        with optuna_log_manager.silent_optimization():
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(),
            )
            study.optimize(objective, n_trials=30)
            return study.best_params["random_state"]

    def side_labeling(self):
        mix = 3  ### GMM mix参数
        L = 5

        close_arr = self.merged_bar[:, 2]
        high_arr = self.merged_bar[:, 3][L:]
        low_arr = self.merged_bar[:, 4][L:]

        log_return = np.log(close_arr[1:] / close_arr[:-1])[L - 1 :]
        log_return_L = np.log(close_arr[L:] / close_arr[:-L])
        HL_diff = np.log(high_arr / low_arr)

        X = np.column_stack([HL_diff, log_return_L, log_return])

        datelist = np.asarray(
            [
                pd.Timestamp(helpers.timestamp_to_time(i))
                for i in self.merged_bar[:, 0][L:]
            ]
        )
        closeidx = self.merged_bar[:, 2][L:]

        assert len(datelist) == len(closeidx)
        assert len(datelist) == len(X)

        gmm = GMMHMM(
            n_components=2,
            n_mix=mix,
            covariance_type="diag",
            n_iter=1000,
            # weights_prior=2,
            means_weight=0.5,
            random_state=self._get_gmmhmm_random_state(),
        )
        gmm.fit(X)
        latent_states_sequence = gmm.predict(X)
        data = pd.DataFrame(
            {
                "datelist": datelist,
                "logreturn": log_return,
                "state": latent_states_sequence,
            }
        ).set_index("datelist")
        ret_dict = {}
        for i in data["state"].unique():
            ret = data[data["state"] == i]["logreturn"].sum()
            ret_dict[i] = ret
        pos_label = 0
        for k, v in ret_dict.items():
            if v > 0:
                pos_label = k
                break
        self.side_label = (latent_states_sequence == pos_label).astype(int)

    def get_df_feature(self):
        self.df_feature = pd.DataFrame(
            feature_bundle(self.merged_bar[:-1], sequential=True),
            index=self.merged_bar[:-1, 0].astype(int),
        )
        self.df_feature = self.df_feature.iloc[self.df_feature.isna().sum().max() :]
        self.side_label = self.side_label[-len(self.df_feature) :]
        assert len(self.df_feature) == len(self.side_label)

    def side_feature_selection(self):
        selector = RFCQSelector(verbose=False)
        selector.fit(self.df_feature, self.side_label)
        side_res = pd.Series(
            selector.relevance_, index=selector.variables_
        ).sort_values(ascending=False)
        self.side_feature_names = side_res[side_res > 0].index.tolist()

    def train_side_model(self):
        mask = self.df_feature.index < self.train_test_split_timestamp
        feature_masked = self.df_feature[self.side_feature_names][mask]
        label_masked = self.side_label[mask]
        params = {
            "objective": "binary",
            "metric": "auc",
            "num_threads": -1,
            "verbose": -1,
            "is_unbalance": True,
            "extra_trees": False,
            "num_leaves": 62,
            "max_depth": 294,
            "min_gain_to_split": 0.3876711100653668,
            "min_data_in_leaf": 200,
            "lambda_l1": 10,
            "lambda_l2": 50,
            "num_boost_round": 600,
        }
        dtrain = lgb.Dataset(feature_masked, label_masked)
        self.side_model = lgb.train(params, dtrain)

        self.df_feature["model"] = self.side_model.predict(
            self.df_feature[self.side_feature_names]
        )

    def meta_labeling(self):
        side_res = self.side_model.predict(self.df_feature[self.side_feature_names])
        side_pred_label = np.where(side_res > 0.5, 1, -1)
        log_ret = np.log(self.merged_bar[1:, 2] / self.merged_bar[:-1, 2])[
            -len(side_pred_label) :
        ]
        assert len(log_ret) == len(side_pred_label)
        meta_label = np.full_like(side_pred_label, 0)

        for idx, (i, r) in enumerate(zip(side_pred_label[:-1], log_ret[:-1])):
            if i == 1:
                # side模型做多
                if log_ret[idx + 1] > 0:
                    # 方向正确
                    meta_label[idx] = 1
                else:
                    # 方向错误
                    meta_label[idx] = 0
            else:
                # side模型做空
                if log_ret[idx + 1] < 0:
                    meta_label[idx] = 1
                else:
                    meta_label[idx] = 0

        start_idx = 0
        cumsum_ret = 0
        for idx, (meta, side, ret) in enumerate(
            zip(meta_label, side_pred_label, log_ret)
        ):
            if meta == 1:
                if idx > 0 and meta_label[idx - 1] == 0:
                    # 开始持仓
                    start_idx = idx
                else:
                    # 继续持仓
                    cumsum_ret += ret * (1 if side == 1 else -1)
            elif meta == 0:
                if idx > 0 and meta_label[idx - 1] == 1:
                    # 结束持仓
                    cumsum_ret += ret * (1 if side == 1 else -1)
                    end_idx = idx
                    if cumsum_ret < 0:
                        # 如果收益为负，则认为判断错误
                        assert start_idx < end_idx, (
                            "start_idx must be less than end_idx"
                        )
                        meta_label[start_idx:end_idx] = 0
                    # 重置收益
                    cumsum_ret = 0
                    start_idx = 0
                else:
                    continue

        self.meta_label = meta_label

    def meta_feature_selection(self):
        selector = RFCQSelector(verbose=False)
        selector.fit(self.df_feature, self.meta_label)
        meta_res = pd.Series(
            selector.relevance_, index=selector.variables_
        ).sort_values(ascending=False)
        self.meta_feature_names = meta_res[meta_res > 0].index.tolist()

    def train_meta_model(self):
        mask = self.df_feature.index < self.train_test_split_timestamp
        feature_masked = self.df_feature[self.meta_feature_names][mask]
        label_masked = self.meta_label[mask]
        params = {
            "objective": "binary",
            "metric": "auc",
            "num_threads": -1,
            "verbose": -1,
            "is_unbalance": True,
            "extra_trees": False,
            "boosting": "dart",
            "num_leaves": 377,
            "max_depth": 133,
            "min_gain_to_split": 0.07755900215445423,
            "min_data_in_leaf": 186,
            "lambda_l1": 3,
            "lambda_l2": 88,
            "num_boost_round": 1000,
        }
        dtrain = lgb.Dataset(feature_masked, label_masked)
        self.meta_model = lgb.train(params, dtrain)

    def backtest(self):
        log_ret = np.log(self.merged_bar[1:, 2] / self.merged_bar[:-1, 2])[:-1]
        test_bars = self.merged_bar[
            self.merged_bar[:, 0] >= self.train_test_split_timestamp
        ][:-1]
        log_ret = log_ret[-len(test_bars) :]

        side_features = self.df_feature[self.side_feature_names]
        side_features = side_features[
            side_features.index >= self.train_test_split_timestamp
        ]
        side_pred = self.side_model.predict(side_features)
        side_pred_label = np.where(side_pred > 0.5, 1, -1)

        meta_features = self.df_feature[self.meta_feature_names]
        meta_features = meta_features[
            meta_features.index >= self.train_test_split_timestamp
        ]
        meta_pred = self.meta_model.predict(meta_features)
        meta_pred_label = np.where(meta_pred > 0.5, 1, 0)

        assert len(log_ret) == len(side_pred_label)
        assert len(log_ret) == len(meta_pred_label)

        total_return = 0
        one_return = 0
        max_drawdown = 0
        for idx, (side, meta, ret) in enumerate(
            zip(side_pred_label, meta_pred_label, log_ret)
        ):
            if meta == 1:
                if (idx == 0) or (meta_pred_label[idx - 1] == 0):
                    # 开始持仓
                    one_return -= self.trading_fee
                else:
                    # 继续持仓
                    if side != side_pred_label[idx - 1]:
                        # 反方向调仓需要更多手续费
                        one_return -= self.trading_fee * 2
                    one_return += ret * side
            else:
                if (idx == 0) or (meta_pred_label[idx - 1] == 0):
                    continue
                elif meta_pred_label[idx - 1] == 1:
                    # 结束持仓
                    one_return += ret * side - self.trading_fee
                    total_return += one_return
                    if one_return < max_drawdown:
                        max_drawdown = one_return
                    one_return = 0

        print(f"{total_return = :.2f}, {max_drawdown = :.2f}")
        calmar_ratio = total_return / abs(max_drawdown)
        return calmar_ratio


def tune_pipeline(trial: optuna.Trial):
    pipeline = BacktestPipeline("data/btc_1m.npy")
    n1 = trial.suggest_int("n1", 1, 300)
    n2 = trial.suggest_int("n2", 1, 300)
    n_entropy = trial.suggest_int("n_entropy", 30, 300)
    pipeline.init_bar_container(n1, n2, n_entropy)
    raw_threshold_array = pipeline.get_threshold_array()
    threshold_min = np.sum(raw_threshold_array) / (len(pipeline.raw_candles) // 30)
    threshold_max = np.sum(raw_threshold_array) / (len(pipeline.raw_candles) // 330)
    pipeline.set_threshold(
        trial.suggest_float("threshold", threshold_min, threshold_max)
    )

    pipeline.generate_merged_bar()
    pipeline.side_labeling()
    pipeline.get_df_feature()
    pipeline.side_feature_selection()
    pipeline.train_side_model()

    pipeline.meta_labeling()
    pipeline.meta_feature_selection()
    pipeline.train_meta_model()

    calmar_ratio = pipeline.backtest()
    return calmar_ratio
