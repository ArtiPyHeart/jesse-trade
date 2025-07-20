from contextlib import contextmanager

import catboost as ctb
import numpy as np
import optuna
import pandas as pd
from hmmlearn.hmm import GMMHMM
from jesse import helpers
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from custom_indicators.all_features import feature_bundle
from custom_indicators.toolbox.bar.fusion.base import FusionBarContainerBase
from custom_indicators.toolbox.entropy.apen_sampen import sample_entropy_numba
from custom_indicators.toolbox.feature_selection.rfcq_selector import RFCQSelector
from custom_indicators.utils.math_tools import log_ret_from_candles


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
        self,
        n1: int,
        n2: int,
        n_entropy: int,
        en_div_thres: float,
        threshold: float,
        max_bars=500000,
    ):
        super().__init__(max_bars, threshold)
        self.N_1 = n1
        self.N_2 = n2
        self.N_ENTROPY = n_entropy
        self.EN_DIV_THRES = en_div_thres

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
            entropy_log_ret_list = log_ret_from_candles(
                candles[self.max_lookback - self.N_ENTROPY :], self.N_ENTROPY
            )
        else:
            entropy_log_ret_list = log_ret_from_candles(candles, self.N_ENTROPY)

        entropy_array = Parallel()(
            delayed(sample_entropy_numba)(i) for i in entropy_log_ret_list
        )
        entropy_array = np.array(entropy_array) / self.EN_DIV_THRES

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
    def side_model_split_timestamp(self):
        return helpers.date_to_timestamp("2024-09-01")

    @property
    def meta_model_split_timestamp(self):
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

    def init_bar_container(self, n1, n2, n_entropy, en_div_thres, threshold=0.5):
        self.bar_container = TuningBarContainer(
            n1, n2, n_entropy, en_div_thres, threshold
        )

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

            try:
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
            except Exception:
                return -100

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
                sampler=optuna.samplers.TPESampler(n_startup_trials=5),
            )
            study.optimize(objective, n_trials=10)
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
            feature_bundle(self.merged_bar[:-1], sequential=True, lightweighted=True),
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

    def train_side_model(self) -> float:
        mask = self.df_feature.index < self.side_model_split_timestamp
        feature_masked = self.df_feature[self.side_feature_names][mask]
        label_masked = self.side_label[mask]
        # params = {
        #     "objective": "binary",
        #     "metric": "auc",
        #     "num_threads": -1,
        #     "verbose": -1,
        #     "is_unbalance": True,
        #     "extra_trees": False,
        #     "num_leaves": 62,
        #     "max_depth": 294,
        #     "min_gain_to_split": 0.3876711100653668,
        #     "min_data_in_leaf": 200,
        #     "lambda_l1": 10,
        #     "lambda_l2": 50,
        #     "num_boost_round": 600,
        # }
        self.side_model = ctb.CatBoostClassifier(
            verbose=False,
            auto_class_weights="Balanced",
            thread_count=-1,
            early_stopping_rounds=50,
            subsample=0.8,
            bootstrap_type="Bernoulli",
            rsm=0.7,
        )
        self.side_model.fit(feature_masked, label_masked)

        self.df_feature["model"] = self.side_model.predict_proba(
            self.df_feature[self.side_feature_names]
        )[:, 1]

        testset_mask = self.df_feature.index >= self.side_model_split_timestamp
        test_auc = roc_auc_score(
            self.side_label[testset_mask],
            self.df_feature["model"][testset_mask],
        )
        return test_auc

    def meta_labeling(self):
        side_res = self.side_model.predict_proba(
            self.df_feature[self.side_feature_names]
        )[:, 1]
        side_pred_label = np.where(side_res > 0.5, 1, -1)

        close_prices = self.merged_bar[:, 2]
        len_gap = len(close_prices) - len(side_pred_label)
        if len_gap > 0:
            close_prices = close_prices[len_gap - 1 : -1]
        assert len(close_prices) == len(side_pred_label)

        meta_label = np.zeros(len(side_pred_label))

        start_idx = 0
        cumsum_ret = 0
        start_price = 0
        for idx, (i, p) in enumerate(zip(side_pred_label, close_prices)):
            if i == 1 or i == -1:
                if idx == 0:
                    # 开始持仓
                    start_idx = idx
                    start_price = p
                    cumsum_ret -= self.trading_fee
                elif side_pred_label[idx - 1] != i:
                    # 反向持仓，先结算收益
                    cumsum_ret -= self.trading_fee
                    cumsum_ret += np.log(p / start_price) * side_pred_label[idx - 1]
                    if cumsum_ret > 0:
                        meta_label[start_idx:idx] = 1
                    cumsum_ret = 0
                    start_price = p
                    start_idx = idx
                    cumsum_ret -= self.trading_fee
                else:
                    # 继续持仓
                    continue
            else:
                raise ValueError(f"side_pred_label[{idx}] = {i} is not valid")
        else:
            last_price = self.merged_bar[-1, 2]
            # 结算最后一根bar的持仓, 可能还没有结算，所以先不加trade fee
            if i == side_pred_label[idx - 1]:
                # 已经开仓，结算
                cumsum_ret += (
                    np.log(last_price / start_price) * side_pred_label[idx - 1]
                )
            else:
                # 反向开仓
                cumsum_ret -= self.trading_fee
                cumsum_ret += (
                    np.log(last_price / start_price) * side_pred_label[idx - 1]
                )

            if cumsum_ret > 0:
                meta_label[start_idx:] = 1

        self.meta_label = meta_label
        return meta_label

    def meta_feature_selection(self):
        selector: RFCQSelector = RFCQSelector(verbose=False)
        selector.fit(self.df_feature, self.meta_label)
        meta_res = pd.Series(
            selector.relevance_, index=selector.variables_
        ).sort_values(ascending=False)
        self.meta_feature_names = meta_res[meta_res > 0].index.tolist()

    def train_meta_model(self) -> tuple[float, float, float]:
        mask = self.df_feature.index < self.meta_model_split_timestamp
        feature_masked = self.df_feature[self.meta_feature_names][mask]
        label_masked = self.meta_label[mask]

        # METRIC = "f1"
        #
        # def eval_metric(preds, eval_dataset):
        #     metric_name = METRIC
        #     y_true = eval_dataset.get_label()
        #     value = f1_score(y_true, preds > 0.5, average="weighted")
        #     higher_better = True
        #     return metric_name, value, higher_better
        #
        # def objective(trial: optuna.Trial):
        #     params = {
        #         "objective": "binary",
        #         "is_unbalance": True,
        #         "num_threads": -1,
        #         "verbose": -1,
        #         "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        #         "boosting": trial.suggest_categorical("boosting", ["gbdt", "dart"]),
        #         "num_leaves": trial.suggest_int("num_leaves", 31, 500),
        #         "max_depth": trial.suggest_int("max_depth", 30, 1000),
        #         "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-8, 1),
        #         "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
        #         "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 100),
        #         "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 100),
        #     }
        #     dtrain = lgb.Dataset(feature_masked, label_masked)
        #     # dtest = lgb.Dataset(meta_features_test, meta_label_test)
        #     model_res = lgb.cv(
        #         params,
        #         dtrain,
        #         num_boost_round=trial.suggest_int("num_boost_round", 100, 1000),
        #         stratified=True,
        #         feval=eval_metric,
        #     )
        #     return model_res[f"valid {METRIC}-mean"][-1]
        #
        # with optuna_log_manager.silent_optimization():
        #     study = optuna.create_study(
        #         direction="maximize",
        #         pruner=optuna.pruners.HyperbandPruner(),
        #         sampler=optuna.samplers.TPESampler(n_startup_trials=5),
        #     )
        #     study.optimize(objective, n_trials=10, n_jobs=1)
        #
        # params = {
        #     "objective": "binary",
        #     "num_threads": -1,
        #     "verbose": -1,
        #     "is_unbalance": True,
        #     **study.best_params,
        # }
        # dtrain = lgb.Dataset(feature_masked, label_masked)
        self.meta_model = ctb.CatBoostClassifier(
            verbose=False,
            auto_class_weights="Balanced",
            thread_count=-1,
            early_stopping_rounds=50,
            subsample=0.8,
            bootstrap_type="Bernoulli",
            rsm=0.7,
        )
        self.meta_model.fit(feature_masked, label_masked)

        testset_mask = self.df_feature.index >= self.meta_model_split_timestamp
        meta_pred_proba = self.meta_model.predict_proba(
            self.df_feature[self.meta_feature_names][testset_mask]
        )[:, 1]
        test_precision = precision_score(
            self.meta_label[testset_mask],
            (meta_pred_proba > 0.5).astype(int),
            zero_division=0,
            average="weighted",
        )
        test_recall = recall_score(
            self.meta_label[testset_mask],
            (meta_pred_proba > 0.5).astype(int),
            zero_division=0,
            average="weighted",
        )
        test_f1 = f1_score(
            self.meta_label[testset_mask],
            (meta_pred_proba > 0.5).astype(int),
            average="weighted",
        )
        return test_f1, test_precision, test_recall

    def backtest(self):
        side_features = self.df_feature[self.side_feature_names]
        side_features = side_features[
            side_features.index >= self.meta_model_split_timestamp
        ]
        side_pred = self.side_model.predict(side_features)
        side_pred_label = np.where(side_pred > 0.5, 1, -1)

        meta_features = self.df_feature[self.meta_feature_names]
        meta_features = meta_features[
            meta_features.index >= self.meta_model_split_timestamp
        ]
        meta_pred = self.meta_model.predict(meta_features)
        meta_pred_label = np.where(meta_pred > 0.5, 1, 0)

        close_prices = self.merged_bar[:, 2]
        len_gap = len(close_prices) - len(side_pred_label)
        if len_gap > 0:
            close_prices = close_prices[len_gap - 1 : -1]
        assert len(close_prices) == len(side_pred_label)

        one_return = 0
        start_price = 0
        log_ret_list = []
        for idx, (side, meta, p) in enumerate(
            zip(side_pred_label, meta_pred_label, close_prices)
        ):
            if meta == 1:
                if (idx == 0) or (meta_pred_label[idx - 1] == 0):
                    # 开始持仓
                    start_price = p
                    one_return -= self.trading_fee
                    log_ret_list.append(-self.trading_fee)
                elif side_pred_label[idx - 1] != side:
                    # 反向调仓
                    # 先结算
                    one_return -= self.trading_fee
                    one_return += np.log(p / start_price) * side_pred_label[idx - 1]
                    log_ret_list.append(one_return)

                    # 再开仓
                    one_return = 0
                    start_price = p
                    one_return -= self.trading_fee
                    log_ret_list.append(-self.trading_fee)
                else:
                    # 继续持仓
                    continue
            else:
                if (idx == 0) or (meta_pred_label[idx - 1] == 0):
                    continue
                elif meta_pred_label[idx - 1] == 1:
                    # 结算持仓
                    one_return -= self.trading_fee
                    one_return += np.log(p / start_price) * side_pred_label[idx - 1]
                    log_ret_list.append(one_return)
                    one_return = 0
                    start_price = 0
        else:
            last_price = self.merged_bar[-1, 2]
            # 结算最后一根bar的持仓
            if meta == 1:
                if meta_pred_label[idx - 1] == 1:  # 上一根已经开仓，假设持仓继续延续
                    if side == side_pred_label[idx - 1]:
                        one_return += np.log(last_price / start_price) * side
                    else:
                        one_return -= self.trading_fee
                        one_return += (
                            np.log(last_price / start_price) * side_pred_label[idx - 1]
                        )

                    log_ret_list.append(one_return)
                else:
                    # 上一根没有开仓，只结算手续费
                    one_return = -self.trading_fee
                    log_ret_list.append(one_return)

        log_ret_list = pd.Series(log_ret_list)
        cumulative_returns = np.exp(np.cumsum(log_ret_list))
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        total_return = log_ret_list.sum()

        print(f"{total_return = :.2f}, {max_drawdown = :.2f}")
        calmar_ratio = total_return / abs(max_drawdown)
        return calmar_ratio


def tune_pipeline(trial: optuna.Trial):
    pipeline = BacktestPipeline("data/btc_1m.npy")
    n1 = trial.suggest_int("n1", 1, 300)
    n2 = trial.suggest_int("n2", 1, 300)
    n_entropy = trial.suggest_int("n_entropy", 30, 300)
    en_div_thres = trial.suggest_float("en_div_thres", 1, 20)
    pipeline.init_bar_container(n1, n2, n_entropy, en_div_thres)
    raw_threshold_array = pipeline.get_threshold_array()
    threshold_min = np.sum(raw_threshold_array) / (len(pipeline.raw_candles) // 60)
    threshold_max = np.sum(raw_threshold_array) / (len(pipeline.raw_candles) // 540)
    if threshold_min < 0:
        return -1000

    pipeline.set_threshold(
        trial.suggest_float("threshold", threshold_min, threshold_max)
    )

    pipeline.generate_merged_bar()
    pipeline.side_labeling()
    pipeline.get_df_feature()
    pipeline.side_feature_selection()
    side_auc = pipeline.train_side_model()

    meta_label = pipeline.meta_labeling()
    pipeline.meta_feature_selection()
    meta_f1, meta_precision, meta_recall = pipeline.train_meta_model()

    calmar_ratio = pipeline.backtest()

    mask_train = pipeline.df_feature.index < pipeline.meta_model_split_timestamp
    mask_test = pipeline.df_feature.index >= pipeline.meta_model_split_timestamp

    meta_label_train_count = np.unique(meta_label[mask_train], return_counts=True)
    meta_label_test_count = np.unique(meta_label[mask_test], return_counts=True)

    print(
        f"{side_auc = :.6f} {meta_f1 = :.6f} {meta_precision = :.6f} {meta_recall = :.6f} {calmar_ratio = :.6f} {meta_label_train_count = } {meta_label_test_count = }"
    )
    return calmar_ratio * side_auc**3 * meta_f1**2


if __name__ == "__main__":
    from optuna_config import create_robust_study, safe_optimize

    # 方法1: 创建新的研究
    study_name = "backtest_tuning_find_v1_bar"

    with optuna_log_manager.verbose_optimization():
        study = create_robust_study(
            study_name=study_name,
            storage_dir="optuna_storage",
            direction="maximize",
            n_startup_trials=10000,
            percentile_for_pruning=20.0,
        )

        # 使用安全优化函数
        safe_optimize(
            study=study,
            objective=tune_pipeline,
            n_trials=12000,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )
