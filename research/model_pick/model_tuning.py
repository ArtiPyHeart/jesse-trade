import gc

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from jesse.helpers import date_to_timestamp
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold

from src.utils.drop_na import drop_na_and_align_x_and_y

METRIC = "f1"


def eval_metric(preds, eval_dataset):
    metric_name = METRIC
    y_true = eval_dataset.get_label()
    value = f1_score(y_true, preds > 0.5, average="weighted")
    higher_better = True
    return metric_name, value, higher_better


class ModelTuning:
    def __init__(self, train_test_split_date: str, x: pd.DataFrame, y: np.ndarray):
        self.split_date = train_test_split_date

        self.X = x
        self.Y = y

        train_mask = self.X.index.to_numpy() < date_to_timestamp(self.split_date)

        self.train_X = x[train_mask]
        self.train_Y = y[train_mask]

        assert len(self.train_X) == len(self.train_Y)

    def tuning_classifier_direct(
        self, train_x: pd.DataFrame, train_y: np.ndarray
    ) -> tuple[dict, float]:
        """
        直接使用预计算特征调参（分类器）

        与 tuning_classifier 的区别：不依赖 FeatureSelector，
        直接使用传入的特征 DataFrame。

        Args:
            train_x: 预计算的特征 DataFrame（已对齐、无 NaN）
            train_y: 标签数组

        Returns:
            (best_params, best_score): 最优参数和最优 F1 分数
        """
        x, y = drop_na_and_align_x_and_y(train_x, train_y)
        print(f"{train_x.shape[1]} features for tuning")

        # LightGBM prefers contiguous float32 arrays
        x = np.ascontiguousarray(x.to_numpy(dtype=np.float32))

        # 固定max_bin参数，使用 free_raw_data=True 释放原始数据
        dtrain = lgb.Dataset(x, y, free_raw_data=True, params={"max_bin": 255})
        cv_folds = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y)
        )

        def objective(trial):
            boosting_type = trial.suggest_categorical("boosting", ["gbdt", "dart"])
            params = {
                "objective": "binary",
                "num_threads": -1,
                "verbose": -1,
                "is_unbalance": trial.suggest_categorical(
                    "is_unbalance", [True, False]
                ),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "boosting": boosting_type,
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 31, 512),
                "max_depth": trial.suggest_int("max_depth", 10, 50),
                "min_gain_to_split": trial.suggest_float(
                    "min_gain_to_split", 1e-8, 1, log=True
                ),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "feature_pre_filter": False,
            }
            if boosting_type == "dart":
                params["drop_rate"] = trial.suggest_float("drop_rate", 0.1, 0.5)
                params["skip_drop"] = trial.suggest_float("skip_drop", 0.1, 0.5)

            num_boost_round = trial.suggest_int("num_boost_round", 300, 1500)
            pruning_cb = LightGBMPruningCallback(trial, METRIC)
            callbacks = [
                lgb.early_stopping(stopping_rounds=150, verbose=False),
                pruning_cb,
            ]
            model_res = lgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                folds=cv_folds,
                feval=eval_metric,
                callbacks=callbacks,
            )
            return model_res[f"valid {METRIC}-mean"][-1]

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(),
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=100,
                multivariate=True,
                constant_liar=True,
                warn_independent_sampling=False,
            ),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=300, n_jobs=1, show_progress_bar=True)

        params = {
            "objective": "binary",
            "num_threads": -1,
            "verbose": -1,
            **study.best_params,
        }
        best_value = study.best_value

        del study
        del dtrain
        gc.collect()

        return params, best_value

    def tuning_regressor_direct(
        self, train_x: pd.DataFrame, train_y: np.ndarray
    ) -> tuple[dict, float]:
        """
        直接使用预计算特征调参（回归器）

        与 tuning_regressor 的区别：不依赖 FeatureSelector，
        直接使用传入的特征 DataFrame。

        Args:
            train_x: 预计算的特征 DataFrame（已对齐、无 NaN）
            train_y: 标签数组

        Returns:
            (best_params, best_score): 最优参数和最优 R² 分数
        """
        x, y = drop_na_and_align_x_and_y(train_x, train_y)
        print(f"{train_x.shape[1]} features for tuning")

        # LightGBM prefers contiguous float32 arrays
        x = np.ascontiguousarray(x.to_numpy(dtype=np.float32))

        # 固定max_bin参数，使用 free_raw_data=True 释放原始数据
        dtrain = lgb.Dataset(x, y, free_raw_data=True, params={"max_bin": 255})
        cv_folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(x))

        # 预计算训练集标签的方差，用于计算R²
        y_var = np.var(y)

        def r2_eval(preds, eval_dataset):
            y_true = eval_dataset.get_label()
            mse = np.mean((y_true - preds) ** 2)
            r2 = 1 - (mse / y_var)
            return "r2", r2, True

        def objective(trial):
            boosting_type = trial.suggest_categorical("boosting", ["gbdt", "dart"])
            params = {
                "objective": "regression",
                "num_threads": -1,
                "verbose": -1,
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "boosting": boosting_type,
                "num_leaves": trial.suggest_int("num_leaves", 31, 512),
                "max_depth": trial.suggest_int("max_depth", 10, 50),
                "min_gain_to_split": trial.suggest_float(
                    "min_gain_to_split", 1e-8, 1, log=True
                ),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "min_sum_hessian_in_leaf": trial.suggest_float(
                    "min_sum_hessian_in_leaf", 1e-3, 10, log=True
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.1, log=True
                ),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "feature_pre_filter": False,
            }
            if boosting_type == "dart":
                params["drop_rate"] = trial.suggest_float("drop_rate", 0.1, 0.5)
                params["skip_drop"] = trial.suggest_float("skip_drop", 0.1, 0.5)

            num_boost_round = trial.suggest_int("num_boost_round", 300, 1500)
            pruning_cb = LightGBMPruningCallback(trial, "r2")
            callbacks = [
                lgb.early_stopping(stopping_rounds=150, verbose=False),
                pruning_cb,
            ]
            model_res = lgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                folds=cv_folds,
                feval=r2_eval,
                callbacks=callbacks,
            )
            return model_res["valid r2-mean"][-1]

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(),
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=100,
                multivariate=True,
                constant_liar=True,
                warn_independent_sampling=False,
            ),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=300, n_jobs=1, show_progress_bar=True)

        params = {
            "objective": "regression",
            "num_threads": -1,
            "verbose": -1,
            **study.best_params,
        }
        best_value = study.best_value

        del study
        del dtrain
        gc.collect()

        return params, best_value
