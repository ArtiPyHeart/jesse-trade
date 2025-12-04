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
from .feature_select import FeatureSelector

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

    def tuning_classifier(
        self, selector: FeatureSelector, feature_names: list[str]
    ) -> tuple[dict, float]:
        all_feats = selector.get_all_features(self.train_X)[feature_names]
        print(f"{len(feature_names)} features selected")

        x, y = drop_na_and_align_x_and_y(all_feats, self.train_Y)

        # LightGBM prefers contiguous float32 arrays; cache once to reuse across trials
        x = np.ascontiguousarray(x.to_numpy(dtype=np.float32))

        # 固定max_bin参数，使用 free_raw_data=True 释放原始数据避免内存泄漏
        dtrain = lgb.Dataset(x, y, free_raw_data=True, params={"max_bin": 255})
        cv_folds = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y)
        )

        def objective(trial):
            boosting_type = trial.suggest_categorical("boosting", ["gbdt", "dart"])
            params = {
                "objective": "binary",
                # 不在params中指定metric，因为使用feval时会冲突
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
                # 移除过于激进的性能优化参数，优先保证模型质量
                "feature_pre_filter": False,
            }
            # 如果使用 dart，添加 dart 特有参数
            if boosting_type == "dart":
                params["drop_rate"] = trial.suggest_float("drop_rate", 0.1, 0.5)
                params["skip_drop"] = trial.suggest_float("skip_drop", 0.1, 0.5)

            num_boost_round = trial.suggest_int("num_boost_round", 300, 1500)
            # 注意：使用feval时，LightGBMPruningCallback需要metric名称而非"metric-mean"格式
            # 实际callback接收的是('valid', 'f1', value, is_higher_better, std)格式
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
        # 设置 Optuna 日志级别为警告，隐藏详细日志
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # 使用 show_progress_bar 显示进度条
        # n_jobs=1 在M4 Pro上避免过度并行导致的性能下降
        study.optimize(
            objective, n_trials=300, n_jobs=1, show_progress_bar=True
        )  # 增加试验次数

        params = {
            "objective": "binary",
            "num_threads": -1,
            "verbose": -1,
            **study.best_params,
        }
        best_value = study.best_value

        # 🔧 显式清理 Optuna study 和 Dataset，防止内存泄漏
        del study
        del dtrain
        gc.collect()

        return params, best_value

    def tuning_regressor(
        self, selector: FeatureSelector, feature_names: list[str]
    ) -> tuple[dict, float]:
        all_feats = selector.get_all_features(self.train_X)[feature_names]
        print(f"{len(feature_names)} features selected")

        x, y = drop_na_and_align_x_and_y(all_feats, self.train_Y)

        # LightGBM prefers contiguous float32 arrays; cache once to reuse across trials
        x = np.ascontiguousarray(x.to_numpy(dtype=np.float32))
        # 固定max_bin参数，使用 free_raw_data=True 释放原始数据避免内存泄漏
        dtrain = lgb.Dataset(x, y, free_raw_data=True, params={"max_bin": 255})
        cv_folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(x))

        # 预计算训练集标签的方差，用于计算R²
        y_var = np.var(y)

        # 自定义R²评估函数（用于LightGBM的feval参数）
        def r2_eval(preds, eval_dataset):
            """
            计算R²分数
            R² = 1 - (MSE / Var(y))
            """
            y_true = eval_dataset.get_label()
            mse = np.mean((y_true - preds) ** 2)
            r2 = 1 - (mse / y_var)
            # 返回 (metric_name, metric_value, is_higher_better)
            return "r2", r2, True

        def objective(trial):
            boosting_type = trial.suggest_categorical("boosting", ["gbdt", "dart"])
            params = {
                "objective": "regression",
                # 不在params中指定metric，使用feval
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
                # 添加 regression 特有的参数
                "min_sum_hessian_in_leaf": trial.suggest_float(
                    "min_sum_hessian_in_leaf", 1e-3, 10, log=True
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.02, 0.1, log=True
                ),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                # 移除过于激进的性能优化参数
                "feature_pre_filter": False,
            }

            # 如果使用 dart，添加 dart 特有参数
            if boosting_type == "dart":
                params["drop_rate"] = trial.suggest_float("drop_rate", 0.1, 0.5)
                params["skip_drop"] = trial.suggest_float("skip_drop", 0.1, 0.5)

            num_boost_round = trial.suggest_int("num_boost_round", 300, 1500)

            # 使用LightGBMPruningCallback监控R²指标
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
                feval=r2_eval,  # 使用自定义R²评估函数
                callbacks=callbacks,
            )
            # 返回交叉验证的平均R²分数
            return model_res["valid r2-mean"][-1]

        study = optuna.create_study(
            direction="maximize",  # R²需要最大化
            pruner=optuna.pruners.HyperbandPruner(),
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=100,
                multivariate=True,
                constant_liar=True,
                warn_independent_sampling=False,
            ),
        )
        # 设置 Optuna 日志级别为警告，隐藏详细日志
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # 使用 show_progress_bar 显示进度条
        # n_jobs=1 在M4 Pro上避免过度并行导致的性能下降
        study.optimize(
            objective, n_trials=300, n_jobs=1, show_progress_bar=True
        )  # 增加试验次数

        params = {
            "objective": "regression",
            "num_threads": -1,
            "verbose": -1,
            **study.best_params,
        }
        best_value = study.best_value

        # 🔧 显式清理 Optuna study 和 Dataset，防止内存泄漏
        del study
        del dtrain
        gc.collect()

        return params, best_value  # 返回R²值
