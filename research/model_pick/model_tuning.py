import numpy as np
import optuna
import pandas as pd
from jesse.helpers import date_to_timestamp
from sklearn.metrics import f1_score
import lightgbm as lgb

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

        def objective(trial):
            params = {
                "objective": "binary",
                "metric": METRIC,
                "num_threads": -1,
                "verbose": -1,
                "is_unbalance": trial.suggest_categorical(
                    "is_unbalance", [True, False]
                ),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "boosting": trial.suggest_categorical("boosting", ["gbdt", "dart"]),
                "num_leaves": trial.suggest_int("num_leaves", 31, 300),
                "max_depth": trial.suggest_int("max_depth", 30, 1000),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-8, 1),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 100),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 100),
            }
            dtrain = lgb.Dataset(all_feats, self.train_Y)
            model_res = lgb.cv(
                params,
                dtrain,
                num_boost_round=trial.suggest_int("num_boost_round", 100, 1500),
                feval=eval_metric,
            )
            return model_res[f"valid {METRIC}-mean"][-1]

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(),
            sampler=optuna.samplers.TPESampler(n_startup_trials=50),
        )
        study.optimize(objective, n_trials=150, n_jobs=1)

        params = {
            "objective": "binary",
            "num_threads": -1,
            "verbose": -1,
            **study.best_params,
        }

        return params, study.best_value

    def tuning_regressor(
        self, selector: FeatureSelector, feature_names: list[str]
    ) -> tuple[dict, float]:
        all_feats = selector.get_all_features(self.train_X)[feature_names]

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "rmse",  # 使用正确的别名
                "num_threads": -1,
                "verbose": -1,
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "boosting": trial.suggest_categorical("boosting", ["gbdt", "dart"]),
                "num_leaves": trial.suggest_int("num_leaves", 31, 300),
                "max_depth": trial.suggest_int("max_depth", 30, 1000),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-8, 1),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 100),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 100),
                # 添加 regression 特有的参数
                "min_sum_hessian_in_leaf": trial.suggest_float(
                    "min_sum_hessian_in_leaf", 1e-3, 10
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
            }

            # 如果使用 dart，添加 dart 特有参数
            if params["boosting"] == "dart":
                params["drop_rate"] = trial.suggest_float("drop_rate", 0.1, 0.5)

            dtrain = lgb.Dataset(all_feats, self.train_Y)
            model_res = lgb.cv(
                params,
                dtrain,
                num_boost_round=trial.suggest_int("num_boost_round", 100, 1500),
            )
            return -model_res["valid rmse-mean"][-1]  # 负值因为要最小化 RMSE

        study = optuna.create_study(
            direction="minimize",  # 回归任务通常最小化损失
            pruner=optuna.pruners.HyperbandPruner(),
            sampler=optuna.samplers.TPESampler(n_startup_trials=50),
        )
        study.optimize(objective, n_trials=150, n_jobs=1)

        params = {
            "objective": "regression",
            "num_threads": -1,
            "verbose": -1,
            **study.best_params,
        }

        return params, -study.best_value  # 返回正的 RMSE 值
