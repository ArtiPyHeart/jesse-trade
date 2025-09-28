import numpy as np
import optuna
import pandas as pd
from jesse.helpers import date_to_timestamp
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
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
        print(f"{len(feature_names)} features selected")

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
            # 对于分类任务，使用 StratifiedKFold 保持类别平衡
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            model_res = lgb.cv(
                params,
                dtrain,
                num_boost_round=trial.suggest_int("num_boost_round", 100, 1500),
                folds=folds,
                feval=eval_metric,
            )
            return model_res[f"valid {METRIC}-mean"][-1]

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(),
            sampler=optuna.samplers.TPESampler(n_startup_trials=25),
        )
        # 设置 Optuna 日志级别为警告，隐藏详细日志
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # 使用 show_progress_bar 显示进度条
        study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)

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
        print(f"{len(feature_names)} features selected")

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": ["rmse", "l2"],  # 需要同时指定rmse和l2才能计算r2
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
            # 对于回归任务，使用普通的 KFold 而不是 StratifiedKFold
            folds = KFold(n_splits=5, shuffle=True, random_state=42)

            # 自定义R²评估函数
            def r2_eval(preds, eval_dataset):
                y_true = eval_dataset.get_label()
                # 计算R²
                r2 = r2_score(y_true, preds)
                # LightGBM要求返回(评估名称, 评估值, 是否越大越好)
                return "r2", r2, True

            model_res = lgb.cv(
                params,
                dtrain,
                num_boost_round=trial.suggest_int("num_boost_round", 100, 1500),
                folds=folds,
                feval=r2_eval,  # 使用自定义的R²评估函数
                return_cvbooster=True,
            )
            # 返回R²分数（已经是越大越好，所以不需要取负）
            return model_res["valid r2-mean"][-1]

        study = optuna.create_study(
            direction="maximize",  # R²需要最大化
            pruner=optuna.pruners.HyperbandPruner(),
            sampler=optuna.samplers.TPESampler(n_startup_trials=25),
        )
        # 设置 Optuna 日志级别为警告，隐藏详细日志
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # 使用 show_progress_bar 显示进度条
        study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)

        params = {
            "objective": "regression",
            "num_threads": -1,
            "verbose": -1,
            **study.best_params,
        }

        return params, study.best_value  # 返回R²值
