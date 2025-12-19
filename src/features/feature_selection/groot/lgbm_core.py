"""
LightGBM 训练核心模块

兼容 LightGBM 4.6.0，使用 callbacks 替代 deprecated 参数。
"""

from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd


def set_lgb_parameters(
    y: pd.Series,
    objective: str,
    n_jobs: int = 0,
    lgbm_params: Optional[Dict[str, Any]] = None,
    silent: bool = True,
) -> Dict[str, Any]:
    """
    设置 LightGBM 参数

    Parameters
    ----------
    y : pd.Series
        目标变量，用于检测类别不平衡
    objective : str
        LightGBM objective，如 'binary', 'rmse', 'softmax'
    n_jobs : int, default=0
        并行数，0 表示使用 OpenMP 默认线程数
    lgbm_params : Dict[str, Any], optional
        用户自定义 LightGBM 参数
    silent : bool, default=True
        是否静默模式

    Returns
    -------
    Dict[str, Any]
        LightGBM 参数字典
    """
    params = dict(lgbm_params) if lgbm_params is not None else {}

    params["objective"] = objective
    params["verbosity"] = -1  # 静默 LightGBM 内部日志

    # 处理多分类
    if objective == "softmax":
        params["num_class"] = len(np.unique(y))

    # 处理分类任务的类别不平衡
    clf_objectives = [
        "binary",
        "softmax",
        "multi_logloss",
        "multiclassova",
        "multiclass",
        "multiclass_ova",
        "ova",
        "ovr",
        "binary_logloss",
    ]
    if objective in clf_objectives:
        y_int = y.astype(int)
        y_freq = pd.Series(y_int.fillna(0)).value_counts(normalize=True)
        n_classes = y_freq.size

        # 如果检测到多分类但 objective 不是 softmax
        if n_classes > 2 and objective != "softmax":
            params["objective"] = "softmax"
            params["num_class"] = n_classes
            if not silent:
                print("检测到多分类任务，设置 objective 为 softmax")

        # 处理类别不平衡
        main_class_ratio = y_freq.iloc[0]
        if main_class_ratio > 0.8:
            params["is_unbalance"] = True
            if not silent:
                print("检测到类别不平衡，启用 is_unbalance")

    # 设置线程数
    params["num_threads"] = n_jobs

    # 移除可能冲突的迭代次数参数（由 num_boost_round 控制）
    keys_to_remove = [
        "num_iterations",
        "num_iteration",
        "n_iter",
        "num_tree",
        "num_trees",
        "num_round",
        "num_rounds",
        "nrounds",
        "num_boost_round",
        "n_estimators",
        "max_iter",
    ]
    for key in keys_to_remove:
        params.pop(key, None)

    return params


def train_lgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict[str, Any],
    weight_train: Optional[pd.Series] = None,
    weight_val: Optional[pd.Series] = None,
    num_boost_round: int = 10000,
    early_stopping_rounds: int = 20,
    verbose_eval: int = 0,
) -> Tuple[lgb.Booster, int]:
    """
    训练 LightGBM 模型，兼容 4.6.0

    Parameters
    ----------
    X_train : pd.DataFrame
        训练特征
    y_train : pd.Series
        训练标签
    X_val : pd.DataFrame
        验证特征
    y_val : pd.Series
        验证标签
    params : Dict[str, Any]
        LightGBM 参数
    weight_train : pd.Series, optional
        训练样本权重
    weight_val : pd.Series, optional
        验证样本权重
    num_boost_round : int, default=10000
        最大迭代轮数
    early_stopping_rounds : int, default=20
        早停轮数
    verbose_eval : int, default=0
        日志打印间隔，0 表示不打印

    Returns
    -------
    Tuple[lgb.Booster, int]
        训练好的模型和最佳迭代次数
    """
    # 创建 Dataset
    d_train = lgb.Dataset(
        X_train,
        label=y_train,
        weight=weight_train.values if weight_train is not None else None,
    )
    d_valid = lgb.Dataset(
        X_val,
        label=y_val,
        weight=weight_val.values if weight_val is not None else None,
        reference=d_train,
    )

    # 设置 callbacks（LightGBM 4.x 方式）
    callbacks: List[Any] = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
    ]
    if verbose_eval > 0:
        callbacks.append(lgb.log_evaluation(period=verbose_eval))

    # 训练模型
    model = lgb.train(
        params=params,
        train_set=d_train,
        num_boost_round=num_boost_round,
        valid_sets=[d_train, d_valid],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # 获取最佳迭代次数（可能为 None，需回退）
    best_iteration = model.best_iteration
    if best_iteration is None or best_iteration <= 0:
        best_iteration = model.current_iteration()

    return model, best_iteration


def split_train_val(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    weight: Optional[pd.Series] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    Optional[pd.Series],
    Optional[pd.Series],
]:
    """
    按索引分割训练集和验证集

    Parameters
    ----------
    X : pd.DataFrame
        特征数据
    y : pd.Series
        目标变量
    train_idx : np.ndarray
        训练集索引
    val_idx : np.ndarray
        验证集索引
    weight : pd.Series, optional
        样本权重

    Returns
    -------
    Tuple
        X_train, X_val, y_train, y_val, weight_train, weight_val
    """
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    weight_train = None
    weight_val = None
    if weight is not None:
        weight_train = weight.iloc[train_idx]
        weight_val = weight.iloc[val_idx]

    return X_train, X_val, y_train, y_val, weight_train, weight_val
