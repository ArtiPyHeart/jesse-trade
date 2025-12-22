"""
SHAP 计算适配模块

支持 shap 和 fasttreeshap，处理不同 SHAP 版本和 objective 类型。
"""

import warnings
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd


def get_shap_explainer(
    model: lgb.Booster,
    fastshap: bool = False,
) -> Any:
    """
    获取 SHAP explainer

    Parameters
    ----------
    model : lgb.Booster
        训练好的 LightGBM 模型
    fastshap : bool, default=False
        是否使用 fasttreeshap（需要 numpy<2.0）

    Returns
    -------
    Any
        SHAP TreeExplainer 实例
    """
    if fastshap:
        try:
            from fasttreeshap import TreeExplainer

            return TreeExplainer(model, algorithm="v1")
        except ImportError:
            warnings.warn(
                "fasttreeshap 未安装或不可用，回退到 shap。"
                "如需使用 fasttreeshap，请确保 numpy<2.0。"
            )
            fastshap = False

    if not fastshap:
        import shap

        return shap.TreeExplainer(model)


def compute_shap_importance(
    X: pd.DataFrame,
    model: lgb.Booster,
    objective: str,
    fastshap: bool = False,
    num_class: int = 0,
    max_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """
    计算 SHAP 特征重要性

    Parameters
    ----------
    X : pd.DataFrame
        特征数据
    model : lgb.Booster
        训练好的 LightGBM 模型
    objective : str
        LightGBM objective
    fastshap : bool, default=False
        是否使用 fasttreeshap
    num_class : int, default=0
        多分类任务的类别数量（用于旧版 SHAP 的 bias 处理）
    max_samples : int, optional
        SHAP 计算的最大样本数，超过则下采样
    batch_size : int, optional
        SHAP 分批计算的批大小
    random_state : int, optional
        采样随机种子

    Returns
    -------
    Dict[str, float]
        特征名到重要性的映射
    """
    explainer = get_shap_explainer(model, fastshap=fastshap)

    if max_samples is not None and len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=random_state)

    try:
        batch_size = None if batch_size is None or batch_size <= 0 else batch_size
        n_rows = len(X)
        n_features = X.shape[1]
        new_shap = _is_new_shap_version()
        if batch_size is None or batch_size >= n_rows:
            shap_matrix = explainer.shap_values(X)
            shap_imp = _aggregate_shap_values(
                shap_matrix=shap_matrix,
                objective=objective,
                n_features=n_features,
                new_shap=new_shap,
                fastshap=fastshap,
                num_class=num_class,
            )
        else:
            shap_sum = None
            for start in range(0, n_rows, batch_size):
                X_batch = X.iloc[start : start + batch_size]
                shap_matrix = explainer.shap_values(X_batch)
                batch_imp = _aggregate_shap_values(
                    shap_matrix=shap_matrix,
                    objective=objective,
                    n_features=n_features,
                    new_shap=new_shap,
                    fastshap=fastshap,
                    num_class=num_class,
                )
                batch_weight = len(X_batch)
                if shap_sum is None:
                    shap_sum = np.asarray(batch_imp, dtype=np.float64) * batch_weight
                else:
                    shap_sum += np.asarray(batch_imp, dtype=np.float64) * batch_weight
            shap_imp = shap_sum / n_rows
    except Exception as e:
        raise RuntimeError(f"SHAP 计算失败: {str(e)}")

    # 创建特征名到重要性的映射
    importance = dict(zip(X.columns, shap_imp))

    return importance


def _is_new_shap_version() -> bool:
    try:
        from shap import __version__ as shap_version

        major, minor = map(int, shap_version.split(".")[:2])
        return (major, minor) >= (0, 45)
    except Exception:
        return False


def _aggregate_shap_values(
    shap_matrix: Any,
    objective: str,
    n_features: int,
    new_shap: bool,
    fastshap: bool,
    num_class: int = 0,
) -> np.ndarray:
    """
    聚合 SHAP 值计算特征重要性

    Parameters
    ----------
    shap_matrix : Any
        SHAP values（可能是 array 或 list）
    objective : str
        LightGBM objective
    n_features : int
        特征数量
    new_shap : bool
        是否是新版 SHAP (>=0.45)
    fastshap : bool
        是否使用 fasttreeshap
    num_class : int, default=0
        多分类任务的类别数量

    Returns
    -------
    np.ndarray
        每个特征的重要性分数
    """
    is_multiclass = objective in ["softmax", "multiclass", "multi_logloss"]

    if fastshap:
        # fasttreeshap 的处理
        if not new_shap and is_multiclass and isinstance(shap_matrix, list):
            shap_matrix = np.abs(np.concatenate(shap_matrix, axis=1))
        return np.mean(np.abs(shap_matrix), axis=0)

    # 标准 shap 的处理
    if new_shap:
        # SHAP >= 0.45：bias 在单独的属性中，不再作为列
        if is_multiclass:
            # 多分类：shape = (n_samples, n_features, n_classes) 或 list of arrays (old API)
            if isinstance(shap_matrix, list):
                shap_matrix = np.stack(shap_matrix, axis=2)
            return np.mean(np.abs(shap_matrix).sum(axis=2), axis=0)
        else:
            return np.mean(np.abs(shap_matrix), axis=0)
    else:
        # SHAP < 0.45：bias 作为最后一列
        if is_multiclass and num_class > 0:
            # 旧版 SHAP 多分类: 可能是 list 或 2D array
            if isinstance(shap_matrix, list):
                # list 每个元素 shape (n_samples, n_features + 1)
                per_class = []
                for class_matrix in shap_matrix:
                    per_class.append(np.mean(np.abs(class_matrix[:, :-1]), axis=0))
                return np.sum(per_class, axis=0)
            # array 形状为 (n_samples, (n_features+1) * num_class)
            bias_indices = list(
                range(n_features, (n_features + 1) * num_class, n_features + 1)
            )
            shap_matrix = np.delete(shap_matrix, bias_indices, axis=1)
            # 移除 bias 后，还需要移除最后一列（总 bias）
            shap_imp = np.mean(np.abs(shap_matrix[:, :-1]), axis=0)
            # 聚合所有类的重要性（每 n_features 列为一类）
            if len(shap_imp) == n_features * num_class:
                shap_imp = shap_imp.reshape(num_class, n_features).sum(axis=0)
            return shap_imp
        else:
            # 二分类或回归
            if isinstance(shap_matrix, list):
                shap_matrix = shap_matrix[1]  # 取正类
            return np.mean(np.abs(shap_matrix[:, :-1]), axis=0)


def normalize_importance(importance: Dict[str, float]) -> Dict[str, float]:
    """
    归一化特征重要性（使总和为 1）

    Parameters
    ----------
    importance : Dict[str, float]
        特征重要性字典

    Returns
    -------
    Dict[str, float]
        归一化后的重要性字典
    """
    total = sum(importance.values())
    if total == 0:
        return importance
    return {k: v / total for k, v in importance.items()}
