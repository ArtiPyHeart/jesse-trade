"""
SHAP 计算适配模块

支持 shap 和 fasttreeshap，处理不同 SHAP 版本和 objective 类型。
"""

import warnings
from typing import Any, Dict

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

    Returns
    -------
    Dict[str, float]
        特征名到重要性的映射
    """
    explainer = get_shap_explainer(model, fastshap=fastshap)

    try:
        shap_matrix = explainer.shap_values(X)
    except Exception as e:
        raise RuntimeError(f"SHAP 计算失败: {str(e)}")

    # 检测 SHAP 版本
    try:
        from shap import __version__ as shap_version

        major, minor = map(int, shap_version.split(".")[:2])
        new_shap = (major, minor) >= (0, 45)
    except Exception:
        new_shap = False

    # 计算重要性
    shap_imp = _aggregate_shap_values(
        shap_matrix=shap_matrix,
        objective=objective,
        n_features=X.shape[1],
        new_shap=new_shap,
        fastshap=fastshap,
    )

    # 创建特征名到重要性的映射
    importance = dict(zip(X.columns, shap_imp))

    return importance


def _aggregate_shap_values(
    shap_matrix: Any,
    objective: str,
    n_features: int,
    new_shap: bool,
    fastshap: bool,
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
            # 多分类：shape = (n_samples, n_features, n_classes)
            return np.mean(np.abs(shap_matrix).sum(axis=2), axis=0)
        else:
            return np.mean(np.abs(shap_matrix), axis=0)
    else:
        # SHAP < 0.45：bias 作为最后一列
        if is_multiclass:
            # 需要移除每个类的 bias 列
            # 旧版本多分类可能返回 list
            if isinstance(shap_matrix, list):
                shap_matrix = shap_matrix[1]  # 取正类
            return np.mean(np.abs(shap_matrix[:, :-1]), axis=0)
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
