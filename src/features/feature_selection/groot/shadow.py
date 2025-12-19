"""
Shadow 特征生成模块

创建 shadow features：复制所有特征并随机打乱，用于特征重要性对比。
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def create_shadow_features(
    X: pd.DataFrame,
    random_state: int = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    创建 shadow features：复制所有特征并随机打乱

    Shadow features 是原始特征的随机打乱版本，用作特征重要性的基准。
    如果一个真实特征的重要性低于 shadow 特征的最大重要性，则应被剔除。

    Parameters
    ----------
    X : pd.DataFrame
        原始特征数据框
    random_state : int, optional
        随机种子，用于可重复性

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - 合并后的数据框（原始特征 + shadow 特征）
        - shadow 特征的列名列表
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 创建 shadow 特征：复制并打乱
    X_shadow = X.copy()
    for col in X_shadow.columns:
        np.random.shuffle(X_shadow[col].values)

    # 重命名 shadow 特征
    shadow_names = [f"ShadowVar{i + 1}" for i in range(X.shape[1])]
    X_shadow.columns = shadow_names

    # 合并原始特征和 shadow 特征
    X_combined = pd.concat([X, X_shadow], axis=1)

    return X_combined, shadow_names
