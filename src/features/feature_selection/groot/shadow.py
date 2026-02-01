"""
Shadow 特征生成模块

创建 shadow features：复制所有特征并随机打乱，用于特征重要性对比。
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def create_shadow_features(
    X: Union[pd.DataFrame, np.ndarray],
    random_state: Optional[int] = None,
) -> Tuple[Union[pd.DataFrame, np.ndarray], List[str]]:
    """
    创建 shadow features：复制所有特征并随机打乱

    Shadow features 是原始特征的随机打乱版本，用作特征重要性的基准。
    如果一个真实特征的重要性低于 shadow 特征的最大重要性，则应被剔除。

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        原始特征数据
    random_state : int, optional
        随机种子，用于可重复性

    Returns
    -------
    Tuple[pd.DataFrame or np.ndarray, List[str]]
        - 合并后的数据（原始特征 + shadow 特征）
        - shadow 特征的列名列表
    """
    rng = np.random.default_rng(random_state)

    if isinstance(X, pd.DataFrame):
        X_values = X.to_numpy(copy=False)
        n_rows, n_cols = X_values.shape
        combined = np.empty((n_rows, n_cols * 2), dtype=X_values.dtype)
        combined[:, :n_cols] = X_values
        for j in range(n_cols):
            combined[:, n_cols + j] = rng.permutation(X_values[:, j])

        shadow_names = [f"ShadowVar{i + 1}" for i in range(n_cols)]
        combined_columns = list(X.columns) + shadow_names
        X_combined = pd.DataFrame(combined, columns=combined_columns, index=X.index)
        return X_combined, shadow_names

    if isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        n_rows, n_cols = X.shape
        combined = np.empty((n_rows, n_cols * 2), dtype=X.dtype)
        combined[:, :n_cols] = X
        for j in range(n_cols):
            combined[:, n_cols + j] = rng.permutation(X[:, j])
        shadow_names = [f"ShadowVar{i + 1}" for i in range(n_cols)]
        return combined, shadow_names

    raise TypeError("X must be pandas.DataFrame or numpy.ndarray")
