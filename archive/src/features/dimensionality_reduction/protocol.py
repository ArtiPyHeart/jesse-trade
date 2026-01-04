"""
DimensionReducer Protocol - 降维器通用协议

定义降维器的统一接口，支持 ARD-VAE 及未来其他降维方法。
"""

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DimensionReducerProtocol(Protocol):
    """
    降维器通用协议

    所有降维器必须实现此协议，采用 sklearn 风格的 fit/transform 接口。

    Notes
    -----
    - save/load 使用两个参数：path (目录) 和 model_name (模型名称)
    - 这种设计允许将降维模型与 LightGBM 模型保存在同一目录下
    - 命名规范示例：
        - path/c_L5_N2.safetensors  (降维模型权重)
        - path/c_L5_N2.json          (降维模型元数据)
        - path/model_c_L5_N2.txt     (LightGBM 模型)
    """

    @property
    def is_fitted(self) -> bool:
        """模型是否已训练"""
        ...

    @property
    def n_components(self) -> int:
        """有效降维维度"""
        ...

    def fit(self, X: pd.DataFrame) -> "DimensionReducerProtocol":
        """
        训练降维模型

        Args:
            X: 输入数据，形状为 (n_samples, n_features)

        Returns:
            self，支持链式调用
        """
        ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        应用降维

        Args:
            X: 输入数据，列名需与训练时一致

        Returns:
            降维后的 DataFrame
        """
        ...

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        训练并应用降维

        Args:
            X: 输入数据

        Returns:
            降维后的 DataFrame
        """
        ...

    def save(self, path: str, model_name: str) -> None:
        """
        保存模型

        Args:
            path: 保存目录路径
            model_name: 模型名称字符串（如 "c_L5_N2"）
        """
        ...

    @classmethod
    def load(cls, path: str, model_name: str) -> "DimensionReducerProtocol":
        """
        加载模型

        Args:
            path: 模型目录路径
            model_name: 模型名称字符串

        Returns:
            加载的降维器实例
        """
        ...
