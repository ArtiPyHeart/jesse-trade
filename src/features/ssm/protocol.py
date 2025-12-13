"""
SSM Protocol - 状态空间模型统一协议

定义 DeepSSM 和 LGSSM 的统一接口，支持批量转换和单步推理。
"""

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class SSMProtocol(Protocol):
    """
    状态空间模型统一协议

    所有 SSM 适配器必须实现此协议，以支持：
    - 批量训练和转换 (fit/transform)
    - 单步实时推理 (inference)
    - 状态持久化 (save/load)
    """

    @property
    def is_fitted(self) -> bool:
        """模型是否已训练"""
        ...

    @property
    def state_dim(self) -> int:
        """状态维度（输出维度）"""
        ...

    @property
    def obs_dim(self) -> int:
        """观测维度（输入维度）"""
        ...

    @property
    def prefix(self) -> str:
        """
        列名前缀

        用于生成 DataFrame 列名，如 'deep_ssm' -> 'deep_ssm_0', 'deep_ssm_1', ...
        """
        ...

    def fit(self, X: pd.DataFrame) -> "SSMProtocol":
        """
        训练模型

        Args:
            X: 输入数据，形状为 (n_samples, obs_dim)

        Returns:
            self，支持链式调用
        """
        ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        批量转换

        Args:
            X: 输入数据，形状为 (n_samples, obs_dim)

        Returns:
            转换后的 DataFrame，列名为 {prefix}_{i}
        """
        ...

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        训练并转换

        Args:
            X: 输入数据

        Returns:
            转换后的 DataFrame
        """
        ...

    def inference(self, observation: np.ndarray) -> np.ndarray:
        """
        单步推理

        自动维护内部状态，每次调用更新状态并返回新状态。

        Args:
            observation: 单个观测值，形状为 (obs_dim,)

        Returns:
            当前状态，形状为 (state_dim,)
        """
        ...

    def reset_state(self) -> None:
        """
        重置内部状态

        用于新序列开始时重置状态，适用场景：
        - 策略启动
        - 数据断点
        - 新交易日开始
        """
        ...

    def save(self, path: str) -> None:
        """
        保存模型

        Args:
            path: 保存路径（不含扩展名）
        """
        ...

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SSMProtocol":
        """
        加载模型

        Args:
            path: 模型路径（不含扩展名）
            device: 设备类型

        Returns:
            加载的模型实例
        """
        ...
