"""
SSM Adapters - DeepSSM 和 LGSSM 的适配器实现

提供统一的 SSMProtocol 接口，封装不同 SSM 模型的实现细节。
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.models.deep_ssm import DeepSSM, DeepSSMConfig
from src.models.lgssm import LGSSM, LGSSMConfig


class DeepSSMAdapter:
    """
    DeepSSM 适配器

    封装 DeepSSM 模型，提供统一的 SSMProtocol 接口。
    """

    def __init__(
        self,
        config: Optional[DeepSSMConfig] = None,
        obs_dim: Optional[int] = None,
        prefix: str = "deep_ssm",
        **kwargs,
    ):
        """
        初始化 DeepSSM 适配器

        Args:
            config: DeepSSM 配置
            obs_dim: 观测维度（如果 config 为 None）
            prefix: 列名前缀
            **kwargs: 传递给 DeepSSMConfig 的额外参数
        """
        if config is not None:
            self._model = DeepSSM(config=config)
        elif obs_dim is not None:
            self._model = DeepSSM(obs_dim=obs_dim, **kwargs)
        else:
            # 延迟初始化，等待 fit 时确定 obs_dim
            self._model = None
            self._pending_kwargs = kwargs

        self._realtime_processor = None
        self._prefix = prefix

    @property
    def is_fitted(self) -> bool:
        """模型是否已训练"""
        return self._model is not None and self._model.is_fitted

    @property
    def state_dim(self) -> int:
        """状态维度"""
        if self._model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")
        return self._model.config.state_dim

    @property
    def obs_dim(self) -> int:
        """观测维度"""
        if self._model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")
        return self._model.config.obs_dim

    @property
    def prefix(self) -> str:
        """列名前缀"""
        return self._prefix

    def fit(self, X: pd.DataFrame) -> "DeepSSMAdapter":
        """
        训练模型

        Args:
            X: 输入数据，形状为 (n_samples, obs_dim)

        Returns:
            self
        """
        # 如果模型未初始化，现在初始化
        if self._model is None:
            kwargs = getattr(self, "_pending_kwargs", {})
            self._model = DeepSSM(obs_dim=X.shape[1], **kwargs)

        self._model.fit(X)

        # 创建实时处理器
        self._realtime_processor = self._model.create_realtime_processor()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        批量转换

        处理后会自动同步内部状态，确保后续 inference() 调用的状态连续性。

        Args:
            X: 输入数据

        Returns:
            转换后的 DataFrame，列名为 {prefix}_{i}
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # 使用 return_final_state=True 获取最终状态用于同步
        result, final_state_dict = self._model.transform(X, return_final_state=True)

        # 确保实时处理器已创建
        if self._realtime_processor is None:
            self._realtime_processor = self._model.create_realtime_processor()

        # 同步内部状态，确保后续 inference() 调用的连续性
        if final_state_dict is not None:
            self._realtime_processor.sync_state(final_state_dict)

        return pd.DataFrame(
            result,
            index=X.index,
            columns=[f"{self._prefix}_{i}" for i in range(result.shape[1])],
        )

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        训练并转换

        Args:
            X: 输入数据

        Returns:
            转换后的 DataFrame
        """
        self.fit(X)
        return self.transform(X)

    def inference(self, observation: np.ndarray) -> np.ndarray:
        """
        单步推理

        Args:
            observation: 单个观测值，形状为 (obs_dim,)

        Returns:
            当前状态，形状为 (state_dim,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self._realtime_processor is None:
            self._realtime_processor = self._model.create_realtime_processor()

        return self._realtime_processor.process_single(observation)

    def reset_state(self) -> None:
        """重置内部状态"""
        if self._realtime_processor is not None:
            self._realtime_processor.reset()

    def save(self, path: str) -> None:
        """
        保存模型

        Args:
            path: 保存路径（不含扩展名）
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self._model.save(path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DeepSSMAdapter":
        """
        加载模型

        Args:
            path: 模型路径（不含扩展名）
            device: 设备类型

        Returns:
            加载的适配器实例
        """
        adapter = cls.__new__(cls)
        adapter._model = DeepSSM.load(path, device=device)
        adapter._realtime_processor = adapter._model.create_realtime_processor()
        adapter._prefix = "deep_ssm"
        return adapter


class LGSSMAdapter:
    """
    LGSSM 适配器

    封装 LGSSM 模型，提供统一的 SSMProtocol 接口。

    状态管理说明：
    - 批量 transform 和单步 inference 使用相同的初始状态 (zeros, 0.1*I)
    - 第一个观测跳过 predict 步骤，直接进行 update
    - reset_state() 将状态重置为初始值，并标记下一个观测为第一个观测
    """

    def __init__(
        self,
        config: Optional[LGSSMConfig] = None,
        obs_dim: Optional[int] = None,
        prefix: str = "lg_ssm",
        **kwargs,
    ):
        """
        初始化 LGSSM 适配器

        Args:
            config: LGSSM 配置
            obs_dim: 观测维度（如果 config 为 None）
            prefix: 列名前缀
            **kwargs: 传递给 LGSSMConfig 的额外参数
        """
        if config is not None:
            self._model = LGSSM(config)
        elif obs_dim is not None:
            _config = LGSSMConfig(obs_dim=obs_dim, **kwargs)
            self._model = LGSSM(_config)
        else:
            # 延迟初始化
            self._model = None
            self._pending_kwargs = kwargs

        self._prefix = prefix

        # 实时推理状态
        self._state: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._first_observation: bool = True

    @property
    def is_fitted(self) -> bool:
        """模型是否已训练"""
        return self._model is not None and self._model.is_fitted

    @property
    def state_dim(self) -> int:
        """状态维度"""
        if self._model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")
        return self._model.config.state_dim

    @property
    def obs_dim(self) -> int:
        """观测维度"""
        if self._model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")
        return self._model.config.obs_dim

    @property
    def prefix(self) -> str:
        """列名前缀"""
        return self._prefix

    def fit(self, X: pd.DataFrame) -> "LGSSMAdapter":
        """
        训练模型

        Args:
            X: 输入数据，形状为 (n_samples, obs_dim)

        Returns:
            self
        """
        # 如果模型未初始化，现在初始化
        if self._model is None:
            kwargs = getattr(self, "_pending_kwargs", {})
            _config = LGSSMConfig(obs_dim=X.shape[1], **kwargs)
            self._model = LGSSM(_config)

        self._model.fit(X)

        # 初始化实时推理状态
        self.reset_state()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        批量转换

        处理后会自动同步内部状态，确保后续 inference() 调用的状态连续性。

        Args:
            X: 输入数据

        Returns:
            转换后的 DataFrame，列名为 {prefix}_{i}
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # 使用 return_final_state=True 获取最终状态用于同步
        result, final_state, final_covariance = self._model.transform(
            X, return_final_state=True
        )

        # 同步内部状态，确保后续 inference() 调用的连续性
        self._state = final_state
        self._covariance = final_covariance
        self._first_observation = False  # 已处理过数据，不再是首次观测

        return pd.DataFrame(
            result,
            index=X.index,
            columns=[f"{self._prefix}_{i}" for i in range(result.shape[1])],
        )

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        训练并转换

        Args:
            X: 输入数据

        Returns:
            转换后的 DataFrame
        """
        self.fit(X)
        return self.transform(X)

    def inference(self, observation: np.ndarray) -> np.ndarray:
        """
        单步推理

        自动维护内部状态，每次调用更新状态并返回新状态。

        Args:
            observation: 单个观测值，形状为 (obs_dim,)

        Returns:
            当前状态，形状为 (state_dim,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # 确保状态已初始化
        if self._state is None:
            self.reset_state()

        # 更新状态
        self._state, self._covariance = self._model.update_single(
            observation,
            self._state,
            self._covariance,
            is_first_observation=self._first_observation,
        )

        # 第一个观测后，标记为非首次
        if self._first_observation:
            self._first_observation = False

        return self._state

    def reset_state(self) -> None:
        """
        重置内部状态

        将状态和协方差重置为初始值 (zeros, 0.1*I)，
        并将下一个观测标记为第一个观测。
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call fit() first.")

        self._state, self._covariance = self._model.get_initial_state()
        self._first_observation = True

    def save(self, path: str) -> None:
        """
        保存模型

        Args:
            path: 保存路径（不含扩展名）
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        self._model.save(path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LGSSMAdapter":
        """
        加载模型

        Args:
            path: 模型路径（不含扩展名）
            device: 设备类型

        Returns:
            加载的适配器实例
        """
        adapter = cls.__new__(cls)
        adapter._model = LGSSM.load(path, device=device)
        adapter._prefix = "lg_ssm"

        # 初始化实时推理状态
        adapter._state, adapter._covariance = adapter._model.get_initial_state()
        adapter._first_observation = True

        return adapter
