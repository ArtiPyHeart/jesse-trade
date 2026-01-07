"""
有状态特征基类

有状态特征与无状态特征的核心区别：
1. 需要训练阶段（train）建立模型
2. 模型状态需要持久化
3. 推理时加载已训练的模型
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class StatefulFeatureBase(ABC):
    """
    有状态特征基类

    生命周期：
    - sequential=True 且缓存不存在: train() -> save_state() -> inference(sequential=True)
    - sequential=True 且缓存存在: load_state() -> inference(sequential=True)
    - sequential=False: load_state() -> inference(sequential=False)

    子类必须实现：
    - get_params(): 返回特征参数（用于缓存校验）
    - get_version(): 返回版本号（用于版本管理）
    - train(): 训练模型
    - inference(): 推理（支持 sequential）
    - get_state_dict(): 返回可序列化的状态字典
    - set_state_dict(): 从状态字典恢复
    """

    # 由 Calculator 注入
    cache_dir: Optional[Path] = None
    _feature_name: str = ""

    def __init__(self, candles: np.ndarray, sequential: bool = True):
        """
        初始化特征

        Args:
            candles: K线数据
            sequential: 是否返回序列
        """
        self._candles = candles
        self._sequential = sequential
        self._is_trained = False

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        返回特征参数

        这些参数用于缓存校验。如果参数不匹配，将报错。

        Returns:
            参数字典，必须可 JSON 序列化
        """
        ...

    @abstractmethod
    def get_version(self) -> str:
        """
        返回版本号

        版本号变化时：
        - sequential=True: 自动重训练
        - sequential=False: 报错

        Returns:
            版本字符串，如 "1.0.0"
        """
        ...

    @abstractmethod
    def train(self, candles: np.ndarray) -> None:
        """
        训练模型

        Args:
            candles: 完整的 K 线数据
        """
        ...

    @abstractmethod
    def inference(self, candles: np.ndarray, sequential: bool) -> np.ndarray:
        """
        推理

        Args:
            candles: K 线数据
            sequential: True 返回全量序列，False 返回最后一行

        Returns:
            特征数组
        """
        ...

    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        """
        返回模型状态字典

        状态字典必须能被 safetensors 序列化，即：
        - 键为字符串
        - 值为 np.ndarray 或 torch.Tensor

        Returns:
            状态字典
        """
        ...

    @abstractmethod
    def set_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        从状态字典恢复模型

        Args:
            state_dict: 由 get_state_dict() 返回的状态字典
        """
        ...

    # ========== 基类提供的方法 ==========

    def compute(self, candles: np.ndarray, sequential: bool) -> np.ndarray:
        """
        计算特征（由 Calculator 调用）

        自动处理缓存加载/训练/保存逻辑
        """
        from .cache_manager import StatefulCacheManager

        if self.cache_dir is None:
            raise RuntimeError(
                f"cache_dir not set for stateful feature '{self._feature_name}'. "
                f"Please set stateful_cache_dir when creating SimpleFeatureCalculator."
            )

        cache_manager = StatefulCacheManager(
            cache_dir=self.cache_dir,
            feature_name=self._feature_name,
        )

        # 尝试加载缓存
        cache_status = cache_manager.check_cache(
            params=self.get_params(),
            version=self.get_version(),
        )

        if cache_status == "valid":
            # 缓存有效，直接加载
            state_dict = cache_manager.load_state()
            self.set_state_dict(state_dict)
            self._is_trained = True
            return self.inference(candles, sequential)

        elif cache_status == "not_exists":
            if not sequential:
                raise RuntimeError(
                    f"Stateful feature '{self._feature_name}' cache not found. "
                    f"Please run with sequential=True first to train the model."
                )
            # 训练并保存
            self.train(candles)
            self._is_trained = True
            cache_manager.save_state(
                state_dict=self.get_state_dict(),
                params=self.get_params(),
                version=self.get_version(),
            )
            return self.inference(candles, sequential)

        elif cache_status == "params_mismatch":
            cached_params = cache_manager.get_cached_params()
            raise RuntimeError(
                f"Stateful feature '{self._feature_name}' params mismatch. "
                f"Cached: {cached_params}, "
                f"Current: {self.get_params()}. "
                f"Please call calc.clear_stateful_cache('{self._feature_name}') to clear."
            )

        elif cache_status == "version_outdated":
            if not sequential:
                cached_version = cache_manager.get_cached_version()
                raise RuntimeError(
                    f"Stateful feature '{self._feature_name}' version outdated. "
                    f"Cached: {cached_version}, "
                    f"Current: {self.get_version()}. "
                    f"Please run with sequential=True to retrain."
                )
            # 重训练
            self.train(candles)
            self._is_trained = True
            cache_manager.save_state(
                state_dict=self.get_state_dict(),
                params=self.get_params(),
                version=self.get_version(),
            )
            return self.inference(candles, sequential)

        else:
            raise RuntimeError(f"Unknown cache status: {cache_status}")

    def res(self) -> np.ndarray:
        """
        兼容 class_feature 的接口
        """
        return self.compute(self._candles, self._sequential)

    @property
    def is_trained(self) -> bool:
        """是否已训练"""
        return self._is_trained
