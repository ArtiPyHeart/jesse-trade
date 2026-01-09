"""
简化的特征注册中心

核心设计：
1. 注册时即固化所有参数
2. 统一的调用签名：(candles, sequential) -> np.ndarray
3. 简单的name -> callable映射
"""

from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Any
import numpy as np


class SimpleFeatureRegistry:
    """简化的特征注册中心"""

    def __init__(self):
        # 只存储 name -> callable 的映射
        self._features: Dict[str, Callable[[np.ndarray, bool], np.ndarray]] = {}
        # 存储特征的元信息（可选）
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register_function(
        self,
        name: str,
        func: Callable,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
        returns_multiple: bool = False,
    ) -> None:
        """
        注册函数型特征

        Args:
            name: 特征名称
            func: 特征计算函数
            params: 要固化的参数
            description: 特征描述
            returns_multiple: 是否返回多列
        """
        if params:
            # 使用partial固化参数
            wrapped_func = partial(func, **params)
        else:
            wrapped_func = func

        self._features[name] = wrapped_func
        self._metadata[name] = {
            "description": description,
            "returns_multiple": returns_multiple,
            "type": "function",
        }

    def register_class(
        self,
        name: str,
        cls: type,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
        returns_multiple: bool = False,
    ) -> None:
        """
        注册类型特征

        Args:
            name: 特征名称
            cls: 特征类
            params: 要固化的参数
            description: 特征描述
            returns_multiple: 是否返回多列
        """

        def class_wrapper(
            candles: np.ndarray, sequential: bool = True, return_raw: bool = False
        ) -> np.ndarray:
            """将类包装成函数

            Args:
                candles: K线数据
                sequential: 是否返回序列
                return_raw: 是否返回raw_result（用于需要转换链处理的情况）
            """
            # 创建实例
            if params:
                instance = cls(candles, sequential=sequential, **params)
            else:
                instance = cls(candles, sequential=sequential)

            # 如果需要raw_result（用于转换链处理）
            if return_raw and hasattr(instance, "raw_result"):
                return instance.raw_result

            # 获取结果
            if hasattr(instance, "res"):
                return instance.res()
            elif hasattr(instance, "result"):
                return instance.result()
            elif hasattr(instance, "get"):
                return instance.get()
            else:
                raise ValueError(
                    f"Class feature '{name}' doesn't have a result method "
                    f"(tried: res, result, get)"
                )

        self._features[name] = class_wrapper
        self._metadata[name] = {
            "description": description,
            "returns_multiple": returns_multiple,
            "type": "class",
        }

    def register_stateful_class(
        self,
        name: str,
        cls: type,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
        returns_multiple: bool = False,
    ) -> None:
        """
        注册有状态特征类

        有状态特征需要训练阶段，模型状态会被持久化。

        Args:
            name: 特征名称
            cls: 特征类（必须继承 StatefulFeatureBase）
            params: 要固化的参数
            description: 特征描述
            returns_multiple: 是否返回多列
        """

        def stateful_class_wrapper(
            candles: np.ndarray,
            sequential: bool = True,
            cache_dir: Optional[Path] = None,
            return_raw: bool = False,
        ) -> np.ndarray:
            """将有状态类包装成函数

            Args:
                candles: K线数据
                sequential: 是否返回序列
                cache_dir: 缓存目录（由 Calculator 注入）
                return_raw: 是否返回 raw_result（用于转换链处理）
            """
            # 创建实例
            if params:
                instance = cls(candles, sequential=sequential, **params)
            else:
                instance = cls(candles, sequential=sequential)

            # 注入缓存目录和特征名
            instance.cache_dir = cache_dir
            instance._feature_name = name

            # 如果需要 raw_result（用于转换链处理）
            if return_raw and hasattr(instance, "raw_result"):
                return instance.raw_result

            # 调用 compute（自动处理缓存）
            return instance.compute(candles, sequential)

        self._features[name] = stateful_class_wrapper
        self._metadata[name] = {
            "description": description,
            "returns_multiple": returns_multiple,
            "type": "stateful",
        }

    def get(self, name: str) -> Optional[Callable]:
        """获取特征计算函数"""
        return self._features.get(name)

    def has_feature(self, name: str) -> bool:
        """检查特征是否已注册"""
        return name in self._features

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """获取特征元信息"""
        return self._metadata.get(name)

    def list_features(self) -> Dict[str, Dict[str, Any]]:
        """列出所有已注册的特征"""
        return {name: self._metadata.get(name, {}) for name in self._features.keys()}

    def clear(self) -> None:
        """清空所有注册的特征"""
        self._features.clear()
        self._metadata.clear()


# 全局注册中心实例
_global_registry = SimpleFeatureRegistry()


def get_global_registry() -> SimpleFeatureRegistry:
    """获取全局注册中心"""
    return _global_registry


# 装饰器：用于简化特征注册
def feature(
    name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    description: str = "",
    returns_multiple: bool = False,
):
    """
    装饰器：注册函数型特征

    使用示例:
        @feature(name="rsi_14", params={"period": 14})
        def calculate_rsi(candles, sequential=True, period=14):
            return ta.rsi(candles, period=period, sequential=sequential)
    """

    def decorator(func: Callable) -> Callable:
        feature_name = name or func.__name__
        _global_registry.register_function(
            name=feature_name,
            func=func,
            params=params,
            description=description or func.__doc__ or "",
            returns_multiple=returns_multiple,
        )
        return func

    return decorator


def class_feature(
    name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    description: str = "",
    returns_multiple: bool = False,
):
    """
    装饰器：注册类型特征

    使用示例:
        @class_feature(name="vmd", params={"alpha": 2000}, returns_multiple=True)
        class VMD:
            def __init__(self, candles, sequential=True, alpha=2000):
                ...
            def res(self):
                return self.result_array
    """

    def decorator(cls: type) -> type:
        feature_name = name or cls.__name__.lower()
        _global_registry.register_class(
            name=feature_name,
            cls=cls,
            params=params,
            description=description or cls.__doc__ or "",
            returns_multiple=returns_multiple,
        )
        return cls

    return decorator


def stateful_feature(
    name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    description: str = "",
    returns_multiple: bool = False,
):
    """
    装饰器：注册有状态特征

    有状态特征需要训练阶段，模型状态会被持久化到磁盘。

    使用示例:
        @stateful_feature(name="my_ssm", returns_multiple=True)
        class MySSMFeature(StatefulFeatureBase):
            def __init__(self, candles, sequential=True, hidden_dim=64):
                super().__init__(candles, sequential)
                self.hidden_dim = hidden_dim

            def get_params(self):
                return {"hidden_dim": self.hidden_dim}

            def get_version(self):
                return "1.0.0"

            def train(self, candles):
                ...

            def inference(self, candles, sequential):
                ...

            def get_state_dict(self):
                return {"weights": self.model.weights}

            def set_state_dict(self, state_dict):
                self.model.weights = state_dict["weights"]
    """

    def decorator(cls: type) -> type:
        feature_name = name or cls.__name__.lower()
        _global_registry.register_stateful_class(
            name=feature_name,
            cls=cls,
            params=params,
            description=description or cls.__doc__ or "",
            returns_multiple=returns_multiple,
        )
        return cls

    return decorator
