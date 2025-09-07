"""
简化的特征注册中心

核心设计：
1. 注册时即固化所有参数
2. 统一的调用签名：(candles, sequential) -> np.ndarray
3. 简单的name -> callable映射
"""

from functools import partial
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
        returns_multiple: bool = False
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
            "type": "function"
        }
    
    def register_class(
        self,
        name: str,
        cls: type,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
        returns_multiple: bool = False
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
        def class_wrapper(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
            """将类包装成函数"""
            # 创建实例
            if params:
                instance = cls(candles, sequential=sequential, **params)
            else:
                instance = cls(candles, sequential=sequential)
            
            # 获取结果
            if hasattr(instance, 'res'):
                return instance.res()
            elif hasattr(instance, 'result'):
                return instance.result()
            elif hasattr(instance, 'get'):
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
            "type": "class"
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
        return {
            name: self._metadata.get(name, {})
            for name in self._features.keys()
        }
    
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
    returns_multiple: bool = False
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
            returns_multiple=returns_multiple
        )
        return func
    return decorator


def class_feature(
    name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    description: str = "",
    returns_multiple: bool = False
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
            returns_multiple=returns_multiple
        )
        return cls
    return decorator