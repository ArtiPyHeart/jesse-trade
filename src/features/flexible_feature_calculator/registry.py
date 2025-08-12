from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class FeatureSpec:
    """特征规格定义"""

    name: str
    func: Optional[Callable] = None
    cls: Optional[type] = None
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    returns_multiple: bool = False
    feature_type: str = "function"  # "function" or "class"

    def __post_init__(self):
        if self.func is None and self.cls is None:
            raise ValueError("Either func or cls must be provided")
        if self.func is not None and self.cls is not None:
            raise ValueError("Cannot provide both func and cls")

        self.feature_type = "class" if self.cls is not None else "function"


class FeatureRegistry:
    """特征注册中心"""

    def __init__(self):
        self._features: Dict[str, FeatureSpec] = {}
        self._aliases: Dict[str, str] = {}

    def register_function(
        self,
        name: str,
        func: Callable,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
        returns_multiple: bool = False,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """注册函数型特征"""
        spec = FeatureSpec(
            name=name,
            func=func,
            params=params or {},
            description=description,
            returns_multiple=returns_multiple,
            feature_type="function",
        )
        self._features[name] = spec

        # 注册别名
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

    def register_class(
        self,
        name: str,
        cls: type,
        params: Optional[Dict[str, Any]] = None,
        description: str = "",
        returns_multiple: bool = False,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """注册类型特征"""
        spec = FeatureSpec(
            name=name,
            cls=cls,
            params=params or {},
            description=description,
            returns_multiple=returns_multiple,
            feature_type="class",
        )
        self._features[name] = spec

        # 注册别名
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

    def get(self, name: str) -> Optional[FeatureSpec]:
        """获取特征规格"""
        # 先检查别名
        if name in self._aliases:
            name = self._aliases[name]
        return self._features.get(name)

    def list_features(
        self, feature_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """列出所有已注册的特征"""
        result = {}
        for name, spec in self._features.items():
            if feature_type and spec.feature_type != feature_type:
                continue

            result[name] = {
                "type": spec.feature_type,
                "params": spec.params,
                "description": spec.description,
                "returns_multiple": spec.returns_multiple,
                "aliases": [
                    alias for alias, target in self._aliases.items() if target == name
                ],
            }
        return result

    def has_feature(self, name: str) -> bool:
        """检查特征是否已注册"""
        return name in self._features or name in self._aliases

    def remove_feature(self, name: str) -> bool:
        """移除特征"""
        if name in self._aliases:
            name = self._aliases[name]

        if name in self._features:
            # 移除相关别名
            aliases_to_remove = [
                alias for alias, target in self._aliases.items() if target == name
            ]
            for alias in aliases_to_remove:
                del self._aliases[alias]

            del self._features[name]
            return True
        return False

    def clear(self) -> None:
        """清空所有注册的特征"""
        self._features.clear()
        self._aliases.clear()


def feature(
    name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    description: str = "",
    returns_multiple: bool = False,
    aliases: Optional[List[str]] = None,
    registry: Optional[FeatureRegistry] = None,
):
    """装饰器：注册函数型特征"""

    def decorator(func: Callable) -> Callable:
        feature_name = name or func.__name__
        feature_registry = registry or _global_registry

        feature_registry.register_function(
            name=feature_name,
            func=func,
            params=params,
            description=description or func.__doc__ or "",
            returns_multiple=returns_multiple,
            aliases=aliases,
        )
        return func

    return decorator


def class_feature(
    name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    description: str = "",
    returns_multiple: bool = False,
    aliases: Optional[List[str]] = None,
    registry: Optional[FeatureRegistry] = None,
):
    """装饰器：注册类型特征"""

    def decorator(cls: type) -> type:
        feature_name = name or cls.__name__.lower()
        feature_registry = registry or _global_registry

        feature_registry.register_class(
            name=feature_name,
            cls=cls,
            params=params,
            description=description or cls.__doc__ or "",
            returns_multiple=returns_multiple,
            aliases=aliases,
        )
        return cls

    return decorator


# 全局注册中心
_global_registry = FeatureRegistry()


def get_global_registry() -> FeatureRegistry:
    """获取全局注册中心"""
    return _global_registry
