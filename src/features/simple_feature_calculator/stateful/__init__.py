"""
有状态特征模块

提供有状态特征的基类和缓存管理器，支持需要训练的特征（如神经网络）。
"""

from .base import StatefulFeatureBase
from .cache_manager import StatefulCacheManager

__all__ = ["StatefulFeatureBase", "StatefulCacheManager"]
