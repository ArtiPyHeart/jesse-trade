"""
简化的特征计算器包

提供更简单、更健壮的特征选择和计算功能
"""

from .calculator import SimpleFeatureCalculator
from .registry import (
    SimpleFeatureRegistry,
    get_global_registry,
    feature,
    class_feature
)
from .validator import FeatureOutputValidator
from .transforms import (
    dt,
    ddt,
    lag,
    rolling_mean,
    rolling_std,
    rolling_max,
    rolling_min,
    TransformChain
)

__all__ = [
    # 核心组件
    'SimpleFeatureCalculator',
    'SimpleFeatureRegistry',
    'FeatureOutputValidator',
    'TransformChain',
    
    # 注册相关
    'get_global_registry',
    'feature',
    'class_feature',
    
    # 转换函数
    'dt',
    'ddt',
    'lag',
    'rolling_mean',
    'rolling_std',
    'rolling_max',
    'rolling_min',
]