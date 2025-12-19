"""
GrootCV 特征选择器 - 自建实现

基于 SHAP 值的特征选择方法，使用 shadow features 和交叉验证。
兼容 LightGBM 4.6.0 和 fasttreeshap (numpy 1.x)。
"""

from src.features.feature_selection.groot.grootcv import GrootCV

__all__ = ["GrootCV"]
