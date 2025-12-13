"""
Pipeline Module - 高阶特征流水线

提供 FeaturePipeline 整合原始特征计算、SSM 特征提取、降维模块。

Example:
    >>> from src.features.pipeline import FeaturePipeline, PipelineConfig
    >>>
    >>> # 配置
    >>> config = PipelineConfig(
    ...     raw_feature_names=["rsi", "macd"],
    ...     fracdiff_feature_names=["frac_close_diff"],
    ...     ssm_types=["deep_ssm", "lg_ssm"],
    ... )
    >>>
    >>> # 训练模式
    >>> pipeline = FeaturePipeline(config)
    >>> all_features = pipeline.fit_transform(candles)
    >>> pipeline.save("/path/to/models")
    >>>
    >>> # 加载模式
    >>> pipeline = FeaturePipeline.load("/path/to/models")
    >>> features = pipeline.transform(candles)
    >>>
    >>> # 实时模式
    >>> pipeline.load_candles(current_candles)
    >>> features = pipeline.inference()
"""

from .config import PipelineConfig
from .feature_pipeline import FeaturePipeline

__all__ = [
    "PipelineConfig",
    "FeaturePipeline",
]
