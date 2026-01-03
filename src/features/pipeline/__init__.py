"""
Pipeline Module - 特征处理流水线

二段式架构：
- FeatureMaker：特征加工（SimpleFeatureCalculator + SSM）
- Reducer：特征降维（ARDVAE）

"""

from .feature_maker import FeatureMaker
from .feature_maker_config import FeatureMakerConfig
from .reducer import Reducer
from .reducer_config import ReducerConfig

__all__ = [
    "FeatureMaker",
    "FeatureMakerConfig",
    "Reducer",
    "ReducerConfig",
]
