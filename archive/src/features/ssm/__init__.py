"""
SSM Module - 状态空间模型模块

提供 DeepSSM 和 LGSSM 的统一接口适配器。

Example:
    >>> from src.features.ssm import DeepSSMAdapter, LGSSMAdapter, SSMProtocol
    >>>
    >>> # 使用 DeepSSM
    >>> deep_ssm = DeepSSMAdapter(obs_dim=10)
    >>> deep_ssm.fit(train_data)
    >>> features = deep_ssm.transform(test_data)
    >>>
    >>> # 实时推理
    >>> for obs in observations:
    ...     state = deep_ssm.inference(obs)
"""

from .protocol import SSMProtocol
from .adapters import DeepSSMAdapter, LGSSMAdapter

__all__ = [
    "SSMProtocol",
    "DeepSSMAdapter",
    "LGSSMAdapter",
]
