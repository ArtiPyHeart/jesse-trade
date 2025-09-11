"""
DeepSSM: Deep State Space Model implemented in JAX/NumPyro
用于金融时间序列的深度状态空间模型
使用PyTorch风格初始化确保与原版一致性
"""

from .model import DeepSSM, create_model, init_model_params
from .kalman_filter import deep_ssm_kalman_filter
from .training import train_deep_ssm, save_model, load_model, save_model_npz, load_model_npz
from .inference import DeepSSMRealTime, create_realtime_processor
from .weight_sync import (
    sync_pytorch_to_jax_deepsm,
    compare_model_outputs,
    create_matched_models
)
from .pytorch_init import (
    PyTorchLSTMCell,
    pytorch_lstm_init,
    pytorch_linear_init,
    pytorch_zeros_init
)

__all__ = [
    # 核心模型
    'DeepSSM',
    'create_model',
    'init_model_params',
    
    # 功能模块
    'deep_ssm_kalman_filter', 
    'train_deep_ssm',
    'save_model',
    'load_model',
    'save_model_npz',
    'load_model_npz',
    'DeepSSMRealTime',
    'create_realtime_processor',
    
    # 权重同步工具
    'sync_pytorch_to_jax_deepsm',
    'compare_model_outputs',
    'create_matched_models',
    
    # PyTorch风格初始化器
    'PyTorchLSTMCell',
    'pytorch_lstm_init',
    'pytorch_linear_init',
    'pytorch_zeros_init',
]