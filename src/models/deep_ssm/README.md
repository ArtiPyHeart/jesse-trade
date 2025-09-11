# DeepSSM JAX实现

## 概述
深度状态空间模型（Deep State Space Model）的JAX/Flax实现，用于金融时间序列分析。采用PyTorch风格的初始化确保与原始实现的一致性。

## 核心功能
- 结合LSTM和状态空间模型提取时序特征
- 扩展卡尔曼滤波进行状态估计
- 支持从PyTorch模型迁移权重
- 适用于金融数据的降维和特征提取

## 文件结构
```
src/models/deep_ssm/
├── model.py            # 核心模型定义
├── pytorch_init.py     # PyTorch风格初始化
├── weight_sync.py      # 权重同步工具
├── kalman_filter.py    # 卡尔曼滤波实现
├── training.py         # 训练逻辑
└── inference.py        # 实时推理
```

## 快速使用

```python
from src.models.deep_ssm import create_model, init_model_params, deep_ssm_kalman_filter
import jax.numpy as jnp
from jax import random

# 创建模型
model = create_model(obs_dim=77, state_dim=5, lstm_hidden=64)

# 初始化参数
key = random.PRNGKey(42)
dummy_input = jnp.zeros((1, 10, 77))
params = init_model_params(model, key, dummy_input)

# 生成状态特征
states, P = deep_ssm_kalman_filter(data, model, params)
```

## 权重同步

从PyTorch模型迁移：
```python
from src.models.deep_ssm import sync_pytorch_to_jax_deepsm

# 同步权重
jax_params = sync_pytorch_to_jax_deepsm(torch_model, jax_params)
```

## 依赖
- jax>=0.4.0
- flax>=0.7.0
- numpyro>=0.13.0
- numpy
- scikit-learn