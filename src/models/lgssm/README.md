# Linear Gaussian State Space Model (LGSSM)

线性高斯状态空间模型的生产级实现，用于时间序列特征提取。

## 特性

- ✅ 原生PyTorch实现（无Pyro依赖）
- ✅ 标准卡尔曼滤波与RTS平滑
- ✅ 变分推断（ELBO优化）
- ✅ 实时推理支持
- ✅ 安全的模型保存/加载（`weights_only=True`）
- ✅ 可选的数据标准化
- ✅ 数值稳定性优化

## 安装

该模块已包含在 jesse-trade 项目中，无需额外安装。

## 快速开始

### 基本用法

```python
import numpy as np
from src.models.lgssm import LGSSM, LGSSMConfig

# 生成示例数据
data = np.random.randn(1000, 10)  # 1000个时间步，10维特征

# 创建配置
config = LGSSMConfig(
    state_dim=5,      # 潜在状态维度
    max_epochs=50,    # 最大训练轮数
    learning_rate=0.01,
    use_scaler=True   # 启用数据标准化
)

# 创建并训练模型
model = LGSSM(config)
model.fit(data[:800], data[800:], verbose=True)  # 80/20 训练/验证分割

# 生成状态特征
states = model.predict(data)
print(f"状态特征形状: {states.shape}")  # (1000, 5)
```

### 实时推理（保证与批处理一致）

```python
# 获取与批处理一致的初始状态
initial_state, initial_covariance = model.get_initial_state()

# 处理观测序列
current_state = initial_state
current_covariance = initial_covariance
states_list = []

for i, observation in enumerate(observations):
    # 第一个观测需要特殊处理
    current_state, current_covariance = model.update_single(
        observation,
        current_state,
        current_covariance,
        is_first_observation=(i == 0)  # 重要：第一个观测标记
    )
    states_list.append(current_state)

# 结果与 model.predict(observations) 完全一致
```

#### 简化用法（自动初始化）

```python
# 不提供初始状态时，自动使用默认初始化
state, covariance = model.update_single(first_observation, is_first_observation=True)

# 后续观测
state, covariance = model.update_single(next_observation, state, covariance)
```

### 模型保存与加载

```python
# 保存模型
model.save("lgssm_model.pt")

# 加载模型
loaded_model = LGSSM.load("lgssm_model.pt")
```

## 模型原理

LGSSM 是一个线性高斯状态空间模型：

```
状态转移: z_{t+1} = A @ z_t + w_t,  w_t ~ N(0, Q)
观测模型: y_t = C @ z_t + v_t,      v_t ~ N(0, R)
```

其中：
- `A`: 状态转移矩阵
- `C`: 观测矩阵
- `Q`: 过程噪声协方差
- `R`: 观测噪声协方差

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `state_dim` | 5 | 潜在状态维度 |
| `learning_rate` | 0.01 | 学习率 |
| `max_epochs` | 50 | 最大训练轮数 |
| `patience` | 10 | 早停耐心值 |
| `use_scaler` | True | 是否使用StandardScaler |
| `A_init_scale` | 0.95 | A矩阵初始化尺度 |
| `gradient_clip` | 1.0 | 梯度裁剪阈值 |

## 与DeepSSM的比较

| 特性 | LGSSM | DeepSSM |
|------|-------|---------|
| 模型类型 | 线性 | 非线性（LSTM） |
| 卡尔曼滤波 | 标准 | 扩展 |
| 参数数量 | 少（4个矩阵） | 多（神经网络） |
| 计算效率 | 高 | 中等 |
| 适用场景 | 线性系统 | 复杂非线性系统 |

## 文件结构

```
lgssm/
├── __init__.py           # 模块导出
├── lgssm.py             # 主模型实现
├── kalman_filter.py     # 卡尔曼滤波器
├── test_lgssm.py        # 单元测试
└── README.md            # 本文档
```

## 测试

运行测试套件：

```bash
cd src/models/lgssm
python test_lgssm.py
```

## 示例：金融时间序列特征提取

```python
import pandas as pd
from src.models.lgssm import LGSSM, LGSSMConfig

# 加载金融数据
df = pd.read_csv("financial_features.csv")

# 配置模型
config = LGSSMConfig(
    state_dim=5,          # 5个潜在因子
    max_epochs=100,
    learning_rate=0.01,
    patience=15,
    use_scaler=True,      # 标准化处理
    seed=42               # 可重现性
)

# 训练模型
model = LGSSM(config)
model.fit(df, verbose=True)

# 提取状态特征
states = model.predict(df)

# 保存特征
feature_df = pd.DataFrame(
    states,
    columns=[f"lgssm_feature_{i}" for i in range(5)]
)
feature_df.to_csv("lgssm_features.csv", index=False)
```

## 注意事项

1. **数据预处理**：如果数据已经过差分处理，可以设置 `use_scaler=False`
2. **状态维度选择**：通常5-10维足够捕捉主要动态
3. **数值稳定性**：模型使用对数尺度参数化确保协方差矩阵正定
4. **实时推理**：保存最后的状态和协方差用于增量更新

## 批处理与在线推理的一致性

本实现保证了批处理（`predict`）和在线推理（`update_single`）的完全一致性：

1. **相同的初始化**：使用 `get_initial_state()` 获取与批处理相同的初始状态
2. **相同的算法流程**：第一个观测使用 `is_first_observation=True` 参数
3. **相同的归一化**：两种方法使用相同的数据预处理

```python
# 验证一致性
batch_states = model.predict(data)
sequential_states = []
state, cov = model.get_initial_state()

for i, obs in enumerate(data):
    state, cov = model.update_single(obs, state, cov, is_first_observation=(i==0))
    sequential_states.append(state)

# batch_states 与 sequential_states 完全相同
```

## 相关资源

- [Kalman Filter Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)
- [State Space Models](https://en.wikipedia.org/wiki/State-space_representation)
- [Variational Inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods)