# Linear Gaussian State Space Model (LGSSM)

线性高斯状态空间模型的生产级实现，用于时间序列特征提取。

## 特性

- 原生PyTorch实现（无Pyro依赖）
- 标准卡尔曼滤波与RTS平滑
- 精确边际似然优化（log p(y|θ)）
- 实时推理支持（含NaN容忍）
- 安全的模型保存/加载（SafeTensors格式）
- 可选的数据标准化
- 数值稳定性优化（A矩阵谱半径约束、Q/R方差边界、协方差对称化）

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
states = model.transform(data)
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

# 结果与 model.transform(observations) 完全一致
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
# 保存模型（SafeTensors格式）
model.save("lgssm_model")

# 加载模型
loaded_model = LGSSM.load("lgssm_model")
```

## 模型原理

LGSSM 是一个线性高斯状态空间模型：

```
状态转移: z_{t+1} = A @ z_t + w_t,  w_t ~ N(0, Q)
观测模型: y_t = C @ z_t + v_t,      v_t ~ N(0, R)
初始状态: z_0 ~ N(0, I)
```

其中：
- `A`: 状态转移矩阵
- `C`: 观测矩阵
- `Q`: 过程噪声协方差（对角矩阵）
- `R`: 观测噪声协方差（对角矩阵）

## 训练算法

模型通过最大化边际似然 log p(y|θ) 训练：

```
log p(y|θ) = Σ_t log p(y_t | y_{1:t-1}, θ)
```

这是由 Kalman 滤波器精确计算的创新似然（innovation likelihood）。
对于 LGSSM，这等价于真正的 ELBO（因为 RTS 后验就是精确后验，KL 散度为 0）。

训练流程：
1. **Kalman 滤波**：前向传播计算滤波状态和创新似然
2. **参数更新**：梯度下降 + A矩阵稳定性投影

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `state_dim` | 5 | 潜在状态维度 |
| `learning_rate` | 0.01 | 学习率 |
| `max_epochs` | 100 | 最大训练轮数 |
| `patience` | 10 | 早停耐心值 |
| `use_scaler` | True | 是否使用StandardScaler |
| `A_init_scale` | 0.95 | A矩阵初始化尺度 |
| `gradient_clip` | 1.0 | 梯度裁剪阈值 |
| `A_spectral_max` | 0.999 | A矩阵最大谱半径（稳定性约束） |
| `Q_log_min` | -10.0 | Q对数下界（exp(-10) ≈ 4.5e-5） |
| `Q_log_max` | 10.0 | Q对数上界（exp(10) ≈ 22026） |
| `R_log_min` | -10.0 | R对数下界 |
| `R_log_max` | 10.0 | R对数上界 |

## 数值稳定性

本实现包含多项数值稳定性保护：

- **A 矩阵**：每步优化后投影到谱半径 < 0.999（确保系统稳定）
- **Q/R 方差**：对数参数化 + 上下界 [exp(-10), exp(10)]
- **矩阵运算**：Jitter = 1e-6 保护 solve/slogdet 操作
- **Scaler**：使用有偏估计 + 最小值 1e-8 防止常量特征
- **协方差对称化**：RTS 平滑后强制协方差矩阵对称

## 缺失值处理

- **训练时**：要求完整数据，NaN 会触发 `ValueError`
- **推理时**：自动跳过 NaN 观测，使用预测值代替（Kalman 滤波标准做法）

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
├── kalman_filter.py     # 卡尔曼滤波器 + RTS平滑
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
states = model.transform(df)

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

本实现保证了批处理（`transform`）和在线推理（`update_single`）的完全一致性：

1. **相同的初始化**：使用 `get_initial_state()` 获取与批处理相同的初始状态（z0=0, P0=I）
2. **相同的算法流程**：第一个观测使用 `is_first_observation=True` 参数
3. **相同的归一化**：两种方法使用相同的数据预处理

```python
# 验证一致性
batch_states = model.transform(data)
sequential_states = []
state, cov = model.get_initial_state()

for i, obs in enumerate(data):
    state, cov = model.update_single(obs, state, cov, is_first_observation=(i==0))
    sequential_states.append(state)

# batch_states 与 sequential_states 完全相同
```

## 版本说明

v2.1.0（当前）:
- 改用精确边际似然 log p(y|θ) 训练（等价于 ELBO，代码更简洁）
- 删除所有 ELBO 相关代码
- 训练阶段 NaN 检查（fail-fast）
- RTS 平滑协方差对称化

v2.0.0:
- 使用 RTS 平滑的 ELBO（包含 lag-one covariance）
- 添加 A 矩阵谱半径稳定性约束
- Q/R 方差边界保护
- SafeTensors 格式保存

## 相关资源

- [Kalman Filter Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)
- [State Space Models](https://en.wikipedia.org/wiki/State-space_representation)
- Shumway & Stoffer, "Time Series Analysis and Its Applications", Chapter 6
