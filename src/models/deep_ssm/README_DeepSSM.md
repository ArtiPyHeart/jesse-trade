# DeepSSM (Deep State Space Model)

深度状态空间模型，结合LSTM和扩展卡尔曼滤波进行时间序列特征提取。

## 特性

- 结合深度学习与状态空间建模
- 使用LSTM提取时序特征
- 扩展卡尔曼滤波进行状态估计
- 支持实时流式处理
- 可控的随机种子实现可重复性

## 快速开始

### 基本使用

```python
from src.models import DeepSSM, DeepSSMConfig
import numpy as np

# 准备数据
data = np.random.randn(1000, 10)  # [时间步, 特征数]

# 创建配置
config = DeepSSMConfig(
    obs_dim=10,        # 输入特征维度
    state_dim=5,       # 潜在状态维度
    lstm_hidden=64,    # LSTM隐层维度
    max_epochs=50,     # 最大训练轮数
    seed=42            # 随机种子
)

# 训练模型
model = DeepSSM(config)
model.fit(data)

# 提取特征
features = model.transform(data)
print(f"Features shape: {features.shape}")  # (1000, 5)
```

### 保存和加载

```python
# 保存模型
model.save("deep_ssm_model.pt")

# 加载模型
loaded_model = DeepSSM.load("deep_ssm_model.pt")
```

### 实时处理

```python
# 创建实时处理器
realtime = model.create_realtime_processor()

# 处理单个观测
for new_observation in data_stream:
    feature = realtime.process_single(new_observation)
    # 使用特征进行下游任务
```

## 配置参数

### 模型结构
- `obs_dim`: 观测维度（必需）
- `state_dim`: 潜在状态维度 (默认: 5)
- `lstm_hidden`: LSTM隐层大小 (默认: 64)
- `lstm_layers`: LSTM层数 (默认: 1)

### 训练参数
- `learning_rate`: 学习率 (默认: 1e-3)
- `max_epochs`: 最大训练轮数 (默认: 100)
- `batch_size`: 批大小 (默认: 32)
- `patience`: 早停耐心值 (默认: 10)
- `min_delta`: 损失改善最小阈值 (默认: 1e-4)

### 正则化
- `dropout`: Dropout概率 (默认: 0.1)
- `weight_decay`: 权重衰减 (默认: 1e-5)
- `gradient_clip`: 梯度裁剪 (默认: 1.0)

### 数据处理
- `use_scaler`: 是否使用StandardScaler标准化输入 (默认: True)

### 其他
- `device`: 计算设备 ('auto', 'cpu', 'cuda')
- `seed`: 随机种子（用于可重复性）

## 与原始版本的区别

1. **独立的StandardScaler**: 标准化参数与模型分离存储
2. **PyTorch原生实现**: 不依赖Pyro，使用PyTorch原生优化器
3. **改进的架构**: 添加LayerNorm和Dropout提高泛化能力
4. **生产级功能**: 完整的错误处理、类型注解和文档
5. **灵活的推理**: 支持批处理和流式处理

## 应用场景

- 时间序列特征提取
- 金融数据降维
- 状态估计和滤波
- 异常检测前处理
- 预测模型的特征工程

## 注意事项

1. 输入数据应为浮点型NumPy数组或PyTorch张量
2. 默认会自动进行标准化处理，可通过`use_scaler=False`禁用
3. 对于长序列，考虑使用滑动窗口
4. 实时处理器维护内部状态，处理新序列前需调用`reset()`
5. 如果数据已预处理或不需要标准化，建议设置`use_scaler=False`

## 示例：金融数据特征提取

```python
import pandas as pd
from src.models import DeepSSM, DeepSSMConfig

# 加载金融数据
df = pd.read_csv("price_data.csv")
features = df[['open', 'high', 'low', 'close', 'volume']].values

# 配置针对金融数据优化
config = DeepSSMConfig(
    obs_dim=5,
    state_dim=3,  # 提取3个潜在因子
    lstm_hidden=32,
    max_epochs=100,
    learning_rate=5e-4,
    seed=42
)

# 训练和提取特征
model = DeepSSM(config)
model.fit(features[:800], val_data=features[800:900])
latent_features = model.transform(features)

# 保存供策略使用
model.save("financial_deep_ssm.pt")
pd.DataFrame(latent_features).to_csv("latent_features.csv", index=False)
```

## 示例：使用预标准化数据

```python
from src.models import DeepSSM, DeepSSMConfig
import numpy as np

# 假设数据已经过预处理（如差分、标准化等）
preprocessed_data = np.load("preprocessed_features.npy")

# 禁用内置的StandardScaler
config = DeepSSMConfig(
    obs_dim=preprocessed_data.shape[1],
    state_dim=5,
    use_scaler=False,  # 不使用内置标准化
    max_epochs=100,
    seed=42
)

model = DeepSSM(config)
model.fit(preprocessed_data)
features = model.transform(preprocessed_data)
```