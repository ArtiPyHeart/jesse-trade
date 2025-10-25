# Ripser 持久同调算法

基于 Rust 实现的高性能持久同调算法，用于拓扑数据分析（TDA）。

## 特性

- ✅ **完全正确**：与参考实现 giotto-ph 完美匹配（< 1e-7 差异）
- ⚡ **高性能**：20-50 点 < 10ms，100 点 ~120ms
- 🔧 **易用**：简单的 Python API
- 📊 **功能完整**：支持 H_0, H_1, H_2 同调计算
- 🎯 **生产就绪**：经过全面测试和验证（10/10 测试通过）

## 快速开始

### 基本使用

```python
import numpy as np
from pyrs_indicators.topology import ripser

# 创建点云数据
points = np.array([
    [0, 0],
    [1, 0],
    [0.5, 0.866]  # 等边三角形
])

# 计算持久同调
result = ripser(points, max_dim=1, threshold=2.0)

# 结果
print(f"H_0 pairs: {result['persistence'][0]}")  # 连通分量
print(f"H_1 pairs: {result['persistence'][1]}")  # 环/循环
```

### 参数说明

```python
ripser(
    data,                    # 输入数据：点云 或 距离矩阵
    max_dim=1,              # 最大维度：0, 1, 或 2
    threshold=None,         # 距离阈值（None = 无限制）
    metric='euclidean',     # 距离度量：'euclidean', 'manhattan', 'chebyshev'
    distance_matrix=False,  # data 是否为距离矩阵
    collapse_edges=True     # 是否过滤零长度 pairs
)
```

**返回值**：
```python
{
    'persistence': [
        np.array([[birth, death], ...]),  # H_0 持久性对
        np.array([[birth, death], ...]),  # H_1 持久性对
        ...
    ]
}
```

### 实际应用示例

#### 1. 检测循环/环

```python
import numpy as np
from pyrs_indicators.topology import ripser

# 在圆上采样点
n_points = 20
theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
circle = np.column_stack([np.cos(theta), np.sin(theta)])

# 计算持久同调
result = ripser(circle, max_dim=1, threshold=np.inf)

# 分析结果
h1_pairs = result['persistence'][1]
if h1_pairs.shape[0] > 0:
    persistence = h1_pairs[:, 1] - h1_pairs[:, 0]
    max_idx = np.argmax(persistence)
    print(f"发现主要循环：birth={h1_pairs[max_idx, 0]:.3f}, "
          f"death={h1_pairs[max_idx, 1]:.3f}")
```

#### 2. 时间序列拓扑特征

```python
from pyrs_indicators.topology import ripser
import numpy as np

# 滑动窗口嵌入
def embed_timeseries(signal, window=20, delay=1):
    """时间延迟嵌入"""
    n = len(signal) - (window-1) * delay
    embedded = np.zeros((n, window))
    for i in range(n):
        embedded[i] = signal[i::delay][:window]
    return embedded

# 计算拓扑特征
signal = np.sin(2*np.pi*np.linspace(0, 10, 200))  # 示例信号
embedded = embed_timeseries(signal, window=20)
result = ripser(embedded, max_dim=1, threshold=2.0)

# 提取特征
n_components = result['persistence'][0].shape[0]  # 连通分量数量
n_loops = result['persistence'][1].shape[0]       # 循环数量
```

#### 3. 使用预计算距离矩阵

```python
from scipy.spatial.distance import pdist, squareform
from pyrs_indicators.topology import ripser

# 计算距离矩阵
points = np.random.randn(50, 3)
dist_compressed = pdist(points, metric='euclidean')

# 使用距离矩阵
result = ripser(
    dist_compressed,
    max_dim=1,
    threshold=2.0,
    distance_matrix=True  # 重要：标记为距离矩阵
)
```

### 辅助函数

#### 过滤持久性对

```python
from pyrs_indicators.topology import ripser, filter_persistence

result = ripser(points, max_dim=1)

# 只保留持久性 > 0.1 的特征
filtered = filter_persistence(result, min_persistence=0.1)
```

#### 计算 Betti 数

```python
from pyrs_indicators.topology import ripser, get_betti_numbers

result = ripser(points, max_dim=2)

# 获取 Betti 数（拓扑特征的数量）
betti = get_betti_numbers(result)
print(f"B_0 (连通分量): {betti[0]}")
print(f"B_1 (环): {betti[1]}")
print(f"B_2 (空洞): {betti[2]}")
```

## 性能基准

**随机 2D 点云**（threshold=2.0, MacBook Apple Silicon）：

| 点数 | H_1 时间 | H_0 pairs | H_1 pairs |
|------|---------|-----------|-----------|
| 10   | < 1ms   | 10        | ~2        |
| 20   | < 1ms   | 20        | ~5        |
| 50   | 9ms     | 50        | ~25       |
| 100  | 118ms   | 100       | ~100      |

**典型应用场景**：
- 时间序列分析（20-50 点滑动窗口）：< 10ms ✅
- 实时交易策略：完全满足需求 ✅

## 算法说明

### 持久同调

持久同调是拓扑数据分析的核心工具，用于：
- 发现数据中的拓扑特征（连通分量、环、空洞）
- 量化这些特征的"重要性"（持久性）
- 过滤噪声，保留显著特征

### 维度说明

- **H_0**：连通分量数量
  - 应用：聚类检测、离群点识别
- **H_1**：环/循环数量
  - 应用：周期性检测、循环模式识别
- **H_2**：空洞/空腔数量
  - 应用：3D结构分析、高维流形检测

### 持久性对格式

每个持久性对 `[birth, death]` 表示：
- **birth**：特征出现的尺度
- **death**：特征消失的尺度
- **persistence** = death - birth：特征的持久性（重要性）

较大的 persistence 值表示更显著的特征。

## 验证与测试

本实现已通过以下验证：
- ✅ **数值一致性**：与 giotto-ph ripser_parallel 完美匹配
- ✅ **几何测试**：Circle, Two Circles, Triangle 等全部通过
- ✅ **集成测试**：10/10 测试通过
- ✅ **Codex 验证**：经过 GPT-5 深度代码分析

详见：
- `tests/test_ripser_integration.py` - 集成测试
- `src/ripser/ITERATION_8_NUMERICAL_CONSISTENCY.md` - 数值验证
- `src/ripser/CODEX_COLLABORATION_SUMMARY.md` - Codex 协助总结

## 注意事项

1. **阈值选择**：
   - `threshold=None`：计算完整的持久同调（可能很慢）
   - 合理的阈值可以显著提升性能
   - 建议从数据的最大距离的 50-80% 开始

2. **维度限制**：
   - `max_dim=2` 时间复杂度较高（O(n³)）
   - 对于大数据集（>100点），建议 `max_dim=1`

3. **内存使用**：
   - Simplex 数量随点数指数增长
   - 200 点约需 12,700 simplices
   - 建议监控内存使用

4. **零长度过滤**：
   - `collapse_edges=True`（默认）：遵循标准 Ripser 实践
   - `collapse_edges=False`：保留所有 pairs（包括 birth == death）

## 参考文献

- Ripser: Efficient computation of Vietoris-Rips persistence barcodes (Bauer, 2021)
- giotto-ph: Python bindings for Ripser
- 本实现：完全 Rust 重写，修复了 3 个 critical bugs

## 许可证

与 rust_indicators 项目相同

## 贡献

发现问题或有改进建议？欢迎提 Issue 或 PR！
