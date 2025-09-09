# SimpleFeatureCalculator 技术文档

## 概述

SimpleFeatureCalculator 是一个专为量化交易设计的特征计算框架，提供了统一、灵活的特征计算和转换能力。它解决了原有特征计算系统的复杂性问题，通过清晰的架构设计实现了高效的特征工程。

## 核心设计理念

1. **不截断数据**：与 Jesse 原生系统不同，SimpleFeatureCalculator 不会截断输入数据，而是将完整数据传递给特征函数，由特征自己决定如何处理
2. **统一接口**：所有特征函数遵循统一签名 `(candles, sequential) -> np.ndarray`
3. **分离关注点**：基础特征计算、转换链、验证逻辑完全分离
4. **智能缓存**：多级缓存策略，避免重复计算

## 架构组件

### 1. Calculator (calculator.py)
主入口类，负责：
- 数据加载和管理
- 特征名解析（基础特征名 + 列索引 + 转换链）
- 缓存管理
- 协调各组件工作

**关键方法**：
- `load(candles, sequential)`: 加载K线数据，不做任何截断
- `get(features)`: 批量获取特征
- `_compute_feature()`: 核心计算逻辑
- `_parse_feature_name()`: 解析特征名（如 `vmd_win512_1_lag1` → base='vmd_win512', column=1, transforms=['lag1']）
- `_process_raw_result_for_transform()`: 专门处理类特征的 raw_result 用于转换链

### 2. Registry (registry.py)
特征注册中心，管理所有可用特征：

**两种注册方式**：
1. **函数型特征**：直接注册函数
   ```python
   @feature(name="rsi", params={"period": 14})
   def calculate_rsi(candles, sequential):
       # 计算逻辑
       return result
   ```

2. **类特征**（用于复杂窗口计算）：
   ```python
   @class_feature(name="vmd_win512", returns_multiple=True)
   class VMDFeature:
       def __init__(self, candles, sequential):
           self.indicator = VMD_NRBO(candles, window, sequential)
       
       @property
       def raw_result(self):
           # 暴露原始窗口数据供转换链使用
           return self.indicator.raw_result
       
       def res(self):
           return self.indicator.res()
   ```

**特殊处理**：
- 类特征支持 `return_raw` 参数，用于获取原始窗口数据进行转换
- 函数型特征通过 `partial` 固化参数

### 3. Transforms (transforms.py)
独立的转换函数库，直接操作 numpy 数组：

**内置转换**：
- `dt`: 一阶差分
- `ddt`: 二阶差分
- `lag`: 滞后n期
- `rolling_mean/std/max/min`: 滚动窗口统计

**TransformChain 类**：
- 管理转换链的解析和应用
- 支持参数化转换（如 `mean20`, `lag1`）
- 智能处理一维/二维数组

### 4. Validator (validator.py)
严格的输出验证，确保特征输出符合规范：

**验证规则**：
- `sequential=True`: 输出长度必须等于输入K线长度
- `sequential=False`: 单值或单行输出
- 多列特征必须返回二维数组
- 提供详细的错误信息和修复建议

### 5. IndicatorBase 集成
与 `src/indicators/prod/_indicator_base/_cls_ind.py` 的配合：

**关键设计**：
- `raw_result`: 保存所有窗口的原始数据 `List[np.ndarray]`
- `res()`: 从每个窗口提取最后一行组成时间序列
- 转换链直接操作 `raw_result` 而不是 `res()` 的结果

**数据流**：
1. `sequential=True`: 计算所有窗口 → raw_result 包含所有窗口数据
2. `sequential=False`: 只计算最后一个窗口 → raw_result 包含一个窗口
3. 转换时：获取 raw_result → 提取每个窗口的最后一行 → 应用转换

## 特征命名规范

完整特征名格式：`{base_name}_{column_index}_{transform1}_{transform2}_...`

示例：
- `rsi`: 基础RSI指标
- `rsi_dt`: RSI的一阶差分
- `vmd_win512_0`: VMD第0列
- `vmd_win512_1_lag1`: VMD第1列滞后1期
- `ac_1_mean20_dt`: AC第1列的20期均值的一阶差分

## 关键实现细节

### 1. 多列特征的处理
- **函数型多列特征**：直接返回二维数组
- **类特征多列**：通过 `raw_result` 暴露窗口数据
- **列索引提取**：在转换前处理，保持维度一致性

### 2. 缓存策略
```python
# 缓存键设计
cache_key = (feature_name, sequential)  # 基础缓存
cache_key = (f"{feature_name}_seq", True)  # 强制sequential
cache_key = (f"{feature_name}_raw", sequential)  # raw_result缓存
```

### 3. Sequential 模式的区别
- `sequential=True`: 返回完整时间序列
- `sequential=False`: 只返回最新值
- 转换链需要时会强制使用 `sequential=True` 获取完整数据

### 4. 错误处理
- 详细的验证错误信息
- 提供修复建议
- 保留原始错误栈用于调试

## 使用示例

```python
from src.features.simple_feature_calculator import SimpleFeatureCalculator

# 初始化
calc = SimpleFeatureCalculator()

# 加载数据（不会截断）
calc.load(candles, sequential=False)

# 获取特征
features = calc.get([
    'rsi',                    # 基础特征
    'vmd_win512_0',          # 多列特征的第0列
    'vmd_win512_1_lag1',     # 带转换的多列特征
    'rsi_mean20_dt'          # 链式转换
])
```

## 扩展指南

### 添加新的函数型特征
```python
@feature(name="my_feature", returns_multiple=False)
def my_feature_func(candles: np.ndarray, sequential: bool) -> np.ndarray:
    # 实现逻辑
    return result
```

### 添加新的类特征
```python
@class_feature(name="my_class_feature", returns_multiple=True)
class MyFeature:
    def __init__(self, candles, sequential, **kwargs):
        self.indicator = MyIndicator(candles, sequential=sequential)
    
    @property
    def raw_result(self):
        return self.indicator.raw_result
    
    def res(self):
        return self.indicator.res()
```

## 性能优化建议

1. **批量获取**：一次性获取多个特征，利用缓存
2. **合理使用 sequential**：非必要不使用 sequential=True
3. **转换链优化**：尽量减少转换步骤