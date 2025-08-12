# FlexibleFeatureCalculator

灵活、可扩展的特征计算器，用于量化交易特征工程。相比原始的 `FeatureCalculator`，提供更好的模块化设计和更强大的转换功能。

## 核心特性

### 1. 灵活的特征注册系统
- 支持函数型和类型特征
- 动态注册新特征
- 参数化特征定义
- 特征别名支持

### 2. 强大的转换管道
- 链式转换操作
- 内置常用转换器：dt, ddt, lag, mean, std, skew, kurt等
- 支持复杂组合：mean23_lag1, std10_dt_lag5等
- 可自定义转换器

### 3. 智能Sequential处理
- 自动检测转换需求：当请求带转换的特征时，即使用户设置sequential=False，系统也会内部使用sequential=True获取完整序列以进行转换
- 确保转换正确性：转换操作（如dt、mean等）需要完整的时间序列数据
- 最终输出符合用户期望：转换完成后，如果用户要求sequential=False，只返回最终值

### 4. 多列特征智能缓存
- **一次计算，多次使用**：对于`returns_multiple=True`的特征，只计算一次完整结果
- **自动缓存所有列**：第一次请求任意列时，自动缓存所有列（如`multi_0`到`multi_4`）
- **高效访问**：后续请求其他列时直接从缓存返回，无需重复计算
- **大幅提升性能**：特别适合计算成本高的多列特征（如VMD、CWT等）

### 5. 向后兼容
- 与原 FeatureCalculator API 完全兼容
- 相同的 sequential 行为
- 计算结果完全一致

### 5. 更好的可观测性
- 列出所有注册特征
- 查看特征参数和描述
- 缓存机制提升性能

## 快速开始

### 基础使用

```python
from src.features.flexible_feature_calculator import FlexibleFeatureCalculator
import numpy as np

# 创建计算器实例
calculator = FlexibleFeatureCalculator()

# 加载K线数据
candles = np.load("data/btc_1m.npy")
calculator.load(candles, sequential=True)

# 获取特征
features = calculator.get([
    "rsi",                  # 基础RSI
    "rsi_dt",              # RSI的一阶差分
    "rsi_mean20_lag5",     # RSI的20期均值后滞后5期
])
```

### 注册自定义特征

#### 函数型特征

```python
from src.features.flexible_feature_calculator import feature

# 特征名称可以包含任意参数信息
@feature(name="rsi_14_smoothed", description="Smoothed RSI with period 14")
def rsi_14_smoothed(candles: np.ndarray, sequential: bool = True, **kwargs):
    # 参数已经在名称中体现，不需要额外解析
    return calculate_smoothed_rsi(candles, period=14)

@feature(name="ma_20_ema_50_cross", description="MA20 and EMA50 crossover")
def ma_cross(candles: np.ndarray, sequential: bool = True, **kwargs):
    # 复杂的命名模式完全支持
    return calculate_ma_cross(candles, ma_period=20, ema_period=50)

# 使用
calculator.get(["rsi_14_smoothed"])
calculator.get(["rsi_14_smoothed_dt"])  # 自动支持转换
calculator.get(["ma_20_ema_50_cross_lag5"])  # 转换链正常工作
```

**重要**：特征名称的灵活性
- 特征名称可以包含任意格式：`indicator_v2_optimized_2024`、`strategy_param1_param2_param3`
- 系统不会尝试解析或猜测参数含义
- 只有尾部的转换器（dt、lag、mean等）会被特殊处理
- 参数的实际值在注册时通过函数实现定义

#### 类型特征

```python
from src.features.flexible_feature_calculator import class_feature

@class_feature(name="complex_indicator", params={"window": 32}, returns_multiple=True)
class ComplexIndicator:
    def __init__(self, candles: np.ndarray, window: int = 32, sequential: bool = False, **kwargs):
        self.candles = candles
        self.window = window
        # 初始化计算
    
    def res(self):
        # 返回多列结果
        return np.zeros((len(self.candles), 5))

# 使用多列返回
calculator.get(["complex_indicator_0"])  # 第一列
calculator.get(["complex_indicator_1"])  # 第二列
```

### 动态注册特征

```python
# 动态注册函数特征
calculator.register_feature(
    name="dynamic_feature",
    func=lambda candles, **kwargs: np.ones(len(candles)),
    params={"param1": 100},
    description="Dynamically registered feature",
    aliases=["df", "dyn"]
)

# 通过别名访问
result = calculator.get(["df"])
```

### 复杂转换链

```python
# 支持的转换器
transformations = [
    "feature_dt",              # 一阶差分
    "feature_ddt",             # 二阶差分
    "feature_lag5",            # 滞后5期
    "feature_mean20",          # 20期移动平均
    "feature_std10",           # 10期标准差
    "feature_skew15",          # 15期偏度
    "feature_kurt20",          # 20期峰度
    "feature_sum30",           # 30期求和
    "feature_max10",           # 10期最大值
    "feature_min10",           # 10期最小值
    "feature_median15",        # 15期中位数
    "feature_pct5",            # 5期百分比变化
    "feature_rank20",          # 20期排名
    "feature_zscore30",        # 30期Z分数
]

# 组合转换
complex_features = [
    "rsi_mean20_dt_lag3",      # RSI → 20期均值 → 一阶差分 → 滞后3期
    "adx_14_std10_zscore20",   # ADX(14) → 10期标准差 → 20期Z分数
]
```

### 自定义转换器

```python
# 注册自定义转换器
def my_transform(array: np.ndarray, param: int) -> np.ndarray:
    # 自定义转换逻辑
    return transformed_array

calculator.register_transformer("mytrans", my_transform)

# 使用自定义转换器
calculator.get(["feature_mytrans10"])  # 应用参数为10的自定义转换
```

## 特征列表管理

```python
# 列出所有已注册的特征
all_features = calculator.list_features()
for name, info in all_features.items():
    print(f"{name}: {info['description']}")
    print(f"  Type: {info['type']}")
    print(f"  Params: {info['params']}")
    print(f"  Aliases: {info['aliases']}")

# 仅列出函数型特征
function_features = calculator.list_features(feature_type="function")

# 仅列出类型特征
class_features = calculator.list_features(feature_type="class")
```

## 性能优化

### 缓存机制

```python
# 特征计算结果会自动缓存
calculator.get(["rsi"])  # 第一次计算
calculator.get(["rsi"])  # 使用缓存，不重复计算

# 清空缓存
calculator.clear_cache()
```

### 多列特征优化

```python
# 对于返回多列的特征，系统会智能缓存所有列
# 例如，VMD特征返回5列数据

# 第一次请求任意一列时，计算并缓存所有列
result0 = calculator.get(["vmd_0"])  # 计算完整VMD，缓存所有5列

# 后续请求其他列直接从缓存返回，无需重新计算
result1 = calculator.get(["vmd_1"])  # 从缓存返回，不重新计算
result2 = calculator.get(["vmd_2"])  # 从缓存返回，不重新计算
result3 = calculator.get(["vmd_3"])  # 从缓存返回，不重新计算
result4 = calculator.get(["vmd_4"])  # 从缓存返回，不重新计算

# 批量请求也只计算一次
results = calculator.get(["vmd_0", "vmd_1", "vmd_2", "vmd_3", "vmd_4"])
# 只计算一次，自动缓存所有列
```

### 批量计算

```python
# 批量获取特征（推荐）
features = calculator.get([
    "rsi", "adx_14", "fisher", 
    "rsi_dt", "adx_14_lag5"
])

# 而不是多次调用
feature1 = calculator.get(["rsi"])
feature2 = calculator.get(["adx_14"])
# ...
```

## 与原 FeatureCalculator 的对比

| 特性 | FeatureCalculator | FlexibleFeatureCalculator |
|-----|------------------|--------------------------|
| 特征注册 | 硬编码在类中 | 动态注册系统 |
| 转换支持 | dt, ddt, lag | 14种内置转换器，可扩展 |
| 组合转换 | 不支持 | 支持任意组合 |
| 特征列表 | 无法查看 | list_features() |
| 扩展性 | 需修改源码 | 装饰器/动态注册 |
| 性能 | 缓存机制 | 相同的缓存机制 |
| API兼容性 | - | 100%兼容 |

## 迁移指南

从 FeatureCalculator 迁移非常简单：

```python
# 原代码
from src.features.all_features import FeatureCalculator

calc = FeatureCalculator()
calc.load(candles, sequential=True)
features = calc.get(["rsi", "rsi_dt"])

# 新代码（仅需更改导入）
from src.features.flexible_feature_calculator import FlexibleFeatureCalculator
from src.features.flexible_feature_calculator.features import builtin  # 导入内置特征

calc = FlexibleFeatureCalculator()
calc.load(candles, sequential=True)
features = calc.get(["rsi", "rsi_dt"])  # 完全相同的使用方式
```

## 扩展示例

### 添加新的技术指标族

```python
# indicators_extension.py
from src.features.flexible_feature_calculator import feature

# 批量注册Bollinger Bands相关特征
@feature(name="bb_upper", params={"period": 20, "std": 2})
def bb_upper(candles, period=20, std=2, sequential=True, **kwargs):
    # Bollinger Bands上轨
    pass

@feature(name="bb_lower", params={"period": 20, "std": 2})
def bb_lower(candles, period=20, std=2, sequential=True, **kwargs):
    # Bollinger Bands下轨
    pass

@feature(name="bb_width", params={"period": 20, "std": 2})
def bb_width(candles, period=20, std=2, sequential=True, **kwargs):
    # Bollinger Bands宽度
    pass
```

### 创建特征工厂

```python
class FeatureFactory:
    """特征工厂，批量创建相似特征"""
    
    @staticmethod
    def create_ma_features(periods=[5, 10, 20, 50, 100, 200]):
        """创建多个周期的移动平均特征"""
        for period in periods:
            @feature(name=f"ma_{period}", params={"period": period})
            def ma_feature(candles, period=period, sequential=True, **kwargs):
                return ta.sma(candles, period, sequential=sequential)
    
    @staticmethod
    def create_volatility_features():
        """创建波动率相关特征族"""
        # ATR不同周期
        for period in [7, 14, 21]:
            @feature(name=f"atr_{period}", params={"period": period})
            def atr_feature(candles, period=period, sequential=True, **kwargs):
                return ta.atr(candles, period, sequential=sequential)

# 初始化特征
FeatureFactory.create_ma_features()
FeatureFactory.create_volatility_features()
```

## 最佳实践

1. **模块化组织特征**：将相关特征组织在独立模块中
2. **使用装饰器注册**：比动态注册更清晰
3. **提供默认参数**：让特征更易用
4. **编写特征描述**：帮助其他人理解特征含义
5. **批量获取特征**：减少重复计算
6. **合理使用缓存**：在数据更新时清空缓存

## 常见问题

**Q: 如何处理特征计算中的NaN值？**
A: 与原FeatureCalculator保持一致，在sequential=True时使用np.nan填充至K线长度。

**Q: 类型特征如何处理多列返回？**
A: 使用 `feature_name_0`, `feature_name_1` 等访问各列。

**Q: 转换器的执行顺序是什么？**
A: 从左到右依次执行，如 `feature_mean20_dt_lag5` 执行顺序为：mean20 → dt → lag5。

**Q: Sequential参数如何影响带转换的特征？**
A: 当请求带转换的特征（如 `rsi_dt`）时，即使用户设置了 `sequential=False`，系统会：
1. 内部使用 `sequential=True` 调用基础特征以获取完整序列
2. 对完整序列应用转换操作
3. 如果用户要求 `sequential=False`，最终只返回转换后的最后一个值

这确保了转换操作的正确性，因为像 dt（差分）、mean（均值）等操作需要完整的时间序列数据。

**Q: 如何调试特征计算？**
A: 可以分步获取中间结果：
```python
base = calculator.get(["feature"])
after_mean = calculator.get(["feature_mean20"])
after_dt = calculator.get(["feature_mean20_dt"])
final = calculator.get(["feature_mean20_dt_lag5"])
```

## 贡献指南

欢迎贡献新的特征和转换器！请遵循以下规范：

1. 特征命名使用小写下划线
2. 提供清晰的文档字符串
3. 包含单元测试
4. 确保与原FeatureCalculator兼容（如适用）

## 许可证

与主项目相同。