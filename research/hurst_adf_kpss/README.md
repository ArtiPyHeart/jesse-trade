# Hurst-ADF-KPSS 三重检验系统

适用于Jesse框架的K线趋势性三重检验分析工具，通过Hurst指数、ADF检验和KPSS检验的组合，科学判断价格序列的趋势特征。

**注意**: 本模块位于`research/`目录下，仅供线下研究分析使用，不包含`sequential`模式，不用于生产环境。

## 理论基础

### 三个核心指标

1. **Hurst指数** - 衡量时间序列的长期记忆性和趋势持续性
   - H < 0.5: 反持续性（均值回归）
   - H = 0.5: 随机游走（无趋势）
   - H > 0.5: 持续性（趋势）
   - H > 0.6: 强趋势潜力

2. **ADF检验** - 检验序列是否存在单位根（非平稳性）
   - p值 > 0.05: 非平稳（存在单位根）
   - p值 ≤ 0.05: 平稳（不存在单位根）

3. **KPSS检验** - 检验序列是否以平稳为零假设（与ADF互补）
   - p值 < 0.05: 非平稳（拒绝平稳假设）
   - p值 ≥ 0.05: 平稳（接受平稳假设）

### 趋势类型判定

![趋势分析准则](趋势分析.png)

根据三个指标的组合，系统会自动判定以下趋势类型：

- **强趋势且非平稳（适合趋势策略）**: H>0.55, ADF_p>0.05, KPSS_p<0.05
- **趋势但平稳（短期趋势可能）**: H>0.55, ADF_p<0.05, KPSS_p>0.05
- **震荡平稳（不适合趋势策略）**: H<0.5, ADF_p<0.05, KPSS_p>0.05
- **矛盾（需进一步验证）**: H>0.55, ADF_p>0.05, KPSS_p>0.05
- **弱趋势或反趋势**: 其他情况

## 快速开始

### 1. 基本使用 - 快速趋势检查

```python
import numpy as np
from research.hurst_adf_kpss.trend_analyzer import quick_trend_check

# 加载Jesse风格K线数据
# K线格式: [timestamp, open, close, high, low, volume]
candles = np.load("data/merged_candles.npy")

# 快速检查最近60根K线的趋势
hurst, trend_type, score = quick_trend_check(candles, window=60)

print(f"Hurst指数: {hurst:.4f}")
print(f"趋势类型: {trend_type}")
print(f"趋势评分: {score}/5")

# 根据评分给出策略建议
if score >= 4:
    print("建议: 适合使用趋势跟踪策略")
elif score >= 3:
    print("建议: 部分时段可使用趋势策略")
else:
    print("建议: 不适合趋势策略，考虑震荡策略")
```

### 2. 详细分析 - 单窗口检验

```python
from research.hurst_adf_kpss.trend_analyzer import TrendAnalyzer

# 创建分析器
analyzer = TrendAnalyzer(candles)

# 分析指定窗口（如最后100根K线）
result = analyzer.analyze_window(
    start_idx=len(candles)-100,
    end_idx=len(candles)-1
)

# 查看详细结果
print(f"Hurst指数: {result['hurst']:.4f}")
print(f"ADF p值: {result['adf']['pvalue']:.4f}")
print(f"KPSS p值: {result['kpss']['pvalue']:.4f}")
print(f"趋势评分: {result['trend_score']}/5")
print(f"趋势类型: {result['trend_type']}")
print(f"使用的滞后参数: min_lag={result['min_lag']}, max_lag={result['max_lag']}")
```

### 3. 滑动窗口分析

```python
# 对多个窗口大小进行滑动分析
analyzer = TrendAnalyzer(candles)

# 分析20、40、60三个窗口，步长为10
results = analyzer.sliding_window_analysis(
    windows=[20, 40, 60],
    step=10
)

# 查看每个窗口的分析结果
for window_size, window_results in results.items():
    print(f"\n窗口大小 {window_size}:")
    print(f"  分析窗口数: {len(window_results)}")
    for i, res in enumerate(window_results[:3]):  # 显示前3个
        print(f"  窗口{i+1}: {res['trend_type']}")
```

## 高级功能

### 1. 获取统计表格

```python
# 进行滑动窗口分析后
results = analyzer.sliding_window_analysis(
    windows=[20, 40, 60],
    step=5
)

# 获取格式化的统计表格
stats_df = analyzer.get_statistics_table()
print(stats_df)

# 输出格式：
#    窗口大小  总窗口数  平均分  中位数  最高分占比  低分占比  评分标准差
# 0      60      189   2.98    3.0     26.98    46.03      1.48
# 1      40      193   2.73    3.0     13.99    44.56      1.26
# 2      20      197   2.33    2.0     10.66    55.33      1.33
```

### 2. 打印统计摘要

```python
# 打印格式化的统计摘要
analyzer.print_statistics_summary()

# 可选：保存到.md文件
analyzer.print_statistics_summary(save_to_file="statistics_summary.md")
```

输出示例：

**控制台输出（Markdown格式）：**
```markdown
# 各窗口趋势适配性评分统计

| 窗口大小 | 总窗口数 | 平均分 | 中位数 | 最高分占比 | 低分占比 | 评分标准差 |
|:--------:|:--------:|:------:|:------:|:----------:|:--------:|:----------:|
| 60 | 189 | 2.98 | 3.0 | 26.98% | 46.03% | 1.48 |
| 40 | 193 | 2.73 | 3.0 | 13.99% | 44.56% | 1.26 |
| 20 | 197 | 2.33 | 2.0 | 10.66% | 55.33% | 1.33 |

## 分析总结

- **最佳窗口大小**: 60 （平均评分 2.98）
- **最差窗口大小**: 20 （平均评分 2.33）
- **最稳定窗口**: 40 （标准差 1.26）
- **高趋势占比最高**: 60 （5分占比 27.0%）
```

保存到文件时，会自动保存为`.md`格式，并包含详细数据表格。

### 3. 导出为DataFrame

```python
# 获取完整的分析结果DataFrame
df_results = analyzer.get_results_dataframe()

# DataFrame包含以下列：
# - 窗口大小, 窗口编号, 起始索引, 结束索引
# - Hurst指数, ADF统计量, ADF p值, ADF滞后期
# - KPSS统计量, KPSS p值, KPSS警告
# - 趋势适配性评分, 趋势类型
# - min_lag, 实际max_lag
# - 起始时间戳, 结束时间戳

# 保存为CSV
df_results.to_csv("trend_analysis_results.csv", index=False)

# 筛选高评分窗口
high_score_windows = df_results[df_results['趋势适配性评分'] >= 4]
print(f"高评分窗口数: {len(high_score_windows)}")
```

### 4. 自定义滞后参数

```python
# 自定义Hurst计算的滞后参数
analyzer = TrendAnalyzer(candles)

# 使用自定义函数计算滞后参数
results = analyzer.sliding_window_analysis(
    windows=[50, 100],
    step=10,
    min_lag_func=lambda w: max(10, w // 5),  # 最小滞后为窗口的1/5
    max_lag_func=lambda w: min(50, w // 2)   # 最大滞后为窗口的1/2，但不超过50
)
```

## 趋势评分系统

系统使用0-5分的评分系统，评分规则如下：

- Hurst > 0.6: +2分
- Hurst > 0.55: +1分
- ADF p值 > 0.05（非平稳）: +1分
- KPSS p值 < 0.05（拒绝平稳）: +1分
- 三重确认奖励（同时满足上述三个条件）: +1分

评分解读：
- **5分**: 强趋势特征，非常适合趋势策略
- **4分**: 较强趋势特征，适合趋势策略
- **3分**: 中等趋势特征，部分时段适合趋势策略
- **2分**: 弱趋势特征，谨慎使用趋势策略
- **0-1分**: 无明显趋势，不适合趋势策略

## 注意事项

1. **数据要求**
   - 输入必须是Jesse风格的numpy array，shape=(n, 6)
   - 列顺序：[timestamp, open, close, high, low, volume]
   - 至少需要20个数据点进行有效分析

2. **窗口大小选择**
   - 小窗口（20-30）: 捕捉短期趋势，但噪音较大
   - 中窗口（40-60）: 平衡短期和中期趋势
   - 大窗口（80-100+）: 识别长期趋势，但反应较慢

3. **KPSS警告处理**
   - 系统会自动捕获KPSS的插值警告
   - 警告信息会显示在趋势类型描述中
   - "p值可能更小"：实际p值可能小于显示值
   - "p值可能更大"：实际p值可能大于显示值

4. **性能优化**
   - 大数据集建议增大step参数减少计算量
   - 滑动窗口分析会对每个窗口进行三重检验，计算密集
   - 可以先用快速检查函数初步判断

## 实际应用示例

### 策略集成示例

```python
def should_use_trend_strategy(candles, window=60, min_score=4):
    """
    判断当前是否应该使用趋势策略
    
    Parameters:
    -----------
    candles: Jesse K线数据
    window: 分析窗口大小
    min_score: 最低趋势评分要求
    
    Returns:
    --------
    bool: 是否使用趋势策略
    """
    hurst, trend_type, score = quick_trend_check(candles, window)
    
    if score >= min_score:
        print(f"趋势策略激活: {trend_type}")
        return True
    else:
        print(f"趋势不足: {trend_type}")
        return False

# 在Jesse策略中使用
if should_use_trend_strategy(self.candles):
    # 执行趋势跟踪逻辑
    pass
else:
    # 执行震荡或其他策略
    pass
```

### 多时间框架分析

```python
def multi_timeframe_trend_analysis(candles):
    """
    多时间框架趋势一致性分析
    """
    windows = [30, 60, 120]  # 短期、中期、长期
    scores = []
    
    for window in windows:
        if len(candles) >= window:
            _, _, score = quick_trend_check(candles, window)
            scores.append(score)
    
    # 判断趋势一致性
    if all(s >= 4 for s in scores):
        return "强趋势一致"
    elif all(s >= 3 for s in scores):
        return "中等趋势一致"
    elif any(s >= 4 for s in scores):
        return "部分时间框架有趋势"
    else:
        return "无明显趋势"
```

## 测试文件

- `test_trend_analyzer.py` - 基础功能测试
- `test_enhanced_features.py` - 增强功能测试
- `test_statistics_table.py` - 统计表格功能测试
- `test_improved_output.py` - 输出格式测试

## 作者与维护

此工具是jesse-trade量化交易系统的一部分，专门针对Jesse框架的K线格式优化。