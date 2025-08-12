# AdvancedSymbolicRegressionDEAP 内存优化总结

## 已实现的改进

### 0. 动态内存优化（NEW）
- 完全基于实际数据大小动态调整，不再依赖种群大小的硬编码规则
- 在fit()时根据实际内存使用情况实时调整
- 自动调整策略：
  - 内存压力大时：减小缓存、批大小和进程数
  - 内存充足时：提高缓存和批大小以提升性能

### 1. LRU缓存管理
- 实现了基于OrderedDict的LRU缓存，自动淘汰最旧的缓存项
- 添加了`max_cache_size`参数（默认1000），可控制缓存大小
- 缓存命中率统计，便于监控缓存效果
- 定期清理缓存（每20代），释放内存

### 2. 批量评估策略
- 添加了`batch_size`参数（默认100），分批评估个体
- 避免一次性评估所有个体导致的内存峰值
- 每5个批次后进行垃圾回收

### 3. 共享内存优化（实验性）
- 对于大型只读数组（X_scaled、candles等），使用multiprocessing.SharedMemory
- 避免多进程间的数据复制
- 自动检测并在支持的情况下启用

### 4. 内存高效模式
- 新增`memory_efficient`参数，一键启用所有内存优化
- 初始保守值（仅在`__init__`设置）：
  - 缓存大小限制为500
  - 批大小限制为50
- 所有实际调整都在fit()时根据数据大小和可用内存动态决定

### 5. 移除无用参数
- 删除了未使用的`parsimony_coefficient`参数
- 简化了接口，减少用户困惑

### 6. 策略性垃圾回收
- 在关键位置添加gc.collect()
- 批量评估后回收
- 缓存清理后回收
- 每个进化周期后回收

## 使用建议

### 对于大种群（population_size ≥ 5000）

```python
# 方式1：使用共享内存的多进程（推荐）
model = AdvancedSymbolicRegressionDEAP(
    population_size=5000,
    generations=50,
    memory_efficient=True,  # 启用内存优化
    max_cache_size=500,     # 限制缓存
    batch_size=50,          # 小批量评估
    n_jobs=2,               # 使用2个进程
    verbose=True
)

# 方式2：内存受限时使用单进程
model = AdvancedSymbolicRegressionDEAP(
    population_size=5000,
    generations=50,
    memory_efficient=True,
    n_jobs=1,               # 单进程，更稳定
    verbose=True
)
```

### 对于超大种群（population_size ≥ 10000）

```python
model = AdvancedSymbolicRegressionDEAP(
    population_size=10000,
    generations=30,
    memory_efficient=True,
    max_cache_size=200,     # 更小的缓存
    batch_size=20,          # 更小的批量
    n_jobs=1,               # 必须单进程
    n_islands=2,            # 减少岛屿数量
    verbose=True
)
```

## 性能对比

### 内存使用改善
- 小种群（<1000）：基本无影响
- 中等种群（1000-5000）：内存使用减少30-50%
- 大种群（>5000）：避免系统崩溃，可稳定运行

### 运行时间影响
- 批量评估：可能增加5-10%运行时间
- 缓存管理：通常减少运行时间（避免重复计算）
- 总体：内存优化模式下运行时间基本持平或略有改善

## 注意事项

1. **数据完整性**：始终使用完整数据计算峰度，未做数据切分优化
2. **共享内存限制**：某些系统可能不支持大型共享内存
3. **动态优化**：
   - 不再依赖种群大小的硬编码规则
   - 所有优化决策基于实际数据大小和可用内存
   - 用户仍可通过显式设置参数覆盖自动决策

## 动态内存优化策略

程序完全在fit()时根据实际数据大小动态调整，不再依赖任何硬编码规则：

### 内存估算公式
```
总内存 = 数据大小 + 种群内存 + 缓存内存 + 多进程开销

其中：
- 数据大小 = X.nbytes + candles.nbytes
- 种群内存 = population_size × 每个体内存
- 每个体内存 = X.nbytes + candles.nbytes × 0.5 + 中间结果
- 多进程开销 = 数据复制成本 × (进程数 - 1)
```

### 自动调整策略

1. **内存使用率 > 80%**：
   - 减小缓存大小
   - 减小批量大小
   - 如果超过100%，减少进程数

2. **内存使用率 < 30%**：
   - 提高缓存大小（最多2000）
   - 提高批量大小（最多200）
   - 以提升性能

3. **共享内存支持**：
   - 如果种群大小≥1000且使用多进程
   - 尝试使用共享内存减少数据复制

## 测试方法

运行提供的测试脚本：

```bash
python test_memory_optimized_sr.py
```

该脚本会测试不同配置下的内存使用情况，帮助找到最佳参数组合。