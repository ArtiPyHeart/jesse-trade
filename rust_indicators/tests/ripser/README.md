# Ripser 测试套件

这个目录包含 Ripser Rust 实现的所有测试。

## 测试策略

每个迭代都遵循以下测试流程：

1. **参考数据生成**: 使用 `giotto-ph` 生成标准参考数据
2. **数值验证**: 对比 Rust 实现与参考数据的差异
3. **性能基准**: 测量性能提升（vs giotto-ph）
4. **边界测试**: 测试边界条件和异常情况

## 测试文件

### 迭代 1: 二项式系数表
- `test_binomial.py`: 二项式系数计算验证

### 迭代 2: 距离矩阵
- `test_distance_matrix.py`: 稠密/稀疏距离矩阵验证

### 迭代 3-4: 简单复形
- `test_simplex.py`: 简单复形编解码、边枚举验证

### 迭代 5: 上同调计算
- `test_cohomology.py`: 上同调算法验证

### 迭代 6: 端到端集成
- `test_integration.py`: 完整流程验证

## 参考数据

`reference_data/` 目录存储 giotto-ph 生成的参考数据：

- `simple_circle.pkl`: 简单圆形点云
- `noisy_torus.pkl`: 带噪声环面
- `timeseries_1d.pkl`: 1D 时间序列（实际使用场景）

## 运行测试

```bash
# 运行所有测试
pytest tests/ripser/

# 运行特定测试
pytest tests/ripser/test_binomial.py

# 显示详细输出
pytest tests/ripser/ -v

# 显示打印输出
pytest tests/ripser/ -s
```

## 验收标准

- ✅ 所有测试通过率 100%
- ✅ 数值误差 < 1e-6
- ✅ 性能提升 > 2x（vs giotto-ph）
