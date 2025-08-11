# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在处理本仓库代码时提供指导。

## 项目概述

jesse-trade 是一个基于 Jesse 框架构建的复杂量化交易系统，旨在通过快速迭代和将先进的数学/物理概念应用于交易，稳定盈利地管理百万美元级别的投资组合。

## 核心原则

1. **算法正确性优先**：任何计算错误都可能造成重大财务损失。在优化可读性或性能之前，始终确保算法计算的正确性。
2. **专业代码质量**：提交前审查所有代码。遵循最佳实践并维持生产就绪标准。
3. **科学方法**：将先进的数学和物理概念应用于交易。团队由优秀的数学家和物理学家组成。

## 架构

## 开发命令

```bash
# 安装生产环境依赖
chmod +x ./install.sh
./install.sh

# 在生产依赖上添加开发依赖
pip install -r requirements-dev.txt
```

## 编码指南

### Python 最佳实践

1. **内部函数**：为内部函数/模块使用下划线前缀以减少认知负荷
2. **算法优化**：优化算法效率（如使用 Numba 优化）时必须进行单元测试以确保算法计算的正确性
3. **指标实现**：遵循 `custom_indicators/prod_indicator/accumulated_swing_index.py` 中的模式：
   - 正确处理 `sequential` 参数
   - 当 `sequential=True` 时返回所有值
   - 当 `sequential=False` 时仅返回最新值

### 其他语言重构至Python

- **EasyLanguage**：EasyLanguage 使用角度，Python 使用弧度。始终使用 `custom_indicators/utils/math.py` 进行三角函数转换

## 项目特定模式

### 自定义K线

项目通常使用由符号回归（gplearn/DEAP）挖掘的自定义K线来替代通常的时间轴k线

### 指标开发

创建新指标时：
1. 始终处理边缘情况并验证输入
2. 尽可能使用向量化操作