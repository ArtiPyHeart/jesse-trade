# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在处理本仓库代码时提供指导。

## 项目概述

jesse-trade 是一个基于 Jesse 框架构建的复杂量化交易系统，旨在通过快速迭代和将先进的数学/物理概念应用于交易，稳定盈利地管理百万美元级别的投资组合。

## 核心原则

1. **算法正确性优先**：任何计算错误都可能造成重大财务损失。在优化可读性或性能之前，始终确保算法计算的正确性。
2. **专业代码质量**：提交前审查所有代码。遵循最佳实践并维持生产就绪标准。
3. **科学方法**：将先进的数学和物理概念应用于交易。团队由优秀的数学家和物理学家组成。

## 项目结构

### 模块职责

- **src/**: 包含所有核心功能代码，可在生产和研究环境中使用
- **research/**: 研究专用代码，包括实验性算法和分析工具，仅用于线下离线研究
- **strategies/**: Jesse框架策略实现，每个策略独立目录
- **extern/**: 第三方代码、教材资料、参考实现等

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
3. **单元测试**：简单单元测试直接在当前文件下使用if __name__ == "__main__":的方式直接运行，复杂测试则在`tests/`文件夹下独立构建
4. **指标实现**：src/中的指标需遵循 `src/indicators/prod/accumulated_swing_index.py` 中的模式（research/中的模块不用）：
   - 正确处理 `sequential` 参数
   - 当 `sequential=True` 时返回所有值，通常用于计算所有指标值用于模型训练
   - 当 `sequential=False` 时仅返回最新值，通常用于线上生产环境返回最新值用于模型预测
5. **数据密集型编程**: 对于所有数据密集型文件（npy/csv/excel/parquet等），统一使用numpy与pandas进行操作
6. **暴露原始错误**: 尽量暴露代码的原始错误，除非需要提供给用户更高的可读性。使用assert拦截非法输入，尽量不要使用try except来掩盖错误
7. **单元测试准则**: 功能实现后必须构建并运行单元测试。成功的单元测试在测试成功后可以直接删除。

### 其他语言重构至Python

- **EasyLanguage**：EasyLanguage 使用角度，Python 使用弧度。始终使用 `src/utils/math_tools.py` 进行三角函数转换

## 项目特定模式

### 自定义K线

无论是原始时间轴k线还是自定义轴k线，都需要遵循Jesse风格的k线形式：
- 二维numpy array，6列数据，列含义分别为timestamp, open, close, high, low, volume
- 可以使用以下方式将jesse numpy candle转化为dataframe：
```python
import numpy as np
from jesse.utils import numpy_candles_to_dataframe
candles = np.load("data/btc_1m.npy")
df_candles = numpy_candles_to_dataframe(candles)
```
dataframe版本K线同样有6列：date, open, close, high, low, volume，index与date列相同，均为pandas datetime格式

项目通常使用由符号回归（gplearn/DEAP）挖掘的自定义K线来替代通常的时间轴K线：
- **融合算法**：`src/bars/fusion/` 包含DEAP和其他符号回归实现
- **传统K线**：`src/bars/traditional/` 包含Dollar Bar、Entropy Bar、Range Bar等
- **调优流程**：使用 `tuning_pipeline.py` 和 `research/optuna_config.py` 进行参数优化

### 指标开发

创建新指标时：
1. 始终处理边缘情况并验证输入
2. 尽可能使用向量化操作
3. 可用的生产指标放在 `src/indicators/prod/`，对于新实现的指标，如果相对简单（单文件），可直接放入prod，如果是一大类新的指标，则在`src/indicators/`下创建新的类别
4. 可靠性存疑的实验性指标放在 `src/indicators/experimental/`
6. 非函数的class类型指标需继承 `src/indicators/prod/_indicator_base/_cls_ind.py` 基类，可参考[cls_vmd_indicator.py](src/indicators/prod/emd/cls_vmd_indicator.py)中的VMD_NRBO指标的实现方式

## 注意事项

### 测试文件命名

- 对于需要使用unittest和pytest进行测试的函数和文件，使用test_开头，pycharm会自动识别并触发测试
- 但如果是直接使用python运行的文件和函数，不要使用test_开头

### 代码组织

- 不要将研究代码直接用于生产，需要经过严格测试后移至 `src/`
- 保持 `strategies/` 中每个策略的独立性，避免交叉依赖
- `extern/` 中的代码仅供参考，不要直接导入使用

### 特征长度要求

所有由K线加工出的特征在`sequential=True`时必须保持与K线numpy array长度一致（len方法一致），如果不一致（通常是因为时间序列上较早时间上有缺失），必须使用`np.nan`填充至一致长度。在特征输入模型前，会以缺失最多的特征为基准，对所有特征进行统一的nan去除