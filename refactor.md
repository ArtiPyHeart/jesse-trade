# Jesse-Trade 代码库重构方案

## 1. 当前结构分析

### 现有问题
- **混杂的代码组织**：研究代码（notebooks）与生产代码混在一起
- **第三方代码管理**：`extern/` 目录包含大量第三方代码，应该明确区分
- **缺少分层**：缺少清晰的核心业务逻辑层
- **运行时文件混杂**：配置、数据、输出等运行时文件与源代码混在一起
- **模块职责不清**：部分模块功能重叠，边界不清晰

### 现有优势
- `strategies/` 和 `custom_indicators/` 已有基本组织
- 使用了 Numba 等性能优化工具
- 有明确的 CLAUDE.md 项目说明

## 2. 推荐的目录结构

```
jesse-trade/
├── src/                          # 核心源代码
│   ├── core/                     # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── config/               # 配置管理
│   │   │   ├── __init__.py
│   │   │   ├── settings.py      # 全局设置
│   │   │   └── optuna_config.py # Optuna配置（从根目录移入）
│   │   ├── data/                 # 数据处理层
│   │   │   ├── __init__.py
│   │   │   ├── loaders/         # 数据加载器
│   │   │   ├── processors/      # 数据预处理
│   │   │   └── storage/         # 数据存储接口
│   │   └── utils/                # 通用工具
│   │       ├── __init__.py
│   │       ├── math_tools.py    # 从 custom_indicators/utils/ 移入
│   │       ├── import_tools.py  # 从 custom_indicators/utils/ 移入
│   │       └── logging.py       # 新建日志工具
│   │
│   ├── bars/                     # 自定义K线模块（原bar/）
│   │   ├── __init__.py
│   │   ├── builders/             # K线构建器
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # 从 bar/build.py 重构
│   │   │   ├── dollar_bar.py    # 从 bar/dollar_bar.py 移入
│   │   │   ├── entropy_bar.py   # 从 bar/entropy_bar_v2.py 移入
│   │   │   └── range_bar.py     # 从 bar/range_bar.py 移入
│   │   ├── fusion/               # 融合算法（保持原结构）
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── deap_v1.py
│   │   │   └── v0.py
│   │   └── tuning/               # 参数调优
│   │       ├── __init__.py
│   │       ├── optimizers/      # 优化器
│   │       └── pipeline.py      # 从 tuning_pipeline.py 移入
│   │
│   ├── indicators/               # 指标库（原custom_indicators/）
│   │   ├── __init__.py
│   │   ├── jesse_style/         # Jesse风格指标
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # 从 _indicator_base/_cls_ind.py 移入
│   │   │   ├── momentum/        # 动量类指标
│   │   │   │   ├── __init__.py
│   │   │   │   ├── rsi.py       # 从 prod_indicator/ 移入相关指标
│   │   │   │   ├── stochastic.py
│   │   │   │   └── cci.py
│   │   │   ├── volatility/      # 波动率类指标
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bollinger.py # 从 mod_bollinger.py 移入
│   │   │   │   └── yang_zhang.py # 从 volitility_indicator/ 移入
│   │   │   ├── volume/          # 成交量类指标
│   │   │   │   ├── __init__.py
│   │   │   │   ├── obv.py       # 从 norm_on_balance_volume.py 移入
│   │   │   │   └── cmf.py       # 从 chaiken_money_flow.py 移入
│   │   │   └── trend/           # 趋势类指标
│   │   │       ├── __init__.py
│   │   │       ├── ma.py        # 从 cmma.py 等移入
│   │   │       └── td_sequential.py # 从 prod_indicator/ 移入
│   │   ├── advanced/            # 高级指标
│   │   │   ├── __init__.py
│   │   │   ├── entropy/         # 熵指标（从 prod_indicator/entropy/ 移入）
│   │   │   │   ├── __init__.py
│   │   │   │   ├── appr_en.py
│   │   │   │   └── samp_en.py
│   │   │   ├── microstructure/  # 微观结构（从 prod_indicator/micro_structure/ 移入）
│   │   │   │   ├── __init__.py
│   │   │   │   ├── first_gen.py
│   │   │   │   └── second_gen.py
│   │   │   ├── wavelets/        # 小波分析（从 prod_indicator/wavelets/ 移入）
│   │   │   │   ├── __init__.py
│   │   │   │   └── cls_cwt_swt.py
│   │   │   ├── dominant_cycle/  # 主导周期（保持原结构）
│   │   │   │   ├── __init__.py
│   │   │   │   ├── dual_differentiator.py
│   │   │   │   ├── homodyne.py
│   │   │   │   └── phase_accumulation.py
│   │   │   └── fractional/      # 分数阶（从 prod_indicator/diff/ 移入）
│   │   │       ├── __init__.py
│   │   │       ├── frac.py
│   │   │       └── frac_ffd.py
│   │   └── experimental/        # 实验性指标（原beta_indicator/）
│   │       ├── __init__.py
│   │       └── hilbert_transformer.py
│   │
│   ├── features/                 # 特征工程
│   │   ├── __init__.py
│   │   ├── extraction/          # 特征提取
│   │   │   ├── __init__.py
│   │   │   └── all_features.py  # 从 custom_indicators/all_features.py 移入
│   │   ├── selection/           # 特征选择（从 toolbox/feature_selection/ 移入）
│   │   │   ├── __init__.py
│   │   │   ├── rfcq_selector.py
│   │   │   ├── fcq_selector.py
│   │   │   └── catfcq_selector.py
│   │   └── engineering/         # 特征构造
│   │       ├── __init__.py
│   │       ├── labeling/        # 标签生成（从 toolbox/labeler/ 移入）
│   │       │   ├── __init__.py
│   │       │   ├── triple_barrier.py
│   │       │   └── zigzag_labeler.py
│   │       ├── transforms/      # 特征变换（从 toolbox/ 移入）
│   │       │   ├── __init__.py
│   │       │   ├── transform.py
│   │       │   └── fracdiff.py
│   │       └── sizing/          # 仓位管理
│   │           ├── __init__.py
│   │           └── bet_sizing.py # 从 toolbox/bet_sizing.py 移入
│   │
│   ├── strategies/               # Jesse策略（保持现有结构）
│   │   ├── __init__.py
│   │   ├── BinanceBtcDBar5hAllOrNothing/
│   │   ├── BinanceBtcEntropyBarV1/
│   │   ├── DemoStrategy/
│   │   ├── ExampleStrategy/
│   │   ├── MLV1Partial/
│   │   └── TaStrategy/
│   │
│   └── ml/                       # 机器学习模块
│       ├── __init__.py
│       ├── models/               # 模型定义
│       ├── training/             # 训练流程
│       └── evaluation/           # 评估工具
│
├── research/                     # 研究代码（与生产代码分离）
│   ├── notebooks/                # Jupyter notebooks
│   │   ├── bar_research/        # K线研究
│   │   │   ├── bar_research.ipynb
│   │   │   ├── bar_gp_research.ipynb
│   │   │   ├── bar_deap_research.ipynb
│   │   │   └── bar_entropy_research.ipynb
│   │   ├── feature_analysis/    # 特征分析
│   │   │   ├── 1_labels.ipynb
│   │   │   ├── 2_features.ipynb
│   │   │   ├── 3_feature_selection.ipynb
│   │   │   └── 4_models.ipynb
│   │   └── tuning/              # 调优实验
│   │       └── check_tuning_pipeline.ipynb
│   └── scripts/                  # 研究脚本
│       └── bar_symbolic_regression/
│           ├── style_gplearn.py
│           ├── symbolic_regression_deap.py
│           └── symbolic_regression_deap_advanced.py
│
├── tests/                        # 测试代码
│   ├── unit/                     # 单元测试
│   │   ├── indicators/
│   │   ├── bars/
│   │   └── features/
│   ├── integration/              # 集成测试
│   └── fixtures/                 # 测试数据
│       └── gmm_params.py
│
├── data/                         # 数据目录（.gitignore）
│   ├── raw/                      # 原始数据
│   │   └── btc_1m.npy
│   ├── processed/                # 处理后数据
│   └── cache/                    # 缓存数据
│
├── outputs/                      # 输出目录（.gitignore）
│   ├── models/                   # 训练的模型
│   ├── logs/                     # 日志文件
│   └── reports/                  # 报告
│
├── configs/                      # 配置文件
│   ├── jesse/                    # Jesse配置
│   ├── database/                 # 数据库配置（pgbouncer等）
│   └── environments/             # 环境配置
│
├── scripts/                      # 运维脚本
│   ├── install.sh
│   ├── run.sh
│   └── deploy/                   # 部署脚本
│
├── docker/                       # Docker配置
├── docs/                         # 文档
│   ├── CLAUDE.md                # 项目说明（从根目录移入）
│   ├── README.md                 # 项目README（从根目录移入）
│   └── OPTUNA_USAGE.md          # Optuna使用说明（从根目录移入）
│
└── vendor/                       # 第三方代码（原extern/）
    ├── mlfinlab/
    ├── AI策略课程代码/
    └── ...

```

## 3. PyCharm 重构步骤指南

### 第一阶段：准备工作

1. **创建分支**
   ```bash
   git checkout -b refactor/project-structure
   ```

2. **备份当前代码**
   ```bash
   git add .
   git commit -m "Backup before refactoring"
   ```

3. **创建新目录结构**
   - 在 PyCharm 中右键项目根目录
   - New > Directory 创建 `src/`, `research/`, `configs/` 等

### 第二阶段：使用 PyCharm 重构功能

#### 移动和重命名文件（Refactor > Move/Rename）

1. **移动 bar 模块**
   - 选中 `bar/` 目录
   - Refactor > Move (F6)
   - 目标路径：`src/bars/`
   - ✅ 勾选 "Search for references"
   - ✅ 勾选 "Search in comments and strings"

2. **重命名和重组 custom_indicators**
   - 选中 `custom_indicators/` 
   - Refactor > Move 到 `src/indicators/`
   - 逐个子模块进行重组：
     - `prod_indicator/` → 分散到 `jesse_style/` 和 `advanced/`
     - `beta_indicator/` → `experimental/`
     - `toolbox/` → 分散到 `src/features/`

3. **提取特征工程模块**
   - 选中 `custom_indicators/toolbox/feature_selection/`
   - Refactor > Move 到 `src/features/selection/`
   - 选中 `custom_indicators/toolbox/labeler/`
   - Refactor > Move 到 `src/features/engineering/labeling/`

#### 更新导入路径（Refactor > Optimize Imports）

1. **批量更新导入**
   - 选中 `src/` 目录
   - Code > Optimize Imports (Ctrl+Alt+O)
   - 在弹出的对话框中检查导入变更

2. **手动修复相对导入**
   - 使用 Find and Replace (Ctrl+Shift+R)
   - 搜索模式：`from \.\.(\w+)`
   - 替换为绝对导入：`from src.$1`

### 第三阶段：重组研究代码

1. **移动 Notebooks**
   - 创建 `research/notebooks/` 目录结构
   - 将所有 `.ipynb` 文件移动到对应子目录
   - 更新 notebook 中的导入路径

2. **移动研究脚本**
   - `bar_research/` → `research/scripts/bar_symbolic_regression/`
   - 保持文件名不变，仅调整位置

### 第四阶段：整理配置和数据

1. **配置文件迁移**
   - `optuna_config.py` → `src/core/config/optuna_config.py`
   - `pgbouncer/` → `configs/database/pgbouncer/`
   - 创建 `configs/environments/` 并添加环境配置文件

2. **数据目录重组**
   - `data/btc_1m.npy` → `data/raw/btc_1m.npy`
   - `storage/temp/` → `data/cache/`
   - `storage/logs/` → `outputs/logs/`

### 第五阶段：创建包初始化文件

为每个新模块创建 `__init__.py`：

```python
# src/__init__.py
"""Jesse-Trade 核心源代码包"""
__version__ = "1.0.0"

# src/bars/__init__.py
"""自定义K线构建模块"""
from .builders import *

# src/indicators/__init__.py
"""技术指标库"""
from .jesse_style import *
from .advanced import *

# src/features/__init__.py
"""特征工程模块"""
from .extraction import *
from .selection import *
from .engineering import *
```

### 第六阶段：更新配置文件

1. **更新 setup.py 或 pyproject.toml**
   ```python
   packages=find_packages(where="src"),
   package_dir={"": "src"},
   ```

2. **更新 .gitignore**
   ```
   # 数据文件
   /data/
   !/data/.gitkeep
   
   # 输出文件
   /outputs/
   !/outputs/.gitkeep
   
   # 缓存
   __pycache__/
   *.pyc
   .pytest_cache/
   ```

3. **更新 requirements.txt**
   - 检查并更新依赖版本
   - 分离 requirements.txt 和 requirements-dev.txt

## 4. 命名规范和最佳实践

### 文件和模块命名
- **模块**：使用小写字母和下划线 `snake_case.py`
- **包目录**：使用小写字母，避免下划线 `features/`
- **测试文件**：前缀 `test_` 如 `test_entropy_bar.py`

### 代码命名规范
```python
# 常量
MAX_BARS_PER_DAY = 1440
DEFAULT_THRESHOLD = 0.5

# 类名（PascalCase）
class EntropyBarBuilder:
    pass

# 函数和方法（snake_case）
def calculate_entropy(data: np.ndarray) -> float:
    pass

# 私有成员（前缀单下划线）
class IndicatorBase:
    def __init__(self):
        self._internal_state = None
    
    def _calculate_internal(self):
        pass

# 模块级私有（前缀单下划线）
_INTERNAL_CONSTANT = 100

def _internal_helper():
    pass
```

### 导入顺序
```python
# 1. 标准库
import os
import sys
from typing import List, Optional

# 2. 第三方库
import numpy as np
import pandas as pd
from numba import njit

# 3. Jesse相关
from jesse import helpers
from jesse.strategies import Strategy

# 4. 项目内部模块
from src.bars.builders import EntropyBar
from src.indicators.jesse_style.momentum import RSI
from src.features.selection import RFCQSelector
```

## 5. 测试结构设置

### 创建测试文件
```python
# tests/unit/bars/test_entropy_bar.py
import pytest
from src.bars.builders.entropy_bar import EntropyBar

class TestEntropyBar:
    def test_build_bar(self):
        # 测试代码
        pass

# tests/integration/test_strategy_execution.py
import pytest
from jesse import research
from src.strategies.DemoStrategy import DemoStrategy

def test_strategy_backtest():
    # 集成测试代码
    pass
```

### 运行测试
```bash
# 运行所有测试
pytest tests/

# 运行特定模块测试
pytest tests/unit/bars/

# 运行带覆盖率的测试
pytest --cov=src tests/
```

## 6. 迁移检查清单

### 代码迁移
- [ ] 所有 Python 文件已移动到新位置
- [ ] 所有导入路径已更新
- [ ] 相对导入改为绝对导入
- [ ] 循环依赖已解决

### 配置更新
- [ ] setup.py/pyproject.toml 已更新
- [ ] requirements.txt 已整理
- [ ] .gitignore 已更新
- [ ] CI/CD 配置已调整

### 文档更新
- [ ] README.md 已更新
- [ ] CLAUDE.md 已移动并更新
- [ ] API 文档已重新生成

### 测试验证
- [ ] 单元测试全部通过
- [ ] 集成测试全部通过
- [ ] Jesse 回测正常运行
- [ ] 导入无错误

## 7. 回滚计划

如果重构出现问题：

```bash
# 回滚到备份点
git reset --hard HEAD~1

# 或者切换回主分支
git checkout main
```

## 8. 后续优化建议

1. **添加类型注解**
   - 为所有公共API添加类型注解
   - 使用 `mypy` 进行类型检查

2. **文档字符串**
   - 为所有模块、类、函数添加docstring
   - 使用 Sphinx 生成文档

3. **性能优化**
   - 继续使用 Numba 优化计算密集型代码
   - 考虑使用 Cython 或 Rust 扩展

4. **代码质量工具**
   - 配置 `black` 自动格式化
   - 配置 `flake8` 或 `ruff` 代码检查
   - 配置 `pre-commit` hooks

5. **持续集成**
   - 设置 GitHub Actions 或 GitLab CI
   - 自动运行测试和代码质量检查

## 9. PyCharm 快捷键参考

| 操作 | 快捷键 |
|------|--------|
| 重命名 | Shift+F6 |
| 移动文件/目录 | F6 |
| 安全删除 | Alt+Delete |
| 提取变量 | Ctrl+Alt+V |
| 提取方法 | Ctrl+Alt+M |
| 优化导入 | Ctrl+Alt+O |
| 查找使用 | Alt+F7 |
| 全局搜索 | Ctrl+Shift+F |
| 全局替换 | Ctrl+Shift+R |

## 10. 注意事项

1. **逐步迁移**：不要一次性重构所有代码，分阶段进行
2. **保持可运行**：每个阶段结束后确保代码可运行
3. **版本控制**：频繁提交，便于回滚
4. **团队沟通**：如果是团队项目，提前沟通重构计划
5. **文档更新**：及时更新文档，特别是 API 变更

---

## 附录：示例迁移脚本

```python
#!/usr/bin/env python
"""
辅助重构脚本：更新所有文件中的导入路径
"""

import os
import re
from pathlib import Path

def update_imports(root_dir: str):
    """批量更新导入路径"""
    
    # 定义导入映射
    import_mappings = {
        r'from bar\.': 'from src.bars.',
        r'from custom_indicators\.': 'from src.indicators.',
        r'import bar\.': 'import src.bars.',
        r'import custom_indicators\.': 'import src.indicators.',
    }
    
    # 遍历所有 Python 文件
    for py_file in Path(root_dir).rglob('*.py'):
        content = py_file.read_text()
        modified = False
        
        # 应用所有映射
        for pattern, replacement in import_mappings.items():
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                modified = True
        
        # 写回文件
        if modified:
            py_file.write_text(content)
            print(f"Updated: {py_file}")

if __name__ == "__main__":
    update_imports("./src")
```

---

本文档将持续更新，建议在重构过程中记录遇到的问题和解决方案。