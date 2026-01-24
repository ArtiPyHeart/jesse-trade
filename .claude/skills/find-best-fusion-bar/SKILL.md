---
name: find-best-fusion-bar
description: Guided development of custom Fusion Bars (trend axes) with statistical validation. Helps users design threshold formulas, develop FusionBar classes, and optimize parameters via Optuna. Use when users mention "构建新的趋势轴", "寻找最佳fusion bar", "开发自定义轴", or want to create custom K-line aggregation methods.
---

# 寻找最佳趋势轴 (Fusion Bar)

## 任务概述
通过统计验证，帮助用户构建并优化自定义趋势轴（Fusion Bar），验证其是否适合趋势交易。

**核心原则**：
- 输入的原始 K 线固定为 1 分钟 Jesse candles
- 每个公式独立开发、独立优化，逐一完成
- Optuna 优化必须充分运行（建议 1000+ trials，可后台运行数小时）
- 最终交付带有优化后默认参数的 FusionBar class

## 触发条件
用户消息包含以下**完整短语**时触发此 skill（避免误触发）：
- "构建新的趋势轴" / "开发新的趋势轴" / "寻找最佳趋势轴"
- "构建新的 fusion bar" / "寻找最佳 fusion bar"
- "开发自定义轴" / "构建自定义轴"
- "帮我优化 fusion bar 参数"

**不触发**的情况（仅讨论概念）：
- "什么是趋势轴" / "fusion bar 是什么"
- "查看 DemoBar 代码"

## 工作流程

### 阶段 1：公式确认
1. **提取公式**：从用户描述或 md 文件中提取阈值计算公式
2. **与用户确认**：
   - 公式的数学表达式
   - 公式中各参数的含义和搜索范围
   - 阈值合并条件（`threshold` 的语义）
3. **多公式处理**：如有多个公式，创建 md 文件记录，逐一处理

**确认模板**：
```markdown
## 公式确认

### 阈值计算公式
`threshold_value = abs(close - close_lag1) * (high - low) / close`

### 参数说明
| 参数名 | 含义 | 搜索范围 | 类型 |
|--------|------|----------|------|
| clip_r | 噪声过滤阈值 | [0.0001, 0.01] | float (log) |
| threshold | 累积阈值 | [0.5, 5.0] | float |

### 合并逻辑
当 `sum(threshold_values) >= threshold` 时，生成新的 Fusion Bar
```

### 阶段 2：开发 FusionBar Class
1. **创建文件**：`src/bars/fusion/{name}.py`
2. **继承基类**：`FusionBarContainerBase`
3. **实现方法**：
   - `__init__`: 初始化参数（暂用默认值）
   - `max_lookback`: 返回公式需要的历史 K 线数量
   - `get_thresholds`: 实现阈值计算公式

**代码模板**：
```python
import numpy as np
from src.bars.fusion.base import FusionBarContainerBase


class {ClassName}(FusionBarContainerBase):
    """
    {公式描述}

    Parameters:
    -----------
    {参数文档}
    """

    def __init__(
        self,
        max_bars: int = -1,
        # 其他参数（暂用占位默认值）
        threshold: float = 1.0,
    ):
        super().__init__(max_bars, threshold)
        # 存储其他参数

    @property
    def max_lookback(self) -> int:
        return {需要的历史 K 线数量}

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        # 实现阈值计算公式
        # candles: [timestamp, open, close, high, low, volume]
        pass
```

### 阶段 3：准备数据
1. **获取 K 线数据**：使用 `research.get_candles`
2. **保存到本地**：`data/{symbol}_1m.npy`

**数据获取脚本**（在项目根目录运行）：
```python
import numpy as np
import jesse.helpers as helpers
from jesse.research import get_candles

# 获取 BTC 三年数据
_, candles = get_candles(
    "Binance Perpetual Futures",
    "BTC-USDT",
    "1m",
    helpers.date_to_timestamp("2022-01-01"),
    helpers.date_to_timestamp("2025-01-01"),
    warmup_candles_num=0,
    caching=True,
    is_for_jesse=False,
)
candles = candles[candles[:, 5] > 0]  # 过滤无效数据
np.save("data/btc_1m.npy", candles)
print(f"保存了 {len(candles):,} 根 1 分钟 K 线")
```

### 阶段 4：Optuna 优化
1. **使用 TrendOptimizer**：从 `research.trend_optimizer.optimizer` 导入
2. **配置参数范围**：根据阶段 1 确认的范围
3. **充分运行**：建议 1000+ trials，**必须后台运行**

**优化脚本**：
```python
import numpy as np
from src.bars.fusion.{module} import {ClassName}
from research.trend_optimizer.optimizer import TrendOptimizer

candles = np.load("data/btc_1m.npy")

optimizer = TrendOptimizer(
    fusion_bar_cls={ClassName},
    candles=candles,
    window_sizes=(20, 40, 60),
    n_top_results=10,
)

results = optimizer.optimize(
    n_trials=1000,  # 建议 1000+
    # 参数范围
    clip_r=(0.0001, 0.01, "log"),
    threshold=(0.5, 5.0),
)

# 输出结果
print("\n最优参数：")
for r in results[:5]:
    print(f"  rank={r.rank}, score={r.final_score:.3f}, params={r.params}")
```

**关键要求**：
- 使用 `run_in_background=True` 后台运行
- 运行时间可能需要数小时，不要中途停止
- 使用 `Read` 工具定期检查 output 文件查看进度

### 阶段 5：填入默认参数
1. **提取最优参数**：从优化结果中获取 rank=1 的参数
2. **更新 class**：将最优参数作为 `__init__` 的默认值
3. **验证**：使用 `evaluate_detailed` 生成评估报告

**验证脚本**：
```python
from src.bars.fusion.{module} import {ClassName}
from research.trend_optimizer.evaluator import MultiWindowEvaluator

candles = np.load("data/btc_1m.npy")

# 使用最优参数
container = {ClassName}()  # 使用默认参数
container.update_with_candles(candles)
fusion_bars = container.get_fusion_bars()

# 生成详细报告
evaluator = MultiWindowEvaluator()
report = evaluator.evaluate_detailed(fusion_bars)
print(report)
```

### 阶段 6：基准对比与交付
1. **与 DemoBar 对比**：使用相同数据生成两个评估报告
2. **展示对比结果**：综合评分、分项得分、等级
3. **质量判定**：
   - 新轴 ≥ DemoBar：正常交付
   - 新轴 < DemoBar 且差距 ≤ 10 分：提示用户"略低于基准，建议谨慎使用"
   - 新轴 < DemoBar 且差距 > 10 分：**警告用户"明显低于基准，此轴可能不适合趋势交易"**
4. **告知文件位置**：`src/bars/fusion/{name}.py`
5. **说明后续步骤**：用户需进行机器学习建模与回测

**对比脚本**：
```python
import numpy as np
from src.bars.fusion.demo import DemoBar
from src.bars.fusion.{module} import {ClassName}
from research.trend_optimizer.evaluator import MultiWindowEvaluator

candles = np.load("data/btc_1m.npy")
evaluator = MultiWindowEvaluator()

# DemoBar 基准
demo = DemoBar()
demo.update_with_candles(candles)
demo_report = evaluator.evaluate_detailed(demo.get_fusion_bars())

# 新轴
new_bar = {ClassName}()
new_bar.update_with_candles(candles)
new_report = evaluator.evaluate_detailed(new_bar.get_fusion_bars())

# 对比
print(f"DemoBar:  {demo_report.overall_score:.1f}/100 ({demo_report.overall_grade})")
print(f"新轴:     {new_report.overall_score:.1f}/100 ({new_report.overall_grade})")
print(f"差距:     {new_report.overall_score - demo_report.overall_score:+.1f}")
```

## 重要提醒

### 优化时间要求
- **最低要求**：500 trials
- **推荐配置**：1000+ trials
- **后台运行**：必须使用 `run_in_background=True`
- **不要偷懒**：宁愿多等几小时，也不要提前终止

### 多公式处理
如果用户提供多个公式：
1. 创建 `research/fusion_bar_drafts/{日期}_{主题}.md` 记录所有公式
2. 逐一完成：确认 → 开发 → 优化 → 交付
3. 每完成一个，向用户报告进度

### 参数范围约定
| 参数类型 | Optuna 格式 | 示例 |
|----------|-------------|------|
| 普通浮点 | `(min, max)` | `threshold=(0.5, 5.0)` |
| 对数浮点 | `(min, max, "log")` | `clip_r=(0.0001, 0.01, "log")` |
| 整数 | `(min, max, "int")` | `window=(10, 100, "int")` |
| 分类 | `([v1, v2, ...], "category")` | `mode=(["fast", "slow"], "category")` |

### 文件位置
- FusionBar 类：`src/bars/fusion/{name}.py`
- 基类：`src/bars/fusion/base.py`
- 参考实现：`src/bars/fusion/demo.py` (DemoBar)
- 评估器：`research/trend_optimizer/evaluator.py`
- 优化器：`research/trend_optimizer/optimizer.py`
- 数据：`data/{symbol}_1m.npy`

## 示例对话

**用户**：我想构建一个新的趋势轴，公式是 `abs(close - open) / (high - low + 1e-8)`，用 BTC 三年数据优化

**Claude**：
1. 确认公式和参数范围
2. 创建 `src/bars/fusion/body_ratio.py`
3. 获取并保存 BTC 数据
4. 后台运行 1000+ trials 优化
5. 填入最优参数并交付

---

## 自更新机制

**本 skill 是一个持续迭代的文档**，在实践中不断积累最佳实践。

### 何时更新此文档
遇到以下情况时，Claude 应主动更新本 skill：

1. **被用户纠正**：用户指出流程中的错误或更好的做法
2. **频繁出错**：某个步骤反复失败，找到根本原因后记录解决方案
3. **发现更优方法**：实践中发现比文档描述更高效的方法
4. **新增公式类型**：遇到需要特殊处理的公式模式
5. **工具/API 变更**：依赖的代码接口发生变化

### 更新内容格式
更新时应添加到下方「实践经验」章节，格式：
```markdown
### [日期] 经验标题
**场景**：简述遇到的问题或情况
**解决方案**：具体的解决方法
**教训**：总结性的经验（可选）
```

### 更新原则
- **只记录验证过的成功经验**，不记录猜测
- **保持简洁**，避免冗余描述
- **标注日期**，便于追溯
- **如涉及流程变更**，同步更新上方「工作流程」章节

---

## 实践经验

> 此章节记录实践中积累的经验教训，由 Claude 在执行任务时自动更新。

### [2025-01] 初始版本
**场景**：首次创建此 skill
**内容**：基于 DemoBar 和 TrendOptimizer 的现有实现，整理出标准化的 6 阶段流程

<!--
未来经验记录示例：

### [2025-02] 数据获取需在项目根目录运行
**场景**：在子目录运行数据获取脚本时，jesse 无法读取 .env 配置
**解决方案**：确保所有涉及 jesse.research.get_candles 的脚本都在项目根目录运行
**教训**：CLAUDE.md 中已有此提醒，但容易遗忘

### [2025-03] clip_r 参数建议使用 log 采样
**场景**：clip_r 范围 [0.0001, 0.01] 跨越两个数量级
**解决方案**：使用 `clip_r=(0.0001, 0.01, "log")` 而非线性采样
**教训**：跨数量级的参数优先考虑对数采样
-->
