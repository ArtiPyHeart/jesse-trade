---
name: find-best-fusion-bar
description: Guided development of custom Fusion Bars (trend axes) with statistical validation. Helps users design threshold formulas, develop FusionBar classes, and optimize parameters via Optuna. Use when users mention "构建新的趋势轴", "寻找最佳fusion bar", "开发自定义轴", or want to create custom K-line aggregation methods.
---

# 寻找最佳趋势轴 (Fusion Bar)

## 任务概述

通过统计验证，帮助用户构建并优化自定义趋势轴（Fusion Bar），验证其是否适合趋势交易。

**核心原则**：
- 输入固定为 1 分钟 Jesse candles
- 每个公式独立开发、独立优化，逐一完成
- Optuna 优化必须充分运行（2500+ trials，后台运行）
- 最终交付带有优化后默认参数的 FusionBar class

## 触发条件

用户消息包含以下**完整短语**时触发：
- "构建新的趋势轴" / "开发新的趋势轴" / "寻找最佳趋势轴"
- "构建新的 fusion bar" / "寻找最佳 fusion bar"
- "开发自定义轴" / "构建自定义轴"
- "帮我优化 fusion bar 参数"

**不触发**："什么是趋势轴"、"fusion bar 是什么"、"查看 DemoBar 代码"

---

## 工作流程

### 阶段 1：公式确认

1. **提取公式**：从用户描述或 md 文件中提取阈值计算公式
2. **与用户确认**：公式表达式、参数含义、阈值合并条件
3. **多公式处理**：创建 `research/fusion_bar_drafts/{日期}_{主题}.md` 记录，逐一处理

### 阶段 2：开发 FusionBar Class

创建 `src/bars/fusion/{name}.py`，继承 `FusionBarContainerBase`：

```python
import numpy as np
from src.bars.fusion.base import FusionBarContainerBase


class {ClassName}(FusionBarContainerBase):
    """
    {公式描述}

    公式：{数学表达式}

    Parameters
    ----------
    max_bars : int
        最大bar数量，-1表示不限制
    {其他参数说明}
    threshold : float
        累积阈值

    Benchmark (BTC 2022-2025, 1min)
    -------------------------------
    配置: {tier} rank {rank} (Optuna score: {score})
    输入: {n} 根 1min K线 → 输出: {m} 根 Fusion Bar
    压缩比: {ratio}:1 (约 {hours} 小时/根)

    评估结果:
      综合评分: {score}/100 ({grade})
      趋势性: {t}/5 | 一致性: {c}/5 | 稳定性: {s}/5
    """

    def __init__(
        self,
        max_bars: int = -1,
        # 门控参数（暂用占位值，优化后替换）
        param: float = 0.4,
        threshold: float = 1e-6,
        epsilon: float = 1e-10,
    ):
        super().__init__(max_bars, threshold)
        self.param = param
        self.epsilon = epsilon

    @property
    def max_lookback(self) -> int:
        return {需要的历史K线数量}

    def get_thresholds(self, candles: np.ndarray) -> np.ndarray:
        # candles: [timestamp, open, close, high, low, volume]
        # 返回每根K线的阈值贡献
        pass
```

### 阶段 3：准备数据

**数据获取脚本**（项目根目录运行）：
```python
import numpy as np
import jesse.helpers as helpers
from jesse.research import get_candles

_, candles = get_candles(
    "Binance Perpetual Futures", "BTC-USDT", "1m",
    helpers.date_to_timestamp("2022-01-01"),
    helpers.date_to_timestamp("2025-01-01"),
    warmup_candles_num=0, caching=True, is_for_jesse=False,
)
candles = candles[candles[:, 5] > 0]
np.save("data/btc_1m.npy", candles)
print(f"保存了 {len(candles):,} 根 1 分钟 K 线")
```

### 阶段 3.5：参数范围校准（关键！）

**不要凭直觉设定参数范围！** 必须先用真实数据统计阈值分布。

#### 目标 Bar 数量范围

| 基准 | 计算方式 | 说明 |
|------|----------|------|
| 上限（30min）| `n_candles // 30` | Bar 太多 → 评估极慢 |
| 下限（6h）| `n_candles // 360` | Bar 太少 → 交易机会不足 |

#### 校准脚本

```python
import numpy as np
from src.bars.fusion.{module} import {ClassName}

candles = np.load("data/btc_1m.npy")
n = len(candles)
bar_max, bar_min = n // 30, n // 360
print(f"K线: {n:,}, 目标bar范围: {bar_min:,} ~ {bar_max:,}")

# 阈值分布
bar = {ClassName}(threshold=1.0)
th = bar.get_thresholds(candles)
print(f"阈值: min={np.min(th):.2e}, max={np.max(th):.2e}, median={np.median(th):.2e}")

# 二分搜索目标threshold
def find_th(target, lo, hi):
    for _ in range(30):
        mid = (lo + hi) / 2
        bar = {ClassName}(threshold=mid)
        bar.update_with_candles(candles)
        if len(bar.get_fusion_bars()) > target:
            lo = mid
        else:
            hi = mid
    return mid

th_max = find_th(bar_max, 1e-10, 1e-1)
th_min = find_th(bar_min, 1e-10, 1e-1)
print(f"建议范围: threshold=({th_max:.2e}, {th_min:.2e}, 'log')")
```

### 阶段 4：Optuna 优化

**优化脚本**（必须使用 `optimize_and_return_study` + pandas 保存 CSV）：

```python
import numpy as np
import pandas as pd
from src.bars.fusion.{module} import {ClassName}
from research.trend_optimizer.optimizer import TrendOptimizer
from research.trend_optimizer.evaluator import extract_top_n_by_tiers

candles = np.load("data/btc_1m.npy")
n_candles = len(candles)
TARGET_BAR_RANGE = (n_candles // 360, n_candles // 30)

optimizer = TrendOptimizer(
    fusion_bar_cls={ClassName},
    candles=candles,
    window_sizes=(20, 40, 60),
    n_top_results=5,
)

# 必须用 optimize_and_return_study，不要用 optimize！
study = optimizer.optimize_and_return_study(
    n_trials=2500,
    param=(...),  # 来自校准
    threshold=(..., ..., "log"),
)

# 分层提取 (10 tier × 5 rank = 50 条)
tiered = extract_top_n_by_tiers(study, TARGET_BAR_RANGE, n_per_tier=5)  # 默认 10 分区

# 用 pandas 保存 CSV（简洁！）
rows = []
for tier in tiered:
    for r in tier.results:
        rows.append({
            "tier": tier.tier_name,
            "rank": r.rank,
            "score": r.final_score,
            "bars": r.fusion_bar_count,
            "compression": n_candles / r.fusion_bar_count if r.fusion_bar_count > 0 else 0,
            **r.params,
        })
df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
df.to_csv("research/{name}_tiered.csv", index=False)
```

**运行命令**：
```bash
PYTHONPATH=/path/to/jesse-trade python research/optimize_{name}.py
```

### 阶段 5：用户选择参数

**重要：最终参数由用户决定，不要自动选择！**

分层结果（10 tier）便于权衡：
- **tier1**（bar 最少）：趋势性可能最强，但交易机会最少
- **tier10**（bar 最多）：交易机会多，但趋势性可能略低

用户选择后：
1. 更新 class 默认参数
2. 运行评估，填充 docstring 中的 Benchmark
3. 清理临时文件：`research/calibrate_{name}.py`、`research/optimize_{name}.py`、`research/{name}_tiered.csv`

### 阶段 6：基准对比与交付

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
demo_bars = demo.get_fusion_bars()

# 新轴
new = {ClassName}()
new.update_with_candles(candles)
new_bars = new.get_fusion_bars()

print(f"DemoBar: {len(demo_bars):,} bars")
print(f"新轴: {len(new_bars):,} bars ({len(new_bars)/len(demo_bars):.0%} of DemoBar)")

demo_report = evaluator.evaluate_detailed(demo_bars)
new_report = evaluator.evaluate_detailed(new_bars)
print(f"DemoBar: {demo_report.overall_score:.1f}/100")
print(f"新轴: {new_report.overall_score:.1f}/100")
```

**质量判定**：
- Bar 数量 < DemoBar 25%：⚠️ 交易机会极少
- 评分 < DemoBar - 10：⚠️ 趋势性不足

---

## 重要提醒

### 脚本运行环境
所有 `research/` 脚本必须设置 PYTHONPATH：
```bash
PYTHONPATH=/path/to/jesse-trade python research/xxx.py
```

### 优化配置
- **trials**: 2500+（探索比微调更重要）
- **探索比例**: 99% 随机探索 + 1% 收束
- **后台运行**: 必须使用 `run_in_background=True`

### 参数格式
| 类型 | 格式 | 示例 |
|------|------|------|
| 普通浮点 | `(min, max)` | `beta=(0.1, 2.0)` |
| 对数浮点 | `(min, max, "log")` | `threshold=(1e-5, 1e-3, "log")` |
| 整数 | `(min, max, "int")` | `window=(10, 100, "int")` |

### 文件位置
- FusionBar 类：`src/bars/fusion/{name}.py`
- 基类：`src/bars/fusion/base.py`
- 评估器：`research/trend_optimizer/evaluator.py`
- 优化器：`research/trend_optimizer/optimizer.py`
- 数据：`data/{symbol}_1m.npy`

---

## 实践经验

> 此章节记录关键教训，避免重复踩坑。

### 参数范围必须数据驱动
**问题**：直接套用其他轴的参数范围，导致所有 trial 的 bar 数量都是 0 或 1
**原因**：不同公式的量级可能差异 1000 倍以上
**方案**：阶段 3.5 强制校准

### 必须用 optimize_and_return_study
**问题**：使用 `optimize()` 只返回 top 5，无法获取 5-tier 分层结果
**方案**：必须用 `optimize_and_return_study()` + `extract_top_n_by_tiers()`
**代价**：2500 trials 需要 16-19 小时，方法用错只能重跑

### 用 pandas 保存 CSV
**问题**：用 csv 模块手动写 CSV 代码冗长易错
**方案**：整理为 `list[dict]` → `pd.DataFrame(rows)` → `df.to_csv()`

### 好的趋势轴特征
- ≥3分 窗口占比 > 65%
- 1分+2分 窗口占比 < 35%
- 均分 > 3.0
