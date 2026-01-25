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

    Benchmark (BTC 2022-2025, 1min):
    --------------------------------
    {原始K线数} 根 1min K线 → {FusionBar数} 根 Fusion Bar
    压缩比 {压缩比}:1，约 {平均时长} 生成一根
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

### 阶段 3.5：参数范围校准（关键步骤）

**不要凭直觉设定参数范围！** 必须先用真实数据统计阈值分布，再确定 Optuna 搜索范围。

#### 目标 Bar 数量范围（重要！）

Fusion bar 数量应控制在 **30分钟 K线数量** 到 **6小时 K线数量** 之间：

| 基准 | 计算方式 | 100,000 根 1min K线 |
|------|----------|---------------------|
| 上限（30min）| `candle_count / 30` | ~3,333 bars |
| 下限（6h）| `candle_count / 360` | ~278 bars |

**为什么要限制这个范围**：
- **Bar 太多**（> 30min 基准）→ TrendValidator 评估极慢，单个 trial 可能从 7 秒变成 100+ 秒，严重影响调参效率
- **Bar 太少**（< 6h 基准）→ 交易机会太少，没有实际意义

#### 校准步骤

**重要：校准必须使用全量数据！** 校准数据量必须与 Optuna 优化时使用的数据量一致，否则 threshold 范围会出现严重偏差。

1. **计算阈值分布**：用新 class 的 `get_thresholds()` 在**全量数据**上计算
2. **统计关键指标**：min, max, mean, median, p5, p95
3. **统计累积阈值**：100/500/1000/5000 根 K 线后的累积值
4. **测试不同 threshold**：观察生成的 fusion bar 数量，**确保落在目标范围内**
5. **确定搜索范围**：根据统计结果设定合理范围

**校准脚本**：
```python
import numpy as np
from src.bars.fusion.{module} import {ClassName}

# 使用全量数据！
candles = np.load("data/btc_1m.npy")
n_candles = len(candles)
print(f"K线数量: {n_candles:,}")
print(f"时间跨度: {n_candles / 60 / 24:.1f} 天 ({n_candles / 60 / 24 / 365:.1f} 年)")

# 目标 bar 数量范围
bar_max = n_candles // 30   # 30min 基准（上限）
bar_min = n_candles // 360  # 6h 基准（下限）
print(f"目标 bar 数量范围: {bar_min:,} ~ {bar_max:,}")

# 计算阈值分布
bar = {ClassName}(threshold=1.0)
thresholds = bar.get_thresholds(candles[:n_candles])

print(f"\n阈值统计:")
print(f"  min:    {np.min(thresholds):.2e}")
print(f"  max:    {np.max(thresholds):.2e}")
print(f"  mean:   {np.mean(thresholds):.2e}")
print(f"  median: {np.median(thresholds):.2e}")
print(f"  p5:     {np.percentile(thresholds, 5):.2e}")
print(f"  p95:    {np.percentile(thresholds, 95):.2e}")

# 累积阈值
cumsum = np.cumsum(thresholds)
print(f"\n累积阈值:")
for i in [100, 500, 1000, 5000]:
    if i <= len(cumsum):
        print(f"  {i}根后: {cumsum[i-1]:.2e}")

# 二分搜索找到目标范围对应的 threshold
def find_threshold_for_bar_count(target_bars, lo, hi, candles, BarClass):
    for _ in range(30):
        mid = (lo + hi) / 2
        bar = BarClass(threshold=mid)
        bar.update_with_candles(candles)
        fusion = bar.get_fusion_bars()
        if len(fusion) > target_bars:
            lo = mid
        else:
            hi = mid
    return mid

# 找到目标范围的 threshold 边界
print(f"\n搜索目标 bar 数量对应的 threshold...")
th_for_max_bars = find_threshold_for_bar_count(bar_max, 1e-10, 1e-1, candles[:n_candles], {ClassName})
th_for_min_bars = find_threshold_for_bar_count(bar_min, 1e-10, 1e-1, candles[:n_candles], {ClassName})

print(f"  {bar_max:,} bars (30min) → threshold ≈ {th_for_max_bars:.2e}")
print(f"  {bar_min:,} bars (6h)    → threshold ≈ {th_for_min_bars:.2e}")

# 验证
print(f"\n验证 threshold 范围内的 bar 数量:")
for th in [th_for_max_bars, (th_for_max_bars + th_for_min_bars) / 2, th_for_min_bars]:
    bar = {ClassName}(threshold=th)
    bar.update_with_candles(candles[:n_candles])
    fusion = bar.get_fusion_bars()
    in_range = "✓" if bar_min <= len(fusion) <= bar_max else "✗"
    print(f"  threshold={th:.2e}: {len(fusion):,} bars {in_range}")

print(f"\n=== 建议的 Optuna 搜索范围 ===")
print(f"  threshold: ({th_for_max_bars:.2e}, {th_for_min_bars:.2e}, 'log')")
```

**范围确定原则**：
| 参数 | 范围确定方法 |
|------|-------------|
| clip_r | 从 p5 到 p50，使用 log scale |
| threshold | **必须确保 bar 数量落在 30min~6h 范围内**，使用 log scale |

### 阶段 4：Optuna 优化
1. **使用 TrendOptimizer**：从 `research.trend_optimizer.optimizer` 导入
2. **配置参数范围**：**根据阶段 3.5 校准的范围**（不要用直觉！）
3. **充分运行**：建议 1000+ trials，**必须后台运行**

**优化脚本**：
```python
import csv
import numpy as np
from src.bars.fusion.{module} import {ClassName}
from research.trend_optimizer.optimizer import TrendOptimizer
from research.trend_optimizer.evaluator import extract_top_n_by_tiers

candles = np.load("data/btc_1m.npy")

# 目标 bar 范围（来自校准）
TARGET_BAR_RANGE = ({min_bars}, {max_bars})

optimizer = TrendOptimizer(
    fusion_bar_cls={ClassName},
    candles=candles,
    window_sizes=(20, 40, 60),
    n_top_results=10,
)

# 运行优化（返回 study 对象用于分层提取）
study = optimizer.optimize_and_return_study(
    n_trials=2500,  # 最低 1000，推荐 2500+
    # 参数范围（来自阶段 3.5 校准结果）
    param1=(...),
    param2=(...),
)

# 分层提取结果（长/中/短周期各 top 5）
tiered_results = extract_top_n_by_tiers(study, TARGET_BAR_RANGE, n_per_tier=5)

# 输出结果
print("\n" + "=" * 80)
for tier in tiered_results:
    print(f"\n=== {tier.tier_name.upper()} 周期 ({tier.tier_range[0]:,} ~ {tier.tier_range[1]:,} bars) ===")
    print(f"{'Rank':<6}{'Score':<10}{'Bars':<12}{'压缩比':<10}{'params'}")
    print("-" * 70)
    for r in tier.results:
        compression = len(candles) / r.fusion_bar_count if r.fusion_bar_count > 0 else 0
        print(f"{r.rank:<6}{r.final_score:<10.3f}{r.fusion_bar_count:<12,}{compression:<10.1f}{r.params}")

# 保存分层 CSV（共 15 行）
csv_path = "research/{name}_tiered_top15.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tier", "rank", "score", "bars", "compression", "param1", "param2"])
    for tier in tiered_results:
        for r in tier.results:
            compression = len(candles) / r.fusion_bar_count if r.fusion_bar_count > 0 else 0
            writer.writerow([
                tier.tier_name,
                r.rank,
                f"{r.final_score:.4f}",
                r.fusion_bar_count,
                f"{compression:.1f}",
                # 根据实际参数调整
                f"{r.params['param1']:.6e}",
                f"{r.params['param2']:.6e}",
            ])

print(f"\n结果已保存到: {csv_path}")
print("请查看 CSV 后选择一个配置，告诉我 tier + rank 用于设置默认参数。")
```

**关键要求**：
- 使用 `run_in_background=True` 后台运行
- 运行时间可能需要数小时，不要中途停止
- 使用 `Read` 工具定期检查 output 文件查看进度

### 阶段 5：用户选择参数

**重要：最终参数由用户决定，不要自动选择最高分！**

分层结果便于分析权衡：
- **Long 周期**（bar 少）：趋势性可能更强，但交易机会少
- **Medium 周期**：平衡选择
- **Short 周期**（bar 多）：交易机会多，但趋势性可能略低

**如果发现 Long 周期始终显著占优势**，这是一个重要信号：
- 该自定义轴可能不适合频繁交易
- 需要考虑是否符合策略需求

#### 用户确认后再更新

1. **等待用户选择**：明确询问用户选择哪个 tier + rank
2. **更新 class**：
   - 将用户选择的参数作为 `__init__` 的默认值
   - 更新 docstring 中的 Benchmark section，填入实际压缩比例：
     ```
     Benchmark (BTC 2022-2025, 1min):
     --------------------------------
     1,576,765 根 1min K线 → 20,605 根 Fusion Bar
     压缩比 76.5:1，约 1.3 小时生成一根
     ```
3. **验证**：使用 `evaluate_detailed` 生成评估报告
4. **清理**：删除优化脚本和 CSV 文件（如 `research/optimize_{name}.py`、`research/{name}_tiered_top15.csv`）

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
2. **展示对比结果**：
   - 综合评分、分项得分、等级
   - **Fusion bar 数量**（重要！bar 数量 = 交易机会）
   - 压缩比（1min candles / fusion bars）
3. **质量判定**：
   - **趋势性判定**：
     - 新轴 ≥ DemoBar：正常
     - 新轴 < DemoBar 且差距 ≤ 10 分：提示"略低于基准，建议谨慎使用"
     - 新轴 < DemoBar 且差距 > 10 分：**警告"明显低于基准，此轴可能不适合趋势交易"**
   - **Bar 数量判定**（同样重要）：
     - 新轴 bar 数量 ≥ DemoBar 的 50%：正常
     - 新轴 bar 数量 < DemoBar 的 50%：**警告"交易机会较少"**
     - 新轴 bar 数量 < DemoBar 的 25%：**强烈警告"交易机会极少，需确认是否符合策略需求"**
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
demo_fusion = demo.get_fusion_bars()
demo_report = evaluator.evaluate_detailed(demo_fusion)

# 新轴
new_bar = {ClassName}()
new_bar.update_with_candles(candles)
new_fusion = new_bar.get_fusion_bars()
new_report = evaluator.evaluate_detailed(new_fusion)

# 对比：趋势性得分
print("=== 趋势性得分 ===")
print(f"DemoBar:  {demo_report.overall_score:.1f}/100 ({demo_report.overall_grade})")
print(f"新轴:     {new_report.overall_score:.1f}/100 ({new_report.overall_grade})")
print(f"差距:     {new_report.overall_score - demo_report.overall_score:+.1f}")

# 对比：Bar 数量（交易机会）
print("\n=== Bar 数量（交易机会）===")
print(f"原始 K 线:   {len(candles):,}")
print(f"DemoBar:     {len(demo_fusion):,} bars (压缩比 {len(candles)/len(demo_fusion):.1f}:1)")
print(f"新轴:        {len(new_fusion):,} bars (压缩比 {len(candles)/len(new_fusion):.1f}:1)")
bar_ratio = len(new_fusion) / len(demo_fusion)
print(f"新轴/DemoBar: {bar_ratio:.1%}")

# 质量判定
print("\n=== 质量判定 ===")
if bar_ratio < 0.25:
    print("⚠️ 强烈警告：交易机会极少（< DemoBar 的 25%），需确认是否符合策略需求")
elif bar_ratio < 0.5:
    print("⚠️ 警告：交易机会较少（< DemoBar 的 50%）")
else:
    print("✓ Bar 数量正常")
```

## 重要提醒

### 脚本运行环境
运行校准脚本和优化脚本时，必须设置 `PYTHONPATH`：
```bash
PYTHONPATH=/path/to/jesse-trade python research/calibrate_xxx.py
PYTHONPATH=/path/to/jesse-trade python research/optimize_xxx.py
```

**为什么需要**：脚本中使用 `from src.bars.fusion.xxx import XxxBar` 导入，如果不设置 PYTHONPATH 会报 `ModuleNotFoundError: No module named 'src.bars'`。

### 优化时间要求
- **最低要求**：1000 trials
- **推荐配置**：2500+ trials（实测表明更多 trials 能找到显著更优的参数）
- **探索策略**：99% 随机探索 + 1% 收束（找轴过程探索比精细微调更重要）
- **后台运行**：必须使用 `run_in_background=True`
- **不要偷懒**：宁愿多等几小时，也不要提前终止

### 后台任务管理最佳实践
1. **启动后短暂检查**：发起后台任务后，用 `tail` 检查输出确认任务正常运行（看到 trial 开始打印即可）
2. **确认后停止等待**：确认正常运行后，不再主动轮询，停下来等待用户通知
3. **用户通知后检查结果**：当用户告知任务完成时，读取输出文件和 CSV 结果
4. **避免不必要的轮询**：优化任务可能运行数小时，频繁检查没有意义

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

### [2026-01] 参数范围必须通过数据统计确定，不能凭直觉
**场景**：开发 LogReturnBar 时，直接套用 DemoBar 的参数范围 `threshold=(0.5, 5.0)`，导致 Optuna 优化时所有 trial 的 fusion_bar_count 都是 0 或 1
**根因**：LogReturnBar 的公式 `|ln(C_t/C_{t-1})| × ln(H_t/L_t)` 量级约 1e-6，而 DemoBar 的公式量级约 1e-3，相差 1000 倍
**解决方案**：新增「阶段 3.5：参数范围校准」，强制要求在 Optuna 优化前先统计阈值分布，用实际数据确定搜索范围
**教训**：不同公式的量级可能差异巨大，凭直觉设定范围必然踩坑。数据驱动优于直觉判断

### [2026-01] 趋势性得分相近不代表质量相近，必须同时考虑 bar 数量
**场景**：LogReturnBar 优化后趋势性得分 68.0/100，与 DemoBar 的 68.6/100 几乎相同，但 bar 数量只有 5,091 vs 20,605（仅 25%）
**问题**：bar 数量少 = 交易机会少
**解决方案**：
1. 更新阶段 6 对比脚本，增加 bar 数量和压缩比的展示
2. 新增 bar 数量质量判定：< 50% 警告，< 25% 强烈警告
**教训**：评价 fusion bar 质量需要多维度考量，趋势性只是其中一个维度。bar 数量决定交易机会，压缩比决定信息密度，都应纳入评估

### [2026-01] threshold 搜索范围过宽会导致 Optuna 优化极慢
**场景**：RSGateBar 校准时使用 `threshold=(1e-6, 1e-3, "log")` 范围，导致部分 trial 产生过多 fusion bars（数万根），TrendValidator 评估时间从 7 秒暴增到 100+ 秒
**根因**：threshold 越小 → fusion bars 越多 → TrendValidator 滑动窗口越多 → ADF/KPSS 计算量剧增
**解决方案**：
1. 在阶段 3.5 增加「目标 Bar 数量范围」约束：30min K线数量（上限）到 6h K线数量（下限）
2. 校准脚本增加二分搜索，自动找到目标范围对应的 threshold 边界
3. 只在该范围内搜索，避免产生过多或过少的 bars
**教训**：搜索范围不仅要考虑数值正确性，还要考虑计算效率。bar 数量是连接参数和计算成本的关键桥梁

### [2026-01] 运行脚本必须设置 PYTHONPATH
**场景**：后台运行 `python research/optimize_rs_gate.py` 失败，报错 `ModuleNotFoundError: No module named 'src.bars'`
**根因**：脚本中使用 `from src.bars.fusion.rs_gate import RSGateBar`，Python 默认不把项目根目录加入 sys.path
**解决方案**：运行脚本时设置 `PYTHONPATH=/path/to/jesse-trade python script.py`
**教训**：所有在 `research/` 目录下的脚本如果导入 `src.*` 模块，都需要设置 PYTHONPATH

### [2026-01] 探索比微调更重要：99% 随机探索 + 1% 收束
**场景**：用户观察到更多 trials 能找到显著更优的参数（如 Trial 221 的 3.2919 vs 早期最优值）
**决策**：
1. 将 `EXPLORATION_STARTUP_RATIO` 从 0.8 提高到 0.99
2. 推荐 trials 数量从 1000 提高到 2500+
**原理**：
- 找轴是一个高维搜索问题，参数空间复杂
- TPE 采样器的 `n_startup_trials` 决定了随机探索阶段的长度
- 过早收束会陷入局部最优，错过更好的参数组合
**实现**：`optimizer.py` 中 `EXPLORATION_STARTUP_RATIO = 0.99`，99% 的 trials 用于随机探索，只有最后 1% 用于贝叶斯优化收束
**教训**：对于复杂参数空间，宁愿多探索也不要过早精细化

### [2026-01] 分层提取避免 bar 少的配置总是占优势
**场景**：优化结果中 bar 数量少的配置总是得分更高，因为长周期自然趋势性更强
**根因**：
- 更少的 bar = 更长的周期 = 噪声被自然平滑
- Hurst/ADF/KPSS 在长周期数据上更容易显示"趋势性强"
- 这是时间尺度的固有特性，不是 fusion bar 公式的优势
**解决方案**：
1. 新增 `extract_top_n_by_tiers()` 函数，按 bar 数量分层（long/medium/short）
2. 每层独立提取 top 5，共 15 个结果
3. 探索比例提高到 99%，确保各区间均匀采样
**实现**：
- `evaluator.py` 新增 `TieredResult` 和 `extract_top_n_by_tiers()`
- `optimizer.py` 新增 `optimize_and_return_study()` 返回 study 对象
**教训**：
- 如果 long 周期始终显著占优势，说明该轴可能不适合频繁交易
- 分层分析能暴露这个信号，帮助用户做出明智决策
