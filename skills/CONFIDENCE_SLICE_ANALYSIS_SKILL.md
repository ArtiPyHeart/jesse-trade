# Claude SKILL: 置信度切片分析与过滤器配置

## 任务概述
分析ML模型的置信度切片图表，根据曲线形态配置过滤器以校准模型的多空交易行为。通过双重验证机制确保判定准确性。

**核心原则**：
- 所有分析仅针对**有样本的切片**（ratio>0），零样本切片在准备阶段直接忽略
- 判定基于**曲线形态**而非样本量，样本少切片与样本多的切片遵循相同准则。切片分析的结果应当只和曲线形态有关，哪怕样本量极少，我们也应当关注曲线形态而非样本量。
- **用户必须指定目标策略**（如 `BinanceBtcDeapV1Voting`），未指定则立即停止询问

## 核心逻辑
- 机器学习拟合的是**面板数据**，而量化投资需要考虑**时序性**
- 即便模型拥有较好的拟合效果，在时序上依然存在亏损的可能性
- 模型的置信度（predict proba）反映的是模型开多开空的倾向性。
  - 分类模型以0.5为界，大于0.5意味着模型倾向于开仓多头，小于0.5意味着倾向于开仓多头
  - 回归模型以0为界，大于0意味着模型倾向于开仓多头，小于0意味着模型倾向于开仓空头

### 解决方案
- 通过置信度切片在测试集上对模型预测概率进行细粒度切分
- 结合实际盈亏制作当前粒度的真实盈亏曲线
- 通过校准后的多个模型投票决定交易行为
    - 所有模型一致预期多头/空头 → 开启相应持仓
    - 模型预测有分歧 → 空仓（利用分歧减少交易次数并降低磨损）

## 模型命名约定（自动解析）
路径格式：`temp/{类型}_L{LAG}_N{PRED_NEXT}/`

| 模型类型 | LOWER_BOUND | UPPER_BOUND | THRESHOLD |
|---------|-------------|-------------|-----------|
| c (分类) | 0.0 | 1.0 | 0.5 |
| r (回归) | -1.0 | 1.0 | 0.0 |

**回归模型边界 Round 机制**：
- 绘图时存在 round 机制：小于 LOWER_BOUND 的值归入 LOWER_BOUND 切片，大于 UPPER_BOUND 的值归入 UPPER_BOUND 切片
- **配置过滤器时**：若要设定最边缘切片的过滤器，应使用更大的范围值（如 `-10` 作为下界，`10` 作为上界）
- 分类模型无此问题，预测概率值始终在 [0, 1] 范围内

**示例**：
- `temp/c_L3_N1` → model_type="c", lag=3, pred_next=1, threshold=0.5
- `temp/r_L7_N3` → model_type="r", lag=7, pred_next=3, threshold=0

## 策略验证（强制）
分析前必须验证：
1. 策略目录 `strategies/{STRATEGY_NAME}` 存在
2. 配置文件 `strategies/{STRATEGY_NAME}/models/config.py` 存在
3. `LGBMContainer` 类可从配置文件导入

**任何验证失败立即停止**，不得猜测或创建文件

## 决策准则

**核心原则**：从整体趋势判断，横盘是无害的，反复无常的盈亏切换才是有害的。

**辅助判断方法**：想象用一条线性回归直线拟合收益曲线，通过斜率辅助判断：
- 斜率大幅 > 0 → good（整体上升趋势）
- 斜率大幅 < 0 → reverse（整体下降趋势）
- 斜率 ≈ 0 → giveup（无明确方向性）

每张图片包含以下关键信息：
- **切片区间**：`[lower, upper]`，如 [0.84, 0.86]
- **交易方向**：(做多) 或 (做空)
- **样本占比**：该区间样本数占总样本的比例
- **最终收益**：曲线终点的累积盈亏值
- **平均收益**：红色虚线表示的均值
- **曲线形态**：整体走势特征

### A. 不干预（good）
**判定条件**（满足任一即可）：
1. **整体上升趋势**：最终收益>0 且 平均收益>0，允许：
   - 中途横盘震荡（震荡幅度小于前期上升幅度的50%）
   - 小幅回调（亏损幅度不超过累计收益的50%）
   - 阶段性盘整后继续上升
2. **稳定盈利**：曲线长期稳定在盈利区域

**关键特征**：起点到终点整体向上，大部分时间保持在平均收益线以上

**操作**：无需过滤器

### B. 反向操作（reverse）
**判定条件**（满足任一即可）：
1. **整体下降趋势**：最终收益<0 且 平均收益<0，允许：
   - 中途横盘震荡（震荡幅度小于前期下降幅度的50%）
   - 小幅反弹后继续下跌（反弹幅度不超过累计亏损的50%）
   - 阶段性企稳后继续下降
2. **稳定亏损**：曲线长期稳定在亏损区域

**关键特征**：起点到终点整体向下，大部分时间保持在平均收益线以下

**操作**：`mc.add_reverse_filter(lower_bound, upper_bound)`

### C. 放弃交易（giveup）
**判定条件**（满足任一即可）：
1. **剧烈盈亏切换**：曲线反复穿越零线（或平均收益线），呈现：
   - **S型/W型/M型波动**：先盈利→亏损→盈利，或反之，盈亏幅度相当
   - **半圆型/U型/V型波动**：先盈利后亏损（或反之），终点有向零轴回归趋势
2. **无方向性**：曲线盈利较小或趋近零，整体无明显方向性

**关键特征**：盈利和亏损状态的反复切换（非横盘），难以判断未来走势

**不应判定为giveup**：
- ❌ 在盈利区长期横盘（应为good）
- ❌ 在亏损区长期横盘（应为reverse）
- ❌ 有小幅回调但整体向上（应为good）
- ❌ 有小幅反弹但整体向下（应为reverse）

**操作**：`mc.add_giveup_filter(lower_bound, upper_bound)`

---

**判定优先级**：
1. 看整体趋势（起点→终点，最终收益与平均收益的符号一致性）
2. 检查是否存在剧烈盈亏反转形态
3. 横盘不影响good/reverse判定

---

## 工作流程

### 阶段1：准备
1. 提取用户指定的策略名称（未指定则停止）
2. 执行策略验证（失败则停止）
3. 解析模型参数（从路径提取）
4. 扫描目标目录获取所有切片图片（Glob）
5. **过滤零样本切片**：
   - 从文件名解析样本占比（`slice_{lower}_{upper}_ratio_{ratio}.jpg`）
   - 过滤掉 ratio=0.0000 的切片，记录非零样本切片数量 `N_nonzero`
6. 检查进度文件：`{模型名称}_round1.jsonl`、`_round2.jsonl`、`_final.jsonl`

**非零样本切片统计脚本**：
```python
from pathlib import Path
import re

model_name = "c_L4_N2"
model_dir = Path(f"temp/{model_name}")

pattern = re.compile(r"slice_(-?[\d.]+)_(-?[\d.]+)_ratio_([\d.]+)\.jpg")
nonzero_slices = []

for img_file in model_dir.glob("slice_*.jpg"):
    match = pattern.match(img_file.name)
    if match:
        lower, upper, ratio = match.groups()
        if float(ratio) > 0:
            nonzero_slices.append({
                "file": img_file,
                "slice": f"[{lower}, {upper}]",
                "ratio": float(ratio)
            })

N_nonzero = len(nonzero_slices)
print(f"非零样本切片数量: {N_nonzero}")
```

### 阶段2：双轮并行分析

采用「分批次启动 + 分组独立写入 + 主进程合并」策略，避免资源限制和并发写入冲突。

**分批次规则**：
- 仅分析 `N_nonzero` 个非零样本切片
- 每个 agent 分析 **10张图片**
- 每批次最多启动 **5个 agent**
- 图片总数 >50张时分多批次，等待当前批次完成后再启动下一批次

**批次计算示例**：
| N_nonzero | 分组数 | 批次数 | 每批次agent数 |
|-----------|--------|--------|--------------|
| 85张 | 9组 | 2批 | 5+4 |
| 150张 | 15组 | 3批 | 5+5+5 |

**执行流程**（每轮相同）：
1. 图片分组（10张/组）
2. 分批次启动 agents，每个 agent 写入独立文件（如 `_round1_g01.jsonl`）
3. 所有批次完成后，主进程合并分组文件
4. 验证完整性：已分析切片数 = `N_nonzero`
5. 若缺失，启动补充 agent

**合并与验证脚本**：
```python
import json
import re
from pathlib import Path

model_name = "c_L6_N2"
round_num = 1
model_dir = Path(f"temp/{model_name}")

# 统计非零样本切片
pattern = re.compile(r"slice_(-?[\d.]+)_(-?[\d.]+)_ratio_([\d.]+)\.jpg")
expected_slices = set()
for img_file in model_dir.glob("slice_*.jpg"):
    match = pattern.match(img_file.name)
    if match:
        lower, upper, ratio = match.groups()
        if float(ratio) > 0:
            expected_slices.add(f"[{lower}, {upper}]")
N_nonzero = len(expected_slices)

# 合并分组文件
group_files = sorted(model_dir.glob(f"{model_name}_round{round_num}_g*.jsonl"))
all_records = []
for gf in group_files:
    with open(gf, 'r') as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line.strip()))

# 写入合并文件并删除分组文件
output_file = model_dir / f"{model_name}_round{round_num}.jsonl"
with open(output_file, 'w') as f:
    for record in all_records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
for gf in group_files:
    gf.unlink()

# 验证完整性
existing_slices = {r['slice'] for r in all_records}
missing_slices = expected_slices - existing_slices
if missing_slices:
    print(f"缺失 {len(missing_slices)} 个切片，需补充分析")
else:
    print(f"完整性验证通过（{N_nonzero} 个非零样本切片）")
```

### 阶段3：裁判
1. 对比两轮决策
2. 一致 → 直接写入 `_final.jsonl`
3. 不一致 → 启动裁判 agent：
   - 提供图片、两轮决策和理由
   - 综合判定（形态优先，数值辅助）
   - 无法确定 → 采用 giveup（保守原则）
4. 裁判结果写入 `_final.jsonl`

### 阶段4：生成配置
1. 读取 `_final.jsonl` 分类汇总
2. 生成可执行代码 `apply_filters_{模型名称}.py`：
```python
from strategies.{STRATEGY_NAME}.models.config import LGBMContainer

mc = LGBMContainer(
    model_type="c",
    lag=3,
    pred_next=1,
    threshold=0.5,
)

# 从 final.jsonl 自动生成
mc.add_reverse_filter(0.02, 0.04)
mc.add_giveup_filter(0.48, 0.50)
# ...

mc.save_filters()
```
3. 运行代码，保存配置到策略 models 目录
4. 立即删除临时 Python 文件

## JSONL 格式
**round1/round2**：
```jsonl
{"slice": "[0.02, 0.04]", "decision": "reverse", "reason": "曲线持续下降，最终收益-0.0856", "sample_ratio": 0.0148}
```

**final**：
```jsonl
{"slice": "[0.02, 0.04]", "decision": "good", "reason": "裁判判定：综合评估为good", "source": "referee", "sample_ratio": 0.0148}
```

**字段**：slice, decision (good/reverse/giveup), reason, sample_ratio, source (consensus/referee)

## 核心要求

1. **强制验证**：策略目录、config.py、LGBMContainer，失败立即停止
2. **非零样本原则**：零样本切片在阶段1过滤，不分析、不配置过滤器；完整性验证基于 `N_nonzero`
3. **双重验证**：两轮独立分析，不一致由裁判判定
4. **分组独立写入**：禁止多 agent 同时写入同一文件，主进程负责合并
5. **形态判定**：基于曲线整体趋势，样本少但>0的切片遵循相同准则
6. **动态代码**：必须使用用户指定的策略名称
