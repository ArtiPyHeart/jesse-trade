# Claude SKILL: 置信度切片分析与过滤器配置

## 任务概述
分析ML模型的置信度切片图表，根据曲线形态配置过滤器以校准交易行为。通过双重验证机制确保判定准确性。

## 核心逻辑
- 面板数据拟合 + 时序盈亏曲线 → 识别有效/反向/随机区间
- 多模型投票：一致则开仓，分歧则空仓
- **用户必须指定目标策略**（如 `BinanceBtcDeapV1Voting`），未指定则立即停止询问

## 模型命名约定（自动解析）
路径格式：`temp/{类型}_L{LAG}_N{PRED_NEXT}`

| 模型类型 | LOWER_BOUND | UPPER_BOUND | THRESHOLD |
|---------|-------------|-------------|-----------|
| c (分类) | 0.0 | 1.0 | 0.5 |
| r (回归) | -1.0 | 1.0 | 0.0 |

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
每张图片分析后做出以下三种决策之一：

### A. 不干预（good）
- 曲线单调上升 + 平均收益>0 + 最终收益>0
- 操作：无需过滤器

### B. 反向操作（reverse）
- 曲线单调下降 + 平均收益<0 + 最终收益<0
- 操作：`mc.add_reverse_filter(lower_bound, upper_bound)`

### C. 放弃交易（giveup）
- 样本占比=0 / 曲线S型波动 / 半圆形走势 / 平均收益≈0但波动剧烈
- 操作：`mc.add_giveup_filter(lower_bound, upper_bound)`

**小样本原则**：交易次数少时，以曲线走向为主，只要相对单调即可判定

## 工作流程

### 阶段1：准备
1. 提取用户指定的策略名称（未指定则停止）
2. 执行策略验证（失败则停止）
3. 解析模型参数（从路径提取）
4. 扫描目标目录获取所有切片图片（Glob）
5. 检查进度文件：`temp/{模型名称}_round1.jsonl`、`_round2.jsonl`、`_final.jsonl`

### 阶段2：双轮并行分析
**第一轮**：
- 图片分组（5-10张/组）
- 启动多个 child agents 并行处理
- 每个 agent 分析曲线特征，做出决策，写入 `_round1.jsonl`
- 验证 JSONL 完整性（行数=图片数，格式正确）

**第二轮**：
- 用相同分组再次启动 child agents **独立分析**
- 写入 `_round2.jsonl`
- 验证 JSONL 完整性

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
    model_type="c",  # 从路径解析
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

**字段**：slice (区间), decision (good/reverse/giveup), reason (理由), sample_ratio (占比), source (consensus/referee)

## 核心要求
1. **强制验证**：策略目录、config.py、LGBMContainer，失败立即停止
2. **双重验证**：两轮独立分析，不一致由裁判判定
3. **并行处理**：Task tool + child agents（5-10张/组）
4. **完整性检查**：每轮后验证 JSONL（行数、格式），通过后才推进
5. **保守原则**：裁判无法确定或样本=0 → giveup
6. **形态优先**：曲线形态是主要判断依据，数值指标为辅助
7. **动态代码**：必须使用用户指定的策略名称
