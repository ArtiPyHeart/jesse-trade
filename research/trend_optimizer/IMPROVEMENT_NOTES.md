# 趋势评价体系改进方向

> 基于 2025-01 与 Codex 的深度讨论整理

## 零、已实现功能

### EvaluationReport (详细评估报告)

新增 `MultiWindowEvaluator.evaluate_detailed()` 方法，返回 `EvaluationReport` 对象：

```python
report = evaluator.evaluate_detailed(fusion_bars)
print(report)  # 格式化输出
```

**报告结构**：
- **综合评分**: 0-100 分 + A/B/C/D/F 等级
- **分项得分** (0-5):
  - 趋势性: 各窗口 mean_score 均值
  - 一致性: 基于窗口得分标准差（归一化）
  - 稳定性: 基于高分/低分比例
- **窗口得分**: 各窗口详细得分
- **统计概览**: Bar 数量、评估窗口数等
- **三重检验统计**: Hurst/ADF/KPSS 各项比例

**综合评分公式**：
```python
overall_score = trend * 10 + consistency * 6 + stability * 4
# 满分 = 50 + 30 + 20 = 100
# 权重: 趋势性 50%, 一致性 30%, 稳定性 20%
```

**等级划分**：A ≥ 80 | B ≥ 65 | C ≥ 50 | D ≥ 35 | F < 35

---

## 一、现有方法评估

### 1.1 当前评价体系

```
TrendValidator (三重趋势验证器)
├── Hurst 指数 - 长期记忆性
├── ADF 检验 - 单位根检验
└── KPSS 检验 - 平稳性检验

评分规则 (0-5分):
- Hurst > 0.6: +2
- Hurst > 0.55: +1
- ADF p > 0.05 (非平稳): +1
- KPSS p < 0.05 (非平稳): +1
- 三重共识: +1

MultiWindowEvaluator:
- 多窗口 (20, 40, 60) 等权聚合
```

### 1.2 已验证的有效性

- 通过 Optuna 优化找到 DemoBar 最佳参数
- 回测中配合模型组合取得良好结果
- **结论：信号链路成立，不需要推倒重来**

### 1.3 理论层面的局限性

| 问题 | 说明 |
|------|------|
| 重复计分 | ADF/KPSS 是同一问题的正反检验，与 Hurst 在价格水平上高度相关 |
| 缺失维度 | 只衡量"非平稳性"，未直接衡量"方向一致性/可交易趋势强度" |
| 检验假设 | ADF/KPSS 对异方差、重尾敏感，fusion bar 可能不满足经典假设 |
| 阈值依赖 | 0.55/0.6/0.05 缺乏普适性，短样本下 Hurst 0.55 可能只是噪声 |

---

## 二、改进方向

### 2.1 MVP：时间分段稳定性惩罚

**改动最小，收益最明显**

```python
# 把评估区间分成 3-5 段
segment_scores = [evaluate(segment) for segment in segments]

# 新的 final_score
final_score = mean(segment_scores) - λ * std(segment_scores)
# λ 建议 0.4-0.6
```

**原理**：抑制"偶然区间高分"的参数，提升跨时间段稳健性

### 2.2 引入 Kaufman 效率比率 (ER)

**直接衡量"方向一致性"**

```python
# 公式
ER_t(n) = |P_t - P_{t-n}| / sum_{i=1..n}(|P_{t-i+1} - P_{t-i}|)

# ER ∈ [0, 1]
# 趋势性强 → ER → 1 (路径接近直线)
# 震荡 → ER → 0 (净位移小、路径长)

# 连续评分
score_ER = clip((ER - 0.15) / (0.45 - 0.15), 0, 1)

# 阈值参考
# ER < 0.2: 低趋势效率（偏噪声）
# ER > 0.4: 高趋势效率（可趋势建模）
```

**与现有指标的关系**：ER 只看"路径效率"，不直接判断平稳性，正交信息强

### 2.3 方差比检验 (Variance Ratio Test)

**相对 ADF/KPSS 的优势**：
- 直接回答"是否像随机游走"
- 对"短期自相关/动量/均值回复"更敏感
- 更贴近交易需求

```python
# 核心指标
VR(k) = Var(r_t + ... + r_{t-k+1}) / (k * Var(r_t))

# 解读
# VR ≈ 1: 随机游走
# VR > 1: 正自相关/动量
# VR < 1: 负自相关/均值回复

# 评分（趋势偏好）
score_VR = clip((VR - 1) / c, 0, 1)  # 趋势类关注 VR > 1
```

**推荐实现**：
- 第一阶段：Lo–MacKinlay at k = [2, 5, 10]（廉价）
- 第二阶段：Chow–Denning 多尺度联合检验（稳健）

### 2.4 改成 log returns 做检验

**问题**：价格水平序列天然偏向非平稳，缺乏区分度

**建议**：
- 平稳性/随机游走检验 → 在 log returns 上做
- 效率/趋势结构指标 → 在 log price 上做

**评分规则调整**：改成分位数排名，而非绝对阈值
```python
score_ADF = percentile_rank(-ADF_pvalue)
score_KPSS = percentile_rank(1 - KPSS_pvalue)
```

### 2.5 用连续评分替代硬阈值

```python
# 原来
if hurst > 0.6:
    score += 2
elif hurst > 0.55:
    score += 1

# 改进
score_hurst = clip((hurst - 0.5) / (0.7 - 0.5), 0, 1) * 2
```

**优点**：避免边界抖动，更平滑的优化曲面

### 2.6 跨窗口一致性惩罚

```python
final_score = mean(window_scores) - λ * std(window_scores)

# λ 取值范围
# 稳健性优先: λ = 0.6 ~ 1.0
# 探索型: λ = 0.2 ~ 0.5

# 通过回测敏感性确定 λ
# 1. 设定 λ 网格: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# 2. 对每个 λ 进行"筛选+回测"
# 3. 目标函数: CAGR - α * MaxDD
# 4. 选使 objective 与稳定性更优的 λ
```

### 2.7 两阶段评价（效率优化）

```
[Candles]
   -> Fusion Bars
   -> Stage 1: feature_fast (ER, acf1, VR(k=5), simple stats)  [O(n)]
   -> score_fast = w · f_fast
   -> filter: score_fast >= quantile(score_fast, 0.6)
   -> Stage 2: feature_heavy (ADF, KPSS, multi-k VR, Hurst DFA)
   -> score_heavy
   -> final_score = mean(score_heavy) - λ * std(score_heavy)
```

**第一阶段廉价指标**：
- Kaufman ER
- Rolling autocorr at lag 1-2
- Variance ratio at single k
- Return kurtosis/skew

---

## 三、验证策略

### 3.1 避免过拟合

| 方法 | 说明 |
|------|------|
| Walk-Forward | 滚动训练/验证，评估跨段一致性 |
| 多标的一致性 | 同一评价体系在不同币种上的 Top-K 重叠率 |
| 参数稳定性 | 最佳参数附近应形成"高分平原"而非尖峰 |
| 噪声敏感性 | 对输入做小扰动，排名不应剧烈变化 |
| 基线对照 | 与旧方法、简单基线并列比较 |

### 3.2 验证"改进确实更好"

```python
# 1. 保留法/时间滚动验证
for train_period, test_period in walk_forward_splits:
    params = optimize(train_period)
    test_score = evaluate(params, test_period)

# 2. 看"跨段一致性"而非单段极值

# 3. 参数邻域稳定性
neighborhood_scores = [evaluate(params + noise) for noise in perturbations]
stability = 1 / std(neighborhood_scores)
```

---

## 四、执行建议

### 优先级排序

```
1. [MVP] 时间分段稳定性惩罚
   - 改动最小，效果明显
   - 可立即实施

2. [中期] 引入 Kaufman ER
   - 补充"方向一致性"维度
   - 与现有指标正交

3. [中期] 连续评分替代硬阈值
   - 改善优化曲面平滑性

4. [长期] 两阶段评价 pipeline
   - 效率优化
   - 需要更多工程投入
```

### 核心原则

- **小幅改动、可逆、可对照**
- 不追求更高回测分数，而是提升稳健性
- 保留原评分（已验证有效），增量改进

---

## 五、参考资料

- Kaufman, P.J. "Trading Systems and Methods"
- Lo, A.W. & MacKinlay, A.C. "Stock Market Prices Do Not Follow Random Walks"
- Hurst, H.E. "Long-term storage capacity of reservoirs"
- statsmodels: `adfuller`, `kpss`, `VarianceRatio`
